"""Task 1.3, 2.1-2.5: Data parsing and feature engineering for MovieLens 1M.

Reads raw .dat files from MinIO bucket 'raw-data',
saves processed output as Delta Lake tables into MinIO bucket 'processed'.
"""

import io
import json
import boto3
import pandas as pd
from botocore.client import Config
from typing import Dict, List, Tuple
from deltalake import DeltaTable, write_deltalake


# ── MinIO connection settings ─────────────────────────────────────────────────
MINIO_ENDPOINT   = "http://localhost:9000"
MINIO_ACCESS_KEY = "minioadmin"
MINIO_SECRET_KEY = "minioadmin123"
RAW_BUCKET       = "raw-data"
PROCESSED_BUCKET = "processed"

STORAGE_OPTIONS = {
    "AWS_ACCESS_KEY_ID":          MINIO_ACCESS_KEY,
    "AWS_SECRET_ACCESS_KEY":      MINIO_SECRET_KEY,
    "AWS_ENDPOINT_URL":           MINIO_ENDPOINT,
    "AWS_REGION":                 "us-east-1",
    "AWS_ALLOW_HTTP":             "true",
    "AWS_S3_ALLOW_UNSAFE_RENAME": "true",
}


# ── MinIO helpers ─────────────────────────────────────────────────────────────
def _s3_client() -> boto3.client:
    return boto3.client(
        "s3",
        endpoint_url=MINIO_ENDPOINT,
        aws_access_key_id=MINIO_ACCESS_KEY,
        aws_secret_access_key=MINIO_SECRET_KEY,
        config=Config(signature_version="s3v4"),
    )


def _ensure_bucket(client: boto3.client, bucket: str) -> None:
    existing = [b["Name"] for b in client.list_buckets()["Buckets"]]
    if bucket not in existing:
        client.create_bucket(Bucket=bucket)
        print(f"  Created bucket: {bucket}")


def _read_tsv(client: boto3.client, key: str) -> pd.DataFrame:
    """Read a tab-separated file from MinIO, stripping ':type' suffixes from headers."""
    obj = client.get_object(Bucket=RAW_BUCKET, Key=key)
    df = pd.read_csv(io.BytesIO(obj["Body"].read()), sep="\t", header=0, encoding="utf-8")
    df.columns = [col.split(":")[0] for col in df.columns]
    return df


def _clear_and_write_delta(uri: str, df: pd.DataFrame, split_name: str) -> None:
    """Check for existing Delta table, delete all rows, then write new data."""
    if DeltaTable.is_deltatable(uri, storage_options=STORAGE_OPTIONS):
        dt = DeltaTable(uri, storage_options=STORAGE_OPTIONS)
        dt.delete()
        print(f"  [{split_name}] Deleted existing data from Delta table")
        write_deltalake(uri, df, storage_options=STORAGE_OPTIONS, mode="append")
    else:
        print(f"  [{split_name}] No existing table found, creating new")
        write_deltalake(uri, df, storage_options=STORAGE_OPTIONS, mode="overwrite")
    print(f"  [{split_name}] Written {len(df):>7,} new rows → {uri}")


def _records_to_df(records: List[Dict]) -> pd.DataFrame:
    """Flatten list of sequence dicts into a DataFrame for Delta Lake."""
    rows = []
    for r in records:
        rows.append({
            "user_id":    int(r["user_id"]),
            "item_seq":   r["item_seq"],
            "genre_seq":  r["genre_seq"],   # list of lists → Arrow list<list<int64>>
            "time_seq":   r["time_seq"],
            "target":     int(r["target"]),
            "target_time": int(r["target_time"]),
            "age_idx":    int(r["user_profile"]["age_idx"]),
            "gender_idx": int(r["user_profile"]["gender_idx"]),
            "occupation": int(r["user_profile"]["occupation"]),
        })
    return pd.DataFrame(rows)


# ── Preprocessor ──────────────────────────────────────────────────────────────
class MovieLensPreprocessor:
    """Preprocessor for MovieLens 1M dataset following CineMate Mamba4Rec requirements."""

    GENRES = [
        'Action', 'Adventure', 'Animation', "Children's", 'Comedy', 'Crime',
        'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical',
        'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'
    ]

    def __init__(self, max_genres: int = 3):
        self.max_genres = max_genres
        self.genre_to_idx = {g: i + 1 for i, g in enumerate(self.GENRES)}  # 0 = padding
        self.gender_encoder = {'F': 0, 'M': 1}
        self.age_encoder = None
        self.users_df = None
        self.movies_df = None
        self.ratings_df = None

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Task 1.3: Read the 3 files from MinIO raw-data bucket."""
        print(f"Loading MovieLens 1M from MinIO bucket '{RAW_BUCKET}'...")
        client = _s3_client()

        self.users_df = _read_tsv(client, "ml-1m.user")
        # rename to match downstream expectations
        self.users_df = self.users_df.rename(columns={"user_id": "user_id"})

        self.movies_df = _read_tsv(client, "ml-1m.item")
        self.movies_df = self.movies_df.rename(columns={
            "item_id": "movie_id",
            "movie_title": "title",
            "genre": "genres",
        })

        self.ratings_df = _read_tsv(client, "ml-1m.inter")
        self.ratings_df = self.ratings_df.rename(columns={"item_id": "movie_id"})

        print(f"  Users: {len(self.users_df)}, Movies: {len(self.movies_df)}, Ratings: {len(self.ratings_df)}")
        return self.users_df, self.movies_df, self.ratings_df

    def encode_static_features(self) -> pd.DataFrame:
        """
        Task 2.1: Encode static user features.

        - Gender: LabelEncode (F=0, M=1)
        - Age: LabelEncode (0-N)
        - Occupation: Keep as-is (already numeric 0-20)
        """
        users = self.users_df.copy()

        users['gender_idx'] = users['gender'].map(self.gender_encoder)

        age_values = sorted(users['age'].unique())
        self.age_encoder = {age: idx for idx, age in enumerate(age_values)}
        users['age_idx'] = users['age'].map(self.age_encoder)

        print(f"  Age groups: {self.age_encoder}")
        print(f"  Gender encoding: {self.gender_encoder}")
        return users

    def encode_genres(self) -> pd.DataFrame:
        """
        Task 2.2: Process multi-hot genres.

        Map Genres column to list of genre IDs, pad/truncate to max_genres.
        """
        movies = self.movies_df.copy()

        def genres_to_ids(genres_str: str) -> List[int]:
            genres = genres_str.split(' ')
            ids = [self.genre_to_idx.get(g, 0) for g in genres if g in self.genre_to_idx]
            if len(ids) < self.max_genres:
                ids = ids + [0] * (self.max_genres - len(ids))
            else:
                ids = ids[:self.max_genres]
            return ids

        movies['genre_ids'] = movies['genres'].apply(genres_to_ids)
        print(f"  Genre vocabulary size: {len(self.genre_to_idx) + 1} (including padding)")
        return movies

    def extract_time_context(self) -> pd.DataFrame:
        """
        Task 2.3: Extract time context from timestamp.

        TimeOfDay:
        - 0: Matinee    (06:00–17:59)
        - 1: Prime Time (18:00–21:59)
        - 2: Late Night (22:00–05:59)
        """
        ratings = self.ratings_df.copy()
        ratings['datetime'] = pd.to_datetime(ratings['timestamp'], unit='s')
        ratings['hour'] = ratings['datetime'].dt.hour

        def get_time_of_day(hour: int) -> int:
            if 6 <= hour < 18:
                return 0
            elif 18 <= hour < 22:
                return 1
            else:
                return 2

        ratings['time_of_day'] = ratings['hour'].apply(get_time_of_day)

        time_dist = ratings['time_of_day'].value_counts().sort_index()
        print("  Time of Day distribution:")
        print(f"    0 Matinee    : {time_dist.get(0, 0)}")
        print(f"    1 Prime Time : {time_dist.get(1, 0)}")
        print(f"    2 Late Night : {time_dist.get(2, 0)}")
        return ratings

    def build_sequences(
        self,
        users_df: pd.DataFrame,
        movies_df: pd.DataFrame,
        ratings_df: pd.DataFrame,
        min_seq_len: int = 5,
    ) -> Dict:
        """
        Task 2.4: Construct per-user behaviour sequences.

        Group by UserID, sort by Timestamp.
        Output: {user_id: {item_seq, genre_seq, time_seq, user_profile, timestamps}}
        """
        print("Building user sequences...")

        data = ratings_df.merge(users_df, on='user_id')
        data = data.merge(movies_df[['movie_id', 'genre_ids']], on='movie_id')
        data = data[data['rating'] >= 3]

        unique_movies = data['movie_id'].unique()
        self.movie_id_map = {old: new + 1 for new, old in enumerate(sorted(unique_movies))}
        self.reverse_movie_map = {v: k for k, v in self.movie_id_map.items()}
        data['item_idx'] = data['movie_id'].map(self.movie_id_map)

        data = data.sort_values(['user_id', 'timestamp'])

        sequences = {}
        for user_id, group in data.groupby('user_id'):
            if len(group) < min_seq_len:
                continue
            user_info = group.iloc[0]
            sequences[user_id] = {
                'item_seq':  group['item_idx'].tolist(),
                'genre_seq': group['genre_ids'].tolist(),
                'time_seq':  group['time_of_day'].tolist(),
                'user_profile': {
                    'age_idx':    int(user_info['age_idx']),
                    'gender_idx': int(user_info['gender_idx']),
                    'occupation': int(user_info['occupation']),
                },
                'timestamps': group['timestamp'].tolist(),
            }

        print(f"  Built sequences for {len(sequences)} users")
        print(f"  Total unique items: {len(self.movie_id_map)}")
        return sequences

    def split_sequences(
        self,
        sequences: Dict,
        max_seq_len: int = 50,
    ) -> Tuple[List, List, List]:
        """
        Task 2.5: Leave-One-Out split.

        - Test : last item (N)
        - Val  : second-to-last item (N-1)
        - Train: sliding window on the rest
        """
        print("Splitting sequences...")

        train_data, val_data, test_data = [], [], []

        for user_id, seq_data in sequences.items():
            item_seq    = seq_data['item_seq']
            genre_seq   = seq_data['genre_seq']
            time_seq    = seq_data['time_seq']
            user_profile = seq_data['user_profile']

            if len(item_seq) < 3:
                continue

            # Test
            test_data.append({
                'user_id':    user_id,
                'item_seq':   item_seq[:-1][-max_seq_len:],
                'genre_seq':  genre_seq[:-1][-max_seq_len:],
                'time_seq':   time_seq[:-1][-max_seq_len:],
                'target':     item_seq[-1],
                'target_time': time_seq[-1],
                'user_profile': user_profile,
            })

            # Val
            val_data.append({
                'user_id':    user_id,
                'item_seq':   item_seq[:-2][-max_seq_len:],
                'genre_seq':  genre_seq[:-2][-max_seq_len:],
                'time_seq':   time_seq[:-2][-max_seq_len:],
                'target':     item_seq[-2],
                'target_time': time_seq[-2],
                'user_profile': user_profile,
            })

            # Train (sliding window)
            train_seq    = item_seq[:-2]
            train_genres = genre_seq[:-2]
            train_times  = time_seq[:-2]

            for i in range(1, len(train_seq)):
                train_data.append({
                    'user_id':    user_id,
                    'item_seq':   train_seq[:i][-max_seq_len:],
                    'genre_seq':  train_genres[:i][-max_seq_len:],
                    'time_seq':   train_times[:i][-max_seq_len:],
                    'target':     train_seq[i],
                    'target_time': train_times[i],
                    'user_profile': user_profile,
                })

        print(f"  Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
        return train_data, val_data, test_data

    def process_all(self) -> Tuple[List, List, List, Dict]:
        """
        Run the full preprocessing pipeline end-to-end.

        Reads from MinIO 'raw-data', writes Delta Lake tables to 'processed'.
        Returns (train_data, val_data, test_data, metadata).
        """
        # ── 1. Load ────────────────────────────────────────────────────────────
        self.load_data()

        # ── 2. Feature engineering ─────────────────────────────────────────────
        print("\n[Feature engineering]")
        users_df   = self.encode_static_features()
        movies_df  = self.encode_genres()
        ratings_df = self.extract_time_context()

        # ── 3. Sequences ───────────────────────────────────────────────────────
        print("\n[Sequences]")
        sequences = self.build_sequences(users_df, movies_df, ratings_df)

        # ── 4. Split ───────────────────────────────────────────────────────────
        print("\n[Split]")
        train_data, val_data, test_data = self.split_sequences(sequences)

        # ── 5. Metadata ────────────────────────────────────────────────────────
        metadata = {
            'num_items':           len(self.movie_id_map) + 1,
            'num_genres':          len(self.GENRES) + 1,
            'num_ages':            len(self.age_encoder),
            'num_genders':         2,
            'num_occupations':     21,
            'num_time_slots':      3,
            'max_genres_per_item': self.max_genres,
            'genre_to_idx':        self.genre_to_idx,
            'age_encoder':         {int(k): v for k, v in self.age_encoder.items()},
            'movie_id_map':        {int(k): int(v) for k, v in self.movie_id_map.items()},
            'reverse_movie_map':   {int(k): int(v) for k, v in self.reverse_movie_map.items()},
        }

        # ── 6. Save to Delta Lake ──────────────────────────────────────────────
        print("\n[Saving to Delta Lake in MinIO]")
        client = _s3_client()
        _ensure_bucket(client, PROCESSED_BUCKET)

        for split_name, split_data in [("train", train_data), ("val", val_data), ("test", test_data)]:
            df  = _records_to_df(split_data)
            uri = f"s3://{PROCESSED_BUCKET}/{split_name}"
            _clear_and_write_delta(uri, df, split_name)

        # Save metadata as JSON
        client.put_object(
            Bucket=PROCESSED_BUCKET,
            Key="metadata.json",
            Body=json.dumps(metadata, indent=2).encode(),
            ContentType="application/json",
        )
        print(f"  Saved metadata                  → s3://{PROCESSED_BUCKET}/metadata.json")

        return train_data, val_data, test_data, metadata


# ── Entry point ───────────────────────────────────────────────────────────────
def main() -> None:
    preprocessor = MovieLensPreprocessor()
    _, _, _, metadata = preprocessor.process_all()

    print("\n[Done] Metadata summary:")
    for k, v in metadata.items():
        if not isinstance(v, dict):
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
