"""Tasks 1.3, 2.1-2.5: Data parsing and feature engineering for MovieLens 1M.

Reads raw Parquet files from MinIO bucket 'raw-data' (written by IngestPipeline),
preprocesses them, and saves the results as Delta Lake tables into
s3://offline-store/delta/ (train / val / test splits).

Data lineage:
  Kafka → IngestPipeline → s3://raw-data/new_user|new_movie|new_rating/year=.../...
                │
          preprocess.py  ← THIS FILE
                │
  s3://offline-store/delta/train|val|test
                │
  materialization.py → DuckDB staging → Feast → Redis
"""

import io
import json
import logging
import os

import boto3
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from botocore.client import Config
from deltalake import DeltaTable, write_deltalake

logger = logging.getLogger(__name__)

# ── MinIO connection settings ─────────────────────────────────────────────────
MINIO_ENDPOINT   = os.environ.get("AWS_ENDPOINT_URL", "http://localhost:9000")
MINIO_ACCESS_KEY = os.environ.get("AWS_ACCESS_KEY_ID", "minioadmin")
MINIO_SECRET_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY", "minioadmin123")
RAW_BUCKET       = "raw-data"
DELTA_BUCKET     = "offline-store"

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


def _read_parquet_prefix(client: boto3.client, bucket: str, prefix: str) -> pd.DataFrame:
    """Read all Parquet files under a bucket/prefix and return as a single DataFrame."""
    paginator = client.get_paginator("list_objects_v2")
    tables: list[pa.Table] = []

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not key.endswith(".parquet"):
                continue
            response = client.get_object(Bucket=bucket, Key=key)
            buf = io.BytesIO(response["Body"].read())
            tables.append(pq.read_table(buf))

    if not tables:
        logger.warning("No Parquet files found under s3://%s/%s", bucket, prefix)
        return pd.DataFrame()

    df = pa.concat_tables(tables).to_pandas()
    logger.info("Read %d rows from s3://%s/%s", len(df), bucket, prefix)
    return df


def _write_delta(uri: str, df: pd.DataFrame, split_name: str) -> None:
    """Write or overwrite a Delta Lake table at uri."""
    action = "Overwriting" if DeltaTable.is_deltatable(uri, storage_options=STORAGE_OPTIONS) else "Creating"
    print(f"  [{split_name}] {action} Delta table")
    write_deltalake(uri, df, storage_options=STORAGE_OPTIONS, mode="overwrite", schema_mode="overwrite")
    print(f"  [{split_name}] Written {len(df):>7,} rows → {uri}")


def _records_to_df(records: list[dict]) -> pd.DataFrame:
    """Flatten sequence records into a DataFrame for Delta Lake."""
    rows = []
    for r in records:
        rows.append({
            "user_id":         int(r["user_id"]),
            "event_timestamp": pd.Timestamp(r["last_timestamp"], unit="s", tz="UTC"),
            "item_seq":        r["item_seq"],
            "genre_seq":       r["genre_seq"],
            "time_seq":        r["time_seq"],
            "target":          int(r["target"]),
            "target_time":     int(r["target_time"]),
            "age_idx":         int(r["user_profile"]["age_idx"]),
            "gender_idx":      int(r["user_profile"]["gender_idx"]),
            "occupation":      int(r["user_profile"]["occupation"]),
        })
    return pd.DataFrame(rows)


# ── Preprocessor ──────────────────────────────────────────────────────────────
class MovieLensPreprocessor:
    """Preprocessor for MovieLens 1M dataset.

    Reads Parquet files written by IngestPipeline from s3://raw-data/:
      - new_user/   → user_id, gender, age, occupation, zip_code
      - new_movie/  → movie_id, title, genres (list[str])
      - new_rating/ → user_id, movie_id, rating, rating_timestamp (Unix int)

    Produces train/val/test Delta Lake tables at s3://offline-store/delta/.
    """

    GENRES = [
        "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime",
        "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical",
        "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
    ]

    def __init__(self, max_genres: int = 3):
        self.max_genres    = max_genres
        self.genre_to_idx  = {g: i + 1 for i, g in enumerate(self.GENRES)}  # 0 = padding
        self.gender_encoder = {"F": 0, "M": 1}
        self.age_encoder   = None
        self.movie_id_map  = None
        self.reverse_movie_map = None
        self.users_df  = None
        self.movies_df = None
        self.ratings_df = None

    # ── Task 1.3: Load ────────────────────────────────────────────────────────
    def load_data(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Task 1.3: Read raw Parquet files from MinIO raw-data bucket."""
        print(f"Loading MovieLens 1M from MinIO bucket '{RAW_BUCKET}'...")
        client = _s3_client()

        # Users: IngestPipeline writes to raw-data/new_user/year=.../...
        users_df = _read_parquet_prefix(client, RAW_BUCKET, "new_user/")
        if users_df.empty:
            raise RuntimeError(
                "No user data in s3://raw-data/new_user/. "
                "Run simulate_ingestion.py first."
            )
        self.users_df = users_df[["user_id", "gender", "age", "occupation", "zip_code"]].copy()
        self.users_df["user_id"]    = pd.to_numeric(self.users_df["user_id"],    errors="coerce")
        self.users_df["age"]        = pd.to_numeric(self.users_df["age"],        errors="coerce")
        self.users_df["occupation"] = pd.to_numeric(self.users_df["occupation"], errors="coerce")
        self.users_df = self.users_df.dropna(subset=["user_id"]).drop_duplicates("user_id")

        # Movies: IngestPipeline writes to raw-data/new_movie/year=.../...
        movies_df = _read_parquet_prefix(client, RAW_BUCKET, "new_movie/")
        if movies_df.empty:
            raise RuntimeError(
                "No movie data in s3://raw-data/new_movie/. "
                "Run simulate_ingestion.py first."
            )
        self.movies_df = movies_df[["movie_id", "title", "genres"]].copy()
        self.movies_df["movie_id"] = pd.to_numeric(self.movies_df["movie_id"], errors="coerce")
        self.movies_df = self.movies_df.dropna(subset=["movie_id"]).drop_duplicates("movie_id")

        # Ratings: IngestPipeline writes to raw-data/new_rating/year=.../...
        ratings_df = _read_parquet_prefix(client, RAW_BUCKET, "new_rating/")
        if ratings_df.empty:
            raise RuntimeError(
                "No rating data in s3://raw-data/new_rating/. "
                "Run simulate_ingestion.py first."
            )
        self.ratings_df = ratings_df[["user_id", "movie_id", "rating", "rating_timestamp"]].copy()
        self.ratings_df["user_id"]          = pd.to_numeric(self.ratings_df["user_id"],          errors="coerce")
        self.ratings_df["movie_id"]         = pd.to_numeric(self.ratings_df["movie_id"],         errors="coerce")
        self.ratings_df["rating"]           = pd.to_numeric(self.ratings_df["rating"],           errors="coerce")
        self.ratings_df["rating_timestamp"] = pd.to_numeric(self.ratings_df["rating_timestamp"], errors="coerce")
        self.ratings_df = self.ratings_df.dropna(subset=["user_id", "movie_id", "rating"])
        # Use original Unix timestamp column; rename to 'timestamp' for downstream methods
        self.ratings_df = self.ratings_df.rename(columns={"rating_timestamp": "timestamp"})

        print(
            f"  Users: {len(self.users_df):,}  "
            f"Movies: {len(self.movies_df):,}  "
            f"Ratings: {len(self.ratings_df):,}"
        )
        return self.users_df, self.movies_df, self.ratings_df

    # ── Task 2.1: Static user features ───────────────────────────────────────
    def encode_static_features(self) -> pd.DataFrame:
        """Task 2.1: LabelEncode gender and age; keep occupation numeric."""
        users = self.users_df.copy()

        users["gender_idx"] = users["gender"].str.strip().str.upper().map(self.gender_encoder)

        age_values = sorted(users["age"].dropna().unique())
        self.age_encoder = {age: idx for idx, age in enumerate(age_values)}
        users["age_idx"] = users["age"].map(self.age_encoder)

        print(f"  Age groups: {self.age_encoder}")
        print(f"  Gender encoding: {self.gender_encoder}")
        return users

    # ── Task 2.2: Genre encoding ──────────────────────────────────────────────
    def encode_genres(self) -> pd.DataFrame:
        """Task 2.2: Map genres to IDs, pad/truncate to max_genres."""
        movies = self.movies_df.copy()

        def genres_to_ids(genres) -> list[int]:
            # genres is a list[str] from IngestPipeline (simulate_ingestion splits by |)
            if isinstance(genres, list):
                genre_list = genres
            else:
                # fallback: pipe-separated or space-separated string
                sep = "|" if "|" in str(genres) else " "
                genre_list = str(genres).split(sep)

            ids = [self.genre_to_idx.get(g.strip(), 0) for g in genre_list if g.strip()]
            # Pad to max_genres with 0
            ids = (ids + [0] * self.max_genres)[: self.max_genres]
            return ids

        movies["genre_ids"] = movies["genres"].apply(genres_to_ids)
        print(f"  Genre vocabulary size: {len(self.genre_to_idx) + 1} (including padding=0)")
        return movies

    # ── Task 2.3: Time context ────────────────────────────────────────────────
    def extract_time_context(self) -> pd.DataFrame:
        """Task 2.3: Derive time-of-day slot from Unix rating timestamp.

        Slots:
          0 — Matinee    (06:00–17:59)
          1 — Prime Time (18:00–21:59)
          2 — Late Night (22:00–05:59)
        """
        ratings = self.ratings_df.copy()
        ratings["datetime"] = pd.to_datetime(ratings["timestamp"], unit="s", utc=True)
        ratings["hour"]     = ratings["datetime"].dt.hour

        def time_of_day(hour: int) -> int:
            if 6 <= hour < 18:
                return 0
            if 18 <= hour < 22:
                return 1
            return 2

        ratings["time_of_day"] = ratings["hour"].apply(time_of_day)

        dist = ratings["time_of_day"].value_counts().sort_index()
        print("  Time-of-Day distribution:")
        print(f"    0 Matinee    : {dist.get(0, 0):>8,}")
        print(f"    1 Prime Time : {dist.get(1, 0):>8,}")
        print(f"    2 Late Night : {dist.get(2, 0):>8,}")
        return ratings

    # ── Task 2.4: Build sequences ─────────────────────────────────────────────
    def build_sequences(
        self,
        users_df: pd.DataFrame,
        movies_df: pd.DataFrame,
        ratings_df: pd.DataFrame,
        min_seq_len: int = 5,
    ) -> dict:
        """Task 2.4: Group ratings per user, sorted by timestamp, filter rating >= 3."""
        print("Building user sequences...")

        data = ratings_df.merge(users_df, on="user_id")
        data = data.merge(movies_df[["movie_id", "genre_ids"]], on="movie_id")
        data = data[data["rating"] >= 3].copy()

        # Remap movie IDs to contiguous 1-indexed integers
        unique_movies = sorted(data["movie_id"].unique())
        self.movie_id_map     = {old: new + 1 for new, old in enumerate(unique_movies)}
        self.reverse_movie_map = {v: k for k, v in self.movie_id_map.items()}
        data["item_idx"] = data["movie_id"].map(self.movie_id_map)

        data = data.sort_values(["user_id", "timestamp"])

        sequences: dict = {}
        for user_id, group in data.groupby("user_id"):
            if len(group) < min_seq_len:
                continue
            user_info = group.iloc[0]
            sequences[user_id] = {
                "item_seq":  group["item_idx"].tolist(),
                "genre_seq": group["genre_ids"].tolist(),
                "time_seq":  group["time_of_day"].tolist(),
                "user_profile": {
                    "age_idx":    int(user_info["age_idx"]),
                    "gender_idx": int(user_info["gender_idx"]),
                    "occupation": int(user_info["occupation"]),
                },
                "timestamps": group["timestamp"].tolist(),
            }

        print(f"  Built sequences for {len(sequences):,} users")
        print(f"  Total unique items: {len(self.movie_id_map):,}")
        return sequences

    # ── Task 2.5: Leave-one-out split ─────────────────────────────────────────
    def split_sequences(
        self,
        sequences: dict,
        max_seq_len: int = 50,
    ) -> tuple[list, list, list]:
        """Task 2.5: Leave-one-out split into train / val / test.

        - Test  : last interaction (N)
        - Val   : second-to-last (N-1)
        - Train : sliding windows over the remaining history
        """
        print("Splitting sequences (leave-one-out)...")
        train_data, val_data, test_data = [], [], []

        for user_id, seq in sequences.items():
            items      = seq["item_seq"]
            genres     = seq["genre_seq"]
            times      = seq["time_seq"]
            timestamps = seq["timestamps"]
            profile    = seq["user_profile"]

            if len(items) < 3:
                continue

            # Test
            test_data.append({
                "user_id":        user_id,
                "item_seq":       items[:-1][-max_seq_len:],
                "genre_seq":      genres[:-1][-max_seq_len:],
                "time_seq":       times[:-1][-max_seq_len:],
                "target":         items[-1],
                "target_time":    times[-1],
                "last_timestamp": timestamps[-1],
                "user_profile":   profile,
            })

            # Val
            val_data.append({
                "user_id":        user_id,
                "item_seq":       items[:-2][-max_seq_len:],
                "genre_seq":      genres[:-2][-max_seq_len:],
                "time_seq":       times[:-2][-max_seq_len:],
                "target":         items[-2],
                "target_time":    times[-2],
                "last_timestamp": timestamps[-2],
                "user_profile":   profile,
            })

            # Train: sliding windows over history[:-2]
            history_items      = items[:-2]
            history_genres     = genres[:-2]
            history_times      = times[:-2]
            history_timestamps = timestamps[:-2]
            for i in range(1, len(history_items)):
                train_data.append({
                    "user_id":        user_id,
                    "item_seq":       history_items[:i][-max_seq_len:],
                    "genre_seq":      history_genres[:i][-max_seq_len:],
                    "time_seq":       history_times[:i][-max_seq_len:],
                    "target":         history_items[i],
                    "target_time":    history_times[i],
                    "last_timestamp": history_timestamps[i],
                    "user_profile":   profile,
                })

        print(f"  Train: {len(train_data):,}  Val: {len(val_data):,}  Test: {len(test_data):,}")
        return train_data, val_data, test_data

    # ── Full pipeline ─────────────────────────────────────────────────────────
    def process_all(self) -> tuple[list, list, list, dict]:
        """Run the full preprocessing pipeline end-to-end.

        Steps:
          1. Load raw Parquet from s3://raw-data/
          2. Feature engineering (tasks 2.1-2.3)
          3. Build user sequences (task 2.4)
          4. Leave-one-out split (task 2.5)
          5. Write Delta Lake tables to s3://offline-store/delta/train|val|test
          6. Write metadata.json to s3://offline-store/metadata.json

        Returns:
            (train_data, val_data, test_data, metadata)
        """
        # 1. Load
        self.load_data()

        # 2. Feature engineering
        print("\n[Feature engineering]")
        users_df   = self.encode_static_features()
        movies_df  = self.encode_genres()
        ratings_df = self.extract_time_context()

        # 3. Sequences
        print("\n[Building sequences]")
        sequences = self.build_sequences(users_df, movies_df, ratings_df)

        # 4. Split
        print("\n[Splitting]")
        train_data, val_data, test_data = self.split_sequences(sequences)

        # 5. Metadata
        metadata = {
            "num_items":           len(self.movie_id_map) + 1,
            "num_genres":          len(self.GENRES) + 1,
            "num_ages":            len(self.age_encoder),
            "num_genders":         2,
            "num_occupations":     21,
            "num_time_slots":      3,
            "max_genres_per_item": self.max_genres,
            "genre_to_idx":        self.genre_to_idx,
            "age_encoder":         {int(k): v for k, v in self.age_encoder.items()},
            "movie_id_map":        {int(k): int(v) for k, v in self.movie_id_map.items()},
            "reverse_movie_map":   {int(k): int(v) for k, v in self.reverse_movie_map.items()},
        }

        # 6. Save to Delta Lake in s3://offline-store/delta/
        print("\n[Saving to Delta Lake → s3://offline-store/delta/]")
        for split_name, split_data in [("train", train_data), ("val", val_data), ("test", test_data)]:
            df  = _records_to_df(split_data)
            uri = f"s3://{DELTA_BUCKET}/delta/{split_name}"
            _write_delta(uri, df, split_name)

        # Save metadata JSON
        client = _s3_client()
        client.put_object(
            Bucket=DELTA_BUCKET,
            Key="metadata.json",
            Body=json.dumps(metadata, indent=2).encode(),
            ContentType="application/json",
        )
        print(f"  Metadata → s3://{DELTA_BUCKET}/metadata.json")

        return train_data, val_data, test_data, metadata


# ── Entry point ───────────────────────────────────────────────────────────────
def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )
    preprocessor = MovieLensPreprocessor()
    _, _, _, metadata = preprocessor.process_all()

    print("\n[Done] Metadata summary:")
    for k, v in metadata.items():
        if not isinstance(v, dict):
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
