"""
Materialization jobs:

1. run_materialization()       — original generic pipeline (Delta → Feast → Redis)
2. run_movielens_materialization() — NEW: processes raw MovieLens Parquet files
   from s3://raw-data/ into three Feast-ready staged Parquets plus a metadata.json.

   Data flow for Mamba4Rec:
     s3://raw-data/new_user/**/*.parquet
     s3://raw-data/new_movie/**/*.parquet      →  DuckDB (in-memory)  →
     s3://raw-data/new_rating/**/*.parquet         feature engineering    →
                                                                           ↓
     s3://offline-store/parquet/user_features/staged.parquet
     s3://offline-store/parquet/movie_features/staged.parquet
     s3://offline-store/parquet/rating_events/staged.parquet
     s3://offline-store/parquet/metadata.json
"""

import json
import logging
import os
from datetime import datetime, timedelta

import pandas as pd

from feast import FeatureStore
from src.core.config import settings
from src.core.duckdb_client import get_duckdb_connection, register_delta_as_table, stage_to_parquet
from src.core.storage import upload_bytes

logger = logging.getLogger(__name__)

# ── MovieLens genre mapping (fixed for ML-1M) ────────────────────────────────
GENRE_TO_IDX: dict[str, int] = {
    "Action": 1,
    "Adventure": 2,
    "Animation": 3,
    "Children's": 4,
    "Comedy": 5,
    "Crime": 6,
    "Documentary": 7,
    "Drama": 8,
    "Fantasy": 9,
    "Film-Noir": 10,
    "Horror": 11,
    "Musical": 12,
    "Mystery": 13,
    "Romance": 14,
    "Sci-Fi": 15,
    "Thriller": 16,
    "War": 17,
    "Western": 18,
}

AGE_TO_IDX: dict[int, int] = {1: 0, 18: 1, 25: 2, 35: 3, 45: 4, 50: 5, 56: 6}

MAMBA_METADATA_KEY = "parquet/metadata.json"
OFFLINE_BUCKET = "offline-store"

# S3 paths
DELTA_PATH = "s3://offline-store/delta/train"
PARQUET_OUTPUT = "s3://offline-store/parquet/users/staged.parquet"

# Storage options for deltalake (boto3 style)
STORAGE_OPTIONS = {
    "endpoint_url": settings.minio_endpoint,
    "aws_access_key_id": settings.aws_access_key_id,
    "aws_secret_access_key": settings.aws_secret_access_key,
    "aws_allow_http": "true",
    "aws_region": "us-east-1",
}

# SQL applied inside DuckDB before writing staged Parquet.
# Extend with any feature engineering needed before materialization.
STAGING_SQL = """
    SELECT
        user_id,
        event_timestamp,
        gender_idx,
        age_idx,
        occupation,
        target,
        target_time
    FROM train
    WHERE event_timestamp IS NOT NULL
"""


def _set_aws_env() -> None:
    """
    Set AWS_* environment variables so that Feast's DuckDB offline store
    can configure its own httpfs session when executing queries.
    """
    endpoint = settings.minio_endpoint.replace("http://", "").replace("https://", "")
    os.environ.setdefault("AWS_ACCESS_KEY_ID", settings.aws_access_key_id)
    os.environ.setdefault("AWS_SECRET_ACCESS_KEY", settings.aws_secret_access_key)
    os.environ.setdefault("AWS_ENDPOINT_URL", settings.minio_endpoint)
    os.environ.setdefault("AWS_S3_ENDPOINT", endpoint)


def delta_to_duckdb_to_parquet() -> int:
    """
    Stage data: Delta Lake → DuckDB (in-memory) → Parquet on MinIO.

    Returns:
        Number of rows staged.
    """
    conn = get_duckdb_connection()

    row_count = register_delta_as_table(
        conn=conn,
        table_name="train",
        delta_path=DELTA_PATH,
        storage_options=STORAGE_OPTIONS,
    )
    logger.info("Loaded %d rows from Delta Lake into DuckDB", row_count)

    stage_to_parquet(
        conn=conn,
        query=STAGING_SQL,
        parquet_s3_path=PARQUET_OUTPUT,
    )
    logger.info("Staged Parquet written to %s", PARQUET_OUTPUT)
    conn.close()
    return row_count


def run_materialization(end_date: datetime | None = None) -> None:
    """
    Full materialization pipeline:
      1. Delta → DuckDB → staged Parquet (MinIO)
      2. Feast DuckDB offline store → Redis online store

    Args:
        end_date: Upper bound for feature timestamps. Defaults to now (UTC).
    """
    _set_aws_env()

    rows = delta_to_duckdb_to_parquet()
    if rows == 0:
        logger.warning("No rows found in Delta table — skipping Feast materialization")
        return

    store = FeatureStore(repo_path="feast/")
    end = end_date or datetime.utcnow()
    start = end - timedelta(days=7)

    store.materialize(start_date=start, end_date=end)
    logger.info(
        "Feast materialization complete: %d rows, window [%s, %s]",
        rows,
        start.isoformat(),
        end.isoformat(),
    )


def _encode_genres(genres_list, max_genres: int = 3) -> list:
    """Map a list of genre strings to padded integer index list."""
    if genres_list is None:
        return [0] * max_genres
    indices = [GENRE_TO_IDX.get(g, 0) for g in genres_list][:max_genres]
    while len(indices) < max_genres:
        indices.append(0)
    return indices


def run_movielens_materialization() -> dict:
    """
    Stage raw MovieLens Parquet files (written by the Kafka ingest pipeline) into
    three Feast-compatible Parquet files and a metadata JSON, all on MinIO.

    Reads from:
        s3://raw-data/new_user/**/*.parquet
        s3://raw-data/new_movie/**/*.parquet
        s3://raw-data/new_rating/**/*.parquet

    Writes to:
        s3://offline-store/parquet/user_features/staged.parquet
        s3://offline-store/parquet/movie_features/staged.parquet
        s3://offline-store/parquet/rating_events/staged.parquet
        s3://offline-store/parquet/metadata.json

    Returns:
        metadata dict with vocab sizes and ID mappings.

    This function is idempotent — re-running it overwrites existing staged files.
    """
    _set_aws_env()
    conn = get_duckdb_connection()

    logger.info("Loading raw MovieLens Parquet files from MinIO …")

    # ── Read raw Parquet files from MinIO ──────────────────────────────────────
    conn.execute("""
        CREATE OR REPLACE VIEW new_user AS
        SELECT * FROM read_parquet('s3://raw-data/new_user/**/*.parquet')
    """)
    conn.execute("""
        CREATE OR REPLACE VIEW new_movie AS
        SELECT * FROM read_parquet('s3://raw-data/new_movie/**/*.parquet')
    """)
    conn.execute("""
        CREATE OR REPLACE VIEW new_rating AS
        SELECT * FROM read_parquet('s3://raw-data/new_rating/**/*.parquet')
    """)

    user_count  = conn.execute("SELECT COUNT(*) FROM new_user").fetchone()[0]
    movie_count = conn.execute("SELECT COUNT(*) FROM new_movie").fetchone()[0]
    rating_count = conn.execute("SELECT COUNT(*) FROM new_rating").fetchone()[0]
    logger.info("Loaded: %d users, %d movies, %d ratings", user_count, movie_count, rating_count)

    if user_count == 0 or movie_count == 0 or rating_count == 0:
        raise RuntimeError(
            "Raw data is missing in MinIO. "
            "Please run the simulator first: python scripts/simulate_ingestion.py"
        )

    # ── Stage 1: User features ─────────────────────────────────────────────────
    logger.info("Staging user features …")
    stage_to_parquet(
        conn=conn,
        query="""
            SELECT
                user_id,
                MAX(event_timestamp)  AS event_timestamp,
                CASE gender WHEN 'F' THEN 0 ELSE 1 END AS gender_idx,
                CASE age
                    WHEN 1  THEN 0
                    WHEN 18 THEN 1
                    WHEN 25 THEN 2
                    WHEN 35 THEN 3
                    WHEN 45 THEN 4
                    WHEN 50 THEN 5
                    ELSE 6
                END AS age_idx,
                occupation
            FROM new_user
            GROUP BY user_id, gender, age, occupation
        """,
        parquet_s3_path="s3://offline-store/parquet/user_features/staged.parquet",
    )

    # ── Stage 2: Movie features (re-index IDs + encode genres) ────────────────
    logger.info("Staging movie features …")
    movie_df: pd.DataFrame = conn.execute(
        "SELECT movie_id, genres, MAX(event_timestamp) AS event_timestamp "
        "FROM new_movie GROUP BY movie_id, genres"
    ).df()

    movie_df = movie_df.sort_values("movie_id").reset_index(drop=True)
    movie_df["internal_movie_id"] = range(1, len(movie_df) + 1)

    genre_indices = movie_df["genres"].apply(_encode_genres)
    movie_df["genre_idx_0"] = genre_indices.apply(lambda x: x[0])
    movie_df["genre_idx_1"] = genre_indices.apply(lambda x: x[1])
    movie_df["genre_idx_2"] = genre_indices.apply(lambda x: x[2])

    movie_features_df = movie_df[[
        "movie_id", "internal_movie_id", "event_timestamp",
        "genre_idx_0", "genre_idx_1", "genre_idx_2",
    ]]

    # Register as DuckDB table and write to S3
    conn.register("movie_features_tbl", movie_features_df)
    stage_to_parquet(
        conn=conn,
        query="SELECT * FROM movie_features_tbl",
        parquet_s3_path="s3://offline-store/parquet/movie_features/staged.parquet",
    )

    # Build movie_id ↔ internal_id maps for metadata
    movie_id_map: dict[int, int] = dict(zip(
        movie_df["movie_id"].astype(int),
        movie_df["internal_movie_id"].astype(int),
    ))
    reverse_movie_map: dict[int, int] = {v: k for k, v in movie_id_map.items()}

    # Also store title information (used at inference)
    movies_info_df = conn.execute(
        "SELECT movie_id, title, genres FROM new_movie"
    ).df().drop_duplicates("movie_id")

    # ── Stage 3: Rating events ─────────────────────────────────────────────────
    logger.info("Staging rating events …")
    conn.register("movie_id_map_tbl", movie_df[["movie_id", "internal_movie_id"]])
    stage_to_parquet(
        conn=conn,
        query="""
            SELECT
                r.user_id,
                r.event_timestamp,
                m.internal_movie_id,
                CAST(r.rating AS FLOAT)  AS rating,
                CASE
                    WHEN EXTRACT(HOUR FROM to_timestamp(r.rating_timestamp)) BETWEEN 6  AND 17 THEN 0
                    WHEN EXTRACT(HOUR FROM to_timestamp(r.rating_timestamp)) BETWEEN 18 AND 21 THEN 1
                    ELSE 2
                END AS time_slot
            FROM new_rating r
            JOIN movie_id_map_tbl m ON r.movie_id = m.movie_id
            WHERE r.rating >= 3
        """,
        parquet_s3_path="s3://offline-store/parquet/rating_events/staged.parquet",
    )

    # ── Stage 4: Metadata JSON ─────────────────────────────────────────────────
    num_users = int(conn.execute("SELECT COUNT(DISTINCT user_id) FROM new_user").fetchone()[0])
    num_occupations = int(conn.execute("SELECT MAX(occupation)+1 FROM new_user").fetchone()[0])

    metadata: dict = {
        "num_items":           len(movie_id_map) + 1,   # +1 for padding idx 0
        "num_genres":          len(GENRE_TO_IDX) + 1,   # +1 for padding idx 0
        "num_ages":            7,
        "num_genders":         2,
        "num_occupations":     max(21, num_occupations),
        "num_time_slots":      3,
        "max_genres_per_item": 3,
        "num_users":           num_users,
        "movie_id_map":        {str(k): v for k, v in movie_id_map.items()},
        "reverse_movie_map":   {str(k): v for k, v in reverse_movie_map.items()},
        "genre_to_idx":        GENRE_TO_IDX,
    }

    upload_bytes(
        bucket=OFFLINE_BUCKET,
        key=MAMBA_METADATA_KEY,
        data=json.dumps(metadata).encode(),
        content_type="application/json",
    )

    # Upload movies_info as JSON for inference-time title lookups
    movies_info_records = movies_info_df.to_dict(orient="records")
    upload_bytes(
        bucket=OFFLINE_BUCKET,
        key="parquet/movies_info.json",
        data=json.dumps(movies_info_records, default=list).encode(),
        content_type="application/json",
    )

    conn.close()
    logger.info(
        "Mamba4Rec materialization complete. "
        "num_items=%d, num_genres=%d, num_users=%d",
        metadata["num_items"],
        metadata["num_genres"],
        metadata["num_users"],
    )
    return metadata


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_materialization()
