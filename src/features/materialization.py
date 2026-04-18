"""
Materialization job: Delta Lake → DuckDB staging → Parquet → Feast → Redis.

Revised data flow
-----------------
                               ┌─────────────────────────────────┐
  Kafka consumer               │         MinIO (S3)               │
  writes Delta ──────────────► │  s3://offline-store/delta/       │
                               │  raw_events/                     │
                               └────────────┬────────────────────┘
                                            │ deltalake → PyArrow
                                            ▼
                               ┌─────────────────────────────────┐
                               │   DuckDB (in-memory)             │
                               │   register_delta_as_table()      │
                               │   → SQL transforms               │
                               │   → stage_to_parquet()           │
                               └────────────┬────────────────────┘
                                            │ httpfs COPY TO S3
                                            ▼
                               ┌─────────────────────────────────┐
                               │         MinIO (S3)               │
                               │  s3://offline-store/parquet/     │
                               │  raw_events/staged.parquet       │
                               └────────────┬────────────────────┘
                                            │ Feast DuckDB offline store
                                            ▼
                               ┌─────────────────────────────────┐
                               │   Redis (online store)           │
                               │   feast.materialize()            │
                               └─────────────────────────────────┘
"""

import logging
import os
from datetime import datetime, timedelta

from feast import FeatureStore

from src.core.config import settings
from src.core.duckdb_client import (
    get_duckdb_connection,
    register_delta_as_table,
    stage_to_parquet,
)

logger = logging.getLogger(__name__)

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


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_materialization()
