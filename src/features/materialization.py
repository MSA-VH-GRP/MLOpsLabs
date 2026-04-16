"""
Materialization job: Delta Lake → Parquet → Feast → Redis.

Steps:
1. Read Delta table from s3://offline-store/delta/raw_events/
2. Write as Parquet to s3://offline-store/parquet/raw_events/
3. Run `feast materialize-incremental` to push features into Redis
"""

import logging
from datetime import datetime, timedelta

import pyarrow.parquet as pq
from deltalake import DeltaTable
from feast import FeatureStore

from src.core.config import settings

logger = logging.getLogger(__name__)

DELTA_PATH = "s3://offline-store/delta/raw_events"
PARQUET_PATH = "s3://offline-store/parquet/raw_events"

STORAGE_OPTIONS = {
    "endpoint_url": settings.minio_endpoint,
    "aws_access_key_id": settings.aws_access_key_id,
    "aws_secret_access_key": settings.aws_secret_access_key,
}


def delta_to_parquet() -> None:
    dt = DeltaTable(DELTA_PATH, storage_options=STORAGE_OPTIONS)
    arrow_table = dt.to_pyarrow()
    pq.write_to_dataset(
        arrow_table,
        root_path=PARQUET_PATH,
        filesystem=_get_s3fs(),
    )
    logger.info("Wrote %d rows to Parquet", len(arrow_table))


def run_materialization(end_date: datetime | None = None) -> None:
    delta_to_parquet()
    store = FeatureStore(repo_path="feast/")
    end = end_date or datetime.utcnow()
    start = end - timedelta(days=7)
    store.materialize(start_date=start, end_date=end)
    logger.info("Materialization complete up to %s", end.isoformat())


def _get_s3fs():
    import s3fs
    return s3fs.S3FileSystem(
        endpoint_url=settings.minio_endpoint,
        key=settings.aws_access_key_id,
        secret=settings.aws_secret_access_key,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_materialization()
