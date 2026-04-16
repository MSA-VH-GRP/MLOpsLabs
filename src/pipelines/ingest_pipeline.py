"""
Ingest pipeline: Kafka → Delta Lake on MinIO.

Reads raw events from Kafka and writes them as Delta Lake tables
into s3://offline-store/delta/<table_name>/ using the deltalake library.
"""

import logging

import pandas as pd
from deltalake import write_deltalake

from src.core.config import settings
from src.pipelines.consumer import BaseConsumer
from src.pipelines.topics import RAW_EVENTS

logger = logging.getLogger(__name__)

DELTA_TABLE_PATH = "s3://offline-store/delta/raw_events"

STORAGE_OPTIONS = {
    "endpoint_url": settings.minio_endpoint,
    "aws_access_key_id": settings.aws_access_key_id,
    "aws_secret_access_key": settings.aws_secret_access_key,
    "allow_unsafe_rename": "true",
}


class IngestPipeline(BaseConsumer):
    def __init__(self):
        super().__init__(topics=[RAW_EVENTS])
        self._buffer: list[dict] = []
        self._batch_size = 100

    def process(self, message: dict) -> None:
        self._buffer.append(message)
        if len(self._buffer) >= self._batch_size:
            self._flush()

    def _flush(self) -> None:
        if not self._buffer:
            return
        df = pd.DataFrame(self._buffer)
        df["event_timestamp"] = pd.to_datetime(df["timestamp"])
        write_deltalake(
            DELTA_TABLE_PATH,
            df,
            storage_options=STORAGE_OPTIONS,
            mode="append",
        )
        logger.info("Flushed %d records to Delta Lake", len(self._buffer))
        self._buffer.clear()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    IngestPipeline().run()
