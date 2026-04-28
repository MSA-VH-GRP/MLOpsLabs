"""
Ingest pipeline: Kafka → Plain Parquet on MinIO.

Reads raw events from Kafka and writes them as partitioned plain Parquet files
into s3://raw-data/<table_name>/ using the pandas and pyarrow libraries.
"""

import logging
import uuid

import pandas as pd

from src.core.config import settings
from src.pipelines.consumer import BaseConsumer
from src.pipelines.topics import NEW_MOVIE, NEW_RATING, NEW_USER, RAW_EVENTS

logger = logging.getLogger(__name__)

STORAGE_OPTIONS = {
    "client_kwargs": {
        "endpoint_url": settings.minio_endpoint,
        "use_ssl": False,
    },
    "key": settings.aws_access_key_id,
    "secret": settings.aws_secret_access_key,
}

def get_table_path(topic: str) -> str:
    return f"s3://raw-data/{topic.replace('-', '_')}"

class IngestPipeline(BaseConsumer):
    def __init__(self):
        super().__init__(topics=[RAW_EVENTS, NEW_USER, NEW_MOVIE, NEW_RATING])
        self._buffers: dict[str, list[dict]] = {}
        self._batch_size = 1000

    def process(self, message: dict, topic: str) -> None:
        if topic not in self._buffers:
            self._buffers[topic] = []
        self._buffers[topic].append(message)
        if len(self._buffers[topic]) >= self._batch_size:
            self._flush(topic)

    def _flush(self, topic: str = None) -> None:
        topics_to_flush = [topic] if topic else list(self._buffers.keys())
        for t in topics_to_flush:
            buffer = self._buffers.get(t, [])
            if not buffer:
                continue
            
            df = pd.DataFrame(buffer)
            # Support both event_timestamp and specific record timestamps depending on topic
            if "timestamp" in df.columns:
                df["event_timestamp"] = pd.to_datetime(df["timestamp"], format='ISO8601')
            elif "event_timestamp" not in df.columns:
                df["event_timestamp"] = pd.Timestamp.now(tz="UTC")
                
            # Time partitioning columns
            df["year"] = df["event_timestamp"].dt.year
            df["month"] = df["event_timestamp"].dt.month
            df["day"] = df["event_timestamp"].dt.day
            
            table_path = get_table_path(t)
            
            # Write each partition as a separate plain Parquet file manually to assure exact schema
            for (year, month, day), group in df.groupby(["year", "month", "day"]):
                partition_path = f"{table_path}/year={year}/month={month:02d}/day={day:02d}"
                file_name = f"part-{uuid.uuid4().hex}.parquet"
                full_path = f"{partition_path}/{file_name}"

                group.drop(columns=["year", "month", "day"]).to_parquet(
                    full_path, 
                    storage_options=STORAGE_OPTIONS, 
                    engine="pyarrow", 
                    index=False
                )
                logger.info("Flushed %d records to Parquet (path: %s)", len(group), full_path)
            
            buffer.clear()

    def shutdown(self) -> None:
        """Flush remaining buffers on clean shutdown."""
        self._flush()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    IngestPipeline().run()
