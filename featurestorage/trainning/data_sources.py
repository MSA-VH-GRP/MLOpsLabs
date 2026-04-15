"""
Data source descriptors for Delta Lake tables stored on MinIO.

Each ``DeltaLakeSource`` wraps one Delta table (train / val / test) and
exposes two representations:

1. ``scan_sql``            — the DuckDB ``delta_scan()`` fragment used by
                             DuckDBDeltaEngine to execute offline queries.
2. ``to_feast_file_source`` — a Feast ``FileSource`` used to register the
                             source in the feature registry for lineage and
                             schema tracking.

The Feast FileSource metadata points to the same S3 URI that DuckDB reads.
When Feast needs to execute a query it delegates to DuckDBDeltaEngine rather
than using its own file reader, so the FileSource is registry-only metadata.
"""
from __future__ import annotations

from dataclasses import dataclass

from feast import FileSource
from feast.data_format import ParquetFormat

from .config import MINIO_CONFIG


@dataclass(frozen=True)
class DeltaLakeSource:
    """
    Descriptor for a single Delta Lake table on MinIO.

    Attributes
    ----------
    name:            Feast source name (appears in the registry).
    table:           Sub-path inside the processed bucket (e.g. ``"train"``).
    timestamp_field: Column used as the event timestamp for point-in-time joins.
    description:     Human-readable label shown in the Feast UI / registry.
    """

    name: str
    table: str
    timestamp_field: str
    description: str = ""

    @property
    def delta_uri(self) -> str:
        """Full S3 URI: ``s3://processed/<table>``."""
        return MINIO_CONFIG.delta_uri(self.table)

    @property
    def scan_sql(self) -> str:
        """
        DuckDB ``delta_scan()`` fragment for inline use in SQL strings.

        Example::

            sql = f"SELECT * FROM {source.scan_sql} WHERE user_id = 42"
        """
        return f"delta_scan('{self.delta_uri}')"

    def to_feast_file_source(self) -> FileSource:
        """
        Feast ``FileSource`` for registry metadata.

        The actual data reading is handled by ``DuckDBDeltaEngine``, not
        Feast's file reader, so ParquetFormat is used as a format hint only.
        """
        return FileSource(
            name=self.name,
            path=self.delta_uri,
            file_format=ParquetFormat(),
            timestamp_field=self.timestamp_field,
            description=self.description,
        )


# ── Registered Delta Lake sources ─────────────────────────────────────────────
# These match the three Delta tables written by preprocess.py:
#   s3://processed/train   s3://processed/val   s3://processed/test

train_source = DeltaLakeSource(
    name="movielens_train",
    table="train",
    timestamp_field="target_time",
    description="MovieLens 1M training split — sliding-window user interaction sequences",
)

val_source = DeltaLakeSource(
    name="movielens_val",
    table="val",
    timestamp_field="target_time",
    description="MovieLens 1M validation split — leave-one-out second-to-last item",
)

test_source = DeltaLakeSource(
    name="movielens_test",
    table="test",
    timestamp_field="target_time",
    description="MovieLens 1M test split — leave-one-out last item",
)
