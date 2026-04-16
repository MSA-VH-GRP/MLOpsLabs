"""
Raw feature definitions for the DuckDB offline store.

The offline store uses DuckDB (via httpfs) to query Parquet files staged on
MinIO by materialization.py. The FileSource below points to the output of
that staging step: s3://offline-store/parquet/raw_events/staged.parquet

Data lineage:
  Kafka  →  Delta Lake (s3://offline-store/delta/)
         →  DuckDB staging (src/features/materialization.py)
         →  Parquet      (s3://offline-store/parquet/raw_events/)
         →  Feast DuckDB offline store  ←── defined here
         →  Redis online store (via feast materialize)
"""

from datetime import timedelta

from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float32, Int64, String

# ── Entity ──────────────────────────────────────────────────────────────────
event_entity = Entity(
    name="event_id",
    description="Unique identifier for each ingested event",
)

# ── Offline source ───────────────────────────────────────────────────────────
# Points to the Parquet file written by DuckDB's stage_to_parquet().
# The DuckDB offline store reads this file via its httpfs S3 extension.
# S3 credentials are injected via AWS_* environment variables at runtime
# (set in materialization.py → _set_aws_env()).
raw_event_source = FileSource(
    path="s3://offline-store/parquet/raw_events/staged.parquet",
    timestamp_field="event_timestamp",
    s3_endpoint_override="http://minio:9000",
)

# ── Feature View ─────────────────────────────────────────────────────────────
# Schema matches the columns produced by STAGING_SQL in materialization.py.
raw_event_feature_view = FeatureView(
    name="raw_event_features",
    entities=[event_entity],
    ttl=timedelta(days=7),
    schema=[
        Field(name="feature_1", dtype=Float32),
        Field(name="feature_2", dtype=Float32),
        Field(name="category",  dtype=String),
        Field(name="count",     dtype=Int64),
    ],
    source=raw_event_source,
    description="Features extracted from raw events — staged via DuckDB from Delta Lake",
)
