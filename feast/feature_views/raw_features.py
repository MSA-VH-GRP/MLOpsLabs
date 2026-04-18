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
from feast.types import Int64

# ── Entity ──────────────────────────────────────────────────────────────────
user_entity = Entity(
    name="user_id",
    description="Unique identifier for each user",
)

# ── Offline source ───────────────────────────────────────────────────────────
# Points to the Parquet file written by DuckDB's stage_to_parquet().
# Source: delta/train → materialization.py STAGING_SQL → this Parquet.
# S3 credentials are injected via AWS_* environment variables at runtime
# (set in materialization.py → _set_aws_env()).
raw_event_source = FileSource(
    path="s3://offline-store/parquet/users/staged.parquet",
    timestamp_field="event_timestamp",
    s3_endpoint_override="http://minio:9000",
)

# ── Feature View ─────────────────────────────────────────────────────────────
# Schema matches the flat columns produced by STAGING_SQL in materialization.py.
# item_seq / genre_seq / time_seq stay in Delta only — consumed by trainer.py.
raw_event_feature_view = FeatureView(
    name="raw_event_features",
    entities=[user_entity],
    ttl=timedelta(days=7),
    schema=[
        Field(name="gender_idx",  dtype=Int64),
        Field(name="age_idx",     dtype=Int64),
        Field(name="occupation",  dtype=Int64),
        Field(name="target",      dtype=Int64),
        Field(name="target_time", dtype=Int64),
    ],
    source=raw_event_source,
    description="Flat user features from MovieLens — staged via DuckDB from delta/train",
)
