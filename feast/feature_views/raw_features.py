"""
Raw feature definitions.

Defines:
- Entity: the primary key for feature lookups (e.g. event_id or user_id)
- FileSource: points to materialized Parquet files in MinIO offline-store bucket
- FeatureView: registers feature schema with Feast
"""

from datetime import timedelta

from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float32, Int64, String

# ── Entity ──────────────────────────────────────────────────────────────────
# Replace 'event_id' with your actual primary key (e.g. user_id, item_id).
event_entity = Entity(
    name="event_id",
    description="Unique identifier for each ingested event",
)

# ── Offline source ───────────────────────────────────────────────────────────
# Points to the Parquet staging prefix written by materialization.py.
# When running outside Docker use the full s3:// path; inside Docker the
# API container mounts the feast/ directory and uses the same path.
raw_event_source = FileSource(
    path="s3://offline-store/parquet/raw_events/",
    timestamp_field="event_timestamp",
    s3_endpoint_override="http://minio:9000",
)

# ── Feature View ─────────────────────────────────────────────────────────────
raw_event_feature_view = FeatureView(
    name="raw_event_features",
    entities=[event_entity],
    ttl=timedelta(days=7),
    schema=[
        Field(name="feature_1", dtype=Float32),
        Field(name="feature_2", dtype=Float32),
        Field(name="category", dtype=String),
        Field(name="count", dtype=Int64),
    ],
    source=raw_event_source,
    description="Raw features extracted from ingested Kafka events",
)
