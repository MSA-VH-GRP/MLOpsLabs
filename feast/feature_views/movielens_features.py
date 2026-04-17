"""
Feast feature definitions for MovieLens / Mamba4Rec.

Three feature views staged by src/features/materialization.run_movielens_materialization():

  user_profile_features   — static user demographics (age, gender, occupation)
  movie_genre_features    — static movie attributes (re-indexed ID + genre indices)
  rating_event_features   — per-interaction events (internal_movie_id, rating, time_slot)

Data lineage:
  simulate_ingestion.py
    ↓  Kafka topics: new-user / new-movie / new-rating
  IngestPipeline  →  s3://raw-data/{topic}/**/*.parquet
    ↓  run_movielens_materialization()  (DuckDB + Python)
  s3://offline-store/parquet/user_features/staged.parquet
  s3://offline-store/parquet/movie_features/staged.parquet
  s3://offline-store/parquet/rating_events/staged.parquet
    ↓  feast apply
  Feast DuckDB offline store  (used in get_historical_features Point-in-Time joins)
"""

from datetime import timedelta

from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float32, Int64, String

# ── Entities ─────────────────────────────────────────────────────────────────

user_entity = Entity(
    name="user_id",
    description="MovieLens user identifier",
)

movie_entity = Entity(
    name="movie_id",
    description="MovieLens movie identifier (original ID)",
)

# ── Offline sources ───────────────────────────────────────────────────────────

_S3_ENDPOINT = "http://minio:9000"

user_profile_source = FileSource(
    path="s3://offline-store/parquet/user_features/staged.parquet",
    timestamp_field="event_timestamp",
    s3_endpoint_override=_S3_ENDPOINT,
)

movie_genre_source = FileSource(
    path="s3://offline-store/parquet/movie_features/staged.parquet",
    timestamp_field="event_timestamp",
    s3_endpoint_override=_S3_ENDPOINT,
)

rating_event_source = FileSource(
    path="s3://offline-store/parquet/rating_events/staged.parquet",
    timestamp_field="event_timestamp",
    s3_endpoint_override=_S3_ENDPOINT,
)

# ── Feature Views ─────────────────────────────────────────────────────────────

user_profile_features = FeatureView(
    name="user_profile_features",
    entities=[user_entity],
    ttl=timedelta(days=3650),   # static — effectively never expires
    schema=[
        Field(name="gender_idx",  dtype=Int64),
        Field(name="age_idx",     dtype=Int64),
        Field(name="occupation",  dtype=Int64),
    ],
    source=user_profile_source,
    description="Static user demographic features encoded for Mamba4Rec",
)

movie_genre_features = FeatureView(
    name="movie_genre_features",
    entities=[movie_entity],
    ttl=timedelta(days=3650),   # static — effectively never expires
    schema=[
        Field(name="internal_movie_id", dtype=Int64),
        Field(name="genre_idx_0",       dtype=Int64),
        Field(name="genre_idx_1",       dtype=Int64),
        Field(name="genre_idx_2",       dtype=Int64),
    ],
    source=movie_genre_source,
    description="Movie features: re-indexed ID and top-3 genre indices (0=padding)",
)

rating_event_features = FeatureView(
    name="rating_event_features",
    entities=[user_entity],
    ttl=timedelta(days=3650),
    schema=[
        Field(name="internal_movie_id", dtype=Int64),
        Field(name="rating",            dtype=Float32),
        Field(name="time_slot",         dtype=Int64),
    ],
    source=rating_event_source,
    description=(
        "Per-user rating events after re-indexing and time-slot encoding. "
        "Used for Point-in-Time historical feature retrieval in Mamba4Rec training."
    ),
)
