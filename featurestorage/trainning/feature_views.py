"""
Feast FeatureView definitions for the MovieLens 1M dataset.

Two feature groups are registered:

┌─────────────────────┬────────────────────────────────────────────────────────┐
│ View                │ Features                                               │
├─────────────────────┼────────────────────────────────────────────────────────┤
│ user_profile        │ age_idx, gender_idx, occupation                        │
│                     │ Static demographics — constant across all rows for a   │
│                     │ given user_id.                                          │
├─────────────────────┼────────────────────────────────────────────────────────┤
│ user_interactions   │ item_seq, genre_seq, time_seq,                         │
│                     │ target, target_time                                     │
│                     │ Sequential interaction history up to a point in time.  │
│                     │ Each row is one training example (sliding window).     │
└─────────────────────┴────────────────────────────────────────────────────────┘

Column schemas mirror the Delta Lake tables written by preprocess.py
(_records_to_df function).  Feast uses these definitions for registry
metadata and schema validation; the actual data fetch is executed by
DuckDBDeltaEngine via delta_scan().
"""
from datetime import timedelta

from feast import FeatureView, Field
from feast.types import Array, Int64

from .data_sources import train_source
from .entities import user_entity


# ── User Profile ──────────────────────────────────────────────────────────────

user_profile_view = FeatureView(
    name="user_profile",
    entities=[user_entity],
    ttl=timedelta(days=365),
    schema=[
        Field(
            name="age_idx",
            dtype=Int64,
            description="Label-encoded age group (0–N, encoding stored in metadata.json)",
        ),
        Field(
            name="gender_idx",
            dtype=Int64,
            description="Label-encoded gender: 0 = F, 1 = M",
        ),
        Field(
            name="occupation",
            dtype=Int64,
            description="Occupation code as-is from MovieLens (0–20)",
        ),
    ],
    source=train_source.to_feast_file_source(),
    description="Static user demographic features, denormalised into every split row",
    tags={"team": "mlops", "feature_group": "user", "model": "mamba4rec"},
)


# ── User Interaction Sequences ────────────────────────────────────────────────

user_interaction_view = FeatureView(
    name="user_interactions",
    entities=[user_entity],
    ttl=timedelta(days=365),
    schema=[
        Field(
            name="item_seq",
            dtype=Array(Int64),
            description=(
                "Chronological sequence of re-indexed item IDs seen by the user "
                "(max length 50, as produced by the sliding-window split)"
            ),
        ),
        Field(
            name="genre_seq",
            dtype=Array(Int64),
            description=(
                "Flattened genre IDs per item in item_seq "
                "(each item has up to max_genres=3 genre IDs, padded with 0)"
            ),
        ),
        Field(
            name="time_seq",
            dtype=Array(Int64),
            description="Time-of-day bucket per interaction: 0=Matinee, 1=Prime Time, 2=Late Night",
        ),
        Field(
            name="target",
            dtype=Int64,
            description="Next item to predict (the training label)",
        ),
        Field(
            name="target_time",
            dtype=Int64,
            description="Time-of-day bucket of the target interaction",
        ),
    ],
    source=train_source.to_feast_file_source(),
    description=(
        "Per-user sequential interaction history for next-item prediction. "
        "One row per sliding-window training example."
    ),
    tags={"team": "mlops", "feature_group": "sequence", "model": "mamba4rec"},
)
