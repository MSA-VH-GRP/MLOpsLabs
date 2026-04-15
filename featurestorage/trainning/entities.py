"""
Feast entity definitions for the MovieLens feature store.

An entity is the primary key that groups feature rows.  All feature views
in this project are keyed on ``user_id``, matching the ``user_id`` column
written to Delta Lake by preprocess.py.
"""
from feast import Entity, ValueType

user_entity = Entity(
    name="user_id",
    value_type=ValueType.INT64,
    description="Unique user identifier from MovieLens 1M (range 1–6 040)",
)
