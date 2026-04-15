"""
TrainingFeatureStore — the main API surface for the training pipeline.

Architecture
------------

::

    Training Pipeline
          │
          ▼
    TrainingFeatureStore          ← this module (Feast business layer)
          │
          ├─ feast.FeatureStore   ← registry: entity/view metadata in registry.db
          │
          └─ DuckDBDeltaEngine    ← offline query engine
                 │
                 ├─ httpfs ext    ← S3-compatible MinIO access
                 └─ delta  ext    ← Delta Lake format decoding
                        │
               s3://processed/{train|val|test}
                   (Delta tables from preprocess.py)
                        │
                        ▼
               pd.DataFrame  ──▶  Training Pipeline

Feast is responsible for:
  • Feature registry and schema catalogue (``apply()``)
  • Entity / feature-view definitions (entities.py, feature_views.py)
  • Feature reference naming ("view:feature")

DuckDBDeltaEngine is responsible for:
  • Actual offline data retrieval via ``delta_scan()``
  • Projection, filtering, and joining in-process (no extra network hop)

Usage
-----
::

    store = TrainingFeatureStore()
    store.apply()                                    # one-time registry setup

    # Full training split
    df = store.get_dataset("train")

    # Specific users only
    df = store.get_dataset("train", user_ids=[1, 42, 100])

    # Only user demographic features
    df = store.get_user_features([1, 42, 100])

    # Feast-style historical feature retrieval
    entity_df = pd.DataFrame({"user_id": [1, 42, 100]})
    df = store.get_historical_features(
        entity_df,
        feature_refs=[
            "user_profile:age_idx",
            "user_profile:gender_idx",
            "user_interactions:item_seq",
        ],
    )
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd
from feast import FeatureStore

from .config import DUCKDB_CONFIG, MINIO_CONFIG, DuckDBConfig, MinIOConfig
from .duckdb_engine import DuckDBDeltaEngine
from .entities import user_entity
from .feature_views import user_interaction_view, user_profile_view

# Canonical column groupings — mirrors _records_to_df in preprocess.py
_PROFILE_COLS  = ["age_idx", "gender_idx", "occupation"]
_SEQUENCE_COLS = ["item_seq", "genre_seq", "time_seq", "target", "target_time"]
_ALL_FEATURE_COLS = _SEQUENCE_COLS + _PROFILE_COLS

# Mapping: Feast feature-view name → column list it owns
_VIEW_COLS = {
    "user_profile":      _PROFILE_COLS,
    "user_interactions": _SEQUENCE_COLS,
}


class TrainingFeatureStore:
    """
    Feast-backed feature store for the training pipeline.

    Parameters
    ----------
    repo_path:  Directory containing ``feature_store.yaml`` and
                ``registry.db``.  Defaults to the module's own directory.
    minio:      MinIO connection settings (defaults to ``MINIO_CONFIG``).
    duckdb_cfg: DuckDB tuning settings (defaults to ``DUCKDB_CONFIG``).
    """

    def __init__(
        self,
        repo_path: str = ".",
        minio: Optional[MinIOConfig] = None,
        duckdb_cfg: Optional[DuckDBConfig] = None,
    ) -> None:
        self._feast  = FeatureStore(repo_path=repo_path)
        self._engine = DuckDBDeltaEngine(
            minio=minio or MINIO_CONFIG,
            duckdb_cfg=duckdb_cfg or DUCKDB_CONFIG,
        )

    # ── Registry ──────────────────────────────────────────────────────────────

    def apply(self) -> None:
        """
        Register all entities and feature views in the Feast registry.

        Equivalent to running ``feast apply`` on the command line.
        Call once before the first ``get_historical_features()`` in a new
        environment.
        """
        self._feast.apply([user_entity, user_profile_view, user_interaction_view])
        print(
            "[Feast] Registered: user_entity | "
            "user_profile_view | user_interaction_view"
        )

    # ── Data Retrieval ────────────────────────────────────────────────────────

    def get_dataset(
        self,
        split: str = "train",
        user_ids: Optional[List[int]] = None,
    ) -> pd.DataFrame:
        """
        Load a full split from Delta Lake via DuckDB.

        Returns all feature columns joined on ``user_id``.

        Parameters
        ----------
        split:    ``"train"``, ``"val"``, or ``"test"``.
        user_ids: Optional list of user IDs to filter to.

        Returns
        -------
        DataFrame with columns:
            user_id, item_seq, genre_seq, time_seq, target, target_time,
            age_idx, gender_idx, occupation
        """
        where = self._user_filter(user_ids)
        cols  = ", ".join(["user_id"] + _ALL_FEATURE_COLS)
        sql   = f"""
            SELECT {cols}
            FROM   {self._engine.scan(split)}
            {where}
            ORDER  BY user_id
        """
        return self._engine.query(sql)

    def get_user_features(self, user_ids: List[int]) -> pd.DataFrame:
        """
        Retrieve static user profile features for a list of user IDs.

        Deduplicates by ``user_id`` because demographics are constant across
        all training examples for a given user.

        Returns
        -------
        DataFrame with columns: user_id, age_idx, gender_idx, occupation
        """
        ids = self._ids_literal(user_ids)
        sql = f"""
            SELECT DISTINCT user_id, age_idx, gender_idx, occupation
            FROM   {self._engine.scan("train")}
            WHERE  user_id IN ({ids})
            ORDER  BY user_id
        """
        return self._engine.query(sql)

    def get_sequence_features(
        self,
        split: str = "train",
        user_ids: Optional[List[int]] = None,
    ) -> pd.DataFrame:
        """
        Retrieve interaction sequences for a split, optionally filtered by user.

        Returns
        -------
        DataFrame with columns:
            user_id, item_seq, genre_seq, time_seq, target, target_time
        """
        where = self._user_filter(user_ids)
        cols  = ", ".join(["user_id"] + _SEQUENCE_COLS)
        sql   = f"""
            SELECT {cols}
            FROM   {self._engine.scan(split)}
            {where}
            ORDER  BY user_id
        """
        return self._engine.query(sql)

    def get_historical_features(
        self,
        entity_df: pd.DataFrame,
        feature_refs: List[str],
        split: str = "train",
    ) -> pd.DataFrame:
        """
        Feast-compatible historical feature retrieval.

        Joins ``entity_df`` (must contain a ``user_id`` column) with the
        requested features from the registered feature views.  The query is
        executed by DuckDBDeltaEngine against the specified Delta Lake split.

        Parameters
        ----------
        entity_df:    DataFrame with at least a ``user_id`` column.
        feature_refs: List of ``"view_name:feature_name"`` strings.
                      Supported views: ``user_profile``, ``user_interactions``.
        split:        Delta Lake split to query (``"train"`` / ``"val"`` /
                      ``"test"``).

        Returns
        -------
        ``entity_df`` left-joined with the requested feature columns.

        Example
        -------
        ::

            entity_df = pd.DataFrame({"user_id": [1, 42, 100]})
            df = store.get_historical_features(
                entity_df,
                feature_refs=[
                    "user_profile:age_idx",
                    "user_profile:gender_idx",
                    "user_interactions:item_seq",
                    "user_interactions:target",
                ],
            )
        """
        # Resolve feature references → concrete column names
        requested: List[str] = []
        for ref in feature_refs:
            try:
                view_name, feat_name = ref.split(":", 1)
            except ValueError:
                raise ValueError(
                    f"Invalid feature_ref '{ref}'. "
                    "Expected format: 'view_name:feature_name'."
                )
            allowed = _VIEW_COLS.get(view_name)
            if allowed is None:
                raise ValueError(
                    f"Unknown feature view '{view_name}'. "
                    f"Available views: {list(_VIEW_COLS)}"
                )
            if feat_name not in allowed:
                raise ValueError(
                    f"Feature '{feat_name}' not in view '{view_name}'. "
                    f"Available features: {allowed}"
                )
            if feat_name not in requested:
                requested.append(feat_name)

        user_ids = entity_df["user_id"].tolist()
        ids_str  = self._ids_literal(user_ids)
        cols     = ", ".join(["user_id"] + requested)
        sql      = f"""
            SELECT {cols}
            FROM   {self._engine.scan(split)}
            WHERE  user_id IN ({ids_str})
            ORDER  BY user_id
        """
        features = self._engine.query(sql)
        return entity_df.merge(features, on="user_id", how="left")

    # ── Materialization ───────────────────────────────────────────────────────

    def materialize_incremental(
        self,
        end_date: Optional[datetime] = None,
    ) -> None:
        """
        Push feature values from Delta Lake → Redis (online store).

        Feast tracks the last materialization timestamp per feature view in
        registry.db.  Only rows newer than that timestamp are pushed, so
        repeated calls are safe and cheap (idempotent).

        Note on timestamps
        ------------------
        The Delta tables use ``target_time`` as the timestamp field, which
        stores time-of-day buckets (0/1/2) rather than real Unix timestamps.
        Feast interprets these as epoch + N seconds (1970-01-01 00:00:00–02).
        On the *first* run all rows are materialised; subsequent runs push
        nothing new because the stored last-run time is far in the future.
        This is the correct behaviour for a static MovieLens dataset.

        Parameters
        ----------
        end_date: Upper bound for materialization. Defaults to now (UTC).
        """
        end_date = end_date or datetime.now(tz=timezone.utc)
        self._feast.materialize_incremental(end_date=end_date)

    # ── Online serving ────────────────────────────────────────────────────────

    def get_online_features(
        self,
        feature_refs: List[str],
        entity_rows: List[Dict[str, Any]],
    ) -> pd.DataFrame:
        """
        Retrieve feature values from Redis for low-latency inference serving.

        Reads from the online store (Redis); does not touch Delta Lake.
        Requires ``materialize_incremental()`` to have been run at least once.

        Parameters
        ----------
        feature_refs: List of ``"view_name:feature_name"`` strings.
        entity_rows:  List of dicts, each with a ``user_id`` key.

        Returns
        -------
        DataFrame with one row per entity and one column per feature ref.

        Example
        -------
        ::

            df = store.get_online_features(
                feature_refs=["user_profile:age_idx", "user_profile:gender_idx"],
                entity_rows=[{"user_id": 1}, {"user_id": 42}],
            )
        """
        response = self._feast.get_online_features(
            features=feature_refs,
            entity_rows=entity_rows,
        )
        return response.to_df()

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def close(self) -> None:
        """Release the DuckDB connection."""
        self._engine.close()

    def __enter__(self) -> "TrainingFeatureStore":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    # ── Format helpers ────────────────────────────────────────────────────────

    def get_dataset_records(
        self,
        split: str = "train",
        user_ids: Optional[List[int]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Load a split and return it as a list of dicts matching the expected
        training format used by Mamba4Rec.

        Each record mirrors the structure of the original ``.pkl`` files:

        .. code-block:: python

            {
                "user_id":      int,
                "item_seq":     list[int],
                "genre_seq":    list[list[int]],
                "time_seq":     list[int],
                "target":       int,
                "target_time":  int,
                "user_profile": {"age_idx": int, "gender_idx": int, "occupation": int},
            }

        Parameters
        ----------
        split:    ``"train"``, ``"val"``, or ``"test"``.
        user_ids: Optional list of user IDs to filter to.
        """
        df = self.get_dataset(split, user_ids=user_ids)
        records = []
        for row in df.itertuples(index=False):
            records.append({
                "user_id":     int(row.user_id),
                "item_seq":    list(int(x) for x in row.item_seq),
                "genre_seq":   [list(int(g) for g in genre) for genre in row.genre_seq],
                "time_seq":    list(int(x) for x in row.time_seq),
                "target":      int(row.target),
                "target_time": int(row.target_time),
                "user_profile": {
                    "age_idx":    int(row.age_idx),
                    "gender_idx": int(row.gender_idx),
                    "occupation": int(row.occupation),
                },
            })
        return records

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _ids_literal(user_ids: List[int]) -> str:
        """Format a list of ints as a SQL IN-list literal."""
        return ", ".join(map(str, user_ids))

    @staticmethod
    def _user_filter(user_ids: Optional[List[int]]) -> str:
        """Return a WHERE clause string, or empty string if no filter."""
        if not user_ids:
            return ""
        ids = ", ".join(map(str, user_ids))
        return f"WHERE user_id IN ({ids})"
