"""Feature retrieval + inference logic for the predict endpoint.

Flow:
  1. Fetch features from Redis online store (Feast).
  2. On cache miss (all-null row), fall back to DuckDB offline store (staged Parquet).
  3. Run inference with the requested MLflow model.
"""

import logging

import mlflow
import pandas as pd

from feast import FeatureStore
from src.core.config import settings
from src.core.duckdb_client import get_duckdb_connection

logger = logging.getLogger(__name__)

# Must match trainer.py FEATURE_COLS — columns the model was trained on.
FEATURE_COLS = ["gender_idx", "age_idx", "occupation", "target_time"]
# All columns defined in the Feast feature view schema (needed for write_to_online_store).
_ONLINE_STORE_COLS = FEATURE_COLS + ["target"]
_PARQUET_PATH = "s3://offline-store/parquet/users/staged.parquet"


def _fetch_offline_features(entity_ids: list[str]) -> pd.DataFrame:
    """Read the latest feature row per entity from the DuckDB offline store.

    Only called for entity_ids that were missing from Redis.
    """
    from src.features.materialization import _set_aws_env
    _set_aws_env()

    int_ids = ", ".join(str(int(eid)) for eid in entity_ids)
    all_cols = "user_id, event_timestamp, " + ", ".join(_ONLINE_STORE_COLS)
    conn = get_duckdb_connection()
    try:
        return conn.execute(f"""
            SELECT {all_cols}
            FROM (
                SELECT {all_cols},
                       ROW_NUMBER() OVER (
                           PARTITION BY user_id
                           ORDER BY event_timestamp DESC
                       ) AS _rn
                FROM read_parquet('{_PARQUET_PATH}')
                WHERE user_id IN ({int_ids})
            ) t
            WHERE _rn = 1
        """).df()
    finally:
        conn.close()


def _resolve_features(
    store: FeatureStore,
    entity_ids: list[str],
    feature_service: str,
) -> pd.DataFrame:
    """Return a DataFrame of features for every entity_id.

    Tries Redis first; fills any cache misses from the DuckDB offline store.
    """
    entity_rows = [{"user_id": eid} for eid in entity_ids]
    online_result = store.get_online_features(
        features=store.get_feature_service(feature_service),
        entity_rows=entity_rows,
    ).to_dict()

    df = pd.DataFrame(online_result)

    miss_mask = df[FEATURE_COLS].isnull().all(axis=1)
    logger.info(
        "Fetched features for %d entities from Redis, %d cache misses",
        len(entity_ids), miss_mask.sum(),
    )
    if miss_mask.any():
        missed_ids = [eid for eid, miss in zip(entity_ids, miss_mask) if miss]
        logger.warning(
            "Redis cache miss for %d/%d entities %s — fetching from DuckDB offline store",
            len(missed_ids), len(entity_ids), missed_ids,
        )
        offline_df = _fetch_offline_features(missed_ids)
        rows_to_cache = []
        for idx, eid in enumerate(entity_ids):
            if not miss_mask.iloc[idx]:
                continue
            row = offline_df[offline_df["user_id"] == int(eid)]
            if row.empty:
                logger.error("Entity %s not found in offline store — prediction will be unreliable", eid)
                continue
            for col in FEATURE_COLS:
                df.at[idx, col] = row.iloc[0][col]
            rows_to_cache.append(row.iloc[[0]])

        if rows_to_cache:
            push_df = pd.concat(rows_to_cache, ignore_index=True)
            try:
                store.write_to_online_store("raw_event_features", push_df)
                logger.warning("Wrote %d rows back to Redis online store", len(push_df))
            except Exception:
                logger.warning("Failed to write back to Redis — next request will hit DuckDB again", exc_info=True)

    return df


def _load_model(model_name: str, model_alias: str):
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    return mlflow.pyfunc.load_model(f"models:/{model_name}@{model_alias}")


def run_predict(
    store: FeatureStore,
    entity_ids: list[str],
    feature_service: str,
    model_name: str,
    model_alias: str,
) -> list[dict]:
    """Resolve features and return predictions for every entity_id.

    Returns:
        List of {"entity_id": ..., "prediction": ...} dicts.
    """
    df = _resolve_features(store, entity_ids, feature_service)

    model = _load_model(model_name, model_alias)

    X = df[FEATURE_COLS]
    raw_preds = model.predict(X)

    return [
        {"entity_id": eid, "prediction": pred.item() if hasattr(pred, "item") else pred}
        for eid, pred in zip(entity_ids, raw_preds)
    ]
