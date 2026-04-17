"""
Model training: fetches offline features, trains a sklearn model, logs to MLflow.
"""

import logging

import mlflow
import mlflow.sklearn
import pandas as pd
from feast import FeatureStore
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from src.core.config import settings

logger = logging.getLogger(__name__)

MODEL_REGISTRY = {
    "random_forest": RandomForestClassifier,
    "logistic_regression": LogisticRegression,
}


def run_training(
    experiment_name: str,
    model_type: str,
    hyperparams: dict,
    feature_view: str,
) -> dict:
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(experiment_name)

    store = FeatureStore(repo_path="feast/")

    # Build entity_df from the staged offline Parquet (all event_id + event_timestamp pairs)
    from src.core.duckdb_client import get_duckdb_connection
    _conn = get_duckdb_connection()
    try:
        entity_df = _conn.execute(
            "SELECT event_id, event_timestamp "
            "FROM read_parquet('s3://offline-store/parquet/raw_events/staged.parquet')"
        ).df()
    finally:
        _conn.close()

    if entity_df.empty:
        raise ValueError("No entities found in offline store — run materialization first.")

    feature_df = store.get_historical_features(
        entity_df=entity_df,
        features=[f"{feature_view}:feature_1", f"{feature_view}:feature_2"],
    ).to_df()

    if feature_df.empty:
        raise ValueError("No training data returned from Feast offline store.")

    X = feature_df.drop(columns=["event_id", "event_timestamp", "label"], errors="ignore")
    y = feature_df.get("label", pd.Series([0] * len(feature_df)))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model_cls = MODEL_REGISTRY.get(model_type, RandomForestClassifier)

    with mlflow.start_run() as run:
        mlflow.log_params({"model_type": model_type, **hyperparams})
        model = model_cls(**hyperparams)
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        mlflow.log_metric("accuracy", score)
        mlflow.sklearn.log_model(model, artifact_path="model", registered_model_name=model_type)

        run_id = run.info.run_id
        logger.info("Training complete: run_id=%s accuracy=%.4f", run_id, score)

    client = mlflow.MlflowClient()
    versions = client.get_latest_versions(model_type)
    latest_version = max(int(v.version) for v in versions) if versions else 1

    return {"run_id": run_id, "model_version": latest_version, "status": "registered"}
