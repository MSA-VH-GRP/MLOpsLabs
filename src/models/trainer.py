"""
Model training: fetches offline features, trains a sklearn model, logs to MLflow.
"""

import logging

import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from src.core.config import settings
from src.core.duckdb_client import get_duckdb_connection
from src.features.materialization import _set_aws_env

logger = logging.getLogger(__name__)

PARQUET_PATH = "s3://offline-store/parquet/users/staged.parquet"
FEATURE_COLS = ["gender_idx", "age_idx", "occupation", "target_time"]
TARGET_COL   = "target"

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
    print(f"Starting training: experiment={experiment_name}, model={model_type}, hyperparams={hyperparams}, feature_view={feature_view}")
    # Mamba4Rec path — delegate to dedicated trainer
    if model_type == "mamba4rec":
        from src.training.mamba_trainer import run_mamba_training
        return run_mamba_training(experiment_name=experiment_name, hyperparams=hyperparams)

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(experiment_name)

    _set_aws_env()

    conn = get_duckdb_connection()
    try:
        feature_df = conn.execute(
            f"SELECT user_id, {', '.join(FEATURE_COLS)}, {TARGET_COL} "
            f"FROM read_parquet('{PARQUET_PATH}')"
        ).df()
    finally:
        conn.close()

    if feature_df.empty:
        raise ValueError("No training data found — run materialization first.")

    X = feature_df[FEATURE_COLS]
    y = feature_df[TARGET_COL]
    logger.info("Fetched training data: %d rows, features=%s", len(feature_df), FEATURE_COLS)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model_cls = MODEL_REGISTRY.get(model_type, RandomForestClassifier)

    with mlflow.start_run() as run:
        mlflow.log_params({"model_type": model_type, "feature_view": feature_view, **hyperparams})
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
