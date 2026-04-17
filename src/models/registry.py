"""MLflow model registry helpers."""

import mlflow
from mlflow import MlflowClient

from src.core.config import settings


def get_client() -> MlflowClient:
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    return MlflowClient()


def set_alias(model_name: str, version: int, alias: str) -> None:
    get_client().set_registered_model_alias(model_name, alias, str(version))


def get_latest_version(model_name: str) -> int:
    versions = get_client().get_latest_versions(model_name)
    return max(int(v.version) for v in versions) if versions else 0


def load_model(model_name: str, alias: str = "champion"):
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    return mlflow.pyfunc.load_model(f"models:/{model_name}@{alias}")
