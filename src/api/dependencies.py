"""FastAPI dependency injection — shared clients and stores."""

from functools import lru_cache

import mlflow

from feast import FeatureStore
from src.core.cache import get_redis
from src.core.config import settings
from src.core.kafka_producer import get_producer
from src.core.storage import get_s3_client


@lru_cache
def get_feature_store() -> FeatureStore:
    return FeatureStore(repo_path="feast/")


def get_mlflow_client() -> mlflow.MlflowClient:
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    return mlflow.MlflowClient()


__all__ = [
    "get_feature_store",
    "get_mlflow_client",
    "get_redis",
    "get_producer",
    "get_s3_client",
]
