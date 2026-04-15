"""
Feature storage — training pipeline.

Public API
----------
::

    from featurestorage.trainning import TrainingFeatureStore

    with TrainingFeatureStore() as store:
        store.apply()                          # one-time Feast registry setup
        df = store.get_dataset("train")        # full training split
"""
from .feature_retrieval import TrainingFeatureStore
from .duckdb_engine import DuckDBDeltaEngine
from .config import MINIO_CONFIG, DUCKDB_CONFIG, MinIOConfig, DuckDBConfig

__all__ = [
    "TrainingFeatureStore",
    "DuckDBDeltaEngine",
    "MINIO_CONFIG",
    "DUCKDB_CONFIG",
    "MinIOConfig",
    "DuckDBConfig",
]
