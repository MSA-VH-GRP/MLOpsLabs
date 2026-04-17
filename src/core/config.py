"""
Centralised configuration loaded from environment variables.
All services read their connection details from here.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # MinIO / S3
    minio_endpoint: str = "http://localhost:9000"
    aws_access_key_id: str = "minioadmin"
    aws_secret_access_key: str = "minioadmin123"

    # MLflow
    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_s3_endpoint_url: str = "http://localhost:9000"
    mlflow_experiment_name: str = "default"

    # Redis
    redis_url: str = "redis://localhost:6379"

    # Kafka
    kafka_bootstrap_servers: str = "localhost:9092"
    kafka_ingest_topic: str = "raw-events"
    kafka_group_id: str = "mlops-consumer-group"

    # App
    app_env: str = "development"
    log_level: str = "info"


settings = Settings()
