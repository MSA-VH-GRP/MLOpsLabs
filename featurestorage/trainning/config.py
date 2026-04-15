"""
MinIO and DuckDB configuration for the training feature store.

Values are read from environment variables with sensible dev defaults
that match the docker-compose.yml service settings.
"""
import os
from dataclasses import dataclass


@dataclass
class MinIOConfig:
    endpoint: str = "http://localhost:9000"
    access_key: str = "minioadmin"
    secret_key: str = "minioadmin123"
    processed_bucket: str = "processed"
    raw_bucket: str = "raw-data"
    region: str = "us-east-1"

    @property
    def endpoint_host(self) -> str:
        """Endpoint without the protocol prefix — used by DuckDB's s3_endpoint setting."""
        return self.endpoint.replace("http://", "").replace("https://", "")

    @property
    def use_ssl(self) -> bool:
        return self.endpoint.startswith("https://")

    def delta_uri(self, table: str) -> str:
        """S3 URI for a Delta Lake table in the processed bucket."""
        return f"s3://{self.processed_bucket}/{table}"


@dataclass
class DuckDBConfig:
    db_path: str = ":memory:"
    threads: int = 4
    memory_limit: str = "2GB"


# Module-level singletons — shared across all store instances unless overridden.
MINIO_CONFIG = MinIOConfig(
    endpoint=os.getenv("MINIO_ENDPOINT", "http://localhost:9000"),
    access_key=os.getenv("MINIO_ACCESS_KEY", "minioadmin"),
    secret_key=os.getenv("MINIO_SECRET_KEY", "minioadmin123"),
)

DUCKDB_CONFIG = DuckDBConfig(
    threads=int(os.getenv("DUCKDB_THREADS", "4")),
    memory_limit=os.getenv("DUCKDB_MEMORY_LIMIT", "2GB"),
)
