"""GET /health — connectivity check for all downstream services."""

import httpx
from fastapi import APIRouter

from src.core.cache import ping as redis_ping
from src.core.config import settings
from src.core.kafka_producer import get_producer
from src.core.storage import get_s3_client

router = APIRouter()


@router.get("/health")
async def health():
    checks = {}

    # Redis
    checks["redis"] = redis_ping()

    # MLflow
    try:
        async with httpx.AsyncClient(timeout=3) as client:
            r = await client.get(f"{settings.mlflow_tracking_uri}/health")
            checks["mlflow"] = r.status_code == 200
    except Exception:
        checks["mlflow"] = False

    # Kafka (producer metadata)
    try:
        producer = get_producer()
        meta = producer.list_topics(timeout=3)
        checks["kafka"] = meta is not None
    except Exception:
        checks["kafka"] = False

    # MinIO (S3 list buckets)
    try:
        s3 = get_s3_client()
        s3.list_buckets()
        checks["minio"] = True
    except Exception:
        checks["minio"] = False

    status = "ok" if all(checks.values()) else "degraded"
    return {"status": status, "checks": checks}
