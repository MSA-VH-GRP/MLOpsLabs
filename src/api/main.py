"""FastAPI application factory."""

from contextlib import asynccontextmanager

import mlflow
from fastapi import FastAPI
from prometheus_client import make_asgi_app

from src.api.routers import health, ingest, predict, train
from src.core.config import settings
from src.core.metrics import REGISTRY

import logging
from contextlib import asynccontextmanager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    yield


app = FastAPI(
    title="MLOps API",
    description="Inference serving, data ingestion trigger, and model training endpoint",
    version="0.1.0",
    lifespan=lifespan,
)

# Expose Prometheus metrics using the SAME registry as custom metrics
metrics_app = make_asgi_app(registry=REGISTRY)
app.mount("/metrics", metrics_app)

app.include_router(health.router, tags=["health"])
app.include_router(ingest.router, tags=["ingest"])
app.include_router(train.router, tags=["train"])
app.include_router(predict.router, tags=["predict"])