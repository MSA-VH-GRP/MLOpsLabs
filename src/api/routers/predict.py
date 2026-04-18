"""POST /predict — fetch features from Redis, run inference, cache result."""

import logging
import re
import time

import mlflow
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Request
from feast import FeatureStore

from src.api.dependencies import get_feature_store
from src.api.schemas.predict import PredictRequest, PredictResponse
from src.core.config import settings
from src.core.metrics import (
    PREDICT_REQUESTS,
    PREDICT_ERRORS,
    PREDICT_LATENCY,
)

logger = logging.getLogger(__name__)
router = APIRouter()

_MODEL_NAME_RE = re.compile(r"^[A-Za-z0-9_.\-]{1,128}$")


def _validate_model_ref(model_name: str, model_alias: str) -> None:
    if not _MODEL_NAME_RE.match(model_name):
        raise HTTPException(status_code=422, detail=f"Invalid model_name: {model_name!r}")
    if not _MODEL_NAME_RE.match(model_alias):
        raise HTTPException(status_code=422, detail=f"Invalid model_alias: {model_alias!r}")


def get_feature_store_with_metrics(request: Request) -> FeatureStore:
    start_time = time.perf_counter()
    request.state.predict_start_time = start_time

    PREDICT_REQUESTS.inc()

    try:
        return get_feature_store()
    except Exception as e:
        PREDICT_ERRORS.inc()
        latency = time.perf_counter() - start_time
        PREDICT_LATENCY.observe(latency)
        logger.exception(f"Feature store dependency failed latency={latency}: {e}")
        raise


@router.post("/predict", response_model=PredictResponse)
async def predict(
    request: PredictRequest,
    http_request: Request,
    store: FeatureStore = Depends(get_feature_store_with_metrics),
):
    logger.info(f"Predict request received: {request.entity_ids}")

    _validate_model_ref(request.model_name, request.model_alias)

    t0 = getattr(http_request.state, "predict_start_time", time.perf_counter())

    try:
        entity_rows = [{"event_id": eid} for eid in request.entity_ids]
        feature_vector = store.get_online_features(
            features=store.get_feature_service(request.feature_service),
            entity_rows=entity_rows,
        ).to_dict()

        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        model_uri = f"models:/{request.model_name}@{request.model_alias}"
        model = mlflow.pyfunc.load_model(model_uri)

        df = pd.DataFrame(feature_vector)
        raw_preds = model.predict(df)

        predictions = [
            {"entity_id": eid, "prediction": pred}
            for eid, pred in zip(request.entity_ids, raw_preds)
        ]

        latency = time.perf_counter() - t0
        PREDICT_LATENCY.observe(latency)
        logger.info(f"Predict success latency={latency}")

        return PredictResponse(
            predictions=predictions,
            model_version=request.model_alias,
            latency_ms=round(latency * 1000, 2),
        )

    except HTTPException:
        raise
    except Exception as e:
        PREDICT_ERRORS.inc()
        latency = time.perf_counter() - t0
        PREDICT_LATENCY.observe(latency)
        logger.exception(f"Predict failed latency={latency}: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")