"""POST /predict — fetch features from Redis, run inference, cache result."""

import logging
import time

import mlflow
import pandas as pd
from fastapi import APIRouter, HTTPException
from feast import FeatureStore

from src.api.dependencies import get_feature_store
from src.api.schemas.predict import PredictRequest, PredictResponse
from src.core.config import settings
from src.core.metrics import (
    PREDICT_REQUESTS,
    PREDICT_ERRORS,
    PREDICT_LATENCY,
    PREDICT_BATCH_SIZE,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    PREDICT_REQUESTS.inc()
    PREDICT_BATCH_SIZE.observe(len(request.entity_ids))
    logger.info(f"Predict request received: {request.entity_ids}")

    t0 = time.perf_counter()

    try:
        # Get feature store INSIDE the function so metrics are counted
        store: FeatureStore = get_feature_store()

        # Fetch features
        entity_rows = [{"event_id": eid} for eid in request.entity_ids]
        feature_vector = store.get_online_features(
            features=store.get_feature_service(request.feature_service),
            entity_rows=entity_rows,
        ).to_dict()

        # Load model
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        model_uri = f"models:/{request.model_name}@{request.model_alias}"
        model = mlflow.pyfunc.load_model(model_uri)

        # Predict
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

    except Exception as e:
        PREDICT_ERRORS.inc()
        latency = time.perf_counter() - t0
        PREDICT_LATENCY.observe(latency)
        logger.exception(f"Predict failed latency={latency}: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")