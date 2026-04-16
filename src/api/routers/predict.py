"""POST /predict — fetch features from Redis, run inference, cache result."""

import time

import mlflow
from fastapi import APIRouter, Depends
from feast import FeatureStore

from src.api.dependencies import get_feature_store
from src.api.schemas.predict import PredictRequest, PredictResponse
from src.core.config import settings

router = APIRouter()


@router.post("/predict", response_model=PredictResponse)
async def predict(
    request: PredictRequest,
    store: FeatureStore = Depends(get_feature_store),
):
    t0 = time.perf_counter()

    # Fetch features from Redis (Feast online store)
    entity_rows = [{"event_id": eid} for eid in request.entity_ids]
    feature_vector = store.get_online_features(
        features=store.get_feature_service(request.feature_service),
        entity_rows=entity_rows,
    ).to_dict()

    # Load model from MLflow registry
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    model_uri = f"models:/{request.model_name}@{request.model_alias}"
    model = mlflow.pyfunc.load_model(model_uri)

    # Run inference
    import pandas as pd
    df = pd.DataFrame(feature_vector)
    raw_preds = model.predict(df)
    predictions = [{"entity_id": eid, "prediction": pred} for eid, pred in zip(request.entity_ids, raw_preds)]

    latency_ms = (time.perf_counter() - t0) * 1000
    return PredictResponse(
        predictions=predictions,
        model_version=request.model_alias,
        latency_ms=round(latency_ms, 2),
    )
