"""
POST /predict          — generic sklearn model inference via Feast + MLflow
POST /predict/mamba    — Mamba4Rec sequential recommendation
"""

import time

import mlflow
import pandas as pd
from fastapi import APIRouter, Depends
from feast import FeatureStore

from src.api.dependencies import get_feature_store
from src.api.schemas.predict import (
    Mamba4RecPredictRequest,
    Mamba4RecPredictResponse,
    PredictRequest,
    PredictResponse,
)
from src.core.config import settings

router = APIRouter()


# ── Generic sklearn predict (unchanged) ───────────────────────────────────────

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
    df = pd.DataFrame(feature_vector)
    raw_preds = model.predict(df)
    predictions = [
        {"entity_id": eid, "prediction": pred}
        for eid, pred in zip(request.entity_ids, raw_preds)
    ]

    latency_ms = (time.perf_counter() - t0) * 1000
    return PredictResponse(
        predictions=predictions,
        model_version=request.model_alias,
        latency_ms=round(latency_ms, 2),
    )


# ── Mamba4Rec sequential recommendation ───────────────────────────────────────

@router.post("/predict/mamba", response_model=Mamba4RecPredictResponse)
async def predict_mamba(request: Mamba4RecPredictRequest):
    """
    Sequential movie recommendation using the trained Mamba4Rec model.

    The predictor is loaded from MLflow on first call and cached in memory.
    Subsequent calls reuse the cached instance for low-latency serving.
    """
    from src.inference.mamba_predictor import get_predictor

    t0 = time.perf_counter()

    predictor = get_predictor(request.model_name, request.model_alias)

    recommendations = predictor.recommend(
        item_history=request.item_history,
        time_history=request.time_history,
        age_idx=request.age_idx,
        gender_idx=request.gender_idx,
        occupation=request.occupation,
        top_k=request.top_k,
        target_time=request.target_time,
    )

    latency_ms = (time.perf_counter() - t0) * 1000
    return Mamba4RecPredictResponse(
        recommendations=recommendations,
        model_version=request.model_alias,
        latency_ms=round(latency_ms, 2),
    )
