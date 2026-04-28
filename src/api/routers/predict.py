"""
POST /predict          — generic sklearn model inference via Feast + MLflow
POST /predict/mamba    — Mamba4Rec sequential recommendation
POST /predict/mock     — synthetic predictions for dashboard demo
"""

import asyncio
import logging
import random
import re
import time

from fastapi import APIRouter, Depends, HTTPException, Request
from feast import FeatureStore

from src.api.dependencies import get_feature_store
from src.api.schemas.predict import (
    Mamba4RecPredictRequest,
    Mamba4RecPredictResponse,
    PredictRequest,
    PredictResponse,
)
from src.core.metrics import (
    FEATURE_MISSING_RATE,
    MODEL_INFO,
    MOCK_PREDICTIONS_TOTAL,
    PREDICTION_DRIFT_SCORE,
    PREDICTION_RATING_TOTAL,
    PREDICT_BATCH_SIZE,
    PREDICT_ERRORS,
    PREDICT_LATENCY,
    PREDICT_REQUESTS,
    SYSTEM_CPU_USAGE,
    SYSTEM_MEMORY_USAGE,
)
from src.models.predict import run_predict

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
    PREDICT_BATCH_SIZE.observe(len(request.entity_ids))

    t0 = getattr(http_request.state, "predict_start_time", time.perf_counter())

    try:
        # run_predict: Redis cache-first → DuckDB offline fallback → MLflow model
        predictions = await asyncio.to_thread(
            run_predict,
            store,
            request.entity_ids,
            request.feature_service,
            request.model_name,
            request.model_alias,
        )

        for item in predictions:
            try:
                rating = int(round(float(item["prediction"])))
                rating = max(1, min(5, rating))
                PREDICTION_RATING_TOTAL.labels(rating=str(rating)).inc()
            except Exception:
                pass

        MODEL_INFO.labels(
            model_name=request.model_name,
            model_alias=request.model_alias,
        ).set(1)

        # synthetic auxiliary gauges for demo
        FEATURE_MISSING_RATE.labels(feature="user_age").set(round(random.uniform(0.00, 0.08), 3))
        FEATURE_MISSING_RATE.labels(feature="movie_genre").set(round(random.uniform(0.00, 0.05), 3))
        FEATURE_MISSING_RATE.labels(feature="timestamp").set(round(random.uniform(0.00, 0.02), 3))
        PREDICTION_DRIFT_SCORE.set(round(random.uniform(0.05, 0.35), 3))
        SYSTEM_CPU_USAGE.set(round(random.uniform(35, 75), 2))
        SYSTEM_MEMORY_USAGE.set(round(random.uniform(40, 85), 2))

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


def _check_model_exists(model_name: str, model_alias: str) -> None:
    """Raise 404 if the MLflow registered model or alias does not exist."""
    import mlflow
    from mlflow.exceptions import RestException
    from src.core.config import settings

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    try:
        mlflow.MlflowClient().get_model_version_by_alias(model_name, model_alias)
    except RestException:
        raise HTTPException(
            status_code=404,
            detail="Prediction failed — model is not existed.",
        )


@router.post("/predict/mamba", response_model=Mamba4RecPredictResponse)
async def predict_mamba(request: Mamba4RecPredictRequest):
    _validate_model_ref(request.model_name, request.model_alias)
    await asyncio.to_thread(_check_model_exists, request.model_name, request.model_alias)

    from src.inference.mamba_predictor import get_predictor

    PREDICT_REQUESTS.inc()
    t0 = time.perf_counter()

    try:
        predictor = get_predictor(request.model_name, request.model_alias)
        recommendations = predictor.recommend(
            item_history=request.item_history,
            time_history=request.time_history,
            age_idx=request.age_idx,
            gender_idx=request.gender_idx,
            occupation=request.occupation,
            top_k=request.top_k,
            target_time=request.target_time,
            now_showing_only=request.now_showing_only,
        )

        MODEL_INFO.labels(
            model_name=request.model_name,
            model_alias=request.model_alias,
        ).set(1)
        PREDICT_BATCH_SIZE.observe(request.top_k)

        for rec in recommendations:
            try:
                score = rec.get("score", 3)
                rating = int(round(float(score)))
                rating = max(1, min(5, rating))
                PREDICTION_RATING_TOTAL.labels(rating=str(rating)).inc()
            except Exception:
                pass

        FEATURE_MISSING_RATE.labels(feature="user_age").set(round(random.uniform(0.00, 0.08), 3))
        FEATURE_MISSING_RATE.labels(feature="item_history").set(round(random.uniform(0.00, 0.03), 3))
        FEATURE_MISSING_RATE.labels(feature="time_history").set(round(random.uniform(0.00, 0.02), 3))
        PREDICTION_DRIFT_SCORE.set(round(random.uniform(0.05, 0.35), 3))
        SYSTEM_CPU_USAGE.set(round(random.uniform(35, 75), 2))
        SYSTEM_MEMORY_USAGE.set(round(random.uniform(40, 85), 2))

        latency = time.perf_counter() - t0
        PREDICT_LATENCY.observe(latency)

        return Mamba4RecPredictResponse(
            recommendations=recommendations,
            model_version=request.model_alias,
            latency_ms=round(latency * 1000, 2),
        )

    except HTTPException:
        raise
    except Exception as exc:
        PREDICT_ERRORS.inc()
        latency = time.perf_counter() - t0
        PREDICT_LATENCY.observe(latency)
        logger.exception("Mamba prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed — check server logs.") from exc


@router.post("/predict/mock", response_model=PredictResponse)
async def predict_mock(request: PredictRequest):
    """
    Mock prediction endpoint for Grafana / Prometheus demo.
    Generates fake rating predictions in range 1-5.
    """
    PREDICT_REQUESTS.inc()
    PREDICT_BATCH_SIZE.observe(len(request.entity_ids))
    MOCK_PREDICTIONS_TOTAL.inc()

    t0 = time.perf_counter()

    try:
        _validate_model_ref(request.model_name, request.model_alias)

        await asyncio.sleep(random.uniform(0.02, 0.12))

        fake_predictions = []
        for eid in request.entity_ids:
            rating = random.randint(1, 5)
            fake_predictions.append({"entity_id": eid, "prediction": rating})
            PREDICTION_RATING_TOTAL.labels(rating=str(rating)).inc()

        MODEL_INFO.labels(
            model_name=request.model_name,
            model_alias=request.model_alias,
        ).set(1)

        FEATURE_MISSING_RATE.labels(feature="user_age").set(round(random.uniform(0.00, 0.08), 3))
        FEATURE_MISSING_RATE.labels(feature="movie_genre").set(round(random.uniform(0.00, 0.05), 3))
        FEATURE_MISSING_RATE.labels(feature="timestamp").set(round(random.uniform(0.00, 0.02), 3))

        PREDICTION_DRIFT_SCORE.set(round(random.uniform(0.05, 0.35), 3))
        SYSTEM_CPU_USAGE.set(round(random.uniform(35, 75), 2))
        SYSTEM_MEMORY_USAGE.set(round(random.uniform(40, 85), 2))

        latency = time.perf_counter() - t0
        PREDICT_LATENCY.observe(latency)

        return PredictResponse(
            predictions=fake_predictions,
            model_version=request.model_alias,
            latency_ms=round(latency * 1000, 2),
        )

    except Exception as e:
        PREDICT_ERRORS.inc()
        latency = time.perf_counter() - t0
        PREDICT_LATENCY.observe(latency)
        logger.exception(f"Mock predict failed latency={latency}: {e}")
        raise HTTPException(status_code=500, detail="Mock prediction failed")
