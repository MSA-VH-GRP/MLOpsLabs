"""POST /predict — validates request, delegates to models.predict, returns response."""

import asyncio
import logging
import re
import time

from fastapi import APIRouter, Depends, HTTPException
from feast import FeatureStore

from src.api.dependencies import get_feature_store
from src.api.schemas.predict import PredictRequest, PredictResponse
from src.models.predict import run_predict

logger = logging.getLogger(__name__)
router = APIRouter()

_MODEL_NAME_RE = re.compile(r"^[A-Za-z0-9_.\-]{1,128}$")


def _validate_model_ref(model_name: str, model_alias: str) -> None:
    if not _MODEL_NAME_RE.match(model_name):
        raise HTTPException(status_code=422, detail=f"Invalid model_name: {model_name!r}")
    if not _MODEL_NAME_RE.match(model_alias):
        raise HTTPException(status_code=422, detail=f"Invalid model_alias: {model_alias!r}")


@router.post("/predict", response_model=PredictResponse)
async def predict(
    request: PredictRequest,
    store: FeatureStore = Depends(get_feature_store),
):
    _validate_model_ref(request.model_name, request.model_alias)
    t0 = time.perf_counter()

    try:
        predictions = await asyncio.to_thread(
            run_predict,
            store,
            request.entity_ids,
            request.feature_service,
            request.model_name,
            request.model_alias,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed — check server logs.") from exc

    latency_ms = (time.perf_counter() - t0) * 1000
    return PredictResponse(
        predictions=predictions,
        model_version=request.model_alias,
        latency_ms=round(latency_ms, 2),
    )
