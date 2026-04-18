"""POST /train — trigger model training and register in MLflow."""

import asyncio
import logging

from fastapi import APIRouter, HTTPException

from src.api.schemas.train import TrainRequest, TrainResponse
from src.models.trainer import run_training

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/train", response_model=TrainResponse)
async def train(request: TrainRequest):
    try:
        result = await asyncio.to_thread(
            run_training,
            experiment_name=request.experiment_name,
            model_type=request.model_type,
            hyperparams=request.hyperparams,
            feature_view=request.feature_view,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Training failed")
        raise HTTPException(status_code=500, detail="Training failed — check server logs.") from exc
    return TrainResponse(**result)
