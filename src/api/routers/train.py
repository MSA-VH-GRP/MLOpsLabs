"""POST /train — trigger model training and register in MLflow."""

import asyncio
import logging

from fastapi import APIRouter

from src.api.schemas.train import TrainRequest, TrainResponse
from src.models.trainer import run_training

logger = logging.getLogger(__name__)
router = APIRouter()


def _run_and_log(request: TrainRequest):
    try:
        result = run_training(
            experiment_name=request.experiment_name,
            model_type=request.model_type,
            hyperparams=request.hyperparams,
            feature_view=request.feature_view,
        )
        logger.info("Training complete: %s", result)
    except Exception:
        logger.exception("Background training failed")


@router.post("/train", status_code=202, response_model=TrainResponse)
async def train(request: TrainRequest):
    asyncio.get_event_loop().run_in_executor(None, _run_and_log, request)
    return TrainResponse(run_id="pending", model_version=0, status="training_started")
