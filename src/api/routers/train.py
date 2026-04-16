"""POST /train — trigger model training and register in MLflow."""

from fastapi import APIRouter, BackgroundTasks

from src.api.schemas.train import TrainRequest, TrainResponse
from src.models.trainer import run_training

router = APIRouter()


@router.post("/train", response_model=TrainResponse)
async def train(request: TrainRequest, background_tasks: BackgroundTasks):
    result = run_training(
        experiment_name=request.experiment_name,
        model_type=request.model_type,
        hyperparams=request.hyperparams,
        feature_view=request.feature_view,
    )
    return TrainResponse(**result)
