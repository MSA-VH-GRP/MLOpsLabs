from pydantic import BaseModel


class TrainRequest(BaseModel):
    experiment_name: str = "default"
    model_type: str = "random_forest"
    hyperparams: dict = {}
    feature_view: str = "raw_event_features"


class TrainResponse(BaseModel):
    run_id: str
    model_version: int
    status: str
