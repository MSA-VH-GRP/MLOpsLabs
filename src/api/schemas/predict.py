from pydantic import BaseModel


class PredictRequest(BaseModel):
    entity_ids: list[str]
    feature_service: str = "inference_features"
    model_name: str
    model_alias: str = "champion"


class PredictResponse(BaseModel):
    predictions: list[dict]
    model_version: str
    latency_ms: float
