"""Unit tests for Pydantic request/response schemas."""

import pytest
from pydantic import ValidationError

from src.api.schemas.ingest import IngestRequest
from src.api.schemas.predict import PredictRequest
from src.api.schemas.train import TrainRequest


def test_ingest_request_valid():
    req = IngestRequest(events=[{"id": "1", "timestamp": "2026-01-01T00:00:00", "payload": {"x": 1}}])
    assert len(req.events) == 1


def test_ingest_request_empty():
    req = IngestRequest(events=[])
    assert req.events == []


def test_predict_request_defaults():
    req = PredictRequest(entity_ids=["a", "b"], model_name="my_model")
    assert req.model_alias == "champion"
    assert req.feature_service == "inference_features"


def test_train_request_defaults():
    req = TrainRequest()
    assert req.model_type == "random_forest"
    assert req.hyperparams == {}
