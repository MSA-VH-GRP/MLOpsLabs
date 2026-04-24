"""Unit tests for src/models/predict.py — one assert per test."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))

import logging
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.api.dependencies import get_feature_store
from src.models.predict import FEATURE_COLS, _resolve_features, run_predict


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _mock_model(preds):
    m = MagicMock()
    m.predict.return_value = preds
    return m


def test_run_predict_returns_predictions_for_each_entity():
    store = get_feature_store()
    with patch("src.models.predict._load_model", return_value=_mock_model([10, 20])):
        result = run_predict(store, ["1", "2"], "inference_features", "rf", "champion")
        print(result)


if __name__ == "__main__":
    test_run_predict_returns_predictions_for_each_entity()
