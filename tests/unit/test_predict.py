"""Unit tests for src/models/predict.py — one assert per test."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))

from unittest.mock import MagicMock, patch

import pandas as pd

from src.models.predict import FEATURE_COLS, run_predict


# ── Helpers ───────────────────────────────────────────────────────────────────

def _mock_store(entity_ids: list[str]) -> MagicMock:
    """Return a MagicMock FeatureStore that returns a deterministic feature DataFrame."""
    df = pd.DataFrame(
        {col: [0.0] * len(entity_ids) for col in FEATURE_COLS},
    )
    df["user_id"] = entity_ids

    online_result = MagicMock()
    online_result.to_dict.return_value = df.to_dict(orient="list")

    store = MagicMock()
    store.get_online_features.return_value = online_result
    return store


def _mock_model(preds):
    m = MagicMock()
    m.predict.return_value = preds
    return m


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_run_predict_returns_predictions_for_each_entity():
    store = _mock_store(["1", "2"])
    with patch("src.models.predict._load_model", return_value=_mock_model([10, 20])):
        result = run_predict(store, ["1", "2"], "inference_features", "rf", "champion")
    assert len(result) == 2
