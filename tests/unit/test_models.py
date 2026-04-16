"""Unit tests for model evaluation helpers."""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.models.evaluator import evaluate, should_promote


def test_evaluate():
    model = RandomForestClassifier(n_estimators=5, random_state=42)
    X = pd.DataFrame({"a": [1, 2, 3, 4], "b": [4, 3, 2, 1]})
    y = pd.Series([0, 1, 0, 1])
    model.fit(X, y)
    metrics = evaluate(model, X, y)
    assert "accuracy" in metrics
    assert 0.0 <= metrics["accuracy"] <= 1.0


def test_should_promote_above_threshold():
    assert should_promote({"accuracy": 0.9}, threshold=0.8) is True


def test_should_promote_below_threshold():
    assert should_promote({"accuracy": 0.5}, threshold=0.8) is False
