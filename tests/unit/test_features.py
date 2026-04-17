"""Unit tests for feature transformations."""

import pandas as pd

from src.features.transformations import extract_hour, normalize


def test_normalize():
    df = pd.DataFrame({"a": [0.0, 5.0, 10.0]})
    result = normalize(df.copy(), ["a"])
    assert result["a"].min() == 0.0
    assert result["a"].max() == 1.0


def test_extract_hour():
    df = pd.DataFrame({"event_timestamp": ["2026-01-01 14:30:00"]})
    result = extract_hour(df)
    assert result["hour_of_day"].iloc[0] == 14
