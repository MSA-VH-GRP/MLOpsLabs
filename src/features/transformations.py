"""
Pure feature transformation functions.
Used by both the offline materialization job and Feast OnDemandFeatureViews.
"""

import pandas as pd


def normalize(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Min-max normalize the specified columns."""
    for col in columns:
        col_min, col_max = df[col].min(), df[col].max()
        if col_max != col_min:
            df[col] = (df[col] - col_min) / (col_max - col_min)
    return df


def extract_hour(df: pd.DataFrame, timestamp_col: str = "event_timestamp") -> pd.DataFrame:
    """Extract hour-of-day from timestamp column as a new feature."""
    df["hour_of_day"] = pd.to_datetime(df[timestamp_col]).dt.hour
    return df
