"""Model evaluation helpers — validate before promoting to champion alias."""

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report


def evaluate(model, X: pd.DataFrame, y: pd.Series) -> dict:
    preds = model.predict(X)
    acc = accuracy_score(y, preds)
    report = classification_report(y, preds, output_dict=True)
    return {"accuracy": acc, "report": report}


def should_promote(metrics: dict, threshold: float = 0.8) -> bool:
    return metrics.get("accuracy", 0) >= threshold
