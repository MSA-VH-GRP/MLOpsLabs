from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry

REGISTRY = CollectorRegistry()

PREDICT_REQUESTS = Counter(
    "predict_requests_total",
    "Total predict requests",
    registry=REGISTRY,
)

PREDICT_ERRORS = Counter(
    "predict_errors_total",
    "Total predict errors",
    registry=REGISTRY,
)

PREDICT_LATENCY = Histogram(
    "predict_latency_seconds",
    "Predict latency",
    registry=REGISTRY,
)

PREDICT_BATCH_SIZE = Histogram(
    "predict_batch_size",
    "Batch size",
    registry=REGISTRY,
)

HEALTH_STATUS = Gauge(
    "service_health_status",
    "Health status of services",
    ["service"],
    registry=REGISTRY,
)

# ---- ML metrics for dashboard ----

PREDICTION_RATING_TOTAL = Counter(
    "prediction_rating_total",
    "Count of predicted rating values",
    ["rating"],
    registry=REGISTRY,
)

MODEL_INFO = Gauge(
    "model_info",
    "Current model info",
    ["model_name", "model_alias"],
    registry=REGISTRY,
)

FEATURE_MISSING_RATE = Gauge(
    "feature_missing_rate",
    "Missing rate for feature inputs",
    ["feature"],
    registry=REGISTRY,
)

PREDICTION_DRIFT_SCORE = Gauge(
    "prediction_drift_score",
    "Synthetic prediction drift score",
    registry=REGISTRY,
)

MOCK_PREDICTIONS_TOTAL = Counter(
    "mock_predictions_total",
    "Total mock predictions generated",
    registry=REGISTRY,
)