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