"""Prometheus instrumentation for the FastAPI app."""

from prometheus_fastapi_instrumentator import Instrumentator


def setup_metrics(app) -> None:
    Instrumentator(
        should_group_status_codes=True,
        should_ignore_untemplated=True,
        excluded_handlers=["/metrics", "/health"],
    ).instrument(app).expose(app, endpoint="/metrics")
