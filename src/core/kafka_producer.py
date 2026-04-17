"""Kafka producer client wrapper."""

import json

from confluent_kafka import Producer

from src.core.config import settings

_producer: Producer | None = None


def get_producer() -> Producer:
    global _producer
    if _producer is None:
        _producer = Producer({"bootstrap.servers": settings.kafka_bootstrap_servers})
    return _producer


def produce(topic: str, key: str, value: dict) -> None:
    producer = get_producer()
    producer.produce(
        topic=topic,
        key=key.encode("utf-8"),
        value=json.dumps(value).encode("utf-8"),
    )
    # poll() drains the delivery-report queue without blocking; the caller
    # is responsible for calling flush() after a batch of produce() calls.
    producer.poll(0)


def flush() -> None:
    """Block until all queued messages have been delivered."""
    get_producer().flush()
