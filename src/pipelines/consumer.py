"""Base Kafka consumer."""

import json
import logging
from abc import ABC, abstractmethod

from confluent_kafka import Consumer, KafkaError, KafkaException

from src.core.config import settings

logger = logging.getLogger(__name__)


class BaseConsumer(ABC):
    def __init__(self, topics: list[str], group_id: str | None = None):
        self._consumer = Consumer({
            "bootstrap.servers": settings.kafka_bootstrap_servers,
            "group.id": group_id or settings.kafka_group_id,
            "auto.offset.reset": "earliest",
        })
        self._consumer.subscribe(topics)

    def run(self) -> None:
        logger.info("Consumer started")
        try:
            while True:
                msg = self._consumer.poll(timeout=1.0)
                if msg is None:
                    continue
                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        continue
                    raise KafkaException(msg.error())
                self.process(json.loads(msg.value().decode("utf-8")), msg.topic())
        finally:
            self.shutdown()
            self._consumer.close()

    @abstractmethod
    def process(self, message: dict, topic: str) -> None:
        """Handle a single decoded message."""

    def shutdown(self) -> None:
        """Called gracefully on consumer exit."""
        pass
