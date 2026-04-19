"""
ShowingPipeline — Kafka consumer for the 'showing-movies' topic.

Listens for events that declare which movies are currently showing in theaters
and maintains a live Redis SET ("showing:active") of internal_movie_ids.

Expected message payload (JSON):
    {
        "internal_movie_id": 42,          # 1-based internal ID from materialization
        "start_date": "2026-04-19",       # ISO-8601 date string (inclusive)
        "end_date":   "2026-05-10"        # ISO-8601 date string (inclusive)
    }

Redis key layout:
    showing:active          → SET of internal_movie_id strings currently showing
    showing:meta:<id>       → HASH {start_date, end_date, added_at}  (TTL-managed)

The consumer evaluates start_date / end_date against today's date on every
message so that a late-arriving or replayed event is handled correctly:
  • start_date <= today <= end_date  → SADD  showing:active
  • otherwise                        → SREM  showing:active  (no longer showing)
"""

import logging
from datetime import date, datetime, timezone

import redis

from src.core.config import settings
from src.pipelines.consumer import BaseConsumer
from src.pipelines.topics import SHOWING_MOVIES

logger = logging.getLogger(__name__)

# Redis key constants
REDIS_SHOWING_ACTIVE = "showing:active"
REDIS_META_PREFIX    = "showing:meta:"


class ShowingPipeline(BaseConsumer):
    """
    Kafka consumer that keeps Redis in sync with the current theater schedule.

    One instance is enough; run it as a long-lived process alongside
    IngestPipeline.
    """

    def __init__(self) -> None:
        super().__init__(
            topics=[SHOWING_MOVIES],
            group_id="showing-movies-consumer-group",
        )
        self._redis = redis.from_url(settings.redis_url, decode_responses=True)
        logger.info(
            "ShowingPipeline connected to Redis at %s", settings.redis_url
        )

    # ── Message handler ────────────────────────────────────────────────────────

    def process(self, message: dict, topic: str) -> None:  # noqa: ARG002
        """
        Evaluate a single showing-update event and update Redis accordingly.

        Args:
            message: Decoded JSON payload from Kafka.
            topic:   Source topic name (always SHOWING_MOVIES here).
        """
        internal_movie_id = message.get("internal_movie_id")
        start_date_str    = message.get("start_date")
        end_date_str      = message.get("end_date")

        if not all([internal_movie_id, start_date_str, end_date_str]):
            logger.warning(
                "Skipping malformed showing-update message (missing fields): %s",
                message,
            )
            return

        try:
            start_dt = date.fromisoformat(start_date_str)
            end_dt   = date.fromisoformat(end_date_str)
        except ValueError as exc:
            logger.warning("Invalid date format in message %s: %s", message, exc)
            return

        today    = date.today()
        movie_id = str(internal_movie_id)

        if start_dt <= today <= end_dt:
            # Movie is currently showing — add to active set
            self._redis.sadd(REDIS_SHOWING_ACTIVE, movie_id)

            # Store metadata for observability / debugging
            meta_key = f"{REDIS_META_PREFIX}{movie_id}"
            self._redis.hset(
                meta_key,
                mapping={
                    "start_date": start_date_str,
                    "end_date":   end_date_str,
                    "added_at":   datetime.now(timezone.utc).isoformat(),
                },
            )
            # TTL on the meta hash — expire 1 day after end_date
            days_until_expiry = (end_dt - today).days + 1
            self._redis.expire(meta_key, days_until_expiry * 86_400)

            logger.info(
                "Movie %s added to showing:active (showing %s → %s)",
                movie_id, start_date_str, end_date_str,
            )
        else:
            # Movie is no longer (or not yet) showing — remove from active set
            removed = self._redis.srem(REDIS_SHOWING_ACTIVE, movie_id)
            if removed:
                logger.info(
                    "Movie %s removed from showing:active (schedule: %s → %s, today: %s)",
                    movie_id, start_date_str, end_date_str, today,
                )

    # ── Graceful shutdown ──────────────────────────────────────────────────────

    def shutdown(self) -> None:
        logger.info("ShowingPipeline shutting down — Redis connection closed.")
        self._redis.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ShowingPipeline().run()
