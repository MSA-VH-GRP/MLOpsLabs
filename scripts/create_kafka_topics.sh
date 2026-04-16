#!/usr/bin/env bash
# Create Kafka topics required by the MLOps pipeline.
# Run once after `docker compose up -d kafka`.

set -euo pipefail

BROKER="${KAFKA_BOOTSTRAP_SERVERS:-localhost:9092}"
PARTITIONS="${KAFKA_NUM_PARTITIONS:-3}"
REPLICATION="${KAFKA_REPLICATION_FACTOR:-1}"

TOPICS=(
  "raw-events"
  "processed-events"
)

for topic in "${TOPICS[@]}"; do
  echo "Creating topic: $topic"
  docker exec mlops-kafka kafka-topics.sh \
    --bootstrap-server "$BROKER" \
    --create \
    --if-not-exists \
    --topic "$topic" \
    --partitions "$PARTITIONS" \
    --replication-factor "$REPLICATION"
done

echo "Done."
