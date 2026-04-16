"""POST /ingest — validate payload and produce events to Kafka."""

from fastapi import APIRouter

from src.api.schemas.ingest import IngestRequest, IngestResponse
from src.core.config import settings
from src.core.kafka_producer import produce

router = APIRouter()


@router.post("/ingest", response_model=IngestResponse)
async def ingest(request: IngestRequest):
    for event in request.events:
        produce(
            topic=settings.kafka_ingest_topic,
            key=event.id,
            value={"id": event.id, "timestamp": event.timestamp, "payload": event.payload},
        )
    return IngestResponse(accepted=len(request.events), topic=settings.kafka_ingest_topic)
