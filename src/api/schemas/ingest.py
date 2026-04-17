from pydantic import BaseModel


class EventPayload(BaseModel):
    id: str
    timestamp: str
    payload: dict


class IngestRequest(BaseModel):
    events: list[EventPayload]


class IngestResponse(BaseModel):
    accepted: int
    topic: str
