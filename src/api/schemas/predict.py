from typing import List, Literal, Optional

from pydantic import BaseModel

# ── Existing sklearn / generic predict schemas (unchanged) ────────────────────


class PredictRequest(BaseModel):
    entity_ids: list[str]
    feature_service: str = "inference_features"
    model_name: str
    model_alias: str = "champion"


class PredictResponse(BaseModel):
    predictions: list[dict]
    model_version: str
    latency_ms: float


# ── Mamba4Rec sequential recommendation schemas ───────────────────────────────


class Mamba4RecPredictRequest(BaseModel):
    """
    Request schema for POST /predict/mamba.

    item_history and time_history use *internal* movie IDs (1-based integer indices
    produced by the materialization pipeline). The metadata.json artifact stored in
    MLflow contains the movie_id_map for translating original MovieLens IDs to
    internal indices if needed.
    """

    model_name: str = "mamba4rec"
    model_alias: str = "champion"

    # Ordered watch history (most-recent last)
    item_history: List[int]     # internal_movie_id values
    time_history: List[int]     # time slot per item: 0=Matinee, 1=Prime Time, 2=Late Night

    # Encoded user profile (from metadata.json mappings)
    age_idx:    int = 0   # 0–6 (seven ML-1M age groups)
    gender_idx: int = 1   # 0=Female, 1=Male
    occupation: int = 0   # 0–20

    top_k:       int = 10
    target_time: int = 1  # time slot for this session

    # Candidate pre-filtering: restrict scoring to currently showing movies
    # (fetched from Redis SET "showing:active" at serving time).
    # When False (default), the model scores every item in the catalog.
    now_showing_only: bool = False


class Mamba4RecPredictResponse(BaseModel):
    """Response schema for POST /predict/mamba."""

    recommendations: List[dict]   # [{rank, movie_id, title, genres, score, time_slot}]
    model_version: str
    latency_ms: float
