"""
MambaPredictor — loads a trained Mamba4Rec model from MLflow and serves recommendations.

Usage:
    predictor = get_predictor("mamba4rec", "champion")   # cached
    recs = predictor.recommend(item_history=[...], time_history=[...], ...)
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import mlflow
import mlflow.pytorch
import numpy as np
import torch

from src.core.config import settings

logger = logging.getLogger(__name__)

TIME_SLOTS = {
    0: "Matinee (6:00-17:59)",
    1: "Prime Time (18:00-21:59)",
    2: "Late Night (22:00-5:59)",
}


class MambaPredictor:
    """
    Inference wrapper around a trained Mamba4Rec model loaded from MLflow.

    The metadata dict (stored as an MLflow artifact) supplies vocab sizes and
    the reverse_movie_map so raw internal IDs are translated back to original
    MovieLens movie IDs for the API response.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        metadata: Dict,
        movies_info: List[Dict],
        device: torch.device,
    ):
        self.model = model
        self.model.eval()
        self.metadata = metadata
        self.device = device

        # Build lookup: original movie_id → title / genres
        self.movie_title: Dict[int, str] = {}
        self.movie_genres: Dict[int, str] = {}
        for row in movies_info:
            mid = int(row["movie_id"])
            self.movie_title[mid]  = row.get("title", f"Movie {mid}")
            genres = row.get("genres", [])
            self.movie_genres[mid] = "|".join(genres) if isinstance(genres, list) else str(genres)

        # reverse_movie_map: internal_id (str key) → original_movie_id
        self.reverse_movie_map: Dict[int, int] = {
            int(k): int(v) for k, v in metadata.get("reverse_movie_map", {}).items()
        }

    # ── Factory: load from MLflow ───────────────────────────────────────────────

    @classmethod
    def from_mlflow(
        cls,
        model_name: str = "mamba4rec",
        alias: str = "champion",
    ) -> "MambaPredictor":
        """
        Load the model and artifacts from MLflow model registry.

        Steps:
            1. mlflow.pytorch.load_model(f"models:/{model_name}@{alias}")
            2. Resolve the run_id for the registered version
            3. Download data/ artifacts (metadata.json, movies_info.json)
        """
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)

        model_uri = f"models:/{model_name}@{alias}"
        logger.info("Loading model from MLflow: %s", model_uri)
        model = mlflow.pytorch.load_model(model_uri)

        # Resolve run_id for this alias
        client = mlflow.MlflowClient()
        version_info = client.get_model_version_by_alias(model_name, alias)
        run_id = version_info.run_id

        # Download data/ artifacts to a local temp directory
        artifacts_dir = mlflow.artifacts.download_artifacts(
            run_id=run_id, artifact_path="data"
        )
        artifacts_path = Path(artifacts_dir)

        with open(artifacts_path / "metadata.json", encoding="utf-8") as f:
            metadata = json.load(f)

        movies_info: List[Dict] = []
        movies_info_path = artifacts_path / "movies_info.json"
        if movies_info_path.exists():
            with open(movies_info_path, encoding="utf-8") as f:
                movies_info = json.load(f)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("MambaPredictor loaded on device: %s", device)
        return cls(model, metadata, movies_info, device)

    # ── Inference ──────────────────────────────────────────────────────────────

    def _prepare_input(
        self,
        item_history: List[int],
        time_history: List[int],
        genre_lookup: Optional[Dict[int, List[int]]] = None,
        max_seq_len: int = 50,
    ) -> Dict[str, torch.Tensor]:
        """
        Pad sequences and build the model input dict.

        Args:
            item_history:  List of internal movie IDs (1-based)
            time_history:  Corresponding time slots
            genre_lookup:  internal_movie_id → [g0, g1, g2] (optional)
            max_seq_len:   Truncate / pad to this length
        """
        def pad_seq(seq, max_len, pad_val=0):
            if len(seq) >= max_len:
                return seq[-max_len:]
            return [pad_val] * (max_len - len(seq)) + seq

        item_seq = pad_seq(item_history, max_seq_len)
        time_seq = pad_seq(time_history, max_seq_len)

        if genre_lookup:
            genre_seq = [genre_lookup.get(i, [0, 0, 0]) for i in item_seq]
        else:
            genre_seq = [[0, 0, 0]] * max_seq_len

        return {
            "item_seq":  torch.LongTensor([item_seq]).to(self.device),
            "genre_seq": torch.LongTensor([genre_seq]).to(self.device),
            "time_seq":  torch.LongTensor([time_seq]).to(self.device),
        }

    @torch.no_grad()
    def recommend(
        self,
        item_history: List[int],
        time_history: List[int],
        age_idx: int,
        gender_idx: int,
        occupation: int,
        top_k: int = 10,
        target_time: int = 1,
        now_showing_only: bool = False,
    ) -> List[Dict]:
        """
        Return top-K movie recommendations.

        Args:
            item_history:    Ordered list of watched internal movie IDs
            time_history:    Time slot for each watched movie
            age_idx:         0–6 (ML-1M age groups)
            gender_idx:      0=F, 1=M
            occupation:      0–20
            top_k:           Number of recommendations
            target_time:     Time slot for this session
            now_showing_only: If True, restrict to a random subset (demo)

        Returns:
            List of recommendation dicts: rank, movie_id, title, genres, score, time_slot
        """
        max_seq_len = self.metadata.get("max_seq_len", 50)
        inputs = self._prepare_input(item_history, time_history, max_seq_len=max_seq_len)

        age_t    = torch.LongTensor([[age_idx]]).to(self.device)
        gender_t = torch.LongTensor([[gender_idx]]).to(self.device)
        occ_t    = torch.LongTensor([[occupation]]).to(self.device)

        scores = self.model.predict_scores(
            inputs["item_seq"],
            inputs["genre_seq"],
            inputs["time_seq"],
            age_t, gender_t, occ_t,
        ).squeeze(0).cpu().numpy()

        # Exclude padding
        scores[0] = -np.inf

        # Exclude already watched items
        for idx in item_history:
            if 0 <= idx < len(scores):
                scores[idx] = -np.inf

        if now_showing_only:
            all_ids = list(range(1, self.metadata["num_items"]))
            np.random.seed(42)
            now_showing = set(np.random.choice(all_ids, size=min(200, len(all_ids)), replace=False))
            mask = np.full_like(scores, -np.inf)
            for idx in now_showing:
                if idx < len(mask):
                    mask[idx] = 0
            scores = np.where(mask == 0, scores, -np.inf)

        top_indices = np.argsort(scores)[::-1][:top_k]

        recommendations = []
        for rank, idx in enumerate(top_indices):
            original_movie_id = self.reverse_movie_map.get(int(idx))
            if original_movie_id is None:
                continue
            recommendations.append({
                "rank":      rank + 1,
                "movie_id":  original_movie_id,
                "title":     self.movie_title.get(original_movie_id, f"Movie {original_movie_id}"),
                "genres":    self.movie_genres.get(original_movie_id, "Unknown"),
                "score":     float(scores[idx]),
                "time_slot": TIME_SLOTS.get(target_time, "Unknown"),
            })

        return recommendations


# ── Module-level cache: keyed by (model_name, alias) ──────────────────────────

_predictor_cache: Dict[tuple, MambaPredictor] = {}


def get_predictor(model_name: str = "mamba4rec", alias: str = "champion") -> MambaPredictor:
    """
    Return a cached MambaPredictor for the given (model_name, alias).
    Loads from MLflow on first call; subsequent calls return the cached instance.
    """
    key = (model_name, alias)
    if key not in _predictor_cache:
        logger.info("Cache miss — loading MambaPredictor from MLflow (%s@%s)", model_name, alias)
        _predictor_cache[key] = MambaPredictor.from_mlflow(model_name, alias)
    return _predictor_cache[key]
