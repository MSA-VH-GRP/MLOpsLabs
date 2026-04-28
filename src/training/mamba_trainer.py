"""
Mamba4Rec training loop integrated with Feast (offline store) and MLflow.

Entry point: run_mamba_training(experiment_name, hyperparams) -> dict

Data flow inside run_mamba_training():
  1. run_movielens_materialization()          — stage raw Parquets → Feast Parquets + metadata.json
  2. load_metadata_from_minio()              — download metadata.json from MinIO
  3. build_entity_df()                       — all (user_id, event_timestamp) rating-event rows
  4. store.get_historical_features(PIT join) — enrich each row with movie + user features
  5. build_sequences_and_split()             — group by user → leave-one-out train/val/test
  6. create_dataloaders()                    — DataLoaders
  7. Mamba4RecTrainer.fit() + MLflow logging
  8. mlflow.pytorch.log_model()              — register "mamba4rec" in MLflow model registry
"""

import json
import logging
import math
import os
import tempfile
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import ibis
import mlflow
import mlflow.pytorch
import pandas as pd
import torch
import torch.nn as nn
from feast import FeatureStore
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, ReduceLROnPlateau, SequentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.core.config import settings
from src.core.duckdb_client import get_duckdb_connection
from src.core.storage import download_bytes
from src.data.movielens_dataset import create_dataloaders
from src.models.mamba4rec import Mamba4Rec, create_mamba4rec
from src.models.mamba_evaluator import compute_metrics, evaluate_model

logger = logging.getLogger(__name__)

OFFLINE_BUCKET = "offline-store"
METADATA_KEY = "parquet/metadata.json"

# Default hyperparameters  (v2 — optimised)
DEFAULTS = {
    # ── Model architecture ────────────────────────────────────────────────────
    # v5 fix: pack_padded_sequence + masked TUPE pos_bias (see mamba4rec.py)
    "d_model":              64,
    "d_state":              16,
    "n_layers":              3,
    "d_conv":                4,
    "expand":                2,
    "dropout":             0.3,   # v5: increased from 0.2 — more regularisation
    "max_seq_len":          50,
    # ── Training ──────────────────────────────────────────────────────────────
    "batch_size":          128,   # v5: smaller batch — more gradient updates per epoch
    "learning_rate":      2e-4,   # v5: lower than v4 (5e-4 peaked too early at ep4)
    "weight_decay":       1e-3,   # v5: lighter L2 (1e-2 was too restrictive)
    "epochs":               50,
    "patience":             10,   # v5: tighter patience — stop earlier if no improve
    "min_seq_len":           5,
    "max_train_per_user":   20,
    # ── Optimiser extras ──────────────────────────────────────────────────────
    "label_smoothing":     0.1,
    "warmup_epochs":        10,   # v5: longer warmup — LR stabilises before decay
    # ── LR scheduler ──────────────────────────────────────────────────────────
    # v5: ReduceLROnPlateau instead of aggressive cosine annealing.
    # LR halves when val NDCG@10 doesn't improve for 3 epochs.
    "lr_scheduler":       "plateau",   # "plateau" | "cosine"
    "plateau_patience":      3,        # epochs without improvement before LR halves
    "plateau_factor":       0.5,       # LR multiplier on plateau
    "eta_min":            1e-5,        # floor for cosine annealing (if used)
    # ── Ablation flags ────────────────────────────────────────────────────────
    # Set to False to isolate the contribution of each architectural component.
    "use_sum_token":      True,        # learnable SUM token appended at end of seq
    "use_tupe":           True,        # positional bias injected AFTER backbone
    # ── Time interval (TiSASRec-style) ────────────────────────────────────────
    # Consecutive time-gap between interactions, discretised via user-relative
    # normalisation (floor(Δt / Δt_min)) and clipped to num_time_interval_bins.
    "use_time_interval":       False,
    "num_time_interval_bins":  256,    # vocabulary size for interval embeddings
    # ── User profile fusion mode ──────────────────────────────────────────────
    # "broadcast"   — user_emb added to input sequence (original behaviour)
    # "film"        — FiLM scale+shift on last_hidden after backbone
    # "head"        — simple additive 2-layer MLP head
    # "gated_head"  — sigmoid-gated MLP head (recommended for Mamba)
    # "normed_head" — MLP head + LayerNorm
    # "hybrid"      — light broadcast (alpha) + gated_head at output
    "user_fusion_mode":        "broadcast",
    # ── Two-stage training (for head-based fusion modes) ─────────────────────
    # Stage 1: freeze user_head_proj — backbone learns pure sequential patterns
    # Stage 2: unfreeze all — head learns to inject user signal
    "two_stage_training":      False,
    "stage1_epochs":           20,    # how many epochs to freeze the head projection
    "stage1_lr":               2e-4,  # LR during stage 1 (backbone-only)
    # ── Backbone override ─────────────────────────────────────────────────────
    # force_gru=True forces GRU (SimplifiedMamba) even when mamba_ssm is installed.
    # Use this for fair GRU vs Mamba comparisons at controlled sequence lengths.
    "force_gru":               False,
}


# ─── Helper: environment variables for MinIO / Feast ──────────────────────────

def _set_aws_env() -> None:
    """Inject AWS_* env vars so Feast's DuckDB offline store can reach MinIO."""
    endpoint = settings.minio_endpoint.replace("http://", "").replace("https://", "")
    os.environ.setdefault("AWS_ACCESS_KEY_ID",     settings.aws_access_key_id)
    os.environ.setdefault("AWS_SECRET_ACCESS_KEY", settings.aws_secret_access_key)
    os.environ.setdefault("AWS_ENDPOINT_URL",      settings.minio_endpoint)
    os.environ.setdefault("AWS_S3_ENDPOINT",       endpoint)


# ─── Helper: load metadata from MinIO ─────────────────────────────────────────

def load_metadata_from_minio() -> Dict:
    """Download metadata.json from MinIO and return as dict."""
    raw = download_bytes(bucket=OFFLINE_BUCKET, key=METADATA_KEY)
    return json.loads(raw.decode())


# ─── Helper: build entity_df from staged rating events ────────────────────────

def build_entity_df() -> pd.DataFrame:
    """
    Query the staged rating_events Parquet on MinIO and return a DataFrame of
    all (user_id, event_timestamp) pairs.

    Feast will use this as the left-side of the Point-in-Time join:
    each row selects the feature values current at that exact timestamp.
    """
    _set_aws_env()
    conn = get_duckdb_connection()
    df = conn.execute(
        "SELECT user_id, event_timestamp "
        "FROM read_parquet('s3://offline-store/parquet/rating_events/staged.parquet')"
    ).df()
    conn.close()

    # Feast requires event_timestamp to be a timezone-aware datetime
    df["event_timestamp"] = pd.to_datetime(df["event_timestamp"], utc=True)
    df["user_id"] = df["user_id"].astype(int)
    logger.info("entity_df: %d rating events for PIT join", len(df))
    return df


# ─── Helper: load movie genres lookup ─────────────────────────────────────────

def load_movie_genre_lookup() -> Dict[int, List[int]]:
    """
    Read staged movie_features Parquet and return a dict:
        internal_movie_id -> [genre_idx_0, genre_idx_1, genre_idx_2]
    Used to build genre_seq for each item in a user's history.
    """
    _set_aws_env()
    conn = get_duckdb_connection()
    df = conn.execute(
        "SELECT internal_movie_id, genre_idx_0, genre_idx_1, genre_idx_2 "
        "FROM read_parquet('s3://offline-store/parquet/movie_features/staged.parquet')"
    ).df()
    conn.close()
    return {
        int(row.internal_movie_id): [int(row.genre_idx_0), int(row.genre_idx_1), int(row.genre_idx_2)]
        for row in df.itertuples()
    }


# ─── Helper: build sequences and apply leave-one-out split ────────────────────

def _compute_delta_seq(
    timestamps: List[int],
    num_bins: int = 256,
) -> List[int]:
    """
    TiSASRec-style consecutive time-gap encoding.

    For each position i, compute the time elapsed since the previous interaction:
        raw_delta[i] = timestamps[i] - timestamps[i-1]   (seconds)
        raw_delta[0] = 0  (no previous interaction)

    Normalise by the user's minimum non-zero gap so the scale is user-relative:
        delta_norm[i] = floor(raw_delta[i] / min_delta)

    Clip to [0, num_bins - 1] so an Embedding table of size num_bins suffices.
    Index 0 is reserved for "no previous item" (first position) and for padding.
    """
    if len(timestamps) == 0:
        return []

    raw = [0] + [max(0, timestamps[i] - timestamps[i - 1]) for i in range(1, len(timestamps))]

    non_zero = [d for d in raw if d > 0]
    min_delta = min(non_zero) if non_zero else 1  # fallback for single-interaction users

    result = []
    for d in raw:
        if d == 0:
            result.append(0)
        else:
            idx = int(math.floor(d / min_delta))
            result.append(min(idx, num_bins - 1))
    return result


def build_sequences_and_split(
    feature_df: pd.DataFrame,
    genre_lookup: Dict[int, List[int]],
    max_seq_len: int = 50,
    min_seq_len: int = 5,
    max_train_per_user: int = 15,
    num_time_interval_bins: int = 256,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Group feature_df by user_id, sort by event_timestamp, and produce
    train / val / test sample lists compatible with SequentialDataset / EvalDataset.

    Leave-one-out strategy:
        test  = last item
        val   = second-to-last item (history = all except last)
        train = sliding windows over the LAST max_train_per_user positions only
                (caps memory: avoids ~380K Python-dict samples on ML-1M)

    Each sample dict:
        {
            "item_seq":    List[int],
            "time_seq":    List[int],
            "delta_seq":   List[int],   # TiSASRec time intervals (0 = pad / no prev)
            "genre_seq":   List[List[int]],
            "user_profile": {"age_idx": int, "gender_idx": int, "occupation": int},
            "target":      int,
            "target_time": int,
        }
    """
    train_data: List[Dict] = []
    val_data:   List[Dict] = []
    test_data:  List[Dict] = []

    grouped = feature_df.sort_values("event_timestamp").groupby("user_id")

    for user_id, group in grouped:
        group = group.sort_values("event_timestamp").reset_index(drop=True)

        items = group["internal_movie_id"].astype(int).tolist()
        times = group["time_slot"].astype(int).tolist()
        genres = [genre_lookup.get(item, [0, 0, 0]) for item in items]

        # ── TiSASRec: Unix timestamps → consecutive normalised deltas ──────────
        # event_timestamp may be tz-aware datetime or int; convert to int seconds.
        ts_series = pd.to_datetime(group["event_timestamp"], utc=True)
        unix_ts: List[int] = (ts_series.astype("int64") // 10 ** 9).tolist()
        deltas = _compute_delta_seq(unix_ts, num_bins=num_time_interval_bins)

        # Use the last row's user profile (static, so any row is fine)
        last = group.iloc[-1]
        user_profile = {
            "age_idx":    int(last["age_idx"]),
            "gender_idx": int(last["gender_idx"]),
            "occupation": int(last["occupation"]),
        }

        # Need at least min_seq_len interactions
        if len(items) < min_seq_len:
            continue

        # ── Test: last item ────────────────────────────────────────────────────
        test_data.append({
            "item_seq":    items[:-1][-max_seq_len:],
            "time_seq":    times[:-1][-max_seq_len:],
            "delta_seq":   deltas[:-1][-max_seq_len:],
            "genre_seq":   genres[:-1][-max_seq_len:],
            "user_profile": user_profile,
            "target":       items[-1],
            "target_time":  times[-1],
        })

        # ── Val: second-to-last item ───────────────────────────────────────────
        if len(items) >= 2:
            val_data.append({
                "item_seq":    items[:-2][-max_seq_len:],
                "time_seq":    times[:-2][-max_seq_len:],
                "delta_seq":   deltas[:-2][-max_seq_len:],
                "genre_seq":   genres[:-2][-max_seq_len:],
                "user_profile": user_profile,
                "target":       items[-2],
                "target_time":  times[-2],
            })

        # ── Train: last max_train_per_user sliding-window positions ───────────
        # Full sliding window on ML-1M creates ~380K Python-dict samples
        # (~3.4 GB).  Capping to the last max_train_per_user positions per user
        # reduces this to ~6040 × 15 = ~90K samples (~800 MB).
        train_items  = items[:-2]
        train_times  = times[:-2]
        train_genres = genres[:-2]
        train_deltas = deltas[:-2]

        # Only generate samples for the last max_train_per_user target positions
        start_pos = max(1, len(train_items) - max_train_per_user)
        for i in range(start_pos, len(train_items)):
            train_data.append({
                "item_seq":    train_items[max(0, i - max_seq_len):i],
                "time_seq":    train_times[max(0, i - max_seq_len):i],
                "delta_seq":   train_deltas[max(0, i - max_seq_len):i],
                "genre_seq":   train_genres[max(0, i - max_seq_len):i],
                "user_profile": user_profile,
                "target":       train_items[i],
                "target_time":  train_times[i],
            })

    logger.info(
        "Sequences built — train: %d, val: %d, test: %d",
        len(train_data), len(val_data), len(test_data),
    )
    return train_data, val_data, test_data


# ─── Trainer class (adapted from old project) ─────────────────────────────────

class Mamba4RecTrainer:
    """Trains a Mamba4Rec model with AdamW + warmup + CosineAnnealingLR."""

    def __init__(
        self,
        model: Mamba4Rec,
        device: torch.device,
        learning_rate: float = 5e-4,
        weight_decay: float = 1e-2,
        label_smoothing: float = 0.1,
    ):
        self.model = model.to(device)
        self.device = device
        # Label smoothing reduces over-confidence on dominant popular items and
        # acts as regularisation against the popularity bias in ML-1M.
        self.criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=label_smoothing)
        self.optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    def train_epoch(self, train_loader: DataLoader) -> float:
        """Run one training epoch, return average loss."""
        self.model.train()
        total_loss = 0.0

        pbar = tqdm(train_loader, desc="Training", leave=False)
        for batch in pbar:
            item_seq   = batch["item_seq"].to(self.device)
            genre_seq  = batch["genre_seq"].to(self.device)
            time_seq   = batch["time_seq"].to(self.device)
            delta_seq  = batch["delta_seq"].to(self.device) if "delta_seq" in batch else None
            age_idx    = batch["age_idx"].to(self.device)
            gender_idx = batch["gender_idx"].to(self.device)
            occupation = batch["occupation"].to(self.device)
            target     = batch["target"].squeeze(-1).to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(
                item_seq, genre_seq, time_seq, age_idx, gender_idx, occupation,
                delta_seq=delta_seq,
            )
            loss = self.criterion(logits, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        return total_loss / len(train_loader)

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Tuple[float, float, float, float]:
        """Return (val_loss≈0, hit@10, ndcg@10, mrr@10)."""
        self.model.eval()
        all_hits, all_ndcgs, all_mrrs = [], [], []

        for batch in tqdm(val_loader, desc="Validating", leave=False):
            item_seq   = batch["item_seq"].to(self.device)
            genre_seq  = batch["genre_seq"].to(self.device)
            time_seq   = batch["time_seq"].to(self.device)
            delta_seq  = batch["delta_seq"].to(self.device) if "delta_seq" in batch else None
            age_idx    = batch["age_idx"].to(self.device)
            gender_idx = batch["gender_idx"].to(self.device)
            occupation = batch["occupation"].to(self.device)
            candidates = batch["candidates"].to(self.device)

            scores = self.model.predict_scores(
                item_seq, genre_seq, time_seq, age_idx, gender_idx, occupation,
                candidate_items=candidates,
                delta_seq=delta_seq,
            )
            hit, ndcg, mrr = compute_metrics(scores, k=10)
            all_hits.append(hit)
            all_ndcgs.append(ndcg)
            all_mrrs.append(mrr)

        avg_hit  = torch.cat(all_hits).mean().item()
        avg_ndcg = torch.cat(all_ndcgs).mean().item()
        avg_mrr  = torch.cat(all_mrrs).mean().item()
        return 0.0, avg_hit, avg_ndcg, avg_mrr


# ─── Feast / ibis S3 patch ────────────────────────────────────────────────────

def _patch_feast_duckdb_s3() -> None:
    """
    Configure the default ibis/DuckDB backend with MinIO S3 settings and patch
    feast.infra.offline_stores.duckdb._read_data_source to use it.

    Problem: feast's DuckDB offline store calls ibis.read_parquet(path) without
    any S3 credentials or endpoint configuration, causing 403/NoneType errors
    for MinIO-hosted Parquet files.

    Fix: create one DuckDB connection, configure httpfs for MinIO, set it as
    ibis's default backend (ibis.set_backend), and patch _read_data_source to
    call read_parquet on that same connection.  This ensures entity_table
    (ibis.memtable) and feature tables (ibis.read_parquet) share one backend
    and can be joined together by Feast.
    """
    import feast.infra.offline_stores.duckdb as _feast_duckdb

    if getattr(_feast_duckdb, "_minio_patched", False):
        return  # already applied

    endpoint  = os.environ.get("AWS_S3_ENDPOINT",       "minio:9000")
    key_id    = os.environ.get("AWS_ACCESS_KEY_ID",     "minioadmin")
    secret    = os.environ.get("AWS_SECRET_ACCESS_KEY", "minioadmin123")

    # Build one DuckDB connection and configure MinIO S3 access on it.
    con = ibis.duckdb.connect()
    con.raw_sql("INSTALL httpfs; LOAD httpfs;")
    con.raw_sql(f"SET s3_endpoint='{endpoint}';")
    con.raw_sql(f"SET s3_access_key_id='{key_id}';")
    con.raw_sql(f"SET s3_secret_access_key='{secret}';")
    con.raw_sql("SET s3_use_ssl=false;")
    con.raw_sql("SET s3_url_style='path';")

    # Make this the default ibis backend so ibis.memtable / ibis.read_parquet
    # all share the same (configured) DuckDB connection.
    ibis.set_backend(con)
    logger.info("ibis default DuckDB backend configured for MinIO (endpoint=%s)", endpoint)

    def _patched_read_data_source(data_source, repo_path: str):
        path = data_source.path
        is_delta = (
            hasattr(data_source, "file_format")
            and data_source.file_format.__class__.__name__ == "DeltaFormat"
        )
        if is_delta:
            storage_options: Dict = {}
            if getattr(data_source, "s3_endpoint_override", None):
                storage_options["AWS_ENDPOINT_URL"] = data_source.s3_endpoint_override
            return ibis.read_delta(path, storage_options=storage_options)

        # Parquet (or anything else): use the already-configured default backend.
        logger.debug("read_parquet via configured backend: %s", path)
        return ibis.read_parquet(path)

    _feast_duckdb._read_data_source = _patched_read_data_source
    _feast_duckdb._minio_patched = True
    logger.info("feast DuckDB _read_data_source patched for MinIO S3")


# ─── Main entry point ─────────────────────────────────────────────────────────

def run_mamba_training(experiment_name: str, hyperparams: Dict) -> Dict:
    """
    Full Mamba4Rec training pipeline (called by the dispatcher in trainer.py).

    Args:
        experiment_name: MLflow experiment name
        hyperparams:     Overrides for DEFAULTS dict

    Returns:
        {"run_id": str, "model_version": int, "status": "registered"}
    """
    hp = {**DEFAULTS, **hyperparams}

    # ── Step 1: Materialize MovieLens features (skip if already done) ─────────
    _set_aws_env()
    from src.core.storage import get_s3_client as _s3c

    _STAGED_KEYS = [
        "parquet/rating_events/staged.parquet",
        "parquet/user_features/staged.parquet",
        "parquet/movie_features/staged.parquet",
        "parquet/metadata.json",
    ]

    def _staged_files_exist() -> bool:
        try:
            existing = {
                obj["Key"]
                for obj in _s3c().list_objects_v2(
                    Bucket="offline-store", Prefix="parquet/"
                ).get("Contents", [])
            }
            return all(k in existing for k in _STAGED_KEYS)
        except Exception:
            return False

    if _staged_files_exist():
        logger.info("Step 1: Staged files already present in MinIO — skipping materialization.")
    else:
        logger.info("Step 1: Running MovieLens materialization …")
        from src.features.materialization import run_movielens_materialization
        run_movielens_materialization()

    # ── Step 2: Load metadata ─────────────────────────────────────────────────
    logger.info("Step 2: Loading metadata from MinIO …")
    metadata = load_metadata_from_minio()

    # ── Step 3: Retrieve all rating events + user profile via DuckDB join ────
    # User features (age, gender, occupation) are static in MovieLens — a simple
    # LEFT JOIN is equivalent to a PIT join and is far more memory-efficient than
    # Feast's ibis-based PIT join over 1.7M rows.
    logger.info("Step 3: Joining rating events with user + movie features …")
    _set_aws_env()

    conn = get_duckdb_connection()
    try:
        feature_df = conn.execute("""
            SELECT
                r.user_id,
                r.event_timestamp,
                r.internal_movie_id,
                r.rating,
                r.time_slot,
                u.gender_idx,
                u.age_idx,
                u.occupation
            FROM read_parquet('s3://offline-store/parquet/rating_events/staged.parquet') r
            LEFT JOIN read_parquet('s3://offline-store/parquet/user_features/staged.parquet')  u
                ON r.user_id = u.user_id
        """).df()
    finally:
        conn.close()

    feature_df = feature_df.dropna(
        subset=["internal_movie_id", "age_idx", "gender_idx", "occupation"]
    )
    logger.info("Feature DataFrame shape after join: %s", feature_df.shape)

    # ── Step 4: Build sequences and split ────────────────────────────────────
    logger.info("Step 4: Building sequences and applying leave-one-out split …")
    genre_lookup = load_movie_genre_lookup()
    train_data, val_data, test_data = build_sequences_and_split(
        feature_df,
        genre_lookup=genre_lookup,
        max_seq_len=hp["max_seq_len"],
        min_seq_len=hp["min_seq_len"],
        max_train_per_user=hp["max_train_per_user"],
        num_time_interval_bins=hp.get("num_time_interval_bins", 256),
    )

    # ── Step 5: Create DataLoaders ────────────────────────────────────────────
    logger.info("Step 5: Creating DataLoaders (batch_size=%d) …", hp["batch_size"])
    train_loader, val_loader, test_loader = create_dataloaders(
        train_data, val_data, test_data, metadata,
        batch_size=hp["batch_size"],
        num_workers=0,            # Windows-safe
        max_seq_len=hp["max_seq_len"],
    )

    # ── Step 6: Train with MLflow tracking ───────────────────────────────────
    logger.info("Step 6: Starting training …")
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(experiment_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training on device: %s", device)

    from src.models.mamba4rec import MAMBA_AVAILABLE

    with mlflow.start_run() as run:
        # Log hyperparameters
        mlflow.log_params({
            "model_type":           "mamba4rec",
            "d_model":              hp["d_model"],
            "d_state":              hp["d_state"],
            "n_layers":             hp["n_layers"],
            "d_conv":               hp["d_conv"],
            "expand":               hp["expand"],
            "dropout":              hp["dropout"],
            "max_seq_len":          hp["max_seq_len"],
            "batch_size":           hp["batch_size"],
            "learning_rate":        hp["learning_rate"],
            "weight_decay":         hp["weight_decay"],
            "epochs":               hp["epochs"],
            "patience":             hp["patience"],
            "max_train_per_user":   hp["max_train_per_user"],
            "label_smoothing":      hp["label_smoothing"],
            "warmup_epochs":        hp["warmup_epochs"],
            "lr_scheduler":         hp.get("lr_scheduler", "cosine"),
            "plateau_patience":     hp.get("plateau_patience", 3),
            "plateau_factor":       hp.get("plateau_factor", 0.5),
            "eta_min":              hp.get("eta_min", 1e-5),
            "num_items":            metadata["num_items"],
            "num_genres":           metadata["num_genres"],
            "mamba_ssm_available":  MAMBA_AVAILABLE,
            "use_sum_token":           hp.get("use_sum_token", True),
            "use_tupe":                hp.get("use_tupe", True),
            "use_time_interval":       hp.get("use_time_interval", False),
            "num_time_interval_bins":  hp.get("num_time_interval_bins", 256),
            "user_fusion_mode":        hp.get("user_fusion_mode", "broadcast"),
            "two_stage_training":      hp.get("two_stage_training", False),
            "stage1_epochs":           hp.get("stage1_epochs", 20),
            "force_gru":               hp.get("force_gru", False),
        })

        # Instantiate model
        model = create_mamba4rec(
            metadata,
            d_model=hp["d_model"],
            d_state=hp["d_state"],
            n_layers=hp["n_layers"],
            d_conv=hp["d_conv"],
            expand=hp["expand"],
            dropout=hp["dropout"],
            max_seq_len=hp["max_seq_len"],
            use_sum_token=hp.get("use_sum_token", True),
            use_tupe=hp.get("use_tupe", True),
            use_time_interval=hp.get("use_time_interval", False),
            num_time_interval_bins=hp.get("num_time_interval_bins", 256),
            user_fusion_mode=hp.get("user_fusion_mode", "broadcast"),
            force_gru=hp.get("force_gru", False),
        )

        trainer = Mamba4RecTrainer(
            model=model,
            device=device,
            learning_rate=hp["learning_rate"],
            weight_decay=hp["weight_decay"],
            label_smoothing=hp["label_smoothing"],
        )

        # v5 LR schedule:
        #   Phase 1: Linear warmup for warmup_epochs (LR: 0.1x → 1.0x)
        #   Phase 2 (plateau):   ReduceLROnPlateau — halves LR when val NDCG
        #                        stagnates for plateau_patience epochs.
        #   Phase 2 (cosine):    CosineAnnealingLR from peak LR down to eta_min.
        # ReduceLROnPlateau is preferred: it adapts to the training dynamics
        # rather than decaying on a fixed schedule that may overshoot.
        warmup_epochs = max(1, hp["warmup_epochs"])
        warmup_sched = LinearLR(
            trainer.optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_epochs,
        )

        use_plateau = hp.get("lr_scheduler", "plateau") == "plateau"
        if use_plateau:
            # ReduceLROnPlateau monitors val NDCG; step() called with the metric
            post_warmup_sched = ReduceLROnPlateau(
                trainer.optimizer,
                mode="max",
                factor=hp.get("plateau_factor", 0.5),
                patience=hp.get("plateau_patience", 3),
                min_lr=hp.get("eta_min", 1e-5),
            )
        else:
            post_warmup_sched = CosineAnnealingLR(
                trainer.optimizer,
                T_max=max(1, hp["epochs"] - warmup_epochs),
                eta_min=hp.get("eta_min", 1e-5),
            )

        best_ndcg = 0.0
        patience_counter = 0
        best_model_state = None

        # ── Two-stage training setup ──────────────────────────────────────────
        # Stage 1: freeze head projection so backbone learns uncontaminated
        #          sequential patterns. Stage 2: unfreeze and fine-tune together.
        _HEAD_PARAM_KEYS = ("user_head_proj", "user_gate", "user_head_norm", "hybrid_alpha")
        two_stage = (
            hp.get("two_stage_training", False)
            and hp.get("user_fusion_mode", "broadcast") in
                ("head", "gated_head", "normed_head", "hybrid")
        )
        stage1_epochs = hp.get("stage1_epochs", 20)
        _in_stage1 = False  # tracks whether we toggled param groups this epoch

        def _set_head_frozen(frozen: bool) -> None:
            for name, param in model.named_parameters():
                if any(k in name for k in _HEAD_PARAM_KEYS):
                    param.requires_grad = not frozen

        if two_stage:
            _set_head_frozen(True)
            logger.info("Two-stage training: Stage 1 — backbone-only for %d epochs", stage1_epochs)

        for epoch in range(hp["epochs"]):
            # Transition from Stage 1 → Stage 2
            if two_stage and epoch == stage1_epochs:
                _set_head_frozen(False)
                logger.info("Two-stage training: Stage 2 — all params unfrozen at epoch %d", epoch + 1)
            logger.info("Epoch %d/%d", epoch + 1, hp["epochs"])

            train_loss = trainer.train_epoch(train_loader)
            _, hit10, ndcg10, mrr10 = trainer.validate(val_loader)

            mlflow.log_metrics(
                {
                    "train_loss":    train_loss,
                    "val_hit_at_10":  hit10,
                    "val_ndcg_at_10": ndcg10,
                    "val_mrr_at_10":  mrr10,
                    "learning_rate":  trainer.optimizer.param_groups[0]["lr"],
                },
                step=epoch,
            )

            logger.info(
                "Epoch %d — loss=%.4f  hit@10=%.4f  ndcg@10=%.4f  mrr@10=%.4f  lr=%.2e",
                epoch + 1, train_loss, hit10, ndcg10, mrr10,
                trainer.optimizer.param_groups[0]["lr"],
            )

            # Step schedulers
            if epoch < warmup_epochs:
                warmup_sched.step()
            else:
                if use_plateau:
                    post_warmup_sched.step(ndcg10)   # plateau monitors val NDCG
                else:
                    post_warmup_sched.step()

            if ndcg10 > best_ndcg:
                best_ndcg = ndcg10
                patience_counter = 0
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                logger.info("  → New best NDCG@10: %.4f", best_ndcg)
            else:
                patience_counter += 1
                if patience_counter >= hp["patience"]:
                    logger.info("Early stopping at epoch %d", epoch + 1)
                    break

        # Restore best model weights
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            model.to(device)

        # Final test evaluation
        logger.info("Running final test evaluation …")
        test_metrics = evaluate_model(model, test_loader, device, ks=[5, 10, 20])
        mlflow.log_metrics(
            {
                f"test_{k.lower().replace('@', '_at_')}": v
                for k, v in test_metrics.items()
            }
        )
        logger.info("Test metrics: %s", test_metrics)

        # Log PyTorch model to MLflow registry
        mlflow.pytorch.log_model(
            model,
            artifact_path="model",
            registered_model_name="mamba4rec",
        )

        # Log metadata.json and movies_info.json as MLflow artifacts.
        # Write to a temp directory with fixed filenames so the predictor can find them.
        with tempfile.TemporaryDirectory() as tmpdir:
            meta_path = os.path.join(tmpdir, "metadata.json")
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f)
            mlflow.log_artifact(meta_path, artifact_path="data")

            # movies_info.json — download from MinIO then re-log
            from src.core.storage import download_bytes as _dl
            try:
                movies_info_bytes = _dl(bucket=OFFLINE_BUCKET, key="parquet/movies_info.json")
                info_path = os.path.join(tmpdir, "movies_info.json")
                with open(info_path, "wb") as f:
                    f.write(movies_info_bytes)
                mlflow.log_artifact(info_path, artifact_path="data")
            except Exception as exc:
                logger.warning("Could not upload movies_info.json to MLflow: %s", exc)

        run_id = run.info.run_id
        logger.info("MLflow run completed: run_id=%s", run_id)

    # Retrieve the latest registered version and promote it to "champion"
    client = mlflow.MlflowClient()
    versions = client.get_latest_versions("mamba4rec")
    latest_version = max(int(v.version) for v in versions) if versions else 1

    try:
        client.set_registered_model_alias("mamba4rec", "champion", str(latest_version))
        logger.info("MLflow alias 'champion' → mamba4rec v%d", latest_version)
    except Exception as exc:
        logger.warning("Could not set champion alias: %s", exc)

    return {"run_id": run_id, "model_version": latest_version, "status": "registered"}
