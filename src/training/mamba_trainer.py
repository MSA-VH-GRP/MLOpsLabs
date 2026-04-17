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
import os
import tempfile
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import mlflow
import mlflow.pytorch
import pandas as pd
import torch
import torch.nn as nn
from feast import FeatureStore
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
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

# Default hyperparameters
DEFAULTS = {
    "d_model":       64,
    "d_state":       16,
    "n_layers":      2,
    "d_conv":        4,
    "expand":        2,
    "dropout":       0.1,
    "max_seq_len":   50,
    "batch_size":    256,
    "learning_rate": 1e-3,
    "weight_decay":  1e-4,
    "epochs":        50,
    "patience":      10,
    "min_seq_len":   5,
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

def build_sequences_and_split(
    feature_df: pd.DataFrame,
    genre_lookup: Dict[int, List[int]],
    max_seq_len: int = 50,
    min_seq_len: int = 5,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Group feature_df by user_id, sort by event_timestamp, and produce
    train / val / test sample lists compatible with SequentialDataset / EvalDataset.

    Leave-one-out strategy:
        test  = last item
        val   = second-to-last item (history = all except last)
        train = sliding windows over the rest (one sample per position)

    Each sample dict:
        {
            "item_seq":    List[int],
            "time_seq":    List[int],
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
            "item_seq":    items[:-1],
            "time_seq":    times[:-1],
            "genre_seq":   genres[:-1],
            "user_profile": user_profile,
            "target":       items[-1],
            "target_time":  times[-1],
        })

        # ── Val: second-to-last item ───────────────────────────────────────────
        if len(items) >= 2:
            val_data.append({
                "item_seq":    items[:-2],
                "time_seq":    times[:-2],
                "genre_seq":   genres[:-2],
                "user_profile": user_profile,
                "target":       items[-2],
                "target_time":  times[-2],
            })

        # ── Train: sliding windows over items[:-2] ─────────────────────────────
        train_items  = items[:-2]
        train_times  = times[:-2]
        train_genres = genres[:-2]

        for i in range(1, len(train_items)):
            train_data.append({
                "item_seq":    train_items[:i],
                "time_seq":    train_times[:i],
                "genre_seq":   train_genres[:i],
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
    """Trains a Mamba4Rec model with AdamW + CosineAnnealingLR."""

    def __init__(
        self,
        model: Mamba4Rec,
        device: torch.device,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
    ):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
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
            age_idx    = batch["age_idx"].to(self.device)
            gender_idx = batch["gender_idx"].to(self.device)
            occupation = batch["occupation"].to(self.device)
            target     = batch["target"].squeeze(-1).to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(item_seq, genre_seq, time_seq, age_idx, gender_idx, occupation)
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
            age_idx    = batch["age_idx"].to(self.device)
            gender_idx = batch["gender_idx"].to(self.device)
            occupation = batch["occupation"].to(self.device)
            candidates = batch["candidates"].to(self.device)

            scores = self.model.predict_scores(
                item_seq, genre_seq, time_seq, age_idx, gender_idx, occupation,
                candidate_items=candidates,
            )
            hit, ndcg, mrr = compute_metrics(scores, k=10)
            all_hits.append(hit)
            all_ndcgs.append(ndcg)
            all_mrrs.append(mrr)

        avg_hit  = torch.cat(all_hits).mean().item()
        avg_ndcg = torch.cat(all_ndcgs).mean().item()
        avg_mrr  = torch.cat(all_mrrs).mean().item()
        return 0.0, avg_hit, avg_ndcg, avg_mrr


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

    # ── Step 1: Materialize MovieLens features into Feast ─────────────────────
    logger.info("Step 1: Running MovieLens materialization …")
    from src.features.materialization import run_movielens_materialization
    run_movielens_materialization()

    # ── Step 2: Load metadata ─────────────────────────────────────────────────
    logger.info("Step 2: Loading metadata from MinIO …")
    metadata = load_metadata_from_minio()

    # ── Step 3: Retrieve all rating events + user profile via Feast PIT join ──
    logger.info("Step 3: Fetching historical features from Feast (PIT join) …")
    _set_aws_env()
    store = FeatureStore(repo_path="feast/")

    entity_df = build_entity_df()

    feature_df = store.get_historical_features(
        entity_df=entity_df,
        features=[
            "rating_event_features:internal_movie_id",
            "rating_event_features:rating",
            "rating_event_features:time_slot",
            "user_profile_features:gender_idx",
            "user_profile_features:age_idx",
            "user_profile_features:occupation",
        ],
    ).to_df()

    # Drop rows where the PIT join found no matching feature (e.g. new users)
    feature_df = feature_df.dropna(
        subset=["internal_movie_id", "age_idx", "gender_idx", "occupation"]
    )
    logger.info("Feature DataFrame shape after PIT join: %s", feature_df.shape)

    # ── Step 4: Build sequences and split ────────────────────────────────────
    logger.info("Step 4: Building sequences and applying leave-one-out split …")
    genre_lookup = load_movie_genre_lookup()
    train_data, val_data, test_data = build_sequences_and_split(
        feature_df,
        genre_lookup=genre_lookup,
        max_seq_len=hp["max_seq_len"],
        min_seq_len=hp["min_seq_len"],
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
            "model_type":          "mamba4rec",
            "d_model":             hp["d_model"],
            "d_state":             hp["d_state"],
            "n_layers":            hp["n_layers"],
            "d_conv":              hp["d_conv"],
            "expand":              hp["expand"],
            "dropout":             hp["dropout"],
            "max_seq_len":         hp["max_seq_len"],
            "batch_size":          hp["batch_size"],
            "learning_rate":       hp["learning_rate"],
            "weight_decay":        hp["weight_decay"],
            "epochs":              hp["epochs"],
            "patience":            hp["patience"],
            "num_items":           metadata["num_items"],
            "num_genres":          metadata["num_genres"],
            "mamba_ssm_available": MAMBA_AVAILABLE,
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
        )

        trainer = Mamba4RecTrainer(
            model=model,
            device=device,
            learning_rate=hp["learning_rate"],
            weight_decay=hp["weight_decay"],
        )

        scheduler = CosineAnnealingLR(trainer.optimizer, T_max=hp["epochs"], eta_min=1e-6)

        best_ndcg = 0.0
        patience_counter = 0
        best_model_state = None

        for epoch in range(hp["epochs"]):
            logger.info("Epoch %d/%d", epoch + 1, hp["epochs"])

            train_loss = trainer.train_epoch(train_loader)
            _, hit10, ndcg10, mrr10 = trainer.validate(val_loader)

            mlflow.log_metrics(
                {
                    "train_loss":  train_loss,
                    "val_hit@10":  hit10,
                    "val_ndcg@10": ndcg10,
                    "val_mrr@10":  mrr10,
                },
                step=epoch,
            )

            logger.info(
                "Epoch %d — loss=%.4f  hit@10=%.4f  ndcg@10=%.4f  mrr@10=%.4f",
                epoch + 1, train_loss, hit10, ndcg10, mrr10,
            )

            scheduler.step()

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

    # Retrieve the latest registered version
    client = mlflow.MlflowClient()
    versions = client.get_latest_versions("mamba4rec")
    latest_version = max(int(v.version) for v in versions) if versions else 1

    return {"run_id": run_id, "model_version": latest_version, "status": "registered"}
