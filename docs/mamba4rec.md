# Mamba4Rec — Sequential Movie Recommendation

Mamba4Rec is a sequential recommendation model based on the **Mamba State Space Model (SSM)** backbone. It learns to predict the next movie a user will watch from their ordered viewing history combined with contextual features (genre, time of day, user demographics).

---

## Model Architecture

```
Input per user:
  item_seq    = [movie_1, movie_2, ..., movie_T]   (internal IDs, 0 = padding)
  time_seq    = [slot_1,  slot_2,  ..., slot_T]    (0/1/2)
  genre_seq   = [[g0,g1,g2], ..., [g0,g1,g2]]      (up to 3 genres per item)
  age_idx, gender_idx, occupation                  (user profile scalars)

                    ┌──────────────────────────────────────┐
                    │          Embedding Fusion             │
                    │                                       │
                    │  Item Emb                             │
                    │  + Genre Pool (mean over ≤3 genres)   │
                    │  + User Profile (age+gender+occ)      │  broadcast
                    │  + Time Slot Emb                      │
                    │  + Positional Emb                     │
                    │  → Dropout → Padding Mask             │
                    └───────────────┬──────────────────────┘
                                    │
                    ┌───────────────▼──────────────────────┐
                    │         MambaBlock × N                │
                    │  LayerNorm → SSM → Residual           │
                    │                                       │
                    │  mamba_ssm installed? → official SSM  │
                    │  otherwise           → SimplifiedMamba│
                    │                        (GRU-based)    │
                    └───────────────┬──────────────────────┘
                                    │  last valid hidden state
                                    ▼
                          Linear (d_model → num_items)
                                    │
                                    ▼
                            next-item logits
```

### Embeddings

| Embedding | Shape | Notes |
|---|---|---|
| Item | `(num_items, d_model)` | `padding_idx=0` |
| Genre | `(num_genres, d_model)` | Mean-pooled across ≤3 genres per item; `padding_idx=0` |
| Age | `(7, d_model)` | 7 ML-1M age groups |
| Gender | `(2, d_model)` | 0=Female, 1=Male |
| Occupation | `(21, d_model)` | ML-1M occupation codes 0–20 |
| Time Slot | `(3, d_model)` | 0=Matinee, 1=Prime Time, 2=Late Night |
| Position | `(max_seq_len, d_model)` | Standard learned positional encoding |

### Default Hyperparameters

| Parameter | Default | Description |
|---|---|---|
| `d_model` | 64 | Embedding / hidden dimension |
| `d_state` | 16 | SSM state dimension |
| `n_layers` | 2 | Number of stacked MambaBlocks |
| `d_conv` | 4 | Depthwise conv kernel size in Mamba |
| `expand` | 2 | Inner dimension expansion factor |
| `dropout` | 0.1 | Applied after embedding fusion |
| `max_seq_len` | 50 | Input sequence length (shorter seqs are left-padded) |

### SimplifiedMamba (CPU fallback)

When `mamba_ssm` is not installed (no CUDA or Windows without NVCC), `MambaBlock` automatically falls back to `SimplifiedMamba`:
- 1D depthwise convolution over the sequence
- GRU for the recurrence approximation
- SiLU gating

This runs on CPU without any compilation and produces similar qualitative results.

To use the official Mamba kernel (GPU required):
```bash
pip install ".[mamba-gpu]"
```

---

## Training Pipeline

### Full flow inside `run_mamba_training()`

```
POST /train  {model_type: "mamba4rec"}
    │
    ▼  Step 1
run_movielens_materialization()
→ reads raw-data Parquets from MinIO
→ stages user_features / movie_features / rating_events / metadata.json
→ (idempotent — safe to re-run)
    │
    ▼  Step 2
load_metadata_from_minio()
→ downloads s3://offline-store/parquet/metadata.json
→ provides vocab sizes for model initialisation
    │
    ▼  Step 3
build_entity_df()
→ queries rating_events.parquet via DuckDB
→ returns all (user_id, event_timestamp) pairs
    │
    ▼  Step 4 — Feast Point-in-Time join
store.get_historical_features(entity_df, features=[...])
→ rating_event_features  (value at exact timestamp)
→ user_profile_features  (latest value ≤ timestamp)
→ returns enriched DataFrame
    │
    ▼  Step 5
build_sequences_and_split()
→ group by user_id, sort by event_timestamp
→ filter users with < 5 interactions
→ leave-one-out split:
    test  = last item per user
    val   = second-to-last item per user
    train = sliding windows over the rest
    │
    ▼  Step 6
create_dataloaders()   (num_workers=0 for Windows/WSL safety)
    │
    ▼  Step 7 — MLflow tracking
mlflow.start_run()
→ log_params (all hyperparameters + vocab sizes)
→ per epoch: train_loss, val_hit@10, val_ndcg@10, val_mrr@10
→ early stopping on val_ndcg@10 (patience = 10)
→ restore best weights
→ evaluate on test set: Hit/NDCG/MRR @ {5, 10, 20}
→ mlflow.pytorch.log_model() → registered as "mamba4rec"
→ log_artifact: data/metadata.json, data/movies_info.json
```

### MLflow logged data

| Type | Keys |
|---|---|
| Params | `model_type`, `d_model`, `d_state`, `n_layers`, `d_conv`, `expand`, `dropout`, `max_seq_len`, `batch_size`, `learning_rate`, `weight_decay`, `epochs`, `patience`, `num_items`, `num_genres`, `mamba_ssm_available` |
| Metrics (per epoch) | `train_loss`, `val_hit@10`, `val_ndcg@10`, `val_mrr@10` |
| Metrics (final) | `test_hit@5/10/20`, `test_ndcg@5/10/20`, `test_mrr@5/10/20` |
| Artifacts | `model/` (PyTorch), `data/metadata.json`, `data/movies_info.json` |

---

## Evaluation Metrics

All metrics are computed over a **candidate set** of 1 target + 99 randomly sampled negatives per user interaction.

| Metric | Formula |
|---|---|
| **Hit@K** | `1` if target item appears in top-K predictions, else `0` |
| **NDCG@K** | `1 / log₂(rank + 1)` if `rank ≤ K`, else `0` |
| **MRR@K** | `1 / rank` if `rank ≤ K`, else `0` |

Reported for **K ∈ {5, 10, 20}**.

---

## Running the Pipeline (WSL + venv)

### Step 1 — Install Python 3.11

```bash
sudo apt-get update
sudo apt-get install -y software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt-get update
sudo apt-get install -y python3.11 python3.11-venv python3.11-dev
python3.11 --version
```

### Step 2 — Create venv and install dependencies

```bash
cd "/mnt/d/Study/MSE/AI in DevOps, DataOps, MLOps/FinalProject"

python3.11 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"
```

### Step 3 — Start Docker services

```bash
docker compose up -d
docker compose ps   # wait until all healthy
```

### Step 4 — Create Kafka topics

```bash
bash scripts/create_kafka_topics.sh
```

### Step 5 — Run ingestion consumer (background)

```bash
python -m src.pipelines.ingest_pipeline &
CONSUMER_PID=$!
```

### Step 6 — Stream MovieLens 1M into Kafka

```bash
python scripts/simulate_ingestion.py --speedup 10000
kill $CONSUMER_PID   # stop consumer after simulator finishes
```

Verify data arrived: open **http://localhost:9001** → bucket `raw-data` should contain `new_user/`, `new_movie/`, `new_rating/`.

### Step 7 — Register Feast feature views

```bash
bash scripts/feast_apply.sh
```

### Step 8 — Start the API server

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

Open a second terminal with the venv activated for the next steps.

### Step 9 — Trigger training

```bash
curl -s -X POST http://localhost:8000/train \
  -H "Content-Type: application/json" \
  -d '{
    "experiment_name": "mamba4rec-v1",
    "model_type": "mamba4rec",
    "hyperparams": {
      "d_model": 64, "d_state": 16, "n_layers": 2,
      "d_conv": 4, "expand": 2, "dropout": 0.1,
      "max_seq_len": 50, "batch_size": 256,
      "learning_rate": 0.001, "weight_decay": 0.0001,
      "epochs": 50, "patience": 10
    }
  }' | python3 -m json.tool
```

Monitor at **http://localhost:5000**.

**Quick smoke test (2 epochs):**

```bash
curl -s -X POST http://localhost:8000/train \
  -H "Content-Type: application/json" \
  -d '{
    "experiment_name": "mamba4rec-smoke",
    "model_type": "mamba4rec",
    "hyperparams": {"epochs": 2, "batch_size": 64}
  }' | python3 -m json.tool
```

### Step 10 — Set champion alias

After training, assign the `champion` alias via the MLflow UI (**http://localhost:5000 → Models → mamba4rec → version 1 → Add alias**), or:

```bash
python3 -c "
import mlflow
mlflow.set_tracking_uri('http://localhost:5000')
client = mlflow.MlflowClient()
client.set_registered_model_alias('mamba4rec', 'champion', '1')
print('Done.')
"
```

### Step 11 — Test predictions

```bash
curl -s -X POST http://localhost:8000/predict/mamba \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "mamba4rec",
    "model_alias": "champion",
    "item_history": [1, 5, 10, 20, 50],
    "time_history": [1, 0, 2, 1, 1],
    "age_idx": 2,
    "gender_idx": 1,
    "occupation": 4,
    "top_k": 5,
    "target_time": 1
  }' | python3 -m json.tool
```

---

## Internal IDs

`item_history` in prediction requests uses **internal movie IDs** (1-based integers assigned during materialization, sorted by original MovieLens `movie_id`). The mapping is stored in:
- `s3://offline-store/parquet/metadata.json` → `movie_id_map` / `reverse_movie_map`
- MLflow artifact `data/metadata.json` (attached to every training run)

To look up an internal ID for a given original MovieLens movie ID:
```python
import json
from src.core.storage import download_bytes

metadata = json.loads(download_bytes("offline-store", "parquet/metadata.json"))
internal_id = metadata["movie_id_map"]["593"]   # Silence of the Lambs
```
