# Development Guide

---

## Setup

### Prerequisites

- Python 3.11
- Docker Desktop (or Docker Engine + Compose plugin)
- WSL2 recommended on Windows for running the training pipeline

### Install dependencies

```bash
# Create venv (Linux / WSL)
python3.11 -m venv .venv
source .venv/bin/activate

# CPU-only PyTorch (default)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"

# GPU (requires CUDA + NVCC)
pip install torch
pip install -e ".[dev,mamba-gpu]"
```

### Environment variables

Copy `.env.example` to `.env`. All services read from `.env` via Docker Compose.

Key variables:

| Variable | Default | Description |
|---|---|---|
| `MINIO_ROOT_USER` | `minioadmin` | MinIO access key |
| `MINIO_ROOT_PASSWORD` | `minioadmin123` | MinIO secret key |
| `MLFLOW_TRACKING_URI` | `http://localhost:5000` | MLflow server |
| `REDIS_URL` | `redis://localhost:6379` | Redis for Feast online store |
| `KAFKA_BOOTSTRAP_SERVERS` | `localhost:9092` | Kafka broker |

---

## Running Tests

```bash
# All tests
pytest

# Unit tests only
pytest tests/unit/

# Integration tests only (requires Docker services running)
pytest tests/integration/

# With output
pytest -s -v
```

Test layout:

```
tests/
├── conftest.py              # shared fixtures
├── unit/
│   ├── test_features.py     # materialization helpers
│   ├── test_schemas.py      # Pydantic request/response schemas
│   └── test_models.py       # model utilities
└── integration/
    ├── test_api.py          # FastAPI endpoint tests (httpx)
    └── test_pipeline.py     # end-to-end pipeline tests
```

---

## Linting

```bash
# Check
ruff check .

# Auto-fix
ruff check . --fix

# Format
ruff format .
```

Config in `pyproject.toml`: line length 100, rules `E F I UP`, target Python 3.11.

---

## Type Checking

```bash
mypy src/
```

`strict = false`; `ignore_missing_imports = true` — add per-module overrides in `pyproject.toml` as needed.

---

## Project Structure

```
.
├── docker-compose.yml          # full infrastructure stack
├── pyproject.toml              # dependencies, tool config
├── .env.example
├── feast/                      # Feast feature store config
│   ├── feature_store.yaml      # offline=DuckDB, online=Redis
│   └── feature_views/
│       ├── feature_views.py    # generic raw_event_features
│       └── movielens_features.py  # Mamba4Rec feature views
├── scripts/
│   ├── simulate_ingestion.py   # streams MovieLens 1M → Kafka
│   ├── create_kafka_topics.sh  # creates new-user/movie/rating + raw-events
│   └── feast_apply.sh          # runs `feast apply`
├── src/
│   ├── api/                    # FastAPI application
│   │   ├── main.py
│   │   ├── routers/
│   │   │   ├── health.py
│   │   │   ├── ingest.py
│   │   │   ├── train.py
│   │   │   └── predict.py      # /predict and /predict/mamba
│   │   └── schemas/
│   │       ├── train.py
│   │       └── predict.py
│   ├── core/
│   │   ├── config.py           # settings from env
│   │   ├── duckdb_client.py    # DuckDB + MinIO connection helper
│   │   └── storage.py          # MinIO upload/download helpers
│   ├── data/
│   │   └── movielens_dataset.py  # SequentialDataset, EvalDataset, create_dataloaders
│   ├── features/
│   │   └── materialization.py  # generic + run_movielens_materialization()
│   ├── inference/
│   │   └── mamba_predictor.py  # MambaPredictor, in-memory cache
│   ├── models/
│   │   ├── trainer.py          # sklearn + mamba4rec dispatcher
│   │   ├── mamba4rec.py        # Mamba4Rec architecture
│   │   └── mamba_evaluator.py  # Hit/NDCG/MRR evaluation
│   ├── pipelines/
│   │   └── ingest_pipeline.py  # Kafka consumer → MinIO Parquet
│   └── training/
│       └── mamba_trainer.py    # run_mamba_training() orchestrator
├── monitoring/
│   ├── prometheus.yml
│   └── grafana/
│       ├── dashboards/         # place JSON dashboard files here
│       └── provisioning/       # auto-loaded by Grafana
├── tests/
└── docs/
```

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `GET /health` returns `redis: false` | Redis container not healthy | `docker compose restart redis` |
| `mamba_ssm` import error on Windows | CUDA unavailable | Normal — `SimplifiedMamba` (GRU) fallback is used automatically |
| `POST /train` fails with "bucket not found" | MinIO buckets not initialised | `docker compose up minio-init` |
| Feast `RegistryNotFoundException` | `feast apply` not run | `bash scripts/feast_apply.sh` |
| DataLoader hangs on Windows | `num_workers > 0` | Already fixed: `num_workers=0` is hard-coded in `create_dataloaders()` |
| `metadata.json` not found in predictor | Training run artifacts missing | Re-run training; check MLflow UI → run → Artifacts → `data/` |
| Kafka consumer stuck at startup | Topics not created yet | `bash scripts/create_kafka_topics.sh` |
| DuckDB `httpfs` S3 error | MinIO credentials wrong or service down | Check `.env` and `docker compose ps minio` |
| `POST /predict/mamba` slow on first call | Model loading from MLflow | Expected — model is cached in memory after first call |
