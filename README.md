# MLOpsLabs — Final Project

A full MLOps pipeline built for the **MSE AI in DevOps, DataOps & MLOps** course.  
Covers data ingestion, feature engineering, model training (sklearn + **Mamba4Rec**), serving, and observability — all running locally via Docker Compose.

---

## Architecture Overview

```
  MovieLens 1M Dataset
        │
        ▼
  simulate_ingestion.py  ──►  Kafka (new-user / new-movie / new-rating)
        │
        ▼
  IngestPipeline (consumer)
  → s3://raw-data/new_user/**, new_movie/**, new_rating/**   (Parquet)
        │
        ├──────────────────────────────────────────────────────┐
        │  Generic pipeline                                    │  Mamba4Rec pipeline
        │  (raw-events → Delta Lake → DuckDB → Feast → Redis) │  run_movielens_materialization()
        │                                                      │  → user/movie/rating_events.parquet
        │                                                      │  → metadata.json
        │                                                      │  Feast PIT join → sequences
        │                                                      │  Mamba4RecTrainer.fit()
        └──────────────────────────────────────────────────────┘
                                    │
                           MLflow Model Registry
                          "mamba4rec" (PyTorch)
                                    │
              ┌─────────────────────┼──────────────────────┐
              ▼                     ▼                      ▼
        POST /train          POST /predict          POST /predict/mamba
        POST /ingest         GET  /health           GET  /metrics

  Observability: Prometheus ──► Grafana
```

---

## Tech Stack

| Component | Technology |
|---|---|
| Data Ingestion | Apache Kafka 4.2 (KRaft) |
| Raw Storage | MinIO (S3-compatible) |
| Offline Store | MinIO + DuckDB + Delta Lake |
| Feature Store | Feast + DuckDB offline + Redis online |
| Model (generic) | scikit-learn |
| Model (Mamba4Rec) | PyTorch — State Space Model sequential recommender |
| Model Registry | MLflow 3.1 |
| Inference API | FastAPI + Uvicorn |
| Monitoring | Prometheus 3.3 + Grafana 12 |

---

## Quick Start

```bash
# 1. Clone
git clone <repo-url> && cd MLOpsLabs
cp .env.example .env

# 2. Start all services
docker compose up -d
docker compose ps   # wait until all services are healthy

# 3. Verify
curl http://localhost:8000/health
```

**Service URLs**

| Service | URL | Credentials |
|---|---|---|
| FastAPI docs | http://localhost:8000/docs | — |
| MLflow UI | http://localhost:5000 | — |
| MinIO Console | http://localhost:9001 | `minioadmin` / `minioadmin123` |
| Prometheus | http://localhost:9090 | — |
| Grafana | http://localhost:3000 | `admin` / `admin123` |

---

## Documentation

| Doc | Contents |
|---|---|
| [docs/data-pipeline.md](docs/data-pipeline.md) | Data flow, Kafka ingestion, MinIO buckets, DuckDB staging, Feast feature views, materialization |
| [docs/mamba4rec.md](docs/mamba4rec.md) | Mamba4Rec model architecture, training loop, evaluation metrics, WSL + venv run guide |
| [docs/api.md](docs/api.md) | Full API reference for all endpoints |
| [docs/monitoring.md](docs/monitoring.md) | Prometheus targets, Grafana dashboards |
| [docs/development.md](docs/development.md) | Dev setup, tests, linting, project structure, troubleshooting |

---

## Stopping the Stack

```bash
docker compose down       # stop (data preserved)
docker compose down -v    # stop + delete all volumes (full reset)
```
