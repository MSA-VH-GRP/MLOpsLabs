# MLOpsLabs — Final Project

A full MLOps pipeline built for the **MSE AI in DevOps, DataOps & MLOps** course.  
Covers data ingestion, feature engineering, model training, serving, and observability — all running locally via Docker Compose.

---

## Architecture Overview

```
                        ┌─────────────────────────────────────────────┐
                        │              Data Ingestion                  │
  External Events ──►   │   POST /ingest  ──►  Kafka (raw-events)     │
                        └──────────────────────┬──────────────────────┘
                                               │ Kafka Consumer
                                               ▼
                        ┌─────────────────────────────────────────────┐
                        │           Offline Store (MinIO)              │
                        │  s3://offline-store/delta/   (Delta Lake)    │
                        │  s3://offline-store/parquet/ (Feast staging) │
                        └──────────────────────┬──────────────────────┘
                                               │ Feast Materialization
                                               ▼
                        ┌─────────────────────────────────────────────┐
                        │        Online Store (Redis)                  │
                        │  Feast feature vectors, inference cache      │
                        └──────────────────────┬──────────────────────┘
                                               │
               ┌───────────────────────────────┼──────────────────────┐
               │                               │                      │
               ▼                               ▼                      ▼
   ┌──────────────────┐          ┌─────────────────────┐   ┌─────────────────┐
   │  POST /train     │          │   POST /predict      │   │  GET /health    │
   │  MLflow autolog  │          │   Feast → MLflow     │   │  GET /metrics   │
   │  Model Registry  │          │   model inference    │   │  (Prometheus)   │
   └────────┬─────────┘          └─────────────────────┘   └─────────────────┘
            │
            ▼
   ┌──────────────────┐
   │  MLflow Registry │
   │  s3://mlflow-    │
   │  artifacts/      │
   └──────────────────┘

   Observability: Prometheus ──► Grafana
   Scraped: FastAPI · Redis · Kafka · MinIO
```

---

## Tech Stack

| Component | Technology | Purpose |
|---|---|---|
| **Data Ingestion** | Apache Kafka 4.2 (KRaft) | Event streaming into the pipeline |
| **Raw Storage** | MinIO (S3-compatible) | Raw data landing zone (`raw-data` bucket) |
| **Offline Store** | MinIO + Delta Lake | ACID feature tables (`offline-store` bucket) |
| **Feature Store** | Feast + DuckDB | Feature registry, serving definitions |
| **Online Store** | Redis 8 | Low-latency feature serving + cache |
| **Model Registry** | MLflow 3.1 | Experiment tracking, model versioning |
| **Inference API** | FastAPI + Uvicorn | `/ingest`, `/train`, `/predict`, `/health` |
| **Monitoring** | Prometheus 3.3 + Grafana 12 | Metrics collection and dashboards |

---

## Project Structure

```
MLOpsLabs/
├── docker-compose.yml          # All 9 services
├── pyproject.toml              # Python dependencies (PEP 621)
├── .env.example                # Environment variable template
│
├── src/
│   ├── api/                    # FastAPI application
│   │   ├── Dockerfile
│   │   ├── main.py             # App factory + lifespan
│   │   ├── dependencies.py     # DI: Feast store, MLflow client, Redis
│   │   ├── metrics.py          # Prometheus instrumentation
│   │   ├── routers/
│   │   │   ├── health.py       # GET  /health
│   │   │   ├── ingest.py       # POST /ingest
│   │   │   ├── train.py        # POST /train
│   │   │   └── predict.py      # POST /predict
│   │   └── schemas/            # Pydantic request/response models
│   │
│   ├── pipelines/
│   │   ├── consumer.py         # Base Kafka consumer
│   │   ├── ingest_pipeline.py  # Kafka → Delta Lake (MinIO)
│   │   └── topics.py           # Topic name constants
│   │
│   ├── features/
│   │   ├── transformations.py  # Pure feature transform functions
│   │   └── materialization.py  # Delta → Parquet → Feast → Redis
│   │
│   ├── models/
│   │   ├── trainer.py          # Train loop + MLflow autolog
│   │   ├── registry.py         # MLflow model registry helpers
│   │   └── evaluator.py        # Metrics + promotion threshold
│   │
│   └── core/
│       ├── config.py           # Pydantic settings (reads from env)
│       ├── storage.py          # MinIO/S3 client wrapper
│       ├── cache.py            # Redis client wrapper
│       └── kafka_producer.py   # Kafka producer wrapper
│
├── feast/
│   ├── feature_store.yaml      # DuckDB registry + Redis online store
│   ├── registry/               # DuckDB registry file (gitignored)
│   └── feature_views/
│       ├── raw_features.py     # Entity, FileSource, FeatureView
│       └── serving_features.py # FeatureService for /predict
│
├── monitoring/
│   ├── prometheus.yml          # Scrape config (5 targets)
│   └── grafana/
│       ├── provisioning/       # Auto-provision datasource + dashboards
│       └── dashboards/         # Drop custom .json dashboards here
│
├── notebooks/
│   └── 01_eda.ipynb            # Explore Delta tables from MinIO
│
├── tests/
│   ├── unit/                   # Feature transforms, schemas, evaluator
│   └── integration/            # API endpoints (requires live stack)
│
└── scripts/
    ├── create_kafka_topics.sh  # One-time Kafka topic creation
    └── feast_apply.sh          # Register Feast feature definitions
```

---

## Quick Start

### 1. Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) >= 4.x
- Git

### 2. Clone & configure

```bash
git clone https://github.com/MSA-VH-GRP/MLOpsLabs.git
cd MLOpsLabs
cp .env.example .env          # Edit credentials if needed
```

### 3. Start all services

```bash
docker compose up -d
```

First run pulls ~2 GB of images and builds the API image (~3-5 min).  
Subsequent starts take ~30 seconds.

### 4. Verify everything is healthy

```bash
docker compose ps
```

Expected output — all services `(healthy)` or `Up`:

```
mlops-api              Up (healthy)   0.0.0.0:8000->8000/tcp
mlops-grafana          Up (healthy)   0.0.0.0:3000->3000/tcp
mlops-kafka            Up (healthy)   0.0.0.0:9092->9092/tcp
mlops-kafka-exporter   Up (healthy)   0.0.0.0:9308->9308/tcp
mlops-minio            Up (healthy)   0.0.0.0:9000-9001->9000-9001/tcp
mlops-mlflow           Up (healthy)   0.0.0.0:5000->5000/tcp
mlops-prometheus       Up (healthy)   0.0.0.0:9090->9090/tcp
mlops-redis            Up (healthy)   0.0.0.0:6379->6379/tcp
mlops-redis-exporter   Up             0.0.0.0:9121->9121/tcp
```

```bash
curl http://localhost:8000/health
# {"status":"ok","checks":{"redis":true,"mlflow":true,"kafka":true,"minio":true}}
```

---

## Service URLs

| Service | URL | Credentials |
|---|---|---|
| **FastAPI** (docs) | http://localhost:8000/docs | — |
| **FastAPI** (health) | http://localhost:8000/health | — |
| **FastAPI** (metrics) | http://localhost:8000/metrics | — |
| **MLflow UI** | http://localhost:5000 | — |
| **MinIO Console** | http://localhost:9001 | `minioadmin` / `minioadmin123` |
| **Prometheus** | http://localhost:9090 | — |
| **Grafana** | http://localhost:3000 | `admin` / `admin123` |
| **Redis** | `localhost:6379` | — |
| **Kafka** | `localhost:9092` | — |

---

## API Endpoints

### `GET /health`
Checks connectivity to all downstream services.
```json
{"status": "ok", "checks": {"redis": true, "mlflow": true, "kafka": true, "minio": true}}
```

### `POST /ingest`
Publishes events to the Kafka `raw-events` topic.
```json
// Request
{
  "events": [
    {"id": "evt-001", "timestamp": "2026-04-16T09:00:00", "payload": {"x": 1.5, "y": 2.3}}
  ]
}

// Response
{"accepted": 1, "topic": "raw-events"}
```

### `POST /train`
Triggers model training, logs to MLflow, and registers the model.
```json
// Request
{
  "experiment_name": "default",
  "model_type": "random_forest",
  "hyperparams": {"n_estimators": 100, "max_depth": 5},
  "feature_view": "raw_event_features"
}

// Response
{"run_id": "abc123...", "model_version": 1, "status": "registered"}
```

### `POST /predict`
Fetches features from Redis (Feast), runs inference with an MLflow model.
```json
// Request
{
  "entity_ids": ["evt-001", "evt-002"],
  "feature_service": "inference_features",
  "model_name": "random_forest",
  "model_alias": "champion"
}

// Response
{
  "predictions": [{"entity_id": "evt-001", "prediction": 1}],
  "model_version": "champion",
  "latency_ms": 12.4
}
```

---

## Data Flow

```
1. POST /ingest  →  Kafka (raw-events topic)
2. IngestPipeline (consumer)  →  Delta Lake on MinIO (offline-store/delta/)
3. materialization.py  →  Delta → Parquet (offline-store/parquet/)
4. feast materialize  →  Parquet → Redis (online store)
5. POST /predict  →  Redis features + MLflow model → prediction
```

### MinIO Buckets

| Bucket | Contents |
|---|---|
| `raw-data` | Raw uploaded files / objects |
| `offline-store` | Delta Lake tables + Parquet staging for Feast |
| `mlflow-artifacts` | MLflow models, plots, metrics |

---

## Feast Feature Store

Feature definitions live in `feast/feature_views/`.

Apply changes to the registry:
```bash
# From project root (with Python env active)
bash scripts/feast_apply.sh
# or directly:
feast -c feast/ apply
```

Run materialization (Delta → Parquet → Redis):
```bash
python -m src.features.materialization
```

---

## Development Setup

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install all dependencies (including dev tools)
pip install -e ".[dev]"

# Run unit tests (no live services needed)
pytest tests/unit/ -v

# Run integration tests (requires docker compose up -d)
pytest tests/integration/ -v -m integration
```

---

## Monitoring

### Prometheus Targets
All 5 targets should show **UP** at http://localhost:9090/targets:

| Job | Endpoint | What it scrapes |
|---|---|---|
| `fastapi` | `api:8000/metrics` | Request latency, throughput, errors |
| `redis` | `redis-exporter:9121` | Memory, hit rate, connected clients |
| `kafka` | `kafka-exporter:9308` | Consumer lag, topic message rates |
| `minio` | `minio:9000/minio/v2/metrics/cluster` | Storage usage, request rates |
| `prometheus` | `localhost:9090` | Self-monitoring |

### Grafana Dashboards
Log in at http://localhost:3000 (`admin` / `admin123`).  
The Prometheus datasource is auto-provisioned.

To add community dashboards:
1. Go to **Dashboards → Import**
2. Use these Grafana.com IDs:
   - Redis: `11835`
   - Kafka: `7589`
   - Node Exporter: `1860`

---

## Stopping the Stack

```bash
docker compose down          # Stop containers (data preserved)
docker compose down -v       # Stop + delete all volumes (data loss!)
```

---

## Troubleshooting

| Issue | Fix |
|---|---|
| `mlops-api` won't start | Check `docker logs mlops-api` — usually a missing dep or import error |
| Kafka health check fails | Ensure `KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://kafka:9092` (internal hostname) |
| MLflow health check fails | Uses Python urllib, not curl — ensure `ghcr.io/mlflow/mlflow:v3.1.0` image |
| MinIO buckets missing | Re-run `docker compose up minio-init` to recreate buckets |
| Prometheus target DOWN | Check the exporter container is running: `docker compose ps` |
