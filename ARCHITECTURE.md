# Architecture

## Overview

MLOpsLabs is a full-stack MLOps reference implementation built around the **MovieLens 1M** dataset. It demonstrates streaming ingestion, feature engineering, dual-model training, real-time inference, and observability — all wired together with Docker Compose.

---

## System Diagram

```
┌──────────────────────────────────────────────────────────────────────┐
│                         Data Ingestion Layer                         │
│                                                                      │
│   scripts/simulate_ingestion.py                                      │
│          │                                                           │
│          ▼                                                           │
│   ┌─────────────┐   topics: raw-events, new-user,                   │
│   │    Kafka    │            new-movie, new-rating                   │
│   └──────┬──────┘                                                   │
│          │                                                           │
│          ▼                                                           │
│   ┌─────────────┐   Parquet / Delta Lake                            │
│   │    MinIO    │◄── consumer service (ingest_pipeline.py)          │
│   └──────┬──────┘                                                   │
└──────────┼───────────────────────────────────────────────────────────┘
           │
┌──────────┼───────────────────────────────────────────────────────────┐
│          │              Feature Store Layer                          │
│          ▼                                                           │
│   ┌─────────────┐   DuckDB reads Parquet from MinIO via httpfs      │
│   │   DuckDB    │                                                    │
│   └──────┬──────┘                                                   │
│          │   Feast PIT join + materialization                        │
│          ▼                                                           │
│   ┌─────────────┐   online serving (key-value)                      │
│   │    Redis    │                                                    │
│   └──────┬──────┘                                                   │
└──────────┼───────────────────────────────────────────────────────────┘
           │
┌──────────┼───────────────────────────────────────────────────────────┐
│          │              Model Training Layer                         │
│          ▼                                                           │
│   ┌──────────────────────────────────┐                              │
│   │          Model Trainer           │                              │
│   │  ┌────────────┐  ┌───────────┐  │                              │
│   │  │  sklearn   │  │ Mamba4Rec │  │                              │
│   │  │ (generic)  │  │ (PyTorch) │  │                              │
│   │  └────────────┘  └───────────┘  │                              │
│   └──────────────┬───────────────────┘                              │
│                  │   log metrics + register model                    │
│                  ▼                                                   │
│          ┌─────────────┐                                            │
│          │   MLflow    │                                            │
│          └─────────────┘                                            │
└──────────────────────────────────────────────────────────────────────┘
           │
┌──────────┼───────────────────────────────────────────────────────────┐
│          │               Inference Layer                             │
│          ▼                                                           │
│   ┌──────────────────────────────────┐                              │
│   │          FastAPI (api)           │                              │
│   │  POST /predict        (sklearn)  │                              │
│   │  POST /predict/mamba  (Mamba4Rec)│                              │
│   │  POST /train          (trigger)  │                              │
│   │  POST /ingest         (trigger)  │                              │
│   │  GET  /health                    │                              │
│   └──────────────────────────────────┘                              │
└──────────────────────────────────────────────────────────────────────┘
           │
┌──────────┼───────────────────────────────────────────────────────────┐
│          │             Observability Layer                           │
│          ▼                                                           │
│   ┌──────────────┐   scrapes /metrics                               │
│   │  Prometheus  │◄──── api, redis-exporter, kafka-exporter, minio  │
│   └──────┬───────┘                                                  │
│          ▼                                                           │
│   ┌─────────────┐                                                   │
│   │   Grafana   │                                                   │
│   └─────────────┘                                                   │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Services

| Service | Image | Port | Role |
|---|---|---|---|
| `kafka` | apache/kafka:4.2.0 | 9092 | Message broker (KRaft, no Zookeeper) |
| `minio` | minio/minio | 9000 / 9001 | S3-compatible raw storage (Parquet / Delta Lake) |
| `redis` | redis:8.6.2 | 6379 | Online feature store + prediction cache |
| `mlflow` | ghcr.io/mlflow/mlflow:v3.1.0 | 5001 | Experiment tracking & model registry |
| `prometheus` | prom/prometheus:v3.3.1 | 9090 | Metrics collection |
| `grafana` | grafana/grafana:12.0.1 | 3000 | Dashboards |
| `redis-exporter` | oliver006/redis_exporter:v1.74.0 | 9121 | Redis → Prometheus bridge |
| `kafka-exporter` | danielqsj/kafka-exporter:v1.9.0 | 9308 | Kafka → Prometheus bridge |
| `consumer` | custom (src/api/Dockerfile) | — | Kafka consumer → MinIO ingest pipeline |
| `api` | custom (src/api/Dockerfile) | 8000 | FastAPI inference server |

---

## Source Layout

```
src/
├── api/            # FastAPI routers, schemas, metrics, Dockerfile
├── core/           # Config (Pydantic), DuckDB client, Kafka producer, MinIO, Redis
├── data/           # SequentialDataset, EvalDataset (MovieLens loader)
├── features/       # Feast materialization, feature transformations
├── inference/      # MambaPredictor with in-memory model cache
├── models/         # Mamba4Rec architecture, sklearn trainer, MLflow registry
├── pipelines/      # Kafka consumer → Parquet writer
├── preprocessing/  # Data cleaning & normalisation
└── training/       # End-to-end Mamba training orchestrator
```

---

## Models

### Generic Recommender (scikit-learn)
- Trained on materialised Feast features
- Registered to MLflow as `generic-recommender`
- Served via `POST /predict`

### Mamba4Rec (PyTorch)
- State Space Model (SSM) for sequential recommendation
- Falls back to a GRU-based `SimplifiedMamba` when `mamba-ssm` is not installed (CPU environments)
- Trained via `src/training/mamba_trainer.py`: Feast PIT join → item sequences → SSM training loop
- Evaluated with Hit@K, NDCG@K, MRR
- Registered to MLflow as `mamba4rec`
- Served via `POST /predict/mamba`

---

## Data Flow (end-to-end)

1. `simulate_ingestion.py` streams MovieLens 1M rows to Kafka topics.
2. The `consumer` service reads events and writes Parquet files to MinIO (`s3://raw-data/`).
3. `POST /train` triggers Feast materialisation (DuckDB reads Parquet → Redis online store), then trains the chosen model and registers it in MLflow.
4. `POST /predict` / `POST /predict/mamba` fetches online features from Redis, runs inference, and returns ranked recommendations.
5. Prometheus scrapes the `/metrics` endpoint; Grafana visualises latency, throughput, drift, and infrastructure health.

---

## Configuration

All runtime configuration is loaded from environment variables via `src/core/config.py` (Pydantic `BaseSettings`). Copy `.env.example` to `.env` before starting.

Key variables:

| Variable | Default | Description |
|---|---|---|
| `KAFKA_BOOTSTRAP_SERVERS` | `localhost:9092` | Kafka broker address |
| `MINIO_ROOT_USER` | `minioadmin` | MinIO access key |
| `MINIO_ROOT_PASSWORD` | `minioadmin123` | MinIO secret key |
| `REDIS_URL` | `redis://localhost:6379` | Redis connection string |
| `MLFLOW_TRACKING_URI` | `http://localhost:5000` | MLflow server |
| `LOG_LEVEL` | `info` | Application log level |

---

## Service URLs (local)

| Service | URL |
|---|---|
| FastAPI docs | http://localhost:8000/docs |
| MLflow UI | http://localhost:5001 |
| MinIO console | http://localhost:9001 |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 |
