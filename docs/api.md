# API Reference

Base URL: `http://localhost:8000`  
Interactive docs: http://localhost:8000/docs

---

## GET /health

Returns connectivity status for all downstream services.

**Response `200`**
```json
{
  "status": "ok",
  "checks": {
    "redis": true,
    "mlflow": true,
    "kafka": true,
    "minio": true
  }
}
```

---

## GET /metrics

Prometheus metrics endpoint scraped by the monitoring stack.

---

## POST /ingest

Publish events to the Kafka `raw-events` topic.

**Request**
```json
{
  "events": [
    {
      "id": "evt-001",
      "timestamp": "2026-04-16T09:00:00",
      "payload": {"x": 1.5, "y": 2.3}
    }
  ]
}
```

**Response `200`**
```json
{"accepted": 1, "topic": "raw-events"}
```

---

## POST /train

Trigger model training and register the result in MLflow.

Dispatches based on `model_type`:
- `"random_forest"` / `"logistic_regression"` → sklearn path (reads from Feast offline store)
- `"mamba4rec"` → Mamba4Rec path (materializes MovieLens features, PIT join, sequential training)

### sklearn

**Request**
```json
{
  "experiment_name": "baseline",
  "model_type": "random_forest",
  "hyperparams": {
    "n_estimators": 100,
    "max_depth": 5
  },
  "feature_view": "raw_event_features"
}
```

### Mamba4Rec

**Request**
```json
{
  "experiment_name": "mamba4rec-v1",
  "model_type": "mamba4rec",
  "hyperparams": {
    "d_model": 64,
    "d_state": 16,
    "n_layers": 2,
    "d_conv": 4,
    "expand": 2,
    "dropout": 0.1,
    "max_seq_len": 50,
    "batch_size": 256,
    "learning_rate": 0.001,
    "weight_decay": 0.0001,
    "epochs": 50,
    "patience": 10
  }
}
```

All `hyperparams` fields are optional — omitted fields use their defaults (see [mamba4rec.md](mamba4rec.md)).

**Response `200`** (both model types)
```json
{
  "run_id": "abc123def456...",
  "model_version": 1,
  "status": "registered"
}
```

---

## POST /predict

Generic sklearn inference. Fetches features from the Feast Redis online store, runs inference with an MLflow-registered model.

**Request**
```json
{
  "entity_ids": ["evt-001", "evt-002"],
  "feature_service": "inference_features",
  "model_name": "random_forest",
  "model_alias": "champion"
}
```

| Field | Default | Description |
|---|---|---|
| `entity_ids` | required | List of `event_id` values to fetch features for |
| `feature_service` | `"inference_features"` | Feast FeatureService name |
| `model_name` | required | Registered model name in MLflow |
| `model_alias` | `"champion"` | MLflow model alias |

**Response `200`**
```json
{
  "predictions": [
    {"entity_id": "evt-001", "prediction": 1},
    {"entity_id": "evt-002", "prediction": 0}
  ],
  "model_version": "champion",
  "latency_ms": 12.4
}
```

---

## POST /predict/mamba

Mamba4Rec sequential movie recommendation. The predictor is loaded from MLflow on first call and **cached in memory** — subsequent calls are fast.

**Request**
```json
{
  "model_name": "mamba4rec",
  "model_alias": "champion",
  "item_history": [1, 5, 10, 20, 50],
  "time_history": [1, 0, 2, 1, 1],
  "age_idx": 2,
  "gender_idx": 1,
  "occupation": 4,
  "top_k": 5,
  "target_time": 1
}
```

**Field reference**

| Field | Type | Default | Description |
|---|---|---|---|
| `model_name` | string | `"mamba4rec"` | MLflow registered model name |
| `model_alias` | string | `"champion"` | MLflow model alias |
| `item_history` | `List[int]` | required | Ordered watch history as **internal movie IDs** (1-based) |
| `time_history` | `List[int]` | required | Time slot for each item in history |
| `age_idx` | int | `0` | 0=<18, 1=18-24, 2=25-34, 3=35-44, 4=45-49, 5=50-55, 6=56+ |
| `gender_idx` | int | `1` | 0=Female, 1=Male |
| `occupation` | int | `0` | ML-1M occupation code (0–20) |
| `top_k` | int | `10` | Number of recommendations to return |
| `target_time` | int | `1` | Session time slot (used for response label) |

**Time slot values**

| Value | Label | Hours |
|---|---|---|
| `0` | Matinee | 06:00–17:59 |
| `1` | Prime Time | 18:00–21:59 |
| `2` | Late Night | 22:00–05:59 |

**Response `200`**
```json
{
  "recommendations": [
    {
      "rank": 1,
      "movie_id": 593,
      "title": "Silence of the Lambs, The (1991)",
      "genres": "Crime|Thriller",
      "score": 12.34,
      "time_slot": "Prime Time (18:00-21:59)"
    },
    {
      "rank": 2,
      "movie_id": 318,
      "title": "Shawshank Redemption, The (1994)",
      "genres": "Drama",
      "score": 11.89,
      "time_slot": "Prime Time (18:00-21:59)"
    }
  ],
  "model_version": "champion",
  "latency_ms": 45.2
}
```

> **Internal IDs note:** `item_history` uses internal IDs assigned during materialization, not original MovieLens IDs. The mapping is in `metadata.json` (logged as a MLflow artifact on every training run). See [mamba4rec.md — Internal IDs](mamba4rec.md#internal-ids) for details.
