# Data Pipeline

End-to-end data flow from raw MovieLens events to Feast-ready features.

---

## Overview

```
simulate_ingestion.py  (reads MovieLens 1M, streams to Kafka)
        │
        │  Topics: new-user / new-movie / new-rating
        ▼
IngestPipeline  (Kafka consumer)
        │
        ▼  Parquet, date-partitioned
┌──────────────────────────────────────────────────┐
│                  MinIO: raw-data                  │
│  s3://raw-data/new_user/year=.../month=.../...    │
│  s3://raw-data/new_movie/...                      │
│  s3://raw-data/new_rating/...                     │
└──────────────────────────┬───────────────────────┘
                           │
             ┌─────────────┴──────────────────────┐
             │                                    │
             ▼                                    ▼
   Generic pipeline                   Mamba4Rec pipeline
   (raw-events → Delta Lake)          run_movielens_materialization()
   → DuckDB SQL transforms            → DuckDB encodes + re-indexes
   → staged.parquet                   → user_features.parquet
   → Feast raw_event_features         → movie_features.parquet
   → Redis online store               → rating_events.parquet
                                      → metadata.json
                                             │
                                      feast apply
                                             │
                                      Feast PIT join
                                      get_historical_features()
                                             │
                                      Training sequences
```

---

## Simulator

`scripts/simulate_ingestion.py` reads the MovieLens 1M dataset and publishes all events to Kafka.

```bash
# Fast playback (finishes in minutes)
python scripts/simulate_ingestion.py --speedup 10000

# Skip specific event types
python scripts/simulate_ingestion.py --skip-users --skip-movies
```

**Kafka topics and message schemas:**

| Topic | Key fields |
|---|---|
| `new-user` | `user_id`, `gender`, `age`, `occupation`, `zip_code`, `timestamp` |
| `new-movie` | `movie_id`, `title`, `genres` (list of strings), `timestamp` |
| `new-rating` | `user_id`, `movie_id`, `rating`, `rating_timestamp`, `timestamp` |

---

## Kafka Ingestion Pipeline

`src/pipelines/ingest_pipeline.py` consumes all three topics and flushes batches to MinIO as Parquet.

**Output paths:**
```
s3://raw-data/new_user/year={y}/month={m}/day={d}/part-{uuid}.parquet
s3://raw-data/new_movie/year={y}/month={m}/day={d}/part-{uuid}.parquet
s3://raw-data/new_rating/year={y}/month={m}/day={d}/part-{uuid}.parquet
```

Run the consumer:
```bash
python -m src.pipelines.ingest_pipeline
```

---

## MinIO Buckets

| Bucket | Contents |
|---|---|
| `raw-data` | Raw Parquet files from Kafka ingestion |
| `offline-store` | Staged Parquet for Feast (`parquet/`), Delta Lake (`delta/`), metadata JSON |
| `mlflow-artifacts` | MLflow model files, metrics, plots |

Access the MinIO console at **http://localhost:9001** (`minioadmin` / `minioadmin123`).

---

## Mamba4Rec Materialization

`src/features/materialization.run_movielens_materialization()` reads the three raw-data Parquets and stages Feast-ready files on MinIO. It is called automatically at the start of every `POST /train` with `model_type="mamba4rec"` and is **idempotent** — re-running overwrites existing staged files.

### Stage 1 — User features

Reads `s3://raw-data/new_user/**/*.parquet`.

Encodes:
- `gender`: `F → 0`, `M → 1`
- `age`: maps ML-1M age groups to 0–6

```
Output: s3://offline-store/parquet/user_features/staged.parquet
Columns: user_id, event_timestamp, gender_idx, age_idx, occupation
```

### Stage 2 — Movie features

Reads `s3://raw-data/new_movie/**/*.parquet`.

Transforms:
- Assigns `internal_movie_id` (1-based, sorted by original `movie_id`)
- Encodes up to 3 genres per movie from the fixed 18-genre ML-1M mapping (1–18, 0 = padding)

```
Output: s3://offline-store/parquet/movie_features/staged.parquet
Columns: movie_id, internal_movie_id, event_timestamp, genre_idx_0, genre_idx_1, genre_idx_2
```

### Stage 3 — Rating events

Reads `s3://raw-data/new_rating/**/*.parquet`, joins with the movie re-index map.

Transforms:
- Joins `movie_id → internal_movie_id`
- Filters ratings < 3 (only positive interactions)
- Computes `time_slot` from Unix `rating_timestamp`:
  - `0` = Matinee (06:00–17:59)
  - `1` = Prime Time (18:00–21:59)
  - `2` = Late Night (22:00–05:59)

```
Output: s3://offline-store/parquet/rating_events/staged.parquet
Columns: user_id, event_timestamp, internal_movie_id, rating, time_slot
```

### Stage 4 — Metadata JSON

```
Output: s3://offline-store/parquet/metadata.json
```

Contains:
```json
{
  "num_items":           3707,
  "num_genres":          19,
  "num_ages":            7,
  "num_genders":         2,
  "num_occupations":     21,
  "num_time_slots":      3,
  "max_genres_per_item": 3,
  "movie_id_map":        {"1": 1, "2": 2, ...},
  "reverse_movie_map":   {"1": 1, "2": 2, ...},
  "genre_to_idx":        {"Action": 1, "Adventure": 2, ...}
}
```

---

## Feast Feature Store

Configuration: `feast/feature_store.yaml`
- Offline store: **DuckDB** (queries Parquet on MinIO via httpfs)
- Online store: **Redis** (low-latency serving)
- Registry: **DuckDB file** (`feast/registry/registry.db`)

### Feature views

#### Generic pipeline

| Name | Entity | Features |
|---|---|---|
| `raw_event_features` | `event_id` | `feature_1`, `feature_2`, `category`, `count` |

**Feature service** `inference_features` bundles `raw_event_features` for `POST /predict`.

#### Mamba4Rec pipeline (`feast/feature_views/movielens_features.py`)

| Name | Entity | Features | TTL |
|---|---|---|---|
| `user_profile_features` | `user_id` | `gender_idx`, `age_idx`, `occupation` | 10 years |
| `movie_genre_features` | `movie_id` | `internal_movie_id`, `genre_idx_0/1/2` | 10 years |
| `rating_event_features` | `user_id` | `internal_movie_id`, `rating`, `time_slot` | 10 years |

### Register feature definitions

```bash
bash scripts/feast_apply.sh
# or:
feast -c feast/ apply
```

### Point-in-Time join for Mamba4Rec training

```python
entity_df = pd.DataFrame(
    columns=["user_id", "event_timestamp"]
)  # one row per (user_id, event_timestamp) from rating_events

feature_df = store.get_historical_features(
    entity_df=entity_df,
    features=[
        "rating_event_features:internal_movie_id",  # value AT that timestamp
        "rating_event_features:rating",
        "rating_event_features:time_slot",
        "user_profile_features:gender_idx",          # latest value ≤ timestamp
        "user_profile_features:age_idx",
        "user_profile_features:occupation",
    ],
).to_df()
```

### Generic materialization (Delta → Feast → Redis)

```bash
python -m src.features.materialization
```

---

## DuckDB Client

`src/core/duckdb_client.py` provides a configured DuckDB connection with MinIO S3 credentials pre-loaded.

```python
from src.core.duckdb_client import get_duckdb_connection

conn = get_duckdb_connection()

# Ad-hoc query on any staged Parquet
df = conn.execute("""
    SELECT * FROM read_parquet('s3://offline-store/parquet/rating_events/staged.parquet')
    LIMIT 10
""").df()

conn.close()
```

Key helpers:

| Function | Description |
|---|---|
| `get_duckdb_connection()` | In-memory DuckDB with S3/MinIO pre-configured |
| `register_delta_as_table(conn, name, path)` | Load a Delta Lake table into DuckDB via PyArrow |
| `stage_to_parquet(conn, query, s3_path)` | Execute SQL and write result to MinIO as Parquet |
