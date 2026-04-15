# MLOpsLabs

MLOps pipeline for MovieLens 1M — preprocessing, feature store, and training data retrieval.

## Architecture

```
Raw data (MinIO)
    └── preprocess.py ──▶ Delta Lake tables (MinIO: s3://processed/)
                                └── featurestorage/ ──▶ training data (main.ipynb)
```

## Setup

### 1. Prerequisites

- Python 3.12
- Docker

### 2. Start infrastructure

```bash
docker compose up -d
```

This starts:
- **MinIO** — object storage at `http://localhost:9000` (console: `http://localhost:9001`)
- **Redis** — online feature store at `localhost:6379`
- **Kafka** — message broker at `localhost:9092`

### 3. Upload raw data to MinIO

Open `http://localhost:9001/browser/raw-data` and upload the 3 MovieLens 1M files:

| File | Contents |
|------|----------|
| `ml-1m.inter` | Ratings (user_id, movie_id, rating, timestamp) |
| `ml-1m.item`  | Movies (movie_id, title, genres) |
| `ml-1m.user`  | Users (user_id, gender, age, occupation) |

MinIO credentials: user `minioadmin` / password `minioadmin123`

### 4. Create Python environment

```bash
python3.12 -m venv .venv
source .venv/bin/activate          # macOS / Linux
# .venv\Scripts\activate           # Windows
pip install -r requirements.txt
```

### 5. Run preprocessing

Reads raw files from MinIO, writes Delta Lake tables to `s3://processed/`.

```bash
python preprocess.py
```

Expected output:
```
train:  818,363 rows → s3://processed/train
val  :    6,038 rows → s3://processed/val
test :    6,038 rows → s3://processed/test
```

### 6. Set up the feature store

Registers entities and feature views in `registry.db` (Feast metadata).

```bash
python featurestorage/trainning/apply.py --registry-only
```

### 7. Run the notebook

Open `main.ipynb` in VS Code or JupyterLab and run all cells.

The notebook loads train/val/test datasets from the feature store and verifies
they match the expected Mamba4Rec format.
