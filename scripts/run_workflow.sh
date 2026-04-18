#!/usr/bin/env bash
# Full pre-training workflow script
# Run from project root with venv activated:
#   source .venv-wsl/bin/activate
#   bash scripts/run_workflow.sh

set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "============================================"
echo " MLOps Pre-Training Workflow"
echo "============================================"

# ── Step 1: Fix Docker hostnames in /etc/hosts ──
echo ""
echo "[1/4] Fixing Docker hostnames..."
if ! grep -q "kafka" /etc/hosts; then
    echo "127.0.0.1 kafka redis minio mlflow" >> /etc/hosts
    echo "  ✓ Added Docker hostnames to /etc/hosts"
else
    echo "  ✓ Docker hostnames already set"
fi

# ── Step 2: Ingest pipeline + Simulator ─────────
echo ""
echo "[2/4] Starting ingest pipeline (background)..."
export KAFKA_BOOTSTRAP_SERVERS=kafka:9092
nohup python3 -m src.pipelines.ingest_pipeline > /tmp/ingest_pipeline.log 2>&1 &
INGEST_PID=$!
echo "  ✓ IngestPipeline running (PID=$INGEST_PID)"

echo ""
echo "  Running MovieLens simulator (speedup=10000x)..."
echo "  This will ingest 6040 users, 3883 movies, 1M ratings..."
KAFKA_BOOTSTRAP_SERVERS=kafka:9092 python3 scripts/simulate_ingestion.py --speedup 10000
echo "  ✓ Simulator done"

echo ""
echo "  Waiting 5s for ingest pipeline to flush remaining messages..."
sleep 5
kill $INGEST_PID 2>/dev/null && echo "  ✓ IngestPipeline stopped" || echo "  IngestPipeline already stopped"

echo ""
echo "  IngestPipeline log (last 10 lines):"
tail -10 /tmp/ingest_pipeline.log || true

# ── Step 3: Materialization ──────────────────────
echo ""
echo "[3/4] Running MovieLens materialization..."
export AWS_ACCESS_KEY_ID=minioadmin
export AWS_SECRET_ACCESS_KEY=minioadmin123
export AWS_ENDPOINT_URL=http://localhost:9000
export AWS_DEFAULT_REGION=us-east-1
python3 -c "
import logging
logging.basicConfig(level=logging.INFO, format='  %(levelname)s %(message)s')
from src.features.materialization import run_movielens_materialization
meta = run_movielens_materialization()
print(f'  ✓ Done: {meta[\"num_users\"]} users, {meta[\"num_items\"]} items')
"

# ── Step 4: Feast apply (after Parquet files exist) ──
echo ""
echo "[4/4] Running feast apply..."
export AWS_ACCESS_KEY_ID=minioadmin
export AWS_SECRET_ACCESS_KEY=minioadmin123
export AWS_ENDPOINT_URL=http://localhost:9000
export AWS_DEFAULT_REGION=us-east-1

# Create placeholder for raw_events (generic pipeline) so feast can read schema
python3 -c "
import io, boto3, pandas as pd
from botocore.config import Config

s3 = boto3.client(
    's3',
    endpoint_url='http://localhost:9000',
    aws_access_key_id='minioadmin',
    aws_secret_access_key='minioadmin123',
    config=Config(signature_version='s3v4'),
    region_name='us-east-1',
)
df = pd.DataFrame({
    'event_id':        pd.Series([], dtype='str'),
    'event_timestamp': pd.Series([], dtype='datetime64[us, UTC]'),
    'feature_1':       pd.Series([], dtype='float32'),
    'feature_2':       pd.Series([], dtype='float32'),
    'category':        pd.Series([], dtype='str'),
    'count':           pd.Series([], dtype='int64'),
})
buf = io.BytesIO()
df.to_parquet(buf, engine='pyarrow', index=False)
buf.seek(0)
s3.put_object(Bucket='offline-store', Key='parquet/raw_events/staged.parquet', Body=buf.getvalue())
print('  ✓ Placeholder raw_events/staged.parquet created')
"

python3 -c "
import sys
sys.argv = ['feast', '-c', 'feast/', 'apply']
from feast.cli.cli import cli
cli()
"
echo "  ✓ Feast feature definitions registered"

echo ""
echo "============================================"
echo " Pre-training complete! Ready to train."
echo "============================================"
echo ""
echo "Run training:"
echo "  curl -X POST http://localhost:8000/train \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"experiment_name\": \"mamba4rec\", \"model_type\": \"mamba4rec\", \"hyperparams\": {\"d_model\": 64, \"n_layers\": 2, \"epochs\": 5}}'"
