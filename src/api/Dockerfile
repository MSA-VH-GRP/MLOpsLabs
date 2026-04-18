FROM python:3.11-slim

WORKDIR /app

# System deps for confluent-kafka and pyarrow
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    librdkafka-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
# Install CPU-only torch first from the PyTorch index (avoids downloading 2GB CUDA wheels)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -e .

COPY src/ ./src/
COPY feast/ ./feast/

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
