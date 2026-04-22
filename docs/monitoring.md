# Monitoring

Prometheus scrapes metrics from five targets; Grafana visualises them.

---

## Service URLs

| Service | URL | Credentials |
|---|---|---|
| Prometheus | http://localhost:9090 | — |
| Grafana | http://localhost:3000 | `admin` / `admin123` |

---

## Prometheus Scrape Targets

`monitoring/prometheus.yml` defines five scrape jobs (15 s interval, 10 s timeout):

| Job | Target | What it collects |
|---|---|---|
| `fastapi` | `api:8000/metrics` | Request counts, latency histograms, in-flight requests (via `prometheus-fastapi-instrumentator`) |
| `redis` | `redis-exporter:9121` | Connected clients, memory usage, command rate, hit/miss ratios |
| `kafka` | `kafka-exporter:9308` | Topic offsets, consumer group lag, partition counts |
| `minio` | `minio:9000/minio/v2/metrics/cluster` | Bucket sizes, object counts, S3 request errors |
| `prometheus` | `localhost:9090` | Prometheus self-monitoring (scrape durations, TSDB stats) |

Verify all targets are **UP** at http://localhost:9090/targets.

---

## Grafana Data Source

Prometheus is auto-provisioned as the default data source via `monitoring/grafana/provisioning/datasources/prometheus.yaml`. No manual setup required after `docker compose up`.

---

## Importing Community Dashboards

In Grafana: **Dashboards → Import → Enter Grafana ID**

| Dashboard | ID | Use |
|---|---|---|
| Redis | `11835` | Memory, throughput, hit rate |
| Kafka | `7589` | Topic lag, broker throughput |
| Node Exporter Full | `1860` | Host CPU, memory, disk (requires node-exporter) |

---

## Key Metrics Reference

### FastAPI

| Metric | Description |
|---|---|
| `http_requests_total` | Total requests by method, path, status code |
| `http_request_duration_seconds` | Latency histogram |
| `http_requests_inprogress` | Currently active requests |

### Redis

| Metric | Description |
|---|---|
| `redis_connected_clients` | Active connections |
| `redis_memory_used_bytes` | Memory consumption |
| `redis_keyspace_hits_total` | Cache hit count |
| `redis_keyspace_misses_total` | Cache miss count |

### Kafka

| Metric | Description |
|---|---|
| `kafka_consumergroup_lag` | Consumer group offset lag per partition |
| `kafka_topic_partition_current_offset` | Latest offset per topic/partition |

---

## Custom Dashboards

Place JSON dashboard files in `monitoring/grafana/dashboards/`. They are auto-loaded by the provisioning config at `monitoring/grafana/provisioning/dashboards/dashboards.yaml`.

---

## Prometheus Storage

Metrics are retained for **15 days** (`--storage.tsdb.retention.time=15d`). Stored in the `prometheus_data` Docker volume.

To reload the Prometheus config without restarting:
```bash
curl -X POST http://localhost:9090/-/reload
```
