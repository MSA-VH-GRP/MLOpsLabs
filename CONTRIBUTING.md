# Contributing

## Team

| GitHub handle | Name | Responsibilities |
|---|---|---|
| **KennyDizi** | Đinh Chí Trung | Docker Compose infrastructure, CI/CD, testing, Feast feature store, DuckDB materialisation |
| **ktran2strongtie** | Trần khả | Kafka ingestion pipeline, MinIO / Delta Lake storage |
| **longhoang0305** | Hoàng Đại Thiên Long | API layer, Mamba4Rec training pipeline, MLflow integration |
| **Ken** | Huỳnh Minh Dũng | Monitoring (Prometheus, Grafana dashboards) |

---

## Branching Strategy

```
main
 ├── feature/<short-description>   # new features
 ├── fix/<short-description>        # bug fixes
 └── chore/<short-description>      # infra, docs, tooling
```

- Branch off `main`.
- Open a Pull Request targeting `main`.
- At least one team member must review before merging.
- Delete the branch after merging.

---

## Development Setup

**Prerequisites:** Python 3.11+, Docker, Docker Compose, [uv](https://github.com/astral-sh/uv)

```bash
# 1. Clone and enter the repo
git clone <repo-url>
cd mlops-finalproject

# 2. Copy environment variables
cp .env.example .env

# 3. Install Python dependencies (editable + dev extras)
uv pip install -e ".[dev]"

# 4. Start all infrastructure services
docker compose up -d

# 5. Verify the API is healthy
curl http://localhost:8000/health
```

---

## Running Tests

```bash
# All tests
pytest

# Unit tests only
pytest tests/unit

# Integration tests only (requires running Docker services)
pytest tests/integration

# With verbose output
pytest -s -v
```

---

## Code Style

This project uses **ruff** for linting/formatting and **mypy** for type checking.

```bash
# Check and auto-fix linting issues
ruff check . --fix

# Type checking
mypy src/

# Both in one go
ruff check . --fix && mypy src/
```

Rules: `E, F, I, UP` — line length 100, target Python 3.11.

CI will fail if either tool reports errors.

---

## Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add Mamba4Rec evaluation endpoint
fix: handle missing Redis key in MambaPredictor
chore: update docker-compose kafka image to 4.2.0
docs: add Feast materialisation flow to ARCHITECTURE.md
test: add integration test for /predict/mamba
```

---

## Pull Request Checklist

Before requesting review:

- [ ] Tests pass locally (`pytest`)
- [ ] No linting errors (`ruff check .`)
- [ ] No type errors (`mypy src/`)
- [ ] `docker compose up` starts cleanly
- [ ] PR description explains **what** changed and **why**
- [ ] Relevant docs updated (README, ARCHITECTURE, docs/)

---

## Project Structure Quick Reference

```
src/api/        → FastAPI routers & schemas
src/core/       → Config, service clients (Kafka, MinIO, Redis, DuckDB)
src/models/     → Mamba4Rec architecture, sklearn trainer, MLflow registry
src/features/   → Feast materialisation
src/pipelines/  → Kafka consumer → Parquet writer
src/training/   → Mamba training orchestrator
tests/          → pytest unit + integration tests
monitoring/     → Prometheus config, Grafana dashboards
feast/          → Feast feature store definitions
scripts/        → Ingestion simulation, workflow automation
docs/           → Detailed documentation per subsystem
```
