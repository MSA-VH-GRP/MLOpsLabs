"""
Idempotent setup script for the Feast training feature store.

What it does
------------
Step 1 — ``feast apply``
    Registers entities and feature views in ``registry.db``.
    Safe to re-run: Feast diffs the current definitions against the registry
    and only applies changes.

Step 2 — ``materialize_incremental``
    Pushes feature values from Delta Lake (MinIO) → Redis online store.
    Safe to re-run: Feast tracks the last-run timestamp per feature view in
    ``registry.db`` and only pushes rows newer than that mark.

Prerequisites
-------------
- MinIO is running and ``s3://processed/{train,val,test}`` Delta tables exist
  (run ``preprocess.py`` first).
- Redis is running on localhost:6379
  (``docker compose up -d`` from the project root).

Usage
-----
::

    # Full setup: registry apply + Redis materialization
    python featurestorage/trainning/apply.py

    # Registry only — skip Redis (Redis not running, or first-time registry setup)
    python featurestorage/trainning/apply.py --registry-only

    # Verbose DuckDB/Feast logs
    python featurestorage/trainning/apply.py --verbose
"""
from __future__ import annotations

import argparse
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

# Repo path is the directory this file lives in, so feature_store.yaml and
# registry.db are resolved correctly regardless of the working directory.
_REPO_PATH = str(Path(__file__).parent.resolve())


def _banner(text: str) -> None:
    width = 60
    print("─" * width)
    print(f"  {text}")
    print("─" * width)


def step_apply(store) -> None:
    """Step 1: Register entities and feature views in registry.db."""
    _banner("Step 1 — Feast registry apply")
    store.apply()
    print("  registry.db updated.\n")


def step_materialize(store) -> None:
    """Step 2: Push Delta Lake features → Redis online store."""
    _banner("Step 2 — Materialize incremental → Redis")
    print("  Reading from:  s3://processed/train  (Delta Lake / MinIO)")
    print("  Writing to:    localhost:6379         (Redis)")
    print()

    end_date = datetime.now(tz=timezone.utc)
    store.materialize_incremental(end_date=end_date)

    print(f"\n  Materialized up to: {end_date.isoformat()}")
    print("  Redis online store is ready for inference.\n")


def main(registry_only: bool = False, verbose: bool = False) -> int:
    """
    Run the apply pipeline.

    Returns 0 on success, 1 on any failure.
    """
    # Import here so import errors surface as clean messages, not tracebacks
    try:
        from featurestorage.trainning import TrainingFeatureStore
    except ImportError as exc:
        print(f"[ERROR] Could not import TrainingFeatureStore: {exc}")
        print("  Make sure 'feast' and 'duckdb' are installed:")
        print("  pip install -r requirements.txt")
        return 1

    print()
    print("  Feast Feature Store — apply")
    print(f"  repo_path : {_REPO_PATH}")
    print(f"  timestamp : {datetime.now(tz=timezone.utc).isoformat()}")
    print()

    with TrainingFeatureStore(repo_path=_REPO_PATH) as store:

        # ── Step 1: Registry apply ─────────────────────────────────────────
        try:
            step_apply(store)
        except Exception:
            print("[ERROR] Registry apply failed.")
            if verbose:
                traceback.print_exc()
            else:
                print("  Re-run with --verbose for the full traceback.")
            return 1

        # ── Step 2: Materialize to Redis ───────────────────────────────────
        if registry_only:
            _banner("Step 2 — skipped (--registry-only)")
            print("  Redis materialization was skipped.\n")
        else:
            try:
                step_materialize(store)
            except Exception:
                print("[ERROR] Materialization failed.")
                print("  Common causes:")
                print("    - Redis is not running  →  docker compose up -d redis")
                print("    - MinIO Delta tables are missing  →  python preprocess.py")
                if verbose:
                    traceback.print_exc()
                else:
                    print("  Re-run with --verbose for the full traceback.")
                return 1

    _banner("Done")
    if registry_only:
        print("  registry.db  ✓   Redis  (skipped)\n")
    else:
        print("  registry.db  ✓   Redis  ✓\n")

    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Register Feast features and materialize to Redis.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--registry-only",
        action="store_true",
        help="Only run 'feast apply' — skip Redis materialization.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print full tracebacks on failure.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    sys.exit(main(registry_only=args.registry_only, verbose=args.verbose))
