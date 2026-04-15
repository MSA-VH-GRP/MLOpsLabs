"""
DuckDB query engine for reading Delta Lake tables from MinIO.

Responsibilities
----------------
- Install and load the ``httpfs`` extension for S3-compatible access.
- Install and load the ``delta`` extension to decode the Delta Lake format.
- Pre-configure all S3/MinIO credentials so every query can use
  ``delta_scan('s3://...')`` without extra boilerplate.
- Expose a small, stable query API used by TrainingFeatureStore.

This class is the *only* place in the training pipeline that talks to MinIO.
"""
from __future__ import annotations

import os
from typing import Optional

import duckdb
import pandas as pd

from .config import DUCKDB_CONFIG, MINIO_CONFIG, DuckDBConfig, MinIOConfig


class DuckDBDeltaEngine:
    """
    DuckDB engine configured to read Delta Lake tables stored on MinIO.

    The connection is initialised lazily on first use and kept open for the
    lifetime of the engine instance (connection pooling is not needed here
    because DuckDB is an in-process engine).

    Example
    -------
    ::

        with DuckDBDeltaEngine() as engine:
            # Read a whole split
            df = engine.read_delta("train")

            # Ad-hoc SQL with delta_scan() inline
            df = engine.query(
                f"SELECT user_id, age_idx "
                f"FROM {engine.scan('train')} "
                f"WHERE user_id < 100"
            )
    """

    def __init__(
        self,
        minio: Optional[MinIOConfig] = None,
        duckdb_cfg: Optional[DuckDBConfig] = None,
    ) -> None:
        self._minio = minio or MINIO_CONFIG
        self._cfg = duckdb_cfg or DUCKDB_CONFIG
        self._conn: Optional[duckdb.DuckDBPyConnection] = None

    # ── Connection ────────────────────────────────────────────────────────────

    @property
    def conn(self) -> duckdb.DuckDBPyConnection:
        """Lazily-initialised DuckDB connection with all extensions loaded."""
        if self._conn is None:
            self._conn = self._init_connection()
        return self._conn

    def _init_connection(self) -> duckdb.DuckDBPyConnection:
        # ── delta-rs environment credentials ──────────────────────────────────
        # The DuckDB `delta` extension uses the delta-rs library, which has its
        # own credential chain (env vars → credential file → EC2 IMDS).
        # Setting these env vars prevents the 10-retry timeout against the AWS
        # instance metadata endpoint (169.254.169.254) and ensures MinIO is used.
        os.environ["AWS_ACCESS_KEY_ID"]          = self._minio.access_key
        os.environ["AWS_SECRET_ACCESS_KEY"]      = self._minio.secret_key
        os.environ["AWS_ENDPOINT_URL"]           = self._minio.endpoint
        os.environ["AWS_DEFAULT_REGION"]         = self._minio.region
        os.environ["AWS_ALLOW_HTTP"]             = "true"
        os.environ["AWS_EC2_METADATA_DISABLED"]  = "true"

        conn = duckdb.connect(self._cfg.db_path)

        # ── Performance tuning ────────────────────────────────────────────────
        conn.execute(f"SET threads        = {self._cfg.threads}")
        conn.execute(f"SET memory_limit   = '{self._cfg.memory_limit}'")

        # ── Extensions ────────────────────────────────────────────────────────
        # httpfs : enables S3-compatible object storage access
        # delta  : decodes the Delta Lake transaction log and Parquet data files
        for ext in ("httpfs", "delta"):
            conn.install_extension(ext)
            conn.load_extension(ext)

        # ── MinIO S3 credentials ──────────────────────────────────────────────
        # DuckDB 1.0+ SECRET store is the correct mechanism for the delta
        # extension.  The legacy SET s3_* commands only affect httpfs, not
        # the delta-rs credential chain used by delta_scan().
        use_ssl = "true" if self._minio.use_ssl else "false"
        conn.execute(f"""
            CREATE OR REPLACE SECRET _minio (
                TYPE     S3,
                KEY_ID   '{self._minio.access_key}',
                SECRET   '{self._minio.secret_key}',
                ENDPOINT '{self._minio.endpoint_host}',
                URL_STYLE 'path',
                USE_SSL  {use_ssl},
                REGION   '{self._minio.region}'
            )
        """)

        return conn

    # ── Query API ─────────────────────────────────────────────────────────────

    def query(self, sql: str) -> pd.DataFrame:
        """Execute arbitrary SQL and return the result as a DataFrame."""
        return self.conn.execute(sql).df()

    def read_delta(self, table: str) -> pd.DataFrame:
        """Read an entire Delta Lake table from the processed bucket."""
        return self.query(f"SELECT * FROM {self.scan(table)}")

    def scan(self, table: str) -> str:
        """
        Return the ``delta_scan()`` SQL fragment for *table*.

        Useful for embedding inside larger SQL strings::

            sql = f"SELECT user_id FROM {engine.scan('train')} WHERE ..."
        """
        return f"delta_scan('{self._minio.delta_uri(table)}')"

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def close(self) -> None:
        """Close the DuckDB connection and release resources."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> "DuckDBDeltaEngine":
        return self

    def __exit__(self, *_) -> None:
        self.close()
