"""
DuckDB connection factory configured for MinIO S3 access.

DuckDB acts as the staging engine between Delta Lake (MinIO) and the
Feast offline store. It uses the httpfs extension to read/write Parquet
files directly on MinIO using the S3 protocol.

Usage:
    conn = get_duckdb_connection()
    conn.execute("SELECT * FROM read_parquet('s3://offline-store/parquet/raw_events/**/*.parquet')")
"""

import re

import duckdb

from src.core.config import settings

_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,127}$")


def _validate_identifier(name: str) -> None:
    if not _IDENT_RE.match(name):
        raise ValueError(f"Unsafe SQL identifier: {name!r}")


def get_duckdb_connection() -> duckdb.DuckDBPyConnection:
    """
    Return an in-memory DuckDB connection pre-configured for MinIO S3 access.

    The httpfs extension is installed and loaded, and all S3 settings are
    pointed at the local MinIO instance (path-style URLs, no SSL).
    """
    conn = duckdb.connect(database=":memory:")
    _configure_s3(conn)
    return conn


def get_duckdb_persistent(db_path: str) -> duckdb.DuckDBPyConnection:
    """
    Return a persistent DuckDB connection backed by a local .db file.
    Useful for caching staged data between materialization runs.
    """
    conn = duckdb.connect(database=db_path)
    _configure_s3(conn)
    return conn


def _configure_s3(conn: duckdb.DuckDBPyConnection) -> None:
    """Install httpfs and configure S3 settings for MinIO."""
    # Strip protocol prefix — DuckDB expects host:port only
    endpoint = (
        settings.minio_endpoint
        .replace("http://", "")
        .replace("https://", "")
    )

    conn.execute("INSTALL httpfs; LOAD httpfs;")
    conn.execute(f"""
        SET s3_endpoint        = '{endpoint}';
        SET s3_access_key_id   = '{settings.aws_access_key_id}';
        SET s3_secret_access_key = '{settings.aws_secret_access_key}';
        SET s3_use_ssl         = false;
        SET s3_url_style       = 'path';
    """)


def register_delta_as_table(
    conn: duckdb.DuckDBPyConnection,
    table_name: str,
    delta_path: str,
    storage_options: dict,
) -> int:
    """
    Read a Delta Lake table into DuckDB as a named in-memory table.

    Uses deltalake → PyArrow as the bridge (DuckDB's native delta extension
    requires DuckDB >= 1.1 and is optional). Returns the row count.

    Args:
        conn:            DuckDB connection (from get_duckdb_connection()).
        table_name:      Name to register the table as inside DuckDB.
        delta_path:      S3 path to the Delta table root (e.g. "s3://...").
        storage_options: boto3-compatible dict with endpoint_url, credentials.

    Returns:
        Number of rows loaded.
    """
    from deltalake import DeltaTable

    _validate_identifier(table_name)
    dt = DeltaTable(delta_path, storage_options=storage_options)
    arrow_table = dt.to_pyarrow_table()
    conn.register(table_name, arrow_table)
    row_count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
    return row_count


def stage_to_parquet(
    conn: duckdb.DuckDBPyConnection,
    query: str,
    parquet_s3_path: str,
) -> None:
    """
    Run a DuckDB SQL query and write the result as Parquet to MinIO.

    Uses DuckDB's native COPY TO statement with the httpfs extension.

    Args:
        conn:            DuckDB connection with S3 already configured.
        query:           SQL SELECT that produces the staging data.
        parquet_s3_path: Destination path on MinIO, e.g.
                         "s3://offline-store/parquet/raw_events/data.parquet"
    """
    conn.execute(f"""
        COPY (
            {query}
        ) TO '{parquet_s3_path}'
        (FORMAT PARQUET, OVERWRITE_OR_IGNORE TRUE)
    """)
