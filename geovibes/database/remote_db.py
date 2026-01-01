"""Remote DuckDB connection via httpfs.

Provides transparent access to DuckDB databases on S3 using the httpfs extension.
"""

import logging
from typing import Any

import duckdb
import pandas as pd

logger = logging.getLogger(__name__)


class RemoteDuckDB:
    """Remote DuckDB connection via httpfs.

    Handles connection to DuckDB databases stored on S3, with proper
    error handling and connection lifecycle management.
    """

    def __init__(self):
        """Initialize without connecting."""
        self._conn: duckdb.DuckDBPyConnection | None = None
        self._url: str | None = None

    @property
    def is_connected(self) -> bool:
        """Check if connected to a database."""
        return self._conn is not None

    def connect(self, url: str) -> None:
        """Connect to a DuckDB database.

        Args:
            url: S3 URL (s3://...) or local file URL (file://...) to the database.

        Raises:
            FileNotFoundError: If database doesn't exist.
            PermissionError: If access is denied.
            TimeoutError: If connection times out.
        """
        self._url = url
        self._conn = duckdb.connect()

        try:
            # Load extensions
            self._conn.execute("INSTALL httpfs; LOAD httpfs;")
            self._conn.execute("INSTALL spatial; LOAD spatial;")
            self._conn.execute("INSTALL aws; LOAD aws;")

            # Handle different URL schemes
            if url.startswith("file://"):
                # Local file - just attach directly
                local_path = url[7:]  # Remove file://
                self._conn.execute(f"ATTACH '{local_path}' AS remote (READ_ONLY)")
            elif url.startswith("s3://"):
                # S3 URL - load AWS credentials
                self._conn.execute("CALL load_aws_credentials();")
                self._conn.execute(f"ATTACH '{url}' AS remote (READ_ONLY)")
            else:
                # Treat as local path
                self._conn.execute(f"ATTACH '{url}' AS remote (READ_ONLY)")

            self._conn.execute("USE remote")

            # Verify connection by getting table count
            self._conn.execute("SELECT 1").fetchone()

            # Warm up by loading table metadata (makes subsequent queries fast)
            self._warmup()

        except duckdb.IOException as e:
            self._handle_io_error(e, url)
        except Exception:
            self._conn = None
            raise

    def _warmup(self) -> None:
        """Warm up connection by loading table metadata.

        This ensures subsequent queries are fast (~1ms instead of ~2s).
        """
        try:
            # Touch the geo_embeddings table including geometry to load metadata
            self._conn.execute(
                "SELECT id, geometry FROM geo_embeddings LIMIT 1"
            ).fetchone()
            logger.debug("Connection warmed up")
        except Exception as e:
            logger.warning(f"Warmup query failed (non-fatal): {e}")

    def _handle_io_error(self, error: Exception, url: str):
        """Convert DuckDB IO errors to clear Python exceptions."""
        self._conn = None
        msg = str(error).lower()

        if "403" in msg or "forbidden" in msg:
            raise PermissionError(
                f"Access denied to {url}. Check AWS credentials and bucket permissions."
            )
        elif "404" in msg or "not found" in msg or "does not exist" in msg:
            raise FileNotFoundError(f"Database not found: {url}")
        elif "timeout" in msg:
            raise TimeoutError(f"Connection to {url} timed out")
        else:
            raise RuntimeError(f"Failed to connect to {url}: {error}")

    def query(self, sql: str, params: list[Any] = None) -> pd.DataFrame:
        """Execute query and return DataFrame.

        Args:
            sql: SQL query string.
            params: Optional list of parameters for parameterized queries.

        Returns:
            Query results as pandas DataFrame.
        """
        if not self._conn:
            raise RuntimeError("Not connected to database")

        if params:
            return self._conn.execute(sql, params).fetchdf()
        return self._conn.execute(sql).fetchdf()

    def fetch_by_ids(self, ids: list[int], columns: list[str] = None) -> pd.DataFrame:
        """Fetch rows by ID.

        Args:
            ids: List of row IDs to fetch.
            columns: Optional list of columns to select. Defaults to all.

        Returns:
            DataFrame with requested rows.
        """
        if not ids:
            return pd.DataFrame()

        # Convert numpy int types to Python int for DuckDB compatibility
        ids = [int(i) for i in ids]

        cols = ", ".join(columns) if columns else "*"
        placeholders = ", ".join(["?" for _ in ids])
        sql = f"SELECT {cols} FROM geo_embeddings WHERE id IN ({placeholders})"

        return self.query(sql, params=ids)

    def fetch_embeddings(self, ids: list[int]) -> pd.DataFrame:
        """Fetch embeddings by ID.

        Args:
            ids: List of row IDs to fetch.

        Returns:
            DataFrame with id and embedding columns.
        """
        return self.fetch_by_ids(ids, columns=["id", "embedding"])

    def close(self) -> None:
        """Close connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
            self._url = None

    def __enter__(self) -> "RemoteDuckDB":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()
