"""Tests for remote database access via httpfs.

These tests verify:
1. FaissCache downloads and caches FAISS indexes from S3
2. RemoteDuckDB connects to DuckDB on S3 and executes queries
3. Integration with existing DataManager

Run with: pytest tests/test_remote_db.py -v
"""

import os

import duckdb
import faiss
import numpy as np
import pytest

# These imports will fail until we implement the modules
from geovibes.database.faiss_cache import FaissCache
from geovibes.database.remote_db import RemoteDuckDB

# Mark tests that require network access
pytestmark = pytest.mark.skipif(
    os.environ.get("SKIP_NETWORK_TESTS", "0") == "1",
    reason="Network tests disabled",
)

# Test S3 URLs (uploaded during Phase 0)
S3_DB_URL = "s3://us-west-2.opendata.source.coop/geovibes/search/USA/alabama/2024-01-01-2025-01-01/alabama_google_satellite_embeddings_v1_2024_2025_25_0_10_metadata.db"


class TestFaissCache:
    """Tests for FAISS index caching."""

    @pytest.fixture
    def cache_dir(self, tmp_path):
        """Create temporary cache directory."""
        return tmp_path / "faiss_cache"

    @pytest.fixture
    def sample_faiss_index(self, tmp_path):
        """Create a small FAISS index for testing."""
        index = faiss.IndexFlatL2(64)
        vectors = np.random.rand(100, 64).astype(np.float32)
        index.add(vectors)
        path = tmp_path / "test.faiss"
        faiss.write_index(index, str(path))
        return path

    def test_cache_creates_directory(self, cache_dir):
        """Cache should create its directory if it doesn't exist."""
        assert not cache_dir.exists()
        _ = FaissCache(cache_dir=cache_dir)
        assert cache_dir.exists()

    def test_cache_key_is_deterministic(self, cache_dir):
        """Same URL should always produce same cache key."""
        cache = FaissCache(cache_dir=cache_dir)

        url1 = "s3://bucket/path/index.faiss"
        url2 = "s3://bucket/path/index.faiss"
        url3 = "s3://bucket/other/index.faiss"

        key1 = cache._get_cache_key(url1)
        key2 = cache._get_cache_key(url2)
        key3 = cache._get_cache_key(url3)

        assert key1 == key2, "Same URL should produce same key"
        assert key1 != key3, "Different URLs should produce different keys"

    def test_loads_from_cache_if_exists(self, cache_dir, sample_faiss_index):
        """Should load from cache without network access if file exists."""
        cache = FaissCache(cache_dir=cache_dir)

        # Pre-populate cache manually
        url = "s3://bucket/test.faiss"
        cache_key = cache._get_cache_key(url)
        cache_path = cache_dir / f"{cache_key}.faiss"
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        # Copy the sample index to cache location
        import shutil

        shutil.copy(sample_faiss_index, cache_path)

        # Should load from cache without network
        index = cache.get_index(url)

        assert index is not None
        assert index.ntotal == 100
        assert index.d == 64

    def test_validates_loaded_index(self, cache_dir):
        """Should detect and handle corrupted cache files."""
        cache = FaissCache(cache_dir=cache_dir)

        # Create corrupted cache file
        url = "s3://bucket/test.faiss"
        cache_key = cache._get_cache_key(url)
        cache_path = cache_dir / f"{cache_key}.faiss"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(b"corrupted data - not a valid FAISS index")

        # Should raise an error or re-download
        with pytest.raises((RuntimeError, ValueError, IOError)):
            cache.get_index(url)

    def test_cache_path_property(self, cache_dir):
        """Should expose cache directory path."""
        cache = FaissCache(cache_dir=cache_dir)
        assert cache.cache_dir == cache_dir


class TestRemoteDuckDB:
    """Tests for remote DuckDB connection via httpfs."""

    def test_connects_to_s3_database(self):
        """Should connect to DuckDB on S3 and query."""
        db = RemoteDuckDB()
        db.connect(S3_DB_URL)

        # Should be able to count rows
        result = db.query("SELECT COUNT(*) as cnt FROM geo_embeddings")
        assert result is not None
        assert len(result) == 1
        assert result["cnt"].iloc[0] > 0

        db.close()

    def test_spatial_queries_work(self):
        """Spatial queries should work over httpfs."""
        db = RemoteDuckDB()
        db.connect(S3_DB_URL)

        result = db.query("""
            SELECT id, ST_AsText(geometry) as wkt
            FROM geo_embeddings
            ORDER BY ST_Distance(geometry, ST_Point(-86.5, 32.5))
            LIMIT 1
        """)

        assert result is not None
        assert len(result) == 1
        assert "POINT" in result["wkt"].iloc[0]

        db.close()

    def test_fetch_by_id(self):
        """Should fetch rows by ID efficiently."""
        db = RemoteDuckDB()
        db.connect(S3_DB_URL)

        # Get some IDs first
        ids_df = db.query("SELECT id FROM geo_embeddings LIMIT 10")
        ids = ids_df["id"].tolist()

        # Fetch by ID
        result = db.fetch_by_ids(ids, columns=["id", "geometry"])

        assert result is not None
        assert len(result) == 10

        db.close()

    def test_fetch_embeddings_by_id(self):
        """Should fetch embeddings by ID."""
        db = RemoteDuckDB()
        db.connect(S3_DB_URL)

        # Get some IDs
        ids_df = db.query("SELECT id FROM geo_embeddings LIMIT 5")
        ids = ids_df["id"].tolist()

        # Fetch embeddings
        result = db.fetch_embeddings(ids)

        assert result is not None
        assert len(result) == 5
        # Should have id and embedding columns
        assert "id" in result.columns
        assert "embedding" in result.columns

        db.close()

    def test_handles_not_found_error(self):
        """Should raise FileNotFoundError for missing database."""
        db = RemoteDuckDB()

        with pytest.raises(FileNotFoundError):
            db.connect("s3://us-west-2.opendata.source.coop/nonexistent/path.db")

    def test_context_manager(self):
        """Should work as context manager."""
        with RemoteDuckDB() as db:
            db.connect(S3_DB_URL)
            result = db.query("SELECT COUNT(*) FROM geo_embeddings")
            assert result is not None

    def test_is_connected_property(self):
        """Should track connection state."""
        db = RemoteDuckDB()
        assert not db.is_connected

        db.connect(S3_DB_URL)
        assert db.is_connected

        db.close()
        assert not db.is_connected


class TestRemoteDuckDBLocal:
    """Tests that don't require network - use local DuckDB."""

    @pytest.fixture
    def local_db_path(self, tmp_path):
        """Create a local DuckDB with geo_embeddings table."""
        db_path = tmp_path / "test.db"
        conn = duckdb.connect(str(db_path))
        conn.execute("INSTALL spatial; LOAD spatial;")
        conn.execute("""
            CREATE TABLE geo_embeddings (
                id BIGINT PRIMARY KEY,
                tile_id VARCHAR,
                embedding FLOAT[64],
                geometry GEOMETRY
            )
        """)
        # Insert test data
        for i in range(10):
            emb = np.random.rand(64).astype(np.float32).tolist()
            conn.execute(
                """
                INSERT INTO geo_embeddings (id, tile_id, embedding, geometry)
                VALUES (?, ?, ?, ST_Point(?, ?))
                """,
                [i, f"tile_{i}", emb, -86.5 + i * 0.01, 32.5 + i * 0.01],
            )
        conn.close()
        return db_path

    def test_query_returns_dataframe(self, local_db_path):
        """Query should return pandas DataFrame."""
        db = RemoteDuckDB()
        # Connect to local file (RemoteDuckDB should handle file:// or local paths)
        db.connect(f"file://{local_db_path}")

        result = db.query("SELECT * FROM geo_embeddings LIMIT 5")

        import pandas as pd

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 5

        db.close()

    def test_parameterized_query(self, local_db_path):
        """Should support parameterized queries."""
        db = RemoteDuckDB()
        db.connect(f"file://{local_db_path}")

        result = db.query(
            "SELECT * FROM geo_embeddings WHERE id = ?",
            params=[5],
        )

        assert len(result) == 1
        assert result["id"].iloc[0] == 5

        db.close()


class TestIntegration:
    """Integration tests for the full remote workflow."""

    def test_faiss_local_duckdb_remote_workflow(self):
        """
        Test the hybrid workflow:
        1. Load FAISS index (would be from cache in real use)
        2. Search FAISS for similar vectors
        3. Fetch metadata from remote DuckDB by ID
        """
        # Skip if no network
        db = RemoteDuckDB()
        db.connect(S3_DB_URL)

        # Get a sample ID to fetch
        sample = db.query("""
            SELECT id
            FROM geo_embeddings
            LIMIT 1
        """)

        # In real use, we'd search FAISS locally
        # Here we just verify we can fetch by ID from remote
        result = db.fetch_by_ids([sample["id"].iloc[0]], columns=["id", "geometry"])

        assert len(result) == 1

        db.close()
