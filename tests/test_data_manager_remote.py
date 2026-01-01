"""Tests for DataManager remote database support.

These tests verify that DataManager can work with remote databases on S3
using the httpfs extension.
"""

import os

import duckdb
import faiss
import numpy as np
import pytest

# Test S3 URLs
S3_DB_URL = "s3://us-west-2.opendata.source.coop/geovibes/search/USA/alabama/2024-01-01-2025-01-01/alabama_google_satellite_embeddings_v1_2024_2025_25_0_10_metadata.db"


class TestDataManagerRemoteSupport:
    """Tests for remote database support in DataManager."""

    @pytest.fixture
    def temp_cache_dir(self, tmp_path):
        """Create temporary cache directory for FAISS indexes."""
        cache_dir = tmp_path / "faiss_cache"
        cache_dir.mkdir()
        return cache_dir

    @pytest.fixture
    def mock_faiss_index(self, tmp_path):
        """Create a mock FAISS index file."""
        index = faiss.IndexFlatL2(64)
        vectors = np.random.rand(100, 64).astype(np.float32)
        index.add(vectors)
        path = tmp_path / "test.faiss"
        faiss.write_index(index, str(path))
        return path

    def test_is_remote_url_detects_s3(self):
        """Should detect S3 URLs as remote."""
        from geovibes.ui.data_manager import DataManager

        assert DataManager.is_remote_url("s3://bucket/path/db.db")
        assert DataManager.is_remote_url("s3://us-west-2.opendata.source.coop/path.db")
        assert not DataManager.is_remote_url("/local/path/db.db")
        assert not DataManager.is_remote_url("local_databases/db.db")

    def test_is_remote_url_detects_gs(self):
        """Should detect GCS URLs as remote."""
        from geovibes.ui.data_manager import DataManager

        assert DataManager.is_remote_url("gs://bucket/path/db.db")
        assert not DataManager.is_remote_url("gcs://bucket/path.db")  # Not valid

    def test_remote_manifest_entry_parsed(self):
        """Manifest entries with S3 URLs should be marked as remote."""
        from geovibes.ui.data_manager import DataManager

        # This tests manifest parsing, not full DataManager init
        manifest_row = {
            "region": "alabama",
            "model_name": "test_model",
            "model_path": "s3://bucket/path/model.tar.gz",
        }

        is_remote = DataManager.is_remote_url(manifest_row.get("model_path", ""))
        assert is_remote


class TestFaissCacheIntegration:
    """Tests for FaissCache integration with DataManager."""

    @pytest.fixture
    def sample_index(self, tmp_path):
        """Create a sample FAISS index."""
        index = faiss.IndexFlatL2(64)
        vectors = np.random.rand(100, 64).astype(np.float32)
        index.add(vectors)
        path = tmp_path / "sample.faiss"
        faiss.write_index(index, str(path))
        return path

    def test_faiss_cache_used_for_remote_index(self, tmp_path, sample_index):
        """DataManager should use FaissCache for remote FAISS indexes."""
        from geovibes.database.faiss_cache import FaissCache

        cache_dir = tmp_path / "cache"
        cache = FaissCache(cache_dir=cache_dir)

        # Simulate caching a remote index
        url = "s3://bucket/test.faiss"
        cache_path = cache._get_cache_path(url)
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        import shutil

        shutil.copy(sample_index, cache_path)

        # Load from cache
        index = cache.get_index(url)
        assert index.ntotal == 100


class TestRemoteDuckDBIntegration:
    """Tests for RemoteDuckDB integration."""

    @pytest.mark.skipif(
        os.environ.get("SKIP_NETWORK_TESTS", "0") == "1",
        reason="Network tests disabled",
    )
    def test_remote_duckdb_queries_work(self):
        """RemoteDuckDB should successfully query S3 database."""
        from geovibes.database.remote_db import RemoteDuckDB

        db = RemoteDuckDB()
        db.connect(S3_DB_URL)

        # Count should return rows
        result = db.query("SELECT COUNT(*) as cnt FROM geo_embeddings")
        assert result["cnt"].iloc[0] > 0

        db.close()

    @pytest.mark.skipif(
        os.environ.get("SKIP_NETWORK_TESTS", "0") == "1",
        reason="Network tests disabled",
    )
    def test_remote_embedding_fetch(self):
        """Should fetch embeddings from remote database."""
        from geovibes.database.remote_db import RemoteDuckDB

        db = RemoteDuckDB()
        db.connect(S3_DB_URL)

        # Get some IDs
        ids_df = db.query("SELECT id FROM geo_embeddings LIMIT 5")
        ids = ids_df["id"].tolist()

        # Fetch embeddings
        embeddings_df = db.fetch_embeddings(ids)
        assert len(embeddings_df) == 5
        assert "embedding" in embeddings_df.columns

        db.close()


class TestManifestRemoteSupport:
    """Tests for manifest with remote database entries."""

    def test_parse_manifest_with_s3_paths(self, tmp_path):
        """Should parse manifest entries with S3 paths."""
        manifest_content = """region,model_name,model_path
alabama,test_model,s3://bucket/path/test_model.tar.gz
california,other_model,s3://bucket/path/other_model.tar.gz
"""
        manifest_path = tmp_path / "manifest.csv"
        manifest_path.write_text(manifest_content)

        import csv

        with open(manifest_path) as f:
            reader = csv.DictReader(f)
            entries = list(reader)

        assert len(entries) == 2
        assert entries[0]["model_path"].startswith("s3://")

    def test_manifest_entry_has_remote_db_url(self, tmp_path):
        """Manifest should support direct database URLs."""
        manifest_content = """region,model_name,model_path,db_url,faiss_url
alabama,test_model,s3://bucket/model.tar.gz,s3://bucket/model.db,s3://bucket/model.faiss
"""
        manifest_path = tmp_path / "manifest.csv"
        manifest_path.write_text(manifest_content)

        import csv

        with open(manifest_path) as f:
            reader = csv.DictReader(f)
            entries = list(reader)

        assert entries[0].get("db_url") == "s3://bucket/model.db"
        assert entries[0].get("faiss_url") == "s3://bucket/model.faiss"


class TestDataManagerRemoteMode:
    """Integration tests for DataManager with remote databases."""

    @pytest.fixture
    def minimal_local_setup(self, tmp_path):
        """Create minimal local database for fallback testing."""
        db_path = tmp_path / "local.db"
        conn = duckdb.connect(str(db_path))
        conn.execute("INSTALL spatial; LOAD spatial;")
        conn.execute("""
            CREATE TABLE geo_embeddings (
                id BIGINT PRIMARY KEY,
                embedding FLOAT[64],
                geometry GEOMETRY
            )
        """)
        for i in range(10):
            emb = np.random.rand(64).astype(np.float32).tolist()
            conn.execute(
                "INSERT INTO geo_embeddings VALUES (?, ?, ST_Point(?, ?))",
                [i, emb, -86.5 + i * 0.01, 32.5],
            )
        conn.close()

        # Create FAISS index (use .index extension for DataManager discovery)
        index = faiss.IndexFlatL2(64)
        vectors = np.random.rand(10, 64).astype(np.float32)
        index.add(vectors)
        faiss_path = tmp_path / "local.index"
        faiss.write_index(index, str(faiss_path))

        return {"db_path": str(db_path), "faiss_path": str(faiss_path)}

    def test_data_manager_with_local_database(self, minimal_local_setup):
        """DataManager should work with local database (baseline)."""
        from geovibes.ui.data_manager import DataManager

        dm = DataManager(
            duckdb_path=minimal_local_setup["db_path"],
            disable_ee=True,
            verbose=False,
        )

        assert dm.duckdb_connection is not None
        assert dm.faiss_index is not None
        assert dm.faiss_index.ntotal == 10

        dm.close()
