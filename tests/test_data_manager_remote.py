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


class TestPrefetchGrouping:
    """Tests for row-group-aware prefetch batching."""

    def test_group_ids_for_prefetch_falls_back_to_id_div(self):
        """Should group IDs with id//row_group when no row-group cache is loaded."""
        from geovibes.ui.data_manager import DataManager

        dm = DataManager.__new__(DataManager)
        dm._row_group_cache_connection = None
        dm._row_group_size = 100

        batches, mode, lookup_ms = dm.group_ids_for_prefetch([1, 2, 150, 199, 205])

        assert mode == "id_div"
        assert lookup_ms == 0.0
        normalized = sorted(sorted(batch) for batch in batches)
        assert normalized == [[1, 2], [150, 199], [205]]

    def test_group_ids_for_prefetch_uses_row_group_cache(self):
        """Should use cached physical row-group mapping when available."""
        from geovibes.ui.data_manager import DataManager

        dm = DataManager.__new__(DataManager)
        dm._row_group_size = 100
        dm._row_group_cache_connection = duckdb.connect(":memory:")
        dm._row_group_cache_connection.execute(
            "CREATE TABLE row_group_cache (id BIGINT, row_group BIGINT)"
        )
        dm._row_group_cache_connection.executemany(
            "INSERT INTO row_group_cache VALUES (?, ?)",
            [(1, 4), (2, 4), (205, 9)],
        )

        batches, mode, lookup_ms = dm.group_ids_for_prefetch([205, 1, 2, 350])
        dm._row_group_cache_connection.close()

        assert mode == "row_group_cache"
        assert lookup_ms >= 0.0
        normalized = sorted(sorted(batch) for batch in batches)
        # 350 is absent from cache and should fall back to id//100 => group 3.
        assert normalized == [[1, 2], [205], [350]]

    def test_group_ids_for_prefetch_applies_id_ascending_scheduler(self):
        """Batches should be returned in ascending ID order by default scheduler."""
        from geovibes.ui.data_manager import DataManager

        dm = DataManager.__new__(DataManager)
        dm._row_group_cache_connection = None
        dm._row_group_size = 100
        dm._prefetch_batch_scheduler = "id_ascending"

        batches, mode, _ = dm.group_ids_for_prefetch([205, 150, 2, 1])

        assert mode == "id_div"
        assert batches == [[1, 2], [150], [205]]


class TestQueryModes:
    """Tests for metadata and prefetch query mode toggles."""

    def test_query_search_metadata_values_join_uses_geometry_cache(self):
        from geovibes.ui.data_manager import DataManager

        dm = DataManager.__new__(DataManager)
        dm.external_id_column = "id"
        dm._metadata_query_mode = "values_join"
        dm._geometry_cache_connection = duckdb.connect(":memory:")
        dm._geometry_cache_connection.execute("INSTALL spatial; LOAD spatial;")
        dm._geometry_cache_connection.execute(
            """
            CREATE TABLE geometry_cache AS
            SELECT
                1::BIGINT AS id,
                ST_GeomFromText('POINT(-86.50 32.50)') AS geometry
            UNION ALL
            SELECT
                3::BIGINT AS id,
                ST_GeomFromText('POINT(-86.30 32.60)') AS geometry
            """
        )

        df = dm.query_search_metadata([3, 1])
        dm._geometry_cache_connection.close()

        assert set(df["id"].astype(int).tolist()) == {1, 3}
        assert "geometry_json" in df.columns
        assert "geometry_wkt" in df.columns

    def test_fetch_embedding_map_with_connection_values_join(self):
        from geovibes.ui.data_manager import DataManager

        conn = duckdb.connect(":memory:")
        conn.execute(
            """
            CREATE TABLE geo_embeddings (
                id BIGINT PRIMARY KEY,
                embedding FLOAT[4]
            )
            """
        )
        conn.execute("INSERT INTO geo_embeddings VALUES (1, [1.0, 2.0, 3.0, 4.0])")
        conn.execute("INSERT INTO geo_embeddings VALUES (3, [3.0, 4.0, 5.0, 6.0])")

        dm = DataManager.__new__(DataManager)
        dm._prefetch_query_mode = "values_join"

        out = dm.fetch_embedding_map_with_connection(conn, ["3", "1"])
        conn.close()

        assert set(out.keys()) == {"1", "3"}
        assert out["1"].shape == (4,)
        assert out["3"].dtype == np.float32

    def test_resolve_embedding_select_expr_prefers_native_fixed_array(self, monkeypatch):
        from geovibes.ui.data_manager import DataManager

        dm = DataManager.__new__(DataManager)
        dm.embedding_type = "FLOAT[384]"
        monkeypatch.delenv("GEOVIBES_EMBEDDING_SELECT_EXPR", raising=False)

        expr = dm._resolve_embedding_select_expr()

        assert expr == "embedding"

    def test_resolve_embedding_select_expr_honors_env_override(self, monkeypatch):
        from geovibes.ui.data_manager import DataManager

        dm = DataManager.__new__(DataManager)
        dm.embedding_type = "FLOAT[384]"
        monkeypatch.setenv("GEOVIBES_EMBEDDING_SELECT_EXPR", "CAST(embedding AS FLOAT[])")

        expr = dm._resolve_embedding_select_expr()

        assert expr == "CAST(embedding AS FLOAT[])"


class TestBackgroundConnectionPool:
    """Tests for persistent background connection pooling."""

    class _DummyConn:
        def __init__(self, name):
            self.name = name
            self.closed = False

        def execute(self, _query):
            return self

        def close(self):
            self.closed = True

    def test_background_pool_reuses_connections(self):
        from geovibes.ui.data_manager import DataManager

        dm = DataManager.__new__(DataManager)
        dm.verbose = False
        dm.current_database_path = "s3://bucket/path/metadata.db"
        created = []

        def fake_connect(_path):
            conn = self._DummyConn(f"conn-{len(created)}")
            created.append(conn)
            return conn

        dm._connect_duckdb = fake_connect

        dm.configure_background_connection_pool(2)
        c1 = dm.acquire_background_connection()
        c2 = dm.acquire_background_connection()
        assert len(created) == 2

        dm.release_background_connection(c1)
        dm.release_background_connection(c2)

        stats = dm.background_pool_stats()
        assert stats["size"] == 2
        assert stats["idle"] == 2

        c3 = dm.acquire_background_connection()
        dm.release_background_connection(c3)
        dm._close_background_connection_pool()

        assert all(conn.closed for conn in created)

    def test_background_pool_disabled_for_local_database(self):
        from geovibes.ui.data_manager import DataManager

        dm = DataManager.__new__(DataManager)
        dm.verbose = False
        dm.current_database_path = "/tmp/local.db"

        dm.configure_background_connection_pool(4)
        stats = dm.background_pool_stats()
        assert stats["size"] == 0

    def test_precreate_background_pool_fills_idle_connections(self):
        from geovibes.ui.data_manager import DataManager

        dm = DataManager.__new__(DataManager)
        dm.verbose = False
        dm.current_database_path = "s3://bucket/path/metadata.db"
        created = []

        def fake_connect(_path):
            conn = self._DummyConn(f"conn-{len(created)}")
            created.append(conn)
            return conn

        dm._connect_duckdb = fake_connect

        result = dm.precreate_background_connection_pool(3, n_workers=3)
        stats = dm.background_pool_stats()

        assert stats["size"] == 3
        assert stats["created"] == 3
        assert stats["idle"] == 3
        assert result["created_ok"] == 3.0

        dm._close_background_connection_pool()
        assert all(conn.closed for conn in created)

    def test_suggest_background_pool_size_respects_memory_cap(self, monkeypatch):
        from geovibes.ui.data_manager import DataManager

        dm = DataManager.__new__(DataManager)
        monkeypatch.setenv("GEOVIBES_PREFETCH_WORKER_TARGET", "44")
        monkeypatch.setenv("GEOVIBES_PREFETCH_WORKER_CAP", "44")
        monkeypatch.setenv("GEOVIBES_PREFETCH_WORKER_FLOOR", "16")
        monkeypatch.setenv("GEOVIBES_PREFETCH_WORKER_MEM_MB", "96")
        monkeypatch.setattr(
            DataManager,
            "_mem_available_mb",
            staticmethod(lambda: 300.0),
        )

        size = dm.suggest_background_pool_size()
        assert size == 3
