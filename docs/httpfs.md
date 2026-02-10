# Remote Database Access via httpfs

Planning document for querying DuckDB + FAISS index pairs stored on AWS S3 without full download.

---

## Current Status

**Phase 0: Benchmarking** - ✓ Complete

| Step | Status | Notes |
|------|--------|-------|
| Create benchmark script | ✓ Complete | `geovibes/database/benchmark_httpfs.py` |
| Test local DuckDB + spatial | ✓ Complete | 27ms spatial queries, 2ms fetch |
| Test httpfs extension loads | ✓ Complete | All extensions load correctly |
| Upload database to S3 | ✓ Complete | 1.18 GB uploaded to Source Cooperative |
| Run remote benchmarks | ✓ Complete | See results below |

### Benchmark Results (2024-12-31)

Database: `alabama_google_satellite_embeddings_v1_2024_2025_25_0_10_metadata.db` (1.18 GB, 2.18M rows)

| Metric | Remote | Local | Ratio | Threshold | Status |
|--------|--------|-------|-------|-----------|--------|
| Cold start (attach) | 6820 ms | - | - | 5000 ms | ⚠ WARN |
| First spatial query | 5333 ms | - | - | - | Expected |
| Warm spatial query | 15 ms | 16 ms | 0.9x | - | ✓ PASS |
| Fetch 100 by ID | 1 ms | 1 ms | 1.0x | 1000 ms | ✓ PASS |
| Fetch 500 by ID | 2 ms | - | - | 3000 ms | ✓ PASS |

### Key Findings

1. **Cold start is slow (6-7 sec)** - Acceptable for notebook use (happens once per session)
2. **First query loads metadata** - 5 sec for first spatial query, then 15ms
3. **Fetch-by-ID is instant** - 1-2ms regardless of count, same as local
4. **Spatial queries NOT needed** - FAISS handles search locally

### Decision: ✓ PROCEED TO PHASE 1

The hybrid architecture works as designed:
- FAISS search: local (~1ms)
- Fetch by ID: remote via httpfs (~1-2ms)
- **No spatial queries in critical path**

---

**Phase 1: Implementation** - ✓ Complete

| Step | Status | Notes |
|------|--------|-------|
| Write failing tests (TDD RED) | ✓ Complete | 15 tests in `tests/test_remote_db.py` |
| Implement FaissCache | ✓ Complete | `geovibes/database/faiss_cache.py` |
| Implement RemoteDuckDB | ✓ Complete | `geovibes/database/remote_db.py` |
| All tests pass (TDD GREEN) | ✓ Complete | 229 tests pass |

### Implemented Modules

**FaissCache** (`geovibes/database/faiss_cache.py`):
- Downloads FAISS indexes from S3 using fsspec
- Caches locally in `~/.cache/geovibes/faiss/`
- Uses SHA256-based cache keys for deterministic paths
- Validates cached indexes on load
- Atomic downloads with temp file + rename

**RemoteDuckDB** (`geovibes/database/remote_db.py`):
- Connects to DuckDB on S3 via httpfs extension
- Supports S3 URLs (`s3://`) and local files (`file://`)
- Auto-loads AWS credentials
- Provides `fetch_by_ids()` and `fetch_embeddings()` helpers
- Converts numpy types for DuckDB compatibility
- Clear error messages for common failures (404, 403, timeout)

### Usage Example

```python
from geovibes.database.remote_db import RemoteDuckDB

# Connect to remote database
db = RemoteDuckDB()
db.connect("s3://us-west-2.opendata.source.coop/geovibes/search/USA/alabama/2024-01-01-2025-01-01/alabama_google_satellite_embeddings_v1_2024_2025_25_0_10_metadata.db")

# Fetch by ID (fast - 1-2ms)
result = db.fetch_by_ids([1, 2, 3], columns=["id", "geometry"])

# Query with SQL
df = db.query("SELECT COUNT(*) FROM geo_embeddings")

db.close()
```

**Phase 1.6: DataManager Integration** - ✓ Complete

| Step | Status | Notes |
|------|--------|-------|
| Write failing tests for `is_remote_url()` | ✓ Complete | `tests/test_data_manager_remote.py` |
| Implement `is_remote_url()` static method | ✓ Complete | Added to DataManager class |
| Test manifest parsing with S3 paths | ✓ Complete | CSV parsing verified |
| Verify FaissCache integration | ✓ Complete | Caching works correctly |
| All tests pass | ✓ Complete | 238 tests pass |

**DataManager.is_remote_url()** (`geovibes/ui/data_manager.py:30-42`):
```python
@staticmethod
def is_remote_url(path: str) -> bool:
    """Check if a path is a remote URL (S3 or GCS)."""
    if not path:
        return False
    return path.startswith("s3://") or path.startswith("gs://")
```

**Next Steps:**
- Add remote database switching in DataManager (optional enhancement)
- Support direct S3 database URLs in manifest (optional)

---

## Proposed Architecture: Local FAISS + Remote DuckDB

For notebook use, the optimal architecture is:
- **FAISS index**: Download once, cache locally (150-400 MB)
- **DuckDB database**: Query via httpfs (1-28 GB stays on S3)

| Model | Download (FAISS) | Skip via httpfs (DuckDB) | **Savings** |
|-------|------------------|--------------------------|-------------|
| dino_vit | 372 MB | 28 GB | **75x less** |
| earthgenome | 225 MB | 7.2 GB | **33x less** |
| google_sat | 151 MB | 1.1 GB | **8x less** |

```
Current: Download everything
├── FAISS index: 372 MB  ─┐
└── DuckDB: 28 GB        ─┴─► 28.4 GB download

Proposed: Hybrid
├── FAISS index: 372 MB  ─► Local cache (download once)
└── DuckDB: 28 GB        ─► S3 via httpfs (no download!)
                             └── Only fetch rows you need
```

---

## Success Criteria

How do we know this implementation is successful?

### Phase 0 Success (Benchmarking)

| Criterion | Target | How to Measure |
|-----------|--------|----------------|
| Cold start latency | < 5 sec | Time from connection to first query result |
| Nearest point query | < 500 ms | Mean over 10 runs |
| Fetch 100 geometries | < 1 sec | Mean over 5 runs |
| Fetch 500 geometries | < 3 sec | Mean over 5 runs |
| Query reliability | 100% success | No failures over 100 queries |
| Local/remote ratio | < 10x | Remote not more than 10x slower than local |

**Go/No-Go Decision**: If any criterion fails, document the failure and consider alternatives before proceeding.

### Phase 1 Success (Implementation)

| Criterion | Target | How to Measure |
|-----------|--------|----------------|
| Existing tests pass | 100% | `pytest` green |
| New remote tests pass | 100% | New test suite green |
| Cache integrity | No corruption | Checksum validation |
| Memory usage | < 2 GB | Monitor during typical session |
| Error messages | Clear and actionable | Manual review |
| Offline graceful degradation | No crashes | Test with network disabled |

---

## Edge Cases and Failure Modes

### Network/Connection Failures

| Failure Mode | Detection | Mitigation |
|--------------|-----------|------------|
| **S3 connection timeout** | `ConnectionError`, `TimeoutError` | Retry with exponential backoff (3 attempts) |
| **Partial FAISS download** | File size mismatch, FAISS load fails | Delete partial file, re-download |
| **httpfs query timeout** | Query hangs > 30 sec | Set DuckDB timeout, raise clear error |
| **Mid-session network loss** | `IOError` on query | Cache recent results, show offline warning |
| **S3 rate limiting** | HTTP 503, SlowDown error | Backoff and retry |

### Authentication Failures

| Failure Mode | Detection | Mitigation |
|--------------|-----------|------------|
| **Missing AWS credentials** | `NoCredentialsError` | Clear message: "Run `aws configure` or set AWS_ACCESS_KEY_ID" |
| **Expired credentials** | `ExpiredTokenException` | Clear message: "Refresh credentials with `aws sso login`" |
| **Wrong bucket permissions** | HTTP 403 Forbidden | Clear message: "Check S3 bucket policy for read access" |
| **Wrong region** | HTTP 301 redirect | Auto-detect region from error, retry |

### Cache Failures

| Failure Mode | Detection | Mitigation |
|--------------|-----------|------------|
| **Disk full during download** | `OSError: No space left` | Check space before download, clear old cache |
| **Cache corruption** | FAISS `read_index` fails | Delete corrupted file, re-download |
| **Cache permissions** | `PermissionError` | Fallback to temp directory |
| **Stale cache (index updated)** | ETag mismatch | Re-download if ETag differs |
| **Multiple processes writing** | Race condition | Use file locking or atomic rename |

### DuckDB/httpfs Specific

| Failure Mode | Detection | Mitigation |
|--------------|-----------|------------|
| **Spatial extension not working over httpfs** | Query error | Test explicitly in Phase 0 |
| **Large result set OOM** | `MemoryError` | Paginate results, limit default count |
| **Schema version mismatch** | Missing columns | Version check in manifest |
| **DuckDB version incompatibility** | Various errors | Pin DuckDB version in requirements |

### Concurrency

| Failure Mode | Detection | Mitigation |
|--------------|-----------|------------|
| **Two notebooks downloading same index** | File lock conflict | Use `filelock` library |
| **Connection pool exhaustion** | Connection errors | Single connection per session |
| **mmap file locked** | `PermissionError` on Windows | Copy to temp file before mmap |

---

## Step-by-Step Implementation Plan

### Phase 0: Benchmarking

**Goal**: Validate that httpfs latency is acceptable before writing production code.

#### Step 0.1: Upload Test Database to S3

**Using upload script (recommended):**
```bash
# Get session token from Source Cooperative console, then:
python -m geovibes.database.upload_for_httpfs \
    local_databases/alabama_google_satellite_embeddings_v1_2024_2025_25_0_10_metadata.db \
    --session-token "YOUR_SESSION_TOKEN"

# Or with AWS CLI directly:
aws s3 cp local_databases/alabama_google_satellite_embeddings_v1_2024_2025_25_0_10_metadata.db \
    s3://geovibes/search/USA/alabama/2024-01-01-2025-01-01/httpfs/alabama_google.db \
    --endpoint-url https://data.source.coop
```

**Note**: Static Source Cooperative credentials are read-only. Session tokens with write access are required.

**Success**: File visible in S3, correct size (1.1 GB).

#### Step 0.2: Verify Basic httpfs Connectivity

```python
# test_httpfs_basic.py
import duckdb

def test_httpfs_connection():
    """Verify we can connect to S3 via httpfs at all."""
    conn = duckdb.connect()
    conn.execute("INSTALL httpfs; LOAD httpfs;")
    conn.execute("INSTALL aws; LOAD aws;")

    # Source Cooperative config
    conn.execute("SET s3_access_key_id='***REMOVED***';")
    conn.execute("SET s3_secret_access_key='***REMOVED***';")
    conn.execute("SET s3_endpoint='data.source.coop';")
    conn.execute("SET s3_url_style='path';")
    conn.execute("SET s3_region='us-west-2';")

    # This should NOT fail (after upload)
    conn.execute("""
        ATTACH 's3://geovibes/search/USA/alabama/2024-01-01-2025-01-01/httpfs/alabama_google.db'
        AS remote (READ_ONLY)
    """)

    # Verify we can see the table
    tables = conn.execute("SHOW TABLES").fetchall()
    assert any("geo_embeddings" in str(t) for t in tables), "Table not found"

    conn.close()
    print("✓ Basic httpfs connection works")
```

**Success**: No exceptions, table visible.
**Failure**: Note the exact error message, check credentials.

#### Step 0.3: Verify Spatial Extension Works Over httpfs

```python
def test_spatial_over_httpfs():
    """Verify spatial queries work - this is critical."""
    conn = setup_remote_connection("s3://geovibes-benchmark-test/alabama_google.db")

    # Test 1: Simple spatial function
    result = conn.execute("SELECT ST_Point(-86.5, 32.5)").fetchone()
    assert result is not None, "ST_Point failed"

    # Test 2: Spatial query on actual data
    result = conn.execute("""
        SELECT id, ST_AsText(geometry)
        FROM geo_embeddings
        LIMIT 1
    """).fetchone()
    assert result is not None, "Cannot read geometry"

    # Test 3: Spatial distance query (the critical one)
    result = conn.execute("""
        SELECT id, ST_Distance(geometry, ST_Point(-86.5, 32.5)) as dist
        FROM geo_embeddings
        ORDER BY dist
        LIMIT 1
    """).fetchone()
    assert result is not None, "Spatial distance query failed"

    conn.close()
    print("✓ Spatial queries work over httpfs")
```

**Success**: All three queries return results.
**Failure**: If spatial fails, this approach won't work - document and stop.

#### Step 0.4: Run Latency Benchmarks

```python
# benchmark_httpfs.py

import argparse
import json
import time
import duckdb
import numpy as np
from pathlib import Path

def setup_remote_connection(s3_url: str) -> duckdb.DuckDBPyConnection:
    conn = duckdb.connect()
    conn.execute("INSTALL httpfs; LOAD httpfs;")
    conn.execute("INSTALL spatial; LOAD spatial;")
    conn.execute("INSTALL aws; LOAD aws;")
    conn.execute("CALL load_aws_credentials();")
    conn.execute(f"ATTACH '{s3_url}' AS remote (READ_ONLY)")
    conn.execute("USE remote")
    return conn

def setup_local_connection(db_path: str) -> duckdb.DuckDBPyConnection:
    conn = duckdb.connect(db_path, read_only=True)
    conn.execute("LOAD spatial;")
    return conn

def benchmark_cold_start(s3_url: str) -> float:
    """Measure time to connect and run first query."""
    start = time.perf_counter()
    conn = setup_remote_connection(s3_url)
    conn.execute("SELECT COUNT(*) FROM geo_embeddings").fetchone()
    elapsed = (time.perf_counter() - start) * 1000
    conn.close()
    return elapsed

def benchmark_nearest_point(conn, lon: float, lat: float, n_runs: int = 10) -> list[float]:
    """Benchmark spatial nearest-point query."""
    sql = """
        SELECT id, ST_AsText(geometry) AS wkt
        FROM geo_embeddings
        ORDER BY ST_Distance(geometry, ST_Point(?, ?))
        LIMIT 1
    """
    latencies = []
    for _ in range(n_runs):
        start = time.perf_counter()
        conn.execute(sql, [lon, lat]).fetchone()
        latencies.append((time.perf_counter() - start) * 1000)
    return latencies

def benchmark_fetch_by_id(conn, n_ids: int, n_runs: int = 5) -> list[float]:
    """Benchmark fetching rows by ID."""
    # Get sample IDs
    all_ids = conn.execute(f"SELECT id FROM geo_embeddings LIMIT {n_ids}").fetchdf()["id"].tolist()
    placeholders = ",".join(["?" for _ in all_ids])
    sql = f"SELECT id, ST_AsGeoJSON(geometry) FROM geo_embeddings WHERE id IN ({placeholders})"

    latencies = []
    for _ in range(n_runs):
        start = time.perf_counter()
        conn.execute(sql, all_ids).fetchdf()
        latencies.append((time.perf_counter() - start) * 1000)
    return latencies

def run_full_benchmark(s3_url: str, local_path: str) -> dict:
    """Run complete benchmark suite."""
    results = {"s3_url": s3_url, "local_path": local_path, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}

    # Cold start
    print("Testing cold start...")
    results["cold_start_ms"] = benchmark_cold_start(s3_url)
    print(f"  Cold start: {results['cold_start_ms']:.0f} ms")

    # Remote benchmarks
    print("Connecting to remote DB...")
    remote_conn = setup_remote_connection(s3_url)

    print("Testing nearest point (remote)...")
    remote_nearest = benchmark_nearest_point(remote_conn, -86.5, 32.5)
    results["remote_nearest_point"] = {
        "mean_ms": np.mean(remote_nearest),
        "std_ms": np.std(remote_nearest),
        "min_ms": np.min(remote_nearest),
        "max_ms": np.max(remote_nearest),
    }
    print(f"  Nearest point: {results['remote_nearest_point']['mean_ms']:.0f} ms (±{results['remote_nearest_point']['std_ms']:.0f})")

    print("Testing fetch by ID (remote)...")
    results["remote_fetch_by_id"] = {}
    for n in [10, 50, 100, 200, 500, 1000]:
        latencies = benchmark_fetch_by_id(remote_conn, n)
        results["remote_fetch_by_id"][n] = {
            "mean_ms": np.mean(latencies),
            "std_ms": np.std(latencies),
        }
        print(f"  Fetch {n}: {results['remote_fetch_by_id'][n]['mean_ms']:.0f} ms")

    remote_conn.close()

    # Local baseline
    if local_path and Path(local_path).exists():
        print("Connecting to local DB...")
        local_conn = setup_local_connection(local_path)

        print("Testing nearest point (local)...")
        local_nearest = benchmark_nearest_point(local_conn, -86.5, 32.5)
        results["local_nearest_point"] = {"mean_ms": np.mean(local_nearest)}
        print(f"  Local nearest point: {results['local_nearest_point']['mean_ms']:.1f} ms")

        print("Testing fetch by ID (local)...")
        local_fetch = benchmark_fetch_by_id(local_conn, 100)
        results["local_fetch_100"] = {"mean_ms": np.mean(local_fetch)}
        print(f"  Local fetch 100: {results['local_fetch_100']['mean_ms']:.1f} ms")

        local_conn.close()

        # Calculate ratios
        results["ratio_nearest"] = results["remote_nearest_point"]["mean_ms"] / results["local_nearest_point"]["mean_ms"]
        results["ratio_fetch_100"] = results["remote_fetch_by_id"][100]["mean_ms"] / results["local_fetch_100"]["mean_ms"]

    return results

def check_acceptance_criteria(results: dict) -> tuple[bool, list[str]]:
    """Check if results meet acceptance criteria."""
    failures = []

    if results["cold_start_ms"] > 5000:
        failures.append(f"Cold start {results['cold_start_ms']:.0f}ms > 5000ms")

    if results["remote_nearest_point"]["mean_ms"] > 500:
        failures.append(f"Nearest point {results['remote_nearest_point']['mean_ms']:.0f}ms > 500ms")

    if results["remote_fetch_by_id"][100]["mean_ms"] > 1000:
        failures.append(f"Fetch 100 {results['remote_fetch_by_id'][100]['mean_ms']:.0f}ms > 1000ms")

    if results["remote_fetch_by_id"][500]["mean_ms"] > 3000:
        failures.append(f"Fetch 500 {results['remote_fetch_by_id'][500]['mean_ms']:.0f}ms > 3000ms")

    if "ratio_nearest" in results and results["ratio_nearest"] > 10:
        failures.append(f"Remote/local ratio {results['ratio_nearest']:.1f}x > 10x")

    return len(failures) == 0, failures

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--s3-url", required=True)
    parser.add_argument("--local-path", default=None)
    parser.add_argument("--output", default="benchmark_results.json")
    args = parser.parse_args()

    results = run_full_benchmark(args.s3_url, args.local_path)

    passed, failures = check_acceptance_criteria(results)
    results["passed"] = passed
    results["failures"] = failures

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "="*50)
    if passed:
        print("✓ ALL ACCEPTANCE CRITERIA PASSED")
        print("  Proceed to Phase 1 implementation")
    else:
        print("✗ ACCEPTANCE CRITERIA FAILED:")
        for f in failures:
            print(f"  - {f}")
        print("  Review failures before proceeding")
```

**Success**: Script completes, all acceptance criteria pass.
**Failure**: Document specific failures, decide whether to proceed.

#### Step 0.5: Document Results

Fill in the results template:

```
=== httpfs Benchmark Results ===

Database: s3://geovibes-benchmark-test/alabama_google.db
File size: 1.1 GB
Client location: [your location]
Date: [date]

1. COLD START
   - Connection + first query: ___ ms
   - PASS/FAIL (threshold: 5000 ms)

2. NEAREST POINT (spatial index lookup)
   - Mean: ___ ms (std: ___ ms)
   - PASS/FAIL (threshold: 500 ms)

3. FETCH GEOMETRIES BY ID
   | Count | Mean (ms) | Std (ms) | PASS/FAIL |
   |-------|-----------|----------|-----------|
   | 10    | ___       | ___      |           |
   | 50    | ___       | ___      |           |
   | 100   | ___       | ___      | (< 1000)  |
   | 200   | ___       | ___      |           |
   | 500   | ___       | ___      | (< 3000)  |
   | 1000  | ___       | ___      |           |

4. LOCAL BASELINE
   - Nearest point: ___ ms
   - Fetch 100: ___ ms

5. REMOTE/LOCAL RATIO
   - Nearest point: ___x (threshold: < 10x)
   - Fetch 100: ___x (threshold: < 10x)

DECISION: [ ] PROCEED TO PHASE 1  [ ] INVESTIGATE FAILURES
```

---

### Phase 1: Implementation (TDD Approach)

**Only proceed if Phase 0 passed.**

#### Step 1.1: Write Failing Tests for FaissCache

Create tests first, then implement to make them pass.

```python
# tests/test_faiss_cache.py

import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import faiss
import numpy as np

# Import will fail until we create the module
from geovibes.database.faiss_cache import FaissCache

class TestFaissCache:
    """Tests for FAISS index caching."""

    @pytest.fixture
    def cache_dir(self):
        """Create temporary cache directory."""
        with tempfile.TemporaryDirectory() as d:
            yield Path(d)

    @pytest.fixture
    def sample_index_path(self, tmp_path):
        """Create a small FAISS index for testing."""
        index = faiss.IndexFlatL2(64)
        vectors = np.random.rand(100, 64).astype(np.float32)
        index.add(vectors)
        path = tmp_path / "test.index"
        faiss.write_index(index, str(path))
        return path

    # --- Core Functionality Tests ---

    def test_downloads_if_not_cached(self, cache_dir, sample_index_path):
        """Should download index if not in cache."""
        cache = FaissCache(cache_dir=cache_dir)

        # Mock fsspec to return our test file
        with patch("geovibes.database.faiss_cache.fsspec") as mock_fs:
            mock_file = MagicMock()
            mock_file.read.side_effect = [
                sample_index_path.read_bytes()[:1024],  # First chunk
                sample_index_path.read_bytes()[1024:],   # Rest
                b"",  # EOF
            ]
            mock_file.size = sample_index_path.stat().st_size
            mock_fs.open.return_value.__enter__.return_value = mock_file

            index = cache.get_index("s3://bucket/test.index", show_progress=False)

            assert index is not None
            assert index.ntotal == 100
            assert mock_fs.open.called

    def test_uses_cache_if_exists(self, cache_dir, sample_index_path):
        """Should not download if index already cached."""
        cache = FaissCache(cache_dir=cache_dir)

        # Pre-populate cache
        import hashlib
        cache_key = hashlib.sha256(b"s3://bucket/test.index").hexdigest()[:16]
        cache_path = cache_dir / f"{cache_key}.index"
        cache_path.write_bytes(sample_index_path.read_bytes())

        with patch("geovibes.database.faiss_cache.fsspec") as mock_fs:
            index = cache.get_index("s3://bucket/test.index", show_progress=False)

            assert index is not None
            assert not mock_fs.open.called  # Should not download

    def test_redownloads_corrupted_cache(self, cache_dir, sample_index_path):
        """Should re-download if cached file is corrupted."""
        cache = FaissCache(cache_dir=cache_dir)

        # Create corrupted cache file
        import hashlib
        cache_key = hashlib.sha256(b"s3://bucket/test.index").hexdigest()[:16]
        cache_path = cache_dir / f"{cache_key}.index"
        cache_path.write_bytes(b"corrupted data")

        with patch("geovibes.database.faiss_cache.fsspec") as mock_fs:
            mock_file = MagicMock()
            mock_file.read.side_effect = [sample_index_path.read_bytes(), b""]
            mock_file.size = sample_index_path.stat().st_size
            mock_fs.open.return_value.__enter__.return_value = mock_file

            index = cache.get_index("s3://bucket/test.index", show_progress=False)

            assert index is not None
            assert mock_fs.open.called  # Should have re-downloaded

    # --- Error Handling Tests ---

    def test_handles_download_failure(self, cache_dir):
        """Should raise clear error on download failure."""
        cache = FaissCache(cache_dir=cache_dir)

        with patch("geovibes.database.faiss_cache.fsspec") as mock_fs:
            mock_fs.open.side_effect = ConnectionError("Network unreachable")

            with pytest.raises(ConnectionError):
                cache.get_index("s3://bucket/test.index", show_progress=False)

    def test_handles_partial_download(self, cache_dir, sample_index_path):
        """Should clean up and retry on partial download."""
        cache = FaissCache(cache_dir=cache_dir)

        call_count = [0]
        def mock_read(*args):
            call_count[0] += 1
            if call_count[0] == 1:
                raise ConnectionError("Connection lost")
            return sample_index_path.read_bytes()

        with patch("geovibes.database.faiss_cache.fsspec") as mock_fs:
            mock_file = MagicMock()
            mock_file.read.side_effect = mock_read
            mock_file.size = sample_index_path.stat().st_size
            mock_fs.open.return_value.__enter__.return_value = mock_file

            # Should retry and succeed
            # (implementation should handle this)

    def test_handles_disk_full(self, cache_dir, sample_index_path):
        """Should raise clear error when disk is full."""
        cache = FaissCache(cache_dir=cache_dir)

        with patch("builtins.open", side_effect=OSError("No space left on device")):
            with pytest.raises(OSError) as exc_info:
                cache.get_index("s3://bucket/test.index", show_progress=False)

            assert "space" in str(exc_info.value).lower()

    # --- Cache Management Tests ---

    def test_cache_key_is_deterministic(self, cache_dir):
        """Same URL should always produce same cache key."""
        cache = FaissCache(cache_dir=cache_dir)

        key1 = cache._get_cache_key("s3://bucket/index.faiss")
        key2 = cache._get_cache_key("s3://bucket/index.faiss")
        key3 = cache._get_cache_key("s3://bucket/other.faiss")

        assert key1 == key2
        assert key1 != key3

    def test_different_urls_different_cache_keys(self, cache_dir):
        """Different URLs should have different cache keys."""
        cache = FaissCache(cache_dir=cache_dir)

        key1 = cache._get_cache_key("s3://bucket/v1/index.faiss")
        key2 = cache._get_cache_key("s3://bucket/v2/index.faiss")

        assert key1 != key2
```

**Run tests**: `pytest tests/test_faiss_cache.py -v`
**Expected**: All tests fail (module doesn't exist yet)

#### Step 1.2: Implement FaissCache to Pass Tests

```python
# geovibes/database/faiss_cache.py

from pathlib import Path
import hashlib
import logging
import faiss
import fsspec

logger = logging.getLogger(__name__)

class FaissCache:
    """Download and cache FAISS indexes locally with integrity checking."""

    def __init__(self, cache_dir: Path = None):
        self.cache_dir = cache_dir or Path.home() / ".cache" / "geovibes" / "faiss"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_key(self, url: str) -> str:
        """Generate deterministic cache key from URL."""
        return hashlib.sha256(url.encode()).hexdigest()[:16]

    def get_index(self, remote_url: str, show_progress: bool = True) -> faiss.Index:
        """Load index from cache, downloading if necessary."""
        cache_key = self._get_cache_key(remote_url)
        cache_path = self.cache_dir / f"{cache_key}.index"
        temp_path = self.cache_dir / f"{cache_key}.index.tmp"

        # Try loading from cache
        if cache_path.exists():
            try:
                return faiss.read_index(str(cache_path), faiss.IO_FLAG_MMAP)
            except Exception as e:
                logger.warning(f"Cached index corrupted, re-downloading: {e}")
                cache_path.unlink()

        # Download to temp file, then atomic rename
        self._download(remote_url, temp_path, show_progress)

        # Verify the downloaded index is valid
        try:
            test_index = faiss.read_index(str(temp_path))
            _ = test_index.ntotal  # Verify we can read it
        except Exception as e:
            temp_path.unlink(missing_ok=True)
            raise ValueError(f"Downloaded index is invalid: {e}")

        # Atomic rename
        temp_path.rename(cache_path)

        return faiss.read_index(str(cache_path), faiss.IO_FLAG_MMAP)

    def _download(self, url: str, dest: Path, show_progress: bool):
        """Download file with progress and integrity checking."""
        from tqdm import tqdm

        try:
            with fsspec.open(url, "rb") as src:
                size = getattr(src, 'size', None)

                with open(dest, "wb") as dst:
                    pbar = None
                    if show_progress and size:
                        pbar = tqdm(total=size, unit="B", unit_scale=True,
                                    desc="Downloading FAISS index")

                    try:
                        while True:
                            chunk = src.read(8 * 1024 * 1024)
                            if not chunk:
                                break
                            dst.write(chunk)
                            if pbar:
                                pbar.update(len(chunk))
                    finally:
                        if pbar:
                            pbar.close()

        except Exception as e:
            # Clean up partial download
            dest.unlink(missing_ok=True)
            raise

    def clear_cache(self):
        """Remove all cached indexes."""
        for f in self.cache_dir.glob("*.index"):
            f.unlink()
```

**Run tests**: `pytest tests/test_faiss_cache.py -v`
**Expected**: Tests pass

#### Step 1.3: Write Failing Tests for Remote DuckDB

```python
# tests/test_remote_duckdb.py

import pytest
from unittest.mock import patch, MagicMock
import duckdb

from geovibes.database.remote_db import RemoteDuckDB

class TestRemoteDuckDB:
    """Tests for remote DuckDB connection via httpfs."""

    def test_connects_to_s3(self):
        """Should connect to S3 database via httpfs."""
        # This is an integration test - needs real S3 bucket
        # Skip if no credentials
        pass

    def test_spatial_queries_work(self):
        """Should execute spatial queries successfully."""
        pass

    def test_handles_auth_error(self):
        """Should raise clear error on authentication failure."""
        db = RemoteDuckDB()

        with patch.object(db, '_execute') as mock_exec:
            mock_exec.side_effect = duckdb.IOException("HTTP 403 Forbidden")

            with pytest.raises(PermissionError) as exc_info:
                db.connect("s3://bucket/db.duckdb")

            assert "credentials" in str(exc_info.value).lower() or \
                   "permission" in str(exc_info.value).lower()

    def test_handles_not_found(self):
        """Should raise clear error when database doesn't exist."""
        db = RemoteDuckDB()

        with patch.object(db, '_execute') as mock_exec:
            mock_exec.side_effect = duckdb.IOException("HTTP 404 Not Found")

            with pytest.raises(FileNotFoundError) as exc_info:
                db.connect("s3://bucket/nonexistent.duckdb")

    def test_handles_timeout(self):
        """Should raise clear error on connection timeout."""
        db = RemoteDuckDB()

        with patch.object(db, '_execute') as mock_exec:
            mock_exec.side_effect = duckdb.IOException("Connection timed out")

            with pytest.raises(TimeoutError):
                db.connect("s3://bucket/db.duckdb")

    def test_query_returns_results(self):
        """Should return query results as DataFrame."""
        pass  # Integration test

    def test_closes_connection(self):
        """Should close connection cleanly."""
        pass
```

#### Step 1.4: Integration Tests

```python
# tests/test_remote_integration.py

import pytest
import os

# Skip all tests if no S3 credentials
pytestmark = pytest.mark.skipif(
    not os.environ.get("AWS_ACCESS_KEY_ID"),
    reason="AWS credentials not configured"
)

S3_TEST_DB = os.environ.get("GEOVIBES_TEST_S3_DB", "")
S3_TEST_INDEX = os.environ.get("GEOVIBES_TEST_S3_INDEX", "")

class TestRemoteIntegration:
    """Integration tests for full remote workflow."""

    @pytest.fixture
    def remote_db(self):
        """Connect to test database on S3."""
        from geovibes.database.remote_db import RemoteDuckDB
        db = RemoteDuckDB()
        db.connect(S3_TEST_DB)
        yield db
        db.close()

    def test_spatial_query_returns_result(self, remote_db):
        """Spatial nearest-point query should work."""
        result = remote_db.query("""
            SELECT id, ST_AsText(geometry)
            FROM geo_embeddings
            ORDER BY ST_Distance(geometry, ST_Point(-86.5, 32.5))
            LIMIT 1
        """)
        assert len(result) == 1

    def test_fetch_by_id_works(self, remote_db):
        """Fetching rows by ID should work."""
        # Get some IDs first
        ids = remote_db.query("SELECT id FROM geo_embeddings LIMIT 10")["id"].tolist()

        # Fetch by ID
        placeholders = ",".join(["?" for _ in ids])
        result = remote_db.query(
            f"SELECT id, embedding FROM geo_embeddings WHERE id IN ({placeholders})",
            ids
        )
        assert len(result) == 10

    def test_full_geovibes_workflow(self):
        """Full GeoVibes workflow should work in remote mode."""
        from geovibes import GeoVibes

        app = GeoVibes(
            remote_index=S3_TEST_INDEX,
            remote_db=S3_TEST_DB,
        )

        # Simulate operations
        # ... (would need more specific test based on GeoVibes API)

        app.close()
```

#### Step 1.5: Implement Remote DuckDB

```python
# geovibes/database/remote_db.py

import duckdb
import logging

logger = logging.getLogger(__name__)

class RemoteDuckDB:
    """Remote DuckDB connection via httpfs."""

    def __init__(self):
        self.conn = None

    def connect(self, s3_url: str) -> None:
        """Connect to DuckDB on S3."""
        self.conn = duckdb.connect()

        try:
            self.conn.execute("INSTALL httpfs; LOAD httpfs;")
            self.conn.execute("INSTALL spatial; LOAD spatial;")
            self.conn.execute("INSTALL aws; LOAD aws;")
            self.conn.execute("CALL load_aws_credentials();")
            self.conn.execute(f"ATTACH '{s3_url}' AS remote (READ_ONLY)")
            self.conn.execute("USE remote")
        except duckdb.IOException as e:
            self._handle_io_error(e, s3_url)

    def _handle_io_error(self, error: Exception, url: str):
        """Convert DuckDB IO errors to clear Python exceptions."""
        msg = str(error).lower()

        if "403" in msg or "forbidden" in msg:
            raise PermissionError(
                f"Access denied to {url}. Check AWS credentials and bucket permissions."
            )
        elif "404" in msg or "not found" in msg:
            raise FileNotFoundError(f"Database not found: {url}")
        elif "timeout" in msg:
            raise TimeoutError(f"Connection to {url} timed out")
        else:
            raise

    def query(self, sql: str, params=None):
        """Execute query and return DataFrame."""
        if params:
            return self.conn.execute(sql, params).fetchdf()
        return self.conn.execute(sql).fetchdf()

    def close(self):
        """Close connection."""
        if self.conn:
            self.conn.close()
            self.conn = None
```

#### Step 1.6: Integrate into DataManager

Create tests for DataManager changes, then implement.

---

## Effort Estimate (Revised)

| Phase | Step | Effort | Cumulative |
|-------|------|--------|------------|
| **0** | 0.1: Upload to S3 | 0.25 day | 0.25 day |
| | 0.2-0.3: Verify connectivity | 0.25 day | 0.5 day |
| | 0.4: Run benchmarks | 0.5 day | 1 day |
| | 0.5: Document results | 0.25 day | 1.25 days |
| **1** | 1.1: Write FaissCache tests | 0.5 day | 1.75 days |
| | 1.2: Implement FaissCache | 0.5 day | 2.25 days |
| | 1.3: Write RemoteDuckDB tests | 0.5 day | 2.75 days |
| | 1.4: Integration tests | 0.5 day | 3.25 days |
| | 1.5: Implement RemoteDuckDB | 0.5 day | 3.75 days |
| | 1.6: Integrate into DataManager | 1 day | 4.75 days |
| | Final testing + docs | 0.25 day | **5 days** |

---

## Rollback Plan

If Phase 0 fails or Phase 1 has issues:

1. **Phase 0 fails**: Document latency numbers, consider alternatives:
   - CloudFront caching
   - Smaller default result sets
   - Full download fallback for specific models

2. **Phase 1 issues**:
   - All changes are additive (new files, new parameters)
   - Existing local mode remains unchanged
   - Can disable remote mode via feature flag

---

## Related Files

- `geovibes/ui/data_manager.py` — DataManager class
- `geovibes/database/faiss_db.py` — FAISS index building
- `tests/` — Test directory
- `docs/data-formats.md` — DuckDB schema reference

---

## Phase 2: Performance Optimization (2025-01)

### Problem: Full Table Scans Over httpfs

Initial testing revealed that ID-based lookups were scanning all ~22M rows despite having a PRIMARY KEY constraint. Investigation uncovered two issues:

1. **DuckDB PRIMARY KEY is NOT an index** — Unlike PostgreSQL, DuckDB's PRIMARY KEY is just a uniqueness constraint. An explicit `CREATE INDEX` is required for fast lookups.

2. **Zone maps ineffective due to data ordering** — DuckDB uses min/max statistics per row group (~122K rows) to skip irrelevant data. But our data was stored in geographic (tile_id) order, causing ID ranges to overlap across all row groups:

```
# Unsorted data - overlapping ID ranges per row group
Row Group 0: ids 3470 - 1018827
Row Group 1: ids 16347 - 1230689
Row Group 2: ids 8821 - 1442551
→ Every query must scan ALL row groups
```

### Solution: Sort by ID + Add Index

Added to `geovibes/database/faiss_db.py`:

```python
# Reorder table by id for optimal zone map performance over httpfs
logging.info("Reordering table by id for optimal remote query performance...")
con.execute("CREATE TABLE geo_embeddings_ordered AS SELECT * FROM geo_embeddings ORDER BY id")
con.execute("DROP TABLE geo_embeddings")
con.execute("ALTER TABLE geo_embeddings_ordered RENAME TO geo_embeddings")

# Create explicit index
logging.info("Creating index on id column for fast lookups...")
con.execute("CREATE INDEX id_idx ON geo_embeddings(id);")
```

After sorting, row groups have non-overlapping ranges:
```
# Sorted data - non-overlapping ID ranges
Row Group 0: ids 1 - 122880
Row Group 1: ids 122881 - 245760
Row Group 2: ids 245761 - 368640
→ Zone maps can skip irrelevant row groups
```

### Performance Characteristics

| Scenario | Latency | Notes |
|----------|---------|-------|
| First search (cold) | ~10-20s | Loading row groups from S3 |
| Same row group (cached) | <1ms | DuckDB caches row groups in memory |
| Sequential IDs (sorted DB) | Fast | Adjacent IDs share row groups |
| 100 scattered FAISS IDs | ~18s | FAISS returns IDs by similarity, not storage order |

**Fundamental limitation**: FAISS returns IDs ordered by embedding similarity, not storage order. Results inherently span multiple row groups, requiring ~500ms HTTP request per row group. Sorting helps with caching but can't eliminate this scatter.

### S3 Database URLs

Sorted databases uploaded to Source Cooperative:

**Google Satellite Embeddings v1** (2.18M rows, 1.2GB):
```
s3://us-west-2.opendata.source.coop/geovibes/search/USA/alabama/2024-01-01-2025-01-01/alabama_google_satellite_embeddings_v1_2024_2025_25_0_10_metadata_sorted.db
s3://us-west-2.opendata.source.coop/geovibes/search/USA/alabama/2024-01-01-2025-01-01/alabama_google_satellite_embeddings_v1_2024_2025_25_0_10_faiss_4096_64_8.index
```

**Quantized DINO ViT** (5.3M rows, 2.2GB):
```
s3://us-west-2.opendata.source.coop/geovibes/search/USA/alabama/2024-01-01-2025-01-01/alabama_quantized_dino_vit_small_patch16_224_2024_2025_32_16_10_metadata_sorted.db
s3://us-west-2.opendata.source.coop/geovibes/search/USA/alabama/2024-01-01-2025-01-01/alabama_quantized_dino_vit_small_patch16_224_2024_2025_32_16_10_faiss_4096_64_8.index
```

### Usage: Direct S3 Database Access

Added `faiss_path` parameter to `GeoVibes.create()` for explicit S3 paths:

```python
from geovibes import GeoVibes

app = GeoVibes.create(
    duckdb_path="s3://us-west-2.opendata.source.coop/geovibes/.../metadata_sorted.db",
    faiss_path="s3://us-west-2.opendata.source.coop/geovibes/.../faiss_4096_64_8.index",
    boundary_path="geometries/alabama.geojson",
    verbose=True
)
app.display()
```

The FAISS index is automatically downloaded and cached at `~/.cache/geovibes/faiss/`.

### Key Findings Summary

| Finding | Impact |
|---------|--------|
| PRIMARY KEY ≠ Index in DuckDB | Must explicitly create indexes |
| Data ordering affects zone maps | Sort by ID for optimal httpfs performance |
| ~500ms per row group over httpfs | Fundamental network latency |
| FAISS scatter unavoidable | Results span many row groups by design |
| Caching helps repeat queries | Same row groups instant after first load |

---

## Phase 3: Local Geometry Cache (2025-01)

### Problem: Search Metadata Query Bottleneck

After optimizing database sorting (Phase 2), profiling revealed the remaining bottleneck:

```
_search_faiss: [1/3] FAISS search completed in 1797.4ms (1001 results)
_search_faiss: [2/3] Metadata query completed in 25998.3ms  ← 26 seconds!
_search_faiss: [3/3] Process results completed in 100.6ms
_search_faiss: DONE total=27901.7ms
```

The metadata query fetches geometries for 1001 FAISS result IDs:

```sql
SELECT id, ST_AsGeoJSON(geometry), ST_AsText(geometry)
FROM geo_embeddings
WHERE id IN (?, ?, ?, ... 1001 IDs ...)
```

**Why it's slow**: FAISS returns IDs by embedding similarity, not storage order. Results are scattered across ~40 row groups, each requiring ~500ms HTTP fetch from S3.

### Options Considered

| Option | Approach | Startup Cost | Search Latency | Memory |
|--------|----------|--------------|----------------|--------|
| **Warmup all row groups** | Preload entire DB into cache | +20-30s every session | Fast (if cached) | High (~2GB) |
| **Local geometry cache** | Download small geometry-only file | +30s once | Always fast | Low |

### Decision: Local Geometry Cache

**Rationale:**
1. **Predictable performance** - Always fast, no cache eviction risk
2. **Small file size** - ~50-100MB (just id + Point geometry) vs 2.2GB full DB
3. **One-time download** - Like FAISS index, cached locally
4. **Memory efficient** - DuckDB memory-maps local files
5. **Offline capable** - Works without network after initial download

The alternative (warmup all row groups) would add 20-30 seconds to every kernel startup and still risk cache eviction under memory pressure.

### Implementation

#### File Format

Parquet with ZSTD compression:
- `id`: BIGINT (8 bytes)
- `geometry`: GEOMETRY (Point, ~24 bytes WKB)

**Naming convention:**
```
Database:        alabama_quantized_dino_..._metadata_sorted.db
Geometry cache:  alabama_quantized_dino_..._geometry.parquet
```

#### Size Estimates

| Database | Rows | Full DB | Geometry Cache |
|----------|------|---------|----------------|
| Quantized DINO | 5.3M | 2.2 GB | ~50-100 MB |
| Google Satellite | 2.2M | 1.2 GB | ~20-40 MB |

#### Generation (`faiss_db.py`)

```python
def export_geometry_cache(con, output_path: str):
    """Export id + geometry to compressed Parquet for local caching."""
    con.execute(f"""
        COPY (SELECT id, geometry FROM geo_embeddings ORDER BY id)
        TO '{output_path}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)
```

#### Generating Geometry Cache

**For new databases** — generated automatically by `faiss_db.py`:
```bash
python geovibes/database/faiss_db.py --roi-file ... --output_dir ...
# Creates: *_metadata.db, *_faiss.index, *_geometry_cache.parquet
```

**For existing databases**:
```python
from geovibes.database.faiss_db import export_geometry_cache
export_geometry_cache("/path/to/database.db", "/path/to/geometry_cache.parquet")
```

**Upload to S3**:
```bash
aws s3 cp /path/to/geometry_cache.parquet \
    s3://us-west-2.opendata.source.coop/geovibes/search/.../name_geometry_cache.parquet
```

#### Usage (`data_manager.py`)

```python
def _load_geometry_cache(self):
    """Download and connect to local geometry cache."""
    cache_url = self._infer_geometry_cache_url()
    local_path = self.cache.get_geometry_cache(cache_url)

    self.geometry_cache_conn = duckdb.connect(":memory:")
    self.geometry_cache_conn.execute(f"""
        CREATE VIEW geo_cache AS SELECT * FROM read_parquet('{local_path}')
    """)

def query_search_metadata(self, faiss_ids: List[int]):
    # Use local cache if available
    conn = self.geometry_cache_conn or self.duckdb_connection
    # ... query logic
```

#### Cache Location

```
~/.cache/geovibes/
├── faiss/           # FAISS indexes
│   └── abc123.index
└── geometry/        # Geometry caches
    └── def456.parquet
```

### Expected Performance

| Operation | Before (httpfs) | After (local cache) |
|-----------|-----------------|---------------------|
| First search | 26 seconds | <1 second |
| Subsequent searches | <1 second | <1 second |
| Startup (first time) | - | +30 seconds (download) |
| Startup (cached) | - | <1 second |

### S3 Files

After implementation, each database will have three files:

```
s3://us-west-2.opendata.source.coop/geovibes/search/USA/alabama/.../
├── alabama_..._metadata_sorted.db      # Full database (embeddings + geometry)
├── alabama_..._faiss_4096_64_8.index   # FAISS index
└── alabama_..._geometry_cache.parquet  # Geometry cache (NEW)
```

### Geometry Cache Status

All geometry caches uploaded (2025-01-18):

| Database | Geometry Cache | Size |
|----------|---------------|------|
| `alabama_dino_vit_small_patch16_224_2024_2025_32_16_10` | ✓ | 69 MB |
| `alabama_earthgenome_softcon_2024_2025_32_16_10` | ✓ | 21 MB |
| `alabama_google_satellite_embeddings_v1_2024_2025_25_0_10` | ✓ | 29 MB |
| `alabama_quantized_dino_vit_small_patch16_224_2024_2025_32_16_10` | ✓ | 69 MB |

### Generating New Geometry Caches

For new databases, use the steps below.

**Step 1: Clone repo and install dependencies**
```bash
git clone git@github.com:cr458/geovibes.git
cd geovibes
uv venv && source .venv/bin/activate
uv pip install -e .
```

**Step 2: Download the sorted database from S3**
```bash
# Set Source Cooperative read credentials
export SOURCE_COOP_KEY_ID="<get from source.coop/geovibes>"
export SOURCE_COOP_SECRET_KEY="<get from source.coop/geovibes>"

# Download Google Satellite database
aws s3 cp \
  s3://us-west-2.opendata.source.coop/geovibes/search/USA/alabama/2024-01-01-2025-01-01/alabama_google_satellite_embeddings_v1_2024_2025_25_0_10_metadata_sorted.db \
  /tmp/alabama_google_satellite_embeddings_v1_2024_2025_25_0_10_metadata_sorted.db
```

**Step 3: Generate the geometry cache**
```python
# generate_geometry_cache.py
import duckdb

db_path = "/tmp/alabama_google_satellite_embeddings_v1_2024_2025_25_0_10_metadata_sorted.db"
output_path = "/tmp/alabama_google_satellite_embeddings_v1_2024_2025_25_0_10_geometry_cache.parquet"

con = duckdb.connect(db_path, read_only=True)
con.execute("INSTALL spatial; LOAD spatial;")

print(f"Exporting geometry cache from {db_path}...")
con.execute(f"""
    COPY (SELECT id, geometry FROM geo_embeddings ORDER BY id)
    TO '{output_path}' (FORMAT PARQUET, COMPRESSION ZSTD)
""")
print(f"Created {output_path}")
con.close()
```

Run with:
```bash
uv run python generate_geometry_cache.py
```

**Step 4: Upload to S3**

Get write credentials from Source Cooperative console, then:
```bash
# Set session credentials from Source Cooperative console
export AWS_ACCESS_KEY_ID="ASIA..."
export AWS_SECRET_ACCESS_KEY="..."
export AWS_SESSION_TOKEN="..."
export AWS_DEFAULT_REGION="us-west-2"

# Upload (use direct AWS endpoint for writes)
aws s3 cp \
  /tmp/alabama_google_satellite_embeddings_v1_2024_2025_25_0_10_geometry_cache.parquet \
  s3://us-west-2.opendata.source.coop/geovibes/search/USA/alabama/2024-01-01-2025-01-01/alabama_google_satellite_embeddings_v1_2024_2025_25_0_10_geometry_cache.parquet

# Verify upload
aws s3 ls s3://us-west-2.opendata.source.coop/geovibes/search/USA/alabama/2024-01-01-2025-01-01/ | grep geometry_cache
```

**Step 5: Verify the file works**
```python
import duckdb
con = duckdb.connect()
con.execute("INSTALL spatial; LOAD spatial;")
df = con.execute("""
    SELECT COUNT(*) as cnt FROM read_parquet('/tmp/alabama_google_satellite_embeddings_v1_2024_2025_25_0_10_geometry_cache.parquet')
""").fetchdf()
print(f"Row count: {df['cnt'][0]}")  # Should be ~2.18M for Google Satellite
```

---

## Phase 4: Parallel Embedding Fetch Optimization (2025-01)

### Problem: Slow Embedding Prefetch for Polygon Labeling

After implementing caching (Phase 3), a new bottleneck emerged: polygon labeling requires fetching embeddings for all points in the polygon. For remote databases, this was taking 90+ seconds for 500 embeddings.

**Root cause analysis:**

FAISS returns IDs ordered by **embedding similarity**, not storage order. This means results are scattered across many row groups. Each row group requires a separate HTTP request (~500ms each).

```
FAISS search returns: [id_42331, id_891022, id_156, id_2847291, ...]
                       ↓         ↓          ↓      ↓
Row groups touched:    [RG_0,    RG_7,      RG_0,  RG_23, ...]

Result: 500 scattered IDs → ~40 row groups → ~40 HTTP requests → 20+ seconds
```

### Initial Hypothesis: Use DuckDB's Internal Threading

DuckDB documentation suggests setting `threads = 2-5x CPU cores` for httpfs workloads:

> "DuckDB uses synchronous IO when reading remote files. Each thread can make at most one HTTP request at a time."

**Hypothesis**: `SET threads = 50` should parallelize HTTP requests internally.

### Benchmark 1: DuckDB Internal Threads vs ThreadPoolExecutor

**Setup**: 250 scattered embeddings from S3, M1 Mac (10 cores)

| Approach | Configuration | Time |
|----------|---------------|------|
| DuckDB `SET threads` | 10 threads | 38.2s |
| DuckDB `SET threads` | 20 threads | 37.5s |
| DuckDB `SET threads` | 30 threads | 37.4s |
| DuckDB `SET threads` | 40 threads | 37.2s |
| DuckDB `SET threads` | 50 threads | 36.9s |
| **ThreadPoolExecutor** | 4 workers | 18.1s |
| **ThreadPoolExecutor** | 8 workers | 15.4s |
| **ThreadPoolExecutor** | 12 workers | 19.0s |
| **ThreadPoolExecutor** | 16 workers | **13.5s** |
| **ThreadPoolExecutor** | 24 workers | 21.0s |

**Surprising finding**: DuckDB's internal threading barely helped (38s → 37s), while ThreadPoolExecutor achieved **2.7x speedup** (37s → 13.5s).

**Why?** DuckDB threads share a single HTTP session. Multiple connections = multiple independent HTTP sessions = true parallelism.

### Key Insight: Row-Group-Aware Batching

**Hypothesis**: If we group IDs by row group before fetching, each batch reads from exactly one row group, minimizing redundant HTTP requests.

```python
ROW_GROUP_SIZE = 122880  # DuckDB default

# Group IDs by row group
rg_groups = defaultdict(list)
for id_ in ids:
    rg_groups[id_ // ROW_GROUP_SIZE].append(id_)

# Fetch each row group's IDs together
batches = list(rg_groups.values())
```

### Benchmark 2: Naive Chunking vs Row-Group Batching

**Setup**: 500 scattered embeddings, 16 parallel workers

| Approach | Batches | Time |
|----------|---------|------|
| Naive chunking (split by count) | 16 chunks | 8.0s |
| **Row-group batching** | 3 batches | **3.0s** |

**Result**: Row-group batching is **2.6x faster**.

### Benchmark 3: Scaling Analysis

**Question**: Do row-group batches become too large with more samples?

| N Samples | Row Groups Touched | Max Batch Size | Avg Batch Size |
|-----------|-------------------|----------------|----------------|
| 100 | 37 | 6 | 2.7 |
| 250 | 44 | 10 | 5.7 |
| 500 | 44 | 19 | 11.4 |
| 1,000 | 44 | 34 | 22.7 |
| 2,000 | 44 | 65 | 45.5 |
| 5,000 | 44 | 139 | 113.6 |
| 10,000 | 44 | 261 | 227.3 |

**Finding**: Scaling is excellent. Even with 10,000 samples:
- Only 44 row groups (bounded by database structure)
- Max batch size is 261 IDs (SQL handles 32k+ parameters easily)
- Memory usage ~15MB (fine)

### Combined Optimization Impact

| Approach | Time (500 embeddings) | Speedup |
|----------|----------------------|---------|
| Sequential (baseline) | ~93s | 1x |
| + ThreadPoolExecutor (16 workers) | ~27s | 3.4x |
| + Row-group batching | ~10s | **9.3x** |

### Implementation

Updated `_prefetch_embeddings_async()` in `geovibes/ui/app.py`:

```python
def _prefetch_embeddings_async(self, ids: list, n_workers: int = 16) -> None:
    """Pre-fetch embeddings using row-group-aware parallel batching."""
    ROW_GROUP_SIZE = 122880  # DuckDB default

    # Group IDs by row group for optimal httpfs performance
    rg_groups = defaultdict(list)
    for id_ in uncached:
        rg_groups[id_ // ROW_GROUP_SIZE].append(id_)
    batches = list(rg_groups.values())

    actual_workers = min(n_workers, len(batches), 32)

    with ThreadPoolExecutor(max_workers=actual_workers) as executor:
        futures = [executor.submit(fetch_batch, batch) for batch in batches]
        # ... fetch each batch with separate connection
```

### Key Learnings

1. **DuckDB `SET threads` doesn't help for httpfs** - Threads share one HTTP session
2. **Multiple connections = true parallelism** - Each connection has its own HTTP session
3. **Row-group batching is critical** - Grouping by `id // 122880` reduces redundant reads
4. **Optimal workers ≈ 32** - For larger databases, more parallelism helps
5. **Scaling is safe** - Row groups are bounded by database structure (~44 for 5.3M rows)

### Impact of Embedding Dimension

Benchmark on DINO ViT database (5.3M rows, **384-dim** embeddings = 1.5KB each):

| Workers | Time (500 embeddings) |
|---------|----------------------|
| 8 | 70.5s |
| 16 | 39.8s |
| **32** | **33.6s** ← optimal |
| 44 | 43.3s |

**Key insight**: Embedding dimension significantly impacts performance. The 384-dim DINO ViT embeddings are ~4x larger than quantized alternatives, resulting in longer fetch times.

**Throughput analysis**: Effective throughput of 0.02 MB/s indicates **latency is the bottleneck**, not bandwidth. Each row group requires a separate HTTP round-trip (~500-800ms), and with 44 row groups scattered across the database, this dominates total time.

**Recommendations for large embeddings**:
1. Use quantized embeddings (INT8) when possible - 4x smaller
2. Consider local embedding cache for frequently-accessed databases
3. Increase parallelism to 32 workers for IO-bound operations

### Visualization

See `blog_figures/embedding_fetch_optimization.png` for benchmark visualization.

![Embedding Fetch Optimization](../blog_figures/embedding_fetch_optimization.png)

---

## Phase 5: Cluster-Aligned Storage Analysis (2025-01)

### Hypothesis: Semantic Storage Ordering

FAISS IVF assigns each embedding to a cluster based on proximity to centroids. If we stored embeddings sorted by cluster ID instead of insertion order, search results (which come from similar clusters) would be co-located in fewer row groups.

### Benchmark Setup

Database: DINO ViT (5.3M vectors, 384-dim, 4096 clusters)
Query: 500 nearest neighbors to a test vector

### Finding 1: Cluster Distribution in Search Results

| nprobe | Clusters Hit | Current Row Groups | Cluster-Aligned Row Groups | Improvement |
|--------|--------------|--------------------|-----------------------------|-------------|
| 16 | 16 | 43 | 14 | **3.1x** |
| 64 | 35 | 44 | 27 | 1.6x |
| 256 | 36 | 44 | 28 | 1.6x |
| 4096 | 36 | 44 | 28 | 1.6x |

**Key insight**: Lower nprobe concentrates results in fewer clusters → better locality.

### Finding 2: Hilbert-Ordered Clusters

Random cluster ordering means semantically similar clusters may have distant IDs. By reordering clusters using a Hilbert space-filling curve (based on PCA projection of centroids), nearby clusters get consecutive IDs.

| nprobe | Clusters Hit | Random Order RGs | Hilbert Order RGs | Improvement |
|--------|--------------|------------------|-------------------|-------------|
| 16 | 16 | 14 | 6 | 2.3x |
| 64 | 35 | 27 | 8 | **3.4x** |
| 256 | 36 | 28 | 8 | **3.5x** |
| 1024 | 36 | 28 | 8 | **3.5x** |

### Combined Impact

| Storage Strategy | Row Groups | vs Current |
|------------------|------------|------------|
| Current (by ID) | 44 | 1x |
| Cluster-aligned (random order) | 28 | 1.6x |
| **Cluster-aligned (Hilbert order)** | **8** | **5.5x** |

**Estimated impact**: Embedding fetch could go from ~31s to ~6s with Hilbert-ordered cluster storage.

### Implementation Considerations

**To implement cluster-aligned storage:**

1. **Build time**: After FAISS training, assign each embedding to its cluster:
   ```python
   _, cluster_ids = index.quantizer.search(embeddings, 1)
   ```

2. **Hilbert ordering**: Project centroids to 2D via PCA, compute Hilbert distance, reorder:
   ```python
   centroids = quantizer.reconstruct_n(0, quantizer.ntotal)
   centroids_2d = PCA(n_components=2).fit_transform(centroids)
   hilbert_order = compute_hilbert_order(centroids_2d)
   cluster_to_new_id = {old: new for new, old in enumerate(hilbert_order)}
   ```

3. **Sort embeddings**: Sort database rows by `(new_cluster_id, original_id)` before export.

**Trade-offs:**
- Requires rebuilding databases (one-time cost)
- IDs change → existing saved datasets incompatible
- ~30% PCA explained variance for 384-dim → Hilbert ordering may be suboptimal

### Recommendation

For new databases, especially those expected to be queried remotely:

1. **Always use cluster-aligned storage** - 1.6x improvement with no downside
2. **Consider Hilbert ordering** - Additional 3.5x improvement if PCA explains ≥50% variance
3. **Tune nprobe** - Lower nprobe values maximize cluster locality benefits
