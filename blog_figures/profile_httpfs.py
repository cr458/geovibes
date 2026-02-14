#!/usr/bin/env python3
"""Profile httpfs performance for blog post visualizations.

This script measures the performance characteristics of remote DuckDB access
via httpfs, demonstrating why warmup and local geometry caching are necessary.

Usage:
    python blog_figures/profile_httpfs.py

Output:
    - blog_figures/profile_results.json (raw timing data)
    - Console output with timing breakdowns
"""

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import duckdb
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# S3 URLs for testing (httpfs folder structure)
S3_DB_URL = "s3://us-west-2.opendata.source.coop/geovibes/search/USA/alabama/2024-01-01-2025-01-01/httpfs/alabama_google_satellite_embeddings_v1_2024_2025_25_0_10/metadata.db"
S3_GEOMETRY_CACHE_URL = "s3://us-west-2.opendata.source.coop/geovibes/search/USA/alabama/2024-01-01-2025-01-01/httpfs/alabama_google_satellite_embeddings_v1_2024_2025_25_0_10/geometry_cache.parquet"

# Local cache path
LOCAL_GEOMETRY_CACHE = (
    Path.home() / ".cache" / "geovibes" / "geometry" / "test_geometry_cache.parquet"
)


@dataclass
class TimingResult:
    """Single timing measurement."""

    name: str
    duration_ms: float
    description: str = ""


@dataclass
class ProfileResults:
    """Complete profiling results."""

    # Connection timings
    cold_attach_ms: float = 0.0
    extension_load_ms: float = 0.0
    warmup_query_ms: float = 0.0

    # Query timings (after warmup)
    warm_count_query_ms: float = 0.0
    warm_single_id_fetch_ms: float = 0.0

    # Scattered ID fetch (simulating FAISS results)
    scattered_100_ids_ms: float = 0.0
    scattered_500_ids_ms: float = 0.0
    scattered_1000_ids_ms: float = 0.0

    # Local geometry cache comparison
    local_cache_100_ids_ms: float = 0.0
    local_cache_500_ids_ms: float = 0.0
    local_cache_1000_ids_ms: float = 0.0

    # Metadata
    row_count: int = 0
    db_size_mb: float = 0.0
    geometry_cache_size_mb: float = 0.0

    # Raw runs for statistics
    scattered_runs: dict = field(default_factory=dict)
    local_cache_runs: dict = field(default_factory=dict)


def setup_connection() -> duckdb.DuckDBPyConnection:
    """Set up a fresh DuckDB connection with extensions."""
    conn = duckdb.connect()
    conn.execute("INSTALL httpfs; LOAD httpfs;")
    conn.execute("INSTALL spatial; LOAD spatial;")
    conn.execute("INSTALL aws; LOAD aws;")
    conn.execute("CALL load_aws_credentials();")
    return conn


def time_operation(func, *args, **kwargs) -> tuple[float, any]:
    """Time an operation and return (duration_ms, result)."""
    start = time.perf_counter()
    result = func(*args, **kwargs)
    duration = (time.perf_counter() - start) * 1000
    return duration, result


def generate_scattered_ids(
    conn: duckdb.DuckDBPyConnection, n: int, max_id: int
) -> list[int]:
    """Generate scattered IDs that simulate FAISS search results.

    FAISS returns IDs by embedding similarity, not storage order,
    so results are scattered across many row groups.
    """
    # Get random IDs across the full range
    np.random.seed(42)  # Reproducible
    return sorted(np.random.randint(1, max_id + 1, size=n).tolist())


def profile_cold_start(results: ProfileResults) -> duckdb.DuckDBPyConnection:
    """Profile cold start - connecting to remote database."""
    logger.info("\n" + "=" * 60)
    logger.info("COLD START PROFILING")
    logger.info("=" * 60)

    # Time extension loading
    start = time.perf_counter()
    conn = duckdb.connect()
    conn.execute("INSTALL httpfs; LOAD httpfs;")
    conn.execute("INSTALL spatial; LOAD spatial;")
    conn.execute("INSTALL aws; LOAD aws;")
    conn.execute("CALL load_aws_credentials();")
    results.extension_load_ms = (time.perf_counter() - start) * 1000
    logger.info(f"  Extension load: {results.extension_load_ms:,.0f} ms")

    # Time database attach
    start = time.perf_counter()
    conn.execute(f"ATTACH '{S3_DB_URL}' AS remote (READ_ONLY)")
    conn.execute("USE remote")
    results.cold_attach_ms = (time.perf_counter() - start) * 1000
    logger.info(f"  Database attach: {results.cold_attach_ms:,.0f} ms")

    # Time warmup query (loads table metadata + first row group)
    start = time.perf_counter()
    conn.execute("SELECT id, geometry FROM geo_embeddings LIMIT 1").fetchone()
    results.warmup_query_ms = (time.perf_counter() - start) * 1000
    logger.info(f"  Warmup query: {results.warmup_query_ms:,.0f} ms")

    total = results.extension_load_ms + results.cold_attach_ms + results.warmup_query_ms
    logger.info(f"  TOTAL cold start: {total:,.0f} ms")

    return conn


def profile_warm_queries(conn: duckdb.DuckDBPyConnection, results: ProfileResults):
    """Profile queries after warmup."""
    logger.info("\n" + "=" * 60)
    logger.info("WARM QUERY PROFILING")
    logger.info("=" * 60)

    # Get row count
    duration, df = time_operation(
        lambda: conn.execute("SELECT COUNT(*) as cnt FROM geo_embeddings").fetchdf()
    )
    results.row_count = int(df["cnt"].iloc[0])
    results.warm_count_query_ms = duration
    logger.info(f"  COUNT(*) query: {duration:,.1f} ms ({results.row_count:,} rows)")

    # Single ID fetch
    duration, _ = time_operation(
        lambda: conn.execute(
            "SELECT id, ST_AsGeoJSON(geometry) FROM geo_embeddings WHERE id = 1000"
        ).fetchdf()
    )
    results.warm_single_id_fetch_ms = duration
    logger.info(f"  Single ID fetch: {duration:,.1f} ms")


def profile_scattered_id_fetch(
    conn: duckdb.DuckDBPyConnection, results: ProfileResults
):
    """Profile fetching scattered IDs (simulating FAISS results)."""
    logger.info("\n" + "=" * 60)
    logger.info("SCATTERED ID FETCH (simulating FAISS results)")
    logger.info("=" * 60)
    logger.info("  FAISS returns IDs by similarity, not storage order.")
    logger.info("  Results span many row groups, each requiring HTTP fetch.")

    for n_ids in [100, 500, 1000]:
        ids = generate_scattered_ids(conn, n_ids, results.row_count)

        # Run multiple times for statistics
        runs = []
        for run in range(3):
            # Create fresh connection to avoid caching effects
            if run > 0:
                conn.close()
                conn = setup_connection()
                conn.execute(f"ATTACH '{S3_DB_URL}' AS remote (READ_ONLY)")
                conn.execute("USE remote")
                # Warmup
                conn.execute("SELECT id FROM geo_embeddings LIMIT 1").fetchone()

            placeholders = ", ".join(["?" for _ in ids])
            sql = f"SELECT id, ST_AsGeoJSON(geometry) FROM geo_embeddings WHERE id IN ({placeholders})"

            duration, df = time_operation(lambda: conn.execute(sql, ids).fetchdf())
            runs.append(duration)
            logger.info(f"  {n_ids} scattered IDs (run {run+1}): {duration:,.0f} ms")

        mean_ms = np.mean(runs)
        results.scattered_runs[n_ids] = runs

        if n_ids == 100:
            results.scattered_100_ids_ms = mean_ms
        elif n_ids == 500:
            results.scattered_500_ids_ms = mean_ms
        elif n_ids == 1000:
            results.scattered_1000_ids_ms = mean_ms

        logger.info(f"  → Mean: {mean_ms:,.0f} ms")

    return conn


def download_geometry_cache():
    """Download geometry cache if not present."""
    if LOCAL_GEOMETRY_CACHE.exists():
        logger.info(f"  Using cached geometry file: {LOCAL_GEOMETRY_CACHE}")
        return

    logger.info("  Downloading geometry cache from S3...")
    LOCAL_GEOMETRY_CACHE.parent.mkdir(parents=True, exist_ok=True)

    import fsspec

    with fsspec.open(S3_GEOMETRY_CACHE_URL, "rb") as src:
        with open(LOCAL_GEOMETRY_CACHE, "wb") as dst:
            dst.write(src.read())

    logger.info(f"  Downloaded to: {LOCAL_GEOMETRY_CACHE}")


def profile_local_geometry_cache(results: ProfileResults):
    """Profile fetching from local geometry cache."""
    logger.info("\n" + "=" * 60)
    logger.info("LOCAL GEOMETRY CACHE PROFILING")
    logger.info("=" * 60)

    download_geometry_cache()

    # Get cache file size
    results.geometry_cache_size_mb = LOCAL_GEOMETRY_CACHE.stat().st_size / (1024 * 1024)
    logger.info(f"  Cache size: {results.geometry_cache_size_mb:.1f} MB")

    # Set up local DuckDB with geometry cache
    conn = duckdb.connect(":memory:")
    conn.execute("INSTALL spatial; LOAD spatial;")
    conn.execute(
        f"CREATE VIEW geometry_cache AS SELECT * FROM '{LOCAL_GEOMETRY_CACHE}'"
    )

    for n_ids in [100, 500, 1000]:
        ids = generate_scattered_ids(conn, n_ids, results.row_count)

        runs = []
        for run in range(5):
            placeholders = ", ".join(["?" for _ in ids])
            sql = f"SELECT id, ST_AsGeoJSON(geometry) FROM geometry_cache WHERE id IN ({placeholders})"

            duration, df = time_operation(lambda: conn.execute(sql, ids).fetchdf())
            runs.append(duration)

        mean_ms = np.mean(runs)
        results.local_cache_runs[n_ids] = runs

        if n_ids == 100:
            results.local_cache_100_ids_ms = mean_ms
        elif n_ids == 500:
            results.local_cache_500_ids_ms = mean_ms
        elif n_ids == 1000:
            results.local_cache_1000_ids_ms = mean_ms

        logger.info(
            f"  {n_ids} IDs from local cache: {mean_ms:,.1f} ms (mean of 5 runs)"
        )

    conn.close()


def print_summary(results: ProfileResults):
    """Print summary comparison."""
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY: WHY LOCAL CACHING MATTERS")
    logger.info("=" * 60)

    logger.info("\n  Cold Start Breakdown:")
    logger.info(f"    Extension load:   {results.extension_load_ms:>8,.0f} ms")
    logger.info(f"    Database attach:  {results.cold_attach_ms:>8,.0f} ms")
    logger.info(f"    Warmup query:     {results.warmup_query_ms:>8,.0f} ms")
    total = results.extension_load_ms + results.cold_attach_ms + results.warmup_query_ms
    logger.info("    ─────────────────────────────")
    logger.info(f"    TOTAL:            {total:>8,.0f} ms")

    logger.info("\n  Scattered ID Fetch (FAISS results) - httpfs vs Local Cache:")
    logger.info(f"    {'IDs':<8} {'httpfs':>12} {'Local Cache':>12} {'Speedup':>10}")
    logger.info(f"    {'─'*8} {'─'*12} {'─'*12} {'─'*10}")

    for n_ids in [100, 500, 1000]:
        httpfs_ms = getattr(results, f"scattered_{n_ids}_ids_ms")
        local_ms = getattr(results, f"local_cache_{n_ids}_ids_ms")
        speedup = httpfs_ms / local_ms if local_ms > 0 else 0
        logger.info(
            f"    {n_ids:<8} {httpfs_ms:>10,.0f}ms {local_ms:>10,.1f}ms {speedup:>9.0f}x"
        )


def main():
    """Run profiling and save results."""
    logger.info("=" * 60)
    logger.info("httpfs Performance Profiling for Blog Post")
    logger.info("=" * 60)
    logger.info(f"Database: {S3_DB_URL.split('/')[-1]}")

    results = ProfileResults()

    # Profile cold start
    conn = profile_cold_start(results)

    # Profile warm queries
    profile_warm_queries(conn, results)

    # Profile scattered ID fetch (the bottleneck)
    conn = profile_scattered_id_fetch(conn, results)
    conn.close()

    # Profile local geometry cache (the solution)
    profile_local_geometry_cache(results)

    # Print summary
    print_summary(results)

    # Save results
    output_path = Path(__file__).parent / "profile_results.json"
    with open(output_path, "w") as f:
        json.dump(asdict(results), f, indent=2)
    logger.info(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
