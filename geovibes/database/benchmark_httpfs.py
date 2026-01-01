"""Benchmark DuckDB queries over httpfs vs local access.

Usage:
    # Test remote S3 database
    python -m geovibes.database.benchmark_httpfs \
        --s3-url "s3://bucket/path/database.duckdb" \
        --local-path "local_databases/database.duckdb"

    # Test with specific coordinates (default: Alabama)
    python -m geovibes.database.benchmark_httpfs \
        --s3-url "s3://bucket/db.duckdb" \
        --lon -86.5 --lat 32.5
"""

import argparse
import json
import time
from pathlib import Path

import duckdb
import numpy as np


def setup_remote_connection(
    s3_url: str, endpoint_url: str = None
) -> duckdb.DuckDBPyConnection:
    """Set up DuckDB connection to S3 via httpfs."""
    conn = duckdb.connect()
    conn.execute("INSTALL httpfs; LOAD httpfs;")
    conn.execute("INSTALL spatial; LOAD spatial;")
    conn.execute("INSTALL aws; LOAD aws;")
    conn.execute("CALL load_aws_credentials();")

    if endpoint_url:
        conn.execute(f"SET s3_endpoint='{endpoint_url}';")
        conn.execute("SET s3_url_style='path';")

    conn.execute(f"ATTACH '{s3_url}' AS remote (READ_ONLY)")
    conn.execute("USE remote")
    return conn


def setup_local_connection(db_path: str) -> duckdb.DuckDBPyConnection:
    """Set up local DuckDB connection."""
    conn = duckdb.connect(db_path, read_only=True)
    conn.execute("LOAD spatial;")
    return conn


def benchmark_cold_start(s3_url: str, endpoint_url: str = None) -> float:
    """Measure time to connect and run first query (cold start)."""
    start = time.perf_counter()
    conn = setup_remote_connection(s3_url, endpoint_url)
    conn.execute("SELECT COUNT(*) FROM geo_embeddings").fetchone()
    elapsed_ms = (time.perf_counter() - start) * 1000
    conn.close()
    return elapsed_ms


def benchmark_nearest_point(
    conn: duckdb.DuckDBPyConnection, lon: float, lat: float, n_runs: int = 10
) -> list[float]:
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


def benchmark_fetch_by_id(
    conn: duckdb.DuckDBPyConnection, n_ids: int, n_runs: int = 5
) -> list[float]:
    """Benchmark fetching rows by ID."""
    all_ids = (
        conn.execute(f"SELECT id FROM geo_embeddings LIMIT {n_ids}")
        .fetchdf()["id"]
        .tolist()
    )
    placeholders = ",".join(["?" for _ in all_ids])
    sql = f"SELECT id, ST_AsGeoJSON(geometry) FROM geo_embeddings WHERE id IN ({placeholders})"

    latencies = []
    for _ in range(n_runs):
        start = time.perf_counter()
        conn.execute(sql, all_ids).fetchdf()
        latencies.append((time.perf_counter() - start) * 1000)
    return latencies


def run_full_benchmark(
    s3_url: str,
    local_path: str = None,
    lon: float = -86.5,
    lat: float = 32.5,
    endpoint_url: str = None,
) -> dict:
    """Run complete benchmark suite comparing remote vs local."""
    results = {
        "s3_url": s3_url,
        "local_path": str(local_path) if local_path else None,
        "coordinates": {"lon": lon, "lat": lat},
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    # Cold start
    print("Testing cold start...")
    results["cold_start_ms"] = benchmark_cold_start(s3_url, endpoint_url)
    print(f"  Cold start: {results['cold_start_ms']:.0f} ms")

    # Remote benchmarks
    print("Connecting to remote DB...")
    remote_conn = setup_remote_connection(s3_url, endpoint_url)

    # Get table stats
    row_count = remote_conn.execute("SELECT COUNT(*) FROM geo_embeddings").fetchone()[0]
    results["row_count"] = row_count
    print(f"  Table has {row_count:,} rows")

    print("Testing nearest point (remote)...")
    remote_nearest = benchmark_nearest_point(remote_conn, lon, lat)
    results["remote_nearest_point"] = {
        "mean_ms": float(np.mean(remote_nearest)),
        "std_ms": float(np.std(remote_nearest)),
        "min_ms": float(np.min(remote_nearest)),
        "max_ms": float(np.max(remote_nearest)),
    }
    print(
        f"  Nearest point: {results['remote_nearest_point']['mean_ms']:.0f} ms "
        f"(±{results['remote_nearest_point']['std_ms']:.0f})"
    )

    print("Testing fetch by ID (remote)...")
    results["remote_fetch_by_id"] = {}
    for n in [10, 50, 100, 200, 500, 1000]:
        if n > row_count:
            break
        latencies = benchmark_fetch_by_id(remote_conn, n)
        results["remote_fetch_by_id"][n] = {
            "mean_ms": float(np.mean(latencies)),
            "std_ms": float(np.std(latencies)),
        }
        print(f"  Fetch {n}: {results['remote_fetch_by_id'][n]['mean_ms']:.0f} ms")

    remote_conn.close()

    # Local baseline
    if local_path and Path(local_path).exists():
        print("Connecting to local DB...")
        local_conn = setup_local_connection(str(local_path))

        print("Testing nearest point (local)...")
        local_nearest = benchmark_nearest_point(local_conn, lon, lat)
        results["local_nearest_point"] = {"mean_ms": float(np.mean(local_nearest))}
        print(
            f"  Local nearest point: {results['local_nearest_point']['mean_ms']:.1f} ms"
        )

        print("Testing fetch by ID (local)...")
        local_fetch = benchmark_fetch_by_id(local_conn, 100)
        results["local_fetch_100"] = {"mean_ms": float(np.mean(local_fetch))}
        print(f"  Local fetch 100: {results['local_fetch_100']['mean_ms']:.1f} ms")

        local_conn.close()

        # Calculate ratios
        results["ratio_nearest"] = (
            results["remote_nearest_point"]["mean_ms"]
            / results["local_nearest_point"]["mean_ms"]
        )
        if 100 in results["remote_fetch_by_id"]:
            results["ratio_fetch_100"] = (
                results["remote_fetch_by_id"][100]["mean_ms"]
                / results["local_fetch_100"]["mean_ms"]
            )

    return results


def check_acceptance_criteria(results: dict) -> tuple[bool, list[str]]:
    """Check if results meet acceptance criteria from docs/httpfs.md."""
    failures = []

    if results["cold_start_ms"] > 5000:
        failures.append(
            f"Cold start {results['cold_start_ms']:.0f}ms > 5000ms threshold"
        )

    if results["remote_nearest_point"]["mean_ms"] > 500:
        failures.append(
            f"Nearest point {results['remote_nearest_point']['mean_ms']:.0f}ms > 500ms threshold"
        )

    if (
        100 in results["remote_fetch_by_id"]
        and results["remote_fetch_by_id"][100]["mean_ms"] > 1000
    ):
        failures.append(
            f"Fetch 100 {results['remote_fetch_by_id'][100]['mean_ms']:.0f}ms > 1000ms threshold"
        )

    if (
        500 in results["remote_fetch_by_id"]
        and results["remote_fetch_by_id"][500]["mean_ms"] > 3000
    ):
        failures.append(
            f"Fetch 500 {results['remote_fetch_by_id'][500]['mean_ms']:.0f}ms > 3000ms threshold"
        )

    if "ratio_nearest" in results and results["ratio_nearest"] > 10:
        failures.append(
            f"Remote/local ratio {results['ratio_nearest']:.1f}x > 10x threshold"
        )

    return len(failures) == 0, failures


def print_report(results: dict, passed: bool, failures: list[str]):
    """Print formatted benchmark report."""
    print("\n" + "=" * 60)
    print("                  httpfs BENCHMARK RESULTS")
    print("=" * 60)
    print(f"\nDatabase: {results['s3_url']}")
    print(f"Rows: {results.get('row_count', 'N/A'):,}")
    print(
        f"Coordinates: ({results['coordinates']['lon']}, {results['coordinates']['lat']})"
    )
    print(f"Date: {results['timestamp']}")

    print("\n1. COLD START")
    print(f"   Connection + first query: {results['cold_start_ms']:.0f} ms")
    print(
        f"   {'✓ PASS' if results['cold_start_ms'] <= 5000 else '✗ FAIL'} (threshold: 5000 ms)"
    )

    print("\n2. NEAREST POINT (spatial query)")
    np_result = results["remote_nearest_point"]
    print(f"   Mean: {np_result['mean_ms']:.0f} ms (std: {np_result['std_ms']:.0f} ms)")
    print(f"   Range: {np_result['min_ms']:.0f} - {np_result['max_ms']:.0f} ms")
    print(
        f"   {'✓ PASS' if np_result['mean_ms'] <= 500 else '✗ FAIL'} (threshold: 500 ms)"
    )

    print("\n3. FETCH GEOMETRIES BY ID")
    print("   | Count | Mean (ms) | Std (ms) | Status |")
    print("   |-------|-----------|----------|--------|")
    for n in [10, 50, 100, 200, 500, 1000]:
        if n not in results["remote_fetch_by_id"]:
            continue
        fetch = results["remote_fetch_by_id"][n]
        threshold = 1000 if n == 100 else (3000 if n == 500 else None)
        status = ""
        if threshold:
            status = "✓" if fetch["mean_ms"] <= threshold else "✗"
        print(
            f"   | {n:5} | {fetch['mean_ms']:9.0f} | {fetch['std_ms']:8.0f} | {status:6} |"
        )

    if "local_nearest_point" in results:
        print("\n4. LOCAL BASELINE")
        print(f"   Nearest point: {results['local_nearest_point']['mean_ms']:.1f} ms")
        if "local_fetch_100" in results:
            print(f"   Fetch 100: {results['local_fetch_100']['mean_ms']:.1f} ms")

        print("\n5. REMOTE/LOCAL RATIO")
        if "ratio_nearest" in results:
            ratio = results["ratio_nearest"]
            print(
                f"   Nearest point: {ratio:.1f}x {'✓' if ratio <= 10 else '✗'} (threshold: < 10x)"
            )
        if "ratio_fetch_100" in results:
            ratio = results["ratio_fetch_100"]
            print(
                f"   Fetch 100: {ratio:.1f}x {'✓' if ratio <= 10 else '✗'} (threshold: < 10x)"
            )

    print("\n" + "=" * 60)
    if passed:
        print("✓ ALL ACCEPTANCE CRITERIA PASSED")
        print("  Proceed to Phase 1 implementation")
    else:
        print("✗ ACCEPTANCE CRITERIA FAILED:")
        for failure in failures:
            print(f"  - {failure}")
        print("\n  Review failures before proceeding to Phase 1")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark DuckDB queries over httpfs vs local access",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--s3-url", required=True, help="S3 URL to DuckDB database")
    parser.add_argument("--local-path", help="Local database path for comparison")
    parser.add_argument(
        "--lon", type=float, default=-86.5, help="Longitude for spatial queries"
    )
    parser.add_argument(
        "--lat", type=float, default=32.5, help="Latitude for spatial queries"
    )
    parser.add_argument(
        "--endpoint-url", help="Custom S3 endpoint (e.g., for Source Cooperative)"
    )
    parser.add_argument(
        "--output", default="benchmark_results.json", help="Output JSON file"
    )
    args = parser.parse_args()

    results = run_full_benchmark(
        s3_url=args.s3_url,
        local_path=args.local_path,
        lon=args.lon,
        lat=args.lat,
        endpoint_url=args.endpoint_url,
    )

    passed, failures = check_acceptance_criteria(results)
    results["passed"] = passed
    results["failures"] = failures

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {args.output}")

    print_report(results, passed, failures)


if __name__ == "__main__":
    main()
