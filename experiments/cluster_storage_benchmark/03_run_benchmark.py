#!/usr/bin/env python3
"""
Step 3: Run embedding fetch benchmarks.

Benchmarks real-world access patterns:
- Search with FAISS -> get result IDs
- Fetch embeddings using different methods
- Measure time across multiple trials

Usage:
    uv run python 03_run_benchmark.py
"""

import os
import gc
import json
import time
import yaml
import numpy as np
import faiss
import duckdb
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

# Load config
config_dir = Path(__file__).parent
with open(config_dir / "config.yaml") as f:
    config = yaml.safe_load(f)

# Load S3 URLs (created by upload script)
s3_urls_path = config_dir / "s3_urls.yaml"
if s3_urls_path.exists():
    with open(s3_urls_path) as f:
        s3_urls = yaml.safe_load(f)
else:
    # Use config defaults for testing
    s3_urls = {
        "faiss_url": config["source"]["faiss_url"],
        "variants": {
            "baseline": config["source"]["db_url"],
        },
    }

ROW_GROUP_SIZE = config["duckdb"]["row_group_size"]
N_TRIALS = config["benchmark"]["n_trials"]
N_EMBEDDINGS_LIST = config["benchmark"]["n_embeddings"]
QUERY_IDS = config["benchmark"]["query_ids"]
METHODS = config["benchmark"]["methods"]
NPROBE = config["faiss"]["nprobe"]

# Results storage
results = {
    "metadata": {
        "timestamp": datetime.now().isoformat(),
        "config": config,
        "s3_urls": s3_urls,
    },
    "benchmarks": [],
}


def clear_caches():
    """Clear all caches to ensure fair comparison."""
    gc.collect()

    # Clear DuckDB's internal caches by creating fresh connections
    # Clear filesystem caches (requires sudo, skip if not available)
    try:
        os.system("sync; echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null 2>&1")
    except Exception:
        pass  # Skip if not Linux or no sudo


def setup_connection(db_url: str, n_threads: int = None) -> duckdb.DuckDBPyConnection:
    """Set up DuckDB connection for S3 database."""
    conn = duckdb.connect(":memory:")
    conn.execute("SET enable_progress_bar = false;")
    conn.execute("INSTALL httpfs; LOAD httpfs;")
    conn.execute("INSTALL spatial; LOAD spatial;")
    conn.execute("INSTALL aws; LOAD aws;")
    conn.execute("CALL load_aws_credentials();")

    if n_threads:
        conn.execute(f"SET threads = {n_threads};")

    conn.execute(f"ATTACH '{db_url}' AS remote_db (READ_ONLY)")
    conn.execute("CREATE VIEW geo_embeddings AS SELECT * FROM remote_db.geo_embeddings")

    return conn


def download_faiss_index():
    """Download FAISS index."""
    from geovibes.database.faiss_cache import FaissCache

    cache = FaissCache()
    return cache.get_index(s3_urls["faiss_url"], show_progress=True)


def search_faiss(index, query_embedding, n_results):
    """Search FAISS index and return result IDs."""
    params = faiss.SearchParametersIVF(nprobe=NPROBE)
    distances, ids = index.search(query_embedding, n_results, params=params)
    return ids[0].tolist(), distances[0].tolist()


def fetch_threadpool(db_url: str, ids: list, n_workers: int) -> tuple:
    """Fetch embeddings using ThreadPoolExecutor with row-group batching."""
    # Group by row group
    rg_groups = defaultdict(list)
    for id_ in ids:
        rg_groups[int(id_) // ROW_GROUP_SIZE].append(id_)
    batches = list(rg_groups.values())

    def fetch_batch(batch_ids):
        conn = setup_connection(db_url)
        placeholders = ",".join(["?" for _ in batch_ids])
        df = conn.execute(
            f"""
            SELECT id, CAST(embedding AS FLOAT[]) as embedding
            FROM geo_embeddings
            WHERE id IN ({placeholders})
        """,
            batch_ids,
        ).fetchdf()
        conn.close()
        return len(df)

    actual_workers = min(n_workers, len(batches))
    start = time.perf_counter()

    total = 0
    with ThreadPoolExecutor(max_workers=actual_workers) as executor:
        futures = [executor.submit(fetch_batch, batch) for batch in batches]
        for future in as_completed(futures):
            total += future.result()

    elapsed = time.perf_counter() - start
    return elapsed, total, len(batches)


def fetch_duckdb_native(db_url: str, ids: list, n_threads: int) -> tuple:
    """Fetch embeddings using DuckDB's native threading."""
    conn = setup_connection(db_url, n_threads=n_threads)

    placeholders = ",".join(["?" for _ in ids])
    start = time.perf_counter()

    df = conn.execute(
        f"""
        SELECT id, CAST(embedding AS FLOAT[]) as embedding
        FROM geo_embeddings
        WHERE id IN ({placeholders})
    """,
        ids,
    ).fetchdf()

    elapsed = time.perf_counter() - start
    conn.close()

    # Count row groups touched
    rg_set = set(int(id_) // ROW_GROUP_SIZE for id_ in ids)

    return elapsed, len(df), len(rg_set)


def run_single_benchmark(
    variant: str, db_url: str, method: dict, ids: list, trial: int
) -> dict:
    """Run a single benchmark trial."""
    clear_caches()
    time.sleep(1)  # Allow caches to clear

    if method["type"] == "threadpool":
        elapsed, fetched, batches = fetch_threadpool(db_url, ids, method["n_workers"])
    elif method["type"] == "duckdb_native":
        elapsed, fetched, batches = fetch_duckdb_native(
            db_url, ids, method["n_threads"]
        )
    else:
        raise ValueError(f"Unknown method type: {method['type']}")

    # Count row groups
    rg_set = set(int(id_) // ROW_GROUP_SIZE for id_ in ids)

    return {
        "variant": variant,
        "method": method["name"],
        "n_embeddings": len(ids),
        "trial": trial,
        "elapsed_seconds": elapsed,
        "fetched": fetched,
        "row_groups_touched": len(rg_set),
        "batches": batches,
    }


def main():
    print("=" * 70)
    print("Step 3: Run Embedding Fetch Benchmarks")
    print("=" * 70)

    # Load FAISS index
    print("\n[1/4] Loading FAISS index...")
    index = download_faiss_index()
    print(f"  Index: {index.ntotal:,} vectors")

    # Get query embeddings
    print("\n[2/4] Fetching query embeddings...")
    # Use baseline for getting query embeddings (doesn't matter which variant)
    baseline_url = s3_urls["variants"].get("baseline", config["source"]["db_url"])
    conn = setup_connection(baseline_url)

    query_embeddings = {}
    for qid in QUERY_IDS:
        row = conn.execute(f"""
            SELECT CAST(embedding AS FLOAT[]) as embedding
            FROM geo_embeddings WHERE id = {qid}
        """).fetchone()
        if row:
            query_embeddings[qid] = np.array(row[0], dtype=np.float32).reshape(1, -1)
            print(f"  Query {qid}: loaded")
    conn.close()

    # Run benchmarks
    print("\n[3/4] Running benchmarks...")
    print(f"  Variants: {list(s3_urls['variants'].keys())}")
    print(f"  Methods: {[m['name'] for m in METHODS]}")
    print(f"  Embedding counts: {N_EMBEDDINGS_LIST}")
    print(f"  Trials per config: {N_TRIALS}")

    total_runs = (
        len(s3_urls["variants"])
        * len(METHODS)
        * len(N_EMBEDDINGS_LIST)
        * len(QUERY_IDS)
        * N_TRIALS
    )
    print(f"  Total benchmark runs: {total_runs}")

    run_count = 0
    for variant, db_url in s3_urls["variants"].items():
        print(f"\n  === Variant: {variant} ===")

        for n_embeddings in N_EMBEDDINGS_LIST:
            for qid in QUERY_IDS:
                if qid not in query_embeddings:
                    continue

                # Search to get result IDs
                result_ids, _ = search_faiss(index, query_embeddings[qid], n_embeddings)

                for method in METHODS:
                    for trial in range(N_TRIALS):
                        run_count += 1
                        print(
                            f"\r    [{run_count}/{total_runs}] {variant}/{method['name']}/n={n_embeddings}/q={qid}/t={trial+1}",
                            end="",
                        )

                        result = run_single_benchmark(
                            variant, db_url, method, result_ids, trial + 1
                        )
                        result["query_id"] = qid
                        results["benchmarks"].append(result)

    print("\n")

    # Save results
    print("\n[4/4] Saving results...")
    results_dir = config_dir / "results"
    results_dir.mkdir(exist_ok=True)

    results_file = results_dir / "benchmark_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved to {results_file}")

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    # Aggregate results
    from collections import defaultdict

    summary = defaultdict(lambda: defaultdict(list))

    for r in results["benchmarks"]:
        key = (r["variant"], r["method"], r["n_embeddings"])
        summary[key]["times"].append(r["elapsed_seconds"])
        summary[key]["row_groups"].append(r["row_groups_touched"])

    print(
        f"\n{'Variant':<12} | {'Method':<18} | {'N':>6} | {'Time (s)':>12} | {'Std':>8} | {'RGs':>4}"
    )
    print("-" * 75)

    for (variant, method, n_emb), data in sorted(summary.items()):
        times = data["times"]
        rgs = data["row_groups"]
        mean_time = np.mean(times)
        std_time = np.std(times)
        mean_rg = np.mean(rgs)

        print(
            f"{variant:<12} | {method:<18} | {n_emb:>6} | {mean_time:>10.2f}s | {std_time:>7.2f} | {mean_rg:>4.0f}"
        )

    print("\nBenchmark complete!")


if __name__ == "__main__":
    main()
