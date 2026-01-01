#!/usr/bin/env python
"""Test the FULLY remote architecture: S3 FAISS (cached) + S3 DuckDB.

This demonstrates:
1. FaissCache downloads FAISS index from S3 with progress bar
2. RemoteDuckDB queries database directly on S3
3. Full search workflow without any local files
"""

import time

import numpy as np

from geovibes.database.faiss_cache import FaissCache
from geovibes.database.remote_db import RemoteDuckDB

# S3 URLs
S3_FAISS_URL = "s3://us-west-2.opendata.source.coop/geovibes/search/USA/alabama/2024-01-01-2025-01-01/alabama_google_satellite_embeddings_v1_2024_2025_25_0_10_faiss_4096_64_8.index"
S3_DB_URL = "s3://us-west-2.opendata.source.coop/geovibes/search/USA/alabama/2024-01-01-2025-01-01/alabama_google_satellite_embeddings_v1_2024_2025_25_0_10_metadata.db"


def main():
    print("=" * 60)
    print("Testing FULLY Remote Architecture")
    print("=" * 60)
    print(f"FAISS URL: {S3_FAISS_URL}")
    print(f"DB URL:    {S3_DB_URL}")

    # Step 1: Download/cache FAISS index from S3
    print("\n[1] Loading FAISS index from S3 (with caching)...")
    cache = FaissCache()

    # Check if already cached
    if cache.is_cached(S3_FAISS_URL):
        print("    ✓ Found in cache, loading...")
    else:
        print("    ↓ Downloading from S3 (first time only)...")

    start = time.perf_counter()
    index = cache.get_index(S3_FAISS_URL, show_progress=True)
    load_time = (time.perf_counter() - start) * 1000
    print(f"    ✓ Loaded {index.ntotal:,} vectors in {load_time:.0f}ms")
    print(f"    ✓ Dimension: {index.d}")
    print(f"    ✓ Cache location: {cache.cache_dir}")

    # Step 2: Connect to remote database
    print("\n[2] Connecting to remote DuckDB via httpfs...")
    db = RemoteDuckDB()
    start = time.perf_counter()
    db.connect(S3_DB_URL)
    connect_time = (time.perf_counter() - start) * 1000
    print(f"    ✓ Connected in {connect_time:.0f}ms")

    # Step 3: Get a sample embedding
    print("\n[3] Getting sample embedding from remote DB...")
    start = time.perf_counter()
    sample = db.query("""
        SELECT id, embedding
        FROM geo_embeddings
        LIMIT 1
    """)
    fetch_sample_time = (time.perf_counter() - start) * 1000
    print(f"    ✓ Fetched sample in {fetch_sample_time:.0f}ms")

    query_embedding = np.array(sample["embedding"].iloc[0], dtype=np.float32)
    query_id = sample["id"].iloc[0]
    print(f"    ✓ Query ID: {query_id}")

    # Step 4: Search FAISS locally (cached index)
    print("\n[4] Searching FAISS index (local cache)...")
    k = 100
    start = time.perf_counter()
    distances, indices = index.search(query_embedding.reshape(1, -1), k)
    search_time = (time.perf_counter() - start) * 1000
    print(f"    ✓ Found {k} neighbors in {search_time:.1f}ms")
    print(f"    ✓ Top 5 indices: {indices[0][:5].tolist()}")

    # Step 5: Fetch results from remote database
    print("\n[5] Fetching results from remote database...")
    result_ids = indices[0].tolist()
    start = time.perf_counter()
    results = db.fetch_by_ids(result_ids, columns=["id", "geometry"])
    fetch_time = (time.perf_counter() - start) * 1000
    print(f"    ✓ Fetched {len(results)} results in {fetch_time:.0f}ms")

    # Step 7: Second search to show caching
    print("\n[7] Second search (pages now cached)...")
    start = time.perf_counter()
    results2 = db.fetch_by_ids(result_ids, columns=["id", "geometry"])
    fetch_time2 = (time.perf_counter() - start) * 1000
    print(f"    ✓ Fetched {len(results2)} results in {fetch_time2:.0f}ms")

    # Summary
    print("\n" + "=" * 60)
    print("FULLY REMOTE ARCHITECTURE WORKS!")
    print("=" * 60)
    print("\n  Timings:")
    print(
        f"    FAISS load:           {load_time:.0f}ms {'(cached)' if cache.is_cached(S3_FAISS_URL) else '(downloaded)'}"
    )
    print(f"    DB connect:           {connect_time:.0f}ms")
    print(f"    FAISS search:         {search_time:.1f}ms")
    print(f"    First fetch:          {fetch_time:.0f}ms (loading pages)")
    print(f"    Second fetch (cached): {fetch_time2:.0f}ms")

    print("\n  Data sizes:")
    print("    FAISS index:     158 MB (cached locally)")
    print("    DuckDB:          1.18 GB (stays on S3!)")
    print("    Total download:  158 MB (not 1.34 GB)")

    print("\n  User experience:")
    print("    First run:       Download 158MB FAISS (~100s)")
    print("    Open notebook:   Connect to DB (~10s)")
    print("    First search:    Load data pages (~20s)")
    print("    Refine search:   Instant (~1ms)")
    print()

    db.close()


if __name__ == "__main__":
    main()
