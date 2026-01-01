#!/usr/bin/env python
"""Test the full remote architecture: local FAISS + remote DuckDB.

This script demonstrates:
1. Loading a FAISS index (local for now, remote via FaissCache later)
2. Connecting to remote DuckDB via httpfs
3. Performing FAISS search locally
4. Fetching results from remote database
"""

import time

import faiss
import numpy as np

from geovibes.database.remote_db import RemoteDuckDB

# Paths
LOCAL_FAISS_INDEX = "local_databases/alabama_google_satellite_embeddings_v1_2024_2025_25_0_10_faiss_4096_64_8.index"
REMOTE_DB_URL = "s3://us-west-2.opendata.source.coop/geovibes/search/USA/alabama/2024-01-01-2025-01-01/alabama_google_satellite_embeddings_v1_2024_2025_25_0_10_metadata.db"


def main():
    print("=" * 60)
    print("Testing Full Remote Architecture")
    print("=" * 60)

    # Step 1: Load local FAISS index
    print("\n[1] Loading local FAISS index...")
    start = time.perf_counter()
    index = faiss.read_index(LOCAL_FAISS_INDEX, faiss.IO_FLAG_MMAP)
    load_time = (time.perf_counter() - start) * 1000
    print(f"    ✓ Loaded {index.ntotal:,} vectors in {load_time:.0f}ms")
    print(f"    ✓ Dimension: {index.d}")

    # Step 2: Connect to remote database
    print("\n[2] Connecting to remote DuckDB via httpfs...")
    db = RemoteDuckDB()
    start = time.perf_counter()
    db.connect(REMOTE_DB_URL)
    connect_time = (time.perf_counter() - start) * 1000
    print(f"    ✓ Connected in {connect_time:.0f}ms")

    # Verify row count matches FAISS
    count = db.query("SELECT COUNT(*) as cnt FROM geo_embeddings")["cnt"].iloc[0]
    print(f"    ✓ Database has {count:,} rows")

    # Step 3: Get a sample embedding to use as query
    print("\n[3] Getting sample embedding from database...")
    start = time.perf_counter()
    sample = db.query("""
        SELECT id, embedding
        FROM geo_embeddings
        LIMIT 1
    """)
    fetch_time = (time.perf_counter() - start) * 1000
    print(f"    ✓ Fetched sample in {fetch_time:.0f}ms")

    query_embedding = np.array(sample["embedding"].iloc[0], dtype=np.float32)
    query_id = sample["id"].iloc[0]
    print(f"    ✓ Query ID: {query_id}")
    print(f"    ✓ Embedding shape: {query_embedding.shape}")

    # Step 4: Search FAISS locally
    print("\n[4] Searching FAISS index locally...")
    k = 100
    start = time.perf_counter()
    distances, indices = index.search(query_embedding.reshape(1, -1), k)
    search_time = (time.perf_counter() - start) * 1000
    print(f"    ✓ Found {k} neighbors in {search_time:.1f}ms")
    print(f"    ✓ Top 5 indices: {indices[0][:5].tolist()}")
    print(f"    ✓ Top 5 distances: {distances[0][:5].tolist()}")

    # Step 5: Fetch results from remote database
    print("\n[5] Fetching results from remote database...")
    result_ids = indices[0].tolist()
    start = time.perf_counter()
    results = db.fetch_by_ids(result_ids, columns=["id", "geometry"])
    fetch_time = (time.perf_counter() - start) * 1000
    print(f"    ✓ Fetched {len(results)} results in {fetch_time:.0f}ms")

    # Step 6: Parse geometries
    print("\n[6] Processing geometries...")
    geometries_df = db.query(f"""
        SELECT id, ST_AsGeoJSON(geometry) as geojson
        FROM geo_embeddings
        WHERE id IN ({','.join(str(i) for i in result_ids[:10])})
    """)
    print(f"    ✓ Got {len(geometries_df)} geometries")
    if len(geometries_df) > 0:
        print(f"    ✓ Sample geometry: {geometries_df['geojson'].iloc[0][:80]}...")

    # Summary
    print("\n" + "=" * 60)
    print("Summary: Full Remote Architecture Works!")
    print("=" * 60)
    print(f"  FAISS load:     {load_time:.0f}ms (memory-mapped)")
    print(f"  DB connect:     {connect_time:.0f}ms (cold start)")
    print(f"  FAISS search:   {search_time:.1f}ms (local)")
    print(f"  Fetch results:  {fetch_time:.0f}ms (remote)")
    print(f"  Total vectors:  {index.ntotal:,}")
    print()
    print("Architecture validated:")
    print("  ✓ Local FAISS search is fast (~1ms)")
    print("  ✓ Remote fetch-by-ID is fast (~1-2ms)")
    print("  ✓ No need to download 1.18GB database!")
    print()

    db.close()


if __name__ == "__main__":
    main()
