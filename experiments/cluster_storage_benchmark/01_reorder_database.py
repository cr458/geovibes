#!/usr/bin/env python3
"""
Step 1: Reorder database by different cluster schemes.

Creates three versions of the database:
1. baseline - Copy of original (for fair comparison)
2. hilbert - Sorted by Hilbert-ordered cluster ID
3. two_level - Sorted by super-cluster (44 groups matching row groups)

Usage:
    uv run python 01_reorder_database.py
"""

import time
import yaml
import numpy as np
import faiss
import duckdb
from pathlib import Path
from collections import Counter
from sklearn.decomposition import PCA

# Load config
with open(Path(__file__).parent / "config.yaml") as f:
    config = yaml.safe_load(f)

LOCAL_DIR = Path(config["local_dir"])
LOCAL_DIR.mkdir(parents=True, exist_ok=True)

ROW_GROUP_SIZE = config["duckdb"]["row_group_size"]
N_SUPER_CLUSTERS = config["faiss"]["n_super_clusters"]


def setup_source_connection():
    """Connect to source database on S3."""
    conn = duckdb.connect(":memory:")
    conn.execute("SET enable_progress_bar = false;")
    conn.execute("INSTALL httpfs; LOAD httpfs;")
    conn.execute("INSTALL spatial; LOAD spatial;")
    conn.execute("INSTALL aws; LOAD aws;")
    conn.execute("CALL load_aws_credentials();")
    conn.execute(f"ATTACH '{config['source']['db_url']}' AS remote_db (READ_ONLY)")
    return conn


def download_faiss_index():
    """Download FAISS index to local cache."""
    from geovibes.database.faiss_cache import FaissCache

    cache = FaissCache()
    return cache.get_index(config["source"]["faiss_url"], show_progress=True)


def morton_code(x: float, y: float, order: int = 16) -> int:
    """Compute Morton/Z-order code for 2D point."""
    scale = 2**order
    xi = int(min(max(x, 0), 0.9999) * scale)
    yi = int(min(max(y, 0), 0.9999) * scale)
    z = 0
    for i in range(order):
        z |= ((xi & (1 << i)) << i) | ((yi & (1 << i)) << (i + 1))
    return z


def compute_cluster_assignments(conn, index, batch_size=10000):
    """Compute cluster assignment for every embedding in database."""
    print("  Computing cluster assignments for all embeddings...")

    quantizer = faiss.downcast_index(index.quantizer)
    total_rows = conn.execute(
        "SELECT COUNT(*) FROM remote_db.geo_embeddings"
    ).fetchone()[0]

    # Store assignments: id -> cluster_id
    assignments = {}

    for offset in range(0, total_rows, batch_size):
        df = conn.execute(f"""
            SELECT id, CAST(embedding AS FLOAT[]) as embedding
            FROM remote_db.geo_embeddings
            ORDER BY id
            LIMIT {batch_size} OFFSET {offset}
        """).fetchdf()

        embeddings = np.vstack([np.array(e, dtype=np.float32) for e in df["embedding"]])
        _, cluster_ids = quantizer.search(embeddings, 1)

        for id_, cid in zip(df["id"], cluster_ids.flatten()):
            assignments[int(id_)] = int(cid)

        pct = min(100, (offset + batch_size) * 100 // total_rows)
        print(f"    Progress: {pct}% ({offset + len(df):,}/{total_rows:,})", end="\r")

    print()
    return assignments


def compute_hilbert_ordering(index):
    """Compute Hilbert ordering for cluster centroids."""
    print("  Computing Hilbert ordering for clusters...")

    quantizer = faiss.downcast_index(index.quantizer)
    n_clusters = quantizer.ntotal

    # Get centroids
    centroids = quantizer.reconstruct_n(0, n_clusters)

    # PCA to 2D
    pca = PCA(n_components=2)
    centroids_2d = pca.fit_transform(centroids)
    print(f"    PCA explained variance: {pca.explained_variance_ratio_.sum():.1%}")

    # Normalize to [0, 1]
    c_min = centroids_2d.min(axis=0)
    c_max = centroids_2d.max(axis=0)
    centroids_norm = (centroids_2d - c_min) / (c_max - c_min + 1e-10)

    # Compute Morton codes and sort
    morton_codes = [morton_code(c[0], c[1]) for c in centroids_norm]
    hilbert_order = np.argsort(morton_codes)

    # Map: original cluster ID -> new position
    cluster_to_hilbert = {int(orig): new for new, orig in enumerate(hilbert_order)}

    return cluster_to_hilbert


def compute_two_level_assignments(index):
    """Compute two-level IVF super-cluster assignments."""
    print(f"  Computing two-level IVF with {N_SUPER_CLUSTERS} super-clusters...")

    quantizer = faiss.downcast_index(index.quantizer)
    n_clusters = quantizer.ntotal
    embedding_dim = quantizer.d

    # Get centroids
    centroids = quantizer.reconstruct_n(0, n_clusters)

    # Cluster centroids into super-clusters
    super_quantizer = faiss.IndexFlatL2(embedding_dim)
    super_index = faiss.IndexIVFFlat(super_quantizer, embedding_dim, N_SUPER_CLUSTERS)
    super_index.train(centroids)
    super_index.add(centroids)

    # Assign each cluster to a super-cluster
    _, super_assignments = super_index.quantizer.search(centroids, 1)

    cluster_to_super = {i: int(super_assignments[i, 0]) for i in range(n_clusters)}

    # Report balance
    sizes = Counter(super_assignments.flatten())
    print(
        f"    Super-cluster sizes: min={min(sizes.values())}, max={max(sizes.values())}, avg={np.mean(list(sizes.values())):.1f}"
    )

    return cluster_to_super


def export_reordered_database(conn, sort_key_func, output_path, description):
    """Export database sorted by custom key function."""
    print(f"  Exporting {description}...")

    # Get all data with sort keys
    total_rows = conn.execute(
        "SELECT COUNT(*) FROM remote_db.geo_embeddings"
    ).fetchone()[0]

    # Create output database
    out_conn = duckdb.connect(str(output_path))
    out_conn.execute("INSTALL spatial; LOAD spatial;")

    # Create table with same schema
    out_conn.execute("""
        CREATE TABLE geo_embeddings (
            id BIGINT PRIMARY KEY,
            embedding FLOAT[],
            geometry GEOMETRY
        )
    """)

    # Fetch all data, compute sort keys, and insert in sorted order
    print("    Fetching all data...")
    batch_size = 50000
    all_data = []

    for offset in range(0, total_rows, batch_size):
        df = conn.execute(f"""
            SELECT id, embedding, ST_AsText(geometry) as geom_wkt
            FROM remote_db.geo_embeddings
            ORDER BY id
            LIMIT {batch_size} OFFSET {offset}
        """).fetchdf()

        for _, row in df.iterrows():
            sort_key = sort_key_func(int(row["id"]))
            all_data.append(
                (sort_key, int(row["id"]), row["embedding"], row["geom_wkt"])
            )

        pct = min(100, (offset + batch_size) * 100 // total_rows)
        print(f"      Progress: {pct}%", end="\r")

    print("\n    Sorting by cluster key...")
    all_data.sort(key=lambda x: (x[0], x[1]))  # Sort by (sort_key, id)

    print("    Inserting into new database...")
    for i, (_, id_, embedding, geom_wkt) in enumerate(all_data):
        # Convert embedding to list for DuckDB
        emb_list = list(embedding) if hasattr(embedding, "__iter__") else embedding
        out_conn.execute(
            "INSERT INTO geo_embeddings VALUES (?, ?, ST_GeomFromText(?))",
            [id_, emb_list, geom_wkt],
        )

        if i % 50000 == 0:
            pct = i * 100 // len(all_data)
            print(f"      Progress: {pct}%", end="\r")

    print(f"\n    Exported {len(all_data):,} rows to {output_path}")
    out_conn.close()


def create_baseline_copy(conn, output_path):
    """Create baseline copy with original ordering."""
    print("  Creating baseline copy...")

    total_rows = conn.execute(
        "SELECT COUNT(*) FROM remote_db.geo_embeddings"
    ).fetchone()[0]

    out_conn = duckdb.connect(str(output_path))
    out_conn.execute("INSTALL spatial; LOAD spatial;")

    out_conn.execute("""
        CREATE TABLE geo_embeddings (
            id BIGINT PRIMARY KEY,
            embedding FLOAT[],
            geometry GEOMETRY
        )
    """)

    batch_size = 50000
    for offset in range(0, total_rows, batch_size):
        conn.execute(f"""
            INSERT INTO out_conn.geo_embeddings
            SELECT id, embedding, geometry
            FROM remote_db.geo_embeddings
            ORDER BY id
            LIMIT {batch_size} OFFSET {offset}
        """)

        pct = min(100, (offset + batch_size) * 100 // total_rows)
        print(f"    Progress: {pct}%", end="\r")

    # Simpler approach: direct copy
    out_conn.execute(f"""
        INSERT INTO geo_embeddings
        SELECT id, embedding, geometry
        FROM read_parquet('{LOCAL_DIR}/baseline_temp.parquet')
    """)

    print(f"\n    Exported {total_rows:,} rows to {output_path}")
    out_conn.close()


def main():
    print("=" * 70)
    print("Step 1: Reorder Database by Cluster Schemes")
    print("=" * 70)

    # Download FAISS index
    print("\n[1/5] Loading FAISS index...")
    index = download_faiss_index()
    print(f"  Index: {index.ntotal:,} vectors, nlist={index.nlist}")

    # Connect to source
    print("\n[2/5] Connecting to source database...")
    conn = setup_source_connection()
    total_rows = conn.execute(
        "SELECT COUNT(*) FROM remote_db.geo_embeddings"
    ).fetchone()[0]
    print(f"  Database: {total_rows:,} rows")

    # Compute cluster assignments for all embeddings
    print("\n[3/5] Computing cluster assignments...")
    start = time.time()
    cluster_assignments = compute_cluster_assignments(conn, index)
    print(f"  Completed in {time.time() - start:.1f}s")

    # Compute orderings
    print("\n[4/5] Computing cluster orderings...")
    hilbert_order = compute_hilbert_ordering(index)
    two_level_order = compute_two_level_assignments(index)

    # Compute combined keys
    # Hilbert: (hilbert_cluster_id, original_id)
    # Two-level: (super_cluster_id, cluster_id, original_id)

    def hilbert_sort_key(id_):
        cluster = cluster_assignments.get(id_, 0)
        hilbert_pos = hilbert_order.get(cluster, 0)
        return (hilbert_pos, id_)

    def two_level_sort_key(id_):
        cluster = cluster_assignments.get(id_, 0)
        super_cluster = two_level_order.get(cluster, 0)
        return (super_cluster, cluster, id_)

    def baseline_sort_key(id_):
        return (id_,)

    # Export databases
    print("\n[5/5] Exporting reordered databases...")

    output_files = {
        "baseline": LOCAL_DIR / "baseline_metadata.db",
        "hilbert": LOCAL_DIR / "hilbert_metadata.db",
        "two_level": LOCAL_DIR / "two_level_metadata.db",
    }

    # For efficiency, export to parquet first then create DuckDB
    print("\n  Exporting to intermediate parquet files...")

    # Export baseline (original order)
    print("\n  [5a] Baseline (original order)...")
    conn.execute(f"""
        COPY (SELECT * FROM remote_db.geo_embeddings ORDER BY id)
        TO '{LOCAL_DIR}/baseline.parquet' (FORMAT PARQUET, ROW_GROUP_SIZE {ROW_GROUP_SIZE})
    """)

    # For hilbert and two-level, we need to add sort keys
    print("\n  [5b] Computing sort keys for all rows...")

    # Create a table with sort keys
    conn.execute(
        "CREATE TABLE sort_keys (id BIGINT, hilbert_key BIGINT, two_level_key BIGINT)"
    )

    batch_size = 100000
    for offset in range(0, total_rows, batch_size):
        batch_ids = list(range(offset, min(offset + batch_size, total_rows)))

        rows = []
        for id_ in batch_ids:
            if id_ in cluster_assignments:
                cluster = cluster_assignments[id_]
                hilbert_pos = hilbert_order.get(cluster, 0)
                super_cluster = two_level_order.get(cluster, 0)
                # Combine into single sortable key
                hilbert_key = hilbert_pos * 10_000_000_000 + id_
                two_level_key = (
                    super_cluster * 100_000_000_000 + cluster * 10_000_000 + id_
                )
                rows.append((id_, hilbert_key, two_level_key))

        if rows:
            conn.executemany("INSERT INTO sort_keys VALUES (?, ?, ?)", rows)

        pct = min(100, (offset + batch_size) * 100 // total_rows)
        print(f"    Progress: {pct}%", end="\r")

    print("\n\n  [5c] Hilbert-ordered export...")
    conn.execute(f"""
        COPY (
            SELECT e.id, e.embedding, e.geometry
            FROM remote_db.geo_embeddings e
            JOIN sort_keys s ON e.id = s.id
            ORDER BY s.hilbert_key
        )
        TO '{LOCAL_DIR}/hilbert.parquet' (FORMAT PARQUET, ROW_GROUP_SIZE {ROW_GROUP_SIZE})
    """)

    print("  [5d] Two-level export...")
    conn.execute(f"""
        COPY (
            SELECT e.id, e.embedding, e.geometry
            FROM remote_db.geo_embeddings e
            JOIN sort_keys s ON e.id = s.id
            ORDER BY s.two_level_key
        )
        TO '{LOCAL_DIR}/two_level.parquet' (FORMAT PARQUET, ROW_GROUP_SIZE {ROW_GROUP_SIZE})
    """)

    # Convert parquet to DuckDB
    print("\n  [5e] Converting parquet to DuckDB...")
    for name in ["baseline", "hilbert", "two_level"]:
        parquet_path = LOCAL_DIR / f"{name}.parquet"
        db_path = LOCAL_DIR / f"{name}_metadata.db"

        out_conn = duckdb.connect(str(db_path))
        out_conn.execute("INSTALL spatial; LOAD spatial;")
        out_conn.execute(f"""
            CREATE TABLE geo_embeddings AS
            SELECT * FROM read_parquet('{parquet_path}')
        """)
        out_conn.close()
        print(f"    Created {db_path}")

    conn.close()

    print("\n" + "=" * 70)
    print("Reordering complete!")
    print("=" * 70)
    print(f"\nOutput files in {LOCAL_DIR}:")
    for name, path in output_files.items():
        if path.exists():
            size_mb = path.stat().st_size / 1024 / 1024
            print(f"  {name}: {path.name} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
