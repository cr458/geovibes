#!/usr/bin/env python3
"""Build remote embeddings-only parquet variants for httpfs retrieval benchmarking.

Outputs:
- id-sorted embeddings parquet
- FAISS-list-sorted embeddings parquet
- matching id->row_group sidecar parquet files for each layout

This avoids large local DuckDB rebuilds while still enabling realistic httpfs fetch tests.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import duckdb
import faiss
import fsspec
import numpy as np
import pandas as pd

from geovibes.database.faiss_cache import FaissCache


DEFAULT_DB_URL = (
    "s3://us-west-2.opendata.source.coop/geovibes/search/USA/alabama/"
    "2024-01-01-2025-01-01/httpfs/"
    "alabama_dino_vit_small_patch16_224_2024_2025_32_16_10/metadata.db"
)
DEFAULT_FAISS_URL = (
    "s3://us-west-2.opendata.source.coop/geovibes/search/USA/alabama/"
    "2024-01-01-2025-01-01/httpfs/"
    "alabama_dino_vit_small_patch16_224_2024_2025_32_16_10/faiss.index"
)
DEFAULT_S3_PREFIX = (
    "s3://us-west-2.opendata.source.coop/geovibes/experiments/httpfs_codex/embedding_variants"
)


def setup_connection() -> duckdb.DuckDBPyConnection:
    conn = duckdb.connect(":memory:")
    conn.execute("SET enable_progress_bar=false")
    conn.execute("SET enable_profiling='no_output'")
    conn.execute("PRAGMA disable_profiling")
    conn.execute("SET enable_object_cache=false")
    conn.execute("INSTALL httpfs; LOAD httpfs;")
    conn.execute("INSTALL aws; LOAD aws;")
    conn.execute("CALL load_aws_credentials();")
    return conn


def build_sort_key_table(
    conn: duckdb.DuckDBPyConnection,
    index: faiss.IndexIVF,
    flush_target_rows: int = 250_000,
) -> None:
    """Create TEMP table sort_keys(id, list_id) from FAISS inverted lists."""
    inv = index.invlists
    nlist = int(index.nlist)
    ntotal = int(index.ntotal)

    conn.execute("CREATE TEMP TABLE sort_keys (id BIGINT, list_id INTEGER)")

    pending_ids: list[np.ndarray] = []
    pending_lists: list[np.ndarray] = []
    pending_rows = 0
    inserted = 0

    def flush() -> None:
        nonlocal pending_ids, pending_lists, pending_rows, inserted
        if pending_rows == 0:
            return
        ids_arr = np.concatenate(pending_ids, axis=0)
        list_arr = np.concatenate(pending_lists, axis=0)
        df = pd.DataFrame({"id": ids_arr, "list_id": list_arr})
        conn.register("_sort_keys_chunk", df)
        conn.execute("INSERT INTO sort_keys SELECT id, list_id FROM _sort_keys_chunk")
        conn.unregister("_sort_keys_chunk")
        inserted += len(df)
        pending_ids = []
        pending_lists = []
        pending_rows = 0
        pct = 100.0 * inserted / max(1, ntotal)
        print(f"  sort_keys rows: {inserted:,}/{ntotal:,} ({pct:.1f}%)", flush=True)

    print("Building FAISS list assignment table...")
    for list_id in range(nlist):
        size = int(inv.list_size(list_id))
        if size <= 0:
            continue
        ids = np.array(faiss.rev_swig_ptr(inv.get_ids(list_id), size), dtype=np.int64)
        list_ids = np.full(size, list_id, dtype=np.int32)
        pending_ids.append(ids)
        pending_lists.append(list_ids)
        pending_rows += size
        if pending_rows >= flush_target_rows:
            flush()
    flush()

    stats = conn.execute(
        "SELECT COUNT(*) AS n_rows, COUNT(DISTINCT id) AS n_distinct_ids FROM sort_keys"
    ).fetchone()
    if not stats:
        raise RuntimeError("Failed to build sort_keys table")
    n_rows, n_distinct_ids = int(stats[0]), int(stats[1])
    if n_rows != ntotal or n_distinct_ids != ntotal:
        raise RuntimeError(
            f"sort_keys mismatch: rows={n_rows:,} distinct={n_distinct_ids:,} expected={ntotal:,}"
        )
    print(f"sort_keys ready: {n_rows:,} rows, {n_distinct_ids:,} distinct ids")


def copy_embeddings_id_sorted(
    conn: duckdb.DuckDBPyConnection,
    out_url: str,
    row_group_size: int,
) -> None:
    print(f"Writing id-sorted embeddings parquet -> {out_url}")
    conn.execute(
        f"""
        COPY (
            SELECT id, embedding
            FROM remote_db.geo_embeddings
            ORDER BY id
        )
        TO '{out_url}'
        (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE {int(row_group_size)})
        """
    )


def copy_embeddings_ivf_sorted(
    conn: duckdb.DuckDBPyConnection,
    out_url: str,
    row_group_size: int,
) -> None:
    print(f"Writing ivf-list-sorted embeddings parquet -> {out_url}")
    conn.execute(
        f"""
        COPY (
            SELECT e.id, e.embedding AS embedding
            FROM remote_db.geo_embeddings e
            JOIN sort_keys s USING (id)
            ORDER BY s.list_id, e.id
        )
        TO '{out_url}'
        (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE {int(row_group_size)})
        """
    )


def copy_row_group_sidecar_id_sorted(
    conn: duckdb.DuckDBPyConnection,
    out_url: str,
    row_group_size: int,
) -> None:
    print(f"Writing id-sorted row-group sidecar -> {out_url}")
    conn.execute(
        f"""
        COPY (
            SELECT
                id,
                CAST(FLOOR((ROW_NUMBER() OVER (ORDER BY id) - 1) / {int(row_group_size)}) AS BIGINT) AS row_group
            FROM remote_db.geo_embeddings
            ORDER BY id
        )
        TO '{out_url}'
        (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )


def copy_row_group_sidecar_ivf_sorted(
    conn: duckdb.DuckDBPyConnection,
    out_url: str,
    row_group_size: int,
) -> None:
    print(f"Writing ivf-list-sorted row-group sidecar -> {out_url}")
    conn.execute(
        f"""
        COPY (
            SELECT
                id,
                CAST(FLOOR((ROW_NUMBER() OVER (ORDER BY list_id, id) - 1) / {int(row_group_size)}) AS BIGINT) AS row_group
            FROM sort_keys
            ORDER BY id
        )
        TO '{out_url}'
        (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build remote embeddings-only variants")
    parser.add_argument("--source-db-url", default=DEFAULT_DB_URL)
    parser.add_argument("--faiss-url", default=DEFAULT_FAISS_URL)
    parser.add_argument("--s3-prefix", default=DEFAULT_S3_PREFIX)
    parser.add_argument("--row-group-size", type=int, default=122880)
    parser.add_argument(
        "--tag",
        default="alabama_ivf_layout_v1",
        help="Suffix folder/tag under --s3-prefix",
    )
    parser.add_argument(
        "--emit-manifest",
        action="store_true",
        help="Write manifest JSON to <s3-prefix>/<tag>/manifest.json",
    )
    args = parser.parse_args()

    source_db_url = str(args.source_db_url)
    faiss_url = str(args.faiss_url)
    row_group_size = int(args.row_group_size)
    out_root = f"{str(args.s3_prefix).rstrip('/')}/{args.tag}"

    id_embeddings_url = f"{out_root}/embeddings_id_sorted_rg{row_group_size}.parquet"
    ivf_embeddings_url = f"{out_root}/embeddings_ivflist_sorted_rg{row_group_size}.parquet"
    id_row_groups_url = f"{out_root}/row_groups_id_sorted_rg{row_group_size}.parquet"
    ivf_row_groups_url = f"{out_root}/row_groups_ivflist_sorted_rg{row_group_size}.parquet"
    manifest_url = f"{out_root}/manifest.json"

    print("=" * 88)
    print("Build Embeddings Variants")
    print("=" * 88)
    print(f"Source DB: {source_db_url}")
    print(f"FAISS URL: {faiss_url}")
    print(f"Output root: {out_root}")
    print(f"Row-group size: {row_group_size}")

    t0 = time.perf_counter()
    index = FaissCache().get_index(faiss_url, show_progress=True)
    if not hasattr(index, "invlists"):
        raise RuntimeError(f"Expected IVF index, got {type(index)}")
    print(f"Loaded FAISS index: ntotal={index.ntotal:,} nlist={index.nlist}")

    conn = setup_connection()
    conn.execute(f"ATTACH '{source_db_url}' AS remote_db (READ_ONLY)")

    source_rows = conn.execute("SELECT COUNT(*) FROM remote_db.geo_embeddings").fetchone()[0]
    print(f"Source rows: {int(source_rows):,}")
    if int(source_rows) != int(index.ntotal):
        raise RuntimeError(
            f"Row count mismatch: db={int(source_rows):,} faiss={int(index.ntotal):,}"
        )

    build_sort_key_table(conn, index)

    copy_embeddings_id_sorted(conn, id_embeddings_url, row_group_size)
    copy_embeddings_ivf_sorted(conn, ivf_embeddings_url, row_group_size)
    copy_row_group_sidecar_id_sorted(conn, id_row_groups_url, row_group_size)
    copy_row_group_sidecar_ivf_sorted(conn, ivf_row_groups_url, row_group_size)

    conn.close()

    manifest = {
        "source_db_url": source_db_url,
        "faiss_url": faiss_url,
        "row_group_size": row_group_size,
        "outputs": {
            "embeddings_id_sorted_url": id_embeddings_url,
            "embeddings_ivflist_sorted_url": ivf_embeddings_url,
            "row_groups_id_sorted_url": id_row_groups_url,
            "row_groups_ivflist_sorted_url": ivf_row_groups_url,
        },
        "generated_at_epoch_s": time.time(),
        "elapsed_s": time.perf_counter() - t0,
    }

    if args.emit_manifest:
        with fsspec.open(manifest_url, "w") as f:
            f.write(json.dumps(manifest, indent=2))
        print(f"Wrote manifest: {manifest_url}")

    print("\nVariant URLs:")
    for key, value in manifest["outputs"].items():
        print(f"- {key}: {value}")
    print(f"\nTotal elapsed: {manifest['elapsed_s']:.1f}s")


if __name__ == "__main__":
    main()
