#!/usr/bin/env python3
"""End-to-end httpfs retrieval harness for GeoVibes usage patterns.

This harness profiles the same hot path as interactive usage:
1. point label nearest-point query (remote DuckDB)
2. FAISS search (local)
3. metadata retrieval for search results
4. embedding prefetch for search results

It supports multiple strategy variants so we can benchmark baseline vs optimizations.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import queue
import random
import statistics
import threading
import time
import traceback
from collections import defaultdict, OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb
import faiss
import numpy as np
import pandas as pd
import yaml

from geovibes.database.faiss_cache import FaissCache

NEAREST_POINT_QUERY = """
SELECT  g.id,
        ST_AsText(g.geometry) AS wkt,
        ST_Distance(geometry, ST_Point(?, ?)) AS dist_m,
        g.embedding
FROM    geo_embeddings g
ORDER BY dist_m
LIMIT   1
"""

RUNTIME_SETTINGS: dict[str, Any] = {
    "enable_object_cache": False,
    "extra_sql": [],
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.array(values, dtype=np.float64), q))


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(statistics.fmean(values))


def _as_int_list(ids: list[Any]) -> list[int]:
    return [int(x) for x in ids]


def _is_parquet_source(path_or_url: str) -> bool:
    value = str(path_or_url).strip().lower()
    return value.endswith(".parquet") or value.endswith(".parq")


class IdLruCache:
    """Simple ID-only LRU cache for simulating cross-search embedding reuse."""

    def __init__(self, max_ids: int):
        self.max_ids = max(0, int(max_ids))
        self._store: OrderedDict[int, None] = OrderedDict()

    def contains(self, item: int) -> bool:
        key = int(item)
        if key not in self._store:
            return False
        self._store.move_to_end(key)
        return True

    def add_many(self, items: list[int]) -> None:
        if self.max_ids <= 0:
            return
        for item in items:
            key = int(item)
            self._store[key] = None
            self._store.move_to_end(key)
            while len(self._store) > self.max_ids:
                self._store.popitem(last=False)

    def size(self) -> int:
        return len(self._store)


def configure_runtime_settings(
    *,
    enable_object_cache: bool | None = None,
    extra_sql: list[str] | None = None,
) -> None:
    if enable_object_cache is not None:
        RUNTIME_SETTINGS["enable_object_cache"] = bool(enable_object_cache)
    if extra_sql is not None:
        RUNTIME_SETTINGS["extra_sql"] = [str(x) for x in extra_sql if str(x).strip()]


class RemoteConnectionPool:
    """Simple reusable pool for remote DuckDB connections."""

    def __init__(self, db_url: str, max_size: int):
        self.db_url = db_url
        self.max_size = max(1, int(max_size))
        self._queue: queue.LifoQueue[duckdb.DuckDBPyConnection] = queue.LifoQueue(
            maxsize=self.max_size
        )
        self._created = 0
        self._lock = threading.Lock()

    def acquire(self, timeout_seconds: float = 30.0) -> duckdb.DuckDBPyConnection:
        with self._lock:
            if self._created < self.max_size:
                self._created += 1
                should_create = True
            else:
                should_create = False

        if should_create:
            return setup_remote_connection(self.db_url)

        try:
            return self._queue.get(timeout=max(0.01, float(timeout_seconds)))
        except queue.Empty:
            return setup_remote_connection(self.db_url)

    def release(self, conn: duckdb.DuckDBPyConnection | None) -> None:
        if conn is None:
            return
        try:
            self._queue.put_nowait(conn)
        except queue.Full:
            try:
                conn.close()
            except Exception:
                pass

    def stats(self) -> dict[str, int]:
        return {"size": self.max_size, "created": self._created, "idle": self._queue.qsize()}

    def close(self) -> None:
        while True:
            try:
                conn = self._queue.get_nowait()
            except queue.Empty:
                break
            try:
                conn.close()
            except Exception:
                pass


def setup_remote_connection(db_url: str, n_threads: int | None = None) -> duckdb.DuckDBPyConnection:
    conn = duckdb.connect(":memory:")
    conn.execute("SET enable_progress_bar = false;")
    conn.execute("SET enable_profiling='no_output'")
    conn.execute("PRAGMA disable_profiling")
    if bool(RUNTIME_SETTINGS.get("enable_object_cache", False)):
        conn.execute("SET enable_object_cache=true")
    else:
        conn.execute("SET enable_object_cache=false")
    conn.execute("INSTALL httpfs; LOAD httpfs;")
    conn.execute("INSTALL spatial; LOAD spatial;")
    conn.execute("INSTALL aws; LOAD aws;")
    conn.execute("CALL load_aws_credentials();")
    for sql in list(RUNTIME_SETTINGS.get("extra_sql", []) or []):
        conn.execute(sql)
    if n_threads:
        conn.execute(f"SET threads = {int(n_threads)}")
    if _is_parquet_source(db_url):
        conn.execute(
            f"CREATE VIEW geo_embeddings AS SELECT * FROM read_parquet('{db_url}')"
        )
    else:
        conn.execute(f"ATTACH '{db_url}' AS remote_db (READ_ONLY)")
        conn.execute("CREATE VIEW geo_embeddings AS SELECT * FROM remote_db.geo_embeddings")
    return conn


def setup_geometry_cache_connection(geometry_cache_url: str) -> tuple[duckdb.DuckDBPyConnection, str]:
    cache = FaissCache()
    local_path = cache.get_geometry_cache(geometry_cache_url, show_progress=True)

    conn = duckdb.connect(":memory:")
    conn.execute("INSTALL spatial; LOAD spatial;")
    conn.execute(f"CREATE VIEW geometry_cache AS SELECT * FROM '{local_path}'")
    return conn, str(local_path)


def resolve_local_parquet_path(path_or_url: str) -> Path:
    value = str(path_or_url).strip()
    if value.startswith(("s3://", "gs://", "http://", "https://")):
        cache = FaissCache()
        return cache.get_geometry_cache(value, show_progress=True)
    return Path(value).expanduser().resolve()


def ensure_row_group_cache(
    db_url: str,
    row_group_size: int,
    cache_dir: Path,
) -> Path:
    """Build (or reuse) a local id->row_group parquet sidecar.

    This is intentionally one-time setup work so prefetch grouping does not perform
    expensive remote rowid lookups on every query.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_key = hashlib.sha256(f"{db_url}|{row_group_size}".encode("utf-8")).hexdigest()[:16]
    out_path = cache_dir / f"{cache_key}.parquet"
    if out_path.exists():
        return out_path

    conn = setup_remote_connection(db_url)
    try:
        if _is_parquet_source(db_url):
            # Parquet sources do not expose rowid; use stable scan order.
            conn.execute(
                f"""
                COPY (
                    SELECT
                        id,
                        CAST(FLOOR((ROW_NUMBER() OVER () - 1) / {int(row_group_size)}) AS BIGINT) AS row_group
                    FROM geo_embeddings
                )
                TO '{out_path}' (FORMAT PARQUET, COMPRESSION ZSTD)
                """
            )
        else:
            conn.execute(
                f"""
                COPY (
                    SELECT id, CAST(FLOOR(rowid / {int(row_group_size)}) AS BIGINT) AS row_group
                    FROM remote_db.geo_embeddings
                )
                TO '{out_path}' (FORMAT PARQUET, COMPRESSION ZSTD)
                """
            )
    finally:
        conn.close()

    return out_path


def setup_row_group_cache_connection(row_group_cache_path: Path) -> duckdb.DuckDBPyConnection:
    conn = duckdb.connect(":memory:")
    conn.execute(
        f"CREATE VIEW row_group_cache AS SELECT * FROM '{row_group_cache_path.resolve()}'"
    )
    return conn


def load_faiss_index(faiss_url: str) -> faiss.Index:
    cache = FaissCache()
    return cache.get_index(faiss_url, show_progress=True)


def build_query_vector_from_ids(
    conn: duckdb.DuckDBPyConnection,
    query_ids: list[int],
    embedding_expr: str = "embedding",
) -> np.ndarray:
    ids = _as_int_list(query_ids)
    if len(ids) < 2:
        raise RuntimeError("Need at least two query_ids to build query vector")
    pos_ids = ids[:-1]
    neg_ids = ids[-1:]
    return build_query_vector_from_labeled_ids(
        conn,
        pos_ids=pos_ids,
        neg_ids=neg_ids,
        embedding_expr=embedding_expr,
    )


def fetch_embeddings_by_id(
    conn: duckdb.DuckDBPyConnection,
    ids: list[int],
    embedding_expr: str = "embedding",
) -> dict[int, np.ndarray]:
    if not ids:
        return {}
    int_ids = _as_int_list(ids)
    placeholders = ",".join(["?" for _ in int_ids])
    sql = f"""
    SELECT id, {embedding_expr} AS embedding
    FROM geo_embeddings
    WHERE id IN ({placeholders})
    """
    rows = conn.execute(sql, int_ids).fetchall()
    out: dict[int, np.ndarray] = {}
    for id_val, emb in rows:
        out[int(id_val)] = np.asarray(emb, dtype=np.float32)
    return out


def build_query_vector_from_labeled_ids(
    conn: duckdb.DuckDBPyConnection,
    pos_ids: list[int],
    neg_ids: list[int] | None = None,
    embedding_expr: str = "embedding",
) -> np.ndarray:
    if not pos_ids:
        raise RuntimeError("Need at least one positive label to build query vector")
    neg_ids = neg_ids or []
    all_ids = list(dict.fromkeys([int(x) for x in pos_ids + neg_ids]))
    emb_by_id = fetch_embeddings_by_id(conn, all_ids, embedding_expr=embedding_expr)

    missing = [qid for qid in all_ids if qid not in emb_by_id]
    if missing:
        raise RuntimeError(f"Missing query IDs in DB: {missing}")

    pos = np.mean([emb_by_id[int(qid)] for qid in pos_ids], axis=0)
    if neg_ids:
        neg = np.mean([emb_by_id[int(qid)] for qid in neg_ids], axis=0)
    else:
        neg = np.zeros_like(pos)
    query_vec = 2.0 * pos - neg
    return np.asarray(query_vec, dtype=np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_vec = np.asarray(a, dtype=np.float32)
    b_vec = np.asarray(b, dtype=np.float32)
    denom = float(np.linalg.norm(a_vec) * np.linalg.norm(b_vec))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(a_vec, b_vec) / denom)


def generate_iterative_query_sets(
    conn: duckdb.DuckDBPyConnection,
    index: faiss.Index,
    base_query_sets: list[list[int]],
    *,
    embedding_expr: str,
    feedback_nprobe: int,
    feedback_top_k: int,
    steps_per_seed: int,
    add_positive_per_step: int,
    add_negative_per_step: int,
    max_positive_ids: int,
    max_negative_ids: int,
) -> list[list[int]]:
    generated: list[list[int]] = []
    steps = max(1, int(steps_per_seed))
    feedback_top_k = max(1, int(feedback_top_k))
    add_positive = max(0, int(add_positive_per_step))
    add_negative = max(0, int(add_negative_per_step))
    max_positive = max(1, int(max_positive_ids))
    max_negative = max(1, int(max_negative_ids))

    for seed_ids in base_query_sets:
        ids = _as_int_list(seed_ids)
        if len(ids) < 2:
            continue
        pos_ids = [int(x) for x in ids[:-1]]
        neg_ids = [int(ids[-1])]

        generated.append(pos_ids + neg_ids)

        for _ in range(steps - 1):
            query_vec = build_query_vector_from_labeled_ids(
                conn,
                pos_ids=pos_ids,
                neg_ids=neg_ids,
                embedding_expr=embedding_expr,
            )
            _, nn_ids, _ = run_faiss_search(
                index=index,
                query_vector=query_vec,
                n_neighbors=feedback_top_k,
                nprobe=feedback_nprobe,
            )
            if not nn_ids:
                break

            labeled = set(pos_ids + neg_ids)
            added_pos = 0
            for candidate in nn_ids:
                if candidate in labeled:
                    continue
                pos_ids.append(int(candidate))
                labeled.add(int(candidate))
                added_pos += 1
                if len(pos_ids) > max_positive:
                    pos_ids = pos_ids[-max_positive:]
                if added_pos >= add_positive:
                    break

            added_neg = 0
            if add_negative > 0:
                for candidate in reversed(nn_ids):
                    if candidate in labeled:
                        continue
                    neg_ids.append(int(candidate))
                    labeled.add(int(candidate))
                    added_neg += 1
                    if len(neg_ids) > max_negative:
                        neg_ids = neg_ids[-max_negative:]
                    if added_neg >= add_negative:
                        break

            if added_pos == 0 and added_neg == 0:
                break

            generated.append(pos_ids + neg_ids)

    return generated if generated else base_query_sets


def run_label_stage(
    conn: duckdb.DuckDBPyConnection,
    label_clicks: list[dict[str, float]],
) -> dict[str, Any]:
    latencies_ms: list[float] = []
    returned_ids: list[int] = []

    for click in label_clicks:
        lon = float(click["lon"])
        lat = float(click["lat"])

        start = time.perf_counter()
        row = conn.execute(NEAREST_POINT_QUERY, [lon, lat]).fetchone()
        elapsed = (time.perf_counter() - start) * 1000.0

        latencies_ms.append(elapsed)
        if row and row[0] is not None:
            returned_ids.append(int(row[0]))

    return {
        "count": len(label_clicks),
        "returned_ids": returned_ids,
        "total_ms": float(sum(latencies_ms)),
        "mean_ms": _mean(latencies_ms),
        "p95_ms": _pct(latencies_ms, 95),
    }


def run_faiss_search(
    index: faiss.Index,
    query_vector: np.ndarray,
    n_neighbors: int,
    nprobe: int,
) -> tuple[float, list[int], list[float]]:
    query = query_vector.reshape(1, -1).astype(np.float32)

    start = time.perf_counter()
    try:
        params = faiss.SearchParametersIVF(nprobe=nprobe)
        distances, ids = index.search(query, int(n_neighbors), params=params)
    except Exception:
        distances, ids = index.search(query, int(n_neighbors))
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    id_list = [int(i) for i in ids[0].tolist() if int(i) >= 0]
    dist_list = [float(d) for d in distances[0].tolist()[: len(id_list)]]
    return elapsed_ms, id_list, dist_list


def fetch_search_metadata_remote(
    conn: duckdb.DuckDBPyConnection,
    ids: list[int],
    query_mode: str = "in_list",
) -> tuple[float, pd.DataFrame]:
    if not ids:
        return 0.0, pd.DataFrame()
    if query_mode == "in_list":
        placeholders = ",".join(["?" for _ in ids])
        sql = f"""
        SELECT id,
               ST_AsGeoJSON(geometry) AS geometry_json,
               ST_AsText(geometry) AS geometry_wkt
        FROM geo_embeddings
        WHERE id IN ({placeholders})
        """
        params = ids
    elif query_mode == "values_join":
        values_rows = ",".join(["(?)" for _ in ids])
        sql = f"""
        SELECT g.id,
               ST_AsGeoJSON(g.geometry) AS geometry_json,
               ST_AsText(g.geometry) AS geometry_wkt
        FROM geo_embeddings g
        JOIN (VALUES {values_rows}) AS req(id)
          ON g.id = req.id
        """
        params = ids
    else:
        raise ValueError(f"Unknown metadata query_mode={query_mode}")
    start = time.perf_counter()
    df = conn.execute(sql, params).fetchdf()
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return elapsed_ms, df


def fetch_search_metadata_cache(
    cache_conn: duckdb.DuckDBPyConnection,
    ids: list[int],
    query_mode: str = "in_list",
) -> tuple[float, pd.DataFrame]:
    if not ids:
        return 0.0, pd.DataFrame()
    if query_mode == "in_list":
        placeholders = ",".join(["?" for _ in ids])
        sql = f"""
        SELECT id,
               ST_AsGeoJSON(geometry) AS geometry_json,
               ST_AsText(geometry) AS geometry_wkt
        FROM geometry_cache
        WHERE id IN ({placeholders})
        """
        params = ids
    elif query_mode == "values_join":
        values_rows = ",".join(["(?)" for _ in ids])
        sql = f"""
        SELECT g.id,
               ST_AsGeoJSON(g.geometry) AS geometry_json,
               ST_AsText(g.geometry) AS geometry_wkt
        FROM geometry_cache g
        JOIN (VALUES {values_rows}) AS req(id)
          ON g.id = req.id
        """
        params = ids
    else:
        raise ValueError(f"Unknown metadata query_mode={query_mode}")
    start = time.perf_counter()
    df = cache_conn.execute(sql, params).fetchdf()
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return elapsed_ms, df


def fetch_embeddings_single_query(
    conn: duckdb.DuckDBPyConnection,
    ids: list[int],
    include_geometry: bool = False,
    query_mode: str = "in_list",
    embedding_expr: str = "embedding",
) -> dict[str, Any]:
    if not ids:
        return {"elapsed_ms": 0.0, "fetched": 0, "row_groups": 0, "batches": 0}
    select_embedding = f"{embedding_expr} AS embedding"
    if query_mode == "in_list":
        placeholders = ",".join(["?" for _ in ids])
        if include_geometry:
            sql = f"""
            SELECT id, {select_embedding}, geometry
            FROM geo_embeddings
            WHERE id IN ({placeholders})
            """
        else:
            sql = f"""
            SELECT id, {select_embedding}
            FROM geo_embeddings
            WHERE id IN ({placeholders})
            """
        params = ids
    elif query_mode == "values_join":
        values_rows = ",".join(["(?)" for _ in ids])
        if include_geometry:
            sql = f"""
            SELECT g.id, {select_embedding}, g.geometry
            FROM geo_embeddings g
            JOIN (VALUES {values_rows}) AS req(id)
              ON g.id = req.id
            """
        else:
            sql = f"""
            SELECT g.id, {select_embedding}
            FROM geo_embeddings g
            JOIN (VALUES {values_rows}) AS req(id)
              ON g.id = req.id
            """
        params = ids
    else:
        raise ValueError(f"Unknown prefetch query_mode={query_mode}")

    start = time.perf_counter()
    df = conn.execute(sql, params).fetchdf()
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    return {
        "elapsed_ms": elapsed_ms,
        "fetched": int(len(df)),
        "row_groups": None,
        "batches": 1,
        "rowid_lookup_ms": 0.0,
    }


def fetch_embeddings_duckdb_native(
    db_url: str,
    ids: list[int],
    n_threads: int,
    include_geometry: bool = False,
    query_mode: str = "in_list",
    embedding_expr: str = "embedding",
) -> dict[str, Any]:
    if not ids:
        return {"elapsed_ms": 0.0, "fetched": 0, "row_groups": 0, "batches": 0}

    total_start = time.perf_counter()
    conn = setup_remote_connection(db_url, n_threads=n_threads)
    try:
        select_embedding = f"{embedding_expr} AS embedding"
        if query_mode == "in_list":
            placeholders = ",".join(["?" for _ in ids])
            if include_geometry:
                sql = f"""
                SELECT id, {select_embedding}, geometry
                FROM geo_embeddings
                WHERE id IN ({placeholders})
                """
            else:
                sql = f"""
                SELECT id, {select_embedding}
                FROM geo_embeddings
                WHERE id IN ({placeholders})
                """
            params = ids
        elif query_mode == "values_join":
            values_rows = ",".join(["(?)" for _ in ids])
            if include_geometry:
                sql = f"""
                SELECT g.id, {select_embedding}, g.geometry
                FROM geo_embeddings g
                JOIN (VALUES {values_rows}) AS req(id)
                  ON g.id = req.id
                """
            else:
                sql = f"""
                SELECT g.id, {select_embedding}
                FROM geo_embeddings g
                JOIN (VALUES {values_rows}) AS req(id)
                  ON g.id = req.id
                """
            params = ids
        else:
            raise ValueError(f"Unknown prefetch query_mode={query_mode}")
        df = conn.execute(sql, params).fetchdf()
    finally:
        conn.close()

    elapsed_ms = (time.perf_counter() - total_start) * 1000.0
    return {
        "elapsed_ms": elapsed_ms,
        "fetched": int(len(df)),
        "row_groups": None,
        "batches": 1,
        "rowid_lookup_ms": 0.0,
    }


def _batch_ids_by_id_div(ids: list[int], row_group_size: int) -> tuple[list[list[int]], int]:
    groups: dict[int, list[int]] = defaultdict(list)
    for id_ in ids:
        groups[int(id_) // int(row_group_size)].append(int(id_))
    return list(groups.values()), len(groups)


def _batch_ids_by_rowid(
    conn: duckdb.DuckDBPyConnection,
    ids: list[int],
    row_group_size: int,
) -> tuple[list[list[int]], int, float]:
    if not ids:
        return [], 0, 0.0

    placeholders = ",".join(["?" for _ in ids])
    # rowid is available on the physical table, not on the geo_embeddings view.
    sql = f"SELECT id, rowid FROM remote_db.geo_embeddings WHERE id IN ({placeholders})"

    start = time.perf_counter()
    rowid_df = conn.execute(sql, ids).fetchdf()
    lookup_ms = (time.perf_counter() - start) * 1000.0

    groups: dict[int, list[int]] = defaultdict(list)
    for _, row in rowid_df.iterrows():
        id_ = int(row["id"])
        rowid = int(row["rowid"])
        groups[rowid // int(row_group_size)].append(id_)

    return list(groups.values()), len(groups), lookup_ms


def _batch_ids_by_row_group_cache(
    cache_conn: duckdb.DuckDBPyConnection,
    ids: list[int],
) -> tuple[list[list[int]], int, float]:
    if not ids:
        return [], 0, 0.0

    placeholders = ",".join(["?" for _ in ids])
    sql = f"SELECT id, row_group FROM row_group_cache WHERE id IN ({placeholders})"

    start = time.perf_counter()
    df = cache_conn.execute(sql, ids).fetchdf()
    lookup_ms = (time.perf_counter() - start) * 1000.0

    groups: dict[int, list[int]] = defaultdict(list)
    for _, row in df.iterrows():
        groups[int(row["row_group"])].append(int(row["id"]))

    return list(groups.values()), len(groups), lookup_ms


def _fetch_batch_embeddings(
    conn: duckdb.DuckDBPyConnection,
    batch_ids: list[int],
    include_geometry: bool = False,
    fetch_mode: str = "fetchdf",
    query_mode: str = "in_list",
    embedding_expr: str = "embedding",
) -> int:
    if not batch_ids:
        return 0
    select_embedding = f"{embedding_expr} AS embedding"
    if query_mode == "in_list":
        placeholders = ",".join(["?" for _ in batch_ids])
        if include_geometry:
            sql = f"""
            SELECT id, {select_embedding}, geometry
            FROM geo_embeddings
            WHERE id IN ({placeholders})
            """
        else:
            sql = f"""
            SELECT id, {select_embedding}
            FROM geo_embeddings
            WHERE id IN ({placeholders})
            """
        params = batch_ids
    elif query_mode == "values_join":
        values_rows = ",".join(["(?)" for _ in batch_ids])
        if include_geometry:
            sql = f"""
            SELECT g.id, {select_embedding}, g.geometry
            FROM geo_embeddings g
            JOIN (VALUES {values_rows}) AS req(id)
              ON g.id = req.id
            """
        else:
            sql = f"""
            SELECT g.id, {select_embedding}
            FROM geo_embeddings g
            JOIN (VALUES {values_rows}) AS req(id)
              ON g.id = req.id
            """
        params = batch_ids
    else:
        raise ValueError(f"Unknown prefetch query_mode={query_mode}")
    if fetch_mode == "fetchdf":
        df = conn.execute(sql, params).fetchdf()
        return int(len(df))
    if fetch_mode == "arrow":
        tbl = conn.execute(sql, params).fetch_arrow_table()
        return int(len(tbl))
    if fetch_mode == "fetchall":
        rows = conn.execute(sql, params).fetchall()
        return int(len(rows))
    raise ValueError(f"Unknown fetch_mode={fetch_mode}")


def _schedule_batches(
    batches: list[list[int]],
    batch_scheduler: str = "as_is",
    shuffle_seed: int | None = None,
) -> list[list[int]]:
    if not batches:
        return batches
    scheduler = str(batch_scheduler).strip().lower()
    if scheduler in {"as_is", "none", ""}:
        return batches
    if scheduler == "largest_first":
        return sorted(batches, key=len, reverse=True)
    if scheduler == "smallest_first":
        return sorted(batches, key=len)
    if scheduler == "id_ascending":
        return sorted(batches, key=lambda batch: min(batch) if batch else -1)
    if scheduler == "id_descending":
        return sorted(batches, key=lambda batch: min(batch) if batch else -1, reverse=True)
    if scheduler == "random":
        shuffled = list(batches)
        rng = random.Random(shuffle_seed)
        rng.shuffle(shuffled)
        return shuffled
    raise ValueError(f"Unknown batch_scheduler={batch_scheduler}")


def fetch_embeddings_threadpool(
    db_url: str,
    ids: list[int],
    row_group_size: int,
    n_workers: int,
    group_mode: str,
    reuse_connections: bool,
    rowid_source_conn: duckdb.DuckDBPyConnection | None,
    row_group_cache_conn: duckdb.DuckDBPyConnection | None,
    connection_pool: RemoteConnectionPool | None = None,
    include_geometry: bool = False,
    fetch_mode: str = "fetchdf",
    query_mode: str = "in_list",
    batch_scheduler: str = "as_is",
    batch_shuffle_seed: int | None = None,
    embedding_expr: str = "embedding",
) -> dict[str, Any]:
    if not ids:
        return {"elapsed_ms": 0.0, "fetched": 0, "row_groups": 0, "batches": 0}

    ids = _as_int_list(ids)

    rowid_lookup_ms = 0.0
    if group_mode == "id_div":
        batches, row_groups = _batch_ids_by_id_div(ids, row_group_size)
    elif group_mode == "rowid":
        if rowid_source_conn is None:
            raise RuntimeError("rowid grouping requires rowid_source_conn")
        batches, row_groups, rowid_lookup_ms = _batch_ids_by_rowid(
            rowid_source_conn,
            ids,
            row_group_size,
        )
    elif group_mode == "row_group_cache":
        if row_group_cache_conn is None:
            raise RuntimeError("row_group_cache grouping requires row_group_cache_conn")
        batches, row_groups, rowid_lookup_ms = _batch_ids_by_row_group_cache(
            row_group_cache_conn,
            ids,
        )
    else:
        raise ValueError(f"Unknown group_mode={group_mode}")

    batches = _schedule_batches(
        batches,
        batch_scheduler=batch_scheduler,
        shuffle_seed=batch_shuffle_seed,
    )

    if not batches:
        return {
            "elapsed_ms": rowid_lookup_ms,
            "fetched": 0,
            "row_groups": 0,
            "batches": 0,
            "rowid_lookup_ms": rowid_lookup_ms,
        }

    actual_workers = max(1, min(int(n_workers), len(batches)))
    start = time.perf_counter()

    total_fetched = 0

    if not reuse_connections:
        def task(batch_ids: list[int]) -> int:
            conn = setup_remote_connection(db_url)
            try:
                return _fetch_batch_embeddings(
                    conn,
                    batch_ids,
                    include_geometry=include_geometry,
                    fetch_mode=fetch_mode,
                    query_mode=query_mode,
                    embedding_expr=embedding_expr,
                )
            finally:
                conn.close()

        with ThreadPoolExecutor(max_workers=actual_workers) as executor:
            futures = [executor.submit(task, batch) for batch in batches]
            for fut in as_completed(futures):
                total_fetched += int(fut.result())
    else:
        thread_local = threading.local()
        created_connections: list[duckdb.DuckDBPyConnection] = []
        lock = threading.Lock()

        def get_thread_conn() -> duckdb.DuckDBPyConnection:
            conn = getattr(thread_local, "conn", None)
            if conn is None:
                if connection_pool is None:
                    conn = setup_remote_connection(db_url)
                else:
                    conn = connection_pool.acquire()
                thread_local.conn = conn
                with lock:
                    created_connections.append(conn)
            return conn

        def task(batch_ids: list[int]) -> int:
            conn = get_thread_conn()
            return _fetch_batch_embeddings(
                conn,
                batch_ids,
                include_geometry=include_geometry,
                fetch_mode=fetch_mode,
                query_mode=query_mode,
                embedding_expr=embedding_expr,
            )

        try:
            with ThreadPoolExecutor(max_workers=actual_workers) as executor:
                futures = [executor.submit(task, batch) for batch in batches]
                for fut in as_completed(futures):
                    total_fetched += int(fut.result())
        finally:
            for conn in created_connections:
                try:
                    if connection_pool is None:
                        conn.close()
                    else:
                        connection_pool.release(conn)
                except Exception:
                    pass

    elapsed_ms = (time.perf_counter() - start) * 1000.0

    return {
        "elapsed_ms": elapsed_ms + rowid_lookup_ms,
        "fetched": int(total_fetched),
        "row_groups": int(row_groups),
        "batches": int(len(batches)),
        "rowid_lookup_ms": rowid_lookup_ms,
    }


def fetch_embeddings_two_stage_threadpool(
    db_url: str,
    ids: list[int],
    row_group_size: int,
    n_workers: int,
    group_mode: str,
    reuse_connections: bool,
    rowid_source_conn: duckdb.DuckDBPyConnection | None,
    row_group_cache_conn: duckdb.DuckDBPyConnection | None,
    fast_k: int,
    connection_pool: RemoteConnectionPool | None = None,
    include_geometry: bool = False,
    fetch_mode: str = "fetchdf",
    query_mode: str = "in_list",
    batch_scheduler: str = "as_is",
    batch_shuffle_seed: int | None = None,
    embedding_expr: str = "embedding",
) -> dict[str, Any]:
    """Two-stage fetch: prioritize top-K IDs first, then prefetch tail."""
    if not ids:
        return {
            "elapsed_ms": 0.0,
            "fetched": 0,
            "row_groups": 0,
            "batches": 0,
            "rowid_lookup_ms": 0.0,
            "fast_stage_ms": 0.0,
            "tail_stage_ms": 0.0,
            "fast_stage_fetched": 0,
            "tail_stage_fetched": 0,
        }

    fast_k = max(0, min(int(fast_k), len(ids)))
    fast_ids = ids[:fast_k]
    tail_ids = ids[fast_k:]

    stage1 = fetch_embeddings_threadpool(
        db_url=db_url,
        ids=fast_ids,
        row_group_size=row_group_size,
        n_workers=n_workers,
        group_mode=group_mode,
        reuse_connections=reuse_connections,
        rowid_source_conn=rowid_source_conn,
        row_group_cache_conn=row_group_cache_conn,
        connection_pool=connection_pool,
        include_geometry=include_geometry,
        fetch_mode=fetch_mode,
        query_mode=query_mode,
        batch_scheduler=batch_scheduler,
        batch_shuffle_seed=batch_shuffle_seed,
        embedding_expr=embedding_expr,
    )
    stage2 = fetch_embeddings_threadpool(
        db_url=db_url,
        ids=tail_ids,
        row_group_size=row_group_size,
        n_workers=n_workers,
        group_mode=group_mode,
        reuse_connections=reuse_connections,
        rowid_source_conn=rowid_source_conn,
        row_group_cache_conn=row_group_cache_conn,
        connection_pool=connection_pool,
        include_geometry=include_geometry,
        fetch_mode=fetch_mode,
        query_mode=query_mode,
        batch_scheduler=batch_scheduler,
        batch_shuffle_seed=batch_shuffle_seed,
        embedding_expr=embedding_expr,
    )

    return {
        "elapsed_ms": float(stage1.get("elapsed_ms", 0.0) + stage2.get("elapsed_ms", 0.0)),
        "fetched": int(stage1.get("fetched", 0) + stage2.get("fetched", 0)),
        "row_groups": int((stage1.get("row_groups") or 0) + (stage2.get("row_groups") or 0)),
        "batches": int(stage1.get("batches", 0) + stage2.get("batches", 0)),
        "rowid_lookup_ms": float(
            stage1.get("rowid_lookup_ms", 0.0) + stage2.get("rowid_lookup_ms", 0.0)
        ),
        "fast_stage_ms": float(stage1.get("elapsed_ms", 0.0)),
        "tail_stage_ms": float(stage2.get("elapsed_ms", 0.0)),
        "fast_stage_fetched": int(stage1.get("fetched", 0)),
        "tail_stage_fetched": int(stage2.get("fetched", 0)),
    }


def summarize_results(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("status") != "ok":
            continue
        key = (
            str(row["dataset"]),
            str(row["strategy"]),
            int(row["n_neighbors"]),
        )
        grouped[key].append(row)

    summary: list[dict[str, Any]] = []
    for (dataset, strategy, n_neighbors), items in sorted(grouped.items()):
        entry: dict[str, Any] = {
            "dataset": dataset,
            "strategy": strategy,
            "n_neighbors": n_neighbors,
            "trials": len(items),
        }
        metric_keys = [
            "connect_ms",
            "warmup_ms",
            "label_total_ms",
            "query_expand_ms",
            "faiss_ms",
            "metadata_ms",
            "prefetch_ms",
            "prefetch_fast_ms",
            "prefetch_tail_ms",
            "prefetch_rowid_lookup_ms",
            "search_path_total_ms",
            "trial_total_ms",
        ]
        for key in metric_keys:
            vals = [float(it.get(key, 0.0)) for it in items]
            entry[f"{key}_mean"] = _mean(vals)
            entry[f"{key}_p50"] = _pct(vals, 50)
            entry[f"{key}_p95"] = _pct(vals, 95)

        entry["prefetch_fetched_mean"] = _mean(
            [float(it.get("prefetch_fetched", 0)) for it in items]
        )
        entry["prefetch_batches_mean"] = _mean(
            [float(it.get("prefetch_batches", 0)) for it in items]
        )
        entry["prefetch_candidate_ids_mean"] = _mean(
            [float(it.get("prefetch_candidate_ids", 0)) for it in items]
        )
        entry["prefetch_effective_ids_mean"] = _mean(
            [float(it.get("prefetch_effective_ids", 0)) for it in items]
        )
        entry["prefetch_cache_hits_mean"] = _mean(
            [float(it.get("prefetch_cache_hits", 0)) for it in items]
        )
        entry["prefetch_wasted_ratio_mean"] = _mean(
            [float(it.get("prefetch_wasted_ratio", 0.0)) for it in items]
        )
        entry["embedding_cache_size_mean"] = _mean(
            [float(it.get("embedding_cache_size", 0)) for it in items]
        )
        entry["faiss_overlap_ratio_mean"] = _mean(
            [float(it.get("faiss_overlap_ratio", 0.0)) for it in items]
        )
        entry["query_vector_cosine_to_baseline_mean"] = _mean(
            [float(it.get("query_vector_cosine_to_baseline", 0.0)) for it in items]
        )
        rg_vals = [
            float(it["prefetch_row_groups"])
            for it in items
            if it.get("prefetch_row_groups") is not None
        ]
        entry["prefetch_row_groups_mean"] = _mean(rg_vals) if rg_vals else None

        summary.append(entry)

    return summary


def print_summary_table(summary: list[dict[str, Any]]) -> None:
    if not summary:
        print("No successful runs to summarize.")
        return

    print("\n" + "=" * 120)
    print("HTTPFS HARNESS SUMMARY")
    print("=" * 120)
    print(
        f"{'Dataset':<12} | {'Strategy':<38} | {'N':>6} | {'Search Mean (ms)':>16} | {'Prefetch (ms)':>14} | {'Overlap':>8} | {'Trials':>6}"
    )
    print("-" * 120)

    for row in summary:
        print(
            f"{row['dataset']:<12} | "
            f"{row['strategy']:<38} | "
            f"{row['n_neighbors']:>6} | "
            f"{row['search_path_total_ms_mean']:>16.1f} | "
            f"{row['prefetch_ms_mean']:>14.1f} | "
            f"{row.get('faiss_overlap_ratio_mean', 0.0):>8.3f} | "
            f"{row['trials']:>6}"
        )


def run_single_trial(
    dataset_name: str,
    db_url: str,
    prefetch_db_url: str,
    cache_conn: duckdb.DuckDBPyConnection | None,
    strategy: dict[str, Any],
    index: faiss.Index,
    query_vector: np.ndarray,
    baseline_query_vector: np.ndarray | None,
    query_embedding_expr: str,
    query_ids_used: list[int],
    n_neighbors: int,
    prefetch_top_k: int,
    prefetch_fast_k: int,
    nprobe: int,
    row_group_size: int,
    label_clicks: list[dict[str, float]],
    run_label_queries: bool,
    trial: int,
    row_group_cache_conn: duckdb.DuckDBPyConnection | None,
    connection_pools: dict[str, RemoteConnectionPool],
    capture_faiss_ids: bool,
    embedding_lru_cache: IdLruCache | None,
    user_consumed_k: int,
    reference_faiss_ids: list[int] | None,
) -> dict[str, Any]:
    gc.collect()

    row: dict[str, Any] = {
        "timestamp": _now_iso(),
        "dataset": dataset_name,
        "db_url": db_url,
        "prefetch_db_url": prefetch_db_url,
        "strategy": strategy["name"],
        "metadata_source": strategy["metadata_source"],
        "prefetch_method": strategy["prefetch_method"],
        "prefetch_fetch_mode": str(strategy.get("fetch_mode", "fetchdf")),
        "prefetch_query_mode": str(strategy.get("prefetch_query_mode", "in_list")),
        "metadata_query_mode": str(strategy.get("metadata_query_mode", "in_list")),
        "batch_scheduler": str(strategy.get("batch_scheduler", "as_is")),
        "query_embedding_expr": str(query_embedding_expr),
        "query_ids_used": [int(x) for x in query_ids_used],
        "n_neighbors": int(n_neighbors),
        "prefetch_top_k": int(prefetch_top_k),
        "user_consumed_k": int(user_consumed_k),
        "trial": int(trial),
        "status": "ok",
        "query_expand_ms": 0.0,
        "query_expand_neighbors": 0,
        "query_expand_weight": 0.0,
    }

    conn: duckdb.DuckDBPyConnection | None = None

    try:
        trial_start = time.perf_counter()

        connect_start = time.perf_counter()
        conn = setup_remote_connection(db_url)
        row["connect_ms"] = (time.perf_counter() - connect_start) * 1000.0

        warmup_start = time.perf_counter()
        conn.execute("SELECT id FROM geo_embeddings LIMIT 1").fetchone()
        row["warmup_ms"] = (time.perf_counter() - warmup_start) * 1000.0

        if run_label_queries and label_clicks:
            label_metrics = run_label_stage(conn, label_clicks)
            row["label_total_ms"] = float(label_metrics["total_ms"])
            row["label_mean_ms"] = float(label_metrics["mean_ms"])
            row["label_p95_ms"] = float(label_metrics["p95_ms"])
            row["label_returned_ids"] = int(len(label_metrics["returned_ids"]))
        else:
            row["label_total_ms"] = 0.0
            row["label_mean_ms"] = 0.0
            row["label_p95_ms"] = 0.0
            row["label_returned_ids"] = 0

        effective_query_vector = np.asarray(query_vector, dtype=np.float32)
        if baseline_query_vector is not None:
            row["query_vector_cosine_to_baseline"] = cosine_similarity(
                effective_query_vector, baseline_query_vector
            )
        else:
            row["query_vector_cosine_to_baseline"] = 1.0

        query_expand_top_k = int(strategy.get("query_expand_top_k", 0))
        query_expand_weight = float(strategy.get("query_expand_weight", 0.0))
        query_expand_nprobe = int(strategy.get("query_expand_nprobe", nprobe))
        query_expand_embedding_expr = str(
            strategy.get("query_expand_embedding_expr", query_embedding_expr)
        )
        if query_expand_top_k > 0 and query_expand_weight > 0.0:
            expand_start = time.perf_counter()
            _, expand_ids, _ = run_faiss_search(
                index=index,
                query_vector=effective_query_vector,
                n_neighbors=max(1, query_expand_top_k),
                nprobe=query_expand_nprobe,
            )
            expand_ids = [int(x) for x in expand_ids[:query_expand_top_k]]
            expand_map = fetch_embeddings_by_id(
                conn,
                expand_ids,
                embedding_expr=query_expand_embedding_expr,
            )
            expand_vectors = [
                expand_map[int(id_)] for id_ in expand_ids if int(id_) in expand_map
            ]
            if expand_vectors:
                neighbor_mean = np.mean(expand_vectors, axis=0).astype(np.float32)
                alpha = max(0.0, min(1.0, query_expand_weight))
                effective_query_vector = (
                    (1.0 - alpha) * effective_query_vector + alpha * neighbor_mean
                ).astype(np.float32)
                row["query_expand_neighbors"] = int(len(expand_vectors))
                row["query_expand_weight"] = float(alpha)
            row["query_expand_ms"] = (time.perf_counter() - expand_start) * 1000.0

        faiss_ms, faiss_ids, _ = run_faiss_search(
            index=index,
            query_vector=effective_query_vector,
            n_neighbors=n_neighbors,
            nprobe=nprobe,
        )
        row["faiss_ms"] = float(faiss_ms)
        row["faiss_results"] = int(len(faiss_ids))
        if capture_faiss_ids:
            row["faiss_ids"] = [int(x) for x in faiss_ids]
        if reference_faiss_ids is not None:
            k = min(int(n_neighbors), len(faiss_ids), len(reference_faiss_ids))
            if k > 0:
                overlap = len(set(faiss_ids[:k]) & set(reference_faiss_ids[:k]))
                row["faiss_overlap_k"] = int(k)
                row["faiss_overlap_count"] = int(overlap)
                row["faiss_overlap_ratio"] = float(overlap / k)
            else:
                row["faiss_overlap_k"] = 0
                row["faiss_overlap_count"] = 0
                row["faiss_overlap_ratio"] = 0.0

        metadata_query_mode = str(strategy.get("metadata_query_mode", "in_list"))
        prefetch_top_k_effective = int(strategy.get("prefetch_top_k", prefetch_top_k))
        prefetch_ids_candidate = _as_int_list(faiss_ids[: max(0, prefetch_top_k_effective)])
        row["prefetch_top_k"] = prefetch_top_k_effective
        row["prefetch_candidate_ids"] = int(len(prefetch_ids_candidate))

        cache_hits = 0
        if embedding_lru_cache is not None and prefetch_ids_candidate:
            filtered_ids: list[int] = []
            for id_ in prefetch_ids_candidate:
                if embedding_lru_cache.contains(id_):
                    cache_hits += 1
                else:
                    filtered_ids.append(id_)
            prefetch_ids = filtered_ids
        else:
            prefetch_ids = prefetch_ids_candidate
        row["prefetch_cache_hits"] = int(cache_hits)
        row["prefetch_effective_ids"] = int(len(prefetch_ids))

        method = strategy["prefetch_method"]
        prefetch_include_geometry = bool(strategy.get("prefetch_include_geometry", False))
        prefetch_fetch_mode = str(strategy.get("fetch_mode", "fetchdf"))
        prefetch_query_mode = str(strategy.get("prefetch_query_mode", "in_list"))
        batch_scheduler = str(strategy.get("batch_scheduler", "as_is"))
        batch_shuffle_seed = strategy.get("batch_shuffle_seed")
        embedding_expr = str(strategy.get("embedding_expr", "embedding"))
        pool_mode = bool(strategy.get("use_connection_pool", False))
        pool: RemoteConnectionPool | None = None
        if pool_mode:
            pool_workers = int(strategy.get("n_workers", 32))
            pool_key = f"{prefetch_db_url}|{pool_workers}|{strategy['name']}"
            pool = connection_pools.get(pool_key)
            if pool is None:
                pool = RemoteConnectionPool(db_url=prefetch_db_url, max_size=pool_workers)
                connection_pools[pool_key] = pool

        def run_metadata_stage() -> tuple[float, pd.DataFrame, str]:
            if strategy["metadata_source"] == "geometry_cache" and cache_conn is not None:
                metadata_ms_local, metadata_df_local = fetch_search_metadata_cache(
                    cache_conn,
                    faiss_ids,
                    query_mode=metadata_query_mode,
                )
                return metadata_ms_local, metadata_df_local, "geometry_cache"
            metadata_ms_local, metadata_df_local = fetch_search_metadata_remote(
                conn,
                faiss_ids,
                query_mode=metadata_query_mode,
            )
            return metadata_ms_local, metadata_df_local, "remote"

        def run_prefetch_stage() -> dict[str, Any]:
            if method == "single_query":
                prefetch_conn = conn
                if prefetch_db_url != db_url:
                    prefetch_conn = setup_remote_connection(prefetch_db_url)
                try:
                    return fetch_embeddings_single_query(
                        prefetch_conn,
                        prefetch_ids,
                        include_geometry=prefetch_include_geometry,
                        query_mode=prefetch_query_mode,
                        embedding_expr=embedding_expr,
                    )
                finally:
                    if prefetch_conn is not conn:
                        prefetch_conn.close()
            if method == "duckdb_native_threads":
                return fetch_embeddings_duckdb_native(
                    db_url=prefetch_db_url,
                    ids=prefetch_ids,
                    n_threads=int(strategy.get("n_threads", 32)),
                    include_geometry=prefetch_include_geometry,
                    query_mode=prefetch_query_mode,
                    embedding_expr=embedding_expr,
                )
            if method == "id_div_rowgroup_threadpool_new_conn":
                return fetch_embeddings_threadpool(
                    db_url=prefetch_db_url,
                    ids=prefetch_ids,
                    row_group_size=row_group_size,
                    n_workers=int(strategy.get("n_workers", 32)),
                    group_mode="id_div",
                    reuse_connections=False,
                    rowid_source_conn=None,
                    row_group_cache_conn=None,
                    include_geometry=prefetch_include_geometry,
                    fetch_mode=prefetch_fetch_mode,
                    query_mode=prefetch_query_mode,
                    batch_scheduler=batch_scheduler,
                    batch_shuffle_seed=batch_shuffle_seed,
                    embedding_expr=embedding_expr,
                )
            if method == "id_div_rowgroup_threadpool_reuse_conn":
                return fetch_embeddings_threadpool(
                    db_url=prefetch_db_url,
                    ids=prefetch_ids,
                    row_group_size=row_group_size,
                    n_workers=int(strategy.get("n_workers", 32)),
                    group_mode="id_div",
                    reuse_connections=True,
                    rowid_source_conn=None,
                    row_group_cache_conn=None,
                    connection_pool=pool,
                    include_geometry=prefetch_include_geometry,
                    fetch_mode=prefetch_fetch_mode,
                    query_mode=prefetch_query_mode,
                    batch_scheduler=batch_scheduler,
                    batch_shuffle_seed=batch_shuffle_seed,
                    embedding_expr=embedding_expr,
                )
            if method == "rowid_rowgroup_threadpool_reuse_conn":
                rowid_source_conn_local = conn
                if prefetch_db_url != db_url:
                    rowid_source_conn_local = setup_remote_connection(prefetch_db_url)
                try:
                    return fetch_embeddings_threadpool(
                        db_url=prefetch_db_url,
                        ids=prefetch_ids,
                        row_group_size=row_group_size,
                        n_workers=int(strategy.get("n_workers", 32)),
                        group_mode="rowid",
                        reuse_connections=True,
                        rowid_source_conn=rowid_source_conn_local,
                        row_group_cache_conn=None,
                        connection_pool=pool,
                        include_geometry=prefetch_include_geometry,
                        fetch_mode=prefetch_fetch_mode,
                        query_mode=prefetch_query_mode,
                        batch_scheduler=batch_scheduler,
                        batch_shuffle_seed=batch_shuffle_seed,
                        embedding_expr=embedding_expr,
                    )
                finally:
                    if rowid_source_conn_local is not conn:
                        rowid_source_conn_local.close()
            if method == "rowgroup_cache_threadpool_reuse_conn":
                return fetch_embeddings_threadpool(
                    db_url=prefetch_db_url,
                    ids=prefetch_ids,
                    row_group_size=row_group_size,
                    n_workers=int(strategy.get("n_workers", 32)),
                    group_mode="row_group_cache",
                    reuse_connections=True,
                    rowid_source_conn=None,
                    row_group_cache_conn=row_group_cache_conn,
                    connection_pool=pool,
                    include_geometry=prefetch_include_geometry,
                    fetch_mode=prefetch_fetch_mode,
                    query_mode=prefetch_query_mode,
                    batch_scheduler=batch_scheduler,
                    batch_shuffle_seed=batch_shuffle_seed,
                    embedding_expr=embedding_expr,
                )
            if method == "rowgroup_cache_threadpool_new_conn":
                return fetch_embeddings_threadpool(
                    db_url=prefetch_db_url,
                    ids=prefetch_ids,
                    row_group_size=row_group_size,
                    n_workers=int(strategy.get("n_workers", 32)),
                    group_mode="row_group_cache",
                    reuse_connections=False,
                    rowid_source_conn=None,
                    row_group_cache_conn=row_group_cache_conn,
                    include_geometry=prefetch_include_geometry,
                    fetch_mode=prefetch_fetch_mode,
                    query_mode=prefetch_query_mode,
                    batch_scheduler=batch_scheduler,
                    batch_shuffle_seed=batch_shuffle_seed,
                    embedding_expr=embedding_expr,
                )
            if method == "rowgroup_cache_threadpool_reuse_conn_two_stage":
                return fetch_embeddings_two_stage_threadpool(
                    db_url=prefetch_db_url,
                    ids=prefetch_ids,
                    row_group_size=row_group_size,
                    n_workers=int(strategy.get("n_workers", 32)),
                    group_mode="row_group_cache",
                    reuse_connections=True,
                    rowid_source_conn=None,
                    row_group_cache_conn=row_group_cache_conn,
                    fast_k=int(strategy.get("fast_k", prefetch_fast_k)),
                    connection_pool=pool,
                    include_geometry=prefetch_include_geometry,
                    fetch_mode=prefetch_fetch_mode,
                    query_mode=prefetch_query_mode,
                    batch_scheduler=batch_scheduler,
                    batch_shuffle_seed=batch_shuffle_seed,
                    embedding_expr=embedding_expr,
                )
            if method == "id_div_rowgroup_threadpool_reuse_conn_two_stage":
                return fetch_embeddings_two_stage_threadpool(
                    db_url=prefetch_db_url,
                    ids=prefetch_ids,
                    row_group_size=row_group_size,
                    n_workers=int(strategy.get("n_workers", 32)),
                    group_mode="id_div",
                    reuse_connections=True,
                    rowid_source_conn=None,
                    row_group_cache_conn=None,
                    fast_k=int(strategy.get("fast_k", prefetch_fast_k)),
                    connection_pool=pool,
                    include_geometry=prefetch_include_geometry,
                    fetch_mode=prefetch_fetch_mode,
                    query_mode=prefetch_query_mode,
                    batch_scheduler=batch_scheduler,
                    batch_shuffle_seed=batch_shuffle_seed,
                    embedding_expr=embedding_expr,
                )
            raise RuntimeError(f"Unknown prefetch method: {method}")

        overlap_mode = str(strategy.get("overlap_mode", "none")).strip().lower()
        if overlap_mode == "metadata_prefetch":
            overlap_start = time.perf_counter()
            with ThreadPoolExecutor(max_workers=2) as executor:
                metadata_future = executor.submit(run_metadata_stage)
                prefetch_future = executor.submit(run_prefetch_stage)
                metadata_ms, metadata_df, metadata_source_effective = metadata_future.result()
                prefetch = prefetch_future.result()
            overlap_elapsed_ms = (time.perf_counter() - overlap_start) * 1000.0
            row["overlap_mode_effective"] = "metadata_prefetch"
        else:
            metadata_ms, metadata_df, metadata_source_effective = run_metadata_stage()
            prefetch = run_prefetch_stage()
            overlap_elapsed_ms = float(metadata_ms + float(prefetch.get("elapsed_ms", 0.0)))
            row["overlap_mode_effective"] = "none"

        row["metadata_source_effective"] = metadata_source_effective
        row["metadata_ms"] = float(metadata_ms)
        row["metadata_rows"] = int(len(metadata_df))

        row["prefetch_ms"] = float(prefetch.get("elapsed_ms", 0.0))
        row["prefetch_fetched"] = int(prefetch.get("fetched", 0))
        row["prefetch_batches"] = int(prefetch.get("batches", 0))
        row["prefetch_row_groups"] = prefetch.get("row_groups")
        row["prefetch_rowid_lookup_ms"] = float(prefetch.get("rowid_lookup_ms", 0.0))
        row["prefetch_fast_ms"] = float(prefetch.get("fast_stage_ms", 0.0))
        row["prefetch_tail_ms"] = float(prefetch.get("tail_stage_ms", 0.0))
        row["prefetch_fast_fetched"] = int(prefetch.get("fast_stage_fetched", 0))
        row["prefetch_tail_fetched"] = int(prefetch.get("tail_stage_fetched", 0))
        row["prefetch_wasted_ids"] = max(0, int(row["prefetch_effective_ids"]) - int(user_consumed_k))
        row["prefetch_wasted_ratio"] = (
            float(row["prefetch_wasted_ids"]) / float(row["prefetch_effective_ids"])
            if int(row["prefetch_effective_ids"]) > 0
            else 0.0
        )
        if embedding_lru_cache is not None:
            embedding_lru_cache.add_many(prefetch_ids)
            row["embedding_cache_size"] = embedding_lru_cache.size()
        else:
            row["embedding_cache_size"] = 0

        if row["overlap_mode_effective"] == "metadata_prefetch":
            row["search_path_total_ms"] = float(
                row["query_expand_ms"] + row["faiss_ms"] + overlap_elapsed_ms
            )
        else:
            row["search_path_total_ms"] = float(
                row["query_expand_ms"] + row["faiss_ms"] + row["metadata_ms"] + row["prefetch_ms"]
            )
        row["trial_total_ms"] = float((time.perf_counter() - trial_start) * 1000.0)
        return row

    except Exception as exc:
        row["status"] = "error"
        row["error"] = f"{type(exc).__name__}: {exc}"
        row["traceback"] = traceback.format_exc()
        return row
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass


def run_harness(config: dict[str, Any]) -> dict[str, Any]:
    source_cfg = config["source"]
    workload_cfg = config["workload"]
    strategy_cfg = config["strategies"]

    db_variants: list[dict[str, Any]] = source_cfg["db_variants"]
    faiss_url = source_cfg["faiss_url"]
    query_ids = _as_int_list(source_cfg["query_ids"])
    query_id_triplets_cfg = source_cfg.get("query_id_triplets")
    label_clicks = source_cfg["label_clicks"]

    n_trials = int(workload_cfg["n_trials"])
    n_neighbors_values = [int(x) for x in workload_cfg["n_neighbors_values"]]
    prefetch_top_k = int(workload_cfg["prefetch_top_k"])
    prefetch_fast_k = int(workload_cfg.get("prefetch_fast_k", min(prefetch_top_k, 200)))
    row_group_size = int(workload_cfg.get("row_group_size", 122880))
    nprobe = int(workload_cfg.get("nprobe", 64))
    reference_nprobe = int(workload_cfg.get("reference_nprobe", nprobe))
    capture_faiss_ids = bool(workload_cfg.get("capture_faiss_ids", False))
    pool_scope = str(workload_cfg.get("pool_scope", "run")).strip().lower()
    run_order = str(workload_cfg.get("run_order", "strategy_major")).strip().lower()
    run_seed = workload_cfg.get("run_seed")
    run_label_queries = bool(workload_cfg.get("run_label_queries", True))
    prefetch_user_consumed_k = int(workload_cfg.get("prefetch_user_consumed_k", 100))
    embedding_lru_size = int(workload_cfg.get("embedding_lru_size", 0))
    embedding_cache_scope = str(workload_cfg.get("embedding_cache_scope", "run")).strip().lower()
    query_embedding_expr_default = str(workload_cfg.get("query_embedding_expr", "embedding"))
    reference_query_embedding_expr = str(
        workload_cfg.get("reference_query_embedding_expr", query_embedding_expr_default)
    )
    simulate_feedback = bool(workload_cfg.get("simulate_feedback", False))
    feedback_steps_per_seed = int(workload_cfg.get("feedback_steps_per_seed", 1))
    feedback_top_k = int(workload_cfg.get("feedback_top_k", 100))
    feedback_nprobe = int(workload_cfg.get("feedback_nprobe", nprobe))
    feedback_add_positive = int(workload_cfg.get("feedback_add_positive", 1))
    feedback_add_negative = int(workload_cfg.get("feedback_add_negative", 0))
    feedback_max_positive_ids = int(workload_cfg.get("feedback_max_positive_ids", 8))
    feedback_max_negative_ids = int(workload_cfg.get("feedback_max_negative_ids", 4))
    runtime_settings_cfg = workload_cfg.get("runtime_settings", {}) or {}
    runtime_enable_object_cache = bool(
        runtime_settings_cfg.get("enable_object_cache", False)
    )
    runtime_extra_sql = [str(x) for x in runtime_settings_cfg.get("extra_sql", [])]
    configure_runtime_settings(
        enable_object_cache=runtime_enable_object_cache,
        extra_sql=runtime_extra_sql,
    )

    print("=" * 88)
    print("HTTPFS End-to-End Retrieval Harness")
    print("=" * 88)
    print(f"UTC start: {_now_iso()}")
    print(f"FAISS URL: {faiss_url}")
    print(f"Dataset variants: {[v['name'] for v in db_variants]}")
    print(f"Strategies: {[s['name'] for s in strategy_cfg]}")
    print(
        "Trials: "
        f"{n_trials}, neighbors: {n_neighbors_values}, "
        f"prefetch_top_k={prefetch_top_k}, prefetch_fast_k={prefetch_fast_k}, "
        f"capture_faiss_ids={capture_faiss_ids}, pool_scope={pool_scope}, "
        f"run_order={run_order}, run_label_queries={run_label_queries}, "
        f"embedding_lru_size={embedding_lru_size}, embedding_cache_scope={embedding_cache_scope}, "
        f"runtime_enable_object_cache={runtime_enable_object_cache}, "
        f"query_embedding_expr={query_embedding_expr_default}, "
        f"simulate_feedback={simulate_feedback}"
    )

    index = load_faiss_index(faiss_url)
    print(f"Loaded FAISS index: ntotal={index.ntotal:,} dim={index.d}")

    if query_id_triplets_cfg:
        query_triplets = [_as_int_list(tri) for tri in query_id_triplets_cfg]
    else:
        query_triplets = [query_ids]

    # Build query vectors from the first dataset variant.
    query_conn = setup_remote_connection(db_variants[0]["db_url"])
    try:
        if simulate_feedback:
            query_triplets = generate_iterative_query_sets(
                query_conn,
                index,
                query_triplets,
                embedding_expr=query_embedding_expr_default,
                feedback_nprobe=feedback_nprobe,
                feedback_top_k=feedback_top_k,
                steps_per_seed=feedback_steps_per_seed,
                add_positive_per_step=feedback_add_positive,
                add_negative_per_step=feedback_add_negative,
                max_positive_ids=feedback_max_positive_ids,
                max_negative_ids=feedback_max_negative_ids,
            )
            print(
                "Generated iterative query sets: "
                f"{len(query_triplets)} (from {len(query_id_triplets_cfg or [query_ids])} seeds)"
            )

        query_exprs = {query_embedding_expr_default, reference_query_embedding_expr}
        for strategy in strategy_cfg:
            query_exprs.add(str(strategy.get("query_embedding_expr", query_embedding_expr_default)))

        query_vectors_by_expr: dict[str, dict[tuple[int, ...], np.ndarray]] = {}
        for expr in sorted(query_exprs):
            expr_map: dict[tuple[int, ...], np.ndarray] = {}
            for tri in query_triplets:
                key = tuple(_as_int_list(tri))
                expr_map[key] = build_query_vector_from_ids(
                    query_conn,
                    list(key),
                    embedding_expr=expr,
                )
            query_vectors_by_expr[expr] = expr_map

        reference_ids_by_query: dict[tuple[tuple[int, ...], int], list[int]] = {}
        ref_vecs = query_vectors_by_expr[reference_query_embedding_expr]
        for tri in query_triplets:
            key = tuple(_as_int_list(tri))
            ref_vec = ref_vecs[key]
            for n_neighbors in n_neighbors_values:
                _, ref_ids, _ = run_faiss_search(
                    index=index,
                    query_vector=ref_vec,
                    n_neighbors=n_neighbors,
                    nprobe=reference_nprobe,
                )
                reference_ids_by_query[(key, int(n_neighbors))] = [int(x) for x in ref_ids]
    finally:
        query_conn.close()

    rows: list[dict[str, Any]] = []
    connection_pools: dict[str, RemoteConnectionPool] = {}

    for variant in db_variants:
        dataset_name = str(variant["name"])
        db_url = str(variant["db_url"])
        prefetch_db_url = str(variant.get("prefetch_db_url", db_url))
        geometry_cache_url = variant.get("geometry_cache_url")
        row_group_cache_url = variant.get("row_group_cache_url")

        cache_conn: duckdb.DuckDBPyConnection | None = None
        cache_local_path: str | None = None
        row_group_cache_conn: duckdb.DuckDBPyConnection | None = None
        row_group_cache_path: str | None = None
        if geometry_cache_url:
            try:
                cache_conn, cache_local_path = setup_geometry_cache_connection(
                    str(geometry_cache_url)
                )
                print(f"[{dataset_name}] geometry cache ready: {cache_local_path}")
            except Exception as exc:
                print(f"[{dataset_name}] geometry cache unavailable: {exc}")
                cache_conn = None
                cache_local_path = None

        needs_row_group_cache = any(
            str(s.get("prefetch_method", "")).startswith("rowgroup_cache_threadpool_")
            for s in strategy_cfg
        )
        if needs_row_group_cache:
            rg_cache_dir = (
                Path.home() / ".cache" / "geovibes" / "row_groups"
            )
            if row_group_cache_url:
                rg_path = resolve_local_parquet_path(str(row_group_cache_url))
            else:
                rg_path = ensure_row_group_cache(
                    db_url=prefetch_db_url,
                    row_group_size=row_group_size,
                    cache_dir=rg_cache_dir,
                )
            row_group_cache_conn = setup_row_group_cache_connection(rg_path)
            row_group_cache_path = str(rg_path)
            print(f"[{dataset_name}] row-group cache ready: {row_group_cache_path}")

        embedding_lru_cache: IdLruCache | None = None
        if embedding_lru_size > 0 and embedding_cache_scope in {"run", "variant"}:
            embedding_lru_cache = IdLruCache(embedding_lru_size)

        run_plan: list[tuple[dict[str, Any], int, int]] = []
        if run_order == "strategy_major":
            for strategy in strategy_cfg:
                for n_neighbors in n_neighbors_values:
                    for trial in range(1, n_trials + 1):
                        run_plan.append((strategy, n_neighbors, trial))
        elif run_order == "trial_major":
            for n_neighbors in n_neighbors_values:
                for trial in range(1, n_trials + 1):
                    for strategy in strategy_cfg:
                        run_plan.append((strategy, n_neighbors, trial))
        elif run_order == "randomized":
            for n_neighbors in n_neighbors_values:
                for trial in range(1, n_trials + 1):
                    for strategy in strategy_cfg:
                        run_plan.append((strategy, n_neighbors, trial))
            rng = random.Random(run_seed)
            rng.shuffle(run_plan)
        else:
            raise RuntimeError(
                f"Unknown run_order={run_order}. Use strategy_major, trial_major, or randomized."
            )

        current_cache_strategy_name: str | None = None
        for strategy, n_neighbors, trial in run_plan:
            if pool_scope == "trial":
                for pool in connection_pools.values():
                    try:
                        pool.close()
                    except Exception:
                        pass
                connection_pools = {}
            if embedding_lru_size > 0 and embedding_cache_scope == "trial":
                embedding_lru_cache = IdLruCache(embedding_lru_size)
            elif embedding_lru_size > 0 and embedding_cache_scope == "strategy":
                strategy_name = str(strategy.get("name", ""))
                if current_cache_strategy_name != strategy_name:
                    embedding_lru_cache = IdLruCache(embedding_lru_size)
                    current_cache_strategy_name = strategy_name

            triplet_idx = (trial - 1) % len(query_triplets)
            query_triplet = query_triplets[triplet_idx]
            query_key = tuple(_as_int_list(query_triplet))
            strategy_query_expr = str(
                strategy.get("query_embedding_expr", query_embedding_expr_default)
            )
            query_vector = query_vectors_by_expr[strategy_query_expr][query_key]
            baseline_query_vector = query_vectors_by_expr[reference_query_embedding_expr][query_key]
            reference_faiss_ids = reference_ids_by_query.get((query_key, int(n_neighbors)))
            print(
                f"[{dataset_name}] {strategy['name']} n={n_neighbors} trial={trial}/{n_trials} q={query_triplet}"
            )
            row = run_single_trial(
                dataset_name=dataset_name,
                db_url=db_url,
                prefetch_db_url=prefetch_db_url,
                cache_conn=cache_conn,
                strategy=strategy,
                index=index,
                query_vector=query_vector,
                baseline_query_vector=baseline_query_vector,
                query_embedding_expr=strategy_query_expr,
                query_ids_used=query_triplet,
                n_neighbors=n_neighbors,
                prefetch_top_k=prefetch_top_k,
                prefetch_fast_k=prefetch_fast_k,
                nprobe=nprobe,
                row_group_size=row_group_size,
                label_clicks=label_clicks,
                run_label_queries=run_label_queries,
                trial=trial,
                row_group_cache_conn=row_group_cache_conn,
                connection_pools=connection_pools,
                capture_faiss_ids=capture_faiss_ids,
                embedding_lru_cache=embedding_lru_cache,
                user_consumed_k=prefetch_user_consumed_k,
                reference_faiss_ids=reference_faiss_ids,
            )
            row["geometry_cache_local_path"] = cache_local_path
            row["row_group_cache_local_path"] = row_group_cache_path
            rows.append(row)
            if row["status"] == "ok":
                print(
                    "  -> "
                    f"search_path={row['search_path_total_ms']:.1f}ms "
                    f"(faiss={row['faiss_ms']:.1f}, metadata={row['metadata_ms']:.1f}, prefetch={row['prefetch_ms']:.1f})"
                )
            else:
                print(f"  -> ERROR: {row['error']}")

        if cache_conn is not None:
            cache_conn.close()
        if row_group_cache_conn is not None:
            row_group_cache_conn.close()

    for pool in connection_pools.values():
        try:
            pool.close()
        except Exception:
            pass

    summary = summarize_results(rows)
    print_summary_table(summary)

    return {
        "metadata": {
            "timestamp_utc": _now_iso(),
            "faiss_url": faiss_url,
            "query_ids": query_ids,
            "workload": workload_cfg,
            "strategies": strategy_cfg,
            "db_variants": db_variants,
        },
        "results": rows,
        "summary": summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run end-to-end httpfs retrieval harness")
    parser.add_argument(
        "--config",
        default=str(Path(__file__).parent / "config.yaml"),
        help="Path to harness config YAML",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional output JSON path. Defaults to timestamped file in results/",
    )
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    output_dir = Path(config.get("output", {}).get("directory", "results"))
    if not output_dir.is_absolute():
        output_dir = config_path.parent / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    report = run_harness(config)

    if args.output:
        output_path = Path(args.output).resolve()
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"harness_results_{ts}.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\nSaved harness report: {output_path}")


if __name__ == "__main__":
    main()
