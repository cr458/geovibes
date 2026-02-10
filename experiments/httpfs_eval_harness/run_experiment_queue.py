#!/usr/bin/env python3
"""Run a systematic optimization experiment queue for httpfs retrieval."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parent
RUN_HARNESS = ROOT / "run_harness.py"
RESULTS_DIR = ROOT / "results"
QUEUE_DIR = ROOT / "queue"
QUEUE_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


BASE_SOURCE = {
    "faiss_url": (
        "s3://us-west-2.opendata.source.coop/geovibes/search/USA/alabama/"
        "2024-01-01-2025-01-01/httpfs/"
        "alabama_dino_vit_small_patch16_224_2024_2025_32_16_10/faiss.index"
    ),
    "db_variants": [
        {
            "name": "baseline",
            "db_url": (
                "s3://us-west-2.opendata.source.coop/geovibes/search/USA/alabama/"
                "2024-01-01-2025-01-01/httpfs/"
                "alabama_dino_vit_small_patch16_224_2024_2025_32_16_10/metadata.db"
            ),
            "geometry_cache_url": (
                "s3://us-west-2.opendata.source.coop/geovibes/search/USA/alabama/"
                "2024-01-01-2025-01-01/httpfs/"
                "alabama_dino_vit_small_patch16_224_2024_2025_32_16_10/geometry_cache.parquet"
            ),
        }
    ],
    "query_ids": [2660000, 1000000, 4000000],
    "query_id_triplets": [
        [78824, 1033427, 4970394],
        [592921, 215330, 4384211],
        [1686758, 2662800, 5070803],
    ],
    "label_clicks": [
        {"lon": -86.55, "lat": 32.35},
        {"lon": -86.30, "lat": 32.55},
    ],
}


BASE_WORKLOAD = {
    "n_trials": 1,
    "n_neighbors_values": [1000],
    "prefetch_top_k": 1000,
    "prefetch_fast_k": 100,
    "nprobe": 1024,
    "reference_nprobe": 4096,
    "row_group_size": 122880,
    "pool_scope": "trial",
    "run_order": "trial_major",
    "run_seed": 20260210,
    "capture_faiss_ids": True,
    "run_label_queries": False,
    "query_embedding_expr": "embedding",
    "simulate_feedback": False,
    "feedback_steps_per_seed": 1,
    "feedback_top_k": 100,
    "feedback_nprobe": 1024,
    "feedback_add_positive": 1,
    "feedback_add_negative": 0,
    "feedback_max_positive_ids": 8,
    "feedback_max_negative_ids": 4,
    "prefetch_user_consumed_k": 100,
    "embedding_lru_size": 0,
    "embedding_cache_scope": "run",
    "runtime_settings": {"enable_object_cache": False, "extra_sql": []},
}


def base_strategy(name: str, **overrides: Any) -> dict[str, Any]:
    out = {
        "name": name,
        "metadata_source": "geometry_cache",
        "prefetch_method": "rowgroup_cache_threadpool_reuse_conn",
        "n_workers": 16,
        "use_connection_pool": True,
        "fetch_mode": "fetchall",
    }
    out.update(overrides)
    return out


def make_task(
    task_id: str,
    lane: str,
    description: str,
    *,
    source_overrides: dict[str, Any] | None = None,
    workload_overrides: dict[str, Any] | None = None,
    strategies: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    source = deepcopy(BASE_SOURCE)
    workload = deepcopy(BASE_WORKLOAD)
    if source_overrides:
        source.update(source_overrides)
    if workload_overrides:
        workload.update(workload_overrides)
    return {
        "id": task_id,
        "lane": lane,
        "description": description,
        "config": {
            "source": source,
            "workload": workload,
            "strategies": strategies or [base_strategy("baseline")],
            "output": {"directory": "results"},
        },
    }


def task_queue() -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []

    # Lane A: query-mode rewrite
    tasks.append(
        make_task(
            "q01_query_mode",
            "query",
            "IN-list vs VALUES-join query forms for metadata/prefetch",
            strategies=[
                base_strategy("in_list_default"),
                base_strategy("prefetch_values_join", prefetch_query_mode="values_join"),
                base_strategy("metadata_values_join", metadata_query_mode="values_join"),
                base_strategy(
                    "both_values_join",
                    prefetch_query_mode="values_join",
                    metadata_query_mode="values_join",
                ),
            ],
        )
    )

    # Lane B: row-group-size proxy sweep (grouping granularity)
    for rg in (65536, 122880, 245760):
        tasks.append(
            make_task(
                f"q02_rowgroup_proxy_rg{rg}",
                "layout",
                f"Proxy grouping granularity sweep row_group_size={rg}",
                workload_overrides={"row_group_size": rg},
                strategies=[
                    base_strategy(
                        f"id_div_pool_w16_rg{rg}",
                        prefetch_method="id_div_rowgroup_threadpool_reuse_conn",
                    )
                ],
            )
        )

    # Lane C: FAISS nprobe sweep
    for nprobe in (128, 256, 512, 1024, 2048, 4096):
        tasks.append(
            make_task(
                f"q03_nprobe_{nprobe}",
                "index",
                f"FAISS nprobe sweep nprobe={nprobe}",
                workload_overrides={"nprobe": nprobe},
                strategies=[base_strategy(f"rowgroup_pool_nprobe_{nprobe}")],
            )
        )

    # Lane D: adaptive prefetch depth
    tasks.append(
        make_task(
            "q04_prefetch_depth",
            "prefetch",
            "Adaptive prefetch depth (200/500/1000 + two-stage)",
            strategies=[
                base_strategy("top200", prefetch_top_k=200),
                base_strategy("top500", prefetch_top_k=500),
                base_strategy("top1000", prefetch_top_k=1000),
                base_strategy(
                    "two_stage_fast100_top1000",
                    prefetch_method="rowgroup_cache_threadpool_reuse_conn_two_stage",
                    fast_k=100,
                    prefetch_top_k=1000,
                ),
            ],
        )
    )

    # Lane E: batch scheduling
    tasks.append(
        make_task(
            "q05_batch_scheduler",
            "scheduler",
            "Batch scheduling policy sweep",
            strategies=[
                base_strategy("sched_as_is"),
                base_strategy("sched_largest_first", batch_scheduler="largest_first"),
                base_strategy("sched_id_ascending", batch_scheduler="id_ascending"),
                base_strategy(
                    "sched_random_seed42",
                    batch_scheduler="random",
                    batch_shuffle_seed=42,
                ),
            ],
        )
    )

    # Lane F: pool sizing and lifecycle
    tasks.append(
        make_task(
            "q06_pool_size_strict",
            "pool",
            "Pool size strict (pool_scope=trial)",
            strategies=[
                base_strategy("pool_w8", n_workers=8),
                base_strategy("pool_w16", n_workers=16),
                base_strategy("pool_w32", n_workers=32),
            ],
        )
    )
    tasks.append(
        make_task(
            "q07_pool_size_warm",
            "pool",
            "Pool size warm (pool_scope=run)",
            workload_overrides={"pool_scope": "run"},
            strategies=[
                base_strategy("pool_w8", n_workers=8),
                base_strategy("pool_w16", n_workers=16),
                base_strategy("pool_w32", n_workers=32),
            ],
        )
    )

    # Lane G: cross-search cache policy
    for cache_size in (0, 500, 2000, 5000):
        tasks.append(
            make_task(
                f"q08_cache_lru_{cache_size}",
                "cache",
                f"Embedding ID LRU simulation size={cache_size}",
                workload_overrides={
                    "embedding_lru_size": cache_size,
                    "embedding_cache_scope": "run",
                    "n_trials": 3,
                    "n_neighbors_values": [500],
                    "run_order": "trial_major",
                },
                strategies=[base_strategy(f"cache_lru_{cache_size}", prefetch_top_k=1000)],
            )
        )

    # Lane H: metadata/prefetch overlap
    tasks.append(
        make_task(
            "q09_overlap",
            "overlap",
            "Metadata/prefetch overlap mode",
            strategies=[
                base_strategy("no_overlap_remote_meta", metadata_source="remote"),
                base_strategy(
                    "overlap_remote_meta",
                    metadata_source="remote",
                    overlap_mode="metadata_prefetch",
                ),
            ],
        )
    )

    # Lane I: storage dtype simulation (quantize/dequantize expression)
    quant_expr = (
        "LIST_TRANSFORM(CAST(embedding AS FLOAT[]), "
        "x -> CAST(CAST(ROUND(x * 1000.0) AS SMALLINT) AS FLOAT) / 1000.0)"
    )
    tasks.append(
        make_task(
            "q10_dtype_sim",
            "dtype",
            "Embedding quantize/dequantize expression overhead simulation",
            strategies=[
                base_strategy("dtype_float_base"),
                base_strategy("dtype_quant_sim", embedding_expr=quant_expr),
            ],
        )
    )

    # Lane J: runtime setting matrix
    tasks.append(
        make_task(
            "q11_runtime_object_cache_off",
            "runtime",
            "Runtime setting sweep: enable_object_cache=false",
            workload_overrides={
                "runtime_settings": {"enable_object_cache": False, "extra_sql": []}
            },
        )
    )
    tasks.append(
        make_task(
            "q12_runtime_object_cache_on",
            "runtime",
            "Runtime setting sweep: enable_object_cache=true",
            workload_overrides={
                "runtime_settings": {"enable_object_cache": True, "extra_sql": []}
            },
        )
    )

    # Lane K: cache realism with overlapping search sequence
    overlap_triplets = [
        list(BASE_SOURCE["query_id_triplets"][0]),
        list(BASE_SOURCE["query_id_triplets"][1]),
        list(BASE_SOURCE["query_id_triplets"][0]),
        list(BASE_SOURCE["query_id_triplets"][1]),
        list(BASE_SOURCE["query_id_triplets"][0]),
        list(BASE_SOURCE["query_id_triplets"][1]),
    ]
    for cache_size in (0, 500, 2000):
        tasks.append(
            make_task(
                f"q13_cache_overlap_lru_{cache_size}",
                "cache_overlap",
                f"Overlapping-query cache realism sweep size={cache_size}",
                source_overrides={"query_id_triplets": overlap_triplets},
                workload_overrides={
                    "n_trials": 6,
                    "n_neighbors_values": [500],
                    "run_order": "trial_major",
                    "embedding_lru_size": cache_size,
                    "embedding_cache_scope": "run",
                    "prefetch_user_consumed_k": 100,
                },
                strategies=[base_strategy(f"cache_overlap_lru_{cache_size}")],
            )
        )

    # Lane L: adaptive prefetch confirmation with random run order
    tasks.append(
        make_task(
            "q14_prefetch_depth_randomized",
            "prefetch",
            "Prefetch depth confirmation with randomized run order",
            workload_overrides={
                "n_trials": 3,
                "n_neighbors_values": [1000],
                "run_order": "randomized",
                "run_seed": 20260211,
            },
            strategies=[
                base_strategy("top100", prefetch_top_k=100),
                base_strategy("top200", prefetch_top_k=200),
                base_strategy("top500", prefetch_top_k=500),
                base_strategy("top1000", prefetch_top_k=1000),
            ],
        )
    )

    # Lane M: query-mode confirmation with random run order
    tasks.append(
        make_task(
            "q15_query_mode_randomized",
            "query",
            "Query-mode rewrite confirmation with randomized ordering",
            workload_overrides={
                "n_trials": 3,
                "run_order": "randomized",
                "run_seed": 20260211,
            },
            strategies=[
                base_strategy("in_list_default"),
                base_strategy("metadata_values_join", metadata_query_mode="values_join"),
                base_strategy(
                    "both_values_join",
                    prefetch_query_mode="values_join",
                    metadata_query_mode="values_join",
                ),
            ],
        )
    )

    # Lane N: query-mode isolation (metadata rewrite only)
    tasks.append(
        make_task(
            "q16_query_mode_metadata_only",
            "query",
            "Isolate metadata VALUES-join impact without prefetch rewrite",
            workload_overrides={
                "n_trials": 5,
                "run_order": "randomized",
                "run_seed": 20260212,
            },
            strategies=[
                base_strategy("in_list_default"),
                base_strategy("metadata_values_join", metadata_query_mode="values_join"),
            ],
        )
    )

    # Lane O: pool-size extension with OPus-like worker count
    tasks.append(
        make_task(
            "q17_pool_size_strict_w44",
            "pool",
            "Pool size strict extension including w44",
            strategies=[
                base_strategy("pool_w16", n_workers=16),
                base_strategy("pool_w32", n_workers=32),
                base_strategy("pool_w44", n_workers=44),
            ],
        )
    )
    tasks.append(
        make_task(
            "q18_pool_size_warm_w44",
            "pool",
            "Pool size warm extension including w44",
            workload_overrides={"pool_scope": "run"},
            strategies=[
                base_strategy("pool_w16", n_workers=16),
                base_strategy("pool_w32", n_workers=32),
                base_strategy("pool_w44", n_workers=44),
            ],
        )
    )

    # Lane P: pool-size confirmation with randomized multi-trial runs
    tasks.append(
        make_task(
            "q19_pool_size_strict_w44_randomized",
            "pool",
            "Pool size strict randomized confirmation (w16/w32/w44)",
            workload_overrides={
                "n_trials": 3,
                "run_order": "randomized",
                "run_seed": 20260213,
                "pool_scope": "trial",
            },
            strategies=[
                base_strategy("pool_w16", n_workers=16),
                base_strategy("pool_w32", n_workers=32),
                base_strategy("pool_w44", n_workers=44),
            ],
        )
    )
    tasks.append(
        make_task(
            "q20_pool_size_warm_w44_randomized",
            "pool",
            "Pool size warm randomized confirmation (w16/w32/w44)",
            workload_overrides={
                "n_trials": 3,
                "run_order": "randomized",
                "run_seed": 20260213,
                "pool_scope": "run",
            },
            strategies=[
                base_strategy("pool_w16", n_workers=16),
                base_strategy("pool_w32", n_workers=32),
                base_strategy("pool_w44", n_workers=44),
            ],
        )
    )

    # Lane Q: embedding expression overhead (native fixed array vs cast)
    tasks.append(
        make_task(
            "q21_embedding_expr_native_vs_cast",
            "dtype",
            "Compare native embedding fetch vs CAST(embedding AS FLOAT[]) overhead",
            workload_overrides={
                "n_trials": 3,
                "run_order": "randomized",
                "run_seed": 20260214,
            },
            strategies=[
                base_strategy("embedding_native_expr", embedding_expr="embedding"),
                base_strategy(
                    "embedding_cast_float_list",
                    embedding_expr="CAST(embedding AS FLOAT[])",
                ),
            ],
        )
    )

    # Lane R: notebook-like iterative usage pattern, before/after optimization
    tasks.append(
        make_task(
            "q22_realworld_iterative_baseline_vs_opt",
            "realworld",
            "Notebook-style iterative search sessions over many searches (baseline vs optimized)",
            workload_overrides={
                "n_trials": 30,
                "n_neighbors_values": [1000],
                "prefetch_top_k": 200,
                "prefetch_fast_k": 100,
                "nprobe": 1024,
                "reference_nprobe": 4096,
                "run_order": "trial_major",
                "run_seed": 20260214,
                "simulate_feedback": True,
                "feedback_steps_per_seed": 12,
                "feedback_top_k": 200,
                "feedback_nprobe": 1024,
                "feedback_add_positive": 1,
                "feedback_add_negative": 0,
                "feedback_max_positive_ids": 8,
                "embedding_lru_size": 2000,
                "embedding_cache_scope": "strategy",
            },
            strategies=[
                base_strategy(
                    "baseline_oldlike_w32",
                    prefetch_method="id_div_rowgroup_threadpool_new_conn",
                    n_workers=32,
                    prefetch_top_k=1000,
                    metadata_query_mode="in_list",
                    prefetch_query_mode="in_list",
                ),
                base_strategy(
                    "optimized_current_w44",
                    prefetch_method="rowgroup_cache_threadpool_reuse_conn",
                    n_workers=44,
                    use_connection_pool=True,
                    prefetch_top_k=200,
                    metadata_query_mode="values_join",
                    prefetch_query_mode="in_list",
                    batch_scheduler="id_ascending",
                    fetch_mode="fetchall",
                ),
            ],
        )
    )

    # Lane S: query-vector expansion sweep (neighbor-assisted query)
    tasks.append(
        make_task(
            "q23_query_expansion_sweep",
            "query_expansion",
            "Evaluate adding nearest neighbors into query vector (latency vs overlap)",
            workload_overrides={
                "n_trials": 10,
                "n_neighbors_values": [1000],
                "prefetch_top_k": 200,
                "run_order": "randomized",
                "run_seed": 20260215,
                "simulate_feedback": True,
                "feedback_steps_per_seed": 8,
                "feedback_top_k": 200,
                "feedback_nprobe": 1024,
                "feedback_add_positive": 1,
                "feedback_max_positive_ids": 8,
            },
            strategies=[
                base_strategy(
                    "expand_none",
                    n_workers=44,
                    prefetch_top_k=200,
                    metadata_query_mode="values_join",
                    batch_scheduler="id_ascending",
                ),
                base_strategy(
                    "expand_top8_w015",
                    n_workers=44,
                    prefetch_top_k=200,
                    metadata_query_mode="values_join",
                    batch_scheduler="id_ascending",
                    query_expand_top_k=8,
                    query_expand_weight=0.15,
                    query_expand_nprobe=512,
                ),
                base_strategy(
                    "expand_top16_w025",
                    n_workers=44,
                    prefetch_top_k=200,
                    metadata_query_mode="values_join",
                    batch_scheduler="id_ascending",
                    query_expand_top_k=16,
                    query_expand_weight=0.25,
                    query_expand_nprobe=512,
                ),
            ],
        )
    )

    # Lane T: dtype comparison (float32 DINO vs quantized uint8 DINO), same workload.
    quantized_source = {
        "faiss_url": (
            "s3://us-west-2.opendata.source.coop/geovibes/search/USA/alabama/"
            "2024-01-01-2025-01-01/httpfs/"
            "alabama_quantized_dino_vit_small_patch16_224_2024_2025_32_16_10/faiss.index"
        ),
        "db_variants": [
            {
                "name": "quantized",
                "db_url": (
                    "s3://us-west-2.opendata.source.coop/geovibes/search/USA/alabama/"
                    "2024-01-01-2025-01-01/httpfs/"
                    "alabama_quantized_dino_vit_small_patch16_224_2024_2025_32_16_10/metadata.db"
                ),
                "geometry_cache_url": (
                    "s3://us-west-2.opendata.source.coop/geovibes/search/USA/alabama/"
                    "2024-01-01-2025-01-01/httpfs/"
                    "alabama_quantized_dino_vit_small_patch16_224_2024_2025_32_16_10/geometry_cache.parquet"
                ),
            }
        ],
        "query_ids": [2660000, 1000000, 4000000],
        "query_id_triplets": [
            [78824, 1033427, 4970394],
            [592921, 215330, 4384211],
            [1686758, 2662800, 5070803],
        ],
        "label_clicks": [
            {"lon": -86.55, "lat": 32.35},
            {"lon": -86.30, "lat": 32.55},
        ],
    }
    common_dtype_workload = {
        "n_trials": 10,
        "n_neighbors_values": [1000],
        "prefetch_top_k": 200,
        "prefetch_fast_k": 100,
        "nprobe": 1024,
        "reference_nprobe": 4096,
        "pool_scope": "run",
        "run_order": "trial_major",
        "run_seed": 20260216,
        "simulate_feedback": True,
        "feedback_steps_per_seed": 8,
        "feedback_top_k": 200,
        "feedback_nprobe": 1024,
        "feedback_add_positive": 1,
        "feedback_max_positive_ids": 8,
        "embedding_lru_size": 2000,
        "embedding_cache_scope": "run",
    }
    tasks.append(
        make_task(
            "q24_dtype_float32_session",
            "dtype",
            "Float32 DINO session benchmark for dtype comparison baseline",
            workload_overrides=common_dtype_workload,
            strategies=[
                base_strategy(
                    "float32_opt_session",
                    n_workers=44,
                    prefetch_top_k=200,
                    metadata_query_mode="values_join",
                    batch_scheduler="id_ascending",
                    fetch_mode="fetchall",
                    query_embedding_expr="embedding",
                    embedding_expr="embedding",
                )
            ],
        )
    )
    tasks.append(
        make_task(
            "q25_dtype_quantized_session",
            "dtype",
            "Quantized uint8 DINO session benchmark for dtype comparison",
            source_overrides=quantized_source,
            workload_overrides=common_dtype_workload,
            strategies=[
                base_strategy(
                    "quantized_opt_session",
                    n_workers=44,
                    prefetch_top_k=200,
                    metadata_query_mode="values_join",
                    batch_scheduler="id_ascending",
                    fetch_mode="fetchall",
                    query_embedding_expr="embedding",
                    embedding_expr="embedding",
                )
            ],
        )
    )

    # Lane U: int16 simulation for query/prefetch expression (speed + overlap).
    int16_sim_expr = (
        "LIST_TRANSFORM(embedding, x -> CAST(CAST(ROUND(x * 1024.0) AS SMALLINT) AS FLOAT) / 1024.0)"
    )
    tasks.append(
        make_task(
            "q26_dtype_int16_sim",
            "dtype",
            "INT16 simulation for query vector + prefetch expression on float32 source",
            workload_overrides={
                "n_trials": 10,
                "n_neighbors_values": [1000],
                "prefetch_top_k": 200,
                "run_order": "randomized",
                "run_seed": 20260217,
            },
            strategies=[
                base_strategy(
                    "float32_native",
                    n_workers=44,
                    prefetch_top_k=200,
                    metadata_query_mode="values_join",
                    batch_scheduler="id_ascending",
                    query_embedding_expr="embedding",
                    embedding_expr="embedding",
                ),
                base_strategy(
                    "int16_sim_query_only",
                    n_workers=44,
                    prefetch_top_k=200,
                    metadata_query_mode="values_join",
                    batch_scheduler="id_ascending",
                    query_embedding_expr=int16_sim_expr,
                    embedding_expr="embedding",
                ),
                base_strategy(
                    "int16_sim_query_prefetch",
                    n_workers=44,
                    prefetch_top_k=200,
                    metadata_query_mode="values_join",
                    batch_scheduler="id_ascending",
                    query_embedding_expr=int16_sim_expr,
                    embedding_expr=int16_sim_expr,
                ),
            ],
        )
    )

    # Lane V: prefetch depth refinement around top-100 default
    tasks.append(
        make_task(
            "q27_prefetch_topk_100_vs_200",
            "prefetch",
            "Iterative session validation: prefetch_top_k=100 vs 200",
            workload_overrides={
                "n_trials": 12,
                "n_neighbors_values": [1000],
                "prefetch_top_k": 200,
                "run_order": "trial_major",
                "run_seed": 20260218,
                "simulate_feedback": True,
                "feedback_steps_per_seed": 6,
                "feedback_top_k": 200,
                "feedback_nprobe": 1024,
                "feedback_add_positive": 1,
                "feedback_add_negative": 0,
                "feedback_max_positive_ids": 8,
                "feedback_max_negative_ids": 4,
                "embedding_lru_size": 2000,
                "embedding_cache_scope": "strategy",
                "pool_scope": "trial",
            },
            strategies=[
                base_strategy(
                    "optimized_top100_w44",
                    n_workers=44,
                    prefetch_top_k=100,
                    metadata_query_mode="values_join",
                    prefetch_query_mode="in_list",
                    batch_scheduler="id_ascending",
                    fetch_mode="fetchall",
                ),
                base_strategy(
                    "optimized_top200_w44",
                    n_workers=44,
                    prefetch_top_k=200,
                    metadata_query_mode="values_join",
                    prefetch_query_mode="in_list",
                    batch_scheduler="id_ascending",
                    fetch_mode="fetchall",
                ),
            ],
        )
    )

    overlap_triplets_prefetch = [
        list(BASE_SOURCE["query_id_triplets"][0]),
        list(BASE_SOURCE["query_id_triplets"][1]),
        list(BASE_SOURCE["query_id_triplets"][0]),
        list(BASE_SOURCE["query_id_triplets"][1]),
        list(BASE_SOURCE["query_id_triplets"][0]),
        list(BASE_SOURCE["query_id_triplets"][1]),
    ]
    tasks.append(
        make_task(
            "q28_prefetch_topk_100_vs_200_overlap",
            "prefetch",
            "Overlapping-query validation: prefetch_top_k=100 vs 200",
            source_overrides={"query_id_triplets": overlap_triplets_prefetch},
            workload_overrides={
                "n_trials": 12,
                "n_neighbors_values": [500],
                "prefetch_top_k": 200,
                "run_order": "trial_major",
                "run_seed": 20260218,
                "simulate_feedback": False,
                "embedding_lru_size": 2000,
                "embedding_cache_scope": "strategy",
                "pool_scope": "trial",
            },
            strategies=[
                base_strategy(
                    "overlap_top100_w44",
                    n_workers=44,
                    prefetch_top_k=100,
                    metadata_query_mode="values_join",
                    prefetch_query_mode="in_list",
                    batch_scheduler="id_ascending",
                    fetch_mode="fetchall",
                ),
                base_strategy(
                    "overlap_top200_w44",
                    n_workers=44,
                    prefetch_top_k=200,
                    metadata_query_mode="values_join",
                    prefetch_query_mode="in_list",
                    batch_scheduler="id_ascending",
                    fetch_mode="fetchall",
                ),
            ],
        )
    )

    tasks.append(
        make_task(
            "q29_prefetch_topk_100_vs_200_warmpool",
            "prefetch",
            "Warm-pool validation: prefetch_top_k=100 vs 200",
            workload_overrides={
                "n_trials": 12,
                "n_neighbors_values": [1000],
                "prefetch_top_k": 200,
                "run_order": "trial_major",
                "run_seed": 20260219,
                "simulate_feedback": True,
                "feedback_steps_per_seed": 6,
                "feedback_top_k": 200,
                "feedback_nprobe": 1024,
                "feedback_add_positive": 1,
                "feedback_add_negative": 0,
                "feedback_max_positive_ids": 8,
                "feedback_max_negative_ids": 4,
                "embedding_lru_size": 2000,
                "embedding_cache_scope": "strategy",
                "pool_scope": "run",
            },
            strategies=[
                base_strategy(
                    "optimized_top100_w44",
                    n_workers=44,
                    prefetch_top_k=100,
                    metadata_query_mode="values_join",
                    prefetch_query_mode="in_list",
                    batch_scheduler="id_ascending",
                    fetch_mode="fetchall",
                ),
                base_strategy(
                    "optimized_top200_w44",
                    n_workers=44,
                    prefetch_top_k=200,
                    metadata_query_mode="values_join",
                    prefetch_query_mode="in_list",
                    batch_scheduler="id_ascending",
                    fetch_mode="fetchall",
                ),
            ],
        )
    )

    return tasks


def check_memory(min_available_gb: float = 2.0) -> tuple[bool, float]:
    meminfo = Path("/proc/meminfo")
    if not meminfo.exists():
        return True, 999.0
    info: dict[str, int] = {}
    for line in meminfo.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        key, val = line.split(":", 1)
        try:
            info[key.strip()] = int(val.strip().split()[0])
        except Exception:
            continue
    avail_kb = info.get("MemAvailable", 0)
    avail_gb = avail_kb / (1024 * 1024)
    return avail_gb >= min_available_gb, avail_gb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run systematic httpfs optimization tasks."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate task configs and print queue actions without executing harness runs.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip tasks with an existing successful report in results/.",
    )
    parser.add_argument(
        "--task-prefix",
        type=str,
        default="",
        help="Optional task-id prefix filter (for partial queue execution).",
    )
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=0,
        help="Optional cap on number of tasks to execute after filtering.",
    )
    parser.add_argument(
        "--min-available-gb",
        type=float,
        default=2.0,
        help="Minimum MemAvailable required to start a task.",
    )
    return parser.parse_args()


def summarize_report(report_path: Path) -> dict[str, Any]:
    if not report_path.exists():
        return {}
    data = json.loads(report_path.read_text(encoding="utf-8"))
    summary = data.get("summary", [])
    if not summary:
        return {}
    best = min(summary, key=lambda x: float(x.get("search_path_total_ms_mean", 1e18)))
    return {
        "rows": len(summary),
        "best_strategy": best.get("strategy"),
        "best_search_path_ms_mean": float(best.get("search_path_total_ms_mean", 0.0)),
        "best_prefetch_ms_mean": float(best.get("prefetch_ms_mean", 0.0)),
    }


def run_task(
    task: dict[str, Any],
    *,
    dry_run: bool = False,
    resume: bool = False,
    min_available_gb: float = 2.0,
) -> dict[str, Any]:
    task_id = task["id"]
    cfg_path = QUEUE_DIR / f"{task_id}.yaml"
    out_path = RESULTS_DIR / f"queue_{task_id}.json"

    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(task["config"], f, sort_keys=False)

    if resume and out_path.exists():
        summary = summarize_report(out_path)
        if summary:
            return {
                "task_id": task_id,
                "lane": task["lane"],
                "description": task["description"],
                "status": "skipped_existing",
                "return_code": 0,
                "elapsed_s": 0.0,
                "config_path": str(cfg_path),
                "report_path": str(out_path),
                "summary": summary,
            }

    ok_mem, avail_gb = check_memory(min_available_gb=min_available_gb)
    if not ok_mem:
        return {
            "task_id": task_id,
            "status": "skipped_low_memory",
            "mem_available_gb": avail_gb,
            "config_path": str(cfg_path),
        }

    cmd = [
        sys.executable,
        str(RUN_HARNESS),
        "--config",
        str(cfg_path),
        "--output",
        str(out_path),
    ]
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    start = time.perf_counter()
    if dry_run:
        rc = 0
    else:
        proc = subprocess.run(cmd, env=env, cwd=str(ROOT.parent.parent))
        rc = int(proc.returncode)
    elapsed_s = time.perf_counter() - start

    result: dict[str, Any] = {
        "task_id": task_id,
        "lane": task["lane"],
        "description": task["description"],
        "status": "ok" if rc == 0 else "error",
        "return_code": rc,
        "elapsed_s": elapsed_s,
        "config_path": str(cfg_path),
        "report_path": str(out_path),
        "mem_available_gb_at_start": avail_gb,
    }
    if rc == 0:
        result["summary"] = summarize_report(out_path)
    return result


def main() -> None:
    args = parse_args()
    dry_run = bool(args.dry_run)
    tasks = task_queue()
    if args.task_prefix:
        tasks = [t for t in tasks if t["id"].startswith(args.task_prefix)]
    if args.max_tasks > 0:
        tasks = tasks[: args.max_tasks]

    queue_run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    queue_result_path = RESULTS_DIR / f"queue_run_{queue_run_id}.json"

    print("=" * 88)
    print("HTTPFS Optimization Queue Runner")
    print("=" * 88)
    print(f"UTC start: {now_iso()}")
    print(f"Tasks: {len(tasks)}")
    print(f"Dry run: {dry_run}")
    print(f"Resume mode: {bool(args.resume)}")
    print(f"Min available memory: {args.min_available_gb:.1f} GB")

    rows: list[dict[str, Any]] = []
    for idx, task in enumerate(tasks, start=1):
        print(
            f"\n[{idx}/{len(tasks)}] {task['id']} [{task['lane']}] - {task['description']}"
        )
        row = run_task(
            task,
            dry_run=dry_run,
            resume=bool(args.resume),
            min_available_gb=float(args.min_available_gb),
        )
        rows.append(row)
        print(
            f"  -> status={row['status']} rc={row.get('return_code')} elapsed={row.get('elapsed_s', 0.0):.1f}s"
        )
        summary = row.get("summary") or {}
        if summary:
            print(
                f"  -> best={summary.get('best_strategy')} "
                f"search_mean={summary.get('best_search_path_ms_mean', 0.0):.1f}ms "
                f"prefetch_mean={summary.get('best_prefetch_ms_mean', 0.0):.1f}ms"
            )

    payload = {
        "metadata": {
            "queue_run_id": queue_run_id,
            "timestamp_utc": now_iso(),
            "task_count": len(tasks),
            "dry_run": dry_run,
        },
        "results": rows,
    }
    queue_result_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved queue report: {queue_result_path}")


if __name__ == "__main__":
    main()
