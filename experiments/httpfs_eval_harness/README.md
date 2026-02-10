# HTTPFS End-to-End Retrieval Harness

This harness profiles GeoVibes' real retrieval path over httpfs:

1. Nearest-point query for labeling clicks
2. FAISS local search
3. Search metadata fetch (`id + geometry`)
4. Embedding prefetch for top-N results

It runs multiple strategies against the same query vector and reports per-stage latency.

## Run

```bash
uv run python experiments/httpfs_eval_harness/run_harness.py
```

Custom config:

```bash
uv run python experiments/httpfs_eval_harness/run_harness.py \
  --config experiments/httpfs_eval_harness/config.yaml
```

Custom output:

```bash
uv run python experiments/httpfs_eval_harness/run_harness.py \
  --output /tmp/httpfs_harness_results.json
```

Run the systematic experiment queue:

```bash
uv run python experiments/httpfs_eval_harness/run_experiment_queue.py
```

Resume from completed per-task reports (skip existing successful outputs):

```bash
uv run python experiments/httpfs_eval_harness/run_experiment_queue.py --resume
```

Run only a subset of tasks:

```bash
uv run python experiments/httpfs_eval_harness/run_experiment_queue.py \
  --resume \
  --task-prefix q14 \
  --max-tasks 1
```

Build remote embeddings-only variants (id-sorted + ivf-list-sorted parquet and row-group sidecars):

```bash
uv run python experiments/httpfs_eval_harness/build_embedding_variants.py \
  --tag alabama_ivf_layout_v1 \
  --row-group-size 122880 \
  --emit-manifest
```

Materialize a remote DuckDB artifact from a parquet variant:

```bash
uv run python experiments/httpfs_eval_harness/materialize_duckdb_variant.py \
  --parquet-url s3://.../embeddings_ivflist_sorted_rg122880.parquet \
  --local-db-path /tmp/httpfs_codex/embeddings_ivflist_sorted_rg122880.db \
  --embedding-dim 384 \
  --upload-s3-url s3://.../embeddings_ivflist_sorted_rg122880.db
```

## Output

JSON report in `experiments/httpfs_eval_harness/results/` with:

- `results`: raw per-trial measurements
- `summary`: grouped means/p50/p95 by dataset, strategy, neighbors

Queue runner output:
- task-level reports in `experiments/httpfs_eval_harness/results/queue_<task_id>.json`
- queue execution report in `experiments/httpfs_eval_harness/results/queue_run_<timestamp>.json`

Optional (when `workload.capture_faiss_ids: true`):
- `faiss_ids` in each trial row for overlap/recall analyses across `nprobe` settings
- `query_ids_used` for per-trial query-vector traceability in multi-query runs

Key metrics:

- `label_total_ms`
- `query_expand_ms`
- `faiss_ms`
- `metadata_ms`
- `prefetch_ms`
- `prefetch_fast_ms` (two-stage strategies)
- `prefetch_tail_ms` (two-stage strategies)
- `search_path_total_ms = query_expand + faiss + metadata + prefetch`
- `faiss_overlap_ratio` (vs reference lane/query vector)

## Strategy focus

The harness is designed to isolate improvements relevant to your app:

- Remote metadata vs local geometry cache
- Current prefetch grouping (`id // row_group_size`)
- Physical grouping using `rowid // row_group_size`
- Cached physical grouping via local `id -> row_group` sidecar
- Per-batch new connections vs thread-local connection reuse
- Persistent connection pooling across trials/searches
- Two-stage prefetch (`fast_k` then tail) for readiness-vs-throughput tradeoffs
- DuckDB native threading alternative

## Realism Controls

To reduce cache-inflated results and better represent production patterns:

- `source.query_id_triplets`: rotate through multiple query vectors across trials
- `workload.pool_scope`:
  - `run` (default): keep pooled connections across trials (steady-state)
  - `trial`: reset pooled connections each trial (cache-conservative)
- `workload.run_order`:
  - `strategy_major` (default): run all trials for each strategy in blocks
  - `trial_major`: interleave strategies inside each trial index
  - `randomized`: shuffle the full run plan (optionally with `workload.run_seed`)
- `workload.run_label_queries`:
  - `true` (default): run nearest-point label stage
  - `false`: skip label stage (prefetch-focused experiments)
- `workload.reference_nprobe`:
  - FAISS overlap reference setting for `faiss_overlap_ratio`
- `workload.query_embedding_expr`:
  - SQL expression used to fetch embeddings for query-vector construction (default `embedding`)
- `workload.embedding_cache_scope`:
  - `run`, `variant`, `strategy`, `trial`
  - use `strategy` to avoid cross-strategy cache contamination in interleaved runs
- `workload.simulate_feedback`:
  - `true`: generate iterative query sets by repeatedly adding top FAISS hits as positives
  - controls: `feedback_steps_per_seed`, `feedback_top_k`, `feedback_nprobe`,
    `feedback_add_positive`, `feedback_add_negative`, `feedback_max_positive_ids`,
    `feedback_max_negative_ids`
- Remote connections are configured with `enable_object_cache=false`

Strategy-level prefetch controls:
- `fetch_mode`:
  - `fetchdf` (default)
  - `fetchall` (closest to current app runtime map path)
  - `arrow`
- `query_embedding_expr`: strategy-level override for query-vector embedding expression
- `query_expand_top_k`, `query_expand_weight`, `query_expand_nprobe`:
  optional query-expansion lane knobs (measured via `query_expand_ms`)

Variant-level source controls:
- `prefetch_db_url` (optional): use a different source for embedding prefetch than `db_url`
- `row_group_cache_url` (optional): precomputed row-group sidecar for `prefetch_db_url`

## Notes

- Geometry cache download is handled through `FaissCache`.
- The harness can build a one-time local row-group cache under
  `~/.cache/geovibes/row_groups/` for fast repeated evaluations.
