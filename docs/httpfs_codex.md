# HTTPFS Codex Experiment Log

## 2026-02-08 - Experiment 0 - Harness bootstrap

### Goal
Create an end-to-end harness that measures the real GeoVibes retrieval workflow over httpfs, then use it to compare optimization strategies and guide code changes.

### Added
- `experiments/httpfs_eval_harness/run_harness.py`
- `experiments/httpfs_eval_harness/config.yaml`
- `experiments/httpfs_eval_harness/README.md`

### Harness scope
The harness profiles these stages:
1. `nearest_point` labeling query latency
2. FAISS search latency
3. metadata fetch latency (`id + geometry`)
4. embedding prefetch latency

### Initial strategy matrix
- `baseline_remote_single`
- `app_current_geomcache_iddiv_newconn`
- `opt_geomcache_iddiv_reuse`
- `opt_geomcache_rowid_reuse`
- `opt_geomcache_duckdb_threads`

### Hypotheses
1. Local geometry cache will significantly reduce metadata fetch cost.
2. Reusing remote DuckDB connections in prefetch workers will beat per-batch connection setup.
3. Physical row-group batching by `rowid` will outperform `id // row_group_size` batching.
4. A one-shot DuckDB native threaded fetch may be competitive, but likely slower than row-group-aware parallel fetch for scattered IDs.

### Next
Run baseline harness, capture results, and choose first optimization target from measured bottlenecks.

## 2026-02-08 - Experiment 1 - Prefetch strategy sweep (worker count + grouping mode)

### Goal
Find the fastest embedding prefetch strategy over httpfs for the current Alabama dataset baseline.

### Config
- `experiments/httpfs_eval_harness/config.sweep_prefetch.yaml`
- `n_trials=2`, `n_neighbors=500`, `prefetch_top_k=200`, `nprobe=64`

### Results
- Report: `experiments/httpfs_eval_harness/results/harness_results_20260208_201828.json`
- Top performers (search-path mean):
  - `opt_geomcache_rowgroup_cache_reuse_w32`: `3786.7ms`
  - `opt_geomcache_iddiv_reuse_w16`: `3788.4ms`
  - `opt_geomcache_iddiv_reuse_w32`: `3817.8ms`
- Current-like path:
  - `app_current_geomcache_iddiv_newconn_w32`: `5033.0ms`
  - `app_current_geomcache_iddiv_newconn_w16`: `7115.8ms`
- Slow paths:
  - `opt_geomcache_duckdb_threads_32`: `22891.5ms`
  - `rowgroup_cache + new_conn` variants were consistently slower than `reuse_conn`.

### Conclusion
The largest gain came from avoiding per-batch connection setup. Reused-thread connections were consistently faster than creating a fresh remote DuckDB connection per batch.

## 2026-02-08 - Experiment 2 - UI-like workload confirmation

### Goal
Validate winners under realistic app parameters:
- FAISS `nprobe=4096`
- neighbors `500` and `1000`
- prefetch up to `1000` IDs

### Config
- `experiments/httpfs_eval_harness/config.confirm_ui_like.yaml`
- `n_trials=3`

### Results
- Report: `experiments/httpfs_eval_harness/results/harness_results_20260208_202328.json`
- `n_neighbors=500`:
  - `app_current_geomcache_iddiv_newconn_w32`: `6210.6ms`
  - `opt_geomcache_rowgroup_cache_reuse_w32`: `4608.3ms`
- `n_neighbors=1000`:
  - `app_current_geomcache_iddiv_newconn_w32`: `7066.2ms`
  - `opt_geomcache_rowgroup_cache_reuse_w32`: `5757.4ms` (best, stable p95 `5855.3ms`)

### Conclusion
For heavier retrieval (`1000` neighbors), row-group-cache batching with connection reuse clearly outperformed the current app strategy.

## 2026-02-08 - Experiment 3 - Harness robustness fix

### Goal
Fix method dispatch/setup edge case for row-group-cache strategies.

### Change
- Updated `run_harness.py` cache setup detection:
  - before: only triggered for `rowgroup_cache_threadpool_reuse_conn`
  - after: triggers for all `rowgroup_cache_threadpool_*` methods

### Validation
- Quick run report: `experiments/httpfs_eval_harness/results/harness_results_20260208_202813.json`
- Sanity result (`n=500`, `prefetch_top_k=200`):
  - `app_current_geomcache_iddiv_newconn`: `8785.1ms`
  - `opt_geomcache_rowgroup_cache_reuse`: `4117.8ms`

## 2026-02-08 - Experiment 4 - Final A/B and stability runs

### Goal
A/B compare current runtime prefetch strategy against candidate strategy and verify stability.

### A/B config
- `experiments/httpfs_eval_harness/config.final_compare_5.yaml`
- Strategies:
  - `current_geomcache_iddiv_newconn_w32`
  - `candidate_geomcache_rowgroup_cache_reuse_w32`
- `n_trials=5`, neighbors `500` and `1000`, `nprobe=4096`

### Results
- Report: `experiments/httpfs_eval_harness/results/harness_results_20260208_203523.json`
- `n=500`: current `5936.5ms`, candidate `5811.4ms` (candidate `2.1%` faster mean)
- `n=1000`: current `7016.8ms`, candidate `5850.7ms` (candidate `16.6%` faster mean)

### Additional stability run (`n=500`, 10 trials)
- Config: `experiments/httpfs_eval_harness/config.stability_500.yaml`
- Report: `experiments/httpfs_eval_harness/results/harness_results_20260208_203945.json`
- current: mean `6273.7ms`, p50 `5978.6ms`, p95 `7714.0ms`
- candidate: mean `5110.3ms`, p50 `4614.0ms`, p95 `7200.9ms`

### Conclusion
Candidate remains better under repeated runs, including p95 in the 10-trial stability check.

## 2026-02-08 - Experiment 5 - Runtime integration

### Goal
Apply measured winner to the actual UI prefetch path.

### Implemented code changes
- `geovibes/ui/data_manager.py`
  - Added local row-group sidecar cache (`id -> row_group`) creation/loading for remote DBs.
  - Added `group_ids_for_prefetch(...)` with:
    - primary mode: `row_group_cache`
    - fallback mode: `id_div`
  - Added auxiliary-connection lifecycle management to avoid leaking cache connections.
  - Added row-group cache loading during remote DB connect/switch.
- `geovibes/ui/app.py`
  - `_prefetch_embeddings_async` now:
    - groups IDs using `DataManager.group_ids_for_prefetch(...)`
    - reuses one background DuckDB connection per worker thread
    - keeps progress updates and adds robust cleanup/error logging.

### Test status
- `SKIP_NETWORK_TESTS=1 uv run pytest -q tests/test_remote_db.py tests/test_data_manager_remote.py tests/test_dropdown_grouping.py tests/test_data_manager_boundary.py`
- Result: `20 passed, 17 skipped`

### Added tests
- `tests/test_data_manager_remote.py`
  - Added prefetch grouping tests for both fallback (`id_div`) and cached row-group modes.

## 2026-02-08 - Experiment 6 - Optimization 1+2 benchmark (pool + two-stage)

### Goal
Evaluate:
1. persistent remote DuckDB connection pool across searches
2. two-stage prefetch (`fast_k`) for quicker "ready-to-search"

### Harness changes
- Added connection-pool support and two-stage prefetch methods in:
  - `experiments/httpfs_eval_harness/run_harness.py`
- Added fast/tail timing metrics:
  - `prefetch_fast_ms`, `prefetch_tail_ms`

### Config + report
- Config: `experiments/httpfs_eval_harness/config.opt12_fast100.yaml`
- Report: `experiments/httpfs_eval_harness/results/harness_results_20260208_234424.json`

### Results
- `n=500` search path mean:
  - current (`id_div + new_conn`): `7086.1ms`
  - rowgroup reuse no pool: `4599.5ms`
  - rowgroup reuse + pool: `2992.1ms` (best total)
  - rowgroup two-stage + pool fast100: `3616.6ms`
- `n=1000` search path mean:
  - current (`id_div + new_conn`): `7152.5ms`
  - rowgroup reuse no pool: `6244.6ms`
  - rowgroup reuse + pool: `3584.6ms` (best total)
  - rowgroup two-stage + pool fast100: `3814.7ms`

### Interpretation
- Pooling is the biggest throughput win.
- Two-stage is slightly slower in total prefetch time but gives much faster early readiness:
  - `n=1000`: `prefetch_fast_ms_p50 = 987.0ms` (top-100 ready quickly), while full pooled prefetch p50 is `3451.1ms`.

## 2026-02-08 - Experiment 7 - `fast_k` and worker tuning

### `fast_k` sweep
- Config: `experiments/httpfs_eval_harness/config.fastk_sweep.yaml`
- Report: `experiments/httpfs_eval_harness/results/harness_results_20260208_233711.json`
- For two-stage (`n=1000`), fast-stage p50:
  - `fast_k=100`: `1255.7ms`
  - `fast_k=200`: `1535.8ms`
  - `fast_k=300`: `1736.7ms`
- Selected default: `fast_k=100` for lower "ready" latency.

### Worker sweep
- Config: `experiments/httpfs_eval_harness/config.pool_workers.yaml`
- Report: `experiments/httpfs_eval_harness/results/harness_results_20260208_234002.json`
- Best worker count remained `32` (both one-stage pooled and two-stage pooled).

## 2026-02-08 - Experiment 8 - Runtime integration of optimizations 1+2

### Implemented
- `geovibes/ui/data_manager.py`
  - Added persistent background connection pool:
    - `configure_background_connection_pool(...)`
    - `acquire_background_connection(...)`
    - `release_background_connection(...)`
    - cleanup on close/switch
- `geovibes/ui/app.py`
  - `_prefetch_embeddings_async` now:
    - uses persistent pool for thread workers
    - executes two-stage prefetch (`fast_k=100`, then tail)
    - marks UI as ready after fast stage (tail continues in background)
    - cancels stale tail work when a newer prefetch generation exists

### Regression tests
- `SKIP_NETWORK_TESTS=1 uv run pytest -q tests/test_remote_db.py tests/test_data_manager_remote.py tests/test_dropdown_grouping.py tests/test_data_manager_boundary.py`
- Result: `22 passed, 17 skipped`

### Additional tests
- `tests/test_data_manager_remote.py`
  - Added background connection pool reuse/disable tests.

## Harness coverage for optimization ideas 3-6

### 3) Build-time row-group sidecar
- Harness can measure steady-state impact directly (already doing this).
- It does **not** yet isolate first-build overhead of sidecar creation as a separate metric; that can be added.

### 4) Storage reorder by FAISS locality
- Fully supported via `source.db_variants`; harness can compare reordered DB variants head-to-head.

### 5) Split embedding store from geometry store
- Supported if exposed as variant DB layouts/artifacts; harness already separates metadata and embedding prefetch stages.

### 6) Adaptive FAISS `nprobe`
- Fully supported now through workload config (`nprobe`); can sweep per scenario.

## 2026-02-08 - Experiment 9 - Adaptive `nprobe` sweep (optimization 6)

### Goal
Measure FAISS latency/quality tradeoff for `nprobe` while using the best prefetch throughput path (`rowgroup_reuse_pool_w32`).

### Method
- Added optional FAISS ID capture to harness:
  - workload: `capture_faiss_ids: true`
  - row output includes `faiss_ids`
- Ran nprobe sweep:
  - `128, 256, 512, 1024, 2048, 4096`
- Generated per-nprobe reports:
  - `experiments/httpfs_eval_harness/results/harness_results_20260208_nprobe_*.json`
- Summary:
  - `experiments/httpfs_eval_harness/results/harness_results_20260208_nprobe_summary.json`

### Results
- Search-path mean (`n=1000`, 2 trials):
  - `nprobe=128`: `4472.2ms`
  - `nprobe=256`: `4370.0ms`
  - `nprobe=512`: `4322.1ms`
  - `nprobe=1024`: `4322.8ms`
  - `nprobe=2048`: `4344.3ms`
  - `nprobe=4096`: `4612.7ms`
- FAISS mean:
  - `nprobe=512`: `23.2ms`
  - `nprobe=1024`: `47.7ms`
  - `nprobe=4096`: `204.3ms`
- Overlap vs `nprobe=4096` (`@1000` IDs, first-trial set):
  - `nprobe=128`: recall `0.975`
  - `nprobe=256`: recall `0.995`
  - `nprobe>=512`: recall `1.000`, jaccard `1.000`

### Runtime integration
- `geovibes/ui/app.py`
  - Replaced fixed `nprobe=4096` with configurable adaptive values:
    - `GEOVIBES_FAISS_NPROBE` (default `1024`)
    - `GEOVIBES_FAISS_NPROBE_HIGH` (default `2048`)
    - `GEOVIBES_FAISS_NPROBE_HIGH_THRESHOLD` (default `5000`)
- For current workloads (`n_neighbors <= 1000`), default now uses `nprobe=1024`.

## 2026-02-08 - Experiment 10 - Cache realism and session-mode benchmarking

### Goal
Avoid inflated benchmarks from repeated pooled-session warm-up and validate optimization under:
1. strict cold-session-like conditions
2. warm in-session usage

### Harness updates
- Added multi-query workload support:
  - `source.query_id_triplets`
- Added pool lifecycle mode:
  - `workload.pool_scope: run | trial`
  - `trial` closes connection pools before each trial to reduce cross-trial warm inflation.
- Added `query_ids_used` to trial output for traceability.

### Strict session-mode run (cache-conservative)
- Config: `experiments/httpfs_eval_harness/config.realworld_strict.yaml`
- Report: `experiments/httpfs_eval_harness/results/harness_results_20260208_235908.json`
- Setup:
  - 6 distinct query triplets
  - `pool_scope=trial`
  - `nprobe=1024`, neighbors `1000`
- Results (search-path mean):
  - current (`id_div + new_conn`): `8473.5ms`
  - optimized (`rowgroup + reuse + pool`): `7840.6ms` (`7.5%` faster)
  - two-stage pooled (`fast100`): `9764.3ms` (slower in total throughput)

### Warm in-session run (real user session)
- Config: `experiments/httpfs_eval_harness/config.realworld_warm.yaml`
- Report: `experiments/httpfs_eval_harness/results/harness_results_20260209_000306.json`
- Setup:
  - same 6 triplets
  - `pool_scope=run`
- Results (search-path mean):
  - current: `7657.8ms`
  - optimized (`rowgroup + reuse + pool`): `5841.2ms` (`23.7%` faster)
  - two-stage pooled (`fast100`): `7124.0ms` (`7.0%` faster vs current, but slower than one-stage pooled)

### Interpretation
- The optimization still helps under cache-conservative conditions (not a warm-only artifact).
- Biggest gains appear in warm in-session usage (the dominant real-world notebook interaction mode).
- Two-stage improves readiness but can reduce total prefetch throughput.

## 2026-02-08 - Experiment 11 - Runtime strategy split (throughput vs readiness)

### Problem
Using two-stage prefetch for every call hurt total throughput for post-search prefetch.

### Change
- Updated `GeoVibes` prefetch call sites:
  - label-time prefetch: `two_stage=True` (fast readiness)
  - post-search prefetch: `two_stage=False` (best throughput)
- This keeps low-latency query-vector readiness when user just labeled points, while maximizing background prefetch throughput after search.

### Related code
- `geovibes/ui/app.py`:
  - `_prefetch_embeddings_async(..., two_stage=True)` added
  - label path calls `two_stage=True`
  - search-result path calls `two_stage=False`
- `geovibes/ui/data_manager.py`:
  - background connections now apply runtime settings including `SET enable_object_cache=false`
    to align with main connection behavior.

## 2026-02-09 - Experiment 12 - Fetch materialization realism + randomized ordering

### Goal
Remove benchmark-order bias and verify which prefetch materialization path best matches runtime:
- `fetchdf`
- `fetchall` (current runtime map path)
- `arrow`

### Harness changes
- Added strategy knob: `fetch_mode` (`fetchdf | fetchall | arrow`).
- Added workload knobs:
  - `run_order: strategy_major | trial_major | randomized`
  - `run_seed`

### Configs + reports
- strict: `experiments/httpfs_eval_harness/config.fetchmode_strict.yaml`
  - report: `experiments/httpfs_eval_harness/results/harness_results_20260209_001548.json`
- warm: `experiments/httpfs_eval_harness/config.fetchmode_warm.yaml`
  - report: `experiments/httpfs_eval_harness/results/harness_results_20260209_001928.json`
- strict randomized:
  - config: `experiments/httpfs_eval_harness/config.fetchmode_strict_randomized.yaml`
  - report: `experiments/httpfs_eval_harness/results/harness_results_20260209_002438.json`

### Key result (strict randomized, most defensible)
- `optimized_rowgroup_pool_fetchall_w32`: `8032.7ms`
- `optimized_rowgroup_pool_arrow_w32`: `8179.7ms`
- `optimized_rowgroup_pool_fetchdf_w32`: `8190.6ms`

### Conclusion
- Differences are modest, but `fetchall` is best in the randomized strict run and matches runtime behavior, so keep using map/fetchall style for app-facing prefetch.

## 2026-02-09 - Experiment 13 - FAISS-locality row-group projection before rebuild

### Goal
Estimate upside of reordering by FAISS IVF list locality before spending full rebuild time.

### Method
- Built `id -> projected_row_group` from FAISS inverted-list order (no table rebuild required).
- Replayed captured FAISS ID sets from nprobe sweep reports.
- Compared current row-group touches vs projected IVF-list layout touches.

### Report
- `experiments/httpfs_eval_harness/results/harness_results_20260209_ivf_layout_rowgroup_projection.json`

### Result
- projected row-group reduction was limited in this workload:
  - ~`9%` for `nprobe>=256`
  - ~`13.6%` for `nprobe=128`

### Conclusion
- IVF-locality reorder is not a silver bullet for this dataset/query pattern; expected gain is moderate, not 5-10x.

## 2026-02-09 - Experiment 14 - Remote variant artifact build (reordered + split-store candidates)

### Goal
Generate real remote artifacts for optimization ideas:
1. split embedding store from metadata DB
2. IVF-locality-ordered embedding storage

### Added scripts
- `experiments/httpfs_eval_harness/build_embedding_variants.py`
  - builds remote parquet variants + row-group sidecars
- `experiments/httpfs_eval_harness/materialize_duckdb_variant.py`
  - materializes remote DuckDB artifact from parquet

### Built artifacts
- Prefix:
  - `s3://us-west-2.opendata.source.coop/geovibes/experiments/httpfs_codex/embedding_variants/alabama_ivf_layout_v1/`
- Parquet:
  - `embeddings_id_sorted_rg122880.parquet` (`7.57GB`)
  - `embeddings_ivflist_sorted_rg122880.parquet` (`7.58GB`)
- DuckDB:
  - `embeddings_id_sorted_rg122880.db` (`10.81GB`)
  - `embeddings_ivflist_sorted_rg122880.db` (`10.86GB`)
- Sidecars:
  - `row_groups_id_sorted_rg122880.parquet`
  - `row_groups_ivflist_sorted_rg122880.parquet`

### Build cost
- Full variant build elapsed: ~`6500s` (~`108min`) for the parquet + sidecar generation.

## 2026-02-09 - Experiment 15 - Split-store/reordered artifact evaluation (recall + fetch speed)

### Goal
Validate whether split embedding artifacts (parquet/DuckDB) improve prefetch over httpfs without hurting FAISS recall.

### Recall
- FAISS index/search path unchanged in these tests, so recall is unchanged for the same `nprobe` setting.
- (We still log captured FAISS IDs where applicable for auditability.)

### Measurements
- Report: `experiments/httpfs_eval_harness/results/split_store_microbench_20260209.json`
- Direct prefetch-threadpool (`workers=8`, top-1000 IDs):
  - baseline metadata DB: `22961.1ms`
  - id-sorted parquet: `52549.3ms` (worse)
  - ivf-list parquet: `105240.3ms` (much worse)
- Single-query `id IN (1000)` probe:
  - baseline metadata DB: `146633.0ms`
  - id-sorted split DuckDB: `184288.9ms` (worse)
  - ivf-list split DuckDB: `170654.4ms` (worse)

### Interpretation
- Parquet-backed split store is a clear mismatch for this sparse `id IN (...)` retrieval pattern over httpfs.
- DuckDB split artifacts also regressed in this direct probe.
- IVF-list ordering reduced projected row-group touches somewhat, but not enough to offset observed penalties in this implementation path.

### Decision
- Do **not** adopt split-store/reordered-remote-artifact path in current form.
- Keep optimizing on the proven path:
  - geometry cache + row-group sidecar + pooled background DuckDB connections
  - adaptive `nprobe`
  - stage-specific prefetch behavior (label-time readiness vs search-time throughput).

## 2026-02-09 - OPus branch comparison (`~/httpfs-opus/docs/httpfs_opus.md`)

### What OPus got right
- Correctly identified connection setup/attach overhead as a major contributor.
- Correctly showed that pooled/reused connections can materially improve throughput.
- Correctly emphasized row-group scatter as the main structural bottleneck.

### Gaps / mistakes we should account for
- Several reported “best” numbers are prefetch-only and not full end-to-end search path.
- “Prewarmed” style measurements are not apples-to-apples with first real search unless explicitly separated as startup amortization.
- Their code path does not consistently enforce `enable_object_cache=false`, so cache behavior can be ambiguous.
- Limited query diversity in some sweeps increases risk of overfitting to one query region.

### Net takeaway
- OPus findings are directionally useful and aligned on connection reuse.
- Our harness adds stricter realism controls (query rotation, pool-scope modes, randomized strategy order, captured FAISS IDs, stage-level metrics), and these controls are necessary for reliable decisions.

## 2026-02-10 - Experiment 16 - Systematic queue runner (q01-q12)

### Goal
Run a systematic optimization queue over the real httpfs retrieval path, then checkpoint/resume safely without rerunning finished tasks.

### Harness/runner updates
- `experiments/httpfs_eval_harness/run_experiment_queue.py`
  - Added CLI flags:
    - `--resume`
    - `--task-prefix`
    - `--max-tasks`
    - `--min-available-gb`
  - Added skip-on-existing-report behavior for resumable queue execution.
- Queue reports:
  - `experiments/httpfs_eval_harness/results/queue_run_20260210_010614.json`

### Key q01-q12 outcomes
- Query form:
  - `metadata_values_join` beat `in_list_default` in the first-pass sweep (`9624.9ms` vs `14515.4ms`), but single-trial noise required follow-up confirmation.
- Fallback `id_div` grouping proxy:
  - `row_group_size=65536` best in this sweep (`9232.1ms`) vs `122880` (`9647.6ms`) and `245760` (`13077.6ms`).
- `nprobe` sweep (single-trial queue pass):
  - Best throughput cluster: `128/256/1024/2048` around `9.3-9.5s`.
  - `4096` slower (`9897.5ms`) with much higher FAISS cost.
  - Recall vs `4096` baseline IDs (same query):
    - `128`: `0.894`
    - `256`: `0.984`
    - `512`: `0.996`
    - `1024/2048`: `1.000`
- Prefetch depth:
  - `top200` beat deeper prefetch in one-pass queue (`4326.4ms` best).
- Batch scheduling:
  - `id_ascending` best (`7301.2ms`), then `largest_first`, then `as_is`.
- Pool sizing:
  - strict (`pool_scope=trial`): `w16` best.
  - warm (`pool_scope=run`): `w32` best.
- Runtime object cache:
  - one-pass showed `enable_object_cache=true` slightly faster (`7871.9ms` vs `8291.1ms`), but we keep `false` for conservative benchmarking.

## 2026-02-10 - Experiment 17 - Instrumentation integrity patch

### Goal
Ensure queue metrics are auditable for prefetch waste/cache analysis.

### Changes
- `experiments/httpfs_eval_harness/run_harness.py`
  - Added `user_consumed_k` to each trial row.
  - Added `prefetch_candidate_ids_mean` to summary output.

### Follow-up
- Re-ran cache lane reports to refresh summaries:
  - `queue_q08_cache_lru_0.json`
  - `queue_q08_cache_lru_500.json`
  - `queue_q08_cache_lru_2000.json`
  - `queue_q08_cache_lru_5000.json`

## 2026-02-10 - Experiment 18 - Overlap-aware cache realism (q13)

### Problem
Original LRU lane used disjoint query triplets, producing `0` cache hits and mostly measuring noise.

### Method
- Added overlapping repeated query sequence lane:
  - `q13_cache_overlap_lru_0`
  - `q13_cache_overlap_lru_500`
  - `q13_cache_overlap_lru_2000`
- Reports:
  - `experiments/httpfs_eval_harness/results/queue_q13_cache_overlap_lru_0.json`
  - `experiments/httpfs_eval_harness/results/queue_q13_cache_overlap_lru_500.json`
  - `experiments/httpfs_eval_harness/results/queue_q13_cache_overlap_lru_2000.json`

### Results
- `lru_0`: `8107.2ms` mean, `7696.9ms` p50, cache hits `0`.
- `lru_500`: `8698.2ms` mean, `7478.6ms` p50, cache hits `0` (still too small for alternating 2x500-ID reuse).
- `lru_2000`: `2618.0ms` mean, `103.9ms` p50.
  - `prefetch_cache_hits_mean=333.3`
  - `prefetch_effective_ids_mean=166.7`
  - trials 3-6 had `prefetch_ms=0.0` from full ID reuse.

### Conclusion
- Cross-search embedding reuse is real and very large when working set fits cache.
- Small caches can fail to capture reuse and can look worse due variance.
- If we add application-level embedding LRU, size needs to cover at least two recent result sets for local back-and-forth workflows.

## 2026-02-10 - Experiment 19 - Randomized confirmation sweeps (q14-q16)

### Reports
- `experiments/httpfs_eval_harness/results/queue_q14_prefetch_depth_randomized.json`
- `experiments/httpfs_eval_harness/results/queue_q15_query_mode_randomized.json`
- `experiments/httpfs_eval_harness/results/queue_q16_query_mode_metadata_only.json`
- Queue run:
  - `experiments/httpfs_eval_harness/results/queue_run_20260210_011854.json`
  - `experiments/httpfs_eval_harness/results/queue_run_20260210_012907.json`

### Prefetch depth (q14, randomized, 3 trials)
- `top100`: `4193.6ms` mean (`4092.2ms` prefetch), `wasted_ratio=0.0`
- `top200`: `5098.1ms` mean, `wasted_ratio=0.5`
- `top500`: `6895.5ms` mean, `wasted_ratio=0.8`
- `top1000`: `9659.2ms` mean, `wasted_ratio=0.9`

### Query form stability
- q15 (`in_list_default` vs `metadata_values_join` vs `both_values_join`) showed severe tail instability when `values_join` was used for prefetch:
  - `both_values_join` had ~`105s` prefetch outliers.
- q16 isolated metadata-only rewrite (prefetch kept as `in_list`), 5 randomized trials:
  - `metadata_values_join`: `9307.3ms` mean
  - `in_list_default`: `9389.4ms` mean
  - effect is small but consistent in favor of metadata rewrite, without pathological tails.

### Decision
- Keep metadata query rewrite (`metadata_query_mode=values_join`).
- Do **not** use `values_join` for embedding prefetch queries.

## 2026-02-10 - Experiment 20 - Worker-count extension with OPus parity (q17-q18)

### Goal
Directly validate OPus claim that `44` workers is best by extending our pool-size lane to include `w44`.

### Reports
- `experiments/httpfs_eval_harness/results/queue_q17_pool_size_strict_w44.json`
- `experiments/httpfs_eval_harness/results/queue_q18_pool_size_warm_w44.json`
- `experiments/httpfs_eval_harness/results/queue_q19_pool_size_strict_w44_randomized.json`
- `experiments/httpfs_eval_harness/results/queue_q20_pool_size_warm_w44_randomized.json`
- Queue run:
  - `experiments/httpfs_eval_harness/results/queue_run_20260210_013307.json`
  - `experiments/httpfs_eval_harness/results/queue_run_20260210_013509.json`

### Results
- strict (`pool_scope=trial`, single-trial extension):
  - `w16`: `8720.7ms`
  - `w32`: `6842.2ms`
  - `w44`: `5760.1ms` (best)
- warm (`pool_scope=run`, single-trial extension):
  - `w16`: `9127.8ms`
  - `w32`: `6632.7ms`
  - `w44`: `5825.4ms` (best)
- strict randomized confirmation (`n_trials=3`):
  - `w16`: mean `9158.5ms`, p50 `9172.9ms`
  - `w32`: mean `8546.6ms`, p50 `7958.1ms`
  - `w44`: mean `8214.1ms`, p50 `7615.2ms` (best mean/p50)
- warm randomized confirmation (`n_trials=3`):
  - `w16`: mean `7721.5ms`, p50 `7924.4ms`
  - `w32`: mean `8758.9ms`, p50 `7831.1ms` (worse mean from a large tail)
  - `w44`: mean `7178.6ms`, p50 `7073.6ms` (best mean/p50)

### Interpretation
- This aligns with OPus directionally: higher concurrency can reduce wall-clock prefetch for this dataset.
- In both strict and warm randomized confirmations, `w44` is the strongest overall choice on this VM/dataset.
- Keep memory/CPU guardrails in production:
  - use `w44` when host resources are healthy
  - fall back to `w32` (or `w16`) on constrained hosts.

## 2026-02-10 - Updated implementation conclusions

### Implement next
1. Metadata query rewrite only:
   - use `VALUES` join for metadata stage
   - keep prefetch stage on stable `IN (...)` query form
2. Adaptive prefetch depth:
   - default prefetch target near real consumed count (`~100-200`), not `1000`
   - expose deeper prefetch as optional background behavior
3. Scheduler + workers:
   - keep `id_ascending` batch scheduling
   - target `w44` where host resources permit; keep `w32`/`w16` fallback tiers
4. Add app-level embedding LRU cache:
   - size to cover at least two recent result sets (e.g. >=`2000` IDs)
   - skip prefetch for cached IDs
5. Keep adaptive `nprobe`:
   - `1024` remains a strong default (full-recall in tested comparisons with much lower FAISS time than `4096`)

### Do not implement (current evidence)
- Split parquet/DuckDB embedding store variants over httpfs (regressed).
- IVF-list physical reorder as a primary fix (limited projected gain and poor direct fetch outcomes).
- `values_join` rewrite for prefetch stage (tail-risk behavior in randomized runs).

### OPus cross-check after queue expansion
- Still aligned with OPus on connection reuse importance.
- Additional items we measured that OPus did not fully cover:
  - metadata-only query rewrite
  - scheduler policy effects
  - prefetch waste ratio vs user-consumed IDs
  - cache-size-dependent cross-search reuse thresholds
  - randomized run-order and tail-risk visibility for query forms
- Main OPus caveat remains: several “best” numbers are prefetch-path-centric and not full end-to-end, and prewarmed/cache state handling can overstate generality.

## 2026-02-10 - Experiment 21 - Runtime implementation pass + OPus update review

### Goal
Implement queue-backed optimizations in runtime code and re-review OPus updates for anything still missing.

### Implemented in app/runtime code
- `geovibes/ui/data_manager.py`
  - Added query-mode controls:
    - `GEOVIBES_METADATA_QUERY_MODE` (default `values_join`)
    - `GEOVIBES_PREFETCH_QUERY_MODE` (default `in_list`)
  - Added batch scheduling control:
    - `GEOVIBES_PREFETCH_BATCH_SCHEDULER` (default `id_ascending`)
  - `group_ids_for_prefetch(...)` now emits deterministic id-sorted batches and applies scheduler ordering.
  - `query_search_metadata(...)` now supports metadata `VALUES`-join mode without changing prefetch query mode.
  - `fetch_embedding_map_with_connection(...)` now accepts query mode and defaults to prefetch mode.
- `geovibes/ui/app.py`
  - Added adaptive prefetch budget controls:
    - `GEOVIBES_PREFETCH_TOPK_MIN` (default `100`)
    - `GEOVIBES_PREFETCH_TOPK_MAX` (default `200`)
    - `GEOVIBES_PREFETCH_TOPK_RATIO` (default `0.2`)
  - Search path now prefetches only adaptive top-K (instead of all FAISS IDs).
  - Added worker autotiering controls:
    - `GEOVIBES_PREFETCH_WORKER_TARGET` (default `44`)
    - `GEOVIBES_PREFETCH_WORKER_CAP` (default `44`)
    - `GEOVIBES_PREFETCH_WORKER_FLOOR` (default `16`)
    - `GEOVIBES_PREFETCH_WORKER_MEM_MB` (default `96`)
  - Added memory-aware worker capping from `/proc/meminfo` to avoid overloading small VMs.
  - Added bounded app-level embedding LRU behavior over `state.cached_embeddings`:
    - `GEOVIBES_EMBEDDING_LRU_SIZE` (default `2000`)
    - keeps labeled IDs protected from eviction.
  - Unified cache-touch + cache-write helpers to keep LRU recency coherent.

### Tests
- Added/updated tests in `tests/test_data_manager_remote.py`:
  - scheduler ordering check for `group_ids_for_prefetch`
  - metadata `values_join` path over geometry cache
  - embedding fetch-map `values_join` path
- Validation command:
  - `SKIP_NETWORK_TESTS=1 uv run pytest -q tests/test_data_manager_remote.py tests/test_remote_db.py tests/test_dropdown_grouping.py tests/test_data_manager_boundary.py`
  - Result: `25 passed, 17 skipped`

### OPus update review (new sections in `~/httpfs-opus/docs/httpfs_opus.md`)
OPus added new findings beyond earlier pool/worker work:
1. **Critical type-preservation warning**:
   - Rebuilds that convert `FLOAT[384]` -> `FLOAT[]` can catastrophically regress httpfs fetch.
   - This is plausible and consistent with sparse-access behavior we observed in split-store/parquet regressions.
2. **DB build method sensitivity**:
   - They report internal DuckDB layout/build path materially affects runtime even with same schema/data.
   - This is directionally credible; we have not yet reproduced their exact CTAS-vs-batched-insert deltas in our harness.
3. **Cluster-aligned storage (Hilbert/two-level)**:
   - They report `~16-48%` gains in their branch after fixing build/type issues.
   - We previously rejected IVF-list reorder variants based on our artifact path; this new OPus result suggests we should retest only with strict `FLOAT[384]` + native DuckDB build constraints.

### OPus mistakes / caveats still present
- Some summary claims still mix prefetch-centric and end-to-end metrics.
- Their recommendation to keep `enable_object_cache` true is not yet cleanly reconciled with conservative cache realism settings in our harness.
- A few sections infer “optimal” from limited query sets and low trial counts; our randomized queue still provides stronger robustness controls.

### Next concrete step from this review
- Re-open storage-layout optimization, but only under strict artifact rules:
  - preserve `FLOAT[384]` end-to-end
  - avoid parquet roundtrip artifacts for final DuckDB benchmark DBs
  - compare against current baseline with randomized multi-query harness settings.

## 2026-02-10 - Experiment 22 - `FLOAT[384]` guardrail implementation in artifact scripts

### Goal
Close the exact failure mode called out by OPus (`FLOAT[384]` degrading to `FLOAT[]`) in our own variant build/materialization scripts.

### Changes
- `experiments/httpfs_eval_harness/build_embedding_variants.py`
  - Removed explicit `CAST(... AS FLOAT[])` in parquet export selects.
  - Exports now select `embedding` directly from source table.
- `experiments/httpfs_eval_harness/materialize_duckdb_variant.py`
  - Added `--embedding-dim` (default `384`).
  - Materialization now:
    1. creates explicit schema `embedding FLOAT[embedding_dim]`
    2. inserts with `CAST(embedding AS FLOAT[embedding_dim])`
    3. creates `id_idx` index
    4. prints resulting embedding column type for verification.
- `experiments/httpfs_eval_harness/README.md`
  - Updated materialization command to include `--embedding-dim 384`.

### Validation
- `python -m py_compile` passed for modified scripts.
- Regression suite still passes:
  - `25 passed, 17 skipped` (`tests/test_data_manager_remote.py`, `tests/test_remote_db.py`, `tests/test_dropdown_grouping.py`, `tests/test_data_manager_boundary.py`).

### Impact
- Prevents future benchmark artifacts from silently taking the known slow path due to type degradation.
- Makes storage-layout retests against OPus findings more trustworthy.

## 2026-02-10 - Experiment 23 - Native embedding expression vs explicit cast (`q21`)

### Goal
Check whether `CAST(embedding AS FLOAT[])` is adding avoidable overhead on `FLOAT[384]` tables.

### Report
- `experiments/httpfs_eval_harness/results/queue_q21_embedding_expr_native_vs_cast.json`

### Results
- `embedding_native_expr`: search mean `10593.2ms`, prefetch mean `10469.7ms`
- `embedding_cast_float_list`: search mean `10696.3ms`, prefetch mean `10595.5ms`
- FAISS overlap unchanged (`1.0` for both).

### Conclusion
- Native `embedding` selection is consistently (slightly) faster than explicit cast in this lane.
- Implement native embedding select expression by default, while keeping an env override for compatibility.

## 2026-02-10 - Experiment 24 - Notebook-style iterative sessions (`q22`, corrected)

### Goal
Profile “real usage” iterative search behavior over many searches, baseline vs optimized strategy.

### First run issue (invalid)
- Interleaving strategies with shared embedding LRU caused cross-strategy cache contamination.
- This made the second strategy in sequence unrealistically appear as near-zero prefetch.

### Harness fix
- Added `embedding_cache_scope: strategy` support in harness to isolate cache per strategy.
- Reran `q22` with corrected scope.

### Corrected report
- `experiments/httpfs_eval_harness/results/queue_q22_realworld_iterative_baseline_vs_opt.json`

### Corrected results (30 trials)
- `baseline_oldlike_w32`: search mean `7819.4ms`, p50 `7754.8ms`, p95 `9580.1ms`
- `optimized_current_w44`: search mean `4489.8ms`, p50 `4329.4ms`, p95 `6078.3ms`
- Both kept overlap `1.0` (same FAISS candidate set semantics).

### Conclusion
- Under iterative multi-search workloads, current optimized strategy is materially better than old baseline (`~42.6%` faster mean search-path).

## 2026-02-10 - Experiment 25 - Query expansion sweep (`q23`)

### Goal
Evaluate whether adding nearest neighbors into the query vector improves outcomes enough to justify overhead.

### Report
- `experiments/httpfs_eval_harness/results/queue_q23_query_expansion_sweep.json`

### Results
- `expand_none`: search mean `4755.9ms`, overlap `1.000`
- `expand_top8_w015`: search mean `6440.0ms`, overlap `0.948`, query-expand overhead `1800.6ms`
- `expand_top16_w025`: search mean `8104.3ms`, overlap `0.902`, query-expand overhead `3013.6ms`

### Conclusion
- Query expansion is a regression for this workflow: higher latency and lower overlap vs baseline.
- Do not implement neighbor-assisted query expansion in runtime search path.

## 2026-02-10 - Experiment 26 - Dtype lane: float32 vs quantized uint8 session benchmarks (`q24`, `q25`)

### Goal
Compare end-to-end retrieval behavior for float and quantized DINO variants under the same session-like harness setup.

### Reports
- `experiments/httpfs_eval_harness/results/queue_q24_dtype_float32_session.json`
- `experiments/httpfs_eval_harness/results/queue_q25_dtype_quantized_session.json`

### Results
- `float32_opt_session`: search mean `1828.4ms`, prefetch mean `1718.4ms`
- `quantized_opt_session`: search mean `1648.7ms`, prefetch mean `1524.1ms`
- Quantized lane is faster in this setup (`~9.8%` search mean improvement).

### Important caveat
- These two runs use each variant’s own evolving feedback sequence.
- They show throughput behavior, but are not by themselves a drop-in quality equivalence claim.

## 2026-02-10 - Experiment 27 - Cross-dataset fixed-query overlap (float32 vs quantized)

### Goal
Directly measure top-K overlap on identical fixed query triplets to estimate retrieval consistency vs speed.

### Reports
- `/tmp/httpfs_dtype_float_fixed.json`
- `/tmp/httpfs_dtype_quant_fixed.json`

### Results
- Float fixed-query mean search: `3914.0ms`
- Quantized fixed-query mean search: `2320.3ms` (faster)
- Cross-dataset overlap was extremely low:
  - overlap@100 `0.0017`
  - overlap@200 `0.0008`
  - overlap@500 `0.0007`
  - overlap@1000 `0.0003`

### Interpretation
- Quantized variant behaves like a different retrieval space/model, not a drop-in numeric compression of float outputs.
- It can be used as an alternative speed-focused variant, but requires independent quality acceptance criteria.

## 2026-02-10 - Experiment 28 - Int16 simulation on float source (`q26`)

### Goal
Test int16-style quantization simulation (query-only and query+prefetch expression) without changing storage layout.

### Report
- `experiments/httpfs_eval_harness/results/queue_q26_dtype_int16_sim.json`

### Results
- `float32_native`: search mean `4806.0ms`
- `int16_sim_query_only`: search mean `5074.4ms`
- `int16_sim_query_prefetch`: search mean `4944.6ms`
- Overlap remained effectively unchanged (`~1.0`), but latency regressed.

### Conclusion
- Int16 simulation does not improve performance in this path and is not recommended.

## 2026-02-10 - Final implementation update

### Implemented from this pass
1. Runtime embedding fetch path now defaults to native `embedding` selection:
   - `geovibes/ui/data_manager.py`
   - Detects embedding type and uses native fixed-array reads by default.
   - Supports explicit override via `GEOVIBES_EMBEDDING_SELECT_EXPR`.
2. Harness improvements for realistic/robust evaluation:
   - iterative feedback query-set generation
   - strategy-level query embedding expression
   - query expansion measurement lane (`query_expand_*`)
   - FAISS overlap metrics in summaries
   - `embedding_cache_scope: strategy` to prevent cross-strategy cache contamination.
3. Queue expansion (`q21`-`q26`) for dtype/query-expansion/session validation.

### Final recommendations for PR
- Keep current optimized retrieval shape:
  - metadata `values_join`
  - prefetch `in_list`
  - `id_ascending` batching
  - worker target `w44` with memory-aware fallback
  - adaptive prefetch depth (`~100-200`)
  - app embedding LRU (`>=2000`)
- Keep query vector baseline (no neighbor expansion).
- Use native `embedding` SQL expression over explicit cast for float fixed-array sources.
- Treat quantized datasets as an optional alternate model path, not a direct replacement for float32 semantics.
