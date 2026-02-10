# Cluster-Aligned Storage Benchmark

This experiment compares three storage orderings for optimizing httpfs embedding fetch:

1. **Baseline**: Current storage (by insertion ID)
2. **Hilbert-ordered**: Embeddings sorted by Hilbert-ordered cluster ID
3. **Two-level IVF**: Embeddings sorted by super-cluster (44 groups = row groups)

## Experiment Design

### Independent Variables
- Storage ordering: baseline, hilbert, two-level
- Fetch method: ThreadPoolExecutor (32 workers) vs DuckDB native threads
- Number of embeddings: 100, 500, 1000

### Dependent Variables
- Fetch time (seconds)
- Row groups touched

### Controls
- Same FAISS index for search (ensures same result IDs)
- Cache clearing between attempts
- Multiple trials (n=5) for statistical significance

## Directory Structure

```
cluster_storage_benchmark/
├── README.md                    # This file
├── 01_reorder_database.py       # Reorder DB by cluster schemes
├── 02_upload_to_s3.py           # Upload reordered DBs to S3
├── 03_run_benchmark.py          # Run fetch benchmarks
├── 04_plot_results.py           # Generate bar plots with error bars
├── config.yaml                  # Experiment configuration
└── results/                     # Output directory
    ├── benchmark_results.json
    └── figures/
```

## Quick Start

```bash
# 1. Configure experiment
vim config.yaml

# 2. Reorder databases (creates local copies)
uv run python 01_reorder_database.py

# 3. Upload to S3 test folder
uv run python 02_upload_to_s3.py

# 4. Run benchmarks (on VM with good network)
uv run python 03_run_benchmark.py

# 5. Generate plots
uv run python 04_plot_results.py
```

## Expected Results

| Ordering | Expected Row Groups (500 results) | Expected Speedup |
|----------|-----------------------------------|------------------|
| Baseline | ~44 | 1x |
| Hilbert | ~8 | ~5x |
| Two-level | ~8-12 | ~4-5x |
