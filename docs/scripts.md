# CLI Tools Reference

GeoVibes includes command-line tools for data preparation and database management.

---

## download_embeddings.py

Interactive script to download pre-built databases from the manifest.

### Usage

```bash
uv run download_embeddings.py
```

### Behavior

1. Parses `manifest.csv` for available databases
2. Fetches remote file sizes via HEAD requests
3. Presents interactive selection menu
4. Downloads selected `.tar.gz` archives with resume support
5. Extracts to `local_databases/`

### Manifest Format

CSV file with columns:
- `region` — Geographic region name
- `model_name` — Embedding model identifier
- `model_path` — S3 URL to tarball

### Output

Creates `local_databases/{model_name}/`:
- `{name}.duckdb` — Metadata and embeddings
- `{name}.faiss` — Vector search index

---

## geovibes/database/faiss_db.py

Builds FAISS index and DuckDB database from embedding Parquet files.

### Usage

```bash
# From specific Parquet files
python geovibes/database/faiss_db.py \
  path/to/embeddings/*.parquet \
  --name my_model \
  --output_dir local_databases

# From ROI + MGRS tiles (automatic discovery)
python geovibes/database/faiss_db.py \
  --roi-file geometries/region.geojson \
  --mgrs-reference-file geometries/mgrs_tiles.parquet \
  --embedding-dir s3://bucket/embeddings/ \
  --name region_model \
  --tile-pixels 32 \
  --tile-overlap 16 \
  --tile-resolution 10 \
  --output_dir local_databases
```

### Arguments

#### Input Sources (mutually exclusive)

| Argument | Description |
|----------|-------------|
| `parquet_files` | Positional: paths to Parquet files or directories (local/S3/GCS) |
| `--roi-file` | GeoJSON file defining region of interest |
| `--mgrs-reference-file` | Parquet with MGRS tile geometries (required with `--roi-file`) |
| `--embedding-dir` | Cloud directory containing embeddings (required with `--roi-file`) |

#### Output Configuration

| Argument | Default | Description |
|----------|---------|-------------|
| `--name` | Required | Base name for output files |
| `--output_dir` | `local_databases` | Output directory (local or cloud) |

#### Tile Specification

| Argument | Default | Description |
|----------|---------|-------------|
| `--tile-pixels` | 32 | Tile size in pixels |
| `--tile-overlap` | 16 | Overlap between tiles in pixels |
| `--tile-resolution` | 10 | Ground sample distance (meters/pixel) |

These values are appended to output filename: `{name}_{pixels}_{overlap}_{resolution}.duckdb`

#### Embedding Configuration

| Argument | Default | Description |
|----------|---------|-------------|
| `--embedding_col` | `embedding` | Column name in Parquet files |
| `--dtype` | `FLOAT` | Embedding type: `FLOAT` or `INT8` |

#### FAISS Index Parameters

| Argument | Default | Description |
|----------|---------|-------------|
| `--nlist` | 4096 | Number of IVF clusters |
| `--m` | 64 | PQ sub-quantizers (FLOAT only) |
| `--nbits` | 8 | Bits per sub-quantizer code (FLOAT only) |
| `--batch_size` | 500,000 | Batch size for index population |

#### Development Options

| Argument | Description |
|----------|-------------|
| `--dry-run` | Process only first N files |
| `--dry-run-size` | Number of files for dry run (default: 5) |

### Pipeline Phases

1. **File Discovery**: Find Parquet files (local or cloud)
2. **Caching**: Download cloud files to temp directory with hash-based cache
3. **Ingestion**: Load data into DuckDB with schema detection
4. **Index Building**: Create FAISS IVF index with training and population
5. **Packaging**: Create tarball if output is cloud destination

### Output Files

| File | Description |
|------|-------------|
| `{name}_{pixels}_{overlap}_{resolution}.duckdb` | DuckDB database with `geo_embeddings` table |
| `{name}_{pixels}_{overlap}_{resolution}.faiss` | FAISS IVF index |
| `{name}.tar.gz` | Compressed archive (cloud output only) |

### Log File

Creates `faiss_build.log` with detailed progress.

---

## Example Workflows

### Build index for a new region

```bash
# 1. Prepare ROI geometry
# Create geometries/my_region.geojson with your area of interest

# 2. Build database
python geovibes/database/faiss_db.py \
  --roi-file geometries/my_region.geojson \
  --mgrs-reference-file geometries/mgrs_tiles.parquet \
  --embedding-dir s3://bucket/ssl4eo-embeddings/ \
  --name my_region_ssl4eo \
  --tile-pixels 32 \
  --tile-overlap 16 \
  --tile-resolution 10

# 3. Database is ready at:
# local_databases/my_region_ssl4eo_32_16_10.duckdb
# local_databases/my_region_ssl4eo_32_16_10.faiss
```

### Test pipeline with dry run

```bash
python geovibes/database/faiss_db.py \
  --roi-file geometries/test_region.geojson \
  --mgrs-reference-file geometries/mgrs_tiles.parquet \
  --embedding-dir s3://bucket/embeddings/ \
  --name test \
  --dry-run \
  --dry-run-size 3
```

### Build from local Parquet files

```bash
python geovibes/database/faiss_db.py \
  data/embeddings/*.parquet \
  --name local_model \
  --embedding_col embedding_vector \
  --dtype INT8
```

---

## geovibes/database/benchmark_httpfs.py

Benchmarks DuckDB queries over httpfs (S3) vs local access.

### Usage

```bash
# Benchmark remote S3 database vs local
python -m geovibes.database.benchmark_httpfs \
  --s3-url "s3://bucket/path/database.db" \
  --local-path "local_databases/database.db" \
  --endpoint-url "data.source.coop"

# Custom coordinates
python -m geovibes.database.benchmark_httpfs \
  --s3-url "s3://bucket/db.db" \
  --lon -86.5 --lat 32.5
```

### Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `--s3-url` | Yes | S3 URL to DuckDB database |
| `--local-path` | No | Local database path for comparison |
| `--lon`, `--lat` | No | Coordinates for spatial queries (default: Alabama) |
| `--endpoint-url` | No | Custom S3 endpoint (e.g., `data.source.coop`) |
| `--output` | No | Output JSON file (default: `benchmark_results.json`) |

### Benchmarks Performed

1. **Cold start**: Time to connect and run first query
2. **Nearest point**: Spatial distance query (10 runs)
3. **Fetch by ID**: Fetch 10, 50, 100, 200, 500, 1000 rows (5 runs each)
4. **Local baseline**: Same queries on local database (if provided)

### Acceptance Criteria

| Metric | Threshold |
|--------|-----------|
| Cold start | < 5000 ms |
| Nearest point | < 500 ms |
| Fetch 100 | < 1000 ms |
| Fetch 500 | < 3000 ms |
| Remote/local ratio | < 10x |

---

## geovibes/database/upload_for_httpfs.py

Uploads DuckDB database to Source Cooperative S3 for httpfs benchmarking.

### Usage

```bash
# With session token from Source Cooperative console
python -m geovibes.database.upload_for_httpfs \
  local_databases/database.db \
  --session-token "YOUR_SESSION_TOKEN"
```

### Note

Static Source Cooperative credentials are read-only. Session tokens with write access are required from the Source Cooperative console.

---

## geovibes/database/faiss_cache.py

Downloads and caches FAISS indexes from S3 for local access.

### Usage

```python
from geovibes.database.faiss_cache import FaissCache

cache = FaissCache()  # Uses ~/.cache/geovibes/faiss/

# Download and cache (or load from cache)
index = cache.get_index("s3://bucket/path/index.faiss")

# Check if cached
if cache.is_cached("s3://bucket/path/index.faiss"):
    print("Already downloaded")

# Clear all cached indexes
cache.clear_cache()
```

### Features

- SHA256-based cache keys for deterministic paths
- Atomic downloads (temp file + rename)
- Validates indexes on load
- Memory-mapped loading for efficiency

---

## geovibes/database/remote_db.py

Connects to DuckDB databases on S3 via httpfs extension.

### Usage

```python
from geovibes.database.remote_db import RemoteDuckDB

# Connect to remote database
db = RemoteDuckDB()
db.connect("s3://bucket/path/database.db")

# Query with SQL
df = db.query("SELECT * FROM geo_embeddings LIMIT 10")

# Fetch by ID (optimized for FAISS workflow)
result = db.fetch_by_ids([1, 2, 3], columns=["id", "geometry"])

# Fetch embeddings
embeddings = db.fetch_embeddings([1, 2, 3])

db.close()

# Or use as context manager
with RemoteDuckDB() as db:
    db.connect("s3://bucket/db.db")
    result = db.query("SELECT COUNT(*) FROM geo_embeddings")
```

### Error Handling

| Error | Exception |
|-------|-----------|
| Database not found | `FileNotFoundError` |
| Access denied | `PermissionError` |
| Connection timeout | `TimeoutError` |

---

## Related Files

- `manifest.csv` — Database download manifest
- `geovibes/database/cloud.py` — S3/GCS file operations
- `geovibes/tiling.py` — MGRS tile grid generation
- `docs/httpfs.md` — Remote database access planning document
- `docs/source.md` — Source Cooperative access documentation
