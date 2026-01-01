# Source Cooperative Access

GeoVibes databases are hosted on [Source Cooperative](https://source.coop/), an S3-compatible data repository for open geospatial data.

---

## Bucket Structure

| Component | Value |
|-----------|-------|
| Bucket | `us-west-2.opendata.source.coop` |
| Base path | `geovibes/search/` |
| Region | `us-west-2` |

### File Organization

```
s3://us-west-2.opendata.source.coop/
└── geovibes/
    └── search/
        └── USA/
            └── alabama/
                └── 2024-01-01-2025-01-01/
                    ├── alabama_dino_vit_*.tar.gz
                    ├── alabama_earthgenome_*.tar.gz
                    ├── alabama_google_satellite_*.tar.gz
                    ├── alabama_google_satellite_*_metadata.db  # For httpfs
                    └── ...
```

---

## Read Access (Public)

Static credentials for read-only access:

```python
import boto3
from botocore.config import Config

s3 = boto3.client(
    's3',
    endpoint_url='https://data.source.coop',
    aws_access_key_id='***REMOVED***',
    aws_secret_access_key='***REMOVED***',
    region_name='us-west-2',
    config=Config(signature_version='s3v4')
)
```

### DuckDB Configuration

```python
import duckdb

conn = duckdb.connect()
conn.execute("INSTALL httpfs; LOAD httpfs;")
conn.execute("INSTALL aws; LOAD aws;")

# Source Cooperative config
conn.execute("SET s3_access_key_id='***REMOVED***';")
conn.execute("SET s3_secret_access_key='***REMOVED***';")
conn.execute("SET s3_endpoint='data.source.coop';")
conn.execute("SET s3_url_style='path';")
conn.execute("SET s3_region='us-west-2';")

# Query remote database
conn.execute("ATTACH 's3://geovibes/search/USA/alabama/...' AS remote (READ_ONLY)")
```

---

## Write Access (Authenticated)

Write access requires session tokens from the Source Cooperative console.

### Getting Credentials

1. Log in to [source.coop](https://source.coop/)
2. Navigate to the geovibes repository
3. Generate temporary credentials
4. Export the credentials:

```bash
export AWS_ACCESS_KEY_ID="ASIAWCQM3Z36..."
export AWS_SECRET_ACCESS_KEY="..."
export AWS_SESSION_TOKEN="IQoJb3JpZ2luX2VjEBY..."
export AWS_DEFAULT_REGION="us-west-2"
```

### Upload with boto3

**Important**: Use direct AWS S3 (no custom endpoint) for write access:

```python
import boto3

s3 = boto3.client(
    's3',
    aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'],
    aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY'],
    aws_session_token=os.environ['AWS_SESSION_TOKEN'],
    region_name='us-west-2',
)

# Upload file
s3.upload_file(
    'local_databases/database.db',
    'us-west-2.opendata.source.coop',
    'geovibes/search/USA/alabama/2024-01-01-2025-01-01/database.db'
)
```

### Key Differences

| Operation | Endpoint | Credentials |
|-----------|----------|-------------|
| Read | `https://data.source.coop` | Static (`SCRPTLC...`) |
| Write | Direct AWS S3 (no endpoint) | Session token required |

---

## httpfs Benchmarking

For querying databases via httpfs, see [`docs/httpfs.md`](httpfs.md).

### Quick Test

```bash
# Run benchmarks
python -m geovibes.database.benchmark_httpfs \
  --s3-url "s3://us-west-2.opendata.source.coop/geovibes/search/USA/alabama/2024-01-01-2025-01-01/alabama_google_satellite_embeddings_v1_2024_2025_25_0_10_metadata.db" \
  --local-path "local_databases/alabama_google_satellite_embeddings_v1_2024_2025_25_0_10_metadata.db"
```

---

## Manifest

Available databases are listed in `manifest.csv`:

```csv
region,model_name,model_path
alabama,alabama_dino_vit_...,s3://us-west-2.opendata.source.coop/geovibes/...
```

Use `download_embeddings.py` to download databases from the manifest.

---

## Troubleshooting

### 401 Unauthorized on Write

Session tokens expire. Get fresh credentials from the Source Cooperative console.

### 404 Not Found on Read

- Check bucket name: `us-west-2.opendata.source.coop` (not just `geovibes`)
- For DuckDB with custom endpoint, use short bucket name: `geovibes`

### Different Bucket Names

| Context | Bucket Name |
|---------|-------------|
| boto3 with `endpoint_url` | `geovibes` |
| boto3 direct AWS | `us-west-2.opendata.source.coop` |
| DuckDB with `s3_endpoint` | `geovibes` |
| Full S3 URI | `s3://us-west-2.opendata.source.coop/geovibes/...` |
