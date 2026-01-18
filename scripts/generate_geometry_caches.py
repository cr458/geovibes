#!/usr/bin/env python3
"""Generate geometry cache Parquet files from sorted DuckDB databases on S3.

The geometry cache contains just id + geometry columns for fast local queries,
avoiding the need to fetch scattered IDs over httpfs during search.
"""

import argparse
import logging
import os
import tempfile
from pathlib import Path

import boto3
import duckdb
from botocore.config import Config
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

ENDPOINT_URL = "https://s3.us-west-2.amazonaws.com"
BUCKET = "us-west-2.opendata.source.coop"

DATABASES = [
    {
        "name": "alabama_google_satellite_embeddings_v1_2024_2025_25_0_10",
        "s3_key": "geovibes/search/USA/alabama/2024-01-01-2025-01-01/alabama_google_satellite_embeddings_v1_2024_2025_25_0_10_metadata_sorted.db",
        "output_prefix": "geovibes/search/USA/alabama/2024-01-01-2025-01-01",
    },
    {
        "name": "new-mexico_google_satellite_embeddings_v1_2024_2025_25_0_10",
        "s3_key": "geovibes/search/USA/new-mexico/2024-01-01-2025-01-01/new-mexico_google_satellite_embeddings_v1_2024_2025_25_0_10_metadata_sorted.db",
        "output_prefix": "geovibes/search/USA/new-mexico/2024-01-01-2025-01-01",
    },
    {
        "name": "alabama_earthgenome_softcon_2024_2025_32_16_10",
        "s3_key": "geovibes/search/USA/alabama/2024-01-01-2025-01-01/alabama_earthgenome_softcon_2024_2025_32_16_10_metadata_sorted.db",
        "output_prefix": "geovibes/search/USA/alabama/2024-01-01-2025-01-01",
    },
    {
        "name": "alabama_dino_vit_small_patch16_224_2024_2025_32_16_10",
        "s3_key": "geovibes/search/USA/alabama/2024-01-01-2025-01-01/alabama_dino_vit_small_patch16_224_2024_2025_32_16_10_metadata_sorted.db",
        "output_prefix": "geovibes/search/USA/alabama/2024-01-01-2025-01-01",
    },
    {
        "name": "alabama_clay_naip_v1.5_2019_2020_256_0_1",
        "s3_key": "geovibes/search/USA/alabama/2019-01-01-2020-01-01/alabama_clay_naip_v1.5_2019_2020_256_0_1_metadata_sorted.db",
        "output_prefix": "geovibes/search/USA/alabama/2019-01-01-2020-01-01",
    },
]


def get_s3_client():
    """Create S3 client for Source Cooperative."""
    return boto3.client(
        "s3",
        endpoint_url=ENDPOINT_URL,
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
        region_name="us-west-2",
        config=Config(signature_version="s3v4"),
    )


def download_with_progress(s3_client, bucket, key, dest_path):
    """Download file from S3 with progress bar."""
    response = s3_client.head_object(Bucket=bucket, Key=key)
    total_size = response["ContentLength"]

    logger.info(f"Downloading {key} ({total_size / 1e9:.2f} GB)")

    with tqdm(total=total_size, unit="B", unit_scale=True, desc="Downloading") as pbar:
        s3_client.download_file(bucket, key, str(dest_path), Callback=pbar.update)


def upload_with_progress(s3_client, local_path, bucket, key):
    """Upload file to S3 with progress bar."""
    file_size = local_path.stat().st_size

    logger.info(f"Uploading to {key} ({file_size / 1e6:.1f} MB)")

    with tqdm(total=file_size, unit="B", unit_scale=True, desc="Uploading") as pbar:
        s3_client.upload_file(str(local_path), bucket, key, Callback=pbar.update)

    logger.info(f"✓ Uploaded to s3://{bucket}/{key}")


def generate_geometry_cache(db_path: Path, output_path: Path):
    """Extract id + geometry from database to compressed Parquet."""
    logger.info(f"Generating geometry cache from {db_path.name}")

    con = duckdb.connect(str(db_path), read_only=True)
    con.execute("INSTALL spatial; LOAD spatial;")

    row_count = con.execute("SELECT COUNT(*) FROM geo_embeddings").fetchone()[0]
    logger.info(f"Database has {row_count:,} rows")

    logger.info("Exporting geometry cache (this may take a moment)...")
    con.execute(f"""
        COPY (SELECT id, geometry FROM geo_embeddings ORDER BY id)
        TO '{output_path}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)

    con.close()

    size_mb = output_path.stat().st_size / 1e6
    logger.info(f"✓ Created geometry cache: {output_path.name} ({size_mb:.1f} MB)")


def check_existing(s3_client, bucket, key):
    """Check if a file already exists on S3."""
    try:
        s3_client.head_object(Bucket=bucket, Key=key)
        return True
    except:
        return False


def process_database(db_config, work_dir: Path, skip_existing: bool = True):
    """Process a single database: download, generate cache, upload."""
    name = db_config["name"]
    s3_key = db_config["s3_key"]
    output_prefix = db_config["output_prefix"]

    cache_key = f"{output_prefix}/{name}_geometry_cache.parquet"

    s3 = get_s3_client()

    if skip_existing and check_existing(s3, BUCKET, cache_key):
        logger.info(f"⏭ Skipping {name} - geometry cache already exists")
        return

    logger.info(f"\n{'='*60}")
    logger.info(f"Processing: {name}")
    logger.info(f"{'='*60}")

    # Download sorted database
    db_path = work_dir / Path(s3_key).name
    download_with_progress(s3, BUCKET, s3_key, db_path)

    # Generate geometry cache
    cache_path = work_dir / f"{name}_geometry_cache.parquet"
    generate_geometry_cache(db_path, cache_path)

    # Delete database to save space
    db_path.unlink()
    logger.info("Deleted local database to save space")

    # Upload geometry cache
    upload_with_progress(s3, cache_path, BUCKET, cache_key)

    # Clean up
    cache_path.unlink()

    logger.info(f"✓ Completed: {name}")


def main():
    parser = argparse.ArgumentParser(description="Generate geometry caches for DuckDB databases")
    parser.add_argument("--db", type=str, help="Process only this database (by name)")
    parser.add_argument("--work-dir", type=Path, default=Path("./geometry_cache_work"), help="Working directory")
    parser.add_argument("--no-skip", action="store_true", help="Don't skip existing caches")
    args = parser.parse_args()

    args.work_dir.mkdir(parents=True, exist_ok=True)

    if args.db:
        dbs = [d for d in DATABASES if d["name"] == args.db]
        if not dbs:
            logger.error(f"Database not found: {args.db}")
            logger.info("Available databases:")
            for d in DATABASES:
                logger.info(f"  - {d['name']}")
            return 1
    else:
        dbs = DATABASES

    logger.info(f"Will process {len(dbs)} database(s)")

    for db_config in dbs:
        process_database(db_config, args.work_dir, skip_existing=not args.no_skip)

    logger.info("\n✓ All geometry caches processed!")


if __name__ == "__main__":
    main()
