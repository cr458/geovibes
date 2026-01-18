#!/usr/bin/env python3
"""Download, optimize, and upload DuckDB databases for httpfs performance.

Optimization includes:
1. Sorting table by ID for optimal zone map performance
2. Creating explicit index on ID column
3. Creating spatial R-Tree index (if geometry column exists)
"""

import argparse
import logging
import os
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path

import boto3
import duckdb
from botocore.config import Config
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

ENDPOINT_URL = "https://data.source.coop"
BUCKET = "geovibes"

DATABASES = [
    {
        "name": "alabama_dino_vit_small_patch16_224_2024_2025_32_16_10",
        "tar_key": "search/USA/alabama/2024-01-01-2025-01-01/alabama_dino_vit_small_patch16_224_2024_2025_32_16_10.tar.gz",
        "output_prefix": "search/USA/alabama/2024-01-01-2025-01-01",
    },
    {
        "name": "alabama_earthgenome_softcon_2024_2025_32_16_10",
        "tar_key": "search/USA/alabama/2024-01-01-2025-01-01/alabama_earthgenome_softcon_2024_2025_32_16_10.tar.gz",
        "output_prefix": "search/USA/alabama/2024-01-01-2025-01-01",
    },
    {
        "name": "alabama_clay_naip_v1.5_2019_2020_256_0_1",
        "tar_key": "search/USA/alabama/2019-01-01-2020-01-01/alabama_clay_naip_v1.5_2019_2020_256_0_1.tar.gz",
        "output_prefix": "search/USA/alabama/2019-01-01-2020-01-01",
    },
    {
        "name": "new-mexico_google_satellite_embeddings_v1_2024_2025_25_0_10",
        "tar_key": "search/USA/new-mexico/2024-01-01-2025-01-01/new-mexico_google_satellite_embeddings_v1_2024_2025_25_0_10.tar.gz",
        "output_prefix": "search/USA/new-mexico/2024-01-01-2025-01-01",
    },
]


def get_s3_client(for_write=False):
    """Create S3 client for Source Cooperative."""
    if for_write:
        return boto3.client(
            "s3",
            endpoint_url=ENDPOINT_URL,
            region_name="us-west-2",
            config=Config(signature_version="s3v4"),
        )
    else:
        return boto3.client(
            "s3",
            endpoint_url=ENDPOINT_URL,
            aws_access_key_id="SCRPTLC1EMW2CQLBT59M15YC",
            aws_secret_access_key="RfCYYNR6ZRfjW2thyntnaG6pNFC9Vil1PS943rx2ElSzOdBRCcuAGW5gq9hau72H",
            region_name="us-west-2",
            config=Config(signature_version="s3v4"),
        )


def download_with_progress(s3_client, bucket, key, dest_path):
    """Download file from S3 with progress bar."""
    response = s3_client.head_object(Bucket=bucket, Key=key)
    total_size = response["ContentLength"]

    logger.info(f"Downloading {key} ({total_size / 1e9:.2f} GB)")

    with tqdm(total=total_size, unit="B", unit_scale=True, desc="Downloading") as pbar:
        s3_client.download_file(
            bucket, key, str(dest_path),
            Callback=pbar.update
        )


def upload_with_progress(s3_client, local_path, bucket, key):
    """Upload file to S3 with progress bar."""
    file_size = local_path.stat().st_size

    logger.info(f"Uploading to {key} ({file_size / 1e9:.2f} GB)")

    with tqdm(total=file_size, unit="B", unit_scale=True, desc="Uploading") as pbar:
        s3_client.upload_file(
            str(local_path), bucket, key,
            Callback=pbar.update
        )

    logger.info(f"✓ Uploaded to s3://{bucket}/{key}")


def extract_db_from_tar(tar_path: Path, work_dir: Path) -> Path:
    """Extract _metadata.db file from tar.gz archive."""
    logger.info(f"Extracting database from {tar_path.name}")

    with tarfile.open(tar_path, "r:gz") as tar:
        members = tar.getmembers()
        db_members = [m for m in members if m.name.endswith("_metadata.db")]

        if not db_members:
            raise ValueError(f"No _metadata.db found in {tar_path}")

        db_member = db_members[0]
        logger.info(f"Found: {db_member.name} ({db_member.size / 1e9:.2f} GB)")

        tar.extract(db_member, work_dir)

        extracted_path = work_dir / db_member.name
        if "/" in db_member.name:
            final_path = work_dir / Path(db_member.name).name
            shutil.move(extracted_path, final_path)
            shutil.rmtree(work_dir / db_member.name.split("/")[0], ignore_errors=True)
            return final_path
        return extracted_path


def extract_index_from_tar(tar_path: Path, work_dir: Path) -> Path | None:
    """Extract FAISS .index file from tar.gz archive if present."""
    logger.info(f"Looking for FAISS index in {tar_path.name}")

    with tarfile.open(tar_path, "r:gz") as tar:
        members = tar.getmembers()
        index_members = [m for m in members if m.name.endswith(".index")]

        if not index_members:
            logger.info("No FAISS index found in archive")
            return None

        index_member = index_members[0]
        logger.info(f"Found: {index_member.name} ({index_member.size / 1e6:.1f} MB)")

        tar.extract(index_member, work_dir)

        extracted_path = work_dir / index_member.name
        if "/" in index_member.name:
            final_path = work_dir / Path(index_member.name).name
            shutil.move(extracted_path, final_path)
            return final_path
        return extracted_path


def optimize_database(input_db: Path, output_db: Path):
    """Sort database by ID and create indexes for optimal httpfs performance."""
    logger.info(f"Optimizing database: {input_db.name}")

    con = duckdb.connect(str(input_db), read_only=True)

    row_count = con.execute("SELECT COUNT(*) FROM geo_embeddings").fetchone()[0]
    logger.info(f"Database has {row_count:,} rows")

    columns = con.execute("DESCRIBE geo_embeddings").fetchall()
    column_names = [c[0] for c in columns]
    has_geometry = "geometry" in column_names
    logger.info(f"Has geometry column: {has_geometry}")

    con.close()

    out_con = duckdb.connect(str(output_db))

    if has_geometry:
        out_con.execute("INSTALL spatial; LOAD spatial;")

    logger.info("Attaching source database...")
    out_con.execute(f"ATTACH '{input_db}' AS source (READ_ONLY)")

    logger.info("Creating sorted table (this may take a while)...")
    out_con.execute("""
        CREATE TABLE geo_embeddings AS
        SELECT * FROM source.geo_embeddings
        ORDER BY id
    """)

    out_con.execute("DETACH source")

    if has_geometry:
        logger.info("Creating spatial R-Tree index...")
        out_con.execute("CREATE INDEX geom_spatial_idx ON geo_embeddings USING RTREE (geometry);")

    logger.info("Creating ID index...")
    out_con.execute("CREATE INDEX id_idx ON geo_embeddings(id);")

    final_count = out_con.execute("SELECT COUNT(*) FROM geo_embeddings").fetchone()[0]
    logger.info(f"Optimized database has {final_count:,} rows")

    out_con.close()

    logger.info(f"✓ Optimized database saved: {output_db} ({output_db.stat().st_size / 1e9:.2f} GB)")


def check_existing(s3_client, bucket, key):
    """Check if a file already exists on S3."""
    try:
        s3_client.head_object(Bucket=bucket, Key=key)
        return True
    except:
        return False


def process_database(db_config, work_dir: Path, skip_existing: bool = True):
    """Process a single database: download, optimize, upload."""
    name = db_config["name"]
    tar_key = db_config["tar_key"]
    output_prefix = db_config["output_prefix"]

    sorted_db_key = f"{output_prefix}/{name}_metadata_sorted.db"
    index_key = f"{output_prefix}/{name}_faiss_4096_64_8.index"

    s3_read = get_s3_client(for_write=False)
    s3_write = get_s3_client(for_write=True)

    if skip_existing and check_existing(s3_read, BUCKET, sorted_db_key):
        logger.info(f"⏭ Skipping {name} - sorted DB already exists")
        return

    logger.info(f"\n{'='*60}")
    logger.info(f"Processing: {name}")
    logger.info(f"{'='*60}")

    tar_path = work_dir / Path(tar_key).name
    download_with_progress(s3_read, BUCKET, tar_key, tar_path)

    db_path = extract_db_from_tar(tar_path, work_dir)
    index_path = extract_index_from_tar(tar_path, work_dir)

    tar_path.unlink()
    logger.info("Deleted tar.gz to save space")

    sorted_db_path = work_dir / f"{name}_metadata_sorted.db"
    optimize_database(db_path, sorted_db_path)

    db_path.unlink()
    logger.info("Deleted original DB to save space")

    upload_with_progress(s3_write, sorted_db_path, BUCKET, sorted_db_key)

    if index_path and not check_existing(s3_read, BUCKET, index_key):
        upload_with_progress(s3_write, index_path, BUCKET, index_key)
    elif index_path:
        logger.info(f"⏭ Skipping index upload - already exists")

    sorted_db_path.unlink()
    if index_path:
        index_path.unlink()

    logger.info(f"✓ Completed: {name}")


def main():
    parser = argparse.ArgumentParser(description="Optimize DuckDB databases for httpfs")
    parser.add_argument("--db", type=str, help="Process only this database (by name)")
    parser.add_argument("--work-dir", type=Path, default=Path("./optimize_work"), help="Working directory")
    parser.add_argument("--no-skip", action="store_true", help="Don't skip existing databases")
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

    logger.info("\n✓ All databases processed!")


if __name__ == "__main__":
    main()
