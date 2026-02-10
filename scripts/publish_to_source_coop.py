#!/usr/bin/env python3
"""Publish a local database to Source Cooperative for remote httpfs access.

This script takes a local database (output of faiss_db.py) and:
1. Optimizes it for httpfs (sorts by ID, creates indexes)
2. Generates geometry cache for fast local queries
3. Uploads all files to the correct S3 location

Usage:
    python scripts/publish_to_source_coop.py \
        local_databases/alabama_dino_vit_small_patch16_224_2024_2025_32_16_10_metadata.db \
        --region alabama \
        --date-range 2024-01-01-2025-01-01

    # Auto-detect region from filename
    python scripts/publish_to_source_coop.py \
        local_databases/alabama_dino_vit_small_patch16_224_2024_2025_32_16_10_metadata.db

Output structure on S3:
    s3://us-west-2.opendata.source.coop/geovibes/search/USA/{region}/{date_range}/httpfs/{model_name}/
        metadata.db
        faiss.index
        geometry_cache.parquet
"""

import argparse
import logging
import os
import re
from pathlib import Path

import boto3
import duckdb
from botocore.config import Config
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# S3 configuration for Source Cooperative
S3_ENDPOINT = "https://s3.us-west-2.amazonaws.com"
S3_BUCKET = "us-west-2.opendata.source.coop"
S3_BASE_PREFIX = "geovibes/search/USA"


def get_s3_client():
    """Create S3 client using environment credentials."""
    return boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
        region_name="us-west-2",
        config=Config(signature_version="s3v4"),
    )


def upload_with_progress(s3_client, local_path: Path, bucket: str, key: str):
    """Upload file to S3 with progress bar."""
    file_size = local_path.stat().st_size
    size_str = (
        f"{file_size / 1e9:.2f} GB" if file_size > 1e9 else f"{file_size / 1e6:.1f} MB"
    )

    logger.info(f"Uploading {local_path.name} ({size_str})")

    with tqdm(total=file_size, unit="B", unit_scale=True, desc="Uploading") as pbar:
        s3_client.upload_file(str(local_path), bucket, key, Callback=pbar.update)

    logger.info(f"✓ Uploaded to s3://{bucket}/{key}")


def check_existing(s3_client, bucket: str, key: str) -> bool:
    """Check if a file already exists on S3."""
    try:
        s3_client.head_object(Bucket=bucket, Key=key)
        return True
    except Exception:
        return False


def parse_model_name(db_path: Path) -> str:
    """Extract model name from database filename."""
    # Remove _metadata.db suffix
    name = db_path.stem
    if name.endswith("_metadata"):
        name = name[:-9]
    return name


def parse_region_from_name(model_name: str) -> str | None:
    """Try to extract region from model name (e.g., 'alabama_dino_...' -> 'alabama')."""
    # Common region patterns at start of name
    regions = ["alabama", "new-mexico", "california", "texas", "florida"]
    name_lower = model_name.lower()
    for region in regions:
        if name_lower.startswith(region):
            return region
    # Try to extract first word before underscore
    parts = model_name.split("_")
    if parts:
        return parts[0].lower()
    return None


def parse_date_range_from_name(model_name: str) -> str | None:
    """Try to extract date range from model name (e.g., '..._2024_2025_...' -> '2024-01-01-2025-01-01')."""
    # Look for year patterns like _2024_2025_
    match = re.search(r"_(\d{4})_(\d{4})_", model_name)
    if match:
        start_year, end_year = match.groups()
        return f"{start_year}-01-01-{end_year}-01-01"
    return None


def find_faiss_index(db_path: Path, model_name: str) -> Path | None:
    """Find the FAISS index file for a database."""
    db_dir = db_path.parent

    # Try common patterns
    patterns = [
        f"{model_name}_faiss_*.index",
        f"{model_name}.index",
        f"{model_name}_faiss.index",
    ]

    for pattern in patterns:
        matches = list(db_dir.glob(pattern))
        if matches:
            return matches[0]

    # Also check for index in same directory with any faiss pattern
    all_indexes = list(db_dir.glob("*.index"))
    for idx in all_indexes:
        if model_name in idx.name:
            return idx

    return None


def optimize_database(input_db: Path, output_db: Path) -> None:
    """Sort database by ID and create indexes for optimal httpfs performance."""
    logger.info("Optimizing database for httpfs...")

    con = duckdb.connect(str(input_db), read_only=True)

    row_count = con.execute("SELECT COUNT(*) FROM geo_embeddings").fetchone()[0]
    logger.info(f"  Source database has {row_count:,} rows")

    columns = con.execute("DESCRIBE geo_embeddings").fetchall()
    column_names = [c[0] for c in columns]
    has_geometry = "geometry" in column_names

    con.close()

    # Create optimized database
    out_con = duckdb.connect(str(output_db))

    if has_geometry:
        out_con.execute("INSTALL spatial; LOAD spatial;")

    logger.info("  Attaching source database...")
    out_con.execute(f"ATTACH '{input_db}' AS source (READ_ONLY)")

    logger.info("  Creating sorted table (this may take a while)...")
    out_con.execute(
        """
        CREATE TABLE geo_embeddings AS
        SELECT * FROM source.geo_embeddings
        ORDER BY id
    """
    )

    out_con.execute("DETACH source")

    if has_geometry:
        logger.info("  Creating spatial R-Tree index...")
        out_con.execute(
            "CREATE INDEX geom_spatial_idx ON geo_embeddings USING RTREE (geometry);"
        )

    logger.info("  Creating ID index...")
    out_con.execute("CREATE INDEX id_idx ON geo_embeddings(id);")

    final_count = out_con.execute("SELECT COUNT(*) FROM geo_embeddings").fetchone()[0]
    out_con.close()

    size_gb = output_db.stat().st_size / 1e9
    logger.info(f"✓ Optimized database: {final_count:,} rows, {size_gb:.2f} GB")


def _parse_fixed_array_dim(type_str: str) -> int | None:
    match = re.match(r"^[A-Z0-9_]+\[(\d+)\]$", str(type_str).strip().upper())
    if not match:
        return None
    return int(match.group(1))


def validate_httpfs_layout(db_path: Path) -> None:
    """Validate critical httpfs performance guardrails before publish."""
    logger.info("Validating database layout for httpfs...")
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        embedding_type_row = con.execute(
            "SELECT typeof(embedding) FROM geo_embeddings LIMIT 1"
        ).fetchone()
        if not embedding_type_row or embedding_type_row[0] is None:
            raise RuntimeError("Could not infer embedding column type from geo_embeddings")

        embedding_type = str(embedding_type_row[0]).upper()
        embedding_dim = _parse_fixed_array_dim(embedding_type)
        if embedding_dim is None:
            raise RuntimeError(
                f"Embedding type must be fixed-size for httpfs performance, got '{embedding_type}'"
            )

        index_rows = con.execute(
            "SELECT index_name FROM duckdb_indexes() WHERE table_name='geo_embeddings'"
        ).fetchall()
        index_names = {str(row[0]) for row in index_rows if row and row[0]}
        if "id_idx" not in index_names:
            raise RuntimeError("Missing required id index 'id_idx' on geo_embeddings(id)")

        row_group_count = None
        try:
            storage_rows = con.execute("PRAGMA storage_info('geo_embeddings')").fetchall()
            row_groups = {int(row[0]) for row in storage_rows if row and row[0] is not None}
            row_group_count = len(row_groups)
            if row_group_count <= 0:
                raise RuntimeError("No row groups discovered in geo_embeddings storage layout")
        except Exception as exc:
            logger.warning(f"Could not inspect row-group layout: {exc}")

        logger.info(
            "✓ Layout validation passed: embedding=%s, dim=%d, id_idx=yes%s",
            embedding_type,
            embedding_dim,
            (
                f", row_groups={row_group_count}"
                if row_group_count is not None
                else ""
            ),
        )
    finally:
        con.close()


def generate_geometry_cache(db_path: Path, output_path: Path) -> None:
    """Extract id + geometry from database to compressed Parquet."""
    logger.info("Generating geometry cache...")

    con = duckdb.connect(str(db_path), read_only=True)
    con.execute("INSTALL spatial; LOAD spatial;")

    row_count = con.execute("SELECT COUNT(*) FROM geo_embeddings").fetchone()[0]

    con.execute(
        f"""
        COPY (SELECT id, geometry FROM geo_embeddings ORDER BY id)
        TO '{output_path}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """
    )

    con.close()

    size_mb = output_path.stat().st_size / 1e6
    logger.info(f"✓ Geometry cache: {row_count:,} geometries, {size_mb:.1f} MB")


def publish_database(
    db_path: Path,
    region: str,
    date_range: str,
    work_dir: Path,
    skip_existing: bool = True,
    dry_run: bool = False,
    optimize: bool = False,
) -> None:
    """Publish a database to Source Cooperative."""
    model_name = parse_model_name(db_path)

    # Build S3 prefix
    s3_prefix = f"{S3_BASE_PREFIX}/{region}/{date_range}/httpfs/{model_name}"

    logger.info(f"\n{'=' * 60}")
    logger.info(f"Publishing: {model_name}")
    logger.info(f"Region: {region}")
    logger.info(f"Date range: {date_range}")
    logger.info(f"S3 destination: s3://{S3_BUCKET}/{s3_prefix}/")
    logger.info(f"Optimize: {optimize}")
    logger.info(f"{'=' * 60}\n")

    if dry_run:
        logger.info("DRY RUN - no files will be uploaded")
        return

    s3 = get_s3_client()

    # Check if already published
    db_key = f"{s3_prefix}/metadata.db"
    if skip_existing and check_existing(s3, S3_BUCKET, db_key):
        logger.info("⏭ Already published - use --force to overwrite")
        return

    # Find FAISS index
    faiss_path = find_faiss_index(db_path, model_name)
    if not faiss_path:
        logger.error(f"❌ Could not find FAISS index for {model_name}")
        logger.info(f"   Expected in: {db_path.parent}")
        return

    logger.info(f"Found FAISS index: {faiss_path.name}")

    # Create work directory
    work_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Optimize database (if requested) or use original
    if optimize:
        final_db = work_dir / "metadata.db"
        optimize_database(db_path, final_db)
        cleanup_db = True
    else:
        final_db = db_path
        cleanup_db = False
        logger.info("Skipping optimization (database already optimized by faiss_db.py)")

    validate_httpfs_layout(final_db)

    # Step 2: Generate geometry cache
    geometry_cache = work_dir / "geometry_cache.parquet"
    generate_geometry_cache(final_db, geometry_cache)

    # Step 3: Upload all files
    logger.info("\nUploading to Source Cooperative...")

    upload_with_progress(s3, final_db, S3_BUCKET, f"{s3_prefix}/metadata.db")
    upload_with_progress(s3, faiss_path, S3_BUCKET, f"{s3_prefix}/faiss.index")
    upload_with_progress(
        s3, geometry_cache, S3_BUCKET, f"{s3_prefix}/geometry_cache.parquet"
    )

    # Cleanup
    if cleanup_db:
        final_db.unlink()
    geometry_cache.unlink()

    logger.info(f"\n✓ Successfully published {model_name}")
    logger.info(f"  s3://{S3_BUCKET}/{s3_prefix}/metadata.db")
    logger.info(f"  s3://{S3_BUCKET}/{s3_prefix}/faiss.index")
    logger.info(f"  s3://{S3_BUCKET}/{s3_prefix}/geometry_cache.parquet")


def main():
    parser = argparse.ArgumentParser(
        description="Publish a local database to Source Cooperative",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "db_path",
        type=Path,
        help="Path to local DuckDB database (output of faiss_db.py)",
    )
    parser.add_argument(
        "--region",
        type=str,
        help="Region name (e.g., 'alabama'). Auto-detected from filename if not provided.",
    )
    parser.add_argument(
        "--date-range",
        type=str,
        help="Date range (e.g., '2024-01-01-2025-01-01'). Auto-detected if not provided.",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("./publish_work"),
        help="Working directory for temporary files",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing files on S3",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be uploaded without actually uploading",
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Re-optimize database (sort by ID, create indexes). Not needed for databases created by faiss_db.py after 2026-01-18.",
    )
    args = parser.parse_args()

    if not args.db_path.exists():
        logger.error(f"Database not found: {args.db_path}")
        return 1

    model_name = parse_model_name(args.db_path)

    # Auto-detect region if not provided
    region = args.region
    if not region:
        region = parse_region_from_name(model_name)
        if region:
            logger.info(f"Auto-detected region: {region}")
        else:
            logger.error("Could not auto-detect region. Please provide --region")
            return 1

    # Auto-detect date range if not provided
    date_range = args.date_range
    if not date_range:
        date_range = parse_date_range_from_name(model_name)
        if date_range:
            logger.info(f"Auto-detected date range: {date_range}")
        else:
            logger.error(
                "Could not auto-detect date range. Please provide --date-range"
            )
            return 1

    # Check for AWS credentials
    if not args.dry_run:
        if not os.environ.get("AWS_ACCESS_KEY_ID"):
            logger.error("AWS_ACCESS_KEY_ID environment variable not set")
            logger.info("Set credentials from Source Cooperative console")
            return 1

    publish_database(
        db_path=args.db_path,
        region=region,
        date_range=date_range,
        work_dir=args.work_dir,
        skip_existing=not args.force,
        dry_run=args.dry_run,
        optimize=args.optimize,
    )

    return 0


if __name__ == "__main__":
    exit(main())
