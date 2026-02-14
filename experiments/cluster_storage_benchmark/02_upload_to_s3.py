#!/usr/bin/env python3
"""
Step 2: Upload reordered databases to S3.

Uploads the three database variants to S3 for benchmark testing.

Usage:
    uv run python 02_upload_to_s3.py
"""

import yaml
import boto3
from pathlib import Path
from tqdm import tqdm

# Load config
with open(Path(__file__).parent / "config.yaml") as f:
    config = yaml.safe_load(f)

LOCAL_DIR = Path(config["local_dir"])
S3_BUCKET = config["s3_destination"]["bucket"]
S3_PREFIX = config["s3_destination"]["prefix"]


def upload_with_progress(s3_client, local_path, bucket, key):
    """Upload file to S3 with progress bar."""
    file_size = local_path.stat().st_size

    with tqdm(total=file_size, unit="B", unit_scale=True, desc=local_path.name) as pbar:

        def callback(bytes_transferred):
            pbar.update(bytes_transferred)

        s3_client.upload_file(str(local_path), bucket, key, Callback=callback)


def main():
    print("=" * 70)
    print("Step 2: Upload Reordered Databases to S3")
    print("=" * 70)

    # Initialize S3 client
    s3 = boto3.client("s3")

    # Files to upload
    variants = ["baseline", "hilbert", "two_level"]
    extensions = ["_metadata.db", ".parquet"]

    uploaded = []

    for variant in variants:
        for ext in extensions:
            local_path = LOCAL_DIR / f"{variant}{ext}"

            if not local_path.exists():
                print(f"  Skipping {local_path.name} (not found)")
                continue

            s3_key = f"{S3_PREFIX}/{variant}{ext}"
            s3_url = f"s3://{S3_BUCKET}/{s3_key}"

            print(f"\nUploading {local_path.name}...")
            upload_with_progress(s3, local_path, S3_BUCKET, s3_key)

            uploaded.append((variant, ext, s3_url))

    # Also upload the FAISS index (shared across all variants)
    faiss_cache_dir = Path.home() / ".cache" / "geovibes" / "faiss"
    faiss_files = list(faiss_cache_dir.glob("*.index"))

    if faiss_files:
        # Use the most recent one
        faiss_path = max(faiss_files, key=lambda p: p.stat().st_mtime)
        s3_key = f"{S3_PREFIX}/faiss.index"

        print("\nUploading FAISS index...")
        upload_with_progress(s3, faiss_path, S3_BUCKET, s3_key)
        uploaded.append(("shared", ".index", f"s3://{S3_BUCKET}/{s3_key}"))

    print("\n" + "=" * 70)
    print("Upload complete!")
    print("=" * 70)
    print("\nUploaded files:")
    for variant, ext, url in uploaded:
        print(f"  {variant}{ext}: {url}")

    # Write URLs to config for benchmark script
    urls_file = Path(__file__).parent / "s3_urls.yaml"
    urls = {
        "faiss_url": f"s3://{S3_BUCKET}/{S3_PREFIX}/faiss.index",
        "variants": {
            variant: f"s3://{S3_BUCKET}/{S3_PREFIX}/{variant}_metadata.db"
            for variant in variants
        },
    }

    with open(urls_file, "w") as f:
        yaml.dump(urls, f)
    print(f"\nS3 URLs written to {urls_file}")


if __name__ == "__main__":
    main()
