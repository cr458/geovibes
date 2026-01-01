"""Upload DuckDB database to S3 for httpfs benchmarking.

This script uploads an uncompressed .db file to Source Cooperative S3
so it can be queried via DuckDB httpfs extension.

Usage:
    # With session credentials (from Source Cooperative console)
    python -m geovibes.database.upload_for_httpfs \
        local_databases/alabama_google_satellite_embeddings_v1_2024_2025_25_0_10_metadata.db \
        --session-token "YOUR_SESSION_TOKEN"

    # With AWS profile
    python -m geovibes.database.upload_for_httpfs \
        local_databases/alabama_google_satellite_embeddings_v1_2024_2025_25_0_10_metadata.db \
        --profile my-profile

Note: Static credentials (SCRPTLC1EMW2CQLBT59M15YC) have read-only access.
      Session tokens from Source Cooperative console are required for writes.
"""

import argparse
from pathlib import Path

import boto3
from botocore.config import Config
from tqdm import tqdm

ENDPOINT_URL = "https://data.source.coop"
DEFAULT_BUCKET = "geovibes"
DEFAULT_PREFIX = "search/USA/alabama/2024-01-01-2025-01-01/httpfs"


def get_s3_client(
    access_key: str = None,
    secret_key: str = None,
    session_token: str = None,
    profile: str = None,
):
    """Create S3 client for Source Cooperative."""
    if profile:
        session = boto3.Session(profile_name=profile)
        return session.client(
            "s3",
            endpoint_url=ENDPOINT_URL,
            config=Config(signature_version="s3v4"),
        )

    return boto3.client(
        "s3",
        endpoint_url=ENDPOINT_URL,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        aws_session_token=session_token,
        region_name="us-west-2",
        config=Config(signature_version="s3v4"),
    )


def upload_file(s3_client, local_path: Path, bucket: str, key: str):
    """Upload file with progress bar."""
    file_size = local_path.stat().st_size

    with tqdm(total=file_size, unit="B", unit_scale=True, desc="Uploading") as pbar:
        s3_client.upload_file(
            str(local_path),
            bucket,
            key,
            Callback=pbar.update,
        )

    print(f"✓ Uploaded to s3://{bucket}/{key}")
    return f"s3://{bucket}/{key}"


def main():
    parser = argparse.ArgumentParser(
        description="Upload DuckDB database to S3 for httpfs benchmarking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("db_path", type=Path, help="Path to local DuckDB database")
    parser.add_argument(
        "--bucket",
        default=DEFAULT_BUCKET,
        help=f"S3 bucket (default: {DEFAULT_BUCKET})",
    )
    parser.add_argument(
        "--prefix",
        default=DEFAULT_PREFIX,
        help=f"S3 key prefix (default: {DEFAULT_PREFIX})",
    )
    parser.add_argument("--access-key", help="AWS access key ID")
    parser.add_argument("--secret-key", help="AWS secret access key")
    parser.add_argument(
        "--session-token", help="AWS session token (required for writes)"
    )
    parser.add_argument("--profile", help="AWS profile name")
    args = parser.parse_args()

    if not args.db_path.exists():
        raise FileNotFoundError(f"Database not found: {args.db_path}")

    # Create S3 client
    if args.profile:
        s3 = get_s3_client(profile=args.profile)
    elif args.session_token:
        # Read static credentials and add session token
        s3 = get_s3_client(
            access_key=args.access_key or "ASIAWCQM3Z36DUTPFIRS",
            secret_key=args.secret_key,
            session_token=args.session_token,
        )
    else:
        print("Error: Write access requires --session-token or --profile")
        print("\nTo get a session token:")
        print("  1. Go to Source Cooperative console")
        print("  2. Navigate to your repository")
        print("  3. Get temporary credentials")
        print("  4. Pass --session-token with the token value")
        return 1

    # Construct S3 key
    key = f"{args.prefix}/{args.db_path.name}"

    print(f"Uploading {args.db_path} ({args.db_path.stat().st_size / 1e9:.2f} GB)")
    print(f"Destination: s3://{args.bucket}/{key}")

    s3_url = upload_file(s3, args.db_path, args.bucket, key)

    print("\nTo run benchmarks:")
    print("  uv run python -m geovibes.database.benchmark_httpfs \\")
    print(f"    --s3-url '{s3_url}' \\")
    print(f"    --local-path '{args.db_path}' \\")
    print("    --endpoint-url 'data.source.coop'")


if __name__ == "__main__":
    main()
