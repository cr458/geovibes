#!/usr/bin/env python3
"""Materialize a DuckDB table from remote parquet and optionally upload to S3."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import boto3
import duckdb


def parse_s3_url(url: str) -> tuple[str, str]:
    if not url.startswith("s3://"):
        raise ValueError(f"Expected s3:// URL, got: {url}")
    rest = url[len("s3://") :]
    parts = rest.split("/", 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid S3 URL: {url}")
    return parts[0], parts[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize DuckDB variant from parquet")
    parser.add_argument("--parquet-url", required=True)
    parser.add_argument("--local-db-path", required=True)
    parser.add_argument("--upload-s3-url", default="")
    parser.add_argument("--table-name", default="geo_embeddings")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=384,
        help="Expected fixed embedding dimension for FLOAT[N] storage.",
    )
    args = parser.parse_args()

    parquet_url = str(args.parquet_url)
    local_db_path = Path(args.local_db_path).expanduser().resolve()
    local_db_path.parent.mkdir(parents=True, exist_ok=True)
    table_name = str(args.table_name)
    embedding_dim = max(1, int(args.embedding_dim))

    if local_db_path.exists():
        local_db_path.unlink()

    conn = duckdb.connect(str(local_db_path))
    conn.execute("SET enable_progress_bar=false")
    conn.execute("SET enable_profiling='no_output'")
    conn.execute("PRAGMA disable_profiling")
    conn.execute(f"SET threads={int(args.threads)}")
    conn.execute("INSTALL httpfs; LOAD httpfs;")
    conn.execute("INSTALL aws; LOAD aws;")
    conn.execute("CALL load_aws_credentials();")

    print(f"Materializing {table_name} from parquet...")
    conn.execute(
        f"""
        CREATE TABLE {table_name} (
            id BIGINT PRIMARY KEY,
            embedding FLOAT[{embedding_dim}]
        )
        """
    )
    conn.execute(
        f"""
        INSERT INTO {table_name}
        SELECT CAST(id AS BIGINT) AS id, CAST(embedding AS FLOAT[{embedding_dim}]) AS embedding
        FROM read_parquet('{parquet_url}')
        """
    )
    conn.execute(f"CREATE INDEX IF NOT EXISTS id_idx ON {table_name}(id)")
    emb_type = conn.execute(
        f"SELECT data_type FROM pragma_table_info('{table_name}') WHERE name='embedding'"
    ).fetchone()
    count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
    conn.close()

    size_gb = local_db_path.stat().st_size / (1024**3)
    emb_type_str = emb_type[0] if emb_type else "unknown"
    print(
        f"Local DB ready: {local_db_path} "
        f"({size_gb:.2f} GiB, rows={int(count):,}, embedding_type={emb_type_str})"
    )

    if args.upload_s3_url:
        bucket, key = parse_s3_url(args.upload_s3_url)
        print(f"Uploading to s3://{bucket}/{key} ...")
        s3 = boto3.client("s3")
        s3.upload_file(str(local_db_path), bucket, key)
        print(f"Uploaded: {args.upload_s3_url}")

        head = s3.head_object(Bucket=bucket, Key=key)
        remote_size = int(head.get("ContentLength", 0)) / (1024**3)
        print(f"Remote size: {remote_size:.2f} GiB")


if __name__ == "__main__":
    main()
