#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 -d DATE_RANGE -s SRC_DIR -t DST_PREFIX [-p CONCURRENCY] [-e S3_ENDPOINT]"
  echo "  -d  Date range suffix filter; only files ending with _<DATE_RANGE>.parquet are copied (e.g. 2024-01-01_2025-01-01)"
  echo "  -s  Source GCS dir (e.g. gs://bucket/path)"
  echo "  -t  Destination S3 prefix (e.g. s3://bucket/prefix)"
  echo "  -p  Concurrency (default: 8)"
  echo "  -e  S3 endpoint URL (default: https://data.source.coop)"
  exit 1
}

DATE_RANGE=""
SRC_DIR=""
DST_PREFIX=""
CONCURRENCY=8
S3_ENDPOINT="https://data.source.coop"

while getopts ":d:s:t:p:e:h" opt; do
  case "$opt" in
    d) DATE_RANGE="$OPTARG" ;;
    s) SRC_DIR="$OPTARG" ;;
    t) DST_PREFIX="$OPTARG" ;;
    p) CONCURRENCY="$OPTARG" ;;
    e) S3_ENDPOINT="$OPTARG" ;;
    h|*) usage ;;
  esac
done

[[ -z "$DATE_RANGE" || -z "$SRC_DIR" || -z "$DST_PREFIX" ]] && usage

command -v gsutil >/dev/null 2>&1 || { echo "gsutil not found"; exit 2; }
command -v aws >/dev/null 2>&1 || { echo "aws CLI not found"; exit 2; }

export DATE_RANGE DST_PREFIX S3_ENDPOINT

gsutil ls "${SRC_DIR%/}/"*"_${DATE_RANGE}.parquet" \
| xargs -n1 -P"${CONCURRENCY}" -I{} bash -c '
  src="$1"
  base="${src##*/}"
  dest="${DST_PREFIX%/}/$base"
  echo "Copying $src -> $dest"
  gsutil cat "$src" | aws s3 cp - "$dest" --endpoint-url="$S3_ENDPOINT" --only-show-errors
' _ {}