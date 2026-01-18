#!/bin/bash
# Cleanup original files after S3 migration verification
#
# ONLY RUN THIS AFTER:
# 1. Running migrate_s3_structure.sh
# 2. Verifying new structure works
# 3. Testing access to new file locations
#
# Usage:
#   ./scripts/cleanup_s3_migration.sh

set -euo pipefail

# Base paths
BUCKET="us-west-2.opendata.source.coop"
BASE_PREFIX="geovibes/search/USA/alabama/2024-01-01-2025-01-01"
BASE_URL="s3://${BUCKET}/${BASE_PREFIX}"

# Models
declare -A MODELS=(
    ["dino_vit"]="alabama_dino_vit_small_patch16_224_2024_2025_32_16_10"
    ["earthgenome"]="alabama_earthgenome_softcon_2024_2025_32_16_10"
    ["google_satellite"]="alabama_google_satellite_embeddings_v1_2024_2025_25_0_10"
    ["quantized_dino_vit"]="alabama_quantized_dino_vit_small_patch16_224_2024_2025_32_16_10"
)

echo "=== S3 Migration Cleanup ==="
echo "This will DELETE original files from ${BASE_URL}"
echo ""
read -p "Are you sure you want to proceed? (yes/no): " confirm
if [ "$confirm" != "yes" ]; then
    echo "Aborted."
    exit 1
fi
echo ""

# Delete test file
echo "Deleting test_write.txt..."
aws s3 rm "${BASE_URL}/test_write.txt" --only-show-errors 2>/dev/null || echo "  (not found)"

# Delete unsorted metadata.db
echo "Deleting unsorted metadata.db..."
aws s3 rm "${BASE_URL}/alabama_google_satellite_embeddings_v1_2024_2025_25_0_10_metadata.db" --only-show-errors 2>/dev/null || echo "  (not found)"

# Delete original files for each model
for model_key in "${!MODELS[@]}"; do
    model_name="${MODELS[$model_key]}"
    echo ""
    echo "Cleaning up ${model_name}..."

    # Delete tar.gz (now in archives/)
    echo "  - Deleting ${model_name}.tar.gz"
    aws s3 rm "${BASE_URL}/${model_name}.tar.gz" --only-show-errors 2>/dev/null || echo "    (not found)"

    # Delete metadata_sorted.db (now in httpfs/)
    echo "  - Deleting ${model_name}_metadata_sorted.db"
    aws s3 rm "${BASE_URL}/${model_name}_metadata_sorted.db" --only-show-errors 2>/dev/null || echo "    (not found)"

    # Delete faiss index (now in httpfs/)
    echo "  - Deleting faiss index"
    aws s3 rm "${BASE_URL}/${model_name}_faiss_4096_64_8.index" --only-show-errors 2>/dev/null || echo "    (not found)"

    # Delete geometry_cache.parquet (now in httpfs/)
    echo "  - Deleting ${model_name}_geometry_cache.parquet"
    aws s3 rm "${BASE_URL}/${model_name}_geometry_cache.parquet" --only-show-errors 2>/dev/null || echo "    (not found)"
done

echo ""
echo "=== Cleanup Complete ==="
echo ""
echo "Final structure:"
aws s3 ls "${BASE_URL}/" --recursive | head -30
