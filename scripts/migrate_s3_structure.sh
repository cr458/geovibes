#!/bin/bash
# Migrate S3 bucket from flat structure to archives/ + httpfs/ structure
#
# Usage:
#   # Set AWS credentials first (from Source Cooperative console)
#   export AWS_ACCESS_KEY_ID="ASIA..."
#   export AWS_SECRET_ACCESS_KEY="..."
#   export AWS_SESSION_TOKEN="..."
#   export AWS_DEFAULT_REGION="us-west-2"
#
#   # Run migration
#   ./scripts/migrate_s3_structure.sh

set -euo pipefail

# Base paths
BUCKET="us-west-2.opendata.source.coop"
BASE_PREFIX="geovibes/search/USA/alabama/2024-01-01-2025-01-01"
BASE_URL="s3://${BUCKET}/${BASE_PREFIX}"

# Models to migrate (folder name will be derived from these)
declare -A MODELS=(
    ["dino_vit"]="alabama_dino_vit_small_patch16_224_2024_2025_32_16_10"
    ["earthgenome"]="alabama_earthgenome_softcon_2024_2025_32_16_10"
    ["google_satellite"]="alabama_google_satellite_embeddings_v1_2024_2025_25_0_10"
    ["quantized_dino_vit"]="alabama_quantized_dino_vit_small_patch16_224_2024_2025_32_16_10"
)

echo "=== S3 Structure Migration ==="
echo "Base URL: ${BASE_URL}"
echo ""

# Step 1: Create archives/ folder and move tar.gz files
echo "Step 1: Moving tar.gz files to archives/"
for model_key in "${!MODELS[@]}"; do
    model_name="${MODELS[$model_key]}"
    src="${BASE_URL}/${model_name}.tar.gz"
    dst="${BASE_URL}/archives/${model_name}.tar.gz"

    echo "  Copying ${model_name}.tar.gz -> archives/"
    aws s3 cp "$src" "$dst" --only-show-errors
done
echo "  Done."
echo ""

# Step 2: Create httpfs/ subfolders and copy files
echo "Step 2: Creating httpfs/ structure"
for model_key in "${!MODELS[@]}"; do
    model_name="${MODELS[$model_key]}"
    httpfs_folder="${BASE_URL}/httpfs/${model_name}"

    echo "  Creating ${model_name}/"

    # Copy metadata_sorted.db -> metadata.db
    echo "    - metadata.db"
    aws s3 cp "${BASE_URL}/${model_name}_metadata_sorted.db" "${httpfs_folder}/metadata.db" --only-show-errors

    # Copy faiss index -> faiss.index
    # Find the faiss file (has _faiss_ in name)
    faiss_src=$(aws s3 ls "${BASE_URL}/" | grep "${model_name}_faiss_" | awk '{print $4}' | head -1)
    if [ -n "$faiss_src" ]; then
        echo "    - faiss.index (from ${faiss_src})"
        aws s3 cp "${BASE_URL}/${faiss_src}" "${httpfs_folder}/faiss.index" --only-show-errors
    else
        echo "    - WARNING: No faiss index found for ${model_name}"
    fi

    # Copy geometry_cache.parquet
    echo "    - geometry_cache.parquet"
    aws s3 cp "${BASE_URL}/${model_name}_geometry_cache.parquet" "${httpfs_folder}/geometry_cache.parquet" --only-show-errors
done
echo "  Done."
echo ""

# Step 3: Verify new structure
echo "Step 3: Verifying new structure"
echo "  archives/:"
aws s3 ls "${BASE_URL}/archives/" | awk '{print "    " $4}'
echo ""
echo "  httpfs/:"
for model_key in "${!MODELS[@]}"; do
    model_name="${MODELS[$model_key]}"
    echo "    ${model_name}/:"
    aws s3 ls "${BASE_URL}/httpfs/${model_name}/" | awk '{print "      " $4}'
done
echo ""

# Step 4: Delete obsolete files (commented out for safety - uncomment after verification)
echo "Step 4: Cleanup (commented out - uncomment after verifying step 3)"
echo "  # Delete test_write.txt"
echo "  # aws s3 rm ${BASE_URL}/test_write.txt"
echo ""
echo "  # Delete unsorted metadata.db (if exists)"
echo "  # aws s3 rm ${BASE_URL}/alabama_google_satellite_embeddings_v1_2024_2025_25_0_10_metadata.db"
echo ""
echo "  # Delete original files after verification:"
for model_key in "${!MODELS[@]}"; do
    model_name="${MODELS[$model_key]}"
    echo "  # aws s3 rm ${BASE_URL}/${model_name}.tar.gz"
    echo "  # aws s3 rm ${BASE_URL}/${model_name}_metadata_sorted.db"
    echo "  # aws s3 rm ${BASE_URL}/${model_name}_faiss_*.index"
    echo "  # aws s3 rm ${BASE_URL}/${model_name}_geometry_cache.parquet"
done
echo ""

echo "=== Migration Complete ==="
echo ""
echo "Next steps:"
echo "1. Verify the new structure looks correct (step 3 output above)"
echo "2. Test accessing files from the new locations"
echo "3. Uncomment and run the cleanup commands in step 4"
echo "4. Run: ./scripts/cleanup_s3_migration.sh (after creating it)"
