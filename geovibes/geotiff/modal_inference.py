#!/usr/bin/env python3

import argparse
import json
import logging
import os
import glob
import zipfile
import tempfile
from typing import Optional, List
from pathlib import Path

import modal
from modal import Secret
import geopandas as gpd

MGRS_ID_COLUMN = 'mgrs_id'

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = modal.App("geotiff-embedding")

image = (
    modal.Image.debian_slim()
    .apt_install([
        "g++",
        "gdal-bin",
        "libgdal-dev",
        "python3-gdal",
        "libspatialindex-dev",
        "libgeos-dev",
        "libproj-dev",
        "proj-data",
        "proj-bin",
        "libgeotiff-dev"
    ])
    .env({"GDAL_DATA": "/usr/share/gdal"})
    .pip_install([
        "torch>=2.0.0",
        "torchvision>=0.15.0", 
        "timm>=0.9.0",
        "pillow>=9.0.0",
        "pandas>=1.5.0",
        "pyarrow>=10.0.0",
        "geopandas>=0.14.0",
        "rasterio>=1.3.0",
        "shapely>=2.0.0",
        "tqdm>=4.64.0",
        "numpy>=1.24.0",
        "joblib>=1.2.0",
        "kornia>=0.7.0",
        "pyogrio>=0.7.0",
        "tenacity>=8.0.0",
    ])
    .add_local_dir("src/geotiff", "/root/geotiff")
)


def get_mgrs_tiles_from_roi(roi_file: str, mgrs_tiles_file: str = "geometries/mgrs_tiles.parquet") -> List[str]:
    """Intersect ROI with MGRS tiles to get list of MGRS tile IDs to process."""
    logger.info(f"Loading ROI from: {roi_file}")
    roi_gdf = gpd.read_file(roi_file)
    
    logger.info(f"Loading MGRS tiles from: {mgrs_tiles_file}")
    mgrs_gdf = gpd.read_parquet(mgrs_tiles_file)
    
    # Ensure both are in the same CRS
    if roi_gdf.crs is not None and mgrs_gdf.crs is not None and roi_gdf.crs != mgrs_gdf.crs:
        logger.info(f"Converting ROI from {roi_gdf.crs} to {mgrs_gdf.crs}")
        roi_gdf = roi_gdf.to_crs(mgrs_gdf.crs)
    elif roi_gdf.crs is None:
        logger.warning("ROI file has no CRS defined")
    elif mgrs_gdf.crs is None:
        logger.warning("MGRS tiles file has no CRS defined")
    
    # Get union of all ROI geometries
    roi_union = roi_gdf.unary_union

    intersecting = mgrs_gdf[mgrs_gdf.intersects(roi_union)]
    
    mgrs_ids = intersecting[MGRS_ID_COLUMN].tolist()
    logger.info(f"Found {len(mgrs_ids)} MGRS tiles intersecting with ROI: {mgrs_ids}")
    
    return mgrs_ids


def find_tileset_for_mgrs(mgrs_id: str, tiles_dir: str) -> Optional[str]:
    """Find the tileset file (zip/geoparquet/geojson) for a given MGRS ID."""
    logger.info(f"Searching for tileset for MGRS {mgrs_id} in directory: {tiles_dir}")
    
    # First, let's see what files are actually in the directory
    if os.path.exists(tiles_dir):
        all_files = os.listdir(tiles_dir)
        logger.info(f"Files in {tiles_dir}: {all_files[:10]}...")  # Show first 10 files
    else:
        logger.error(f"Directory does not exist: {tiles_dir}")
        return None
    
    # Search for files containing the MGRS ID
    patterns = [
        f"*{mgrs_id}*.zip",
        f"*{mgrs_id}*.geoparquet", 
        f"*{mgrs_id}*.geojson",
        f"*{mgrs_id}*.gpkg",
        f"*{mgrs_id}*.shp"
    ]
    
    for pattern in patterns:
        search_pattern = os.path.join(tiles_dir, pattern)
        logger.info(f"Searching with pattern: {search_pattern}")
        files = glob.glob(search_pattern)
        if files:
            logger.info(f"Found match: {files[0]}")
            return files[0]  # Return first match
    
    logger.warning(f"No tileset found for MGRS {mgrs_id} using any pattern")
    return None


@app.function(
    image=image,
    cpu=16,
    secrets=[Secret.from_name("gcs-aws-hmac-credentials")],
    timeout=86400,
    memory=16384,
    region='us',
    volumes={
        "/gcs-mount": modal.CloudBucketMount(
            bucket_name="geovibes",
            bucket_endpoint_url="https://storage.googleapis.com",
            secret=Secret.from_name("gcs-aws-hmac-credentials"),
        ),
        "/modal_vol": modal.Volume.from_name("geovibes-modal")
    }
)
def process_mgrs_geotiff_embeddings(
    mgrs_tile_id: str,
    tiles_dir: str,
    output_base_path: str,
    local_dir: str = "imagery/s2",
    date_range: str = "2024-01-01_2025-01-01",
    bands_json: Optional[str] = None,
    model_name: str = "resnet18",
    batch_size: int = 64,
    num_workers: int = 12,
    target_resolution: int = 10,
    enable_quantization: bool = True,
    enable_compile: bool = False
):
    """Process a single MGRS tile using the main function from generate_geotiff_embeddings."""
    
    import sys
    import geopandas as gpd
    import json
    
    bands: Optional[List[str]] = json.loads(bands_json) if bands_json is not None else None

    # Add local directories to path
    sys.path.insert(0, '/root')
    
    from geotiff.embeddings import main
    import torch
    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    torch.multiprocessing.set_sharing_strategy("file_system")
        
    # Debug: List files to ensure mount worked
    logger.info("Checking mounted files:")
    if os.path.exists('/root/geotiff'):
        files = os.listdir('/root/geotiff')
        logger.info(f"Files in /root/geotiff: {files}")
    else:
        logger.error("Mount directory /root/geotiff not found!")
    
    logger.info(f"Using bands: {bands}")
    
    logger.info(f"Starting processing for MGRS tile: {mgrs_tile_id}")
    logger.info(f"Tiles directory: {tiles_dir}")
    logger.info(f"Local directory: {local_dir}")
    logger.info(f"Output base path: {output_base_path}")
    logger.info(f"Date range: {date_range}")
    logger.info(f"Bands: {bands}")
    logger.info(f"Model: {model_name}")
    
    # Find tileset file for this MGRS ID
    tiles_gcs_dir = os.path.join("/gcs-mount", tiles_dir)
    tileset_file = find_tileset_for_mgrs(mgrs_tile_id, tiles_gcs_dir)
    
    if not tileset_file:
        raise FileNotFoundError(f"No tileset file found for MGRS {mgrs_tile_id} in {tiles_gcs_dir}")
    
    logger.info(f"Found tileset: {tileset_file}")
    
    # Set up working directory on Modal volume
    work_dir = f"/modal_vol/{mgrs_tile_id}"
    os.makedirs(work_dir, exist_ok=True)
    logger.info(f"Using Modal volume working directory: {work_dir}")
    
    # Extract/copy tileset to local directory
    logger.info("Extracting tileset...")
    shapefile_local_path = None
    
    if tileset_file.endswith('.zip'):
        # Handle zipped shapefile
        zip_local_path = f"{work_dir}/shapefile.zip"
        os.system(f"cp '{tileset_file}' '{zip_local_path}'")
        
        # Extract zip
        with zipfile.ZipFile(zip_local_path, 'r') as zip_ref:
            zip_ref.extractall(f"{work_dir}/shapefile")
        
        # Find the .shp file
        shp_files = list(Path(f"{work_dir}/shapefile").glob("*.shp"))
        if shp_files:
            shapefile_local_path = str(shp_files[0])
        else:
            raise FileNotFoundError("No .shp file found in the zip archive")
    
    elif tileset_file.endswith('.shp'):
        # Copy all shapefile components
        shapefile_dir = f"{work_dir}/shapefile"
        os.makedirs(shapefile_dir, exist_ok=True)
        
        base_name = os.path.splitext(os.path.basename(tileset_file))[0]
        source_dir = os.path.dirname(tileset_file)
        
        extensions = ['.shp', '.shx', '.dbf', '.prj', '.cpg']
        for ext in extensions:
            src_file = f"{source_dir}/{base_name}{ext}"
            if os.path.exists(src_file):
                dst_file = f"{shapefile_dir}/{base_name}{ext}"
                os.system(f"cp '{src_file}' '{dst_file}'")
        
        shapefile_local_path = f"{shapefile_dir}/{base_name}.shp"
    
    elif tileset_file.endswith(('.geoparquet', '.geojson', '.gpkg')):
        # Copy directly
        shapefile_local_path = f"{work_dir}/tileset{Path(tileset_file).suffix}"
        os.system(f"cp '{tileset_file}' '{shapefile_local_path}'")
    
    else:
        raise ValueError(f"Unsupported tileset format: {tileset_file}")
    
    logger.info(f"Using tileset: {shapefile_local_path}")
    
    # Set up GCS mounted directory for imagery
    gcs_imagery_dir = os.path.join("/gcs-mount", local_dir)
    logger.info(f"Using GCS mounted directory for imagery: {gcs_imagery_dir}")
    
    # Copy band files from GCS to Modal volume for processing
    logger.info("Copying band files to Modal volume...")
    if bands is None:
        bands = ["B4", "B3", "B2"]  # Default bands
    band_file_patterns = [f"{mgrs_tile_id}_{band}_{date_range}.tif" for band in bands]
    copied_files = []
    
    for pattern in band_file_patterns:
        gcs_band_file = os.path.join(gcs_imagery_dir, pattern)
        modal_band_file = os.path.join(work_dir, pattern)
        
        if os.path.exists(gcs_band_file):
            os.system(f"cp '{gcs_band_file}' '{modal_band_file}'")
            copied_files.append(modal_band_file)
            logger.info(f"Copied {pattern} to Modal volume")
        else:
            logger.warning(f"Band file not found: {gcs_band_file}")
    
    if not copied_files:
        logger.error(f"No band files found for {mgrs_tile_id} in {gcs_imagery_dir}")
        if os.path.exists(gcs_imagery_dir):
            available_files = [f for f in os.listdir(gcs_imagery_dir) if mgrs_tile_id in f][:10]
            logger.info(f"Available files with {mgrs_tile_id}: {available_files}")
        raise FileNotFoundError(f"No band files found for {mgrs_tile_id}")
    
    # Set up output paths - write directly to GCS mount
    if output_base_path.startswith('gs://'):
        # Remove gs://bucketname/ prefix and construct GCS mount path
        # gs://geovibes/embeddings/resnet18 -> /gcs-mount/embeddings/resnet18
        path_without_bucket = output_base_path.replace('gs://geovibes/', '')
        gcs_output_dir = os.path.join("/gcs-mount", path_without_bucket)
        final_output_path = os.path.join(gcs_output_dir, f"{mgrs_tile_id}_embeddings.parquet")
        output_gcs_path = f"{output_base_path.rstrip('/')}/{mgrs_tile_id}_embeddings.parquet"
    else:
        # Local path in bucket
        gcs_output_dir = os.path.join("/gcs-mount", output_base_path)
        final_output_path = os.path.join(gcs_output_dir, f"{mgrs_tile_id}_embeddings.parquet")
        output_gcs_path = f"gs://geovibes/{output_base_path.rstrip('/')}/{mgrs_tile_id}_embeddings.parquet"
    
    # Create output directory if needed
    os.makedirs(gcs_output_dir, exist_ok=True)
    logger.info(f"Writing output directly to GCS: {final_output_path}")
    
    # Call the main function from generate_geotiff_embeddings
    # Now all processing happens on the Modal volume, output goes directly to GCS
    try:
        main(
            mgrs_tile_id=mgrs_tile_id,
            bands=bands,
            date_range=date_range,
            local_dir=work_dir,  # Use Modal volume work directory
            tiles_file=shapefile_local_path,
            output_path=final_output_path,  # Write directly to GCS mount
            model_name=model_name,
            batch_size=batch_size,
            num_workers=num_workers,
            enable_quantization=enable_quantization,
            enable_compile=enable_compile
        )
        
        logger.info(f"Output saved to: {output_gcs_path}")
        
        # Get result info
        result_gdf = gpd.read_parquet(final_output_path)
        processed_patches = len(result_gdf)
        feature_dimension = len(result_gdf['embedding'].iloc[0]) if len(result_gdf) > 0 else 0
        
        # Cleanup local files
        os.system(f"rm -rf {work_dir}")
        
        logger.info("Processing complete!")
        return {
            "mgrs_tile_id": mgrs_tile_id,
            "processed_patches": processed_patches,
            "output_path": output_gcs_path,
            "model_used": model_name,
            "feature_dimension": feature_dimension
        }
        
    except Exception as e:
        logger.error(f"Error processing {mgrs_tile_id}: {str(e)}")
        # Cleanup on error
        os.system(f"rm -rf {work_dir}")
        raise


@app.local_entrypoint()
def main(
    config: str,
    roi_file: Optional[str] = None,
    tiles_dir: Optional[str] = None,
    output_base_path: Optional[str] = None,
    local_dir: Optional[str] = None,
    date_range: str = "2024-01-01_2025-01-01",
    model_name: str = "resnet18",
    batch_size: int = 64,
    num_workers: int = 12,
    target_resolution: int = 10,
    enable_quantization: bool = True,
    enable_compile: bool = False
):
    """Launch Modal inference jobs for MGRS tiles using config file."""
    
    # Load config file
    logger.info(f"Loading config from: {config}")
    with open(config, 'r') as f:
        config_params = json.load(f)
    
    # Get MGRS IDs either from ROI intersection or config
    roi_file = config_params.get("roi_file", roi_file)
    if roi_file:
        logger.info(f"Determining MGRS tiles from ROI file: {roi_file}")
        mgrs_tiles_file = config_params.get("mgrs_tiles_file", "geometries/mgrs_tiles.parquet")
        mgrs_ids = get_mgrs_tiles_from_roi(roi_file, mgrs_tiles_file)
    else:
        # Fall back to explicit mgrs_ids from config
        mgrs_ids = config_params.get("mgrs_ids")
        if not mgrs_ids:
            raise ValueError("Config file must contain either 'roi_file' or 'mgrs_ids' field")
    
    # Override parameters with config values if provided
    tiles_dir = config_params.get("tiles_dir", tiles_dir)
    output_base_path = config_params.get("output_base_path", output_base_path)
    local_dir = config_params.get("local_dir", local_dir or "imagery/s2")
    date_range = config_params.get("date_range", date_range)
    bands = config_params.get("bands")
    model_name = config_params.get("model_name", model_name)
    batch_size = config_params.get("batch_size", batch_size)
    num_workers = config_params.get("num_workers", num_workers)
    target_resolution = config_params.get("target_resolution", target_resolution)
    enable_quantization = config_params.get("enable_quantization", enable_quantization)
    enable_compile = config_params.get("enable_compile", enable_compile)
    
    # Validate required parameters
    if not tiles_dir:
        raise ValueError("tiles_dir must be provided either in config or as command line argument")
    if not output_base_path:
        raise ValueError("output_base_path must be provided either in config or as command line argument")
    
    logger.info(f"Launching {len(mgrs_ids)} Modal inference jobs...")
    logger.info(f"MGRS IDs: {mgrs_ids}")
    logger.info(f"Tiles directory: {tiles_dir}")
    logger.info(f"Local directory: {local_dir}")
    logger.info(f"Output base path: {output_base_path}")
    logger.info(f"Using bands: {bands}")
    
    # Launch jobs for each MGRS ID
    results = []
    bands_json = json.dumps(bands) if bands is not None else None
    for mgrs_id in mgrs_ids:
        job = process_mgrs_geotiff_embeddings.spawn(
            mgrs_tile_id=mgrs_id,
            tiles_dir=tiles_dir,
            output_base_path=output_base_path,
            local_dir=local_dir or "imagery/s2",
            date_range=date_range,
            bands_json=bands_json,
            model_name=model_name,
            batch_size=batch_size,
            num_workers=num_workers,
            target_resolution=target_resolution,
            enable_quantization=enable_quantization,
            enable_compile=enable_compile
        )
        results.append(job)
    
    logger.info(f"Spawned {len(results)} jobs. Jobs will run asynchronously.")
    
    # Wait for all to complete
    completed_results = []
    for i, job in enumerate(results):
        mgrs_id = mgrs_ids[i]
        logger.info(f"Waiting for job {mgrs_id}...")
        try:
            result = job.get()
            completed_results.append(result)
            logger.info(f"Job {mgrs_id} completed: {result}")
        except Exception as e:
            logger.error(f"Job {mgrs_id} failed: {e}")
            completed_results.append({"mgrs_tile_id": mgrs_id, "error": str(e)})
    
    logger.info("All jobs completed!")
    return completed_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process MGRS tiles for geotiff embeddings using Modal")
    
    parser.add_argument(
        "--config",
        required=True,
        help="Path to JSON config file containing roi_file/mgrs_ids and other parameters"
    )
    
    parser.add_argument(
        "--roi-file",
        help="Path to ROI geometry file (geojson/shapefile/geoparquet) to intersect with MGRS tiles (overrides config)"
    )
    
    parser.add_argument(
        "--tiles-dir",
        help="GCS directory containing tileset files (overrides config)"
    )
    
    parser.add_argument(
        "--output-base-path",
        help="Base GCS path for output files (overrides config)"
    )
    
    parser.add_argument(
        "--local-dir",
        help="GCS directory containing imagery files (overrides config)"
    )
    
    parser.add_argument(
        "--date-range",
        default="2024-01-01_2025-01-01",
        help="Date range for Sentinel-2 imagery (default: 2024-01-01_2025-01-01)"
    )
    
    parser.add_argument(
        "--model-name",
        default="resnet18",
        help="timm model name to use for embeddings (default: resnet18)"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for processing (default: 64)"
    )
    
    parser.add_argument(
        "--num-workers",
        type=int,
        default=12,
        help="Number of DataLoader workers (default: 12)"
    )
    
    parser.add_argument(
        "--target-resolution",
        type=int,
        default=10,
        help="Target resolution for processing (default: 10)"
    )

    parser.add_argument(
        "--no-quantization",
        action="store_true",
        help="Disable INT8 quantization"
    )
    
    parser.add_argument(
        "--no-compile",
        action="store_true",
        help="Disable torch.compile optimization"
    )
    
    args = parser.parse_args()
    
    params = {
        "config": args.config,
        "roi_file": args.roi_file,
        "tiles_dir": args.tiles_dir,
        "output_base_path": args.output_base_path,
        "local_dir": args.local_dir,
        "date_range": args.date_range,
        "model_name": args.model_name,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "target_resolution": args.target_resolution,
        "enable_quantization": not args.no_quantization,
        "enable_compile": not args.no_compile
    }
    
    # Filter out None values for cleaner output
    params = {k: v for k, v in params.items() if v is not None}
    
    print(f"Running embedding generation with parameters: {params}")
    print("Note: Run with 'modal run' for Modal execution")