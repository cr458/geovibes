import os
from argparse import ArgumentParser
from typing import List
import multiprocessing as mp
import logging
import time

import geopandas as gpd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import timm
from torch.ao.quantization import quantize_dynamic
from tenacity import retry, stop_after_attempt, wait_exponential, before_sleep_log, retry_if_exception_type

try:
    from .geodataset import GeotiffTileDataset
    from .transforms import RescaledImageNetTransform
    from .raster import find_band_files, create_stacked_raster, get_raster_info
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from geodataset import GeotiffTileDataset
    from transforms import RescaledImageNetTransform
    from raster import find_band_files, create_stacked_raster, get_raster_info

logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def worker_init_fn(worker_id):
    """Initialize worker with proper thread settings."""
    cpu_count = mp.cpu_count()
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is not None:
        num_workers = worker_info.num_workers
        threads_per_worker = max(1, cpu_count // num_workers)
    else:
        threads_per_worker = 1
    
    torch.set_num_threads(threads_per_worker)
    
    os.environ['OMP_NUM_THREADS'] = str(threads_per_worker)
    os.environ['MKL_NUM_THREADS'] = str(threads_per_worker)


@retry(
    stop=stop_after_attempt(20),
    wait=wait_exponential(multiplier=2, min=2, max=300),
    retry=retry_if_exception_type((ConnectionError, TimeoutError, OSError, RuntimeError, Exception)),
    before_sleep=before_sleep_log(logger, logging.WARNING)
)
def load_model_with_retry(model_name: str):
    """Load a timm model with exponential backoff retry logic."""
    logger.info(f"Loading {model_name} model...")
    try:
        model = timm.create_model(model_name, pretrained=True, num_classes=0)
        logger.info("Model loaded successfully.")
        return model
    except Exception as e:
        logger.error(f"Failed to load model {model_name}: {e}")
        raise


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=1, max=30),
    retry=retry_if_exception_type((ConnectionError, TimeoutError, OSError, IOError)),
    before_sleep=before_sleep_log(logger, logging.WARNING)
)
def read_tiles_file_with_retry(tiles_file: str):
    """Read tiles file with retry logic for remote file access."""
    logger.info(f"Reading tiles from: {tiles_file}")
    try:
        return gpd.read_file(tiles_file)
    except Exception as e:
        logger.error(f"Failed to read tiles file {tiles_file}: {e}")
        raise


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=1, max=30),
    retry=retry_if_exception_type((ConnectionError, TimeoutError, OSError, IOError)),
    before_sleep=before_sleep_log(logger, logging.WARNING)
)
def save_results_with_retry(results_gdf, output_path: str):
    """Save results to parquet with retry logic for remote file writes."""
    logger.info(f"Saving embeddings to: {output_path}")
    try:
        results_gdf.to_parquet(output_path, index=False, compression='snappy')
        logger.info(f"Successfully saved to: {output_path}")
    except Exception as e:
        logger.error(f"Failed to save results to {output_path}: {e}")
        raise


def optimize_model(model: nn.Module, example_input: torch.Tensor, enable_compile: bool = True) -> nn.Module:
    """Optimize model for CPU inference with optional torch.compile."""
    model.eval()
    
    if enable_compile and hasattr(torch, 'compile'):
        try:
            print("Compiling model with torch.compile...")
            model = torch.compile(model, mode="reduce-overhead")
            with torch.inference_mode():
                _ = model(example_input)
            print("Model compilation successful")
            return model
        except Exception as e:
            print(f"torch.compile failed: {e}, falling back to eager mode")
    
    print("Using eager mode (no compilation)")
    return model


def main(
    mgrs_tile_id: str,
    bands: List[str],
    date_range: str,
    local_dir: str,
    tiles_file: str,
    output_path: str,
    model_name: str = "resnet18",
    batch_size: int = 64,
    num_workers: int = 12,
    target_resolution: int = 10,
    enable_quantization: bool = True,
    enable_compile: bool = True
) -> None:
    """
    Generate embeddings for Sentinel-2 imagery using deep learning models.
    
    This function processes Sentinel-2 satellite imagery to generate feature embeddings
    using pretrained computer vision models. It handles multi-band raster stacking,
    geometric tiling, data loading, and model inference with CPU optimizations.
    
    Args:
        mgrs_tile_id: MGRS tile identifier (e.g., '33TUN')
        bands: List of Sentinel-2 band identifiers to process
        date_range: Date range string for file identification
        local_dir: Local directory containing the band files
        tiles_file: Path to file containing tile geometries
        output_path: Path where embedding results will be saved
        model_name: Name of the pretrained model to use (timm format)
        batch_size: Batch size for inference
        num_workers: Number of data loading workers
        target_resolution: Target spatial resolution in meters
        enable_quantization: Whether to apply INT8 quantization
        enable_compile: Whether to use torch.compile optimization
    """
    cpu_count = mp.cpu_count()
    print(f"System has {cpu_count} CPU cores")
    
    main_threads = max(1, cpu_count - num_workers)
    torch.set_num_threads(main_threads)
    
    print(f"Processing MGRS tile: {mgrs_tile_id}")
    
    # Check if output already exists
    if os.path.exists(output_path):
        print(f"Output file already exists: {output_path}")
        print("Skipping processing. Delete the file to regenerate.")
        return
    
    print(f"Getting GeoTIFF files from: {local_dir}")
    band_paths = find_band_files(mgrs_tile_id, bands, date_range, local_dir)

    print(f"No bands found for MGRS tile {mgrs_tile_id} in {local_dir}")
    print("Attempting to extract tile boundary from tiles_file...")
    
    # Load tiles file to get the boundary
    print(f"Loading tiles from: {tiles_file}")

    tiles_gdf = read_tiles_file_with_retry(tiles_file)
    print(f"Successfully read file: {tiles_file}")

    # Get the overall boundary of all tiles
    overall_bounds = tiles_gdf.total_bounds  # [minx, miny, maxx, maxy]
    print(f"Tile boundary from tiles_file: {overall_bounds}")


    stacked_path = os.path.join(local_dir, f"{mgrs_tile_id}_stacked_{target_resolution}m.tif")
    print(f"Creating stacked GeoTIFF: {stacked_path} at {target_resolution}m resolution")
    create_stacked_raster(band_paths, stacked_path, target_resolution=target_resolution)
    
    
    print(f"Loaded {len(tiles_gdf)} polygons from shapefile")
    
    raster_info = get_raster_info(stacked_path)
    raster_crs = raster_info['crs']
    
    if tiles_gdf.crs != raster_crs:
        print(f"Converting tiles from {tiles_gdf.crs} to {raster_crs}")
        tiles_gdf = tiles_gdf.to_crs(raster_crs)
    
    transform_pipeline = RescaledImageNetTransform()
    
    print("Creating GeoDataFrameDataset...")
    dataset_start = time.time()
    logger.info(f"Creating dataset with {len(tiles_gdf)} geometries...")
    dataset = GeotiffTileDataset(
        path=stacked_path,
        geometries_gdf=tiles_gdf,
        transforms=transform_pipeline,
        target_resolution=target_resolution,
        use_memmap=True,
        cache_windows=False
    )
    dataset_elapsed = time.time() - dataset_start
    logger.info(f"Dataset creation completed in {dataset_elapsed:.2f}s")
    
    logger.info(f"Creating DataLoader with batch_size={batch_size}, num_workers={num_workers}...")
    dataloader_start = time.time()
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=False,  # No CUDA
        persistent_workers=True,
        prefetch_factor=6,
        worker_init_fn=worker_init_fn
    )
    dataloader_elapsed = time.time() - dataloader_start
    logger.info(f"DataLoader creation completed in {dataloader_elapsed:.2f}s")
    
    # Test dataset access
    logger.info("Testing dataset access with first item...")
    test_start = time.time()
    try:
        test_item = dataset[0]
        test_elapsed = time.time() - test_start
        logger.info(f"First dataset item accessed successfully in {test_elapsed:.3f}s, image shape: {test_item['image'].shape}")
    except Exception as e:
        logger.error(f"Failed to access first dataset item: {e}")
        raise
    
    model = load_model_with_retry(model_name)
    
    # Create example input for optimization
    example_input = torch.randn(1, 3, 224, 224)
    
    # Apply quantization if enabled
    if enable_quantization:
        print("Applying INT8 quantization...")
        model = quantize_dynamic(
            model, 
            {nn.Linear, nn.Conv2d},  # Quantize linear and conv layers
            dtype=torch.qint8
        )
    
    # Optimize model (compile or eager mode)
    model = optimize_model(model, example_input, enable_compile=enable_compile)
    model.eval()
    
    print("Generating embeddings...")
    embeddings = []
    tile_ids = []
    geometry_indices = []
    
    with torch.inference_mode():
        for batch in tqdm(dataloader, desc="Processing tiles"):
            images = batch['image']
            batch_tile_ids = batch['tile_id']
            batch_geom_indices = batch['geometry_index']
            
            # CPU inference
            batch_embeddings = model(images)
            
            # Convert and store results
            embeddings.append(batch_embeddings.numpy())
            
            # Handle both tensor and list cases for tile_ids
            if isinstance(batch_tile_ids, torch.Tensor):
                tile_ids.extend(batch_tile_ids.numpy().tolist())
            else:
                tile_ids.extend(batch_tile_ids)
                
            geometry_indices.extend(batch_geom_indices.numpy().tolist())

    
    embeddings = np.vstack(embeddings)
    
    print("Creating output GeoDataFrame...")
    
    # Vectorized result creation
    results_data = {
        'tile_id': [],
        'embedding': [],
        'geometry': [],
        'mgrs_tile_id': []
    }
    
    # Pre-allocate lists
    n_results = len(embeddings)
    results_data['tile_id'] = [None] * n_results
    results_data['embedding'] = [None] * n_results
    results_data['geometry'] = [None] * n_results
    results_data['mgrs_tile_id'] = [mgrs_tile_id] * n_results
    
    # Get additional columns
    extra_cols = [col for col in tiles_gdf.columns if col not in ['geometry', 'tile_id']]
    for col in extra_cols:
        results_data[col] = [None] * n_results
    
    # Fill results efficiently
    for i, (embedding, tile_id, geom_idx) in enumerate(zip(embeddings, tile_ids, geometry_indices)):
        original_row = tiles_gdf.iloc[geom_idx]
        results_data['tile_id'][i] = original_row.get('tile_id', tile_id)
        results_data['embedding'][i] = embedding.tolist()
        results_data['geometry'][i] = original_row.geometry
        
        for col in extra_cols:
            results_data[col][i] = original_row[col]
    
    results_gdf = gpd.GeoDataFrame(results_data, crs=tiles_gdf.crs)
    
    # Compute centroids in UTM if not already in UTM
    if results_gdf.crs and not results_gdf.crs.is_projected:
        # Convert to appropriate UTM for centroid calculation
        sample_centroid = results_gdf.geometry.iloc[0].centroid
        lon = sample_centroid.x
        utm_zone = int((lon + 180) / 6) + 1
        if utm_zone > 60:
            utm_zone = 60
        utm_crs = f"EPSG:326{utm_zone:02d}"
        
        results_utm = results_gdf.to_crs(utm_crs)
        results_utm['centroid_utm'] = results_utm.geometry.centroid
        results_utm['centroid_wgs84'] = results_utm['centroid_utm'].to_crs("EPSG:4326")
        if tiles_gdf.crs is not None:
            results_gdf = results_utm.to_crs(tiles_gdf.crs)
        else:
            results_gdf = results_utm
        results_gdf['centroid_utm'] = results_utm['centroid_utm']
        results_gdf['centroid_wgs84'] = results_utm['centroid_wgs84']
    else:
        # Already in projected CRS, just compute centroids
        results_gdf['centroid_utm'] = results_gdf.geometry.centroid
        results_gdf['centroid_wgs84'] = results_gdf['centroid_utm'].to_crs("EPSG:4326")
    
    # Set centroid_wgs84 as the geometry column and drop other geometry columns
    results_gdf = results_gdf.drop(columns=['geometry', 'centroid_utm'])
    results_gdf = results_gdf.set_geometry('centroid_wgs84')
    results_gdf = results_gdf.rename_geometry('geometry')
    
    save_results_with_retry(results_gdf, output_path)
    
    print(f"Generated {len(embeddings)} embeddings for {mgrs_tile_id}")
    print(f"Output saved to: {output_path}")
    print(f"Keeping temporary files in: {local_dir}")
    
    # Verify output matches input
    if len(results_gdf) != len(tiles_gdf):
        print(f"WARNING: Output has {len(results_gdf)} rows but input had {len(tiles_gdf)} rows")
    else:
        print(f"✓ Output has same number of rows as input: {len(results_gdf)}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('mgrs_tile_id', type=str, help='MGRS tile ID (e.g., 16RCA)')
    parser.add_argument('--bands', type=str, nargs='+', 
                       default=['B4', 'B3', 'B2'], help='Sentinel-2 bands to process')
    parser.add_argument('--date_range', type=str, 
                       default='2024-01-01_2025-01-01', help='Date range string')
    parser.add_argument('--local_dir', type=str, required=True, 
                       help='Local directory for temporary files')
    parser.add_argument('--tiles_file', type=str, 
                       default='', 
                       help='Path to tiles geojson/geoparquet/shapefile with polygons (optional)')
    parser.add_argument('--output_path', type=str, required=True, 
                       help='Output parquet file path')
    parser.add_argument('--model_name', type=str, default='resnet18', 
                       help='Model name for timm')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=12, help='Number of workers')
    parser.add_argument('--target_resolution', type=int, default=10, 
                       help='Target resolution for processing')
    parser.add_argument('--no_quantization', action='store_true', 
                       help='Disable INT8 quantization')
    parser.add_argument('--no_compile', action='store_true',
                       help='Disable torch.compile optimization')
    
    args = parser.parse_args()
    
    main(
        mgrs_tile_id=args.mgrs_tile_id,
        bands=args.bands,
        date_range=args.date_range,
        local_dir=args.local_dir,
        tiles_file=args.tiles_file,
        output_path=args.output_path,
        model_name=args.model_name,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        target_resolution=args.target_resolution,
        enable_quantization=not args.no_quantization,
        enable_compile=not args.no_compile
    )