"""
Raster handling utilities for GeoTIFF processing.

This module provides functionality for working with multi-resolution satellite imagery,
particularly Sentinel-2 data, including band discovery, stacking, and resampling.
"""

import os
from typing import Any, List, Dict

import rasterio
import rasterio.enums
import rasterio.transform


class MultiResolutionBandStacker:
    """
    Handle Sentinel-2 bands with different resolutions.
    
    This class provides utilities for working with Sentinel-2 imagery where different
    spectral bands are provided at different spatial resolutions (10m, 20m, 60m).
    """
    
    BAND_RESOLUTIONS = {
        'B1': 60, 'B2': 10, 'B3': 10, 'B4': 10,
        'B5': 20, 'B6': 20, 'B7': 20, 'B8': 10,
        'B8A': 20, 'B9': 60, 'B10': 60, 'B11': 20, 'B12': 20
    }
    
    def __init__(self, target_resolution: int = 10):
        """
        Initialize the band stacker.
        
        Args:
            target_resolution: Target spatial resolution in meters for output raster.
        """
        self.target_resolution = target_resolution


def find_band_files(mgrs_tile_id: str, bands: List[str], date_range: str, 
                   local_dir: str) -> List[str]:
    """
    Locate GeoTIFF files for specified bands in a local directory.
    
    This function searches for band files using multiple naming patterns to handle
    variations in file naming conventions across different data sources.
    
    Args:
        mgrs_tile_id: MGRS tile identifier (e.g., '33TUN')
        bands: List of band identifiers (e.g., ['B4', 'B3', 'B2'])
        date_range: Date range string used in filename (e.g., '2024-01-01_2025-01-01')
        local_dir: Local directory to search for files
        
    Returns:
        List of file paths for found bands
        
    Note:
        Files are searched in both the main directory and 'imagery/s2' subdirectory.
        Multiple naming patterns are tried to maximize compatibility.
    """
    file_paths = []
    
    for band in bands:
        patterns = [
            f"{mgrs_tile_id}_{band}_{date_range}.tif",
            f"{mgrs_tile_id}_{band.upper()}_{date_range}.tif",
            f"{mgrs_tile_id}_{band.lower()}_{date_range}.tif",
            f"{mgrs_tile_id}_B{band[-1]}_{date_range}.tif" if band.startswith('B') else None,
            f"{mgrs_tile_id}_B{band[-2:]}_{date_range}.tif" if band.startswith('B') and len(band) > 2 else None,
        ]
        
        found = False
        for pattern in patterns:
            if pattern is None:
                continue
                
            file_path = os.path.join(local_dir, pattern)
            if os.path.exists(file_path):
                print(f"Found file: {file_path}")
                file_paths.append(file_path)
                found = True
                break
                
            alt_path = os.path.join(local_dir, "imagery", "s2", pattern)
            if os.path.exists(alt_path):
                print(f"Found file: {alt_path}")
                file_paths.append(alt_path)
                found = True
                break
        
        if not found:
            print(f"Warning: No file found for band {band} in {local_dir}")
            print(f"  Tried patterns: {[p for p in patterns if p is not None]}")
    
    return file_paths


def create_stacked_raster(band_paths: List[str], output_path: str, 
                         target_resolution: int = 10) -> str:
    """
    Create a multi-band stacked GeoTIFF from individual band files.
    
    This function combines multiple single-band GeoTIFF files into a single
    multi-band raster, optionally resampling to a target resolution.
    
    Args:
        band_paths: List of paths to individual band GeoTIFF files
        output_path: Path where the stacked raster will be saved
        target_resolution: Target spatial resolution in meters
        
    Returns:
        Path to the created stacked raster file
        
    Note:
        If the output file already exists, creation is skipped and the existing
        path is returned. Resampling uses bilinear interpolation when needed.
    """
    if os.path.exists(output_path):
        print(f"Stacked GeoTIFF already exists, skipping: {output_path}")
        return output_path
    
    with rasterio.open(band_paths[0]) as template:
        profile = template.profile.copy()
        transform = template.transform
        width = template.width
        height = template.height
        
        print(f"Template band info:")
        print(f"  Resolution: {template.res[0]}m")
        print(f"  Original dimensions: {width}x{height}")
        print(f"  CRS: {template.crs}")
        print(f"  Bounds: {template.bounds}")
        
        if template.res[0] != target_resolution:
            scale_factor = template.res[0] / target_resolution
            width = int(width * scale_factor)
            height = int(height * scale_factor)
            left, bottom, right, top = template.bounds
            transform = rasterio.transform.from_bounds(
                left, bottom, right, top, width, height
            )
            print(f"  Scaling from {template.res[0]}m to {target_resolution}m (factor: {scale_factor:.3f})")
            print(f"  New dimensions: {width}x{height}")
        else:
            print(f"  No scaling needed, already at {target_resolution}m resolution")
    
    profile.update({
        'count': len(band_paths),
        'width': width,
        'height': height,
        'transform': transform,
        'dtype': 'float32',
        'compress': 'lzw',
        'tiled': True,
        # set this small because our tiles are currently small (~25 pixels)
        'blockxsize': 64,
        'blockysize': 64
    })
    
    with rasterio.open(output_path, 'w', **profile) as dst:
        for i, band_path in enumerate(band_paths):
            with rasterio.open(band_path) as src:
                if src.res[0] != target_resolution:
                    scale_factor = src.res[0] / target_resolution
                    new_width = int(src.width * scale_factor)
                    new_height = int(src.height * scale_factor)
                    data = src.read(1, out_shape=(new_height, new_width), 
                                  resampling=rasterio.enums.Resampling.bilinear)
                else:
                    data = src.read(1)
                dst.write(data.astype('float32'), i + 1)
    
    return output_path


def get_raster_info(raster_path: str) -> Dict[str, Any]:
    """
    Extract metadata information from a raster file.
    
    Args:
        raster_path: Path to the raster file
        
    Returns:
        Dictionary containing raster metadata including CRS, bounds, resolution, etc.
    """
    with rasterio.open(raster_path) as src:
        return {
            'crs': src.crs,
            'bounds': src.bounds,
            'width': src.width,
            'height': src.height,
            'resolution': src.res,
            'count': src.count,
            'dtype': src.dtypes[0] if src.count > 0 else None,
            'transform': src.transform
        }