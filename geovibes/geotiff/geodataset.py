from collections.abc import Callable
import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import geometry_window
import torch
from rasterio.enums import Resampling
from rasterio.env import Env
import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)


class GeotiffTileDataset(torch.utils.data.Dataset):
    """Optimized Torch Dataset with memory mapping and caching."""

    def __init__(
        self,
        path: str,  # path to geotiff
        geometries_gdf: gpd.GeoDataFrame,  # GeoDataFrame with geometries
        transforms: Callable | None = None,
        target_size: tuple[int, int] = (224, 224),  # Target size for all patches
        target_resolution: Optional[int] = None, # Target resolution to resample to
        use_memmap: bool = True,  # Use memory mapping
        cache_windows: bool = False,  # Cache a`ll` window calculations
    ) -> None:
        """Initialize the dataset.

        Args:
            path: The path to the geotiff file.
            geometries_gdf: GeoDataFrame containing polygon geometries to extract.
            transforms: The transforms to apply to the image.
            target_size: Target size (height, width) for all extracted patches.
            target_resolution: Target resolution (in meters) to resample the patch to.
            use_memmap: Whether to use memory mapping for reading.
            cache_windows: Whether to cache window calculations.
        """
        self.path = path
        self.transforms = transforms
        self.target_size = target_size
        self.use_memmap = use_memmap
        self.target_resolution = target_resolution
        
        # Configure rasterio environment for optimal performance
        # currently we download the geotiff to a local directory, maybe we should change this to read windows from the cloud directly
        self.env_kwargs = {
            'GDAL_DISABLE_READDIR_ON_OPEN': 'EMPTY_DIR',
            'GDAL_NUM_THREADS': 1,
            'GDAL_CACHEMAX': 128,  # MB as integer
            'AWS_NO_SIGN_REQUEST': 'YES',
            'GDAL_MAX_RAW_BLOCK_CACHE_SIZE': '200000000',
            'GDAL_SWATH_SIZE': '200000000',
            'VSI_CURL_CACHE_SIZE': '200000000',
        }
        
        with Env(**self.env_kwargs):
            with rasterio.open(path) as src:
                self.width, self.height = src.width, src.height
                self.crs = src.crs
                self.transform = src.transform
                self.count = src.count
                self.native_resolution = src.res[0]
                
                # Ensure geometries are in the same CRS as the raster
                if geometries_gdf.crs != self.crs:
                    self.geometries_gdf = geometries_gdf.to_crs(self.crs)
                else:
                    self.geometries_gdf = geometries_gdf.copy()
        
        # Pre-calculate windows if caching is enabled
        self._window_cache = {}
        if cache_windows:
            self._precalculate_windows()
        
        assert len(self) > 0, "No geometries provided"
        
        # Log native tile size for the first geometry
        self._log_native_tile_size()

    def _precalculate_windows(self):
        """Pre-calculate all windows for faster access."""
        logger.info(f"Pre-calculating windows for {len(self.geometries_gdf)} geometries...")
        start_time = time.time()
        
        with Env(**self.env_kwargs):
            with rasterio.open(self.path) as src:
                for idx in range(len(self.geometries_gdf)):
                    if idx % 1000 == 0:
                        logger.info(f"Pre-calculating window {idx}/{len(self.geometries_gdf)}")
                    
                    geometry = self.geometries_gdf.iloc[idx].geometry
                    window = geometry_window(src, [geometry])
                    self._window_cache[idx] = window
        
        elapsed = time.time() - start_time
        logger.info(f"Window pre-calculation completed in {elapsed:.2f}s")

    def _log_native_tile_size(self):
        """Log the native tile size for the first geometry (for debugging)."""
        try:
            with Env(**self.env_kwargs):
                with rasterio.open(self.path) as src:
                    first_geometry = self.geometries_gdf.iloc[0].geometry
                    window = geometry_window(src, [first_geometry])
                    
                    # Read at native resolution to see actual size
                    data = src.read(
                        boundless=False,
                        window=window,
                        fill_value=0,
                        masked=False
                    )
                    
                    # Get geometry bounds and calculate expected size
                    geom_bounds = first_geometry.bounds  # (minx, miny, maxx, maxy)
                    width_meters = geom_bounds[2] - geom_bounds[0]
                    height_meters = geom_bounds[3] - geom_bounds[1]
                    resolution = src.res[0]
                    expected_width = int(width_meters / resolution)
                    expected_height = int(height_meters / resolution)
                    
                    logger.info(f"Raster info: resolution={resolution}m, CRS={src.crs}")
                    logger.info(f"Geometry bounds: {geom_bounds} (width={width_meters:.1f}m, height={height_meters:.1f}m)")
                    logger.info(f"Expected size at {resolution}m: {expected_height}x{expected_width} pixels")
                    logger.info(f"Window: row_off={window.row_off}, col_off={window.col_off}, height={window.height}, width={window.width}")
                    logger.info(f"Native tile size for first geometry: {data.shape} (channels, height, width)")
        except Exception as e:
            logger.warning(f"Could not determine native tile size: {e}")

    def read_window_from_geometry(
        self,
        geometry,
        index: int,
        interpolation: Resampling = Resampling.bilinear,
    ) -> tuple[np.ndarray, dict]:
        """
        Read the window corresponding to a geometry, handling edges by padding.
        
        This method first checks if padding is necessary. If the geometry is
        fully contained within the raster, it performs a direct read. Otherwise,
        it calculates the required padding and applies it using reflection.
        """
        read_start = time.time()
        
        with Env(**self.env_kwargs):
            with rasterio.open(self.path, 'r') as src:
                # Get the window clipped to raster bounds.
                clipped_window = geometry_window(src, [geometry], boundless=False)
                if index not in self._window_cache:
                    self._window_cache[index] = clipped_window

                # Determine resampling scale, if any.
                scale_factor = 1.0
                if self.target_resolution and self.native_resolution != self.target_resolution:
                    scale_factor = self.native_resolution / self.target_resolution
                
                # Calculate the expected shape based on the geometry's metric size.
                geom_bounds = geometry.bounds
                expected_height = int(round((geom_bounds[3] - geom_bounds[1]) / self.native_resolution * scale_factor))
                expected_width = int(round((geom_bounds[2] - geom_bounds[0]) / self.native_resolution * scale_factor))
                
                # Calculate the shape of the data we can actually read.
                read_height = int(clipped_window.height * scale_factor)
                read_width = int(clipped_window.width * scale_factor)
                
                # Check if the readable window is smaller than the geometry's full extent,
                # allowing a 1-pixel tolerance for rounding differences.
                needs_padding = (read_height < expected_height - 1) or (read_width < expected_width - 1)

                if not needs_padding:
                    # Optimized path: Geometry is not on an edge. Read directly.
                    out_shape = (self.count, expected_height, expected_width)
                    data = src.read(
                        boundless=False,
                        window=clipped_window,
                        out_shape=out_shape,
                        resampling=interpolation,
                        fill_value=0,
                        masked=False
                    )
                else:
                    # Padding path: Geometry is on an edge.
                    logger.debug(f"Applying padding for geometry {index}.")
                    full_window = geometry_window(src, [geometry], boundless=True)

                    # Read the partial data available.
                    data = src.read(
                        boundless=False,
                        window=clipped_window,
                        out_shape=(self.count, read_height, read_width),
                        resampling=interpolation,
                        fill_value=0,
                        masked=False
                    )
                    
                    # Calculate padding based on the difference between the full and clipped windows.
                    top_offset = clipped_window.row_off - full_window.row_off
                    left_offset = clipped_window.col_off - full_window.col_off
                    
                    pad_top = int(top_offset * scale_factor)
                    pad_left = int(left_offset * scale_factor)
                    pad_bottom = expected_height - data.shape[1] - pad_top
                    pad_right = expected_width - data.shape[2] - pad_left
                    
                    padding = (
                        (0, 0),
                        (max(0, pad_top), max(0, pad_bottom)),
                        (max(0, pad_left), max(0, pad_right)),
                    )
                    
                    data = np.pad(data, padding, mode='reflect')

                # Final crop/pad to ensure exact target size due to rounding.
                current_h, current_w = data.shape[-2:]
                if current_h != expected_height or current_w != expected_width:
                    final_data = np.zeros((self.count, expected_height, expected_width), dtype=data.dtype)
                    h_copy = min(current_h, expected_height)
                    w_copy = min(current_w, expected_width)
                    final_data[:, :h_copy, :w_copy] = data[:, :h_copy, :w_copy]
                    data = final_data
        
        read_elapsed = time.time() - read_start
        if read_elapsed > 0.1:
            logger.warning(f"Slow read for geometry {index}: {read_elapsed:.3f}s")
        
        geom_info = {
            'geometry': geometry,
            'index': index,
            'window': clipped_window,
            'bounds_pixel': (clipped_window.col_off, clipped_window.row_off, clipped_window.col_off + clipped_window.width, clipped_window.row_off + clipped_window.height)
        }
        
        logger.debug(f"Successfully read and padded window for geometry {index}, final data shape: {data.shape}")
        return data, geom_info

    def __len__(self) -> int:
        return len(self.geometries_gdf)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | int | str]:
        start_time = time.time()
        logger.debug(f"Starting __getitem__ for index {index}")
        
        # Get geometry from GeoDataFrame
        geom_start = time.time()
        row = self.geometries_gdf.iloc[index]
        geometry = row.geometry
        geom_elapsed = time.time() - geom_start
        logger.debug(f"Geometry lookup for index {index}: {geom_elapsed:.4f}s")
        
        # Read window
        read_start = time.time()
        x, _ = self.read_window_from_geometry(geometry, index)
        read_elapsed = time.time() - read_start
        logger.debug(f"Window read for index {index}: {read_elapsed:.4f}s")
        
        # Convert to float32 tensor directly (avoid double conversion)
        tensor_start = time.time()
        x = torch.from_numpy(x.astype(np.float32, copy=False))
        tensor_elapsed = time.time() - tensor_start
        logger.debug(f"Tensor conversion for index {index}: {tensor_elapsed:.4f}s")
        
        # Clip negative values in-place
        clip_start = time.time()
        x.clamp_(min=0)
        clip_elapsed = time.time() - clip_start
        logger.debug(f"Tensor clipping for index {index}: {clip_elapsed:.4f}s")

        # Apply transforms
        transform_start = time.time()
        if self.transforms is not None:
            logger.debug(f"Applying transforms for index {index}")
            x = self.transforms(x)
        transform_elapsed = time.time() - transform_start
        logger.debug(f"Transform application for index {index}: {transform_elapsed:.4f}s")

        # Create result
        result_start = time.time()
        result = {
            "image": x, 
            "geometry_index": index,
            "original_size": torch.tensor([x.shape[1], x.shape[2]]),
            "tile_id": str(row.get('tile_id', index))
        }
        result_elapsed = time.time() - result_start
        logger.debug(f"Result creation for index {index}: {result_elapsed:.4f}s")
        
        total_elapsed = time.time() - start_time
        if total_elapsed > 0.5:  # Log slow items
            logger.warning(f"Slow __getitem__ for index {index}: {total_elapsed:.3f}s total")
        else:
            logger.debug(f"Completed __getitem__ for index {index}: {total_elapsed:.4f}s total")
        
        return result