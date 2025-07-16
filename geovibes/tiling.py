from dataclasses import dataclass, field

import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
import shapely.geometry
import shapely.ops


def get_crs_from_tile(tile_series: pd.Series) -> str:
    """
    Get the CRS from a tile series by reading the 'epsg' column.
    """
    try:
        epsg_code = tile_series['epsg']
        return f"EPSG:{epsg_code}"
    except KeyError:
        raise ValueError("Input series must have an 'epsg' column.")


@dataclass
class MGRSTileGrid:
    """Class for tracking a MGRS tile grid"""
    mgrs_tile_id: str
    crs: str
    tilesize: int
    overlap: int
    resolution: float
    prefix: str = field(init=False)

    def __post_init__(self):
        self.prefix = f"{self.mgrs_tile_id}_{self.crs.split(':')[-1]}_{self.tilesize}_{self.overlap}_{int(self.resolution)}"


def chip_mgrs_tile(
    tile_series: pd.Series, mgrs_tile_grid: MGRSTileGrid, source_crs: pyproj.CRS) -> gpd.GeoDataFrame:
    """
    Top level function to generate chips over an MGRS tile
    """
    xform_utm = pyproj.Transformer.from_crs(source_crs, mgrs_tile_grid.crs, always_xy=True)
    tile_geom_utm = shapely.ops.transform(xform_utm.transform, tile_series.geometry)

    eff_tilesize = mgrs_tile_grid.tilesize * mgrs_tile_grid.resolution
    eff_overlap = mgrs_tile_grid.overlap * mgrs_tile_grid.resolution
    grid_spacing = eff_tilesize - eff_overlap

    bounds_utm = tile_geom_utm.bounds
    sw_utm = bounds_utm[0], bounds_utm[1]
    ne_utm = bounds_utm[2], bounds_utm[3]

    x_diff = ne_utm[0] - sw_utm[0]
    y_diff = ne_utm[1] - sw_utm[1]

    x_samples = round(x_diff / grid_spacing) + 1
    y_samples = round(y_diff / grid_spacing) + 1

    xs = np.arange(0, x_samples) * grid_spacing + sw_utm[0]
    ys = np.arange(0, y_samples) * grid_spacing + sw_utm[1]

    x_grid, y_grid = np.meshgrid(xs, ys)

    return generate_chips(
        x_samples=x_samples,
        y_samples=y_samples,
        x_grid=x_grid,
        y_grid=y_grid,
        eff_tilesize=eff_tilesize,
        mgrs_tile_grid=mgrs_tile_grid,
        tile_geom_utm=tile_geom_utm,
    )


def generate_chips(
    x_samples: int,
    y_samples: int,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    eff_tilesize: float,
    mgrs_tile_grid: MGRSTileGrid,
    tile_geom_utm: shapely.geometry.Polygon,
) -> gpd.GeoDataFrame:
    """
    Generate chips over a grid and return them as a GeoDataFrame.
    """
    tiles = []
    for i in range(x_samples):
        for j in range(y_samples):
            x, y = x_grid[j, i], y_grid[j, i]
            geom = shapely.geometry.Point(x, y).buffer(eff_tilesize / 2, cap_style=3)

            if tile_geom_utm.intersects(geom):
                tile = {
                    'geometry': geom,
                    'tile_id': f"{mgrs_tile_grid.mgrs_tile_id}_{mgrs_tile_grid.tilesize}_{mgrs_tile_grid.overlap}_{int(mgrs_tile_grid.resolution)}_{j}_{i}"
                }
                tiles.append(tile)

    return gpd.GeoDataFrame(tiles, crs=mgrs_tile_grid.crs)