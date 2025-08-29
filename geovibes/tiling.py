from dataclasses import dataclass, field
import logging

import duckdb
import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
import shapely.geometry
from shapely import Geometry
import shapely.ops

@dataclass
class MGRSTileId:
    """Example MGRS tile: 18SJH"""
    utm_zone: int
    latitude_band: str
    grid_square: str

    def __post_init__(self):
        if not isinstance(self.utm_zone, int):
            raise ValueError("utm_zone must be an integer")
        if not 1 <= self.utm_zone <= 60:
            raise ValueError("utm_zone must be between 1 and 60")
        if not isinstance(self.latitude_band, str) or len(self.latitude_band) != 1:
            raise ValueError("latitude_band must be a single character string")
        if not isinstance(self.grid_square, str) or len(self.grid_square) != 2:
            raise ValueError("grid_square must be a two character string")

    def __str__(self):
        return f"{self.utm_zone}{self.latitude_band}{self.grid_square}"

    @property
    def crs(self) -> pyproj.CRS:
        """Get the CRS for this MGRS tile's UTM zone."""
        return get_crs_from_mgrs_tile_id(self)

    @classmethod
    def from_str(cls, mgrs_id: str) -> "MGRSTileId":
        if len(mgrs_id) < 4 or len(mgrs_id) > 5:
            raise ValueError("MGRS ID must be 4 or 5 characters long")
        
        grid_square = mgrs_id[-2:]
        latitude_band = mgrs_id[-3]
        utm_zone = int(mgrs_id[:-3])
            
        return MGRSTileId(
            utm_zone=utm_zone,
            latitude_band=latitude_band,
            grid_square=grid_square)

@dataclass
class MGRSTileGrid:
    """Class for tracking a MGRS tile grid"""
    mgrs_tile_id: MGRSTileId
    tilesize: int
    overlap: int
    resolution: float
    prefix: str = field(init=False)

    def __post_init__(self):
        self.prefix = f"{self.mgrs_tile_id}_{self.crs.to_epsg()}_{self.tilesize}_{self.overlap}_{int(self.resolution)}"

    @property
    def crs(self) -> pyproj.CRS:
        """Get the CRS for this MGRS tile's UTM zone."""
        return self.mgrs_tile_id.crs


def get_crs_from_mgrs_tile_id(mgrs_tile_id: MGRSTileId) -> pyproj.CRS:
    """Get the CRS for a MGRS tile's UTM zone."""
    north_bands = ['N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X']
    base = 32600 if mgrs_tile_id.latitude_band in north_bands else 32700
    epsg = base + mgrs_tile_id.utm_zone
    return pyproj.CRS(f"EPSG:{epsg}")


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


def get_mgrs_tile_ids_for_roi(
    search_geometry: Geometry,
    search_geometry_crs: str | pyproj.CRS | None ,
    mgrs_tiles_file: str = "geometries/mgrs_tiles.parquet",
) -> list[MGRSTileId]:
    """
    Return all MGRSTileIds whose footprints intersect the *search_geometry*.

    Parameters
    ----------
    search_geometry : shapely.geometry
        Region‑of‑interest geometry.
    search_geometry_crs : str | pyproj.CRS | None
        CRS of *search_geometry* (e.g. 'EPSG:4326'). If not provided, will be assumed to be EPSG:4326.
    mgrs_tiles_file : str, default "geometries/mgrs_tiles.parquet"
        Path to the GeoParquet file that stores MGRS tiles. Assumption this is in 
        CRS 4326.

    Returns
    -------
    list[MGRSTileId]
        List of MGRS tile IDs that intersect the search geometry.
    """
    if search_geometry_crs is None:
        search_geometry_crs = "EPSG:4326"
        logging.warning("No CRS provided for search geometry, assuming EPSG:4326")

    if str(search_geometry_crs) != "EPSG:4326":
        logging.info(f"Reprojecting search geometry from {search_geometry_crs} to EPSG:4326")
        transformer = pyproj.Transformer.from_crs(
            search_geometry_crs, "EPSG:4326", always_xy=True
        )
        search_geometry = shapely.ops.transform(transformer.transform, search_geometry)

    con = duckdb.connect()
    con.execute("INSTALL spatial; LOAD spatial;")
    q = f"""
        SELECT mgrs_id
        FROM parquet_scan('{mgrs_tiles_file}')
        WHERE ST_Intersects(geometry,
                            ST_GeomFromText(?))
    """
    result = con.execute(q, [shapely.to_wkt(search_geometry)]).fetchall()
    return [MGRSTileId.from_str(mgrs_id=row[0]) for row in result]
    
def get_mgrs_tile_ids_for_roi_from_roi_file(
    roi_geojson_file: str,
    mgrs_tiles_file: str = "geometries/mgrs_tiles.parquet",
) -> list[MGRSTileId]:
    """
    Return all MGRSTileIdswhose footprints intersect the *search_geometry*.

    Parameters
    ----------
    roi_geojson_file : str
        Path to the ROI file readable by geopandas.read_file or geopandas.read_parquet.
    mgrs_tiles_file : str, default "geometries/mgrs_tiles.parquet"
        Path to the GeoParquet file that stores MGRS tiles. Assumption that this is in 
        CRS 4326.

    Returns
    -------
    list[MGRSTileId]
        List of MGRS tile IDs that intersect the search geometry.
    """
    if roi_geojson_file.endswith('.parquet'):
        roi_gdf = gpd.read_parquet(roi_geojson_file)
    else:
        roi_gdf = gpd.read_file(roi_geojson_file)
    if roi_gdf.crs is None:
        raise ValueError("ROI file must have a CRS")
    return get_mgrs_tile_ids_for_roi(roi_gdf.geometry.union_all(), roi_gdf.crs, mgrs_tiles_file)