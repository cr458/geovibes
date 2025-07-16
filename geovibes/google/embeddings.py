import argparse
import ee
import geopandas as gpd
import pandas as pd
from dataclasses import dataclass, field

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

def get_crs_from_tile(tile_series: pd.Series) -> str:
    """Get the CRS from a tile series by reading the 'epsg' column."""
    try:
        epsg_code = tile_series['epsg']
        return f"EPSG:{epsg_code}"
    except KeyError:
        raise ValueError("Input series must have an 'epsg' column.")

def find_intersecting_mgrs_tiles(roi_file: str, mgrs_reference_file: str) -> gpd.GeoDataFrame:
    """Find MGRS tiles that intersect with the given ROI."""
    print(f"Loading MGRS reference from: {mgrs_reference_file}")
    if mgrs_reference_file.endswith(".parquet"):
        mgrs_gdf = gpd.read_parquet(mgrs_reference_file)
    elif mgrs_reference_file.endswith(".geojson"):
        mgrs_gdf = gpd.read_file(mgrs_reference_file)
    else:
        raise ValueError("MGRS reference file must be a .parquet or .geojson file.")

    print(f"Filtering MGRS tiles by ROI: {roi_file}")
    if roi_file.endswith(".parquet"):
        roi_gdf = gpd.read_parquet(roi_file)
    elif roi_file.endswith(".geojson"):
        roi_gdf = gpd.read_file(roi_file)
    else:
        raise ValueError("ROI file must be a .parquet or .geojson file.")

    if mgrs_gdf.crs != roi_gdf.crs:
        print(f"Warning: MGRS CRS ({mgrs_gdf.crs}) and ROI CRS ({roi_gdf.crs}) differ. Reprojecting ROI.")
        roi_gdf = roi_gdf.to_crs(mgrs_gdf.crs)

    roi_geometry = roi_gdf.union_all()
    intersecting_mask = mgrs_gdf.intersects(roi_geometry)
    intersecting_gdf = mgrs_gdf[intersecting_mask]
    print(f"Found {len(intersecting_gdf)} MGRS tiles intersecting with the ROI.")
    return intersecting_gdf

def aggregate_satellite_embeddings(
    roi_file: str,
    mgrs_reference_file: str,
    year: int,
    gcs_bucket: str,
    gcs_prefix: str,
    gee_asset_path: str,
    tilesize: int,
    overlap: int,
    resolution: float,
):
    """
    Aggregates Google Satellite Embeddings for a collection of tiles and exports the result.
    """
    try:
        ee.Initialize(project='demeterlabs-gee')
        print("Earth Engine initialized successfully.")
    except Exception as e:
        raise RuntimeError("Could not initialize Earth Engine. Please ensure you have authenticated.") from e

    intersecting_tiles_gdf = find_intersecting_mgrs_tiles(roi_file, mgrs_reference_file)

    if intersecting_tiles_gdf.empty:
        print("No intersecting MGRS tiles found. Exiting.")
        return

    print(f"\nFound {len(intersecting_tiles_gdf)} tiles to process. Starting export tasks...")

    # Load the Google Satellite Embedding collection for the specified year.
    start_date = f'{year}-01-01'
    end_date = f'{year+1}-01-01'
    embedding_image = ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL') \
                        .filterDate(start_date, end_date) \
                        .mosaic()

    # Create a list of all 64 band names (A00, A01, ..., A63).
    band_names = [f'A{i:02d}' for i in range(64)]
    embedding_image = embedding_image.select(band_names)

    def per_feature_median(feature):
        """Attach the image's per‑band median to the input feature."""
        stats = embedding_image.reduceRegion(
            reducer=ee.Reducer.median(),
            geometry=feature.geometry(),
            scale=10,          # Native resolution of the image
            tileScale=8,       # Bump this higher (e.g. 16) for very large polygons
            maxPixels=1e13     # Increase if you still hit maxPixels errors
        )
        return feature.set(stats)

    for _, tile_series in intersecting_tiles_gdf.iterrows():
        # 1. Construct asset ID
        crs = get_crs_from_tile(tile_series)
        grid = MGRSTileGrid(
            mgrs_tile_id=tile_series.mgrs_id,
            crs=crs,
            tilesize=tilesize,
            overlap=overlap,
            resolution=resolution,
        )
        asset_id = f"{gee_asset_path}/{grid.prefix}"
        
        print(f"\nProcessing asset: {asset_id}")

        try:
            tile_collection = ee.FeatureCollection(asset_id)
            if tile_collection.size().getInfo() == 0:
                print(f"  Warning: Asset '{asset_id}' is empty. Skipping.")
                continue
        except ee.EEException as e:
            print(f"  Error: Could not load asset '{asset_id}'. Skipping. Details: {e}")
            continue

        stats_collection = tile_collection.map(per_feature_median)

        # Construct a simpler filename and a more descriptive prefix
        gcs_filename = f"{grid.mgrs_tile_id}_{year}.geojson"
        
        # Construct the full prefix path, including the dynamic part from tiling params
        tiling_params_str = f"{tilesize}_{overlap}_{int(resolution)}"
        full_gcs_prefix = f"{gcs_prefix}/{tiling_params_str}" if gcs_prefix else tiling_params_str

        gcs_filename_with_prefix = f"{full_gcs_prefix}/{gcs_filename}"

        task_description = f'export_{grid.mgrs_tile_id}_{year}_{tiling_params_str}'
        
        print(f"  Starting export to gs://{gcs_bucket}/{gcs_filename_with_prefix}")
        task = ee.batch.Export.table.toCloudStorage(
            collection=stats_collection,
            description=task_description,
            bucket=gcs_bucket,
            fileNamePrefix=gcs_filename_with_prefix.split('.')[0],
            fileFormat='GeoJSON'
        )
        task.start()
        print(f"  Task started: {task.id}")

    print("\nAll export tasks started. Monitor progress in the GEE Code Editor.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Aggregate Google Satellite Embeddings over specified GEE tile assets."
    )
    parser.add_argument("--roi_file", type=str, required=True, help="Path to a GeoJSON/GeoParquet file to filter MGRS tiles.")
    parser.add_argument("--mgrs_reference_file", type=str, default='./mgrs_tiles.parquet', help="Path to GeoParquet file with MGRS tile geometries.")
    parser.add_argument("--year", type=int, default=2024, help="The year for which to get the satellite embeddings (e.g., 2023).")
    parser.add_argument("--gcs_bucket", type=str, default='geovibes', help="The GCS bucket to export the final GeoJSON file to.")
    parser.add_argument("--gcs_prefix", type=str, default="embeddings/google_satellite_v1", help="Base GCS prefix/folder within the bucket.")
    parser.add_argument("--gee_asset_path", type=str, default='projects/demeterlabs-gee/assets/tiles', help="GEE asset path for tile assets (e.g., 'projects/user/assets/tiles').")
    parser.add_argument("--tilesize", type=int, default=25, help="Tile size in pixels used to construct asset name.")
    parser.add_argument("--overlap", type=int, default=0, help="Overlap in pixels used to construct asset name.")
    parser.add_argument("--resolution", type=float, default=10.0, help="Resolution in meters per pixel used to construct asset name.")

    args = parser.parse_args()

    aggregate_satellite_embeddings(
        roi_file=args.roi_file,
        mgrs_reference_file=args.mgrs_reference_file,
        year=args.year,
        gcs_bucket=args.gcs_bucket,
        gcs_prefix=args.gcs_prefix,
        gee_asset_path=args.gee_asset_path,
        tilesize=args.tilesize,
        overlap=args.overlap,
        resolution=args.resolution,
    ) 