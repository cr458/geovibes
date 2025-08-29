import argparse
import logging
import pathlib
import sys
from typing import List, Dict, Optional
import os
import io
import re

import geopandas as gpd
import pandas as pd
from shapely.ops import unary_union
from tqdm import tqdm
from joblib import Parallel, delayed
import s3fs

from geovibes.tiling import MGRSTileId, get_crs_from_mgrs_tile_id, get_mgrs_tile_ids_for_roi_from_roi_file


def setup_logging():
    """Configure basic logging settings."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def _load_region(path: str) -> gpd.GeoSeries:
    """Load a vector file and return a single unified geometry in WGS84."""
    path = pathlib.Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    gdf = gpd.read_file(path)
    if gdf.empty:
        raise ValueError("No geometries found in input file")
    if gdf.crs is None or gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(4326)
    geom = unary_union(gdf.geometry)
    return gpd.GeoSeries([geom], crs="EPSG:4326")


def process_and_save_geojson(gcs_path: str, epsg_code: str, output_dir: str, start_time: str, end_time: str, s3_endpoint: Optional[str] = None) -> Optional[str]:
    """
    Reads a GeoJSON from GCS, processes it, and saves it as a local GeoParquet file.
    """
    try:
        stem = pathlib.Path(gcs_path).stem
        m = re.match(r"^(?P<prefix>.+?)_\d{4}$", stem)
        base = m.group("prefix") if m else stem
        output_filename = f"{base}_{start_time}_{end_time}.parquet"
        output_uri = f"{output_dir.rstrip('/')}" + "/" + output_filename
            
        gdf = gpd.read_file(gcs_path).drop(columns=["id"])
        if gdf.empty:
            return None

        # Calculate accurate centroids in the tile's specific UTM projection
        centroids = []
        for _, row in gdf.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                centroids.append(None)
                continue
            
            epsg_str = str(epsg_code)
            target_crs = epsg_str if epsg_str.upper().startswith("EPSG:") else f"EPSG:{epsg_str}"
            centroid_utm = gpd.GeoSeries([geom], crs="EPSG:4326").to_crs(target_crs).centroid
            centroid_wgs84 = centroid_utm.to_crs("EPSG:4326").iloc[0]
            centroids.append(centroid_wgs84)

        gdf['geometry'] = centroids
        gdf = gdf[gdf.geometry.notna()]
        
        band_names = [f'A{i:02d}' for i in range(64)]
        
        # Ensure all band columns exist, fill with 0 if not
        for band in band_names:
            if band not in gdf.columns:
                gdf[band] = 0
                
        gdf['embedding'] = gdf[band_names].values.tolist()

        # Rename 'id' to 'tile_id' if it exists, otherwise create it from filename
        if 'id' in gdf.columns:
            gdf = gdf.rename(columns={'id': 'tile_id'})
        elif 'tile_id' not in gdf.columns:
            mgrs_id = pathlib.Path(gcs_path).stem.split('_')[0]
            gdf['tile_id'] = mgrs_id
        
        # Select final columns
        final_cols = ['tile_id', 'geometry', 'embedding']
        final_gdf = gdf[final_cols]
        
        # Save to local geoparquet file
        if output_dir.startswith("s3://"):
            endpoint = s3_endpoint or "https://data.source.coop"
            buffer = io.BytesIO()
            final_gdf.to_parquet(buffer)
            buffer.seek(0)
            fs = s3fs.S3FileSystem(client_kwargs={"endpoint_url": endpoint}, config_kwargs={"s3": {"addressing_style": "path"}})
            with fs.open(output_uri, "wb") as f:
                f.write(buffer.read())
            return output_uri
        else:
            pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
            final_gdf.to_parquet(output_uri)
            return output_uri
    
    except Exception as e:
        logging.warning(f"Failed to process {gcs_path}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Convert Google Satellite GeoJSON files to GeoParquet format.")
    parser.add_argument("roi_file", help="Path to the ROI file (e.g., aoi.geojson).")
    parser.add_argument("output_dir", help="Directory to save output files. Can be local path or s3:// prefix.")
    parser.add_argument("--mgrs_reference_file", default="/Users/christopherren/geovibes/geometries/mgrs_tiles.parquet", help="Path to the MGRS grid reference file.")
    parser.add_argument("--gcs_bucket", default="geovibes", help="GCS bucket to use for the embeddings.")
    parser.add_argument("--gcs_prefix", type=str, default="embeddings/google_satellite_v1", help="Base GCS prefix/folder within the bucket.")
    parser.add_argument("--year", type=int, default=2024, help="The year for which to get the satellite embeddings (e.g., 2023).")
    parser.add_argument("--tilesize", type=int, default=25, help="Tile size in pixels used to construct asset name.")
    parser.add_argument("--overlap", type=int, default=0, help="Overlap in pixels used to construct asset name.")
    parser.add_argument("--resolution", type=float, default=10.0, help="Resolution in meters per pixel used to construct asset name.")
    parser.add_argument("--workers", type=int, default=-1, help="Number of parallel workers for processing files.")
    parser.add_argument("--start_time", type=str, default="2024-01-01", help="Start date appended to parquet filenames.")
    parser.add_argument("--end_time", type=str, default="2025-01-01", help="End date appended to parquet filenames.")
    parser.add_argument("--s3_endpoint", type=str, default="https://data.source.coop", help="S3 endpoint when output_dir is s3://.")
    args = parser.parse_args()

    setup_logging()

    # Create output directory if it doesn't exist
    output_path = pathlib.Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        # Find intersecting MGRS tiles and their EPSG codes
        mgrs_tile_ids: list[MGRSTileId] = get_mgrs_tile_ids_for_roi_from_roi_file(args.roi_file, args.mgrs_reference_file)
        if not mgrs_tile_ids:
            logging.info("No intersecting MGRS tiles found for the given ROI.")
            return

        local_parquet_files = []
        tasks_to_process = []

        tiling_params_str = f"{args.tilesize}_{args.overlap}_{int(args.resolution)}"
        full_gcs_prefix = f"{args.gcs_prefix}/{tiling_params_str}" if args.gcs_prefix else tiling_params_str

        logging.info("Checking for existing parquet files...")
        for mgrs_tile_id in mgrs_tile_ids:
            expected_filename = f"{mgrs_tile_id}_{args.start_time}_{args.end_time}.parquet"
            local_path = output_path / expected_filename

            if not args.output_dir.startswith("s3://") and local_path.exists():
                local_parquet_files.append(str(local_path))
            else:
                gcs_path = f"gs://{args.gcs_bucket}/{full_gcs_prefix}/{mgrs_tile_id}_{args.year}.geojson"
                tasks_to_process.append((gcs_path, get_crs_from_mgrs_tile_id(mgrs_tile_id), args.output_dir, args.start_time, args.end_time, args.s3_endpoint))
        
        logging.info(f"Found {len(local_parquet_files)} existing parquet files.")
        
        if tasks_to_process:
            logging.info(f"Constructed {len(tasks_to_process)} tasks to process for missing files.")

            # Process and save GeoJSONs in parallel for missing files
            processed_files = Parallel(n_jobs=args.workers, verbose=20)(
                delayed(process_and_save_geojson)(*task)
                for task in tasks_to_process
            )

            newly_processed_files = [f for f in processed_files if f is not None]
            failed_downloads = len(processed_files) - len(newly_processed_files)
            
            if failed_downloads > 0:
                logging.error(f"Failed to download/process {failed_downloads} out of {len(tasks_to_process)} required files.")
                logging.error("Cannot proceed with incomplete data. All required files must be successfully downloaded.")
                sys.exit(1)
            
            local_parquet_files.extend(newly_processed_files)
            logging.info(f"Successfully processed all {len(newly_processed_files)} missing files.")

        # Validate we have all expected files
        expected_file_count = len(mgrs_tile_ids)
        actual_file_count = len(local_parquet_files)
        
        if actual_file_count != expected_file_count:
            logging.error(f"File count mismatch: Expected {expected_file_count} files but have {actual_file_count} files.")
            logging.error("Cannot proceed with incomplete data.")
            sys.exit(1)

        if not local_parquet_files:
            logging.error("No data could be processed or found.")
            sys.exit(1)

        logging.info(f"Validated all {len(local_parquet_files)} required parquet files are present.")
        logging.info("Parquet files created successfully. Use database.py to create DuckDB index.")

    except Exception as e:
        logging.error(f"An error occurred during the process: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()