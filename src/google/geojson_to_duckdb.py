import argparse
import logging
import pathlib
import sys
import time
from typing import List, Set, Optional, Dict

import geopandas as gpd
import duckdb
import pandas as pd
import numpy as np
from shapely.ops import unary_union
from tqdm import tqdm
from joblib import Parallel, delayed
import os

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

def _get_utm_crs(lon: float, lat: float) -> str:
    """Get appropriate UTM CRS for a given longitude/latitude."""
    zone = int((lon + 180) / 6) + 1
    if lat >= 0:
        return f"EPSG:326{zone:02d}"
    else:
        return f"EPSG:327{zone:02d}"

def get_intersecting_mgrs_ids(roi_path: str, mgrs_reference_path: str) -> Dict[str, str]:
    """Find MGRS tiles that intersect with the given ROI and return a dict of tile_id: epsg_code."""
    logging.info("Loading ROI and MGRS reference file...")
    region_gs = _load_region(roi_path)
    mgrs_gdf = gpd.read_parquet(mgrs_reference_path)

    if mgrs_gdf.crs is None or mgrs_gdf.crs.to_epsg() != 4326:
        mgrs_gdf = mgrs_gdf.to_crs(4326)

    logging.info("Finding intersecting MGRS tiles...")
    intersecting_tiles = gpd.sjoin(
        mgrs_gdf, gpd.GeoDataFrame(geometry=region_gs), how="inner", predicate="intersects"
    )

    tile_id_col = "mgrs_id"
    epsg_col = "epsg"
    if tile_id_col not in intersecting_tiles.columns:
        raise ValueError(f"MGRS reference file must have a '{tile_id_col}' column.")
    if epsg_col not in intersecting_tiles.columns:
        raise ValueError(f"MGRS reference file must have an '{epsg_col}' column.")

    return dict(zip(intersecting_tiles[tile_id_col], intersecting_tiles[epsg_col]))

def process_and_save_geojson(gcs_path: str, epsg_code: str, output_dir: str) -> Optional[str]:
    """
    Reads a GeoJSON from GCS, processes it, and saves it as a local GeoParquet file.
    """
    try:
        output_filename = f"{pathlib.Path(gcs_path).stem}.parquet"
        local_path = os.path.join(output_dir, output_filename)
            
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
            
            # Use the provided EPSG code for the projection
            centroid_utm = gpd.GeoSeries([geom], crs="EPSG:4326").to_crs(f"EPSG:{epsg_code}").centroid
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
        final_gdf.to_parquet(local_path)
        
        return local_path
    
    except Exception as e:
        logging.warning(f"Failed to process {gcs_path}: {e}")
        return None

def check_and_clean_embeddings(parquet_files: List[str]) -> List[str]:
    """Check parquet files for NaN embeddings and create cleaned versions."""
    
    logging.info("Checking parquet files for NaN values in embeddings...")
    cleaned_files = []
    total_nan_count = 0
    files_with_nans = []
    
    for parquet_file in tqdm(parquet_files, desc="Checking parquet files"):
        # Read the parquet file as GeoDataFrame to preserve geometry type
        try:
            df = gpd.read_parquet(parquet_file)
        except Exception:
            # Fallback to pandas if geopandas fails
            df = pd.read_parquet(parquet_file)
        original_count = len(df)
        
        # Check for NaN values in embedding column
        if 'embedding' in df.columns:
            # Convert embeddings to numpy arrays for NaN checking
            embeddings = np.array(df['embedding'].tolist())
            
            # Find rows with any NaN values in embeddings
            nan_mask = np.isnan(embeddings).any(axis=1)
            nan_count = nan_mask.sum()
            
            if nan_count > 0:
                logging.warning(f"Found {nan_count} embeddings with NaN values in {pathlib.Path(parquet_file).name}")
                files_with_nans.append(pathlib.Path(parquet_file).name)
                total_nan_count += nan_count
                
                # Remove rows with NaN embeddings
                df_clean = df[~nan_mask].copy()
                
                # Save cleaned version - preserve geometry column type if it's a GeoDataFrame
                clean_filename = parquet_file.replace('.parquet', '_clean.parquet')
                if isinstance(df_clean, gpd.GeoDataFrame):
                    df_clean.to_parquet(clean_filename, index=False)
                else:
                    # Convert to GeoDataFrame if it has geometry column
                    if 'geometry' in df_clean.columns:
                        df_clean = gpd.GeoDataFrame(df_clean)
                    df_clean.to_parquet(clean_filename, index=False)
                cleaned_files.append(clean_filename)
                
                logging.info(f"Cleaned {pathlib.Path(parquet_file).name}: {original_count} -> {len(df_clean)} rows")
            else:
                # No NaN values, use original file
                cleaned_files.append(parquet_file)
        else:
            logging.warning(f"No 'embedding' column found in {pathlib.Path(parquet_file).name}")
            cleaned_files.append(parquet_file)
    
    # Summary report
    if total_nan_count > 0:
        logging.warning(f"NaN SUMMARY:")
        logging.warning(f"  Total embeddings with NaN values: {total_nan_count}")
        logging.warning(f"  Files affected: {len(files_with_nans)}")
        logging.warning(f"  Affected files: {', '.join(files_with_nans)}")
        logging.warning(f"  Cleaned versions created with '_clean.parquet' suffix")
    else:
        logging.info("✅ No NaN values found in any embeddings")
    
    return cleaned_files


def create_duckdb_index(
    parquet_files: List[str], 
    output_file: str, 
    metric: str,
    embedding_column: str = "embedding",
    embedding_dim: int = 64
) -> None:
    """Creates a DuckDB database with HNSW and spatial indexes from local parquet files."""
    
    # Check and clean embeddings before building database
    cleaned_parquet_files = check_and_clean_embeddings(parquet_files)
    
    logging.info(f"Creating DuckDB database at {output_file} with {metric} metric...")
    logging.info(f"Using embedding dimension: {embedding_dim}")

    con = duckdb.connect(database=output_file)

    logging.info("Loading extensions...")
    con.execute("SET enable_progress_bar=true")
    con.execute("INSTALL spatial; LOAD spatial;")
    con.execute("INSTALL vss; LOAD vss;")
    con.execute("SET hnsw_enable_experimental_persistence = true;")

    # Convert string paths to Path objects and create proper path strings
    path_strings = [
        f"'{p_str}'"
        for p_str in (str(pathlib.Path(p).resolve()).replace("\\", "/") for p in cleaned_parquet_files)
    ]
    sql_parquet_files_list_str = "[" + ", ".join(path_strings) + "]"

    create_table_sql = f"""
    CREATE OR REPLACE TABLE geo_embeddings AS
    SELECT
        tile_id AS id,
        CAST({embedding_column} AS FLOAT[{embedding_dim}]) as embedding,
        geometry
    FROM read_parquet({sql_parquet_files_list_str}, union_by_name=true);
    """

    logging.info("Creating table and ingesting data...")
    start_time = time.time()
    con.execute(create_table_sql)
    ingest_time = time.time() - start_time

    row_count = con.execute("SELECT COUNT(*) FROM geo_embeddings;").fetchone()[0]
    logging.info(f"Ingested {row_count} rows in {ingest_time:.2f} seconds")

    if row_count > 0:
        logging.info(f"Creating HNSW index with {metric} metric...")
        start_index_time = time.time()
        con.execute(
            f"CREATE INDEX IF NOT EXISTS emb_hnsw_idx ON geo_embeddings USING HNSW (embedding) WITH (metric = '{metric}');"
        )
        index_time = time.time() - start_index_time
        logging.info(f"HNSW index created in {index_time:.2f} seconds")

        logging.info("Creating RTree spatial index...")
        start_spatial_index_time = time.time()
        con.execute(
            "CREATE INDEX IF NOT EXISTS geom_spatial_idx ON geo_embeddings USING RTREE (geometry);"
        )
        spatial_index_time = time.time() - start_spatial_index_time
        logging.info(f"RTree spatial index created in {spatial_index_time:.2f} seconds")

        db_size_info = con.execute("PRAGMA database_size;").fetchone()
        logging.info(f"Database size: {db_size_info}")

    con.close()
    logging.info(f"Database saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Process Google Satellite GeoJSON files and create a DuckDB index.")
    parser.add_argument("roi_file", help="Path to the ROI file (e.g., aoi.geojson).")
    parser.add_argument("output_dir", help="Directory to save output files.")
    parser.add_argument("output_db_file", help="Path for the output DuckDB database file.")
    parser.add_argument("--mgrs_reference_file", default="/Users/christopherren/geovibes/geometries/mgrs_tiles.parquet", help="Path to the MGRS grid reference file.")
    parser.add_argument("--gcs_bucket", default="geovibes", help="GCS bucket to use for the embeddings.")
    parser.add_argument("--tilesize", type=int, default=100, help="Tile size in pixels used in the embeddings data.")
    parser.add_argument("--overlap", type=int, default=0, help="Overlap in pixels used in the embeddings data.")
    parser.add_argument("--resolution", type=float, default=10.0, help="Resolution in meters per pixel used in the embeddings data.")
    parser.add_argument("--metric", default="cosine", choices=["cosine", "l2sq", "inner_product"], help="Distance metric for HNSW index.")
    parser.add_argument("--embedding_dim", type=int, default=64, help="Dimension of the embedding vectors.")
    parser.add_argument("--workers", type=int, default=-1, help="Number of parallel workers for processing files.")
    args = parser.parse_args()

    setup_logging()

    # Create output directory if it doesn't exist
    output_path = pathlib.Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        # 1. Find intersecting MGRS tiles and their EPSG codes
        mgrs_epsg_map = get_intersecting_mgrs_ids(args.roi_file, args.mgrs_reference_file)
        if not mgrs_epsg_map:
            logging.info("No intersecting MGRS tiles found for the given ROI.")
            return

        local_parquet_files = []
        tasks_to_process = []
        
        # Construct the tiling parameters string
        tiling_params_str = f"{args.tilesize}_{args.overlap}_{int(args.resolution)}"

        logging.info("Checking for existing parquet files...")
        for mgrs_id, epsg_code in mgrs_epsg_map.items():
            expected_filename = f"{mgrs_id}_2024.parquet"
            local_path = output_path / expected_filename

            if local_path.exists():
                local_parquet_files.append(str(local_path))
            else:
                gcs_path = f"gs://{args.gcs_bucket}/embeddings/google_satellite_v1/{tiling_params_str}/{mgrs_id}_2024.geojson"
                tasks_to_process.append((gcs_path, epsg_code, args.output_dir))
        
        logging.info(f"Found {len(local_parquet_files)} existing parquet files.")
        
        if tasks_to_process:
            logging.info(f"Constructed {len(tasks_to_process)} tasks to process for missing files.")

            # Process and save GeoJSONs in parallel for missing files
            processed_files = Parallel(n_jobs=args.workers, verbose=20)(
                delayed(process_and_save_geojson)(*task)
                for task in tqdm(tasks_to_process, desc="Processing GeoJSON files")
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
        expected_file_count = len(mgrs_epsg_map)
        actual_file_count = len(local_parquet_files)
        
        if actual_file_count != expected_file_count:
            logging.error(f"File count mismatch: Expected {expected_file_count} files but have {actual_file_count} files.")
            logging.error("Cannot proceed with incomplete data.")
            sys.exit(1)

        if not local_parquet_files:
            logging.error("No data could be processed or found.")
            sys.exit(1)

        logging.info(f"Validated all {len(local_parquet_files)} required parquet files are present.")
        
        # Create DuckDB index from all local parquet files (existing and newly processed)
        create_duckdb_index(local_parquet_files, args.output_db_file, args.metric, embedding_dim=args.embedding_dim)

    except Exception as e:
        logging.error(f"An error occurred during the process: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()