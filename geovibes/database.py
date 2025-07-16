import argparse
import logging
import pathlib
import time
from typing import List, Optional
import os
import tempfile
import shutil
import glob

import duckdb
import pandas as pd
import geopandas as gpd
import numpy as np
from tqdm import tqdm
import fsspec
from joblib import Parallel, delayed


def setup_logging():
    """Configure basic logging settings."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def get_mgrs_ids_from_roi(roi_file: str, mgrs_reference_file: str) -> List[str]:
    """
    Find MGRS tile IDs that overlap with the region of interest.
    
    Args:
        roi_file: Path to ROI geometry file (geojson/shapefile/geoparquet)
        mgrs_reference_file: Path to MGRS reference file containing MGRS tile geometries
        
    Returns:
        List of MGRS tile IDs that intersect with the ROI
    """
    logging.info(f"Loading ROI from: {roi_file}")
    roi_gdf = gpd.read_file(roi_file)
    
    logging.info(f"Loading MGRS reference from: {mgrs_reference_file}")
    if mgrs_reference_file.endswith('.parquet'):
        mgrs_gdf = gpd.read_parquet(mgrs_reference_file)
    else:
        mgrs_gdf = gpd.read_file(mgrs_reference_file)
    
    # Ensure both are in the same CRS
    if roi_gdf.crs is not None and mgrs_gdf.crs is not None and roi_gdf.crs != mgrs_gdf.crs:
        logging.info(f"Converting ROI from {roi_gdf.crs} to {mgrs_gdf.crs}")
        roi_gdf = roi_gdf.to_crs(mgrs_gdf.crs)
    elif roi_gdf.crs is None:
        logging.warning("ROI file has no CRS defined")
    elif mgrs_gdf.crs is None:
        logging.warning("MGRS reference file has no CRS defined")
    
    # Get union of all ROI geometries
    roi_union = roi_gdf.unary_union
    
    # Find intersecting MGRS tiles
    intersecting = mgrs_gdf[mgrs_gdf.intersects(roi_union)]
    
    # Try common MGRS ID column names
    mgrs_id_columns = ['mgrs_id', 'MGRS', 'mgrs', 'tile_id', 'TILE_ID', 'id', 'ID']
    mgrs_id_column = None
    
    for col in mgrs_id_columns:
        if col in intersecting.columns:
            mgrs_id_column = col
            break
    
    if mgrs_id_column is None:
        raise ValueError(f"No MGRS ID column found. Available columns: {list(intersecting.columns)}")
    
    mgrs_ids = intersecting[mgrs_id_column].tolist()
    logging.info(f"Found {len(mgrs_ids)} MGRS tiles intersecting with ROI: {mgrs_ids}")
    
    return mgrs_ids


def find_embedding_files_for_mgrs_ids(mgrs_ids: List[str], embedding_dir: str) -> List[str]:
    """
    Find parquet files in embedding directory that contain the specified MGRS IDs.
    
    Args:
        mgrs_ids: List of MGRS tile IDs to search for
        embedding_dir: Directory containing embedding parquet files
        
    Returns:
        List of parquet file paths that match the MGRS IDs
    """
    logging.info(f"Searching for embedding files for {len(mgrs_ids)} MGRS IDs in: {embedding_dir}")
    
    found_files = []
    
    # Check if it's a cloud path
    if get_cloud_protocol(embedding_dir):
        # For cloud storage, list all parquet files and filter
        try:
            all_parquet_files = list_cloud_parquet_files(embedding_dir)
            for mgrs_id in mgrs_ids:
                matching_files = [f for f in all_parquet_files if mgrs_id in os.path.basename(f)]
                found_files.extend(matching_files)
                if matching_files:
                    logging.info(f"Found {len(matching_files)} files for MGRS {mgrs_id}")
                else:
                    logging.warning(f"No files found for MGRS {mgrs_id}")
        except Exception as e:
            logging.error(f"Error listing cloud files: {e}")
            return []
    else:
        # For local paths, use glob patterns
        embedding_path = pathlib.Path(embedding_dir)
        if not embedding_path.exists():
            logging.error(f"Embedding directory does not exist: {embedding_dir}")
            return []
        
        for mgrs_id in mgrs_ids:
            # Try multiple patterns to find files containing the MGRS ID
            patterns = [
                f"*{mgrs_id}*.parquet",
                f"{mgrs_id}_*.parquet",
                f"*_{mgrs_id}.parquet",
                f"*{mgrs_id}_embeddings.parquet"
            ]
            
            mgrs_files = []
            for pattern in patterns:
                matches = list(embedding_path.glob(pattern))
                mgrs_files.extend([str(f) for f in matches])
            
            # Remove duplicates
            mgrs_files = list(set(mgrs_files))
            
            if mgrs_files:
                found_files.extend(mgrs_files)
                logging.info(f"Found {len(mgrs_files)} files for MGRS {mgrs_id}: {[os.path.basename(f) for f in mgrs_files]}")
            else:
                logging.warning(f"No embedding files found for MGRS {mgrs_id}")
    
    # Remove duplicates from final list
    found_files = list(set(found_files))
    
    logging.info(f"Total embedding files found: {len(found_files)}")
    if len(found_files) == 0:
        logging.error("No embedding files found for any MGRS IDs!")
        logging.info(f"Searched MGRS IDs: {mgrs_ids}")
        if not get_cloud_protocol(embedding_dir):
            # List some example files for debugging
            example_files = list(pathlib.Path(embedding_dir).glob("*.parquet"))[:10]
            logging.info(f"Example files in directory: {[f.name for f in example_files]}")
    
    return found_files


def check_and_clean_embeddings(parquet_files: List[str]) -> List[str]:
    """Check parquet files for NaN embeddings and create cleaned versions."""
    
    logging.info("Checking parquet files for NaN values in embeddings...")
    
    def process_file(parquet_file):
        """Process a single parquet file for NaN values."""
        try:
            # Try reading as regular parquet first (faster)
            try:
                df = pd.read_parquet(parquet_file)
                # If it has geometry-related columns, try to read as GeoDataFrame
                if any(col in df.columns for col in ['geometry', 'geometry_wkt', 'geom', 'wkt']):
                    try:
                        df = gpd.read_parquet(parquet_file)
                    except:
                        # Keep as regular dataframe if geopandas fails
                        pass
            except Exception as e:
                logging.error(f"Failed to read {parquet_file}: {e}")
                return None, 0, None
                
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
                    
                    # Remove rows with NaN embeddings
                    df_clean = df[~nan_mask].copy()
                    
                    # Save cleaned version
                    clean_filename = parquet_file.replace('.parquet', '_clean.parquet')
                    if isinstance(df_clean, gpd.GeoDataFrame):
                        df_clean.to_parquet(clean_filename, index=False)
                    else:
                        # Just save as regular parquet
                        df_clean.to_parquet(clean_filename, index=False)
                    
                    logging.info(f"Cleaned {pathlib.Path(parquet_file).name}: {original_count} -> {len(df_clean)} rows")
                    return clean_filename, nan_count, pathlib.Path(parquet_file).name
                else:
                    # No NaN values, use original file
                    return parquet_file, 0, None
            else:
                logging.warning(f"No 'embedding' column found in {pathlib.Path(parquet_file).name}")
                return parquet_file, 0, None
                
        except Exception as e:
            logging.error(f"Error processing {parquet_file}: {e}")
            return None, 0, None
    
    # Process files in parallel
    from joblib import Parallel, delayed
    results = Parallel(n_jobs=8)(
        delayed(process_file)(parquet_file) 
        for parquet_file in tqdm(parquet_files, desc="Checking parquet files")
    )
    
    # Collect results
    cleaned_files = []
    total_nan_count = 0
    files_with_nans = []
    
    for result in results:
        if result[0] is not None:  # Successfully processed
            cleaned_files.append(result[0])
            total_nan_count += result[1]
            if result[2] is not None:  # Had NaN values
                files_with_nans.append(result[2])
    
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
    metric: str = "cosine",
    embedding_column: str = "embedding",
    id_column: str = "id",
    geometry_column: Optional[str] = None,
    embedding_dim: Optional[int] = None,
    skip_nan_check: bool = False
) -> None:
    """Creates a DuckDB database with HNSW and spatial indexes from parquet files.
    
    Args:
        parquet_files: List of parquet file paths
        output_file: Output database file path
        metric: Distance metric for HNSW index
        embedding_column: Name of the embedding column
        id_column: Name of the ID column
        geometry_column: Name of the geometry column (auto-detected if None)
        embedding_dim: Dimension of embeddings (auto-detected if None)
        skip_nan_check: Skip NaN checking and cleaning
    """
    
    if not skip_nan_check:
        cleaned_parquet_files = check_and_clean_embeddings(parquet_files)
    else:
        cleaned_parquet_files = parquet_files
    
    # Auto-detect embedding dimension if not provided
    if embedding_dim is None:
        logging.info("Auto-detecting embedding dimension...")
        sample_df = pd.read_parquet(cleaned_parquet_files[0], columns=[embedding_column])
        if embedding_column in sample_df.columns and len(sample_df) > 0:
            first_embedding = sample_df[embedding_column].iloc[0]
            if isinstance(first_embedding, list):
                embedding_dim = len(first_embedding)
            else:
                embedding_dim = len(first_embedding)
            logging.info(f"Detected embedding dimension: {embedding_dim}")
        else:
            raise ValueError(f"Could not detect embedding dimension from column '{embedding_column}'")
    
    # Auto-detect geometry column if not specified
    if geometry_column is None:
        logging.info("Auto-detecting geometry column...")
        sample_df = pd.read_parquet(cleaned_parquet_files[0])
        geometry_candidates = ['geometry', 'geometry_wkt', 'geom', 'wkt', 'geo', 'shape']
        for col in geometry_candidates:
            if col in sample_df.columns:
                geometry_column = col
                logging.info(f"Detected geometry column: {geometry_column}")
                break
        
        if geometry_column is None:
            logging.error("CRITICAL: No geometry column found in parquet files!")
            logging.error(f"Checked for columns: {geometry_candidates}")
            logging.error(f"Available columns: {list(sample_df.columns)}")
            raise ValueError("Geometry column is required for GeoVibes databases. "
                           "Please specify --geometry-column or ensure your parquet files contain a geometry column.")
    
    logging.info(f"Creating DuckDB database at {output_file} with {metric} metric...")
    logging.info(f"Using embedding dimension: {embedding_dim}")
    logging.info(f"ID column: {id_column}, Geometry column: {geometry_column}")

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

    # Build the CREATE TABLE SQL - geometry is always required
    # Check if it's a WKT column that needs conversion
    sample_df = pd.read_parquet(cleaned_parquet_files[0])
    is_wkt_column = geometry_column in ['geometry_wkt', 'wkt', 'geo_wkt'] or \
                   (geometry_column in sample_df.columns and 
                    len(sample_df) > 0 and
                    isinstance(sample_df[geometry_column].iloc[0], str) and 
                    sample_df[geometry_column].iloc[0].startswith(('POINT', 'POLYGON', 'LINESTRING')))
    
    if is_wkt_column:
        # Need to convert WKT to geometry
        create_table_sql = f"""
        CREATE OR REPLACE TABLE geo_embeddings AS
        SELECT
            {id_column} AS id,
            CAST({embedding_column} AS FLOAT[{embedding_dim}]) as embedding,
            ST_GeomFromText({geometry_column}) as geometry
        FROM read_parquet({sql_parquet_files_list_str}, union_by_name=true);
        """
    else:
        # Direct geometry column
        create_table_sql = f"""
        CREATE OR REPLACE TABLE geo_embeddings AS
        SELECT
            {id_column} AS id,
            CAST({embedding_column} AS FLOAT[{embedding_dim}]) as embedding,
            {geometry_column} as geometry
        FROM read_parquet({sql_parquet_files_list_str}, union_by_name=true);
        """

    logging.info("Creating table and ingesting data...")
    start_time = time.time()
    con.execute(create_table_sql)
    ingest_time = time.time() - start_time

    row_count_result = con.execute("SELECT COUNT(*) FROM geo_embeddings;").fetchone()
    row_count = row_count_result[0] if row_count_result else 0
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


def get_cloud_protocol(path: str) -> Optional[str]:
    """Returns 's3' or 'gs' if the path is a cloud path, otherwise None."""
    if path.startswith("s3://"):
        return "s3"
    if path.startswith("gs://"):
        return "gs"
    return None


def upload_to_cloud(local_file: str, cloud_path: str) -> None:
    """Upload a file to a cloud storage location (GCS or S3)."""
    protocol = get_cloud_protocol(cloud_path)
    if not protocol:
        raise ValueError("Cloud path must start with 'gs://' or 's3://'")
    
    logging.info(f"Uploading {local_file} to {cloud_path}...")
    fs = fsspec.filesystem(protocol)
    fs.put(local_file, cloud_path)
    logging.info(f"Upload complete: {cloud_path}")


def _download_single_cloud_file(cloud_path: str, temp_dir: str) -> Optional[str]:
    """Helper function to download a single file from cloud for parallel execution."""
    try:
        protocol = get_cloud_protocol(cloud_path)
        if not protocol:
            logging.error(f"Invalid cloud path provided to worker: {cloud_path}")
            return None
        
        local_filename = os.path.join(temp_dir, os.path.basename(cloud_path))

        if os.path.exists(local_filename):
            return local_filename

        fs = fsspec.filesystem(protocol)
        fs.get(cloud_path, local_filename)
        return local_filename
    except Exception as e:
        logging.error(f"Failed to download {cloud_path} in worker: {e}")
        return None


def download_cloud_files(cloud_paths: List[str], temp_dir: str) -> List[str]:
    """Download parquet files from cloud to a temporary directory in parallel."""
    logging.info(f"Downloading {len(cloud_paths)} files in parallel using joblib...")
    
    local_paths = Parallel(n_jobs=-1)(
        delayed(_download_single_cloud_file)(cloud_path, temp_dir)
        for cloud_path in tqdm(cloud_paths, desc="Queueing cloud downloads")
    )
    
    successful_downloads = [path for path in local_paths if path is not None]
    
    if len(successful_downloads) != len(cloud_paths):
        logging.warning(
            f"Failed to download {len(cloud_paths) - len(successful_downloads)} files. "
            "Check logs for errors."
        )

    return successful_downloads


def list_cloud_parquet_files(cloud_path: str) -> List[str]:
    """List all parquet files in a cloud directory (GCS or S3)."""
    protocol = get_cloud_protocol(cloud_path)
    if not protocol:
        raise ValueError("Cloud path must start with 'gs://' or 's3://'")

    if not cloud_path.endswith('/'):
        cloud_path += '/'
    
    fs = fsspec.filesystem(protocol)
    full_paths = [f"{protocol}://{p}" for p in fs.glob(cloud_path + "*.parquet")]
    return full_paths


def main():
    parser = argparse.ArgumentParser(description="Create a DuckDB database with HNSW and spatial indexes from parquet files.")
    
    # Main input arguments - either parquet files OR roi-based discovery
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("parquet_files", nargs='*', help="Paths to parquet files or directories (local or gs:// or s3://)")
    input_group.add_argument("--roi-file", help="ROI geometry file to intersect with MGRS tiles for automatic file discovery")
    
    parser.add_argument("--output_path", help="Output path for the DuckDB database (local path or gs:// or s3:// path)")
    
    # ROI-based discovery arguments
    parser.add_argument("--mgrs-reference-file", help="MGRS reference file containing MGRS tile geometries (required with --roi-file)")
    parser.add_argument("--embedding-dir", help="Directory containing embedding parquet files (required with --roi-file)")
    
    # Database creation options
    parser.add_argument("--metric", default="cosine", choices=["cosine", "l2sq", "inner_product"], help="Distance metric for HNSW index")
    parser.add_argument("--embedding-column", default="embedding", help="Name of the embedding column")
    parser.add_argument("--id-column", default="id", help="Name of the ID column")
    parser.add_argument("--geometry-column", help="Name of the geometry column (auto-detected if not specified, required for database creation)")
    parser.add_argument("--embedding-dim", type=int, help="Dimension of embeddings (auto-detected if not specified)")
    parser.add_argument("--skip-nan-check", action="store_true", help="Skip NaN checking and cleaning")
    
    args = parser.parse_args()
    
    setup_logging()
    
    # Validate ROI-based arguments
    if args.roi_file:
        if not args.mgrs_reference_file:
            parser.error("--mgrs-reference-file is required when using --roi-file")
        if not args.embedding_dir:
            parser.error("--embedding-dir is required when using --roi-file")
    
    all_parquet_files = []
    cloud_files = []
    
    # Determine input files
    if args.roi_file:
        logging.info("Using ROI-based file discovery")
        
        # Find MGRS IDs that overlap with ROI
        mgrs_ids = get_mgrs_ids_from_roi(args.roi_file, args.mgrs_reference_file)
        
        if not mgrs_ids:
            logging.error("No MGRS tiles found intersecting with ROI")
            return
        
        # Find embedding files for those MGRS IDs
        embedding_files = find_embedding_files_for_mgrs_ids(mgrs_ids, args.embedding_dir)
        
        if not embedding_files:
            logging.error("No embedding files found for intersecting MGRS tiles")
            return
        
        # Add found files to processing lists
        for file_path in embedding_files:
            if get_cloud_protocol(file_path):
                cloud_files.append(file_path)
            else:
                all_parquet_files.append(file_path)
        
        logging.info(f"Found {len(embedding_files)} embedding files for ROI ({len(all_parquet_files)} local, {len(cloud_files)} cloud)")
    
    else:
        logging.info("Using explicit parquet file paths")
        # Use the original parquet file discovery logic
        parquet_files = args.parquet_files or []
        
        for path in parquet_files:
            if get_cloud_protocol(path):
                if path.endswith('.parquet'):
                    cloud_files.append(path)
                else:
                    cloud_parquet_files = list_cloud_parquet_files(path)
                    cloud_files.extend(cloud_parquet_files)
                    logging.info(f"Found {len(cloud_parquet_files)} parquet files in {path}")
            else:
                # Handle local paths
                path_obj = pathlib.Path(path)
                if path_obj.is_file() and path_obj.suffix == '.parquet':
                    all_parquet_files.append(str(path_obj))
                elif path_obj.is_dir():
                    # Find all parquet files in directory
                    parquet_files_in_dir = list(path_obj.glob("*.parquet"))
                    all_parquet_files.extend([str(f) for f in parquet_files_in_dir])
                else:
                    logging.warning(f"Skipping {path} - not a parquet file or directory")
    
    # Download cloud files if any
    temp_dir = None
    if cloud_files:
        temp_dir = tempfile.mkdtemp(prefix="geovibes_db_")
        logging.info(f"Downloading {len(cloud_files)} files from cloud to temporary directory...")
        local_cloud_files = download_cloud_files(cloud_files, temp_dir)
        all_parquet_files.extend(local_cloud_files)
    
    if not all_parquet_files:
        logging.error("No parquet files found!")
        if temp_dir:
            shutil.rmtree(temp_dir)
        return
    
    logging.info(f"Total {len(all_parquet_files)} parquet files to process")
    
    # Determine if output is cloud or local
    if get_cloud_protocol(args.output_path):
        # Create local temp file first
        local_temp_file = "temp_database.db"
        create_duckdb_index(
            all_parquet_files,
            local_temp_file,
            metric=args.metric,
            embedding_column=args.embedding_column,
            id_column=args.id_column,
            geometry_column=args.geometry_column,
            embedding_dim=args.embedding_dim,
            skip_nan_check=args.skip_nan_check
        )
        
        # Upload to cloud
        upload_to_cloud(local_temp_file, args.output_path)
        
        # Clean up temp file
        os.remove(local_temp_file)
        logging.info(f"Cleaned up temporary file: {local_temp_file}")
    else:
        # Direct local output
        create_duckdb_index(
            all_parquet_files,
            args.output_path,
            metric=args.metric,
            embedding_column=args.embedding_column,
            id_column=args.id_column,
            geometry_column=args.geometry_column,
            embedding_dim=args.embedding_dim,
            skip_nan_check=args.skip_nan_check
        )
    
    # Clean up temporary directory if used
    if temp_dir:
        shutil.rmtree(temp_dir)
        logging.info(f"Cleaned up temporary directory: {temp_dir}")


if __name__ == "__main__":
    main()