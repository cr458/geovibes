import argparse
import logging
import pathlib
import time
import os
import tempfile
import shutil
import subprocess
import shlex
import duckdb
import geovibes.database.faiss_db as faiss_db
import faiss
import numpy as np
import pandas as pd
from tqdm import tqdm
import glob
import gc
import fsspec
from joblib import Parallel, delayed
from typing import Optional
import geovibes.tiling as tiling


def get_cloud_protocol(path: str) -> Optional[str]:
    """Returns 's3' or 'gs' if the path is a cloud path, otherwise None."""
    if path.startswith("s3://"):
        return "s3"
    if path.startswith("gs://"):
        return "gs"
    return None


def infer_embedding_dim_from_file(parquet_file: str, embedding_col: str) -> int:
    with duckdb.connect() as con_inf:
        row = con_inf.execute(
            f"SELECT {embedding_col} FROM read_parquet(?) WHERE {embedding_col} IS NOT NULL LIMIT 1;",
            (parquet_file,),
        ).fetchone()
        if not row or row[0] is None:
            raise IngestParquetError(f"Could not infer embedding dimension; column '{embedding_col}' has no non-null rows in {parquet_file}")
        arr = np.asarray(row[0])
        dim = int(arr.shape[0])
        logging.info(f"Inferred embedding_dim={dim} from first row of {parquet_file}")
        logging.info(f"Sample embedding values: {arr[:8].tolist()}")
        return dim



def list_cloud_parquet_files(cloud_path: str) -> list[str]:
    """List all parquet files in a cloud directory (GCS or S3)."""
    protocol = get_cloud_protocol(cloud_path)
    if not protocol:
        raise ValueError("Cloud path must start with 'gs://' or 's3://'")
    if not cloud_path.endswith('/'):
        cloud_path += '/'
    if protocol == "s3":
        endpoint = os.environ.get("S3_ENDPOINT_URL", "https://data.source.coop")
        fs = fsspec.filesystem("s3", client_kwargs={"endpoint_url": endpoint})
    else:
        fs = fsspec.filesystem(protocol)
    return [f"{protocol}://{p}" for p in fs.glob(cloud_path + "*.parquet")]



def _download_single_cloud_file(cloud_path: str, temp_dir: str) -> Optional[str]:
    """Download one cloud file to a temp directory and return local path."""
    protocol = get_cloud_protocol(cloud_path)
    if not protocol:
        logging.error(f"Invalid cloud path provided to worker: {cloud_path}")
        return None
    local_filename = os.path.join(temp_dir, os.path.basename(cloud_path))
    if os.path.exists(local_filename):
        return local_filename
    try:
        if protocol == "s3":
            endpoint = os.environ.get("S3_ENDPOINT_URL", "https://data.source.coop")
            fs = fsspec.filesystem("s3", client_kwargs={"endpoint_url": endpoint})
        else:
            fs = fsspec.filesystem(protocol)
        fs.get(cloud_path, local_filename)
    except Exception as e:
        logging.error(f"Failed to download {cloud_path}: {e}")
        return None
    return local_filename



def download_cloud_files(cloud_paths: list[str], temp_dir: str) -> list[str]:
    """Download parquet files from cloud to a temporary directory in parallel."""
    local_paths = Parallel(n_jobs=-1, prefer="threads", verbose=10)(
        delayed(_download_single_cloud_file)(cloud_path, temp_dir)
        for cloud_path in cloud_paths
    )
    return [path for path in local_paths if path is not None]



def find_embedding_files_for_mgrs_ids(mgrs_ids: list[str], embedding_dir: str) -> list[str]:
    """Find parquet files in embedding directory that contain the specified MGRS IDs."""
    found_files: list[str] = []
    if get_cloud_protocol(embedding_dir):
        try:
            all_parquet_files = list_cloud_parquet_files(embedding_dir)
            for mgrs_id in mgrs_ids:
                matching_files = [f for f in all_parquet_files if mgrs_id in os.path.basename(f)]
                found_files.extend(matching_files)
        except Exception as e:
            logging.error(f"Error listing cloud files: {e}")
            return []
    else:
        embedding_path = pathlib.Path(embedding_dir)
        if not embedding_path.exists():
            logging.error(f"Embedding directory does not exist: {embedding_dir}")
            return []
        for mgrs_id in mgrs_ids:
            patterns = [
                f"*{mgrs_id}*.parquet",
                f"{mgrs_id}_*.parquet",
                f"*_{mgrs_id}.parquet",
                f"*{mgrs_id}_embeddings.parquet",
            ]
            mgrs_files: list[str] = []
            for pattern in patterns:
                matches = list(embedding_path.glob(pattern))
                mgrs_files.extend([str(f) for f in matches])
            mgrs_files = list(set(mgrs_files))
            if mgrs_files:
                found_files.extend(mgrs_files)
    return list(set(found_files))

def setup_logging():
    """Configure basic logging settings."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('faiss_build.log')
        ]
    )

class IngestParquetError(Exception):
    """Exception raised for errors in the ingestion of Parquet files."""
    pass


def ingest_parquet_to_duckdb(parquet_files: list[str], db_path: str, dtype: str, embedding_col: str, mgrs_ids: Optional[list[str]] = None, roi_geometry_wkb: Optional[bytes] = None):
    """
    Ingests data from Parquet files into a DuckDB database.
    Args:
        parquet_files: List of paths to the input Parquet files.
        db_path: Path to the DuckDB database file to be created.
        embedding_dim: The dimension of the vector embeddings.
        dtype: The data type of the embeddings.
        embedding_col: The name of the embedding column in the Parquet files.
        mgrs_ids: Optional list of MGRS tile IDs; used for filtering when geometry is absent but tile_id is present.
        roi_geometry_wkb: Optional single ROI geometry as WKB bytes; if provided and geometry exists, rows are filtered by ST_Intersects.
    """
    if not parquet_files:
        raise IngestParquetError("No parquet files provided for ingestion.")
        
    logging.info(f"Starting ingestion of {len(parquet_files)} parquet files into {db_path}...")

    # --- Schema Verification Step ---
    logging.info("--- Verifying source schema from first Parquet file ---")
    try:
        with duckdb.connect() as con_check:
            df_schema = con_check.execute(f"DESCRIBE SELECT * FROM read_parquet(?);", (parquet_files[0],)).fetchdf()
            available_columns = set(df_schema['column_name'])
            
            # Log the data type of the embedding column
            embedding_col_info = df_schema[df_schema['column_name'] == embedding_col]
            if not embedding_col_info.empty:
                embedding_type = embedding_col_info['column_type'].iloc[0]
                logging.info(f"Detected {embedding_col} column type: {embedding_type}")
            else:
                logging.error(f"Could not find {embedding_col} column in the schema.")

    except Exception as e:
        raise IngestParquetError(f"Could not read schema from {parquet_files[0]}: {e}")
    # --- End Schema Verification ---

    # Inspect the schema of the first file to determine available columns
    try:
        with duckdb.connect() as con_check:
            df_schema = con_check.execute(f"DESCRIBE SELECT * FROM read_parquet(?);", (parquet_files[0],)).fetchdf()
            available_columns = set(df_schema['column_name'])
    except Exception as e:
        raise IngestParquetError(f"Could not read schema from {parquet_files[0]}: {e}")

    has_geometry = 'geometry' in available_columns

    try:
        embedding_dim = infer_embedding_dim_from_file(parquet_files[0], embedding_col)
    except Exception as e:
        raise IngestParquetError(str(e))
    
    # Use 'tile_id' if available, otherwise fall back to 'id'
    id_column_in_parquet = 'tile_id' if 'tile_id' in available_columns else 'id' if 'id' in available_columns else None
    if id_column_in_parquet:
        logging.info(f"Source ID column: '{id_column_in_parquet}'. Geometry column found: {has_geometry}.")
    else: 
        logging.info(f"No 'tile_id' or 'id' column found, will generate ids.")

    with duckdb.connect(database=db_path) as con:
        # Load spatial extension, which is required for creating a spatial index
        if has_geometry:
            logging.info("Installing and loading spatial extension for DuckDB.")
            con.execute("INSTALL spatial; LOAD spatial;")

        logging.info("Creating table 'geo_embeddings' in DuckDB.")
        
        # Check if table already exists
        table_exists = con.execute("""
            SELECT COUNT(*) 
            FROM information_schema.tables 
            WHERE table_name = 'geo_embeddings'
        """).fetchone()[0] > 0
        
        if table_exists:
            existing_count = con.execute("SELECT COUNT(*) FROM geo_embeddings").fetchone()[0]
            logging.warning(f"Table 'geo_embeddings' already exists with {existing_count} rows. Dropping and recreating...")
            con.execute("DROP TABLE geo_embeddings;")
            con.execute("DROP SEQUENCE IF EXISTS seq_geo_embeddings_id;")
        
        # Create a sequence for auto-incrementing IDs
        con.execute("CREATE SEQUENCE IF NOT EXISTS seq_geo_embeddings_id START 1;")
        
        # Define embedding column type based on dtype flag
        if dtype.upper() == 'INT8':
            embedding_sql_type = f"UTINYINT[{embedding_dim}]" # UTINYINT is DuckDB for uint8
        else:
            embedding_sql_type = f"FLOAT[{embedding_dim}]"

        # Create table with dynamic schema
        if id_column_in_parquet:
            create_sql = f"""
            CREATE TABLE geo_embeddings (
                id BIGINT PRIMARY KEY DEFAULT nextval('seq_geo_embeddings_id'),
                tile_id VARCHAR,
                embedding {embedding_sql_type}
                {', geometry GEOMETRY' if has_geometry else ''}
            );
            """
        else:
            create_sql = f"""
            CREATE TABLE geo_embeddings (
                id BIGINT PRIMARY KEY DEFAULT nextval('seq_geo_embeddings_id'),
                embedding {embedding_sql_type}
                {', geometry GEOMETRY' if has_geometry else ''}
            );
            """

        logging.info(f"Creating table 'geo_embeddings' in DuckDB with sql: {create_sql}")
        con.execute(create_sql)
        
        # Ingest all files in a single, efficient query
        try:
            sql_parquet_files_list_str = "['" + "', '".join(parquet_files) + "']"
            
            insert_columns = f"embedding{', geometry' if has_geometry else ''}"
            select_clause = f"{embedding_col}"
            if id_column_in_parquet:
                insert_columns = f"tile_id, {insert_columns}"
                select_clause = f"tile_id, {select_clause}"
            
            # The geometry column from GeoParquet is already understood by DuckDB's spatial extension.
            # No explicit cast is needed if the target column is type GEOMETRY.
            if has_geometry:
                select_clause += ", geometry"

                # Apply ROI filter only when WKB geometry is provided
                if roi_geometry_wkb is not None:
                    logging.info("Applying ROI geometry (WKB) filter during ingestion (ST_Intersects)...")
                    insert_sql = f"""
                        WITH roi AS (
                            SELECT ST_GeomFromWKB(?) AS geom
                        )
                        INSERT INTO geo_embeddings ({insert_columns})
                        SELECT
                            {select_clause}
                        FROM read_parquet({sql_parquet_files_list_str}, union_by_name=true) AS t, roi
                        WHERE ST_Intersects(t.geometry, roi.geom);
                    """
                    con.execute(insert_sql, [roi_geometry_wkb])
                else:
                    insert_sql = f"""
                        INSERT INTO geo_embeddings ({insert_columns})
                        SELECT
                            {select_clause}
                        FROM read_parquet({sql_parquet_files_list_str}, union_by_name=true);
                    """
                    logging.info("Inserting data into DuckDB without ROI filter.")
                    con.execute(insert_sql)
            else:
                # No geometry in source; optionally filter by tile_id using MGRS IDs if available
                where_clause = ""
                if id_column_in_parquet == 'tile_id' and mgrs_ids:
                    logging.info("Applying tile_id filter using MGRS IDs during ingestion (no geometry column available)...")
                    mgrs_list_sql = ", ".join([f"'{mid}'" for mid in mgrs_ids])
                    where_clause = f" WHERE tile_id IN ({mgrs_list_sql})"

                insert_sql = f"""
                    INSERT INTO geo_embeddings ({insert_columns})
                    SELECT
                        {select_clause}
                    FROM read_parquet({sql_parquet_files_list_str}, union_by_name=true){where_clause};
                """
                if where_clause:
                    logging.info("Inserting data into DuckDB with tile filter.")
                else:
                    logging.info("Inserting data into DuckDB without tile filter.")
                con.execute(insert_sql)
        except Exception as e:
            raise IngestParquetError(f"Failed to ingest parquet files: {e}. Please ensure all parquet files have a consistent schema.")

        row_count = con.execute("SELECT COUNT(*) FROM geo_embeddings;").fetchone()[0]
        logging.info(f"Successfully ingested {row_count} rows into DuckDB.")

        if has_geometry:
            logging.info("Creating R-Tree spatial index on geometry column...")
            con.execute("CREATE INDEX geom_spatial_idx ON geo_embeddings USING RTREE (geometry);")
            logging.info("Spatial index created successfully.")

        return embedding_dim

def create_faiss_index(db_path: str, index_path: str, embedding_dim: int, dtype: str, nlist: int, m: int, nbits: int, batch_size: int):
    """
    Creates a FAISS index from embeddings stored in a DuckDB database.

    Args:
        db_path: Path to the DuckDB database.
        index_path: Path to save the final FAISS index.
        embedding_dim: The dimension of the embeddings.
        nlist: The number of cells for the IVF index.
        m: The number of sub-quantizers for Product Quantization.
        nbits: The number of bits per sub-quantizer code.
        batch_size: The number of vectors to process at a time when populating the index.
    """
    logging.info("Starting FAISS index creation.")
    
    with duckdb.connect(database=db_path, read_only=True) as con:
        total_vectors = con.execute("SELECT COUNT(*) FROM geo_embeddings;").fetchone()[0]
        logging.info(f"Total vectors to index: {total_vectors}")

        if total_vectors == 0:
            logging.error("The 'geo_embeddings' table is empty after ingestion. Nothing to index. Aborting.")
            logging.error("This might happen if the input parquet files are empty or have an incompatible schema.")
            return

        # --- Branch logic based on dtype ---
        if dtype.upper() == 'FLOAT':
            logging.info("Building FAISS index for FLOAT data using IndexIVFPQ.")
            # Use IndexIVFPQ for floating point data
            quantizer = faiss.IndexFlatL2(embedding_dim)
            index = faiss.IndexIVFPQ(quantizer, embedding_dim, nlist, m, nbits)
            input_dtype = np.float32
        
        elif dtype.upper() == 'INT8':
            logging.info("Building FAISS index for INT8 data using IndexIVFScalarQuantizer.")
            # Use IndexIVFScalarQuantizer for quantized integer data
            quantizer = faiss.IndexFlatL2(embedding_dim) # The IVF quantizer still operates in float space
            index = faiss.IndexIVFScalarQuantizer(quantizer, embedding_dim, nlist, faiss.ScalarQuantizer.QT_8bit)
            input_dtype = np.uint8

        else:
            raise ValueError(f"Unsupported dtype for FAISS index: {dtype}")
        
        # 2. Train the index on a sample of the data
        logging.info("Training FAISS index...")
        train_sample_size = min(total_vectors, max(total_vectors // 10, 2000000))
        logging.info(f"Using {train_sample_size} vectors for training...")

        training_vectors_df = con.execute(f"SELECT embedding FROM geo_embeddings TABLESAMPLE RESERVOIR({train_sample_size} ROWS);").fetchdf()
        
        if training_vectors_df.empty:
            logging.error("Failed to sample training vectors from the database. Aborting.")
            return
            
        # Training always requires float32 vectors
        training_vectors = np.vstack(training_vectors_df['embedding'].values).astype(np.float32)

        start_time = time.time()
        index.train(training_vectors)
        logging.info(f"Index training completed in {time.time() - start_time:.2f} seconds.")

        del training_vectors
        gc.collect()

        # 3. Populate the index in batches
        logging.info("Populating FAISS index in batches...")
        num_batches = (total_vectors + batch_size - 1) // batch_size
        
        for i in tqdm(range(num_batches), desc="Populating FAISS Index"):
            offset = i * batch_size
            batch_df = con.execute(f"SELECT id, embedding FROM geo_embeddings ORDER BY id LIMIT {batch_size} OFFSET {offset};").fetchdf()
            
            ids = batch_df['id'].values.astype('int64')
            
            # For adding to the index, SQ needs float32, PQ can take float16 or float32.
            # We use float32 for both for simplicity here.
            vectors = np.vstack(batch_df['embedding'].values).astype(np.float32)

            index.add_with_ids(vectors, ids)
            
        logging.info("Index population complete.")
        
        # 4. Save the final index to disk
        logging.info(f"Writing index to {index_path}")
        faiss.write_index(index, index_path)
        logging.info("FAISS index successfully built and saved.")


def main():
    parser = argparse.ArgumentParser(description="Build a FAISS index from geospatial embeddings stored in Parquet files.")

    # argparse cannot include positional args in mutually exclusive groups.
    # Define both inputs and validate exclusivity manually below.
    parser.add_argument("parquet_files", nargs='*', help="Paths to input Parquet files or directories (local or gs:// or s3://)")
    parser.add_argument("--roi-file", dest="roi_file", help="ROI geometry file to intersect with MGRS tiles for automatic file discovery")

    parser.add_argument("--mgrs-reference-file", dest="mgrs_reference_file", help="MGRS reference file containing MGRS tile geometries (required with --roi-file)")
    parser.add_argument("--embedding-dir", dest="embedding_dir", help="Directory containing embedding parquet files (required with --roi-file)")
    parser.add_argument("--name", type=str, required=True, help="A descriptive name for the output files (e.g., 'bali_2024').")
    parser.add_argument("--output_dir", default=".", help="Directory to save the DuckDB and FAISS index files.")
    parser.add_argument("--embedding_col", type=str, default="embedding", help="Name of the embedding column in the Parquet files.")
    parser.add_argument("--dtype", type=str, default="FLOAT", choices=["FLOAT", "INT8"], help="Data type of the embeddings (FLOAT or INT8).")
    parser.add_argument("--nlist", type=int, default=4096, help="Number of clusters (IVF cells) for the FAISS index.")
    parser.add_argument("--m", type=int, default=64, help="Number of sub-quantizers for FAISS Product Quantization (used for FLOAT dtype).")
    parser.add_argument("--nbits", type=int, default=8, help="Number of bits per sub-quantizer code (used for FLOAT dtype).")
    parser.add_argument("--batch_size", type=int, default=500_000, help="Batch size for populating the FAISS index.")
    parser.add_argument("--dry-run", action="store_true", help="Run the script on a small subset of files to test the pipeline.")
    parser.add_argument("--dry-run-size", type=int, default=5, help="Number of files to use in a dry run.")
    
    args = parser.parse_args()
    
    setup_logging()
    
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Construct descriptive filenames
    faiss_params_str = f"faiss_{args.nlist}_{args.m}_{args.nbits}"
    db_filename = f"{args.name}_metadata.db"
    index_filename = f"{args.name}_{faiss_params_str}.index"

    db_path = str(output_dir / db_filename)
    index_path = str(output_dir / index_filename)
    
    # Validate mutual exclusivity and ROI-based arguments
    if getattr(args, "roi_file", None) and args.parquet_files:
        parser.error("Cannot use positional parquet_files with --roi-file; choose one input method.")
    if not getattr(args, "roi_file", None) and not args.parquet_files:
        parser.error("You must provide either parquet file paths or --roi-file.")
    if getattr(args, "roi_file", None):
        if not getattr(args, "mgrs_reference_file", None):
            parser.error("--mgrs-reference-file is required when using --roi-file")
        if not getattr(args, "embedding_dir", None):
            parser.error("--embedding-dir is required when using --roi-file")

    # --- File Discovery (local and cloud, optional ROI) ---
    all_parquet_files: list[str] = []
    cloud_files: list[str] = []
    mgrs_ids: Optional[list[str]] = None
    roi_wkb: Optional[bytes] = None

    if getattr(args, "roi_file", None):
        logging.info("Using ROI-based file discovery")
        mgrs_tile_ids = tiling.get_mgrs_tile_ids_for_roi_from_roi_file(
            roi_geojson_file=args.roi_file,
            mgrs_tiles_file=args.mgrs_reference_file,
        )
        mgrs_ids = [str(t) for t in mgrs_tile_ids]
        if not mgrs_ids:
            logging.error("No MGRS tiles found intersecting with ROI")
            return
        # Derive a single ROI geometry WKB once (to pass into ingestion for row-level filtering)
        try:
            with duckdb.connect() as con_roi:
                con_roi.execute("INSTALL spatial; LOAD spatial;")
                row = con_roi.execute(
                    """
                    WITH roi AS (
                        SELECT ST_Union_Agg(geom) AS geom
                        FROM ST_Read(?)
                    )
                    SELECT ST_AsWKB(geom) FROM roi;
                    """,
                    [args.roi_file],
                ).fetchone()
                if row and row[0] is not None:
                    roi_wkb = bytes(row[0]) if not isinstance(row[0], (bytes, bytearray)) else row[0]
        except Exception as e:
            logging.error(f"Failed to derive ROI WKB from file {args.roi_file}: {e}")
        embedding_files = find_embedding_files_for_mgrs_ids(mgrs_ids, args.embedding_dir)
        if not embedding_files:
            logging.error("No embedding files found for intersecting MGRS tiles")
            return
        for file_path in embedding_files:
            if get_cloud_protocol(file_path):
                cloud_files.append(file_path)
            else:
                all_parquet_files.append(file_path)
        logging.info(f"Found {len(embedding_files)} embedding files for ROI ({len(all_parquet_files)} local, {len(cloud_files)} cloud)")
    else:
        logging.info("Using explicit parquet file paths")
        parquet_inputs = getattr(args, "parquet_files", []) or []
        for path in parquet_inputs:
            if get_cloud_protocol(path):
                if path.endswith('.parquet'):
                    cloud_files.append(path)
                else:
                    cloud_parquet_files = list_cloud_parquet_files(path)
                    cloud_files.extend(cloud_parquet_files)
                    logging.info(f"Found {len(cloud_parquet_files)} parquet files in {path}")
            else:
                path_obj = pathlib.Path(path)
                if path_obj.is_file() and path_obj.suffix == '.parquet':
                    all_parquet_files.append(str(path_obj))
                elif path_obj.is_dir():
                    all_parquet_files.extend(glob.glob(f"{str(path_obj)}/**/*.parquet", recursive=True))
                else:
                    all_parquet_files.extend(glob.glob(path))

    temp_dir = None
    if cloud_files:
        temp_dir = tempfile.mkdtemp(prefix="faiss_db_")
        logging.info(f"Downloading {len(cloud_files)} files from cloud to temporary directory...")
        local_cloud_files = download_cloud_files(cloud_files, temp_dir)
        all_parquet_files.extend(local_cloud_files)

    # Deduplicate and validate
    all_parquet_files = list(dict.fromkeys(all_parquet_files))

    if not all_parquet_files:
        logging.error("No parquet files found!")
        if temp_dir:
            shutil.rmtree(temp_dir)
        return

    logging.info(f"Total {len(all_parquet_files)} parquet files to process")

    files_to_process = all_parquet_files
    if args.dry_run:
        logging.info(f"--- Starting DRY RUN using the first {args.dry_run_size} files. ---")
        files_to_process = files_to_process[: args.dry_run_size]

    if not files_to_process:
        logging.error("No files left to process after applying dry-run filter.")
        if temp_dir:
            shutil.rmtree(temp_dir)
        return
    
    start_total_time = time.time()

    # --- Phase 1: Ingest data into DuckDB ---
    inferred_dim = ingest_parquet_to_duckdb(
        files_to_process,
        db_path,
        args.dtype,
        args.embedding_col,
        mgrs_ids=mgrs_ids,
        roi_geometry_wkb=roi_wkb,
    )
    logging.info(f"Using inferred embedding_dim={inferred_dim} for index creation")

    # --- Phase 2: Build FAISS index ---
    create_faiss_index(db_path, index_path, inferred_dim, args.dtype, args.nlist, args.m, args.nbits, args.batch_size)
    
    logging.info(f"Total process completed in {time.time() - start_total_time:.2f} seconds.")
    logging.info(f"Artifacts created:\n- DuckDB: {db_path}\n- FAISS Index: {index_path}")

    if temp_dir:
        shutil.rmtree(temp_dir)
        logging.info(f"Cleaned up temporary directory: {temp_dir}")

    # Create compressed tarball and upload to output directory
    try:
        tar_name = f"{args.name}.tar.gz"
        is_s3_output = str(output_dir).startswith("s3://")
        tar_local_dir = tempfile.mkdtemp(prefix="faiss_tar_") if is_s3_output else str(output_dir)
        tar_local_path = os.path.join(tar_local_dir, tar_name)

        cmd = f"tar -c {shlex.quote(db_path)} {shlex.quote(index_path)} | pigz > {shlex.quote(tar_local_path)}"
        try:
            subprocess.run(["bash", "-lc", cmd], check=True)
            logging.info(f"Created tarball with pigz: {tar_local_path}")
        except Exception as e:
            fallback_cmd = f"tar -czf {shlex.quote(tar_local_path)} {shlex.quote(db_path)} {shlex.quote(index_path)}"
            subprocess.run(["bash", "-lc", fallback_cmd], check=True)
            logging.info(f"Created tarball with gzip fallback: {tar_local_path}")

        if is_s3_output:
            endpoint = os.environ.get("S3_ENDPOINT_URL", "https://data.source.coop")
            fs = fsspec.filesystem("s3", client_kwargs={"endpoint_url": endpoint})
            s3_dest = str(output_dir).rstrip("/") + "/" + tar_name
            fs.put(tar_local_path, s3_dest)
            logging.info(f"Uploaded tarball to {s3_dest}")
            shutil.rmtree(tar_local_dir, ignore_errors=True)
        else:
            logging.info(f"Tarball available at {tar_local_path}")
    except Exception as e:
        logging.error(f"Failed to create/upload tarball: {e}")

if __name__ == "__main__":
    main() 