import argparse
import logging
import pathlib
import time
import duckdb
import geovibes.database.faiss_db as faiss_db
import numpy as np
import pandas as pd
from tqdm import tqdm
import glob
import gc

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


def ingest_parquet_to_duckdb(parquet_files: list[str], db_path: str, embedding_dim: int, dtype: str, embedding_col: str):
    """
    Ingests data from Parquet files into a DuckDB database.
    Args:
        parquet_files: List of paths to the input Parquet files.
        db_path: Path to the DuckDB database file to be created.
        embedding_dim: The dimension of the vector embeddings.
        dtype: The data type of the embeddings.
        embedding_col: The name of the embedding column in the Parquet files.
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
                select_clause += """,
                    CASE
                        WHEN ST_GeometryType(geometry) = 'POINT' THEN geometry
                        ELSE st_centroid(geometry)
                    END as geometry
                """

            insert_sql = f"""
                INSERT INTO geo_embeddings ({insert_columns})
                SELECT
                    {select_clause}
                FROM read_parquet({sql_parquet_files_list_str}, union_by_name=true);
            """

            logging.info(f"Inserting data into DuckDB with sql: {insert_sql}")
            con.execute(insert_sql)
        except Exception as e:
            raise IngestParquetError(f"Failed to ingest parquet files: {e}. Please ensure all parquet files have a consistent schema.")

        row_count = con.execute("SELECT COUNT(*) FROM geo_embeddings;").fetchone()[0]
        logging.info(f"Successfully ingested {row_count} rows into DuckDB.")

        if has_geometry:
            logging.info("Creating R-Tree spatial index on geometry column...")
            con.execute("CREATE INDEX geom_spatial_idx ON geo_embeddings USING RTREE (geometry);")
            logging.info("Spatial index created successfully.")


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
            quantizer = faiss_db.IndexFlatL2(embedding_dim)
            index = faiss_db.IndexIVFPQ(quantizer, embedding_dim, nlist, m, nbits)
            input_dtype = np.float32
        
        elif dtype.upper() == 'INT8':
            logging.info("Building FAISS index for INT8 data using IndexIVFScalarQuantizer.")
            # Use IndexIVFScalarQuantizer for quantized integer data
            quantizer = faiss_db.IndexFlatL2(embedding_dim) # The IVF quantizer still operates in float space
            index = faiss_db.IndexIVFScalarQuantizer(quantizer, embedding_dim, nlist, faiss_db.ScalarQuantizer.QT_8bit)
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
        faiss_db.write_index(index, index_path)
        logging.info("FAISS index successfully built and saved.")


def main():
    parser = argparse.ArgumentParser(description="Build a FAISS index from geospatial embeddings stored in Parquet files.")
    parser.add_argument("parquet_files", nargs='+', help="Paths to input Parquet files or a glob pattern.")
    parser.add_argument("--name", type=str, required=True, help="A descriptive name for the output files (e.g., 'bali_2024').")
    parser.add_argument("--output_dir", default=".", help="Directory to save the DuckDB and FAISS index files.")
    parser.add_argument("--embedding_dim", type=int, default=384, help="Dimension of the embedding vectors.")
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

    # --- File Discovery ---
    all_files = []
    for path_pattern in args.parquet_files:
        # Check if it's a directory
        if pathlib.Path(path_pattern).is_dir():
            all_files.extend(glob.glob(f"{path_pattern}/**/*.parquet", recursive=True))
        else:
            # Assume it's a glob pattern or a single file
            all_files.extend(glob.glob(path_pattern))
            
    if not all_files:
        logging.error(f"No parquet files found for the given paths: {args.parquet_files}")
        return

    files_to_process = all_files
    if args.dry_run:
        logging.info(f"--- Starting DRY RUN using the first {args.dry_run_size} files. ---")
        files_to_process = files_to_process[:args.dry_run_size]

    if not files_to_process:
        logging.error("No files left to process after applying dry-run filter.")
        return
    
    start_total_time = time.time()

    # --- Phase 1: Ingest data into DuckDB ---
    ingest_parquet_to_duckdb(files_to_process, db_path, args.embedding_dim, args.dtype, args.embedding_col)

    # --- Phase 2: Build FAISS index ---
    create_faiss_index(db_path, index_path, args.embedding_dim, args.dtype, args.nlist, args.m, args.nbits, args.batch_size)
    
    logging.info(f"Total process completed in {time.time() - start_total_time:.2f} seconds.")
    logging.info(f"Artifacts created:\n- DuckDB: {db_path}\n- FAISS Index: {index_path}")

if __name__ == "__main__":
    main() 