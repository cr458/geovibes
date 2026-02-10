import argparse
import hashlib
import logging
import pathlib
import time
import os
import tempfile
import shutil
import subprocess
import shlex
import duckdb
import faiss
import numpy as np
from tqdm import tqdm
import glob
import gc
import fsspec
from typing import Optional
import geovibes.tiling as tiling
from geovibes.database.cloud import (
    download_cloud_files,
    find_embedding_files_for_mgrs_ids,
    get_cloud_protocol,
    list_cloud_parquet_files,
)


def ensure_duckdb_extension(con: duckdb.DuckDBPyConnection, name: str) -> None:
    """Install and load the requested DuckDB extension."""
    con.execute(f"INSTALL {name};")
    con.execute(f"LOAD {name};")


def setup_logging():
    """Configure basic logging settings."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler("faiss_build.log")],
    )


class IngestParquetError(Exception):
    """Exception raised for errors in the ingestion of Parquet files."""

    pass


def _parse_fixed_array_dim(type_str: str) -> Optional[int]:
    value = str(type_str).strip().upper()
    if "[" not in value or not value.endswith("]"):
        return None
    try:
        return int(value.split("[", 1)[1][:-1])
    except ValueError:
        return None


def _validate_httpfs_artifact_layout(
    con: duckdb.DuckDBPyConnection,
    *,
    embedding_dim: int,
    dtype: str,
) -> None:
    """Fail-fast guardrails for remote/httpfs retrieval artifacts."""
    row = con.execute("SELECT typeof(embedding) FROM geo_embeddings LIMIT 1").fetchone()
    if not row or row[0] is None:
        raise IngestParquetError("Could not detect embedding type from geo_embeddings")

    embedding_type = str(row[0]).upper()
    actual_dim = _parse_fixed_array_dim(embedding_type)
    expected_prefix = "UTINYINT" if dtype.upper() == "INT8" else "FLOAT"
    if actual_dim != int(embedding_dim) or not embedding_type.startswith(
        f"{expected_prefix}["
    ):
        raise IngestParquetError(
            f"Unexpected embedding type '{embedding_type}'. "
            f"Expected fixed-size {expected_prefix}[{embedding_dim}]"
        )

    index_rows = con.execute(
        "SELECT index_name FROM duckdb_indexes() WHERE table_name='geo_embeddings'"
    ).fetchall()
    index_names = {str(r[0]) for r in index_rows if r and r[0]}
    if "id_idx" not in index_names:
        raise IngestParquetError("Missing required id index 'id_idx' on geo_embeddings(id)")

    try:
        storage_rows = con.execute("PRAGMA storage_info('geo_embeddings')").fetchall()
        row_groups = {int(r[0]) for r in storage_rows if r and r[0] is not None}
        if not row_groups:
            raise IngestParquetError("No row groups discovered in geo_embeddings")
        logging.info(
            "Artifact layout validation passed: embedding=%s, row_groups=%d",
            embedding_type,
            len(row_groups),
        )
    except IngestParquetError:
        raise
    except Exception as exc:
        logging.warning(f"Could not inspect row-group layout: {exc}")


def infer_embedding_dim_from_file(parquet_file: str, embedding_col: str) -> int:
    with duckdb.connect() as con_inf:
        ensure_duckdb_extension(con_inf, "httpfs")
        row = con_inf.execute(
            f"""
            SELECT {embedding_col}
            FROM read_parquet(?, union_by_name=true)
            WHERE {embedding_col} IS NOT NULL
            LIMIT 1
            """,
            (parquet_file,),
        ).fetchone()

    if not row or row[0] is None:
        raise IngestParquetError(
            f"Could not infer embedding dimension; column '{embedding_col}' has no non-null rows in {parquet_file}"
        )

    embedding_value = row[0]
    if isinstance(embedding_value, np.ndarray):
        dim = int(embedding_value.shape[-1])
    elif isinstance(embedding_value, (list, tuple)):
        dim = len(embedding_value)
    else:
        embedding_array = np.asarray(embedding_value)
        if embedding_array.ndim == 0:
            raise IngestParquetError(
                f"Encountered scalar value in embedding column '{embedding_col}' for {parquet_file}"
            )
        dim = int(embedding_array.shape[-1])

    logging.info(f"Inferred embedding_dim={dim} from first row of {parquet_file}")
    return dim


def ingest_parquet_to_duckdb(
    parquet_files: list[str],
    db_path: str,
    dtype: str,
    embedding_col: str,
    mgrs_ids: Optional[list[str]] = None,
    roi_geometry_wkb: Optional[bytes] = None,
):
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

    logging.info(
        f"Starting ingestion of {len(parquet_files)} parquet files into {db_path}..."
    )

    # --- Schema Verification Step ---
    logging.info("--- Verifying source schema from first Parquet file ---")
    try:
        with duckdb.connect() as con_check:
            ensure_duckdb_extension(con_check, "httpfs")
            df_schema = con_check.execute(
                "DESCRIBE SELECT * FROM read_parquet(?);", (parquet_files[0],)
            ).fetchdf()
            available_columns = set(df_schema["column_name"])

            # Log the data type of the embedding column
            embedding_col_info = df_schema[df_schema["column_name"] == embedding_col]
            if not embedding_col_info.empty:
                embedding_type = embedding_col_info["column_type"].iloc[0]
                logging.info(f"Detected {embedding_col} column type: {embedding_type}")
            else:
                logging.error(f"Could not find {embedding_col} column in the schema.")

    except Exception as e:
        raise IngestParquetError(f"Could not read schema from {parquet_files[0]}: {e}")
    # --- End Schema Verification ---

    # Inspect the schema of the first file to determine available columns
    try:
        with duckdb.connect() as con_check:
            ensure_duckdb_extension(con_check, "httpfs")
            df_schema = con_check.execute(
                "DESCRIBE SELECT * FROM read_parquet(?);", (parquet_files[0],)
            ).fetchdf()
            available_columns = set(df_schema["column_name"])
    except Exception as e:
        raise IngestParquetError(f"Could not read schema from {parquet_files[0]}: {e}")

    has_geometry = "geometry" in available_columns

    try:
        embedding_dim = infer_embedding_dim_from_file(parquet_files[0], embedding_col)
    except Exception as e:
        raise IngestParquetError(str(e))

    # Use 'tile_id' if available, otherwise fall back to 'id'
    id_column_in_parquet = (
        "tile_id"
        if "tile_id" in available_columns
        else "id"
        if "id" in available_columns
        else None
    )
    if id_column_in_parquet:
        logging.info(
            f"Source ID column: '{id_column_in_parquet}'. Geometry column found: {has_geometry}."
        )
    else:
        logging.info("No 'tile_id' or 'id' column found, will generate ids.")

    with duckdb.connect(database=db_path) as con:
        ensure_duckdb_extension(con, "httpfs")

        # Load spatial extension, which is required for creating a spatial index
        if has_geometry:
            logging.info("Installing and loading spatial extension for DuckDB.")
            ensure_duckdb_extension(con, "spatial")

        logging.info("Creating table 'geo_embeddings' in DuckDB.")

        # Check if table already exists
        table_exists = (
            con.execute("""
            SELECT COUNT(*) 
            FROM information_schema.tables 
            WHERE table_name = 'geo_embeddings'
        """).fetchone()[0]
            > 0
        )

        if table_exists:
            existing_count = con.execute(
                "SELECT COUNT(*) FROM geo_embeddings"
            ).fetchone()[0]
            logging.warning(
                f"Table 'geo_embeddings' already exists with {existing_count} rows. Dropping and recreating..."
            )
            con.execute("DROP TABLE geo_embeddings;")
            con.execute("DROP SEQUENCE IF EXISTS seq_geo_embeddings_id;")

        # Create a sequence for auto-incrementing IDs
        con.execute("CREATE SEQUENCE IF NOT EXISTS seq_geo_embeddings_id START 1;")

        # Define embedding column type based on dtype flag
        if dtype.upper() == "INT8":
            embedding_sql_type = (
                f"UTINYINT[{embedding_dim}]"  # UTINYINT is DuckDB for uint8
            )
        else:
            embedding_sql_type = f"FLOAT[{embedding_dim}]"

        # Create table with dynamic schema
        if id_column_in_parquet:
            create_sql = f"""
            CREATE TABLE geo_embeddings (
                id BIGINT PRIMARY KEY DEFAULT nextval('seq_geo_embeddings_id'),
                source_id VARCHAR,
                embedding {embedding_sql_type}
                {", geometry GEOMETRY" if has_geometry else ""}
            );
            """
        else:
            create_sql = f"""
            CREATE TABLE geo_embeddings (
                id BIGINT PRIMARY KEY DEFAULT nextval('seq_geo_embeddings_id'),
                embedding {embedding_sql_type}
                {", geometry GEOMETRY" if has_geometry else ""}
            );
            """

        logging.info(
            f"Creating table 'geo_embeddings' in DuckDB with sql: {create_sql}"
        )
        con.execute(create_sql)

        # Ingest all files in a single, efficient query
        try:
            sql_parquet_files_list_str = "['" + "', '".join(parquet_files) + "']"

            insert_columns = f"embedding{', geometry' if has_geometry else ''}"
            select_clause = f"{embedding_col}"
            if id_column_in_parquet:
                insert_columns = f"source_id, {insert_columns}"
                select_clause = f"{id_column_in_parquet}, {select_clause}"

            # The geometry column from GeoParquet is already understood by DuckDB's spatial extension.
            # No explicit cast is needed if the target column is type GEOMETRY.
            if has_geometry:
                select_clause += ", geometry"

                # Apply ROI filter only when WKB geometry is provided
                if roi_geometry_wkb is not None:
                    logging.info(
                        "Applying ROI geometry (WKB) filter during ingestion (ST_Intersects)..."
                    )
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
                if id_column_in_parquet == "tile_id" and mgrs_ids:
                    logging.info(
                        "Applying tile_id filter using MGRS IDs during ingestion (no geometry column available)..."
                    )
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
            raise IngestParquetError(
                f"Failed to ingest parquet files: {e}. Please ensure all parquet files have a consistent schema."
            )

        row_count = con.execute("SELECT COUNT(*) FROM geo_embeddings;").fetchone()[0]
        logging.info(f"Successfully ingested {row_count} rows into DuckDB.")

        # Reorder table by id for optimal zone map performance over httpfs
        logging.info("Reordering table by id for optimal remote query performance...")
        con.execute(
            "CREATE TABLE geo_embeddings_ordered AS SELECT * FROM geo_embeddings ORDER BY id"
        )
        con.execute("DROP TABLE geo_embeddings")
        con.execute("ALTER TABLE geo_embeddings_ordered RENAME TO geo_embeddings")
        logging.info("Table reordered successfully.")

        if has_geometry:
            logging.info("Creating R-Tree spatial index on geometry column...")
            con.execute(
                "CREATE INDEX geom_spatial_idx ON geo_embeddings USING RTREE (geometry);"
            )
            logging.info("Spatial index created successfully.")

        logging.info("Creating index on id column for fast lookups...")
        con.execute("CREATE INDEX id_idx ON geo_embeddings(id);")
        logging.info("ID index created successfully.")

        _validate_httpfs_artifact_layout(
            con,
            embedding_dim=embedding_dim,
            dtype=dtype,
        )

        return embedding_dim


def create_faiss_index(
    db_path: str,
    index_path: str,
    embedding_dim: int,
    dtype: str,
    nlist: int,
    m: int,
    nbits: int,
    batch_size: int,
):
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
        total_vectors = con.execute("SELECT COUNT(*) FROM geo_embeddings;").fetchone()[
            0
        ]
        logging.info(f"Total vectors to index: {total_vectors}")

        if total_vectors == 0:
            logging.error(
                "The 'geo_embeddings' table is empty after ingestion. Nothing to index. Aborting."
            )
            logging.error(
                "This might happen if the input parquet files are empty or have an incompatible schema."
            )
            return

        # --- Branch logic based on dtype ---
        if dtype.upper() == "FLOAT":
            logging.info("Building FAISS index for FLOAT data using IndexIVFPQ.")
            # Use IndexIVFPQ for floating point data
            quantizer = faiss.IndexFlatL2(embedding_dim)
            index = faiss.IndexIVFPQ(quantizer, embedding_dim, nlist, m, nbits)
            # training vectors always converted to np.float32 below
        elif dtype.upper() == "INT8":
            logging.info(
                "Building FAISS index for INT8 data using IndexIVFScalarQuantizer."
            )
            # Use IndexIVFScalarQuantizer for quantized integer data
            quantizer = faiss.IndexFlatL2(
                embedding_dim
            )  # The IVF quantizer still operates in float space
            index = faiss.IndexIVFScalarQuantizer(
                quantizer, embedding_dim, nlist, faiss.ScalarQuantizer.QT_8bit
            )

        else:
            raise ValueError(f"Unsupported dtype for FAISS index: {dtype}")

        # 2. Train the index on a sample of the data
        logging.info("Training FAISS index...")
        train_sample_size = min(total_vectors, max(total_vectors // 10, 2000000))
        logging.info(f"Using {train_sample_size} vectors for training...")

        training_vectors_df = con.execute(
            f"SELECT embedding FROM geo_embeddings TABLESAMPLE RESERVOIR({train_sample_size} ROWS);"
        ).fetchdf()

        if training_vectors_df.empty:
            logging.error(
                "Failed to sample training vectors from the database. Aborting."
            )
            return

        # Training always requires float32 vectors
        training_vectors = np.vstack(training_vectors_df["embedding"].values).astype(
            np.float32
        )

        start_time = time.time()
        index.train(training_vectors)
        logging.info(
            f"Index training completed in {time.time() - start_time:.2f} seconds."
        )

        del training_vectors
        gc.collect()

        # 3. Populate the index in batches
        logging.info("Populating FAISS index in batches...")
        num_batches = (total_vectors + batch_size - 1) // batch_size

        for i in tqdm(range(num_batches), desc="Populating FAISS Index"):
            offset = i * batch_size
            batch_df = con.execute(
                f"SELECT id, embedding FROM geo_embeddings ORDER BY id LIMIT {batch_size} OFFSET {offset};"
            ).fetchdf()

            ids = batch_df["id"].values.astype("int64")

            # For adding to the index, SQ needs float32, PQ can take float16 or float32.
            # We use float32 for both for simplicity here.
            vectors = np.vstack(batch_df["embedding"].values).astype(np.float32)

            index.add_with_ids(vectors, ids)

        logging.info("Index population complete.")

        # 4. Save the final index to disk
        logging.info(f"Writing index to {index_path}")
        faiss.write_index(index, index_path)
        logging.info("FAISS index successfully built and saved.")


def export_geometry_cache(db_path: str, output_path: str) -> None:
    """Export id + geometry to compressed Parquet for local caching.

    This creates a lightweight file (~50-100MB) containing only the id and
    geometry columns, which can be downloaded and cached locally for fast
    search metadata queries over httpfs.

    Args:
        db_path: Path to the DuckDB database
        output_path: Path for the output Parquet file
    """
    logging.info(f"Exporting geometry cache to {output_path}")
    start_time = time.time()

    con = duckdb.connect(db_path, read_only=True)
    con.execute("INSTALL spatial; LOAD spatial;")

    row_count = con.execute("SELECT COUNT(*) FROM geo_embeddings").fetchone()[0]
    logging.info(f"Exporting {row_count:,} geometries...")

    con.execute(f"""
        COPY (SELECT id, geometry FROM geo_embeddings ORDER BY id)
        TO '{output_path}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)

    con.close()

    file_size_mb = pathlib.Path(output_path).stat().st_size / (1024 * 1024)
    elapsed = time.time() - start_time
    logging.info(f"Geometry cache exported: {file_size_mb:.1f} MB in {elapsed:.1f}s")


def _prefix_name_with_roi(name: str, roi_file: Optional[str]) -> str:
    if not roi_file:
        return name
    roi_stem = pathlib.Path(roi_file).stem
    if not roi_stem:
        return name
    prefix = f"{roi_stem}_"
    if name.lower().startswith(prefix.lower()):
        return name
    return f"{roi_stem}_{name}"


def _append_tile_suffix(
    name: str, tile_pixels: int, tile_overlap: int, tile_resolution: int
) -> str:
    suffix = f"{tile_pixels}_{tile_overlap}_{tile_resolution}"
    if name.lower().endswith(f"_{suffix}".lower()):
        return name
    return f"{name}_{suffix}"


def main():
    parser = argparse.ArgumentParser(
        description="Build a FAISS index from geospatial embeddings stored in Parquet files."
    )

    # argparse cannot include positional args in mutually exclusive groups.
    # Define both inputs and validate exclusivity manually below.
    parser.add_argument(
        "parquet_files",
        nargs="*",
        help="Paths to input Parquet files or directories (local or gs:// or s3://)",
    )
    parser.add_argument(
        "--roi-file",
        dest="roi_file",
        help="ROI geometry file to intersect with MGRS tiles for automatic file discovery",
    )

    parser.add_argument(
        "--mgrs-reference-file",
        dest="mgrs_reference_file",
        help="MGRS reference file containing MGRS tile geometries (required with --roi-file)",
    )
    parser.add_argument(
        "--embedding-dir",
        dest="embedding_dir",
        help="Directory containing embedding parquet files (required with --roi-file)",
    )
    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="A descriptive name for the output files (e.g., 'bali_2024').",
    )
    default_output_dir = pathlib.Path("local_databases")
    parser.add_argument(
        "--output_dir",
        default=str(default_output_dir),
        help="Directory to save the DuckDB and FAISS index files.",
    )
    parser.add_argument(
        "--embedding_col",
        type=str,
        default="embedding",
        help="Name of the embedding column in the Parquet files.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="FLOAT",
        choices=["FLOAT", "INT8"],
        help="Data type of the embeddings (FLOAT or INT8).",
    )
    parser.add_argument(
        "--nlist",
        type=int,
        default=4096,
        help="Number of clusters (IVF cells) for the FAISS index.",
    )
    parser.add_argument(
        "--m",
        type=int,
        default=64,
        help="Number of sub-quantizers for FAISS Product Quantization (used for FLOAT dtype).",
    )
    parser.add_argument(
        "--nbits",
        type=int,
        default=8,
        help="Number of bits per sub-quantizer code (used for FLOAT dtype).",
    )
    parser.add_argument(
        "--tile-pixels",
        type=int,
        default=32,
        help="Tile size in pixels to include in output naming.",
    )
    parser.add_argument(
        "--tile-resolution",
        type=int,
        default=10,
        help="Tile resolution (meters per pixel) to include in output naming.",
    )
    parser.add_argument(
        "--tile-overlap",
        type=int,
        default=16,
        help="Tile overlap in pixels to include in output naming.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=500_000,
        help="Batch size for populating the FAISS index.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run the script on a small subset of files to test the pipeline.",
    )
    parser.add_argument(
        "--dry-run-size",
        type=int,
        default=5,
        help="Number of files to use in a dry run.",
    )

    args = parser.parse_args()

    setup_logging()

    output_destination = args.output_dir
    output_protocol = (
        get_cloud_protocol(output_destination) if output_destination else None
    )
    if output_protocol:
        local_output_dir = pathlib.Path(tempfile.mkdtemp(prefix="faiss_output_"))
        logging.info(
            "Output directory %s is a cloud path (protocol=%s); using temporary workspace %s for artifacts.",
            output_destination,
            output_protocol,
            local_output_dir,
        )
        cleanup_output_dir = True
        tarball_required = True
    else:
        local_output_dir = pathlib.Path(output_destination)
        local_output_dir.mkdir(parents=True, exist_ok=True)
        cleanup_output_dir = False
        tarball_required = False

    if getattr(args, "roi_file", None):
        args.name = _prefix_name_with_roi(args.name, args.roi_file)

    args.name = _append_tile_suffix(
        args.name, args.tile_pixels, args.tile_overlap, args.tile_resolution
    )

    # Construct descriptive filenames
    faiss_params_str = f"faiss_{args.nlist}_{args.m}_{args.nbits}"
    db_filename = f"{args.name}_metadata.db"
    index_filename = f"{args.name}_{faiss_params_str}.index"
    geometry_cache_filename = f"{args.name}_geometry.parquet"

    db_path = str((local_output_dir / db_filename).resolve())
    index_path = str((local_output_dir / index_filename).resolve())
    geometry_cache_path = str((local_output_dir / geometry_cache_filename).resolve())

    # Validate mutual exclusivity and ROI-based arguments
    if getattr(args, "roi_file", None) and args.parquet_files:
        parser.error(
            "Cannot use positional parquet_files with --roi-file; choose one input method."
        )
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
        logging.info(f"Found {len(mgrs_ids)} MGRS tiles intersecting with ROI")
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
                    roi_wkb = (
                        bytes(row[0])
                        if not isinstance(row[0], (bytes, bytearray))
                        else row[0]
                    )
        except Exception as e:
            logging.error(f"Failed to derive ROI WKB from file {args.roi_file}: {e}")

        embedding_files = find_embedding_files_for_mgrs_ids(
            mgrs_ids, args.embedding_dir
        )
        if not embedding_files:
            logging.error("No embedding files found for intersecting MGRS tiles")
            return
        for file_path in embedding_files:
            if get_cloud_protocol(file_path):
                cloud_files.append(file_path)
            else:
                all_parquet_files.append(file_path)
        logging.info(
            f"Found {len(embedding_files)} embedding files for ROI ({len(all_parquet_files)} local, {len(cloud_files)} cloud)"
        )
    else:
        logging.info("Using explicit parquet file paths")
        parquet_inputs = getattr(args, "parquet_files", []) or []
        for path in parquet_inputs:
            if get_cloud_protocol(path):
                if path.endswith(".parquet"):
                    cloud_files.append(path)
                else:
                    cloud_parquet_files = list_cloud_parquet_files(path)
                    cloud_files.extend(cloud_parquet_files)
                    logging.info(
                        f"Found {len(cloud_parquet_files)} parquet files in {path}"
                    )
            else:
                path_obj = pathlib.Path(path)
                if path_obj.is_file() and path_obj.suffix == ".parquet":
                    all_parquet_files.append(str(path_obj))
                elif path_obj.is_dir():
                    all_parquet_files.extend(
                        glob.glob(f"{str(path_obj)}/**/*.parquet", recursive=True)
                    )
                else:
                    all_parquet_files.extend(glob.glob(path))

    temp_dir = None
    temp_dir_created = False
    if cloud_files:
        cache_root = pathlib.Path(tempfile.gettempdir()) / "faiss_db_cache"
        cache_root.mkdir(parents=True, exist_ok=True)
        cloud_digest = hashlib.sha256(
            "\n".join(sorted(cloud_files)).encode()
        ).hexdigest()
        cache_path = cache_root / cloud_digest
        existing_cache = cache_path.exists()
        cache_path.mkdir(parents=True, exist_ok=True)
        temp_dir = str(cache_path)
        temp_dir_created = not existing_cache
        if existing_cache:
            logging.info(f"Reusing cached downloads in {temp_dir}")
        else:
            logging.info(f"Created download cache at {temp_dir}")
        local_cloud_files: list[str] = []
        missing_cloud_files: list[str] = []
        for remote_path in cloud_files:
            local_candidate = os.path.join(temp_dir, os.path.basename(remote_path))
            if os.path.exists(local_candidate) and os.path.getsize(local_candidate) > 0:
                local_cloud_files.append(local_candidate)
            else:
                missing_cloud_files.append(remote_path)

        if missing_cloud_files:
            logging.info(f"Downloading {len(missing_cloud_files)} files from cloud...")
            downloaded = download_cloud_files(missing_cloud_files, temp_dir)
            local_cloud_files.extend(downloaded)
        else:
            logging.info("All cloud files already cached; skipping downloads.")

        all_parquet_files.extend(local_cloud_files)

    # Deduplicate and validate
    all_parquet_files = list(dict.fromkeys(all_parquet_files))

    if not all_parquet_files:
        logging.error("No parquet files found!")
        if temp_dir and temp_dir_created:
            shutil.rmtree(temp_dir)
        return

    logging.info(f"Total {len(all_parquet_files)} parquet files to process")

    files_to_process = all_parquet_files
    if args.dry_run:
        logging.info(
            f"--- Starting DRY RUN using the first {args.dry_run_size} files. ---"
        )
        files_to_process = files_to_process[: args.dry_run_size]

    if not files_to_process:
        logging.error("No files left to process after applying dry-run filter.")
        if temp_dir and temp_dir_created:
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
    create_faiss_index(
        db_path,
        index_path,
        inferred_dim,
        args.dtype,
        args.nlist,
        args.m,
        args.nbits,
        args.batch_size,
    )

    # --- Phase 3: Export geometry cache for httpfs optimization ---
    export_geometry_cache(db_path, geometry_cache_path)

    logging.info(
        f"Total process completed in {time.time() - start_total_time:.2f} seconds."
    )
    logging.info(
        f"Artifacts created:\n- DuckDB: {db_path}\n- FAISS Index: {index_path}\n- Geometry Cache: {geometry_cache_path}"
    )

    if temp_dir and temp_dir_created:
        shutil.rmtree(temp_dir)
        logging.info(f"Cleaned up temporary directory: {temp_dir}")
    elif temp_dir:
        logging.info(f"Preserving download cache directory: {temp_dir}")

    # Create compressed tarball and upload to output directory
    if tarball_required:
        try:
            tar_name = f"{args.name}.tar.gz"
            if output_protocol:
                tar_local_dir = tempfile.mkdtemp(prefix="faiss_tar_")
            else:
                tar_local_dir = str(local_output_dir)
            tar_local_path = os.path.join(tar_local_dir, tar_name)

            cmd = f"tar -c {shlex.quote(db_path)} {shlex.quote(index_path)} {shlex.quote(geometry_cache_path)} | pigz > {shlex.quote(tar_local_path)}"
            try:
                subprocess.run(["bash", "-lc", cmd], check=True)
                logging.info(f"Created tarball with pigz: {tar_local_path}")
            except Exception as e:
                logging.warning(
                    f"pigz tarball creation failed ({e}); falling back to gzip."
                )
                fallback_cmd = f"tar -czf {shlex.quote(tar_local_path)} {shlex.quote(db_path)} {shlex.quote(index_path)} {shlex.quote(geometry_cache_path)}"
                subprocess.run(["bash", "-lc", fallback_cmd], check=True)
                logging.info(f"Created tarball with gzip fallback: {tar_local_path}")

            dest_path = output_destination.rstrip("/") + "/" + tar_name
            fs_kwargs: dict[str, object] = {}
            if output_protocol == "s3":
                endpoint = os.environ.get(
                    "S3_ENDPOINT_URL", "https://s3.us-west-2.amazonaws.com"
                )
                fs_kwargs["client_kwargs"] = {"endpoint_url": endpoint}
                fs_kwargs["anon"] = False
            fs = fsspec.filesystem(output_protocol, **fs_kwargs)
            fs.put(tar_local_path, dest_path)
            logging.info(f"Uploaded tarball to {dest_path}")
            shutil.rmtree(tar_local_dir, ignore_errors=True)
        except Exception as e:
            logging.error(f"Failed to create/upload tarball: {e}")
    else:
        logging.info(
            "Tarball creation skipped for local output directory; files available directly."
        )

    if cleanup_output_dir:
        shutil.rmtree(local_output_dir, ignore_errors=True)
        logging.info(f"Removed temporary local output directory: {local_output_dir}")


if __name__ == "__main__":
    main()
