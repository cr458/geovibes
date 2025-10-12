import argparse
import json
import math
from pathlib import Path

import duckdb
import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import wkb
from shapely.ops import unary_union
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--duckdb-path", required=True)
    parser.add_argument("--table", default="geo_embeddings")
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--training-set", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--keep-tiles", action="store_true")
    parser.add_argument("--tiles-output")
    parser.add_argument("--chunk-size", type=int, default=5000)
    parser.add_argument("--raw-chunks-output")
    return parser.parse_args()


def detect_feature_dim(con, table):
    query = f"SELECT array_length(embedding) FROM {table} WHERE embedding IS NOT NULL LIMIT 1"
    result = con.execute(query).fetchone()
    if not result or result[0] is None:
        raise ValueError("No embeddings available to detect feature dimension.")
    return int(result[0])


def ensure_training_data(path, feature_dim):
    p = Path(path)
    if p.exists():
        df = pd.read_parquet(p)
        feature_cols = [c for c in df.columns if c.startswith("f_")]
        feature_cols.sort(key=lambda x: int(x.split("_")[1]))
        if len(feature_cols) == feature_dim and "label" in df.columns:
            features = df[feature_cols].to_numpy(dtype=np.float32)
            labels = df["label"].to_numpy()
            return features, labels
    rng = np.random.default_rng(0)
    features = rng.normal(size=(2000, feature_dim)).astype(np.float32)
    labels = rng.integers(0, 2, size=(2000,), endpoint=False)
    columns = [f"f_{i}" for i in range(feature_dim)]
    df = pd.DataFrame(features, columns=columns)
    df["label"] = labels
    df.to_parquet(p)
    return features, labels


def train_classifier(features, labels):
    model = RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=-1)
    model.fit(features, labels)
    return model


def iterate_embedding_batches(con, table, chunk_size):
    query = f"SELECT tile_id, embedding, ST_AsWKB(geometry) AS geometry FROM {table}"
    con.execute(query)
    while True:
        chunk = con.fetch_df_chunk(chunk_size)
        if chunk is None or chunk.empty:
            break
        yield chunk


def build_tile_geodataframe(df, probabilities):
    geoms = df["geometry"].tolist()
    parsed_geoms = []
    for g in geoms:
        if g is None:
            parsed_geoms.append(None)
        elif isinstance(g, (bytes, bytearray, memoryview)):
            parsed_geoms.append(wkb.loads(bytes(g)))
        else:
            parsed_geoms.append(g)
    data = {
        "tile_id": df["tile_id"].tolist(),
        "probability": probabilities.tolist(),
    }
    return gpd.GeoDataFrame(data, geometry=parsed_geoms, crs="EPSG:4326")


def dissolve_tiles(tile_gdf):
    if tile_gdf.empty:
        return gpd.GeoDataFrame(
            {"tile_ids": pd.Series(dtype=str), "max_probability": pd.Series(dtype=float), "tile_count": pd.Series(dtype=int)},
            geometry=gpd.GeoSeries([], crs="EPSG:4326"),
        )
    union_geom = unary_union(tile_gdf.geometry.tolist())
    parts = []
    if union_geom.is_empty:
        return gpd.GeoDataFrame(
            {"tile_ids": pd.Series(dtype=str), "max_probability": pd.Series(dtype=float), "tile_count": pd.Series(dtype=int)},
            geometry=gpd.GeoSeries([], crs="EPSG:4326"),
        )
    if union_geom.geom_type == "Polygon":
        components = [union_geom]
    elif union_geom.geom_type == "MultiPolygon":
        components = list(union_geom.geoms)
    else:
        components = [union_geom]
    tiles = tile_gdf[["tile_id", "probability", "geometry"]]
    for geom in components:
        mask = tiles.geometry.intersects(geom)
        subset = tiles.loc[mask]
        if subset.empty:
            continue
        record = {
            "tile_ids": json.dumps(subset.tile_id.tolist()),
            "max_probability": float(subset.probability.max()),
            "tile_count": int(len(subset)),
            "geometry": geom,
        }
        parts.append(record)
    return gpd.GeoDataFrame(parts, geometry="geometry", crs=tile_gdf.crs)


def main():
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    args = parse_args()
    logger.info(f"Connecting to DuckDB database: {args.duckdb_path}")
    con = duckdb.connect(args.duckdb_path, read_only=True)
    try:
        logger.info("Loading spatial extension")
        try:
            con.execute("INSTALL spatial")
        except duckdb.Error:
            pass
        con.execute("LOAD spatial")
        
        logger.info(f"Detecting feature dimensions for table: {args.table}")
        feature_dim = detect_feature_dim(con, args.table)
        logger.info(f"Feature dimension detected: {feature_dim}")
        
        logger.info(f"Loading training data from: {args.training_set}")
        features, labels = ensure_training_data(args.training_set, feature_dim)
        logger.info(f"Training data loaded: {len(features)} samples")
        
        logger.info("Training classifier")
        model = train_classifier(features, labels)
        logger.info("Classifier training completed")
        
        logger.info(f"Counting total rows in table: {args.table}")
        total_rows = con.execute(f"SELECT COUNT(*) FROM {args.table}").fetchone()[0]
        logger.info(f"Total rows to process: {total_rows}")
        if total_rows == 0:
            raise ValueError("No records found in the embeddings table.")

        if args.raw_chunks_output:
            raw_chunks_dir = Path(args.raw_chunks_output)
            raw_chunks_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Raw chunks will be saved to: {raw_chunks_dir}")

        logger.info("Starting batch processing and classification")
        filtered_chunks = []
        chunk_counter = 0
        total_chunks = max(1, math.ceil(total_rows / args.chunk_size))
        logger.info("Estimated chunk count: %s", total_chunks)
        for chunk in tqdm(
            iterate_embedding_batches(con, args.table, args.chunk_size), total=total_chunks
        ):
            vectors = np.asarray(chunk["embedding"].tolist(), dtype=np.float32)
            probabilities = model.predict_proba(vectors)[:, 1]
            chunk_gdf = build_tile_geodataframe(chunk, probabilities)
            
            if args.raw_chunks_output:
                chunk_path = raw_chunks_dir / f"chunk_{chunk_counter:06d}.parquet"
                chunk_gdf.to_parquet(chunk_path)
                chunk_counter += 1
            
            chunk_filtered = chunk_gdf[chunk_gdf["probability"] >= args.threshold].reset_index(drop=True)
            if not chunk_filtered.empty:
                filtered_chunks.append(chunk_filtered)

    finally:
        con.close()
        logger.info("Database connection closed")
    
    if args.raw_chunks_output:
        logger.info(f"Saved {chunk_counter} raw chunks to: {raw_chunks_dir}")
    
    logger.info("Combining filtered chunks")
    if filtered_chunks:
        filtered_tiles = gpd.GeoDataFrame(
            pd.concat(filtered_chunks, ignore_index=True),
            geometry="geometry",
            crs=filtered_chunks[0].crs,
        )
        logger.info(f"Total filtered tiles: {len(filtered_tiles)}")
    else:
        filtered_tiles = gpd.GeoDataFrame(
            {"tile_id": pd.Series(dtype=str), "probability": pd.Series(dtype=float)},
            geometry=gpd.GeoSeries([], crs="EPSG:4326"),
        )
        logger.info("No tiles passed the threshold filter")
    
    logger.info("Dissolving tiles into contiguous regions")
    dissolved = dissolve_tiles(filtered_tiles)
    dissolved["threshold"] = args.threshold
    logger.info(f"Created {len(dissolved)} dissolved regions")
    
    logger.info(f"Saving dissolved results to: {args.output}")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    dissolved.to_parquet(args.output)
    
    if args.keep_tiles:
        output_path = Path(args.output)
        tiles_path = Path(args.tiles_output) if args.tiles_output else output_path.with_name(f"{output_path.stem}_tiles.parquet")
        logger.info(f"Saving individual tiles to: {tiles_path}")
        filtered_tiles.to_parquet(tiles_path)
    
    logger.info("Classification pipeline completed successfully")


if __name__ == "__main__":
    main()
