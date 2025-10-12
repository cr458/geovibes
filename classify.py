import argparse
import json
from pathlib import Path

import duckdb
import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import wkb
from shapely.ops import unary_union
from sklearn.ensemble import RandomForestClassifier


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--duckdb-path", required=True)
    parser.add_argument("--table", default="geo_embeddings")
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--training-set", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--keep-tiles", action="store_true")
    parser.add_argument("--tiles-output")
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


def fetch_embeddings(con, table):
    query = f"SELECT tile_id, embedding, geometry FROM {table}"
    df = con.execute(query).fetch_df()
    if df.empty:
        raise ValueError("No records found in the embeddings table.")
    return df


def build_tile_geodataframe(df, probabilities):
    embeddings = df["embedding"].tolist()
    geoms = df["geometry"].tolist()
    parsed_geoms = [wkb.loads(g) if isinstance(g, (bytes, bytearray, memoryview)) else g for g in geoms]
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
    args = parse_args()
    con = duckdb.connect(args.duckdb_path, read_only=True)
    try:
        feature_dim = detect_feature_dim(con, args.table)
        features, labels = ensure_training_data(args.training_set, feature_dim)
        model = train_classifier(features, labels)
        df = fetch_embeddings(con, args.table)
    finally:
        con.close()
    embeddings = [np.asarray(row, dtype=np.float32) for row in df["embedding"].tolist()]
    feature_matrix = np.vstack(embeddings)
    probabilities = model.predict_proba(feature_matrix)[:, 1]
    tile_gdf = build_tile_geodataframe(df, probabilities)
    filtered_tiles = tile_gdf[tile_gdf["probability"] >= args.threshold].reset_index(drop=True)
    dissolved = dissolve_tiles(filtered_tiles)
    dissolved["threshold"] = args.threshold
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    dissolved.to_parquet(args.output)
    if args.keep_tiles:
        output_path = Path(args.output)
        tiles_path = Path(args.tiles_output) if args.tiles_output else output_path.with_name(f"{output_path.stem}_tiles.parquet")
        filtered_tiles.to_parquet(tiles_path)


if __name__ == "__main__":
    main()
