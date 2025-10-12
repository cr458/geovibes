import argparse
import os
import tempfile
import uuid
from pathlib import Path

import duckdb
import joblib
import numpy as np
import pyarrow as pa
from tqdm import tqdm


def make_test_connection():
    con = duckdb.connect(database=":memory:")
    con.execute("INSTALL spatial;")
    con.execute("LOAD spatial;")
    tmp = tempfile.TemporaryDirectory()
    return con, tmp


def seed_sample_data(con):
    con.execute(
        """
        CREATE TABLE embeddings(
          id BIGINT,
          geom GEOMETRY,
          embedding FLOAT[4]
        );
        """
    )
    con.execute(
        """
        INSERT INTO embeddings VALUES
          (1, ST_GeomFromText('SRID=4326;POINT(-122.4194 37.7749)'), [0.1, 0.2, 0.3, 0.4]),
          (2, ST_GeomFromText('SRID=4326;POINT(12.5683 55.6761)'), [0.2, 0.1, 0.0, 0.5]),
          (3, ST_GeomFromText('SRID=4326;POINT(151.2093 -33.8688)'), [0.4, 0.3, 0.2, 0.1]);
        """
    )


class DummyModel:
    def __init__(self):
        self.weights = np.array([0.5, -0.3, 0.2, 0.1], dtype=np.float32)

    def predict_proba(self, X):
        z = X @ self.weights
        p1 = 1.0 / (1.0 + np.exp(-z))
        p0 = 1.0 - p1
        return np.vstack([p0, p1]).T.astype(np.float32)


def load_model(path="model.pkl"):
    if os.path.exists(path):
        try:
            model = joblib.load(path)
            return model
        except Exception:
            pass
    return DummyModel()


def py_predict_arrow(emb_col):
    if isinstance(emb_col, pa.ChunkedArray):
        chunks = emb_col.chunks
    else:
        chunks = [emb_col]
    out = []
    model = py_predict_arrow.model
    for chunk in chunks:
        X = np.array(chunk.to_pylist(), dtype=np.float32)
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)[:, 1].astype(np.float32)
        else:
            preds = model.predict(X)
            probs = np.asarray(preds, dtype=np.float32)
        out.append(pa.array(probs))
    return pa.chunked_array(out)


def register_py_predict(con, model):
    py_predict_arrow.model = model
    con.create_function(
        "py_predict",
        py_predict_arrow,
        parameters=["FLOAT[]"],
        return_type=duckdb.typing.FLOAT,
        type="arrow",
    )


def score_embeddings(con, table="embeddings", id_col="id", geom_col="geom", emb_col="embedding"):
    return con.sql(
        f"""
        SELECT {id_col} AS id, {geom_col} AS geom, py_predict({emb_col}) AS probability
        FROM {table}
        """
    )


def filter_scores(scored_relation, threshold):
    return scored_relation.filter(f"probability >= {threshold}")


def build_metric_tiles(filtered_relation, tile_size_m):
    half = tile_size_m / 2.0
    return filtered_relation.query(
        "scored",
        f"""
        WITH zoned AS (
          SELECT
            id,
            probability,
            geom,
            CAST(FLOOR((ST_X(geom) + 180) / 6) + 1 AS INTEGER) AS zone,
            CASE WHEN ST_Y(geom) < 0 THEN 'S' ELSE 'N' END AS hemi
          FROM scored
        ),
        reproj AS (
          SELECT
            id,
            probability,
            zone,
            hemi,
            ((CASE WHEN hemi = 'S' THEN 32700 ELSE 32600 END) + zone) AS utm_srid,
            ST_Transform(
              geom,
              'EPSG:4326',
              'EPSG:' || CAST(((CASE WHEN hemi = 'S' THEN 32700 ELSE 32600 END) + zone) AS VARCHAR),
              TRUE
            ) AS geom_utm
          FROM zoned
        )
        SELECT
          id,
          probability,
          zone,
          hemi,
          utm_srid,
          ST_MakeEnvelope(
            ST_X(geom_utm) - {half},
            ST_Y(geom_utm) - {half},
            ST_X(geom_utm) + {half},
            ST_Y(geom_utm) + {half}
          ) AS tile_utm
        FROM reproj
        """,
    )


def union_tiles(tile_relation):
    return tile_relation.query(
        "tiles",
        """
        WITH merged AS (
          SELECT
            zone,
            hemi,
            utm_srid,
            MAX(probability) AS max_probability,
            COUNT(*) AS tile_count,
            ST_Union_Agg(tile_utm) AS union_utm
          FROM tiles
          GROUP BY zone, hemi, utm_srid
        )
        SELECT
          zone,
          hemi,
          utm_srid,
          max_probability,
          tile_count,
          ST_Transform(
            union_utm,
            'EPSG:' || CAST(utm_srid AS VARCHAR),
            'EPSG:4326',
            TRUE
          ) AS geometry
        FROM merged
        """,
    )


def export_geoparquet(con, relation, out_path):
    view_name = f"_codex_export_{uuid.uuid4().hex}"
    relation.create_view(view_name, replace=True)
    try:
        target = Path(out_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        con.execute(
            f"""
            COPY (
              SELECT * FROM {view_name}
            ) TO '{target}'
            WITH (FORMAT PARQUET)
            """
        )
        return str(target)
    finally:
        con.execute(f"DROP VIEW {view_name}")


def run_inference_tiles(
    db_path,
    table,
    id_col,
    geom_col,
    emb_col,
    threshold,
    tile_size_m,
    out_geojson,
    model_path="model.pkl",
    chunk_size=5000,
):
    temp_scored = None
    con = duckdb.connect(database=db_path)
    try:
        try:
            con.execute("INSTALL spatial;")
        except duckdb.Error:
            pass
        con.execute("LOAD spatial;")
        model = load_model(model_path)
        register_py_predict(con, model)
        total_rows = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        chunk_size = max(1, int(chunk_size))
        id_type = None
        for cid, name, dtype, *_ in con.execute(f"PRAGMA table_info('{table}')").fetchall():
            if name == id_col and dtype:
                id_type = dtype
                break
        temp_scored = f"_codex_scored_{uuid.uuid4().hex}"
        con.execute(
            f"""
            CREATE TEMP TABLE {temp_scored} AS
            SELECT CAST(NULL AS VARCHAR) AS id, CAST(NULL AS BLOB) AS geom_wkb, CAST(NULL AS DOUBLE) AS probability
            WHERE FALSE
            """
        )
        if total_rows > 0:
            relation_sql = (
                f"SELECT CAST({id_col} AS VARCHAR) AS id, ST_AsWKB({geom_col}) AS geom_wkb, py_predict({emb_col}) AS probability"
                f" FROM {table}"
            )
            con.execute(relation_sql)
            reader = con.fetch_record_batch(chunk_size)
            tqdm.write("Scoring embeddings...")
            with tqdm(total=total_rows, desc="Scoring", unit="rows", leave=False) as progress:
                while True:
                    try:
                        batch = reader.read_next_batch()
                    except StopIteration:
                        break
                    if batch is None or batch.num_rows == 0:
                        break
                    table_batch = pa.Table.from_batches([batch])
                    temp_batch = f"_codex_batch_{uuid.uuid4().hex}"
                    con.register(temp_batch, table_batch)
                    con.execute(
                        f"INSERT INTO {temp_scored} SELECT id, geom_wkb, probability FROM {temp_batch}"
                    )
                    con.unregister(temp_batch)
                    progress.update(batch.num_rows)
            tqdm.write("Scoring complete")
        id_expr = "id"
        if id_type:
            id_expr = f"CAST(id AS {id_type})"
        scored = con.sql(
            f"SELECT {id_expr} AS id, ST_GeomFromWKB(geom_wkb) AS geom, probability FROM {temp_scored}"
        )
        tqdm.write("Applying threshold filter...")
        filtered = filter_scores(scored, threshold)
        tqdm.write("Building metric tiles...")
        tiles = build_metric_tiles(filtered, tile_size_m)
        tqdm.write("Merging tiles by zone...")
        merged = union_tiles(tiles)
        tqdm.write("Exporting GeoParquet...")
        export_geoparquet(con, merged, out_geojson)
        return out_geojson
    finally:
        if temp_scored is not None:
            try:
                con.execute(f"DROP TABLE IF EXISTS {temp_scored}")
            except Exception:
                pass
        con.close()


def cli_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True)
    parser.add_argument("--table", default="embeddings")
    parser.add_argument("--id-col", default="id")
    parser.add_argument("--geom-col", default="geom")
    parser.add_argument("--emb-col", default="embedding")
    parser.add_argument("--threshold", type=float, default=0.85)
    parser.add_argument("--tile-size-m", type=float, default=100.0)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default="model.pkl")
    parser.add_argument("--chunk-size", type=int, default=5000)
    args = parser.parse_args()
    run_inference_tiles(
        args.db,
        args.table,
        args.id_col,
        args.geom_col,
        args.emb_col,
        args.threshold,
        args.tile_size_m,
        args.out,
        model_path=args.model,
        chunk_size=args.chunk_size,
    )


if __name__ == "__main__":
    cli_main()
