# Codex Implementation Prompt: Out-of-Core Inference with Geospatial Aggregation in DuckDB

## Overview

This prompt instructs Codex to implement a complete, tested workflow for out-of-core inference using a DuckDB database with geospatial data. The workflow will:

1. Run inference on embeddings stored in DuckDB using an Arrow UDF.
2. Filter results above a probability threshold.
3. Create metric tiles (in meters) around points, grouped by UTM zone.
4. Perform a unary union of the resulting geometries.
5. Export the final geometry as a GeoJSON file.

The implementation should be **modular**, **tested incrementally**, and **chained together only after each stage passes its tests**.

---

## Phase 0 — Environment and Fixtures

### Goal

Create an in-memory DuckDB database with spatial support and seed it with a small table of example geometries and embeddings.

### Requirements

* Table: `embeddings(id BIGINT, geom GEOMETRY, embedding FLOAT[4])`
* Three sample points: San Francisco (zone 10N), Copenhagen (zone 33N), Sydney (zone 56S)

### Example Code

```python
import duckdb, tempfile

def make_test_connection():
    con = duckdb.connect(database=':memory:')
    con.execute('INSTALL spatial; LOAD spatial;')
    tmp = tempfile.TemporaryDirectory()
    return con, tmp

def seed_sample_data(con):
    con.execute('''
        CREATE TABLE embeddings(
          id BIGINT,
          geom GEOMETRY,
          embedding FLOAT[4]
        );
    ''')
    con.execute('''
        INSERT INTO embeddings VALUES
          (1, ST_Point(-122.4194, 37.7749), [0.1, 0.2, 0.3, 0.4]),
          (2, ST_Point(12.5683, 55.6761),   [0.2, 0.1, 0.0, 0.5]),
          (3, ST_Point(151.2093, -33.8688), [0.4, 0.3, 0.2, 0.1]);
    ''')
```

### Test Cases

* Assert `SELECT COUNT(*) FROM embeddings` = 3.
* Verify `ST_X` and `ST_Y` return correct coordinates.

---

## Phase 1 — Inference Arrow UDF

### Goal

Implement a **vectorized Arrow UDF** (`py_predict`) that performs batched inference on embeddings.

### Requirements

* Uses a `DummyModel` for testing (sigmoid of dot product).
* Supports both scikit-learn and XGBoost models.

### Example Code

```python
import numpy as np, pyarrow as pa, joblib, os, duckdb

class DummyModel:
    def predict_proba(self, X):
        z = (X @ np.array([0.5, -0.3, 0.2, 0.1], dtype=np.float32))
        p1 = 1.0 / (1.0 + np.exp(-z))
        p0 = 1.0 - p1
        return np.vstack([p0, p1]).T.astype(np.float32)

def load_model(path='model.pkl'):
    if os.path.exists(path):
        try:
            return joblib.load(path)
        except Exception:
            pass
    return DummyModel()

def py_predict_arrow(emb_col: pa.Array) -> pa.Array:
    out_chunks = []
    for chunk in emb_col.chunks:
        X = np.array(chunk.to_pylist(), dtype=np.float32)
        model = py_predict_arrow.model
        if hasattr(model, 'predict_proba'):
            p = model.predict_proba(X)[:, 1].astype(np.float32)
        else:
            p = model.predict(X).astype(np.float32)
        out_chunks.append(pa.array(p))
    return pa.chunked_array(out_chunks)

def register_py_predict(con, model):
    py_predict_arrow.model = model
    con.create_function(
        'py_predict',
        py_predict_arrow,
        parameters=[list[float]],
        return_type=duckdb.typing.FLOAT,
        type='arrow',
    )
```

### Test Cases

```python
con, tmp = make_test_connection()
seed_sample_data(con)
model = load_model()
register_py_predict(con, model)

res = con.execute('SELECT id, py_predict(embedding) AS p FROM embeddings ORDER BY id;').fetchdf()
assert len(res) == 3 and res['p'].between(0, 1).all()
```

---

## Phase 2 — UTM Zone Calculation SQL

### Goal

Generate SQL logic to determine UTM zone, hemisphere, and EPSG code.

### Example Query

```sql
WITH z AS (
  SELECT
    id,
    ST_X(geom) AS lon,
    ST_Y(geom) AS lat,
    CAST(FLOOR((ST_X(geom) + 180) / 6) + 1 AS INTEGER) AS zone,
    CASE WHEN ST_Y(geom) < 0 THEN 'S' ELSE 'N' END AS hemi
  FROM embeddings
)
SELECT
  id, zone, hemi,
  'EPSG:' || CAST((CASE WHEN hemi='S' THEN 32700 ELSE 32600 END) + zone AS VARCHAR) AS epsg
FROM z
ORDER BY id;
```

### Expected Results

| City | Zone | Hemisphere | EPSG  |
| ---- | ---- | ---------- | ----- |
| SF   | 10   | N          | 32610 |
| CPH  | 33   | N          | 32633 |
| SYD  | 56   | S          | 32756 |

---

## Phase 3 — Tile Construction per UTM Zone

### Goal

For each point with probability above threshold:

* Reproject to UTM zone.
* Create a metric tile.

### Example Query

```sql
WITH scored AS (
  SELECT id, geom, py_predict(embedding) AS p FROM embeddings
),
kept AS (
  SELECT * FROM scored WHERE p >= :threshold
),
zoned AS (
  SELECT id, geom, p,
    CAST(FLOOR((ST_X(geom)+180)/6)+1 AS INTEGER) AS zone,
    CASE WHEN ST_Y(geom) < 0 THEN 'S' ELSE 'N' END AS hemi
  FROM kept
),
reproj AS (
  SELECT id, p, zone, hemi,
    ST_Transform(
      geom, 'EPSG:4326',
      'EPSG:' || CAST((CASE WHEN hemi='S' THEN 32700 ELSE 32600 END) + zone AS VARCHAR)
    ) AS geom_utm
  FROM zoned
),
tiled AS (
  SELECT zone, hemi,
    ST_MakeEnvelope(
      ST_X(geom_utm) - (:tile_size_m / 2.0),
      ST_Y(geom_utm) - (:tile_size_m / 2.0),
      ST_X(geom_utm) + (:tile_size_m / 2.0),
      ST_Y(geom_utm) + (:tile_size_m / 2.0)
    ) AS tile_utm
  FROM reproj
)
SELECT * FROM tiled;
```

### Test Case

* With `threshold=0.0` and `tile_size_m=100`, assert each `ST_Area(tile_utm)` ≈ 10,000.

---

## Phase 4 — Unary Union and GeoJSON Export

### Goal

Merge tiles per zone and export final geometry to GeoJSON.

### Example SQL

```sql
WITH scored AS (
  SELECT id, geom, py_predict(embedding) AS p FROM embeddings
),
kept AS ( SELECT * FROM scored WHERE p >= :threshold ),
zoned AS (
  SELECT id, geom, p,
    CAST(FLOOR((ST_X(geom)+180)/6)+1 AS INTEGER) AS zone,
    CASE WHEN ST_Y(geom)<0 THEN 'S' ELSE 'N' END AS hemi
  FROM kept
),
reproj AS (
  SELECT id, p, zone, hemi,
    ST_Transform(geom, 'EPSG:4326',
      'EPSG:' || CAST((CASE WHEN hemi='S' THEN 32700 ELSE 32600 END)+zone AS VARCHAR)
    ) AS geom_utm
  FROM zoned
),
tiled AS (
  SELECT zone, hemi,
    ST_MakeEnvelope(
      ST_X(geom_utm) - (:tile_size_m/2.0),
      ST_Y(geom_utm) - (:tile_size_m/2.0),
      ST_X(geom_utm) + (:tile_size_m/2.0),
      ST_Y(geom_utm) + (:tile_size_m/2.0)
    ) AS tile_utm
  FROM reproj
),
merged AS (
  SELECT zone, hemi, ST_Union_Agg(tile_utm) AS union_utm
  FROM tiled
  GROUP BY zone, hemi
),
result AS (
  SELECT ST_Transform(
    union_utm,
    'EPSG:' || CAST((CASE WHEN hemi='S' THEN 32700 ELSE 32600 END) + zone AS VARCHAR),
    'EPSG:4326'
  ) AS geometry
  FROM merged
)
SELECT * FROM result;
```

### Example Export Helper

```python
import json

def export_geojson(con, sql_text, out_path):
    con.execute(f'''
      COPY (
        {sql_text}
      ) TO '{out_path}'
      WITH (FORMAT GDAL, DRIVER 'GeoJSON', SRS 'EPSG:4326');
    ''')
    with open(out_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    assert data['type'] == 'FeatureCollection'
    return data
```

### Test Case

* Run with `threshold=0.0`, `tile_size_m=100`.
* Verify output GeoJSON file exists and contains ≥1 feature.

---

## Phase 5 — Full Integration & CLI

### Goal

Combine all previous phases into a single callable and CLI.

### Example Python Entrypoint

```python
import argparse

def run_inference_tiles(db_path, table, id_col, geom_col, emb_col, threshold, tile_size_m, out_geojson):
    con = duckdb.connect(database=db_path or ':memory:')
    con.execute('INSTALL spatial; LOAD spatial;')
    model = load_model()
    register_py_predict(con, model)

    PIPELINE_SQL = f'''
    WITH scored AS (
      SELECT {id_col} AS id, {geom_col} AS geom, py_predict({emb_col}) AS p
      FROM {table}
    ),
    kept AS (SELECT * FROM scored WHERE p >= {threshold}),
    zoned AS (
      SELECT id, geom, p,
        CAST(FLOOR((ST_X(geom)+180)/6)+1 AS INTEGER) AS zone,
        CASE WHEN ST_Y(geom)<0 THEN 'S' ELSE 'N' END AS hemi
      FROM kept
    ),
    reproj AS (
      SELECT id, p, zone, hemi,
        ST_Transform(
          geom, 'EPSG:4326',
          'EPSG:' || CAST((CASE WHEN hemi='S' THEN 32700 ELSE 32600 END)+zone AS VARCHAR)
        ) AS geom_utm
      FROM zoned
    ),
    tiled AS (
      SELECT zone, hemi,
        ST_MakeEnvelope(
          ST_X(geom_utm) - ({tile_size_m}/2.0),
          ST_Y(geom_utm) - ({tile_size_m}/2.0),
          ST_X(geom_utm) + ({tile_size_m}/2.0),
          ST_Y(geom_utm) + ({tile_size_m}/2.0)
        ) AS tile_utm
      FROM reproj
    ),
    merged AS (
      SELECT zone, hemi, ST_Union_Agg(tile_utm) AS union_utm
      FROM tiled
      GROUP BY zone, hemi
    ),
    result AS (
      SELECT ST_Transform(
        union_utm,
        'EPSG:' || CAST((CASE WHEN hemi='S' THEN 32700 ELSE 32600 END)+zone AS VARCHAR),
        'EPSG:4326'
      ) AS geometry
      FROM merged
    )
    SELECT * FROM result;
    '''

    export_geojson(con, PIPELINE_SQL, out_geojson)
    return out_geojson

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--db', default=':memory:')
    parser.add_argument('--table', default='embeddings')
    parser.add_argument('--id_col', default='id')
    parser.add_argument('--geom_col', default='geom')
    parser.add_argument('--emb_col', default='embedding')
    parser.add_argument('--threshold', type=float, default=0.85)
    parser.add_argument('--tile_size_m', type=float, default=100.0)
    parser.add_argument('--out', default='predicted_tiles.geojson')
    args = parser.parse_args()

    output = run_inference_tiles(
        args.db, args.table, args.id_col, args.geom_col, args.emb_col,
        args.threshold, args.tile_size_m, args.out
    )
    print(f'Wrote {output}')
```

### Integration Test

```python
con, tmp = make_test_connection()
seed_sample_data(con)
out = f'{tmp.name}/out.geojson'
run_inference_tiles(':memory:', 'embeddings', 'id', 'geom', 'embedding', 0.0, 100, out)
assert os.path.exists(out)
```

---

**End of Codex Prompt**
