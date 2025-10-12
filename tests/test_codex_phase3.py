import numpy as np

from geovibes.codex_inference import (
    build_metric_tiles,
    filter_scores,
    load_model,
    make_test_connection,
    register_py_predict,
    score_embeddings,
    seed_sample_data,
)


def test_build_metric_tiles_generates_metric_squares():
    con, tmp = make_test_connection()
    try:
        seed_sample_data(con)
        model = load_model(path="nonexistent.pkl")
        register_py_predict(con, model)
        scored = score_embeddings(con)
        filtered = filter_scores(scored, 0.0)
        tiles = build_metric_tiles(filtered, 100.0)
        df = tiles.order("id").fetchdf()
        assert list(df.columns) == ["id", "probability", "zone", "hemi", "utm_srid", "tile_utm"]
        zones = df["zone"].tolist()
        assert zones == [10, 33, 56]
        dims = tiles.query(
            "tiles",
            "SELECT ST_XMax(tile_utm) - ST_XMin(tile_utm) AS w, ST_YMax(tile_utm) - ST_YMin(tile_utm) AS h FROM tiles ORDER BY id",
        ).fetchall()
        for w, h in dims:
            assert np.isclose(w, 100.0, atol=1e-6)
            assert np.isclose(h, 100.0, atol=1e-6)
    finally:
        con.close()
        tmp.cleanup()
