from pathlib import Path

from geovibes.codex_inference import (
    build_metric_tiles,
    export_geoparquet,
    filter_scores,
    load_model,
    make_test_connection,
    register_py_predict,
    score_embeddings,
    seed_sample_data,
    union_tiles,
)


def test_union_tiles_and_export_geoparquet(tmp_path):
    con, tmp = make_test_connection()
    try:
        seed_sample_data(con)
        model = load_model(path="nonexistent.pkl")
        register_py_predict(con, model)
        scored = score_embeddings(con)
        filtered = filter_scores(scored, 0.0)
        tiles = build_metric_tiles(filtered, 100.0)
        merged = union_tiles(tiles)
        df = merged.order("zone").fetchdf()
        assert list(df.columns) == [
            "zone",
            "hemi",
            "utm_srid",
            "max_probability",
            "tile_count",
            "geometry",
        ]
        assert df["tile_count"].tolist() == [1, 1, 1]
        out_path = tmp_path / "union.parquet"
        result_path = export_geoparquet(con, merged, out_path)
        assert Path(result_path).exists()
        row_count = con.execute(
            f"SELECT COUNT(*) FROM read_parquet('{result_path}')"
        ).fetchone()[0]
        assert row_count == 3
    finally:
        con.close()
        tmp.cleanup()
