import numpy as np
import pytest

from geovibes.codex_inference import (
    filter_scores,
    load_model,
    make_test_connection,
    register_py_predict,
    score_embeddings,
    seed_sample_data,
)


def test_score_and_filter_embeddings():
    con, tmp = make_test_connection()
    try:
        seed_sample_data(con)
        model = load_model(path="nonexistent.pkl")
        register_py_predict(con, model)
        scored = score_embeddings(con)
        df = scored.order("id").fetchdf()
        assert list(df.columns) == ["id", "geom", "probability"]
        expected = np.array([0.52248484, 0.529964, 0.5399149], dtype=np.float32)
        assert np.allclose(df["probability"].to_numpy(), expected, atol=1e-6)
        filtered = filter_scores(scored, 0.53)
        filtered_ids = filtered.order("id").project("id").fetchall()
        assert filtered_ids == [(3,)]
    finally:
        con.close()
        tmp.cleanup()
