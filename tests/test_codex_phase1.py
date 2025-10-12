import numpy as np

from geovibes.codex_inference import (
    load_model,
    make_test_connection,
    register_py_predict,
    seed_sample_data,
)


def test_py_predict_arrow_outputs_probabilities():
    con, tmp = make_test_connection()
    try:
        seed_sample_data(con)
        model = load_model(path="nonexistent.pkl")
        register_py_predict(con, model)
        df = con.execute(
            "SELECT id, py_predict(embedding) AS p FROM embeddings ORDER BY id"
        ).fetchdf()
        assert len(df) == 3
        assert np.all((df["p"] >= 0.0) & (df["p"] <= 1.0))
    finally:
        con.close()
        tmp.cleanup()
