from geovibes.codex_inference import make_test_connection, seed_sample_data


def test_seed_sample_data_counts_and_coords():
    con, tmp = make_test_connection()
    try:
        seed_sample_data(con)
        assert con.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0] == 3
        rows = con.execute("SELECT id, ST_X(geom), ST_Y(geom) FROM embeddings ORDER BY id").fetchall()
        assert rows == [
            (1, -122.4194, 37.7749),
            (2, 12.5683, 55.6761),
            (3, 151.2093, -33.8688),
        ]
    finally:
        con.close()
        tmp.cleanup()
