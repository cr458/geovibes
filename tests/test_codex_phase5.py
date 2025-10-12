import subprocess
import sys

import duckdb

from geovibes.codex_inference import run_inference_tiles, seed_sample_data


def test_full_pipeline_and_cli(tmp_path):
    db_path = tmp_path / "pipeline.duckdb"
    con = duckdb.connect(str(db_path))
    try:
        con.execute("INSTALL spatial;")
        con.execute("LOAD spatial;")
        seed_sample_data(con)
    finally:
        con.close()
    out_path = tmp_path / "pipeline.parquet"
    run_inference_tiles(
        str(db_path),
        "embeddings",
        "id",
        "geom",
        "embedding",
        0.5,
        100.0,
        str(out_path),
        model_path="nonexistent.pkl",
    )
    tmp_con = duckdb.connect()
    try:
        row_count = tmp_con.execute(
            f"SELECT COUNT(*) FROM read_parquet('{out_path}')"
        ).fetchone()[0]
    finally:
        tmp_con.close()
    assert row_count == 3
    cli_path = tmp_path / "pipeline_cli.parquet"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "geovibes.codex_inference",
            "--db",
            str(db_path),
            "--table",
            "embeddings",
            "--id-col",
            "id",
            "--geom-col",
            "geom",
            "--emb-col",
            "embedding",
            "--threshold",
            "0.5",
            "--tile-size-m",
            "100",
            "--out",
            str(cli_path),
            "--model",
            "nonexistent.pkl",
        ],
        check=True,
    )
    assert cli_path.exists()
    cli_con = duckdb.connect()
    try:
        cli_count = cli_con.execute(
            f"SELECT COUNT(*) FROM read_parquet('{cli_path}')"
        ).fetchone()[0]
    finally:
        cli_con.close()
    assert cli_count == 3
