# AGENTS.md — GeoVibes Agent Notes

- Purpose: GeoVibes is a Jupyter-driven geospatial similarity explorer that lets users label points/polygons on a map, update a query vector (`2*pos_avg - neg_avg`), and search a FAISS index backed by DuckDB metadata.
- Setup: `uv venv .venv && source .venv/bin/activate`, `uv pip install -e .`, `python -m ipykernel install --user --name geovibes --display-name "Python (geovibes)"`.
- Data prep: Run `uv run download_embeddings.py` to interactively pull geometries and DuckDB/FAISS bundles listed in `manifest.csv` into `geometries/` and `local_databases/` (resumable downloads). Start with lighter Google or quantized DBs.
- Notebook: `uv run jupyter lab` then open `vibe_checker.ipynb` and select the `Python (geovibes)` kernel. The UI builds inside the notebook.
- Config: optional `config.yaml` with `start_date`, `end_date`, `enable_ee`. Required env var `MAPTILER_API_KEY` (for basemap tiles); optional `GEOVIBES_ENABLE_EE=1` to allow Sentinel-2 NDVI/NDWI basemaps after `earthengine authenticate`. For GCS-hosted DBs, `GCS_ACCESS_KEY_ID` / `GCS_SECRET_ACCESS_KEY` or default gcloud auth. `.env` is loaded from repo root.

## Architecture Cheat Sheet
- Entry point: `geovibes.ui.app.GeoVibes.create(...)` wires together `DataManager`, `AppState`, `MapManager`, `DatasetManager`, `TilePanel`, `StatusBus`.
- Data discovery: `DataManager` autoloads the first available DB/FAISS pair from `duckdb_path`, `duckdb_directory`, or downloaded artifacts referenced by `manifest.csv`; infers tile specs from filenames like `{name}_{size}_{overlap}_{resolution}` and looks for matching geometry files under `geometries/`.
- Basemaps: defaults to MapTiler; adds Google Hybrid and Hutch tiles. Earth Engine basemaps (S2 RGB/NDVI/NDWI/HSV) are added only when `enable_ee` is true and auth succeeds.
- Labeling/search flow: map click → `DataManager.nearest_point` → `AppState.apply_label` → `AppState.update_query_vector` → FAISS search (`nprobe=4096`, requests `neighbors + half(labels)` then drops already-labeled IDs) → map layer + tile panel update with color-coded distances.
- Tile panel: async thumbnail fetch via `ThreadPoolExecutor` in `geovibes/ui/tiles.py`, uses XYZ sources from `BasemapConfig` and `get_map_image` to center tiles based on `tile_spec` coverage.
- Dataset persistence: `DatasetManager.save_dataset()` writes a timestamped GeoJSON with embeddings and label metadata; `load_from_content` accepts GeoJSON or Parquet with `id/label/embedding` columns.
- Database builder: `geovibes/database/faiss_db.py` ingests Parquet embeddings (local/S3/GCS), builds DuckDB `geo_embeddings` and FAISS IVF-PQ or IVF-SQ indexes, and can filter by ROI/MGRS grids (`geovibes/tiling.py`). Requires DuckDB `httpfs`/`spatial` extensions.

## Common Commands
- Run tests: `pytest` (unit tests live in `tests/`).
- Build DB example: `python geovibes/database/faiss_db.py --roi-file geometries/alabama.geojson --mgrs-reference-file geometries/mgrs_tiles.parquet --embedding-dir s3://... --name model_name --tile-pixels 32 --tile-overlap 16 --tile-resolution 10 --output_dir local_databases` (add `--dry-run --dry-run-size 5` to sample).
- Launch notebook UI quickly from Python: `from geovibes import GeoVibes; GeoVibes.create(duckdb_directory="local_databases")`.

## Gotchas
- No models bundled: ensure `local_databases/` and `geometries/` exist or pass explicit `duckdb_path`/`duckdb_directory`; otherwise `DataManager` raises `FileNotFoundError`.
- Missing `MAPTILER_API_KEY` only affects basemap imagery; core search still works. Earth Engine is entirely optional.
- `DataManager` sets DuckDB to read-only and loads `httpfs` automatically for GCS paths; avoid writing to those DBs.
- Tile thumbnails rely on network XYZ sources; if offline, tiles may show “Image unavailable” but search results still render as map points.
