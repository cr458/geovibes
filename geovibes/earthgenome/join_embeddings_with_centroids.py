import argparse
import logging
import os
import pathlib
import re
import shutil
import tempfile
from typing import List, Optional, Tuple
import io

import fsspec
import geopandas as gpd
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import geovibes.tiling as tiling
from geovibes.database.faiss_db import (
    get_cloud_protocol,
    list_cloud_parquet_files,
    download_cloud_files,
    find_embedding_files_for_mgrs_ids,
)


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')



def discover_embedding_files(
    roi_file: Optional[str],
    mgrs_reference_file: Optional[str],
    embedding_dir: Optional[str],
    explicit_paths: List[str],
) -> Tuple[List[str], List[str]]:
    all_local_files: List[str] = []
    cloud_files: List[str] = []

    if roi_file:
        mgrs_tile_ids = tiling.get_mgrs_tile_ids_for_roi_from_roi_file(
            roi_geojson_file=roi_file,
            mgrs_tiles_file=mgrs_reference_file,
        )
        mgrs_ids = [str(t) for t in mgrs_tile_ids]
        logging.info(f"Found {len(mgrs_ids)} MGRS tiles intersecting with ROI")
        if not mgrs_ids:
            logging.error("No MGRS tiles found intersecting with ROI")
            return [], []
        embedding_files = find_embedding_files_for_mgrs_ids(mgrs_ids, embedding_dir or "")
        if not embedding_files:
            logging.error("No embedding files found for intersecting MGRS tiles")
            return [], []
        
        # Find which MGRS IDs have corresponding files
        found_mgrs_ids = set()
        for file_path in embedding_files:
            mgrs_id = extract_mgrs_id_from_filename(file_path)
            if mgrs_id:
                found_mgrs_ids.add(mgrs_id)
        
        # Log missing MGRS IDs
        missing_mgrs_ids = set(mgrs_ids) - found_mgrs_ids
        if missing_mgrs_ids:
            logging.warning(f"No embedding files found for {len(missing_mgrs_ids)} MGRS IDs: {sorted(missing_mgrs_ids)}")
        
        for file_path in embedding_files:
            if get_cloud_protocol(file_path):
                cloud_files.append(file_path)
            else:
                all_local_files.append(file_path)
        logging.info(
            f"Found {len(embedding_files)} embedding files for ROI ({len(all_local_files)} local, {len(cloud_files)} cloud)"
        )
    else:
        for path in explicit_paths:
            if get_cloud_protocol(path):
                if path.endswith('.parquet'):
                    cloud_files.append(path)
                else:
                    cloud_parquet_files = list_cloud_parquet_files(path)
                    cloud_files.extend(cloud_parquet_files)
                    logging.info(f"Found {len(cloud_parquet_files)} parquet files in {path}")
            else:
                path_obj = pathlib.Path(path)
                if path_obj.is_file() and path_obj.suffix == '.parquet':
                    all_local_files.append(str(path_obj))
                elif path_obj.is_dir():
                    all_local_files.extend([str(p) for p in path_obj.glob('**/*.parquet')])
                else:
                    import glob as _glob
                    all_local_files.extend(_glob.glob(path))

    return all_local_files, cloud_files



def list_tiles_files_for_mgrs(tiles_dir: str, mgrs_id: str) -> List[str]:
    protocol = get_cloud_protocol(tiles_dir)
    pattern = f"{mgrs_id}_*.parquet"
    if protocol:
        fs = fsspec.filesystem(protocol)
        dir_path = tiles_dir if tiles_dir.endswith('/') else tiles_dir + '/'
        matches = fs.glob(dir_path + pattern)
        return [f"{protocol}://{m}" for m in matches]
    else:
        path_obj = pathlib.Path(tiles_dir)
        return [str(p) for p in path_obj.glob(pattern)]



def extract_mgrs_id_from_filename(filename: str) -> Optional[str]:
    base = os.path.basename(filename)
    m = re.search(r"(\d{1,2}[A-Z][A-Z]{2})", base)
    return m.group(1) if m else None



def extract_epsg_from_tiles_filename(filename: str) -> Optional[str]:
    base = os.path.basename(filename)
    m = re.match(r"^(?P<mgrs>\d{1,2}[A-Z][A-Z]{2})_(?P<epsg>\d{4,5})_", base)
    if not m:
        return None
    return f"EPSG:{m.group('epsg')}"



def ensure_tiles_gdf_in_native_crs(tiles_path: str) -> gpd.GeoDataFrame:
    gdf = gpd.read_parquet(tiles_path)
    expected_crs = extract_epsg_from_tiles_filename(tiles_path)
    if expected_crs is None:
        raise ValueError(f"Could not parse EPSG from tiles filename: {tiles_path}")
    if gdf.crs is None:
        gdf.set_crs(expected_crs, inplace=True)
    elif str(gdf.crs) != expected_crs:
        gdf = gdf.to_crs(expected_crs)
    return gdf



def compute_centroids_wgs84(tiles_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    centroids = tiles_gdf.copy()
    centroids['geometry'] = tiles_gdf.geometry.centroid
    if str(centroids.crs) != 'EPSG:4326':
        centroids = centroids.to_crs('EPSG:4326')
    return centroids[['tile_id', 'geometry']]



def build_embedding_column(df: pd.DataFrame) -> pd.DataFrame:
    if 'embedding' in df.columns:
        return df
    vit_cols = [c for c in df.columns if c.lower().startswith('vit')]
    if not vit_cols:
        raise ValueError("No 'embedding' column and no 'vit*' feature columns found")
    def _key(name: str) -> Tuple[str, int]:
        m = re.search(r"(\d+)$", name)
        return (name, int(m.group(1)) if m else -1)
    vit_cols_sorted = sorted(vit_cols, key=lambda x: _key(x))
    emb = df[vit_cols_sorted].to_numpy(dtype=np.float32)
    df = df.copy()
    df['embedding'] = list(emb)
    return df[['tile_id', 'embedding']]



def join_one(embedding_path: str, tiles_dir: str, temp_dir: Optional[str], output_dir: str) -> Optional[str]:
    mgrs_id = extract_mgrs_id_from_filename(embedding_path)
    if mgrs_id is None:
        logging.warning(f"Could not extract MGRS id from {embedding_path}; skipping")
        return None
    candidates = list_tiles_files_for_mgrs(tiles_dir, mgrs_id)
    if not candidates:
        logging.warning(f"No tiles file found for MGRS {mgrs_id} in {tiles_dir}; skipping")
        return None
    tiles_path = candidates[0]
    local_tiles_path = tiles_path
    if get_cloud_protocol(tiles_path):
        tiles_tmp = temp_dir or tempfile.mkdtemp(prefix="eg_tiles_")
        if temp_dir:
            tiles_tmp = os.path.join(temp_dir, "tiles", mgrs_id)
            os.makedirs(tiles_tmp, exist_ok=True)
        dl = download_cloud_files([tiles_path], tiles_tmp)
        if not dl:
            logging.warning(f"Failed to download tiles file {tiles_path}")
            return None
        local_tiles_path = dl[0]

    tiles_gdf = ensure_tiles_gdf_in_native_crs(local_tiles_path)
    centroids = compute_centroids_wgs84(tiles_gdf)

    if get_cloud_protocol(embedding_path):
        emb_tmp = temp_dir or tempfile.mkdtemp(prefix="eg_emb_")
        if temp_dir:
            emb_base = os.path.splitext(os.path.basename(embedding_path))[0]
            emb_tmp = os.path.join(temp_dir, "emb", emb_base)
            os.makedirs(emb_tmp, exist_ok=True)
        dl = download_cloud_files([embedding_path], emb_tmp)
        if not dl:
            logging.warning(f"Failed to download embedding file {embedding_path}")
            return None
        embedding_local = dl[0]
    else:
        embedding_local = embedding_path

    edf = pd.read_parquet(embedding_local)
    edf = build_embedding_column(edf)

    merged = centroids.merge(edf, on='tile_id', how='inner')
    if merged.empty:
        logging.warning(f"Join produced no rows for {os.path.basename(embedding_path)}; skipping")
        return None

    base_name = os.path.basename(embedding_path)
    if output_dir.startswith("s3://"):
        endpoint = os.environ.get("S3_ENDPOINT_URL", "https://data.source.coop")
        dest = output_dir.rstrip('/') + '/' + base_name
        buffer = io.BytesIO()
        gpd.GeoDataFrame(merged, geometry='geometry', crs='EPSG:4326').to_parquet(buffer, index=False)
        buffer.seek(0)
        fs = fsspec.filesystem("s3", client_kwargs={"endpoint_url": endpoint})
        with fs.open(dest, "wb") as f:
            f.write(buffer.read())
        return dest
    else:
        out_path = os.path.join(output_dir, base_name)
        gpd.GeoDataFrame(merged, geometry='geometry', crs='EPSG:4326').to_parquet(out_path, index=False)
        return out_path



def main() -> None:
    parser = argparse.ArgumentParser(description="Join embedding parquet files with centroid geometries computed from tiles.")
    parser.add_argument("embedding_inputs", nargs='*', help="Embedding parquet files or directories (local or gs://)")
    parser.add_argument("--roi-file", dest="roi_file", help="ROI geometry file to intersect with MGRS tiles for automatic file discovery")
    parser.add_argument("--mgrs-reference-file", dest="mgrs_reference_file", help="MGRS reference GeoParquet for ROI discovery (required with --roi-file)")
    parser.add_argument("--embedding-dir", dest="embedding_dir", help="Directory containing embedding parquet files (required with --roi-file)")
    parser.add_argument("--tiles-dir", dest="tiles_dir", required=True, help="Directory containing tiles geometries (local or gs://)")
    parser.add_argument("--output-dir", dest="output_dir", required=True, help="Directory to write output parquet files")
    parser.add_argument("--dry-run", action="store_true", help="Process only a small subset of files")
    parser.add_argument("--dry-run-size", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=8, help="Number of parallel workers")

    args = parser.parse_args()
    setup_logging()

    if getattr(args, "roi_file", None):
        if not getattr(args, "mgrs_reference_file", None):
            parser.error("--mgrs-reference-file is required when using --roi-file")
        if not getattr(args, "embedding_dir", None):
            parser.error("--embedding-dir is required when using --roi-file")

    local_files, cloud_files = discover_embedding_files(
        roi_file=getattr(args, "roi_file", None),
        mgrs_reference_file=getattr(args, "mgrs_reference_file", None),
        embedding_dir=getattr(args, "embedding_dir", None),
        explicit_paths=getattr(args, "embedding_inputs", []) or [],
    )

    temp_dir = None
    if cloud_files:
        temp_dir = tempfile.mkdtemp(prefix="eg_join_")
        logging.info(f"Downloading {len(cloud_files)} embedding files...")
        local_cloud_files = download_cloud_files(cloud_files, temp_dir)
        local_files.extend(local_cloud_files)

    local_files = list(dict.fromkeys(local_files))
    if not local_files:
        logging.error("No embedding parquet files to process")
        if temp_dir:
            shutil.rmtree(temp_dir)
        return

    if args.dry_run:
        local_files = local_files[: args.dry_run_size]
        logging.info(f"Dry run: limiting to {len(local_files)} files")

    if not get_cloud_protocol(args.output_dir):
        pathlib.Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    def _worker(emb_path: str) -> Optional[str]:
        try:
            return join_one(emb_path, args.tiles_dir, temp_dir, args.output_dir)
        except Exception as e:
            logging.exception(f"Failed to process {emb_path}: {e}")
            return None

    results = Parallel(
        n_jobs=args.num_workers, prefer="threads", verbose=10)(delayed(_worker)(emb) for emb in local_files)
    written = [r for r in results if r]
    for out in written:
        logging.info(f"Wrote {out}")

    if temp_dir:
        shutil.rmtree(temp_dir)
        logging.info(f"Cleaned up temporary directory: {temp_dir}")

    logging.info(f"Completed. Wrote {len(written)} files to {args.output_dir}")



if __name__ == "__main__":
    main()

