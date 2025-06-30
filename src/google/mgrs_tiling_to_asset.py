"""
Generate a grid of tiles over an MGRS tile and save it as a GeoParquet file.
"""
import argparse
from dataclasses import dataclass, field

import geopandas as gpd
import numpy as np
import pyproj
import shapely.geometry
import shapely.ops
import pandas as pd
import pyproj
import os
import tempfile
import zipfile
import subprocess
from google.cloud import storage
from tqdm import tqdm
import shutil
from joblib import Parallel, delayed


def get_crs_from_tile(tile_series: pd.Series) -> str:
    """
    Get the CRS from a tile series by reading the 'epsg' column.
    """
    try:
        epsg_code = tile_series['epsg']
        return f"EPSG:{epsg_code}"
    except KeyError:
        raise ValueError("Input series must have an 'epsg' column.")


@dataclass
class MGRSTileGrid:
    """Class for tracking a MGRS tile grid"""
    mgrs_tile_id: str
    crs: str
    tilesize: int
    overlap: int
    resolution: float
    prefix: str = field(init=False)

    def __post_init__(self):
        self.prefix = f"{self.mgrs_tile_id}_{self.crs.split(':')[-1]}_{self.tilesize}_{self.overlap}_{int(self.resolution)}"


def chip_mgrs_tile(tile_series: pd.Series, mgrs_tile_grid: MGRSTileGrid, source_crs: pyproj.CRS) -> gpd.GeoDataFrame:
    """
    Top level function to generate chips over an MGRS tile
    """
    xform_utm = pyproj.Transformer.from_crs(source_crs, mgrs_tile_grid.crs, always_xy=True)
    tile_geom_utm = shapely.ops.transform(xform_utm.transform, tile_series.geometry)

    eff_tilesize = mgrs_tile_grid.tilesize * mgrs_tile_grid.resolution
    eff_overlap = mgrs_tile_grid.overlap * mgrs_tile_grid.resolution
    grid_spacing = eff_tilesize - eff_overlap

    bounds_utm = tile_geom_utm.bounds
    sw_utm = bounds_utm[0], bounds_utm[1]
    ne_utm = bounds_utm[2], bounds_utm[3]

    x_diff = ne_utm[0] - sw_utm[0]
    y_diff = ne_utm[1] - sw_utm[1]

    x_samples = round(x_diff / grid_spacing) + 1
    y_samples = round(y_diff / grid_spacing) + 1

    xs = np.arange(0, x_samples) * grid_spacing + sw_utm[0]
    ys = np.arange(0, y_samples) * grid_spacing + sw_utm[1]

    x_grid, y_grid = np.meshgrid(xs, ys)

    return generate_chips(
        x_samples=x_samples,
        y_samples=y_samples,
        x_grid=x_grid,
        y_grid=y_grid,
        eff_tilesize=eff_tilesize,
        mgrs_tile_grid=mgrs_tile_grid,
        tile_geom_utm=tile_geom_utm,
    )


def generate_chips(
    x_samples: int,
    y_samples: int,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    eff_tilesize: float,
    mgrs_tile_grid: MGRSTileGrid,
    tile_geom_utm: shapely.geometry.Polygon,
) -> gpd.GeoDataFrame:
    """
    Generate chips over a grid and return them as a GeoDataFrame.
    """
    tiles = []
    for i in range(x_samples):
        for j in range(y_samples):
            x, y = x_grid[j, i], y_grid[j, i]
            geom = shapely.geometry.Point(x, y).buffer(eff_tilesize / 2, cap_style=3)

            if tile_geom_utm.intersects(geom):
                tile = {
                    'geometry': geom,
                    'tile_id': f"{mgrs_tile_grid.mgrs_tile_id}_{mgrs_tile_grid.tilesize}_{mgrs_tile_grid.overlap}_{int(mgrs_tile_grid.resolution)}_{j}_{i}"
                }
                tiles.append(tile)

    return gpd.GeoDataFrame(tiles, crs=mgrs_tile_grid.crs)


def write_tiles_to_geoparquet(tiles: gpd.GeoDataFrame, tile_name: str, output_dir: str = "."):
    """
    Write a GeoDataFrame of chips to a GeoParquet file locally.
    """
    output_path = f"{output_dir}/{tile_name}.parquet"
    tiles.to_parquet(output_path)
    print(f"Wrote {len(tiles)} tiles to {output_path}")


def check_gcs_file_exists(gcs_bucket: str, blob_name: str) -> bool:
    """
    Check if a file exists in GCS bucket.
    
    Args:
        gcs_bucket: GCS bucket name
        blob_name: Full blob path (including any prefix)
        
    Returns:
        True if file exists, False otherwise
    """
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(gcs_bucket)
        blob = bucket.blob(blob_name)
        return blob.exists()
    except Exception as e:
        print(f"    Warning: Could not check GCS file existence: {e}")
        return False


def create_local_shapefile_zip(tiles: gpd.GeoDataFrame, tile_name: str, output_dir: str):
    """
    Creates a zipped shapefile locally.
    Returns tuple: (success: bool, message: str, zip_path: str)
    """
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save shapefile to temp dir
            shapefile_path = os.path.join(tmpdir, f"{tile_name}.shp")
            tiles.to_file(shapefile_path, driver='ESRI Shapefile')

            # Zip the shapefile components
            zip_path = os.path.join(tmpdir, f"{tile_name}.zip")
            with zipfile.ZipFile(zip_path, 'w') as zipf:
                for ext in ['.shp', '.shx', '.dbf', '.prj', '.cpg']:
                    source_file = f"{shapefile_path[:-4]}{ext}"
                    if os.path.exists(source_file):
                        zipf.write(source_file, arcname=os.path.basename(source_file))

            # Save zipped shapefile locally
            output_zip_path = os.path.join(output_dir, f"{tile_name}.zip")
            shutil.copy(zip_path, output_zip_path)
            
            return (True, f"Created local zip: {output_zip_path}", output_zip_path)
            
    except Exception as e:
        return (False, f"Failed to create local zip: {str(e)}", "")


def upload_to_gcs_with_gcloud(local_zip_path: str, gcs_bucket: str, gcs_prefix: str):
    """
    Upload a local file to GCS using gcloud command.
    Returns tuple: (success: bool, message: str, gcs_uri: str)
    """
    try:
        filename = os.path.basename(local_zip_path)
        gcs_path = f"{gcs_prefix}/{filename}" if gcs_prefix else filename
        gcs_uri = f"gs://{gcs_bucket}/{gcs_path}"
        
        command = [
            'gcloud', 'storage', 'cp', 
            local_zip_path,
            gcs_uri
        ]
        
        result = subprocess.run(command, check=True)
        return (True, f"Successfully uploaded to {gcs_uri}", gcs_uri)
        
    except subprocess.CalledProcessError as e:
        return (False, f"gcloud upload failed: {e.stderr}", "")
    except Exception as e:
        return (False, f"Upload failed: {str(e)}", "")


def create_gee_asset(gcs_uri: str, gee_asset_path: str, tile_name: str):
    """
    Create a GEE asset from a GCS file.
    Returns tuple: (success: bool, message: str, task_id: str)
    """
    try:
        asset_id = f"{gee_asset_path}/{tile_name}"
        command = [
            'earthengine', 'upload', 'table',
            f'--asset_id={asset_id}',
            gcs_uri
        ]
        
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        task_id = result.stdout.strip()
        
        return (True, f"Started GEE task: {task_id}", task_id)
        
    except subprocess.CalledProcessError as e:
        return (False, f"GEE upload failed: {e.stderr}", "")
    except Exception as e:
        return (False, f"GEE upload failed: {str(e)}", "")


def check_gee_asset_exists(asset_id: str) -> bool:
    """Check if a GEE asset already exists."""
    try:
        command = ['earthengine', 'asset', 'info', asset_id]
        subprocess.run(command, check=True, capture_output=True, text=True)
        return True
    except subprocess.CalledProcessError:
        return False
    except Exception:
        # If any other error occurs, assume it doesn't exist to be safe
        return False


def batch_upload_to_gcs(local_files: list, gcs_bucket: str, gcs_prefix: str, debug: bool = False):
    """
    Upload multiple files to GCS using gcloud storage cp.
    Returns tuple: (success: bool, message: str)
    """
    try:
        if not local_files:
            return (True, "No files to upload")
        
        # Build gcloud command for batch upload
        gcs_dest = f"gs://{gcs_bucket}/{gcs_prefix}/" if gcs_prefix else f"gs://{gcs_bucket}/"
        
        # Use gcloud storage cp - it automatically uses parallel uploads for multiple files
        command = ['gcloud', 'storage', 'cp']
        
        # Add all local files
        command.extend(local_files)
        
        # Add destination
        command.append(gcs_dest)
        
        if debug:
            print(f"Running: {' '.join(command)}")
        
        # Run command and allow output to stream to console
        result = subprocess.run(command, check=True)
        
        return (True, f"Successfully uploaded {len(local_files)} files")
        
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr if e.stderr else str(e)
        return (False, f"gcloud upload failed: {error_msg}")
    except Exception as e:
        return (False, f"Upload failed: {str(e)}")


def process_single_tile(tile_series, source_crs, tilesize, overlap, resolution, buffer_m, 
                       roi_geometry, roi_crs, output_dir, gcs_bucket, gcs_prefix, gee_asset_path, debug=False):
    """
    Process a single MGRS tile: generate chips, optionally filter by ROI, and upload to GCS/GEE.
    
    Args:
        tile_series: pandas Series with MGRS tile data
        source_crs: Source CRS of the MGRS tiles
        tilesize: Tile size in pixels
        overlap: Overlap in pixels
        resolution: Resolution in meters per pixel
        buffer_m: Buffer distance in meters for ROI filtering
        roi_geometry: ROI geometry for filtering (or None)
        roi_crs: CRS of the ROI geometry
        output_dir: Directory to save outputs
        gcs_bucket: GCS bucket name (or None for local only)
        gee_asset_path: GEE asset path (or None for local only)
        
    Returns:
        dict: Processing results with tile_id, success status, and message
    """
    tile_id = tile_series.mgrs_id
    try:
        if debug:
            print(f"Processing MGRS tile: {tile_id}")
        crs = get_crs_from_tile(tile_series)
        grid = MGRSTileGrid(
            mgrs_tile_id=tile_id,
            crs=crs,
            tilesize=tilesize,
            overlap=overlap,
            resolution=resolution,
        )
        
        # Check if file already exists in GCS (only if uploading to GCS/GEE)
        if gcs_bucket and gee_asset_path:
            blob_name = f"{gcs_prefix}/{grid.prefix}.zip" if gcs_prefix else f"{grid.prefix}.zip"
            if check_gcs_file_exists(gcs_bucket, blob_name):
                return {
                    'tile_id': tile_id,
                    'success': True,
                    'message': f"Skipped - file already exists in GCS: gs://{gcs_bucket}/{blob_name}",
                    'chips_generated': 0,
                    'chips_saved': 0,
                    'zip_path': '',
                    'tile_name': grid.prefix
                }
        
        # Check if local zip file already exists
        local_zip_path = os.path.join(output_dir, f"{grid.prefix}.zip")
        if os.path.exists(local_zip_path):
            return {
                'tile_id': tile_id,
                'success': True,
                'message': f"Skipped generation - local zip file already exists: {local_zip_path}",
                'chips_generated': 0,
                'chips_saved': 0,
                'zip_path': local_zip_path,
                'tile_name': grid.prefix
            }
        
        # Generate chips
        tiles = chip_mgrs_tile(tile_series, grid, source_crs=source_crs)
        initial_chip_count = len(tiles)
        if debug:
            print(f"    Generated {initial_chip_count} initial chips for {tile_id}")

        # Filter by ROI if provided
        if roi_geometry and len(tiles) > 0:
            transformer = pyproj.Transformer.from_crs(roi_crs, grid.crs, always_xy=True)
            roi_utm = shapely.ops.transform(transformer.transform, roi_geometry)
            buffered_roi_utm = roi_utm.buffer(buffer_m)
            
            intersecting_mask = tiles.intersects(buffered_roi_utm)
            tiles = tiles[intersecting_mask]
            if debug:
                print(f"    Post-filtering: Kept {len(tiles)} of {initial_chip_count} chips intersecting with the {buffer_m}m buffered ROI for {tile_id}")

        final_chip_count = len(tiles)
        
        if final_chip_count > 0:
            if gcs_bucket and gee_asset_path:
                # Create local zipped shapefile for later upload
                if debug:
                    print(f"    Creating local zip for {final_chip_count} chips for {tile_id}...")
                success, zip_message, zip_path = create_local_shapefile_zip(tiles, grid.prefix, output_dir)
                if success:
                    message = f"Successfully processed and created local zip"
                    if debug:
                        print(f"    Zip creation success for {tile_id}: {zip_message}")
                else:
                    # Return as failed if zip creation failed
                    if debug:
                        print(f"    Zip creation failed for {tile_id}: {zip_message}")
                    return {
                        'tile_id': tile_id,
                        'success': False,
                        'message': f"Zip creation failed: {zip_message}",
                        'chips_generated': initial_chip_count,
                        'chips_saved': 0,
                        'zip_path': '',
                        'tile_name': grid.prefix
                    }
            else:
                write_tiles_to_geoparquet(tiles, grid.prefix, output_dir)
                message = f"Successfully processed and saved locally"
                zip_path = ''
        else:
            message = f"No chips to save after filtering"
            zip_path = ''

        return {
            'tile_id': tile_id,
            'success': True,
            'message': message,
            'chips_generated': initial_chip_count,
            'chips_saved': final_chip_count,
            'zip_path': zip_path,
            'tile_name': grid.prefix
        }

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        return {
            'tile_id': tile_id,
            'success': False,
            'message': f"Error: {str(e)}",
            'error_details': error_details,
            'chips_generated': 0,
            'chips_saved': 0,
            'gcs_uri': ''
        }


def main():
    parser = argparse.ArgumentParser(
        description="Generate tiling grid for MGRS tiles from a file.",
        epilog="""
WORKFLOW:
1. Generate all tiles locally as zip files (parallel)
2. Upload all zip files to GCS using gcloud (batch)
3. Create all GEE assets from uploaded files (batch)

EXAMPLE:
   python mgrs_tiling_to_asset.py --input_file tiles.parquet --roi_file roi.geojson --gcs_bucket mybucket --gee_asset_path mypath
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input_file", type=str, required=True, help="Path to GeoParquet or GeoJSON file with MGRS tile geometries.")
    parser.add_argument("--roi_file", type=str, help="Path to a GeoJSON/GeoParquet file to filter MGRS tiles.")
    parser.add_argument("--tilesize", type=int, default=100, help="Tile size in pixels.")
    parser.add_argument("--overlap", type=int, default=0, help="Overlap in pixels.")
    parser.add_argument("--resolution", type=float, default=10.0, help="Resolution in meters per pixel.")
    parser.add_argument("--buffer_m", type=float, default=100.0, help="Buffer distance in meters for post-filtering chips against the ROI.")
    parser.add_argument("--output_dir", type=str, default=".", help="Directory to save the output GeoParquet files.")
    parser.add_argument("--gcs_bucket", type=str, default='geovibes', help="GCS bucket name to upload zipped shapefiles to.")
    parser.add_argument("--gcs_prefix", type=str, default='tiles', help="GCS prefix/folder within the bucket.")
    parser.add_argument("--gee_asset_path", type=str, default='projects/demeterlabs-gee/assets/tiles', help="GEE asset path for table uploads (e.g., 'users/username/folder').")
    parser.add_argument("--n_jobs", type=int, default=1, help="Number of parallel jobs to run. Use -1 for all available cores.")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode for troubleshooting.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    try:
        if args.input_file.endswith(".parquet"):
            mgrs_gdf = gpd.read_parquet(args.input_file)
        elif args.input_file.endswith(".geojson"):
            mgrs_gdf = gpd.read_file(args.input_file)
        else:
            raise ValueError("Input file must be a .parquet or .geojson file.")
    except Exception as e:
        raise IOError(f"Could not read input file: {args.input_file}") from e

    roi_geometry = None
    if args.roi_file:
        print(f"Filtering MGRS tiles by ROI: {args.roi_file}")
        try:
            if args.roi_file.endswith(".parquet"):
                roi_gdf = gpd.read_parquet(args.roi_file)
            elif args.roi_file.endswith(".geojson"):
                roi_gdf = gpd.read_file(args.roi_file)
            else:
                raise ValueError("ROI file must be a .parquet or .geojson file.")
        except Exception as e:
            raise IOError(f"Could not read ROI file: {args.roi_file}") from e

        if mgrs_gdf.crs != roi_gdf.crs:
            print(f"Warning: MGRS file CRS ({mgrs_gdf.crs}) and ROI file CRS ({roi_gdf.crs}) differ. Reprojecting ROI to match MGRS for intersection.")
            roi_gdf = roi_gdf.to_crs(mgrs_gdf.crs)

        roi_geometry = roi_gdf.union_all()
        intersecting_mask = mgrs_gdf.intersects(roi_geometry)
        mgrs_gdf = mgrs_gdf[intersecting_mask]
        print(f"Found {len(mgrs_gdf)} MGRS tiles intersecting with the ROI.")
        roi_crs = roi_gdf.crs
    else:
        roi_crs = None

    # Prepare tile data for parallel processing
    tile_list = [row for _, row in mgrs_gdf.iterrows()]
    
    print("\n" + "="*80)
    print("PHASE 1: GENERATING LOCAL FILES")
    print("="*80)
    print(f"Processing {len(tile_list)} MGRS tiles using {args.n_jobs} parallel jobs...")
    print(f"Output directory: {args.output_dir}")
    
    if args.gcs_bucket and args.gee_asset_path:
        gcs_path = f"gs://{args.gcs_bucket}/{args.gcs_prefix}" if args.gcs_prefix else f"gs://{args.gcs_bucket}"
        print(f"Will upload to: {gcs_path}")
        print(f"GEE asset path: {args.gee_asset_path}")
    else:
        print("No GCS/GEE configuration - local files only")
    
    # Process tiles in parallel
    results = Parallel(n_jobs=args.n_jobs, verbose=10)(
        delayed(process_single_tile)(
            tile_series=tile_series,
            source_crs=mgrs_gdf.crs,
            tilesize=args.tilesize,
            overlap=args.overlap,
            resolution=args.resolution,
            buffer_m=args.buffer_m,
            roi_geometry=roi_geometry,
            roi_crs=roi_crs,
            output_dir=args.output_dir,
            gcs_bucket=args.gcs_bucket,  # Always pass for file existence check
            gcs_prefix=args.gcs_prefix,
            gee_asset_path=args.gee_asset_path,
            debug=args.debug
        ) for tile_series in tile_list
    )
    
    # Print summary of Phase 1
    print("\n" + "="*80)
    print("PHASE 1 SUMMARY")
    print("="*80)
    
    successful_tiles = [r for r in results if r['success']]
    failed_tiles = [r for r in results if not r['success']]
    skipped_local = [r for r in results if r['success'] and 'local zip file already exists' in r['message']]
    skipped_gcs = [r for r in results if r['success'] and 'already exists in GCS' in r['message']]
    processed_tiles = [r for r in results if r['success'] and 'Skipped' not in r['message']]
    
    total_chips_generated = sum(r['chips_generated'] for r in results)
    total_chips_saved = sum(r['chips_saved'] for r in results)
    
    print(f"Total tiles: {len(results)}")
    print(f"Successfully processed: {len(processed_tiles)}")
    print(f"Skipped (local zip exists): {len(skipped_local)}")
    print(f"Skipped (GCS file exists): {len(skipped_gcs)}")
    print(f"Failed: {len(failed_tiles)}")
    print(f"Total chips generated: {total_chips_generated}")
    print(f"Total chips saved: {total_chips_saved}")
    
    if failed_tiles:
        print(f"\nFailed tiles:")
        for result in failed_tiles:
            print(f"  - {result['tile_id']}: {result['message']}")
    
    # Get list of files available for upload (newly created + existing local files)
    files_to_upload = [r for r in results if r['success'] and 'zip_path' in r and r['zip_path']]
    
    # Get list of files already in GCS, ready for GEE asset creation
    files_for_gee_creation = [
        {'tile_name': r['tile_name'], 'gcs_uri': r['message'].split(' ')[-1]}
        for r in results if r['success'] and 'already exists in GCS' in r['message']
    ]
    
    # Phase 2: Batch upload to GCS
    if args.gcs_bucket and args.gee_asset_path and files_to_upload:
        print("\n" + "="*80)
        print("PHASE 2: BATCH UPLOAD TO GCS")
        print("="*80)
        
        gcs_path = f"gs://{args.gcs_bucket}/{args.gcs_prefix}/" if args.gcs_prefix else f"gs://{args.gcs_bucket}/"
        print(f"Uploading {len(files_to_upload)} files to {gcs_path}")
        
        # Use gcloud to upload all files at once
        local_files = [r['zip_path'] for r in files_to_upload]
        success, message = batch_upload_to_gcs(local_files, args.gcs_bucket, args.gcs_prefix, args.debug)
        
        if success:
            print("✅ Batch upload to GCS successful!")
            
            # Add successfully uploaded files to the GEE creation queue
            for result in files_to_upload:
                filename = os.path.basename(result['zip_path'])
                gcs_blob_path = f"{args.gcs_prefix}/{filename}" if args.gcs_prefix else filename
                gcs_uri = f"gs://{args.gcs_bucket}/{gcs_blob_path}"
                files_for_gee_creation.append({
                    'tile_name': result['tile_name'],
                    'gcs_uri': gcs_uri
                })
        else:
            print(f"❌ Batch upload to GCS failed: {message}")

    # Phase 3: Create GEE assets
    if args.gcs_bucket and args.gee_asset_path and files_for_gee_creation:
        print("\n" + "="*80)
        print("PHASE 3: CREATING GEE ASSETS")
        print("="*80)
        
        gee_results = []
        for item in files_for_gee_creation:
            tile_name = item['tile_name']
            gcs_uri = item['gcs_uri']
            asset_id = f"{args.gee_asset_path}/{tile_name}"
            
            if args.debug:
                print(f"Processing GEE asset for {tile_name}...")
            
            # Check if GEE asset already exists
            if check_gee_asset_exists(asset_id):
                if args.debug:
                    print(f"  Skipping asset creation for {tile_name}, already exists.")
                gee_results.append({
                    'tile_name': tile_name,
                    'success': True,
                    'message': 'Skipped - GEE asset already exists',
                    'task_id': 'existing'
                })
                continue

            # Create GEE asset if it doesn't exist
            if args.debug:
                print(f"Creating GEE asset for {tile_name} from {gcs_uri}")
            
            gee_success, gee_message, task_id = create_gee_asset(gcs_uri, args.gee_asset_path, tile_name)
            gee_results.append({
                'tile_name': tile_name,
                'success': gee_success,
                'message': gee_message,
                'task_id': task_id if gee_success else None
            })
        
        # Print GEE results summary
        successful_assets = [r for r in gee_results if r['success'] and r['task_id'] != 'existing']
        skipped_assets = [r for r in gee_results if r['task_id'] == 'existing']
        failed_assets = [r for r in gee_results if not r['success']]
        
        print(f"New GEE assets created: {len(successful_assets)}")
        print(f"Skipped (asset already exists): {len(skipped_assets)}")
        print(f"GEE asset creation failed: {len(failed_assets)}")
        
        if failed_assets:
            print(f"\nFailed GEE assets:")
            for result in failed_assets:
                print(f"  - {result['tile_name']}: {result['message']}")
        
        if successful_assets and args.debug:
            print(f"\nSuccessful GEE assets:")
            for result in successful_assets:
                print(f"  - {result['tile_name']}: {result['task_id']}")
    
    elif args.gcs_bucket and args.gee_asset_path and not files_for_gee_creation:
        print("\n⚠️  No files to process for GEE asset creation.")

    elif not args.gcs_bucket or not args.gee_asset_path:
        print(f"\n📁 Local files created in: {args.output_dir}")
        print("No GCS/GEE upload (missing --gcs_bucket or --gee_asset_path)")
            
    print("\n" + "="*80)
    print("WORKFLOW COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main() 