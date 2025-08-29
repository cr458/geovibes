"""
Generate a grid of tiles over an MGRS tile and save it as a GeoParquet file.
"""

import argparse
import os
import shutil
import subprocess
import tempfile
import zipfile
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

import geopandas as gpd
import pyproj
import shapely.ops
from google.cloud import storage
from joblib import Parallel, delayed

from geovibes.tiling import MGRSTileGrid, MGRSTileId, chip_mgrs_tile

def write_tiles_to_geoparquet(
    tiles: gpd.GeoDataFrame, tile_name: str, output_dir: str = "."
):
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


def check_gcs_bucket_access(gcs_bucket_name: str) -> tuple[bool, str]:
    """Checks for read access to a GCS bucket."""
    try:
        storage_client = storage.Client()
        storage_client.get_bucket(gcs_bucket_name)
        return (True, f"✅ Access to GCS bucket '{gcs_bucket_name}' confirmed.")
    except Exception as e:
        return (False, f"❌ Failed to access GCS bucket '{gcs_bucket_name}': {str(e)}")


def create_local_shapefile_zip(
    tiles: gpd.GeoDataFrame, tile_name: str, output_dir: str
):
    """
    Creates a zipped shapefile locally.
    Returns tuple: (success: bool, message: str, zip_path: str)
    """
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            shapefile_path = os.path.join(tmpdir, f"{tile_name}.shp")
            tiles.to_file(shapefile_path, driver="ESRI Shapefile")

            zip_path = os.path.join(tmpdir, f"{tile_name}.zip")
            with zipfile.ZipFile(zip_path, "w") as zipf:
                for ext in [".shp", ".shx", ".dbf", ".prj", ".cpg"]:
                    source_file = f"{shapefile_path[:-4]}{ext}"
                    if os.path.exists(source_file):
                        zipf.write(source_file, arcname=os.path.basename(source_file))
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

        command = ["gcloud", "storage", "cp", local_zip_path, gcs_uri]

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
        command = ["earthengine", "upload", "table", f"--asset_id={asset_id}", gcs_uri]

        result = subprocess.run(command, check=True, capture_output=True, text=True)
        task_id = result.stdout.strip()

        return (True, f"Started GEE task: {task_id}", task_id)

    except subprocess.CalledProcessError as e:
        error_msg = e.stderr.strip() if e.stderr else "No error details available"
        stdout_msg = e.stdout.strip() if e.stdout else ""

        if (
            "not authenticated" in error_msg.lower()
            or "not authenticated" in stdout_msg.lower()
        ):
            error_msg = (
                "Not authenticated. Please run 'earthengine authenticate' first."
            )
        elif "permission" in error_msg.lower() or "permission" in stdout_msg.lower():
            error_msg = f"Permission denied. Check access to: {gee_asset_path}"
        elif (
            "already exists" in error_msg.lower()
            or "already exists" in stdout_msg.lower()
        ):
            error_msg = f"Asset already exists: {asset_id}"
        elif "invalid" in error_msg.lower() or "invalid" in stdout_msg.lower():
            error_msg = (
                f"Invalid asset path or GCS URI. Asset: {asset_id}, GCS: {gcs_uri}"
            )
        elif not error_msg and not stdout_msg:
            error_msg = f"Command failed with exit code {e.returncode}. Check 'earthengine' is installed and authenticated."

        full_error = f"GEE upload failed: {error_msg}"
        if stdout_msg and stdout_msg != error_msg:
            full_error += f" | Output: {stdout_msg}"

        return (False, full_error, "")
    except Exception as e:
        return (False, f"GEE upload failed: {str(e)} (Type: {type(e).__name__})", "")


def check_earthengine_auth():
    """Check if earthengine CLI is installed and authenticated."""
    try:
        result = subprocess.run(
            ["earthengine", "ls"], capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            return (True, "Earthengine CLI is authenticated")
        else:
            error_msg = result.stderr.strip() if result.stderr else "Unknown error"
            return (False, f"Earthengine authentication failed: {error_msg}")
    except FileNotFoundError:
        return (
            False,
            "Earthengine CLI not found. Please install the earthengine-api package.",
        )
    except subprocess.TimeoutExpired:
        return (False, "Earthengine CLI timeout - may be authentication issue")
    except Exception as e:
        return (False, f"Error checking earthengine: {str(e)}")


def check_gee_asset_exists(asset_id: str) -> bool:
    """Check if a GEE asset already exists."""
    try:
        command = ["earthengine", "asset", "info", asset_id]
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        return True
    except subprocess.CalledProcessError as e:
        if "does not exist" in (e.stderr or "").lower():
            return False
        elif "not authenticated" in (e.stderr or "").lower():
            print(f"Warning: Not authenticated to check asset existence: {asset_id}")
            return False
        else:
            return False
    except Exception:
        return False


def batch_upload_to_gcs(
    local_files: list, gcs_bucket: str, gcs_prefix: str, debug: bool = False
):
    """
    Upload multiple files to GCS using gcloud storage cp.
    Returns tuple: (success: bool, message: str)
    """
    try:
        if not local_files:
            return (True, "No files to upload")

        gcs_dest = (
            f"gs://{gcs_bucket}/{gcs_prefix}/" if gcs_prefix else f"gs://{gcs_bucket}/"
        )
        command = ["gcloud", "storage", "cp"]
        command.extend(local_files)
        command.append(gcs_dest)
        if debug:
            print(f"Running: {' '.join(command)}")
        result = subprocess.run(command, check=True)
        return (True, f"Successfully uploaded {len(local_files)} files")

    except subprocess.CalledProcessError as e:
        error_msg = e.stderr if e.stderr else str(e)
        return (False, f"gcloud upload failed: {error_msg}")
    except Exception as e:
        return (False, f"Upload failed: {str(e)}")


def process_single_tile(
    tile_series,
    source_crs,
    tilesize,
    overlap,
    resolution,
    buffer_m,
    roi_geometry,
    roi_crs,
    output_dir,
    gcs_bucket,
    gcs_prefix,
    gee_asset_path,
    debug=False,
):
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
        mgrs_tile_id = MGRSTileId.from_str(tile_id)
        grid = MGRSTileGrid(
            mgrs_tile_id=mgrs_tile_id,
            tilesize=tilesize,
            overlap=overlap,
            resolution=resolution,
        )

        if gcs_bucket and gee_asset_path:
            blob_name = (
                f"{gcs_prefix}/{grid.prefix}.zip"
                if gcs_prefix
                else f"{grid.prefix}.zip"
            )
            if check_gcs_file_exists(gcs_bucket, blob_name):
                return {
                    "tile_id": tile_id,
                    "success": True,
                    "message": f"Skipped - file already exists in GCS: gs://{gcs_bucket}/{blob_name}",
                    "chips_generated": 0,
                    "chips_saved": 0,
                    "zip_path": "",
                    "tile_name": grid.prefix,
                }

        local_zip_path = os.path.join(output_dir, f"{grid.prefix}.zip")
        if os.path.exists(local_zip_path):
            return {
                "tile_id": tile_id,
                "success": True,
                "message": f"Skipped generation - local zip file already exists: {local_zip_path}",
                "chips_generated": 0,
                "chips_saved": 0,
                "zip_path": local_zip_path,
                "tile_name": grid.prefix,
            }

        tiles = chip_mgrs_tile(tile_series, grid, source_crs=source_crs)
        initial_chip_count = len(tiles)
        if debug:
            print(f"    Generated {initial_chip_count} initial chips for {tile_id}")

        if roi_geometry and len(tiles) > 0:
            transformer = pyproj.Transformer.from_crs(roi_crs, grid.crs, always_xy=True)
            roi_utm = shapely.ops.transform(transformer.transform, roi_geometry)
            buffered_roi_utm = roi_utm.buffer(buffer_m)

            intersecting_mask = tiles.intersects(buffered_roi_utm)
            tiles = tiles[intersecting_mask]
            if debug:
                print(
                    f"    Post-filtering: Kept {len(tiles)} of {initial_chip_count} chips intersecting with the {buffer_m}m buffered ROI for {tile_id}"
                )

        final_chip_count = len(tiles)

        if final_chip_count > 0:
            if gcs_bucket and gee_asset_path:
                if debug:
                    print(
                        f"    Creating local zip for {final_chip_count} chips for {tile_id}..."
                    )
                success, zip_message, zip_path = create_local_shapefile_zip(
                    tiles, grid.prefix, output_dir
                )
                if success:
                    message = "Successfully processed and created local zip"
                    if debug:
                        print(f"    Zip creation success for {tile_id}: {zip_message}")
                else:
                    if debug:
                        print(f"    Zip creation failed for {tile_id}: {zip_message}")
                    return {
                        "tile_id": tile_id,
                        "success": False,
                        "message": f"Zip creation failed: {zip_message}",
                        "chips_generated": initial_chip_count,
                        "chips_saved": 0,
                        "zip_path": "",
                        "tile_name": grid.prefix,
                    }
            else:
                write_tiles_to_geoparquet(tiles, grid.prefix, output_dir)
                message = "Successfully processed and saved locally"
                zip_path = ""
        else:
            message = "No chips to save after filtering"
            zip_path = ""

        return {
            "tile_id": tile_id,
            "success": True,
            "message": message,
            "chips_generated": initial_chip_count,
            "chips_saved": final_chip_count,
            "zip_path": zip_path,
            "tile_name": grid.prefix,
        }

    except Exception as e:
        import traceback

        error_details = traceback.format_exc()
        return {
            "tile_id": tile_id,
            "success": False,
            "message": f"Error: {str(e)}",
            "error_details": error_details,
            "chips_generated": 0,
            "chips_saved": 0,
            "gcs_uri": "",
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
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mgrs_tile_file",
        type=str,
        required=True,
        help="Path to GeoParquet or GeoJSON file with MGRS tile geometries.",
    )
    parser.add_argument(
        "--roi_file",
        type=str,
        help="Path to a GeoJSON/GeoParquet file to filter MGRS tiles.",
    )
    parser.add_argument("--tilesize", type=int, default=25, help="Tile size in pixels.")
    parser.add_argument("--overlap", type=int, default=0, help="Overlap in pixels.")
    parser.add_argument(
        "--resolution", type=float, default=10.0, help="Resolution in meters per pixel."
    )
    parser.add_argument(
        "--buffer_m",
        type=float,
        default=100.0,
        help="Buffer distance in meters for post-filtering chips against the ROI.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=".",
        help="Directory to save the output GeoParquet files.",
    )
    parser.add_argument(
        "--gcs_bucket",
        type=str,
        default="geovibes",
        help="GCS bucket name to upload zipped shapefiles to.",
    )
    parser.add_argument(
        "--gcs_prefix",
        type=str,
        default="tiles",
        help="GCS prefix/folder within the bucket.",
    )
    parser.add_argument(
        "--gee_asset_path",
        type=str,
        default="projects/demeterlabs-gee/assets/tiles",
        help="GEE asset path for table uploads (e.g., 'users/username/folder').",
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=-1,
        help="Number of parallel jobs to run. Use -1 for all available cores.",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enable debug mode for troubleshooting."
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.gcs_bucket and args.gee_asset_path:
        print("\n" + "=" * 80)
        print("PHASE 0: PRE-FLIGHT CHECKS")
        print("=" * 80)
        gcs_ok, gcs_message = check_gcs_bucket_access(args.gcs_bucket)
        print(gcs_message)
        if not gcs_ok:
            print("Please resolve GCS access issues before proceeding.")
            return

        auth_ok, auth_message = check_earthengine_auth()
        print(f"✅ {auth_message}" if auth_ok else f"❌ {auth_message}")
        if not auth_ok:
            print(
                "Please resolve authentication issues before proceeding with GEE asset creation."
            )
            return

    try:
        if args.mgrs_tile_file.endswith(".parquet"):
            mgrs_gdf = gpd.read_parquet(args.mgrs_tile_file)
        elif args.mgrs_tile_file.endswith(".geojson"):
            mgrs_gdf = gpd.read_file(args.mgrs_tile_file)
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
            print(
                f"Warning: MGRS file CRS ({mgrs_gdf.crs}) and ROI file CRS ({roi_gdf.crs}) differ. Reprojecting ROI to match MGRS for intersection."
            )
            roi_gdf = roi_gdf.to_crs(mgrs_gdf.crs)

        roi_geometry = roi_gdf.union_all()
        intersecting_mask = mgrs_gdf.intersects(roi_geometry)
        mgrs_gdf = mgrs_gdf[intersecting_mask]
        print(f"Found {len(mgrs_gdf)} MGRS tiles intersecting with the ROI.")
        roi_crs = roi_gdf.crs
    else:
        roi_crs = None

    tile_list = [row for _, row in mgrs_gdf.iterrows()]

    print("\n" + "=" * 80)
    print("PHASE 1: GENERATING LOCAL FILES")
    print("=" * 80)
    print(
        f"Processing {len(tile_list)} MGRS tiles using {args.n_jobs} parallel jobs..."
    )
    print(f"Output directory: {args.output_dir}")

    if args.gcs_bucket and args.gee_asset_path:
        gcs_path = (
            f"gs://{args.gcs_bucket}/{args.gcs_prefix}"
            if args.gcs_prefix
            else f"gs://{args.gcs_bucket}"
        )
        print(f"Will upload to: {gcs_path}")
        print(f"GEE asset path: {args.gee_asset_path}")
    else:
        print("No GCS/GEE configuration - local files only")

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
            gcs_bucket=args.gcs_bucket,
            gcs_prefix=args.gcs_prefix,
            gee_asset_path=args.gee_asset_path,
            debug=args.debug,
        )
        for tile_series in tile_list
    )

    print("\n" + "=" * 80)
    print("PHASE 1 SUMMARY")
    print("=" * 80)

    failed_tiles = [r for r in results if not r["success"]]
    skipped_local = [
        r
        for r in results
        if r["success"] and "local zip file already exists" in r["message"]
    ]
    skipped_gcs = [
        r for r in results if r["success"] and "already exists in GCS" in r["message"]
    ]
    processed_tiles = [
        r for r in results if r["success"] and "Skipped" not in r["message"]
    ]

    total_chips_generated = sum(r["chips_generated"] for r in results)
    total_chips_saved = sum(r["chips_saved"] for r in results)

    print(f"Total tiles: {len(results)}")
    print(f"Successfully processed: {len(processed_tiles)}")
    print(f"Skipped (local zip exists): {len(skipped_local)}")
    print(f"Skipped (GCS file exists): {len(skipped_gcs)}")
    print(f"Failed: {len(failed_tiles)}")
    print(f"Total chips generated: {total_chips_generated}")
    print(f"Total chips saved: {total_chips_saved}")

    if failed_tiles:
        print("\nFailed tiles:")
        for result in failed_tiles:
            print(f"  - {result['tile_id']}: {result['message']}")

    files_to_upload = [
        r for r in results if r["success"] and "zip_path" in r and r["zip_path"]
    ]

    files_for_gee_creation = [
        {"tile_name": r["tile_name"], "gcs_uri": r["message"].split(" ")[-1]}
        for r in results
        if r["success"] and "already exists in GCS" in r["message"]
    ]

    if args.gcs_bucket and args.gee_asset_path and files_to_upload:
        print("\n" + "=" * 80)
        print("PHASE 2: BATCH UPLOAD TO GCS")
        print("=" * 80)

        gcs_path = (
            f"gs://{args.gcs_bucket}/{args.gcs_prefix}/"
            if args.gcs_prefix
            else f"gs://{args.gcs_bucket}/"
        )
        print(f"Uploading {len(files_to_upload)} files to {gcs_path}")

        local_files = [r["zip_path"] for r in files_to_upload]
        success, message = batch_upload_to_gcs(
            local_files, args.gcs_bucket, args.gcs_prefix, args.debug
        )

        if success:
            print("✅ Batch upload to GCS successful!")

            for result in files_to_upload:
                filename = os.path.basename(result["zip_path"])
                gcs_blob_path = (
                    f"{args.gcs_prefix}/{filename}" if args.gcs_prefix else filename
                )
                gcs_uri = f"gs://{args.gcs_bucket}/{gcs_blob_path}"
                files_for_gee_creation.append(
                    {"tile_name": result["tile_name"], "gcs_uri": gcs_uri}
                )
        else:
            print(f"❌ Batch upload to GCS failed: {message}")

    if args.gcs_bucket and args.gee_asset_path and files_for_gee_creation:
        print("\n" + "=" * 80)
        print("PHASE 3: CREATING GEE ASSETS")
        print("=" * 80)

        gee_results = []
        for item in files_for_gee_creation:
            tile_name = item["tile_name"]
            gcs_uri = item["gcs_uri"]
            asset_id = f"{args.gee_asset_path}/{tile_name}"

            if args.debug:
                print(f"Processing GEE asset for {tile_name}...")

            asset_exists = check_gee_asset_exists(asset_id)
            if asset_exists:
                if args.debug:
                    print(f"  Skipping asset creation for {tile_name}, already exists.")
                gee_results.append(
                    {
                        "tile_name": tile_name,
                        "success": True,
                        "message": "Skipped - GEE asset already exists",
                        "task_id": "existing",
                    }
                )
                continue

            print(
                f"Asset does not already exist, launching ingestion task for {asset_id}"
            )
            if args.debug:
                print(f"Creating GEE asset for {tile_name} from {gcs_uri}")

            gee_success, gee_message, task_id = create_gee_asset(
                gcs_uri, args.gee_asset_path, tile_name
            )
            gee_results.append(
                {
                    "tile_name": tile_name,
                    "success": gee_success,
                    "message": gee_message,
                    "task_id": task_id if gee_success else None,
                }
            )

        successful_assets = [
            r for r in gee_results if r["success"] and r["task_id"] != "existing"
        ]
        skipped_assets = [r for r in gee_results if r["task_id"] == "existing"]
        failed_assets = [r for r in gee_results if not r["success"]]

        print(f"New GEE assets created: {len(successful_assets)}")
        print(f"Skipped (asset already exists): {len(skipped_assets)}")
        print(f"GEE asset creation failed: {len(failed_assets)}")

        if failed_assets:
            print("\nFailed GEE assets:")
            for result in failed_assets:
                print(f"  - {result['tile_name']}: {result['message']}")

            print("\n💡 Troubleshooting tips:")
            print("   - Ensure you're authenticated: earthengine authenticate")
            print(f"   - Check asset path permissions: {args.gee_asset_path}")
            print("   - Verify GCS files exist and are accessible")
            print("   - Check for network connectivity issues")

        if successful_assets and args.debug:
            print("\nSuccessful GEE assets:")
            for result in successful_assets:
                print(f"  - {result['tile_name']}: {result['task_id']}")

    elif args.gcs_bucket and args.gee_asset_path and not files_for_gee_creation:
        print("\n⚠️  No files to process for GEE asset creation.")

    elif not args.gcs_bucket or not args.gee_asset_path:
        print(f"\n📁 Local files created in: {args.output_dir}")
        print("No GCS/GEE upload (missing --gcs_bucket or --gee_asset_path)")

    print("\n" + "=" * 80)
    print("WORKFLOW COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
