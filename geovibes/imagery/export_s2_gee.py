#!/usr/bin/env python3

import argparse
import os
import sys
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import ee
import geopandas as gpd
import pandas as pd
from google.cloud import storage

sys.path.append(str(Path(__file__).parent.parent.parent))
from geovibes.ee_tools import initialize_ee_with_credentials, get_s2_cloud_masked_collection
from geovibes.tiling import get_mgrs_tile_ids_for_roi_from_roi_file


def load_mgrs_tiles(mgrs_file):
    """Load MGRS tiles from geojson file."""
    try:
        gdf = gpd.read_parquet(mgrs_file)
        return gdf
    except Exception as e:
        raise ValueError(f"Failed to load MGRS tiles from {mgrs_file}: {e}")


def geometry_to_ee_feature(geometry):
    """Convert shapely geometry to Earth Engine geometry."""
    geom_dict = json.loads(gpd.GeoSeries([geometry]).to_json())
    coords = geom_dict['features'][0]['geometry']['coordinates']
    return ee.Geometry.Polygon(coords)


def create_s2_composite(aoi_geometry, start_date, end_date, clear_threshold=0.80):
    """Create cloud-masked Sentinel-2 composite for the area of interest."""
    
    # Get cloud-masked collection
    collection = get_s2_cloud_masked_collection(
        aoi_geometry, 
        start_date, 
        end_date, 
        clear_threshold
    )
    
    # Add NDVI and NDWI to each image in the collection
    def add_indices(image):
        ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
        ndwi = image.normalizedDifference(['B3', 'B8']).rename('NDWI')
        return image.addBands(ndvi).addBands(ndwi)
    
    collection = collection.map(add_indices)
    
    # Define the bands to export
    # Using the most commonly needed Sentinel-2 bands
    bands_to_export = [
        'B2',   # Blue
        'B3',   # Green  
        'B4',   # Red
        'B5',   # Red Edge 1
        'B6',   # Red Edge 2
        'B7',   # Red Edge 3
        'B8',   # NIR
        'B8A',  # NIR Narrow
        'B9',   # Water Vapour
        'B11',  # SWIR 1
        'B12'   # SWIR 2
    ]
    bands_to_export.extend(['NDVI', 'NDWI'])
    
    # Create median composite
    composite = collection.select(bands_to_export).median()

    return composite, bands_to_export


def export_band_to_drive(
        image, band_name, mgrs_code, output_folder, crs, start_date, end_date, geometry, scale=10):
    """Export a single band to Google Drive as GeoTIFF."""
    
    band_image = image.select([band_name])
    
    native_20m_bands = ['B5', 'B6', 'B7', 'B8A', 'B11', 'B12']
    native_60m_bands = ['B9']
    export_scale = 60 if band_name in native_60m_bands else 20 if band_name in native_20m_bands else scale
    
    band_image = band_image.toInt16()
    
    task = ee.batch.Export.image.toDrive(
        image=band_image,
        description=f'{mgrs_code}_{band_name}_{start_date}_{end_date}',
        folder=output_folder,
        fileNamePrefix=f'{mgrs_code}_{band_name}_{start_date}_{end_date}',
        scale=export_scale,
        region=geometry,
        crs=f'EPSG:{crs}',
        maxPixels=1e9,
        fileFormat='GeoTIFF'
    )
    
    return task


def export_band_to_cloud_storage(
        image, band_name, mgrs_code, bucket_name, folder_path, crs, start_date, end_date, geometry, scale=10):
    """Export a single band to Google Cloud Storage as GeoTIFF."""
    
    band_image = image.select([band_name])
    
    native_20m_bands = ['B5', 'B6', 'B7', 'B8A', 'B11', 'B12']
    native_60m_bands = ['B9']
    export_scale = 60 if band_name in native_60m_bands else 20 if band_name in native_20m_bands else scale
    
    band_image = band_image.toInt16()
    
    file_name = f'{mgrs_code}_{band_name}_{start_date}_{end_date}'
    object_name = f'{folder_path}/{file_name}' if folder_path else file_name
    
    task = ee.batch.Export.image.toCloudStorage(
        image=band_image,
        description=f'{mgrs_code}_{band_name}_{start_date}_{end_date}',
        bucket=bucket_name,
        fileNamePrefix=object_name,
        scale=export_scale,
        region=geometry,
        crs=f'EPSG:{crs}',
        maxPixels=1e9,
        fileFormat='GeoTIFF'
    )
    
    return task


def check_gcs_file_exists(bucket, object_name: str) -> bool:
    """Check if a file exists in Google Cloud Storage."""
    blob = bucket.blob(object_name)
    return blob.exists()


def track_task_status(tasks: List[ee.batch.Task], check_interval: int = 30) -> Dict:
    """Track export tasks and their EECU consumption."""
    print(f"\n🔄 Tracking {len(tasks)} export tasks...")
    
    # Generate automatic log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_file = f"ee_export_tasks_{timestamp}.csv"
    print(f"📄 Logging task details to: {csv_file}")
    print("Press Ctrl+C to stop tracking and continue\n")
    
    task_stats = {
        'total_tasks': len(tasks),
        'completed': 0,
        'failed': 0,
        'running': 0,
        'total_eecu': 0.0
    }
    
    # Track task details for DataFrame
    task_details = {}
    for task in tasks:
        task_id = task.id
        task_details[task_id] = {
            'task_name': task.config.get('description', 'Unknown'),
            'start_time': None,
            'end_time': None,
            'runtime_seconds': None,
            'eecu_usage': 0.0,
            'final_state': 'PENDING'
        }
    
    try:
        while True:
            running_tasks = 0
            completed_tasks = 0
            failed_tasks = 0
            total_eecu = 0.0
            
            for task in tasks:
                status = task.status()
                state = status.get('state', 'UNKNOWN')
                task_id = task.id
                
                # Update task details
                task_detail = task_details[task_id]
                
                # Track start time
                if task_detail['start_time'] is None and state in ['RUNNING', 'COMPLETED', 'FAILED', 'CANCELLED']:
                    start_time_ms = status.get('start_timestamp_ms')
                    if start_time_ms:
                        task_detail['start_time'] = datetime.fromtimestamp(start_time_ms / 1000)
                
                # Track end time and final state
                if state in ['COMPLETED', 'FAILED', 'CANCELLED'] and task_detail['end_time'] is None:
                    update_time_ms = status.get('update_timestamp_ms')
                    if update_time_ms:
                        task_detail['end_time'] = datetime.fromtimestamp(update_time_ms / 1000)
                        task_detail['final_state'] = state
                        
                        # Calculate runtime
                        if task_detail['start_time'] and task_detail['end_time']:
                            runtime = task_detail['end_time'] - task_detail['start_time']
                            task_detail['runtime_seconds'] = runtime.total_seconds()
                
                # Update EECU usage
                eecu_usage = status.get('eecu_usage', {}).get('cpu_seconds', 0)
                if eecu_usage:
                    task_detail['eecu_usage'] = eecu_usage
                    total_eecu += eecu_usage
                
                # Count by state
                if state == 'RUNNING':
                    running_tasks += 1
                elif state == 'COMPLETED':
                    completed_tasks += 1
                elif state in ['FAILED', 'CANCELLED']:
                    failed_tasks += 1
            
            task_stats.update({
                'completed': completed_tasks,
                'failed': failed_tasks,
                'running': running_tasks,
                'total_eecu': total_eecu
            })
            
            print(f"\r📊 Status: {completed_tasks} completed, {running_tasks} running, {failed_tasks} failed | EECUs: {total_eecu:.2f}", end="")
            
            if completed_tasks + failed_tasks == len(tasks):
                print("\n✅ All tasks completed!")
                break
                
            time.sleep(check_interval)
            
    except KeyboardInterrupt:
        print("\n⏸️  Stopping task tracking (tasks will continue running)")
    
    # Create DataFrame and save to CSV using pandas
    df = pd.DataFrame.from_dict(task_details, orient='index')
    df.to_csv(csv_file, index=False)
    
    print(f"\n📄 Task details saved to: {csv_file}")
    
    return task_stats


def main():
    parser = argparse.ArgumentParser(
        description='Export cloud-masked Sentinel-2 composite bands for all MGRS tiles intersecting ROI'
    )
    parser.add_argument(
        '--roi_file',
        type=str,
        required=True,
        help="Path to a GeoJSON/GeoParquet file to filter MGRS tiles."
    )
    parser.add_argument(
        '--output-folder',
        default='imagery/s2',
        help='Output folder/path name for files'
    )
    parser.add_argument(
        '--bucket-name',
        default='geovibes',
        help='Google Cloud Storage bucket name (for cloud storage export)'
    )
    parser.add_argument(
        '--destination',
        choices=['drive', 'cloud'],
        default='cloud',
        help='Export destination: drive or cloud (default: cloud)'
    )
    parser.add_argument(
        '--start-date',
        default='2024-01-01',
        help='Start date for composite (YYYY-MM-DD format, default: 2024-01-01)'
    )
    parser.add_argument(
        '--end-date', 
        default='2025-01-01',
        help='End date for composite (YYYY-MM-DD format, default: 2025-01-01)'
    )
    parser.add_argument(
        '--clear-threshold',
        type=float,
        default=0.80,
        help='CloudScore+ clear threshold (0-1, default: 0.80)'
    )
    parser.add_argument(
        '--scale',
        type=int,
        default=10,
        help='Export scale in meters (default: 10)'
    )
    parser.add_argument(
        '--mgrs_reference_file',
        type=str,
        default='./mgrs_tiles.parquet',
        help="Path to GeoParquet file with MGRS tile geometries."
    )
    parser.add_argument(
        '--track-tasks',
        action='store_true',
        help='Track export tasks and EECU consumption'
    )
    
    args = parser.parse_args()
    
    if not initialize_ee_with_credentials():
        print("❌ Failed to initialize Earth Engine. Exiting.")
        return 1
    
    destination = args.destination
    
    if destination == 'cloud' and not args.bucket_name:
        print("❌ Cloud storage export requires --bucket-name argument")
        return 1
    
    # Initialize GCS client if needed
    gcs_bucket = None
    if destination == 'cloud':
        try:
            storage_client = storage.Client()
            gcs_bucket = storage_client.bucket(args.bucket_name)
            print(f"✅ Connected to GCS bucket: {args.bucket_name}")
        except Exception as e:
            print(f"❌ Failed to connect to GCS bucket: {e}")
            return 1
    
    try:
        print(f"🎯 Finding intersecting MGRS tiles for ROI: {args.roi_file}...")
        intersecting_mgrs_ids = get_mgrs_tile_ids_for_roi_from_roi_file(
            roi_geojson_file=args.roi_file,
            mgrs_tiles_file=args.mgrs_reference_file,
        )

        if not intersecting_mgrs_ids:
            raise ValueError("No MGRS tiles intersect with the provided ROI")

        intersecting_mgrs_codes = [str(tile_id) for tile_id in intersecting_mgrs_ids]

        print(f"🗺️  Loading MGRS tile geometries from {args.mgrs_reference_file}")
        mgrs_gdf = load_mgrs_tiles(args.mgrs_reference_file)

        intersecting_gdf = mgrs_gdf[mgrs_gdf['mgrs_id'].isin(intersecting_mgrs_codes)]

        intersecting_tiles = []
        for _, row in intersecting_gdf.iterrows():
            intersecting_tiles.append({
                'mgrs_code': row['mgrs_id'],
                'geometry': row.geometry,
                'epsg_code': row['epsg']
            })
        
        print(f"📍 Found {len(intersecting_tiles)} intersecting MGRS tiles:")
        for tile in intersecting_tiles:
            print(f"   • {tile['mgrs_code']} (EPSG:{tile['epsg_code']})")
        
        print(f"\n🛰️  Creating Sentinel-2 composites from {args.start_date} to {args.end_date}")
        print(f"☁️  Using CloudScore+ threshold: {args.clear_threshold}")
        
        all_tasks = []
        
        if destination == 'drive':
            print(f"📤 Exporting band images to Google Drive folder: {args.output_folder}")
        else:
            print(f"📤 Exporting band images to Cloud Storage: gs://{args.bucket_name}/{args.output_folder}")
        
        for i, tile_info in enumerate(intersecting_tiles, 1):
            print(f"\n🔄 Processing tile {i}/{len(intersecting_tiles)}: {tile_info['mgrs_code']}")
            
            ee_geometry = geometry_to_ee_feature(tile_info['geometry'])
            
            composite, bands = create_s2_composite(
                ee_geometry,
                args.start_date,
                args.end_date,
                args.clear_threshold
            )
            
            # Export each band for this tile
            for band_name in bands:
                file_name = f"{tile_info['mgrs_code']}_{band_name}_{args.start_date}_{args.end_date}.tif"
                
                # Check if file exists
                if destination == 'cloud' and gcs_bucket:
                    object_name = f"{args.output_folder}/{file_name}"
                    if check_gcs_file_exists(gcs_bucket, object_name):
                        print(f"   ⏩ Skipping (exists): {file_name}")
                        continue
                elif destination == 'drive':
                    # NOTE: Checking for existing files in Google Drive is not supported.
                    pass

                if destination == 'drive':
                    task = export_band_to_drive(
                        composite,
                        band_name, 
                        tile_info['mgrs_code'],
                        args.output_folder,
                        tile_info['epsg_code'],
                        args.start_date,
                        args.end_date,
                        ee_geometry,
                        args.scale
                    )
                else:
                    task = export_band_to_cloud_storage(
                        composite,
                        band_name,
                        tile_info['mgrs_code'],
                        args.bucket_name,
                        args.output_folder,
                        tile_info['epsg_code'],
                        args.start_date,
                        args.end_date,
                        ee_geometry,
                        args.scale
                    )
                all_tasks.append(task)
                print(f"   📁 Queued: {file_name}")
        
        if not all_tasks:
            print("\n✅ All files already exist. No new tasks to start.")
            return 0

        # Start all tasks
        print(f"\n🚀 Starting {len(all_tasks)} export tasks...")
        for task in all_tasks:
            task.start()
        
        print(f"✅ Started {len(all_tasks)} export tasks across {len(intersecting_tiles)} MGRS tiles")
        print("🔄 Monitor progress at: https://code.earthengine.google.com/tasks")
        print("📋 Export details:")
        print(f"   ROI File: {args.roi_file}")
        print(f"   MGRS Tiles: {', '.join([t['mgrs_code'] for t in intersecting_tiles])}")
        print(f"   Destination: {destination.upper()}")
        if destination == 'drive':
            print(f"   Output Folder: {args.output_folder}")
        else:
            print(f"   Bucket: {args.bucket_name}")
            print(f"   Path: {args.output_folder}")
        print(f"   Date Range: {args.start_date} to {args.end_date}")
        print(f"   Scale: {args.scale}m")
        print(f"   Bands: {', '.join(bands)}")
        print(f"   Total Exports: {len(all_tasks)}")
        
        if args.track_tasks:
            task_stats = track_task_status(all_tasks)
            print(f"\n📊 Final Statistics:")
            print(f"   Total Tasks: {task_stats['total_tasks']}")
            print(f"   Completed: {task_stats['completed']}")
            print(f"   Failed: {task_stats['failed']}")
            print(f"   Total EECUs: {task_stats['total_eecu']:.2f}")
            print(f"   MGRS Tiles Processed: {len(intersecting_tiles)}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main()) 