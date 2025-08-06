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
from geovibes.ee_tools import (
    initialize_ee_with_credentials, 
    get_s2_ndvi_median,
    get_s2_ndwi_median,
    get_s2_rgb_median
)
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


def create_basemap_image(aoi_geometry, basemap_type, start_date, end_date, clear_threshold=0.80):
    """Create basemap image for the specified type (NDVI, NDWI, or RGB).
    
    Args:
        aoi_geometry: Earth Engine geometry defining area of interest
        basemap_type: Type of basemap ('ndvi', 'ndwi', 'rgb')
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        clear_threshold: CloudScore+ threshold (0-1) for pixel quality
        
    Returns:
        Earth Engine image for the specified basemap type
    """
    if basemap_type.lower() == 'ndvi':
        return get_s2_ndvi_median(aoi_geometry, start_date, end_date, clear_threshold)
    elif basemap_type.lower() == 'ndwi':
        return get_s2_ndwi_median(aoi_geometry, start_date, end_date, clear_threshold)
    elif basemap_type.lower() == 'rgb':
        return get_s2_rgb_median(aoi_geometry, start_date, end_date, clear_threshold)
    else:
        raise ValueError(f"Unsupported basemap type: {basemap_type}")


def detect_export_destination() -> str:
    """Automatically detect whether to use Drive or Cloud Storage."""
    try:
        storage_client = storage.Client()
        for bucket in storage_client.list_buckets(max_results=1):
            return 'cloud'
    except Exception:
        pass
    return 'drive'


def export_basemap_to_drive(
        image, basemap_type, mgrs_code, output_folder, crs, start_date, end_date, geometry, scale=10):
    """Export basemap image to Google Drive as GeoTIFF."""
    
    # Scale basemaps appropriately for visualization
    if basemap_type.lower() in ['ndvi', 'ndwi']:
        # Scale index values from -1,1 to 0-255 for better visualization
        scaled_image = image.multiply(127.5).add(127.5).toUint8()
    else:  # RGB
        # RGB bands are typically in 0-10000 range, scale to 0-255
        scaled_image = image.divide(39.24).toUint8()
    
    task = ee.batch.Export.image.toDrive(
        image=scaled_image,
        description=f'{mgrs_code}_{basemap_type.upper()}_{start_date}_{end_date}',
        folder=output_folder,
        fileNamePrefix=f'{mgrs_code}_{basemap_type.upper()}_{start_date}_{end_date}',
        scale=scale,
        region=geometry,
        crs=f'EPSG:{crs}',
        maxPixels=1e9,
        fileFormat='GeoTIFF'
    )
    
    return task


def export_basemap_to_cloud_storage(
        image, basemap_type, mgrs_code, bucket_name, folder_path, crs, start_date, end_date, geometry, scale=10):
    """Export basemap image to Google Cloud Storage as GeoTIFF."""
    
    # Scale basemaps appropriately for visualization
    if basemap_type.lower() in ['ndvi', 'ndwi']:
        # Scale index values from -1,1 to 0-255 for better visualization
        scaled_image = image.multiply(127.5).add(127.5).toUint8()
    else:  # RGB
        # RGB bands are typically in 0-10000 range, scale to 0-255
        scaled_image = image.divide(39.24).toUint8()
    
    file_name = f'{mgrs_code}_{basemap_type.upper()}_{start_date}_{end_date}'
    object_name = f'{folder_path}/{file_name}' if folder_path else file_name
    
    task = ee.batch.Export.image.toCloudStorage(
        image=scaled_image,
        description=f'{mgrs_code}_{basemap_type.upper()}_{start_date}_{end_date}',
        bucket=bucket_name,
        fileNamePrefix=object_name,
        scale=scale,
        region=geometry,
        crs=f'EPSG:{crs}',
        maxPixels=1e9,
        fileFormat='GeoTIFF'
    )
    
    return task


def track_task_status(tasks: List[ee.batch.Task], check_interval: int = 30) -> Dict:
    """Track export tasks and their EECU consumption."""
    print(f"\n🔄 Tracking {len(tasks)} export tasks...")
    
    # Generate automatic log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_file = f"ee_basemap_export_tasks_{timestamp}.csv"
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
        description='Export NDVI/NDWI/RGB basemaps for all MGRS tiles intersecting ROI'
    )
    parser.add_argument(
        '--roi-file',
        type=str,
        required=True,
        help="Path to a GeoJSON/GeoParquet file to filter MGRS tiles."
    )
    parser.add_argument(
        '--basemap-type',
        choices=['ndvi', 'ndwi', 'rgb', 'all'],
        default='all',
        help='Type of basemap to export: ndvi, ndwi, rgb, or all (default: all)'
    )
    parser.add_argument(
        '--output-folder',
        default='basemaps',
        help='Output folder/path name for files (default: basemaps)'
    )
    parser.add_argument(
        '--bucket-name',
        default='geovibes',
        help='Google Cloud Storage bucket name (for cloud storage export)'
    )
    parser.add_argument(
        '--destination',
        choices=['drive', 'cloud', 'auto'],
        default='auto',
        help='Export destination: drive, cloud, or auto-detect (default: auto)'
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
        '--mgrs-reference-file',
        type=str,
        default='./geometries/mgrs_tiles.parquet',
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
    if destination == 'auto':
        destination = detect_export_destination()
        print(f"🤖 Auto-detected export destination: {destination}")
    
    if destination == 'cloud' and not args.bucket_name:
        print("❌ Cloud storage export requires --bucket-name argument")
        return 1
    
    # Determine which basemap types to export
    if args.basemap_type == 'all':
        basemap_types = ['ndvi', 'ndwi', 'rgb']
    else:
        basemap_types = [args.basemap_type]
    
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
        
        print(f"\n🗺️  Creating {', '.join([t.upper() for t in basemap_types])} basemaps from {args.start_date} to {args.end_date}")
        print(f"☁️  Using CloudScore+ threshold: {args.clear_threshold}")
        
        all_tasks = []
        total_exports = len(intersecting_tiles) * len(basemap_types)
        
        if destination == 'drive':
            print(f"📤 Exporting {total_exports} basemap images to Google Drive folder: {args.output_folder}")
        else:
            print(f"📤 Exporting {total_exports} basemap images to Cloud Storage: gs://{args.bucket_name}/{args.output_folder}")
        
        for i, tile_info in enumerate(intersecting_tiles, 1):
            print(f"\n🔄 Processing tile {i}/{len(intersecting_tiles)}: {tile_info['mgrs_code']}")
            
            ee_geometry = geometry_to_ee_feature(tile_info['geometry'])
            
            # Export each basemap type for this tile
            for basemap_type in basemap_types:
                print(f"   🗺️  Creating {basemap_type.upper()} basemap...")
                
                basemap_image = create_basemap_image(
                    ee_geometry,
                    basemap_type,
                    args.start_date,
                    args.end_date,
                    args.clear_threshold
                )
                
                if destination == 'drive':
                    task = export_basemap_to_drive(
                        basemap_image,
                        basemap_type,
                        tile_info['mgrs_code'],
                        args.output_folder,
                        tile_info['epsg_code'],
                        args.start_date,
                        args.end_date,
                        ee_geometry,
                        args.scale
                    )
                else:
                    task = export_basemap_to_cloud_storage(
                        basemap_image,
                        basemap_type,
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
                print(f"   📁 Queued: {tile_info['mgrs_code']}_{basemap_type.upper()}.tif")
        
        # Start all tasks
        print(f"\n🚀 Starting {len(all_tasks)} export tasks...")
        for task in all_tasks:
            task.start()
        
        print(f"✅ Started {len(all_tasks)} export tasks across {len(intersecting_tiles)} MGRS tiles")
        print("🔄 Monitor progress at: https://code.earthengine.google.com/tasks")
        print("📋 Export details:")
        print(f"   ROI File: {args.roi_file}")
        print(f"   MGRS Tiles: {', '.join([t['mgrs_code'] for t in intersecting_tiles])}")
        print(f"   Basemap Types: {', '.join([t.upper() for t in basemap_types])}")
        print(f"   Destination: {destination.upper()}")
        if destination == 'drive':
            print(f"   Output Folder: {args.output_folder}")
        else:
            print(f"   Bucket: {args.bucket_name}")
            print(f"   Path: {args.output_folder}")
        print(f"   Date Range: {args.start_date} to {args.end_date}")
        print(f"   Scale: {args.scale}m")
        print(f"   Total Exports: {len(all_tasks)}")
        
        if args.track_tasks:
            task_stats = track_task_status(all_tasks)
            print(f"\n📊 Final Statistics:")
            print(f"   Total Tasks: {task_stats['total_tasks']}")
            print(f"   Completed: {task_stats['completed']}")
            print(f"   Failed: {task_stats['failed']}")
            print(f"   Total EECUs: {task_stats['total_eecu']:.2f}")
            print(f"   MGRS Tiles Processed: {len(intersecting_tiles)}")
            print(f"   Basemap Types: {', '.join([t.upper() for t in basemap_types])}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())