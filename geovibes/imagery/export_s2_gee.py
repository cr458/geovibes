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
from shapely.geometry import shape
from google.cloud import storage

sys.path.append(str(Path(__file__).parent.parent))
from ee_tools import initialize_ee_with_credentials, get_s2_cloud_masked_collection


def load_mgrs_tiles(mgrs_file):
    """Load MGRS tiles from geojson file."""
    try:
        gdf = gpd.read_parquet(mgrs_file)
        return gdf
    except Exception as e:
        raise ValueError(f"Failed to load MGRS tiles from {mgrs_file}: {e}")


def load_roi_geometry(roi_file):
    """Load ROI geometry and return union of all geometries."""
    try:
        if roi_file.suffix.lower() in {'.gpq', '.parquet'}:
            gdf = gpd.read_parquet(roi_file)
        else:
            gdf = gpd.read_file(roi_file)
        
        if gdf.empty:
            raise ValueError("No geometries found in ROI file")
        
        # Ensure CRS is WGS84
        if gdf.crs is None or gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs(4326)
        
        # Return union of all geometries
        union_geom = gdf.union_all()
        return union_geom
        
    except Exception as e:
        raise ValueError(f"Failed to load ROI from {roi_file}: {e}")


def find_intersecting_mgrs_tiles(mgrs_gdf, roi_geometry):
    """Find all MGRS tiles that intersect with ROI geometry."""
    # Ensure MGRS tiles are in WGS84
    if mgrs_gdf.crs is None or mgrs_gdf.crs.to_epsg() != 4326:
        mgrs_gdf = mgrs_gdf.to_crs(4326)
    
    # Create ROI GeoDataFrame for spatial join
    roi_gdf = gpd.GeoDataFrame([1], geometry=[roi_geometry], crs="EPSG:4326")
    
    # Find intersecting tiles
    intersecting = gpd.sjoin(mgrs_gdf, roi_gdf, how="inner", predicate="intersects")
    
    if intersecting.empty:
        raise ValueError("No MGRS tiles intersect with the provided ROI")
    
    # Return list of tile info dictionaries
    tiles = []
    for _, row in intersecting.iterrows():
        tiles.append({
            'mgrs_code': row['mgrs_id'],
            'geometry': row.geometry,
            'epsg_code': row['epsg']
        })
    
    return tiles


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
    
    # Create median composite
    composite = collection.select(bands_to_export).median()
    return composite, bands_to_export


def detect_export_destination() -> str:
    """Automatically detect whether to use Drive or Cloud Storage."""
    try:
        storage_client = storage.Client()
        for bucket in storage_client.list_buckets(max_results=1):
            return 'cloud'
    except Exception:
        pass
    return 'drive'


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
        '--roi-file',
        required=True,
        help='Path to ROI file (GeoJSON, GeoParquet, or Shapefile)'
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
        '--mgrs-file',
        default='geometries/mgrs_tiles.parquet',
        help='Path to MGRS geojson file (default: geometries/mgrs_tiles.parquet)'
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
    
    if args.mgrs_file is None:
        script_dir = Path(__file__).parent.parent.parent
        args.mgrs_file = script_dir / 'geometries' / 'mgrs_tiles.parquet'
    
    destination = args.destination
    if destination == 'auto':
        destination = detect_export_destination()
        print(f"🤖 Auto-detected export destination: {destination}")
    
    if destination == 'cloud' and not args.bucket_name:
        print("❌ Cloud storage export requires --bucket-name argument")
        return 1
    
    try:
        print(f"🔍 Loading ROI geometry from {args.roi_file}")
        roi_geometry = load_roi_geometry(Path(args.roi_file))
        
        print(f"🗺️  Loading MGRS tiles from {args.mgrs_file}")
        mgrs_gdf = load_mgrs_tiles(args.mgrs_file)
        
        print("🎯 Finding intersecting MGRS tiles...")
        intersecting_tiles = find_intersecting_mgrs_tiles(mgrs_gdf, roi_geometry)
        
        print(f"📍 Found {len(intersecting_tiles)} intersecting MGRS tiles:")
        for tile in intersecting_tiles:
            print(f"   • {tile['mgrs_code']} (EPSG:{tile['epsg_code']})")
        
        print(f"\n🛰️  Creating Sentinel-2 composites from {args.start_date} to {args.end_date}")
        print(f"☁️  Using CloudScore+ threshold: {args.clear_threshold}")
        
        all_tasks = []
        total_exports = len(intersecting_tiles) * 11  # 11 bands per tile
        
        if destination == 'drive':
            print(f"📤 Exporting {total_exports} band images to Google Drive folder: {args.output_folder}")
        else:
            print(f"📤 Exporting {total_exports} band images to Cloud Storage: gs://{args.bucket_name}/{args.output_folder}")
        
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
                print(f"   📁 Queued: {tile_info['mgrs_code']}_{band_name}.tif")
        
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