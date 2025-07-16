#!/usr/bin/env python3
"""
Script to download XYZ tiles from a tile server for random locations within a geometry.
The tiles are downloaded in parallel with rate limiting to avoid overwhelming the server.
"""

import os
import math
import random
import time
import requests
import geopandas as gpd
from shapely.geometry import Point, box
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import argparse
from tqdm import tqdm


def deg2num(lat_deg, lon_deg, zoom):
    """Convert latitude/longitude to tile numbers at given zoom level."""
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    x = int((lon_deg + 180.0) / 360.0 * n)
    y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return (x, y)


def num2deg(x, y, zoom):
    """Convert tile numbers to latitude/longitude at given zoom level."""
    n = 2.0 ** zoom
    lon_deg = x / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * y / n)))
    lat_deg = math.degrees(lat_rad)
    return (lat_deg, lon_deg)


def tile_bounds(x, y, zoom):
    """Get the bounding box of a tile in lat/lon coordinates."""
    lat_deg_nw, lon_deg_nw = num2deg(x, y, zoom)
    lat_deg_se, lon_deg_se = num2deg(x + 1, y + 1, zoom)
    return box(lon_deg_nw, lat_deg_se, lon_deg_se, lat_deg_nw)


def is_tile_fully_contained(x, y, zoom, geometry):
    """Check if a tile is fully contained within the given geometry."""
    tile_geom = tile_bounds(x, y, zoom)
    return geometry.contains(tile_geom)


def get_tiles_in_geometry(geometry, zoom, max_tiles=-1):
    """Get all tile coordinates that are fully contained within the geometry."""
    bounds = geometry.bounds
    min_lon, min_lat, max_lon, max_lat = bounds
    
    # Get the tile range that covers the geometry
    min_x, max_y = deg2num(min_lat, min_lon, zoom)
    max_x, min_y = deg2num(max_lat, max_lon, zoom)
    
    # Collect all tiles that are fully contained
    contained_tiles = []
    for x in range(min_x, max_x + 1):
        for y in range(min_y, max_y + 1):
            if is_tile_fully_contained(x, y, zoom, geometry):
                contained_tiles.append((x, y))
                if max_tiles != -1 and len(contained_tiles) >= max_tiles:
                    return contained_tiles
    
    return contained_tiles


def download_tile(x, y, zoom, base_url, output_dir, session=None, debug=False):
    """Download a single tile and save it to disk."""
    if session is None:
        session = requests.Session()
    
    # Format the URL
    url = base_url.format(z=zoom, x=x, y=y)
    
    # Create filename
    filename = f"tile_{zoom}_{x}_{y}.webp"
    filepath = output_dir / filename
    
    # Skip if file already exists
    if filepath.exists():
        return {"success": True, "message": f"Skipped existing file: {filename}", "x": x, "y": y}
    
    try:
        # Add a small delay to be nice to the server
        time.sleep(random.uniform(0.1, 0.3))
        
        response = session.get(url, timeout=30)
        
        # Check for common error cases
        if response.status_code == 204:
            # Check zoom level headers
            max_zoom = response.headers.get('x-max-zoom', 'unknown')
            requested_zoom = response.headers.get('x-requested-zoom', zoom)
            return {"success": False, "message": f"No content available - requested zoom {requested_zoom}, max available zoom {max_zoom}", "x": x, "y": y}
        
        response.raise_for_status()
        
        # Check if response has content
        if len(response.content) == 0:
            return {"success": False, "message": f"Empty response from server", "x": x, "y": y}
        
        # Save the tile
        with open(filepath, 'wb') as f:
            f.write(response.content)
        
        if debug:
            content_size = len(response.content)
            return {"success": True, "message": f"Downloaded: {filename} ({content_size} bytes)", "x": x, "y": y}
        else:
            return {"success": True, "message": f"Downloaded: {filename}", "x": x, "y": y}
        
    except Exception as e:
        return {"success": False, "message": f"Failed to download {filename}: {str(e)}", "x": x, "y": y}


def download_tiles_parallel(tile_coords, zoom, base_url, output_dir, max_workers=4, debug=False):
    """Download tiles in parallel with rate limiting."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Create a session for each worker
        session = requests.Session()
        
        # Submit all download tasks
        future_to_tile = {
            executor.submit(download_tile, x, y, zoom, base_url, output_dir, session, debug): (x, y)
            for x, y in tile_coords
        }
        
        # Process completed downloads with progress bar
        for future in tqdm(as_completed(future_to_tile), total=len(tile_coords), desc="Downloading tiles"):
            result = future.result()
            results.append(result)
    
    return results


def test_zoom_level(base_url, zoom, sample_tile_coords):
    """Test if a zoom level is available by trying to download a sample tile."""
    if not sample_tile_coords:
        return True, "No sample tiles to test"
    
    x, y = sample_tile_coords[0]
    url = base_url.format(z=zoom, x=x, y=y)
    
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 204:
            max_zoom = response.headers.get('x-max-zoom', 'unknown')
            return False, f"Zoom level {zoom} not available. Maximum zoom: {max_zoom}"
        response.raise_for_status()
        return True, f"Zoom level {zoom} is available"
    except Exception as e:
        return False, f"Error testing zoom level: {str(e)}"


def main():
    parser = argparse.ArgumentParser(description="Download XYZ tiles for random locations within a geometry")
    parser.add_argument("tile_server",
                       help="Tile server URL, including the {z}/{x}/{y} placeholder")
    parser.add_argument("--geojson_file", default="~/Data/java.geojson", 
                       help="Path to GeoJSON file with geometry")
    parser.add_argument("--output_dir", default="/Users/noah/Data/java_s2_tiles/",
                       help="Output directory for downloaded tiles")
    parser.add_argument("--zoom", type=int, default=12,
                       help="Zoom level for tiles (default: 12)")
    parser.add_argument("--num_tiles", type=int, default=-1,
                       help="Number of random tiles to download")
    parser.add_argument("--max_workers", type=int, default=4,
                       help="Maximum number of parallel download workers")
    parser.add_argument("--max_search_tiles", type=int, default=-1,
                       help="Maximum number of tiles to search when finding contained tiles")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug output")
    
    args = parser.parse_args()
    
    # Expand user path
    geojson_file = os.path.expanduser(args.geojson_file)
    
    print(f"Reading geometry from: {geojson_file}")
    
    # Read the GeoJSON file
    try:
        gdf = gpd.read_file(geojson_file)
    except Exception as e:
        print(f"Error reading GeoJSON file: {e}")
        return
    
    # Get the geometry (assume single geometry or union all)
    if len(gdf) == 1:
        geometry = gdf.geometry.iloc[0]
    else:
        geometry = gdf.union_all()
    
    print(f"Geometry bounds: {geometry.bounds}")
    print(f"Finding tiles at zoom level {args.zoom} that are fully contained in the geometry...")
    
    # Get all tiles that are fully contained in the geometry
    contained_tiles = get_tiles_in_geometry(geometry, args.zoom, args.max_search_tiles)
    
    if not contained_tiles:
        print("No tiles found that are fully contained in the geometry!")
        return
    
    print(f"Found {len(contained_tiles)} tiles fully contained in the geometry")
    
    # Test zoom level availability
    if args.debug:
        print(f"Testing zoom level {args.zoom} availability...")
        zoom_available, zoom_message = test_zoom_level(args.tile_server, args.zoom, contained_tiles[:1])
        print(f"Zoom test result: {zoom_message}")
        if not zoom_available:
            print("Consider using a lower zoom level.")
    
    # Select random tiles to download
    num_to_download = len(contained_tiles) if args.num_tiles == -1 else min(args.num_tiles, len(contained_tiles))
    selected_tiles = random.sample(contained_tiles, num_to_download)
    
    print(f"Randomly selected {num_to_download} tiles to download")
    print(f"Output directory: {args.output_dir}")
    print(f"Using {args.max_workers} parallel workers")
    
    # Download the tiles
    results = download_tiles_parallel(
        selected_tiles, 
        args.zoom, 
        args.tile_server, 
        args.output_dir, 
        args.max_workers,
        args.debug
    )
    
    # Print summary
    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]
    
    print("\n" + "="*60)
    print("DOWNLOAD SUMMARY")
    print("="*60)
    print(f"Total tiles processed: {len(results)}")
    print(f"Successfully downloaded: {len(successful)}")
    print(f"Failed downloads: {len(failed)}")
    
    if failed:
        print("\nFailed downloads:")
        for result in failed:
            print(f"  - Tile {result['x']},{result['y']}: {result['message']}")
    
    if successful:
        print(f"\nTiles saved to: {args.output_dir}")
        sample_tiles = successful[:3]
        print("Sample downloaded tiles:")
        for result in sample_tiles:
            print(f"  - Tile {result['x']},{result['y']}: {result['message']}")


if __name__ == "__main__":
    main() 