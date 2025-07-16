#!/usr/bin/env python3
"""
Script to split 512x512 WebP tiles into 16 128x128 PNG tiles.
This effectively converts zoom level 12 tiles to zoom level 14 tiles.
"""

import os
import argparse
from pathlib import Path
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import re


def parse_tile_filename(filename):
    """
    Parse tile filename to extract zoom, x, y coordinates.
    Expected format: tile_12_3314_2133.webp
    """
    match = re.match(r'tile_(\d+)_(\d+)_(\d+)\.webp$', filename)
    if match:
        zoom, x, y = map(int, match.groups())
        return zoom, x, y
    return None


def split_tile_to_subtiles(input_path, output_dir, target_zoom=14, tile_size=128):
    """
    Split a single 512x512 tile into 16 128x128 tiles.
    
    Args:
        input_path: Path to input WebP file
        output_dir: Output directory for PNG tiles
        target_zoom: Target zoom level (default 14)
        tile_size: Size of output tiles (default 128)
    
    Returns:
        dict: Processing result with success status and message
    """
    try:
        filename = input_path.name
        
        # Parse the original tile coordinates
        parsed = parse_tile_filename(filename)
        if not parsed:
            return {
                "success": False,
                "message": f"Could not parse filename: {filename}",
                "input_file": filename,
                "output_files": []
            }
        
        zoom, x, y = parsed
        
        # Open the input image
        with Image.open(input_path) as img:
            # Verify image size
            if img.size != (512, 512):
                return {
                    "success": False,
                    "message": f"Expected 512x512 image, got {img.size}",
                    "input_file": filename,
                    "output_files": []
                }
            
            # Convert to RGB if necessary (WebP might have transparency)
            if img.mode in ('RGBA', 'LA', 'P'):
                # Create white background for transparent images
                background = Image.new('RGB', img.size, (255, 255, 255))
                if img.mode == 'P':
                    img = img.convert('RGBA')
                background.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
                img = background
            elif img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Calculate new tile coordinates for zoom 14
            # Each tile at zoom N becomes 16 tiles at zoom N+2
            base_x = x * 4  # 2^(14-12) = 4
            base_y = y * 4
            
            output_files = []
            
            # Split into 16 tiles: 4x4 grid
            for dy in range(4):
                for dx in range(4):
                    # Calculate crop box (left, top, right, bottom)
                    left = dx * 128  # 512 / 4 = 128
                    top = dy * 128
                    right = left + 128
                    bottom = top + 128
                    
                    # Crop the tile (already 128x128, no resizing needed)
                    tile = img.crop((left, top, right, bottom))
                    
                    # Calculate new tile coordinates
                    new_x = base_x + dx
                    new_y = base_y + dy
                    
                    # Create output filename
                    output_filename = f"tile_{target_zoom}_{new_x}_{new_y}.png"
                    output_path = output_dir / output_filename
                    
                    # Save as PNG
                    tile.save(output_path, 'PNG', optimize=True)
                    output_files.append(output_filename)
            
            return {
                "success": True,
                "message": f"Split {filename} into {len(output_files)} tiles",
                "input_file": filename,
                "output_files": output_files
            }
            
    except Exception as e:
        return {
            "success": False,
            "message": f"Error processing {filename}: {str(e)}",
            "input_file": filename,
            "output_files": []
        }


def process_tiles_parallel(input_dir, output_dir, max_workers=4, target_zoom=14, tile_size=128):
    """
    Process all WebP tiles in parallel.
    
    Args:
        input_dir: Directory containing input WebP files
        output_dir: Directory for output PNG files
        max_workers: Number of parallel workers
        target_zoom: Target zoom level
        tile_size: Size of output tiles
    
    Returns:
        list: Results from all processing tasks
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all WebP files
    webp_files = list(input_dir.glob("*.webp"))
    
    if not webp_files:
        print(f"No WebP files found in {input_dir}")
        return []
    
    print(f"Found {len(webp_files)} WebP files to process")
    
    results = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all processing tasks
        future_to_file = {
            executor.submit(split_tile_to_subtiles, webp_file, output_dir, target_zoom, tile_size): webp_file
            for webp_file in webp_files
        }
        
        # Process completed tasks with progress bar
        for future in tqdm(as_completed(future_to_file), total=len(webp_files), desc="Processing tiles"):
            result = future.result()
            results.append(result)
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Split 512x512 WebP tiles into 16 128x128 PNG tiles",
        epilog="""
EXAMPLE:
    python split_tiles_to_png.py --input_dir ~/Data/java_s2_tiles/ --output_dir ~/Data/java_s2_tiles_14/
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("--input_dir", default="~/Data/java_s2_tiles/",
                       help="Input directory containing WebP tiles")
    parser.add_argument("--output_dir", default="~/Data/java_s2_tiles_14/",
                       help="Output directory for PNG tiles")
    parser.add_argument("--target_zoom", type=int, default=14,
                       help="Target zoom level for output tiles")
    parser.add_argument("--tile_size", type=int, default=128,
                       help="Size of output tiles in pixels")
    parser.add_argument("--max_workers", type=int, default=4,
                       help="Maximum number of parallel workers")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug output")
    
    args = parser.parse_args()
    
    # Expand user paths
    input_dir = os.path.expanduser(args.input_dir)
    output_dir = os.path.expanduser(args.output_dir)
    
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Target zoom level: {args.target_zoom}")
    print(f"Output tile size: {args.tile_size}x{args.tile_size}")
    print(f"Using {args.max_workers} parallel workers")
    
    # Verify input directory exists
    if not os.path.exists(input_dir):
        print(f"Error: Input directory does not exist: {input_dir}")
        return
    
    # Process all tiles
    results = process_tiles_parallel(
        input_dir=input_dir,
        output_dir=output_dir,
        max_workers=args.max_workers,
        target_zoom=args.target_zoom,
        tile_size=args.tile_size
    )
    
    # Print summary
    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]
    
    total_output_tiles = sum(len(r["output_files"]) for r in successful)
    
    print("\n" + "="*60)
    print("PROCESSING SUMMARY")
    print("="*60)
    print(f"Total input files processed: {len(results)}")
    print(f"Successfully processed: {len(successful)}")
    print(f"Failed: {len(failed)}")
    print(f"Total output tiles created: {total_output_tiles}")
    
    if failed:
        print("\nFailed files:")
        for result in failed:
            print(f"  - {result['input_file']}: {result['message']}")
    
    if successful:
        print(f"\nTiles saved to: {output_dir}")
        
        if args.debug:
            print("\nSample successful conversions:")
            for result in successful[:3]:
                print(f"  - {result['input_file']} -> {len(result['output_files'])} tiles")
                for output_file in result['output_files'][:2]:  # Show first 2 output files
                    print(f"    + {output_file}")
    
    print(f"\nConversion complete! {total_output_tiles} PNG tiles ready at zoom level {args.target_zoom}")


if __name__ == "__main__":
    main() 