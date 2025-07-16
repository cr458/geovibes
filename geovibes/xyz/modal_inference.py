#!/usr/bin/env python3

import argparse
import json
import logging
import os
import re
import time
from typing import List, Tuple, Optional, Dict, Any

import modal
from modal import Secret

import torch
import timm
import numpy as np
from joblib import Parallel, delayed
import torchvision.io as tvio
import torchvision.transforms.functional as F
import pandas as pd
import glob
import mercantile
import pyarrow as pa
import pyarrow.parquet as pq
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = modal.App("tile-embedding-decode-large-batch")

image = modal.Image.debian_slim().pip_install([
    "torch>=2.0.0",
    "torchvision>=0.15.0", 
    "timm>=0.9.0",
    "pillow>=9.0.0",
    "pandas>=1.5.0",
    "pyarrow>=10.0.0",
    "mercantile>=1.2.0",
    "tqdm>=4.64.0",
    "boto3>=1.26.0",
    "numpy>=1.24.0",
    "joblib>=1.2.0"
])


def parse_tile_from_filename(filename: str) -> Optional[Tuple[int, int, int]]:
    pattern = r'tile_(\d+)_(\d+)_(\d+)\.'
    match = re.search(pattern, filename)
    
    if match:
        zoom, x, y = map(int, match.groups())
        return zoom, x, y
    else:
        logger.warning(f"Could not parse tile coordinates from: {filename}")
        return None


class MountedTileDataset(Dataset):
    
    def __init__(self, mount_path: str, transforms, max_files: Optional[int] = None):
        self.mount_path = mount_path
        self.transforms = transforms
        self.max_files = max_files
        
        logger.info(f"Discovering files in {self.mount_path}...")
        all_files = []
        
        pattern = os.path.join(self.mount_path, "**/*.png")
        image_files = glob.glob(pattern, recursive=True)
        
        logger.info(f"Found {len(image_files)} image files")
        
        logger.info("Parsing tile coordinates from filenames...")
        
        def process_file(filepath):
            filename = os.path.basename(filepath)
            coords = parse_tile_from_filename(filename)
            if coords is not None:
                return (filepath, coords)
            return None
        
        results = Parallel(n_jobs=-1, verbose=10)(
            delayed(process_file)(filepath) for filepath in image_files)
        all_files = [result for result in results if result is not None]
        
        if self.max_files and len(all_files) > self.max_files:
            all_files = all_files[:self.max_files]
        
        self.file_list = all_files
        logger.info(f"Discovered {len(all_files)} valid tile images")
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        filepath, coords = self.file_list[idx]
        
        try:
            # 1. zero-copy read from disk (runs in C, GIL released)
            img_bytes = tvio.read_file(filepath)                         # Tensor(uint8)

            # 2. PNG/JPEG → RGB tensor, shape = (C, H, W), dtype = uint8
            img_tensor = tvio.decode_image(img_bytes,
                                        mode=tvio.ImageReadMode.RGB)  # Tensor(uint8)

            # 3. to float32   0-1 range
            img_tensor = img_tensor.float().div_(255.0)                  # Tensor(float32)

            # 4. If your timm/torchvision transforms expect PIL, convert once here
            img_pil = F.to_pil_image(img_tensor, mode="RGB")             # PIL.Image
            img_tensor = self.transforms(img_pil)                        # final tensor

            return {
                "image": img_tensor,
                "filepath": filepath,
                "coords": coords,
                "valid": True,
            }
        except Exception as e:
            logger.warning(f"Failed to load image {filepath}: {e}")
            return {
                "image": torch.zeros(3, 224, 224),
                "filepath": filepath,
                "coords": coords,
                "valid": False,
            }
    


def custom_collate_fn(batch):
    images = torch.stack([item['image'] for item in batch])
    filepaths = [item['filepath'] for item in batch]
    coords = [item['coords'] for item in batch]
    valid = torch.tensor([item['valid'] for item in batch])
    
    return {
        'image': images,
        'filepath': filepaths,
        'coords': coords,
        'valid': valid
    }


def prepare_tile_data(embeddings: np.ndarray, image_filepaths: List[str], tile_coords: List[Tuple[int, int, int]]) -> List[Dict[str, Any]]:
    logger.info("Preparing tile data with geometries...")
    
    tile_data = []
    
    for i, (embedding, filepath, coords) in enumerate(zip(embeddings, image_filepaths, tile_coords)):
        zoom, x, y = coords
        tile_bounds = mercantile.bounds(x, y, zoom)
        center_lon = (tile_bounds.west + tile_bounds.east) / 2
        center_lat = (tile_bounds.south + tile_bounds.north) / 2
        geometry_wkt = f"POINT({center_lon} {center_lat})"
        tile_id = f"{zoom}_{x}_{y}"
        
        tile_data.append({
            'id': tile_id,
            'zoom': zoom,
            'x': x,
            'y': y,
            'embedding': embedding.tolist(),
            'geometry_wkt': geometry_wkt,
            'image_filepath': filepath,
            'west': tile_bounds.west,
            'south': tile_bounds.south,
            'east': tile_bounds.east,
            'north': tile_bounds.north,
            'center_lon': center_lon,
            'center_lat': center_lat
        })
        
        if (i + 1) % 1000 == 0:
            logger.info(f"Processed {i + 1}/{len(embeddings)} tiles...")
    
    logger.info(f"Prepared {len(tile_data)} tile records")
    return tile_data


def write_parquet_chunk(tile_data: List[Dict[str, Any]], output_gcs_path: str, file_counter: int):
    logger.info(f"Writing chunk {file_counter} with {len(tile_data)} records to {output_gcs_path}")
    
    if not output_gcs_path.startswith('gs://'):
        raise ValueError("Output path must start with 'gs://'")
    
    path_parts = output_gcs_path[5:].split('/', 1)
    bucket = path_parts[0]
    prefix = path_parts[1] if len(path_parts) > 1 else ""
    
    df = pd.DataFrame(tile_data)
    chunk_filename = f"embeddings_chunk_{file_counter:06d}.parquet"
    
    if prefix:
        output_dir = f"/gcs-mount/{prefix}"
        os.makedirs(output_dir, exist_ok=True)
        output_file = f"{output_dir}/{chunk_filename}"
    else:
        output_file = f"/gcs-mount/{chunk_filename}"
    
    table = pa.Table.from_pandas(df)
    pq.write_table(table, output_file)
    
    logger.info(f"Wrote chunk {file_counter} to {output_file}")


@app.function(
    image=image,
    gpu="T4",
    secrets=[Secret.from_name("gcs-aws-hmac-credentials")],
    cpu=8,
    timeout=86400,
    memory=32000,
    region='us',
    volumes={
        "/gcs-mount": modal.CloudBucketMount(
            bucket_name="geovibes",
            bucket_endpoint_url="https://storage.googleapis.com",
            secret=Secret.from_name("gcs-aws-hmac-credentials"),
        )
    }
)
def process_tile_embeddings(
    input_gcs_path: str,
    output_gcs_path: str,
    model_name: str = "resnet34.a3_in1k",
    batch_size: int = 256,
    max_files: Optional[int] = None,
    chunk_size: int = 10000,
    num_workers: int = 16,
    amp: bool = False
):
    
    logger.info(f"Starting tile embedding processing")
    logger.info(f"Input: {input_gcs_path}")
    logger.info(f"Output: {output_gcs_path}")
    logger.info(f"Model: {model_name}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"AMP enabled: {amp}")
    
    torch.backends.cuda.matmul.allow_tf32 = True
    
    if not input_gcs_path.startswith('gs://'):
        raise ValueError("Input path must start with 'gs://'")
    
    path_parts = input_gcs_path[5:].split('/', 1)
    bucket = path_parts[0]
    prefix = path_parts[1] if len(path_parts) > 1 else ""
    
    input_mount_path = f"/gcs-mount/{prefix}" if prefix else "/gcs-mount"
    
    logger.info(f"Loading model: {model_name}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        model = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=0,
        )
        model = model.eval().to(device)
        
        data_config = timm.data.resolve_model_data_config(model)
        transforms = timm.data.create_transform(**data_config, is_training=False)
        
        logger.info(f"Model loaded. Feature dimension: {model.num_features}")
        logger.info(f"Using batch size: {batch_size}")
        
    except Exception as e:
        logger.error(f"Failed to load model {model_name}: {e}")
        raise
    
    logger.info("Creating dataset for mounted filesystem access...")
    dataset = MountedTileDataset(input_mount_path, transforms, max_files)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True if device.type == 'cuda' else False,
        persistent_workers=True,
        prefetch_factor=8,
        collate_fn=custom_collate_fn
    )
    
    logger.info(f"Starting streaming inference with DataLoader (batch_size={batch_size}, workers={num_workers})")
    logger.info("Using mounted cloud storage for output files")
    
    batch_embeddings = []
    batch_keys = []
    batch_coords = []
    total_processed = 0
    file_counter = 0
    
    start_time = time.time()
    processed_count = 0
    batch_end_time = time.time()
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(
            tqdm(dataloader, desc="Streaming GPU inference")):
            try:
                batch_start_time = time.time()
                loader_lag = batch_start_time - batch_end_time
                
                valid_mask = batch['valid']
                valid_images = batch['image'][valid_mask]
                valid_filepaths = [batch['filepath'][i] for i in range(len(batch['filepath'])) if valid_mask[i]]
                valid_coords = [batch['coords'][i] for i in range(len(batch['coords'])) if valid_mask[i]]
                
                if len(valid_images) == 0:
                    batch_end_time = time.time()
                    continue
                
                batch_tensor = valid_images.to(device)
                
                if amp:
                    with torch.cuda.amp.autocast():
                        model_embeddings = model(batch_tensor)
                    model_embeddings = model_embeddings.half().cpu().numpy()
                else:
                    model_embeddings = model(batch_tensor)
                    model_embeddings = model_embeddings.cpu().numpy()
                
                batch_embeddings.extend(model_embeddings)
                batch_keys.extend(valid_filepaths)
                batch_coords.extend(valid_coords)
                
                processed_count += len(valid_images)
                batch_end_time = time.time()
                
                if (batch_idx + 1) % 100 == 0:
                    if batch_embeddings:
                        embeddings_array = np.array(batch_embeddings)
                        tile_data = prepare_tile_data(embeddings_array, batch_keys, batch_coords)
                        write_parquet_chunk(tile_data, output_gcs_path, file_counter)
                        total_processed += len(batch_embeddings)
                        file_counter += 1
                        batch_embeddings = []
                        batch_keys = []
                        batch_coords = []
                        logger.info(f"Wrote chunk {file_counter} with {len(tile_data)} records. Total processed: {total_processed}")
                
                if (batch_idx + 1) % 10 == 0:
                    elapsed = time.time() - start_time
                    rate = processed_count / elapsed if elapsed > 0 else 0
                    logger.info(f"Processed {processed_count} images ({rate:.1f} img/s), loader lag: {loader_lag:.3f}s")
                    
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.error(f"GPU OOM error on batch {batch_idx}. Try reducing batch size.")
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
                    raise
                else:
                    logger.error(f"CUDA error on batch {batch_idx}: {e}")
                    raise
    
    if batch_embeddings:
        embeddings_array = np.array(batch_embeddings)
        tile_data = prepare_tile_data(embeddings_array, batch_keys, batch_coords)
        write_parquet_chunk(tile_data, output_gcs_path, file_counter)
        total_processed += len(batch_embeddings)
        file_counter += 1
        logger.info(f"Wrote final chunk {file_counter} with {len(tile_data)} records. Total processed: {total_processed}")
    
    total_time = time.time() - start_time
    logger.info(f"Embedding extraction completed in {total_time:.1f} seconds")
    logger.info(f"Average speed: {total_processed / total_time:.1f} images/second")
    logger.info(f"Total files written: {file_counter}")
    
    logger.info("Processing complete!")
    return {
        "processed_images": total_processed,
        "output_path": output_gcs_path,
        "model_used": model_name,
        "processing_time": total_time,
        "files_written": file_counter
    }


@app.local_entrypoint()
def main(
    input_gcs_path: str,
    output_gcs_path: str,
    model_name: str = "resnet34.a3_in1k",
    batch_size: int = 256,
    max_files: int = None,
    chunk_size: int = 10000,
    num_workers: int = 16,
    amp: bool = False,
    config: str = None
):
    
    config_params = {}
    if config:
        with open(config, 'r') as f:
            config_params = json.load(f)
    
    params = {
        "input_gcs_path": input_gcs_path,
        "output_gcs_path": output_gcs_path,
        "model_name": model_name,
        "batch_size": batch_size,
        "max_files": max_files,
        "chunk_size": chunk_size,
        "num_workers": num_workers,
        "amp": amp,
        **config_params
    }
    
    params = {k: v for k, v in params.items() if v is not None}
    
    print(f"Running with parameters: {params}")
    
    result = process_tile_embeddings.remote(**params)
    
    print(f"Processing complete! Results: {result}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process tile images and generate embeddings using Modal")
    
    parser.add_argument(
        "--input-gcs-path", 
        required=True,
        help="GCS path to input tile images (e.g., gs://bucket/path/to/tiles)"
    )
    
    parser.add_argument(
        "--output-gcs-path",
        required=True, 
        help="GCS path for output parquet files"
    )
    
    parser.add_argument(
        "--model-name",
        default="resnet34.a3_in1k",
        help="timm model name to use for embeddings (default: resnet34.a3_in1k)"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for processing (default: 256)"
    )
    
    parser.add_argument(
        "--max-files",
        type=int,
        help="Maximum number of files to process (for testing)"
    )
    
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=10000,
        help="Number of records per parquet file (default: 10000)"
    )
    
    parser.add_argument(
        "--num-workers",
        type=int,
        default=16,
        help="Number of parallel download threads (default: 16)"
    )
    
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Use automatic mixed precision"
    )
    
    parser.add_argument(
        "--config",
        help="Path to JSON config file with parameters"
    )
    
    args = parser.parse_args()
    
    config = {}
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    
    params = {
        "input_gcs_path": args.input_gcs_path,
        "output_gcs_path": args.output_gcs_path,
        "model_name": args.model_name,
        "batch_size": args.batch_size,
        "max_files": args.max_files,
        "chunk_size": args.chunk_size,
        "num_workers": args.num_workers,
        "amp": args.amp,
        **config
    }
    
    params = {k: v for k, v in params.items() if v is not None}
    
    print(f"Running with parameters: {params}")
    print("Note: Run with 'modal run' for Modal execution or 'python' for direct execution")