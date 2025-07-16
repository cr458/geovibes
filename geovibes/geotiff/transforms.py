import time
import logging
from typing import Tuple

import torch
import torch.nn as nn
import kornia as K

logger = logging.getLogger(__name__)


class RescaleIntensity(nn.Module):
    """Rescale intensity values using pure PyTorch operations."""
    
    def __init__(self, out_min_max: Tuple = (0, 1), percentiles: Tuple = (2, 98)):
        super().__init__()
        self.out_min, self.out_max = out_min_max
        self.percentiles = torch.tensor(percentiles, dtype=torch.float32)
        
    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        start_time = time.time()
        logger.debug(f"RescaleIntensity: Starting with tensor shape {tensor.shape}")
        
        original_shape = tensor.shape
        
        shape_start = time.time()
        if len(original_shape) == 3:
            tensor = tensor.unsqueeze(0)
            logger.debug(f"RescaleIntensity: Unsqueezed 3D tensor to {tensor.shape}")
        elif len(original_shape) == 5:
            tensor = tensor.squeeze(2)
            logger.debug(f"RescaleIntensity: Squeezed 5D tensor to {tensor.shape}")
        shape_elapsed = time.time() - shape_start
        logger.debug(f"RescaleIntensity: Shape handling took {shape_elapsed:.4f}s")
        
        batch_size, channels = tensor.shape[:2]
        
        flatten_start = time.time()
        tensor_flat = tensor.view(batch_size, channels, -1)
        flatten_elapsed = time.time() - flatten_start
        logger.debug(f"RescaleIntensity: Flatten took {flatten_elapsed:.4f}s, shape: {tensor_flat.shape}")
        
        percentile_start = time.time()
        lower = torch.quantile(tensor_flat, self.percentiles[0] / 100.0, dim=2, keepdim=True)
        upper = torch.quantile(tensor_flat, self.percentiles[1] / 100.0, dim=2, keepdim=True)
        percentile_elapsed = time.time() - percentile_start
        logger.debug(f"RescaleIntensity: Percentile calculation took {percentile_elapsed:.4f}s")
        
        reshape_start = time.time()
        lower = lower.view(batch_size, channels, 1, 1)
        upper = upper.view(batch_size, channels, 1, 1)
        reshape_elapsed = time.time() - reshape_start
        logger.debug(f"RescaleIntensity: Reshape took {reshape_elapsed:.4f}s")
        
        clamp_start = time.time()
        tensor = torch.clamp(tensor, lower, upper)
        in_range = upper - lower
        
        in_range = torch.where(in_range == 0, torch.ones_like(in_range), in_range)
        
        tensor = (tensor - lower) / in_range
        out_range = self.out_max - self.out_min
        tensor = tensor * out_range + self.out_min
        clamp_elapsed = time.time() - clamp_start
        logger.debug(f"RescaleIntensity: Clamp and normalize took {clamp_elapsed:.4f}s")
        
        restore_start = time.time()
        if len(original_shape) == 3:
            tensor = tensor.squeeze(0)
            logger.debug(f"RescaleIntensity: Restored to 3D shape {tensor.shape}")
        restore_elapsed = time.time() - restore_start
        logger.debug(f"RescaleIntensity: Shape restore took {restore_elapsed:.4f}s")
        
        total_elapsed = time.time() - start_time
        if total_elapsed > 0.1:
            logger.warning(f"RescaleIntensity: Slow transform took {total_elapsed:.3f}s")
        else:
            logger.debug(f"RescaleIntensity: Total time {total_elapsed:.4f}s")
        
        return tensor


class RescaledImageNetTransform(nn.Module):
    """Fully vectorized transform pipeline for ImageNet preprocessing."""
    
    def __init__(self):
        super().__init__()
        self.rescale = RescaleIntensity((0, 1), (2, 98))
        self.resize = K.geometry.Resize((224, 224))
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    
    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        start_time = time.time()
        logger.debug(f"RescaledImageNetTransform: Starting with tensor shape {x.shape}")
        
        batch_start = time.time()
        is_single = len(x.shape) == 3
        if is_single:
            x = x.unsqueeze(0)
            logger.debug(f"RescaledImageNetTransform: Added batch dimension, shape: {x.shape}")
        batch_elapsed = time.time() - batch_start
        logger.debug(f"RescaledImageNetTransform: Batch handling took {batch_elapsed:.4f}s")
        
        rescale_start = time.time()
        logger.debug(f"RescaledImageNetTransform: Applying rescale transform...")
        x = self.rescale(x)
        rescale_elapsed = time.time() - rescale_start
        logger.debug(f"RescaledImageNetTransform: Rescale took {rescale_elapsed:.4f}s, output shape: {x.shape}")
        
        resize_start = time.time()
        logger.debug(f"RescaledImageNetTransform: Applying resize transform...")
        x = self.resize(x)
        resize_elapsed = time.time() - resize_start
        logger.debug(f"RescaledImageNetTransform: Resize took {resize_elapsed:.4f}s, output shape: {x.shape}")
        
        norm_start = time.time()
        logger.debug(f"RescaledImageNetTransform: Applying normalization...")
        x = (x - self.mean) / self.std
        norm_elapsed = time.time() - norm_start
        logger.debug(f"RescaledImageNetTransform: Normalization took {norm_elapsed:.4f}s")
        
        squeeze_start = time.time()
        if is_single:
            x = x.squeeze(0)
            logger.debug(f"RescaledImageNetTransform: Removed batch dimension, final shape: {x.shape}")
        squeeze_elapsed = time.time() - squeeze_start
        logger.debug(f"RescaledImageNetTransform: Squeeze handling took {squeeze_elapsed:.4f}s")
        
        total_elapsed = time.time() - start_time
        if total_elapsed > 0.2:
            logger.warning(f"RescaledImageNetTransform: Slow transform pipeline took {total_elapsed:.3f}s")
        else:
            logger.debug(f"RescaledImageNetTransform: Total pipeline time {total_elapsed:.4f}s")
        
        return x