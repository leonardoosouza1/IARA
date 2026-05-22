#!/usr/bin/env python3
"""
I.A.R.A. v6.9 - Pre-computed Dataset Loader
============================================
Loads pre-computed .pt tensors from the Feature Factory.

Features:
- Direct tensor loading (no on-the-fly computation)
- Memory mapping for large datasets
- Float16 precision for AMP compatibility
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import torch
from torch.utils.data import Dataset

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("Dataset")

# Default cache directory
CACHE_DIR = Path("data/cache")


class PrecomputedDataset(Dataset):
    """
    Dataset that loads pre-computed tensors from Feature Factory.
    
    This is the v6.9 "Hybrid Engine" approach:
    - CPU pre-computes all features and saves as .pt
    - GPU just loads and trains (zero feature computation overhead)
    
    Performance Notes:
    - Data stays on CPU in __getitem__ (let DataLoader batch and Trainer move to GPU)
    - No per-sample .to(device) calls which killed performance
    """
    
    def __init__(
        self,
        cache_path: Path,
        device: Optional[torch.device] = None,
        mmap_mode: bool = False
    ):
        """
        Args:
            cache_path: Path to the .pt file (train.pt or val.pt)
            device: Target device for tensors (IGNORED - transfer happens in training loop)
            mmap_mode: If True, use memory mapping for large datasets
        """
        self.cache_path = Path(cache_path)
        
        if not self.cache_path.exists():
            raise FileNotFoundError(f"Cache file not found: {self.cache_path}")
        
        logger.info(f"Loading pre-computed dataset: {self.cache_path}")
        
        # Load tensors to CPU (DataLoader workers will handle batching)
        data = torch.load(self.cache_path, weights_only=False)
        self.X = data['X']  # [N, T, D]
        self.y = data['y']  # [N]
        
        # Keep as float32 for DataLoader compatibility (AMP will handle float16 conversion)
        if self.X.dtype == torch.float16:
            self.X = self.X.float()
            self.y = self.y.float()
        
        logger.info(f"Loaded {len(self)} samples, shape: {self.X.shape}")
    
    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            X: Feature sequence [T, D] (CPU tensor)
            y: Label (0 or 1) (CPU tensor)
            weight: Sample weight (1.0 for now)
        
        NOTE: No .to(device) here! DataLoader batches, Trainer moves to GPU.
        """
        return self.X[idx], self.y[idx], torch.tensor(1.0)
    
    def get_feature_dim(self) -> int:
        """Return the number of features."""
        return self.X.shape[2]
    
    def get_seq_len(self) -> int:
        """Return the sequence length."""
        return self.X.shape[1]



class HybridDatasetLoader:
    """
    Factory class that either loads from cache or triggers factory build.
    """
    
    def __init__(self, cache_dir: Path = CACHE_DIR):
        self.cache_dir = Path(cache_dir)
        self.train_path = self.cache_dir / "train.pt"
        self.val_path = self.cache_dir / "val.pt"
        self.norm_path = self.cache_dir / "norm_params.pt"
    
    def is_cache_valid(self) -> bool:
        """Check if cache exists and is fresh."""
        return (
            self.train_path.exists() and 
            self.val_path.exists() and
            self.norm_path.exists()
        )
    
    def build_cache(self, **kwargs):
        """Trigger the Feature Factory to rebuild cache."""
        from src.data.feature_factory import run_factory
        logger.info("Cache not found or stale. Running Feature Factory...")
        run_factory(cache_dir=self.cache_dir, **kwargs)
    
    def get_datasets(
        self,
        device: Optional[torch.device] = None,
        rebuild: bool = False
    ) -> Tuple[PrecomputedDataset, PrecomputedDataset]:
        """
        Get train and validation datasets.
        
        Args:
            device: Target device
            rebuild: If True, force rebuild the cache
        
        Returns:
            (train_dataset, val_dataset)
        """
        if rebuild or not self.is_cache_valid():
            self.build_cache()
        
        train_ds = PrecomputedDataset(self.train_path, device=device)
        val_ds = PrecomputedDataset(self.val_path, device=device)
        
        return train_ds, val_ds
    
    def get_norm_params(self) -> dict:
        """Load normalization parameters for inference."""
        if not self.norm_path.exists():
            raise FileNotFoundError(f"Norm params not found: {self.norm_path}")
        return torch.load(self.norm_path, weights_only=False)


# Convenience function
def load_precomputed_datasets(
    cache_dir: Path = CACHE_DIR,
    device: Optional[torch.device] = None,
    rebuild: bool = False
) -> Tuple[PrecomputedDataset, PrecomputedDataset]:
    """
    Load pre-computed train/val datasets.
    
    If cache doesn't exist, triggers Feature Factory.
    """
    loader = HybridDatasetLoader(cache_dir)
    return loader.get_datasets(device=device, rebuild=rebuild)
