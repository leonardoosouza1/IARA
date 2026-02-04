
import polars as pl
import torch
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SignalCheck")

def check_signal():
    # Load cache
    cache_path = Path("data/cache/train.pt")
    if not cache_path.exists():
        logger.error("Cache not found")
        return

    logger.info("Loading dataset...")
    data = torch.load(cache_path, weights_only=False)
    X = data['X'].numpy() # [N, T, D]
    y = data['y'].numpy() # [N]
    
    logger.info(f"Shape: {X.shape}")
    
    # Check naive correlation of last timestep features with target
    # X[:, -1, :] is the feature vector at the moment of prediction
    last_step_features = X[:, -1, :]
    
    # Feature 15 is labels in some contexts? No, features are 16 dims.
    # Check correlation of each feature with y
    n_features = X.shape[2]
    
    logger.info("--- Feature Correlation with Target (Linear Signal) ---")
    correlations = []
    for i in range(n_features):
        feat = last_step_features[:, i]
        # remove nans/inf
        mask = np.isfinite(feat)
        if mask.sum() < len(feat):
            logger.warning(f"Feature {i} has {len(feat)-mask.sum()} NaNs")
        
        corr = np.corrcoef(feat[mask], y[mask])[0, 1]
        correlations.append((i, corr))
        print(f"Feat {i}: {corr:.4f}")
        
    # Check Class Balance
    pos = y.sum()
    total = len(y)
    print(f"\nClass Balance: {pos/total:.2%}")
    
    # Check Label Consistency
    # If using log returns to label, check if label matches sign of visual return
    # This is harder without raw prices, but let's check basic sanity
    print(f"\nLabels: min={y.min()}, max={y.max()}, mean={y.mean()}")

if __name__ == "__main__":
    check_signal()
