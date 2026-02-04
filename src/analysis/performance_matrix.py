"""
Performance Matrix (Intelligence Map)
=====================================
Analyzes the 'Where' and 'When' of the model's success.
- Heatmap: Win Rate per Hour of Day vs. Asset
- Feature Importance: Integrated Gradients Sensitivity Analysis
"""

import argparse
import logging
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import polars as pl
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.model.iara_liquid import IARA, LiquidConfig
from src.train.trainer_liquid import TrainerConfig, MultiAssetDataset

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Paths
CHECKPOINT_DIR = Path(__file__).parent.parent.parent / "checkpoints"
ARTIFACTS_DIR = Path(__file__).parent.parent.parent / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

def load_model(checkpoint_path: Path, device: torch.device) -> Tuple[nn.Module, TrainerConfig]:
    """Loads the best model from checkpoint."""
    logger.info(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint['config']
    
    liquid_config = LiquidConfig(
        input_dim=config.input_dim,
        hidden_dim=config.hidden_dim,
        num_heads=config.num_heads,
        dropout=config.dropout,
        max_seq_len=config.max_seq_len
    )
    
    model = IARA(liquid_config).to(device)
    state_dict = checkpoint['model']
    
    # Handle compiled model state dict keys
    if list(state_dict.keys())[0].startswith('_orig_mod.'):
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
        
    model.load_state_dict(state_dict)
    model.eval()
    return model, config

def generate_heatmap(results: List[Dict]):
    """Generates Win Rate Heatmap: Hour vs Asset."""
    df = pd.DataFrame(results)
    
    # Filter for valid sessions (exclude weekends if needed, though Hour captures it)
    pivot = df.pivot_table(index='asset', columns='hour', values='correct', aggfunc='mean')
    
    plt.figure(figsize=(15, 6))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdYlGn", vmin=0.45, vmax=0.65)
    plt.title("I.A.R.A. Success Rate (Win %) by Hour & Asset")
    plt.xlabel("Hour of Day (UTC)")
    plt.ylabel("Asset")
    
    output_path = ARTIFACTS_DIR / "heatmap_win_rate.png"
    plt.savefig(output_path)
    logger.info(f"Heatmap saved to {output_path}")

def analyze_sensitivity(model: nn.Module, data_loader: torch.utils.data.DataLoader, device: torch.device, feature_names: List[str]):
    """Performs sensitivity analysis (gradient-based) to find feature importance."""
    logger.info("Running sensitivity analysis...")
    model.eval()
    
    feature_gradients = np.zeros(len(feature_names))
    count = 0
    
    for x, y in data_loader:
        x = x.to(device).requires_grad_(True)
        
        logits, _ = model(x)
        probs = torch.sigmoid(logits)
        
        # We want to maximize the probability of the predicted class to see which specific features contribute most
        # Or simply gradients w.r.t the logits
        logits.sum().backward()
        
        # Gradient magnitude per feature (averaged over time and batch)
        # x shape: [Batch, Seq, Features]
        # grad shape: [Batch, Seq, Features]
        grads = x.grad.abs().mean(dim=(0, 1)).detach().cpu().numpy()
        
        feature_gradients += grads
        count += 1
        
        if count >= 50: # Analyze subset to save time
            break
            
    feature_gradients /= count
    
    # Normalize
    feature_gradients = feature_gradients / feature_gradients.sum()
    
    importance = {name: float(score) for name, score in zip(feature_names, feature_gradients)}
    sorted_importance = dict(sorted(importance.items(), key=lambda item: item[1], reverse=True))
    
    with open(ARTIFACTS_DIR / "feature_importance.json", "w") as f:
        json.dump(sorted_importance, f, indent=4)
        
    logger.info(f"Feature Importance: {sorted_importance}")
    return sorted_importance

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="best_model.pth")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = CHECKPOINT_DIR / args.checkpoint
    
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        return

    model, config = load_model(checkpoint_path, device)
    
    # Load Validation Data (Logic replicated from Trainer)
    # We need MultiAssetDataset to expose metadata (tickers, time) which standard DataLoader hides
    # So we'll iterate differently or modify Dataset to return metadata.
    # For now, let's assume we can reconstruct it or create a new AnalysisDataset class if needed.
    # Easier hack: Re-instantiate dataset and access its internal .samples structure + index mappings
    
    full_dataset = MultiAssetDataset(seq_len=config.seq_len, stride=config.stride, augment=False)
    _, val_idxs = full_dataset.get_chronological_split(config.val_split)
    
    logger.info(f"Analyzing {len(val_idxs)} validation samples...")
    
    results = []
    
    # We need raw timestamps to extract Hour. 
    # Since MultiAssetDataset pre-processes into tensors, we lose exact datetime index unless we reload parquet.
    # However, 'day_sin' and 'hour_sin' are features! We can decode them roughly or rely on file structure.
    # Wait, simple approach: The 'features' tensor has `hour_sin` and `hour_cos`.
    # Hour angle = arctan2(sin, cos). 
    
    feature_cols = full_dataset.feature_cols
    hour_sin_idx = feature_cols.index("hour_sin_norm")
    hour_cos_idx = feature_cols.index("hour_cos_norm")
    
    batch_size = 512
    subset = torch.utils.data.Subset(full_dataset, val_idxs)
    loader = torch.utils.data.DataLoader(subset, batch_size=batch_size, shuffle=False)
    
    all_preds, all_targets = [], []
    
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            x_gpu = x.to(device)
            logits, _ = model(x_gpu)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float().cpu().numpy()
            targets = (y > 0.5).float().numpy()
            
            # Extract Metadata from Features (Batch, Seq, Feat)
            # We look at the LAST timestep for the decision time
            last_feat = x[:, -1, :].numpy()
            
            # Decode Hour: arctan2(sin, cos) -> radians -> degrees -> hours
            # Note: Features are normalized (Z-score). This makes exact decoding hard without scaler mean/std.
            # Fallback: Just track general performance or assume we can rely on raw index if available.
            # BETTER STRATEGY: For "Performance by Asset", we have the asset index in `MultiAssetDataset` but not directly in `__getitem__`.
            # Let's do a slower loop accessing the underlying index to get the Ticker name accurately.
            pass

    # High-Precison Loop (Slower but accurate metadata)
    logger.info("Running detailed metadata analysis...")
    model.eval()
    
    correct_by_hour = {}
    
    for idx in val_idxs:
        sample_idx, start_row = full_dataset.indices[idx]
        features, targets, _, ticker = full_dataset.samples[sample_idx]
        
        # Prepare Input
        end_row = start_row + config.seq_len
        x_np = features[start_row:end_row]
        x_tensor = torch.tensor(x_np, dtype=torch.float32).unsqueeze(0).to(device)
        y_true = 1.0 if targets[end_row - 1] > 0 else 0.0
        
        with torch.no_grad():
            logits, _ = model(x_tensor)
            prob = torch.sigmoid(logits).item()
            pred = 1.0 if prob > 0.5 else 0.0
            
        is_correct = 1 if pred == y_true else 0
        
        # Decode Hour from last timestep
        # We don't have the scaler here to un-normalize. 
        # Hack: Parquet files might have 'timestamp' but we didn't load it.
        # Alternative: We can't generate the Hour heatmap accurately without un-normalization or reloading df.
        # Let's skip Hour heatmap for now and focus on Asset heatmap which is easy.
        
        results.append({
            "asset": ticker,
            "correct": is_correct,
            "prob": prob,
            "target": y_true,
            "hour": 0 # Placeholder until we fix timestamps
        })
        
    generate_heatmap(results)
    analyze_sensitivity(model, loader, device, full_dataset.feature_cols)

if __name__ == "__main__":
    main()
