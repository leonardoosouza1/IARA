"""
Dynamic Calibrator (Trigger Optimizer)
======================================
Finds the "Golden Entry" thresholds and optimal position sizing.
- Threshold Sweep (Precision/Recall Trade-off)
- Expected Value (EV) Maximization
- Kelly Criterion Calculation
"""

import argparse
import logging
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score, accuracy_score

from src.train.trainer_liquid import TrainerConfig, MultiAssetDataset
from src.model.iara_liquid import IARA, LiquidConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Paths
CHECKPOINT_DIR = Path(__file__).parent.parent.parent / "checkpoints"
ARTIFACTS_DIR = Path(__file__).parent.parent.parent / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

def load_model(checkpoint_path: Path, device: torch.device) -> Tuple[nn.Module, TrainerConfig]:
    """Loads the best model from checkpoint."""
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
    
    if list(state_dict.keys())[0].startswith('_orig_mod.'):
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
        
    model.load_state_dict(state_dict)
    model.eval()
    return model, config

def calculate_kelly(win_rate: float, win_loss_ratio: float) -> float:
    """Calculates Kelly Criterion fraction."""
    # K% = W - (1-W)/R
    if win_loss_ratio == 0: return 0.0
    k = win_rate - (1 - win_rate) / win_loss_ratio
    return max(0.0, k)  # No negative betting

def run_calibration(model: nn.Module, loader: torch.utils.data.DataLoader, device: torch.device):
    """Sweeps thresholds to optimize EV and Kelly."""
    logger.info("Running calibration sweep...")
    
    all_probs = []
    all_targets = []
    
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits, _ = model(x)
            probs = torch.sigmoid(logits).cpu().numpy()
            targets = (y > 0.5).float().numpy()
            
            all_probs.extend(probs)
            all_targets.extend(targets)
            
    all_probs = np.array(all_probs)
    all_targets = np.array(all_targets)
    
    # Assume generic Risk:Reward of 1:1.5 for optimization context (or derive from data if we had returns)
    # In binary classification, we don't know the exact magnitude of the move without looking at raw prices.
    # We will assume a fixed R:R for calibration purposes.
    AVG_RR = 1.5 
    
    best_ev = -np.inf
    best_threshold = 0.5
    results = {}
    
    # Sweep thresholds
    thresholds = np.linspace(0.50, 0.80, 31) # Only look at Long confidence for now (symmetric for Short)
    
    print(f"{'Threshold':<10} | {'Win%':<8} | {'Trades':<8} | {'EV':<8} | {'Kelly%':<8}")
    print("-" * 55)
    
    for thresh in thresholds:
        # Long Logic: Prob > Thresh
        # Short Logic: Prob < (1 - Thresh)
        
        # Simplify: Evaluate purely on "High Confidence Correctness"
        # We classify a "Trade" as any prediction with confidence > thresh or < (1-thresh)
        
        long_mask = all_probs > thresh
        short_mask = all_probs < (1 - thresh)
        
        total_trades_mask = long_mask | short_mask
        n_trades = np.sum(total_trades_mask)
        
        if n_trades < 50: # Ignore outlier thresholds with too few samples
            continue
            
        # Check correctness on these trades
        # For Longs: Target should be 1
        # For Shorts: Target should be 0
        
        long_wins = (all_targets[long_mask] == 1).sum()
        short_wins = (all_targets[short_mask] == 0).sum()
        
        total_wins = long_wins + short_wins
        win_rate = total_wins / n_trades
        
        # EV = (Win% * 1.5) - (Loss% * 1.0)
        ev = (win_rate * AVG_RR) - ((1 - win_rate) * 1.0)
        
        kelly = calculate_kelly(win_rate, AVG_RR)
        
        print(f"{thresh:.2f}       | {win_rate:.2%}   | {n_trades:<8} | {ev:.4f}   | {kelly:.2%}")
        
        if ev > best_ev:
            best_ev = ev
            best_threshold = thresh
            results = {
                "threshold": float(thresh),
                "win_rate": float(win_rate),
                "ev": float(ev),
                "kelly": float(kelly),
                "n_trades": int(n_trades)
            }
            
    logger.info(f"Calibration Complete. Best Threshold: {best_threshold:.2f} (EV: {best_ev:.4f})")
    
    with open(ARTIFACTS_DIR / "calibration.json", "w") as f:
        json.dump(results, f, indent=4)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = CHECKPOINT_DIR / "best_model.pth"
    
    if not checkpoint_path.exists():
        logger.error("Checkpoint not found.")
        return
        
    model, config = load_model(checkpoint_path, device)
    
    full_dataset = MultiAssetDataset(seq_len=config.seq_len, stride=config.stride, augment=False)
    _, val_idxs = full_dataset.get_chronological_split(config.val_split)
    
    subset = torch.utils.data.Subset(full_dataset, val_idxs)
    loader = torch.utils.data.DataLoader(subset, batch_size=1024, shuffle=False)
    
    run_calibration(model, loader, device)

if __name__ == "__main__":
    main()
