"""
Test Direction - Evaluate Saci-Nano on Trading Metrics
=======================================================
Calculates Directional Accuracy (Win Rate) and Magnitude Accuracy.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Tuple

import torch
import numpy as np
from tqdm import tqdm

from src.model.saci_nano import SaciNano, SaciNanoConfig
from src.data.tokenizer import FinancialTokenizer
from src.train.dataset import StructuredForexDataset
from src.train.trainer import TrainerConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Paths
CHECKPOINT_DIR = Path(__file__).parent.parent.parent / "checkpoints"
TOKENIZER_PATH = Path(__file__).parent.parent.parent / "data" / "tokenizer.pkl"
DATA_TOKENIZED_DIR = Path(__file__).parent.parent.parent / "data" / "tokenized"


def get_neutral_threshold(tokenizer: FinancialTokenizer, target_col: str) -> float:
    """
    Find the token ID that corresponds to 0.0 value for the target column.
    """
    if target_col not in tokenizer.feature_offsets:
        raise ValueError(f"Target column {target_col} not found in tokenizer offsets.")
        
    offset = tokenizer.feature_offsets[target_col]
    edges = tokenizer.bin_edges[target_col]
    
    # Find which bin contains 0.0
    # Edges: [e0, e1, ..., en]
    # Bin i is [edges[i], edges[i+1])
    # We want index i such that edges[i] <= 0 < edges[i+1]
    
    zero_bin_idx = -1
    
    # Check bounds
    if 0.0 < edges[0]:
        zero_bin_idx = 0 # All positive?
    elif 0.0 > edges[-1]:
        zero_bin_idx = len(edges) - 2 # All negative?
    else:
        # Standard search
        for i in range(len(edges) - 1):
            if edges[i] <= 0.0 <= edges[i+1]:
                zero_bin_idx = i
                break
    
    if zero_bin_idx == -1:
        # Fallback to median if 0 not found range
        zero_bin_idx = tokenizer.n_bins // 2
        logger.warning(f"Could not find 0.0 in {target_col} edges. Using median bin {zero_bin_idx}.")
    else:
        logger.info(f"Found neutral (0.0) at {target_col} bin index {zero_bin_idx}")
        
    return float(offset + zero_bin_idx)


def evaluate_direction(
    model: SaciNano,
    dataset: StructuredForexDataset,
    neutral_thresh: float,
    num_samples: int = 100,
    tolerance: int = 5,
    device: torch.device = torch.device("cpu")
) -> Dict[str, float]:
    """
    Evaluate model on directional and magnitude metrics.
    """
    model.eval()
    
    total_samples = 0
    correct_direction = 0
    correct_magnitude = 0
    
    # Randomly select indices if subset
    if num_samples < len(dataset):
        indices = np.random.choice(len(dataset), num_samples, replace=False)
    else:
        indices = range(len(dataset))
        
    logger.info(f"Evaluating on {len(indices)} samples...")
    
    for idx in tqdm(indices, desc="Evaluating"):
        inputs, targets = dataset[idx]
        
        inputs = inputs.unsqueeze(0).to(device)
        targets = targets.unsqueeze(0).to(device)
        
        with torch.no_grad():
            logits, _ = model(inputs, targets)
            predictions = logits.argmax(dim=-1)
        
        # Valid targets only (exclude -100 mask)
        valid_mask = (targets[0] != -100)
        
        if not valid_mask.any():
            continue
            
        preds = predictions[0][valid_mask].float()
        targs = targets[0][valid_mask].float()
        
        # 1. Magnitude Tolerance Accuracy
        # abs(Pred - True) <= tolerance
        diff = torch.abs(preds - targs)
        magnitude_hits = (diff <= tolerance).sum().item()
        
        # 2. Directional Accuracy (Win Rate)
        # (Pred > Neutral) == (True > Neutral)
        # Using a small epsilon for 0 case if needed, but 0.0 usually hits a bin.
        # The neutral_thresh is the ID of the bin containing 0.
        # If ID > neutral_thresh -> Positive return (roughly)
        # If ID < neutral_thresh -> Negative return (roughly)
        
        pred_dir = torch.sign(preds - neutral_thresh)
        targ_dir = torch.sign(targs - neutral_thresh)
        
        # We consider exact match of sign, including 0
        direction_hits = (pred_dir == targ_dir).sum().item()
        
        total_samples += len(preds)
        correct_magnitude += magnitude_hits
        correct_direction += direction_hits
        
    metrics = {
        "samples": total_samples,
        "magnitude_acc": (correct_magnitude / total_samples * 100) if total_samples > 0 else 0.0,
        "directional_acc": (correct_direction / total_samples * 100) if total_samples > 0 else 0.0
    }
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Saci-Nano Directional Test")
    parser.add_argument("--checkpoint", "-c", type=str, default="latest.pth", help="Checkpoint file")
    parser.add_argument("--limit", "-n", type=int, default=1000, help="Number of samples to test")
    parser.add_argument("--tolerance", "-t", type=int, default=5, help="Bin tolerance for magnitude")
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Tokenizer
    tokenizer = FinancialTokenizer.load(TOKENIZER_PATH)
    
    # Identify Target Column & Neutral Threshold
    # We assume dataset target is 'close_log_ret' usually
    target_col = "close_log_ret"
    neutral_thresh = get_neutral_threshold(tokenizer, target_col)
    
    print(f"Target Column: {target_col}")
    print(f"Neutral Threshold (ID): {neutral_thresh}")
    
    # Load Model
    checkpoint_path = CHECKPOINT_DIR / args.checkpoint
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        return
        
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_config = checkpoint['model_config']
    model = SaciNano(model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    # Load Dataset (Validation Mode)
    dataset = StructuredForexDataset(
        data_dir=DATA_TOKENIZED_DIR,
        seq_len=model.config.max_seq_len,
        is_training=False, # Deterministic
        target_column=target_col
    )
    
    # Evaluate
    metrics = evaluate_direction(
        model, 
        dataset, 
        neutral_thresh, 
        num_samples=args.limit, 
        tolerance=args.tolerance, 
        device=device
    )
    
    print("\n" + "="*50)
    print("SACI-NANO TRADING METRICS")
    print("="*50)
    print(f"Target:             {target_col}")
    print(f"Samples Tested:     {metrics['samples']}")
    print(f"Bin Tolerance:      +/- {args.tolerance}")
    print("-" * 50)
    print(f"Directional Acc (Win Rate): {metrics['directional_acc']:.2f}%")
    print(f"Magnitude Acc (Error <= {args.tolerance}):   {metrics['magnitude_acc']:.2f}%")
    print("="*50 + "\n")


if __name__ == "__main__":
    main()
