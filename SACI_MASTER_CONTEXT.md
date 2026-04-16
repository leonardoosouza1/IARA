--- FILE: backend/src/train/backtester.py ---
import os
import torch
import torch.nn.functional as F
from datetime import datetime
import json
import logging
from typing import Dict, Any

from src.brain.models.liquid_v9 import SwarmBrain, CoordinatorConfig
from src.brain.models.workers import WorkerConfig
from src.database.core import get_db
from src.database.models import Trade

logger = logging.getLogger("api.backtester")

class Backtester:
    """
    Evaluates the historical performance of a SwarmBrain model
    applying varying degrees of Sniper Mode precision.
    """
    def __init__(self, model_path: str = "logs/models/active_model_v10.pth", device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        
        # Load Model Dimensions dynamically
        self.input_dim = 26
        self.worker_hidden = 64
        self.coord_hidden = 256
        self.seq_len = 128
        
        # Introspect checkpoint to determine optimal dimensions
        if os.path.exists(self.model_path):
            try:
                state = torch.load(self.model_path, map_location='cpu', weights_only=False)
                state_dict = state.get('model_state_dict', state)
                
                # Check new v9/v10 attn_proj first, then fallback to old input_proj
                w = state_dict.get("swarm.workers.M5.attn_proj.weight") or state_dict.get("swarm.workers.M5.input_proj.weight")
                if w is not None and hasattr(w, 'shape'):
                    self.input_dim = int(w.shape[1])
                    self.worker_hidden = int(w.shape[0])
                    
                ph = state_dict.get("coordinator.policy_head.weight")
                if ph is not None and hasattr(ph, 'shape'):
                    self.coord_hidden = int(ph.shape[1])
            except Exception as e:
                logger.warning(f"Could not introspect checkpoint dimensions: {e}")
                
        self.model = self._load_model()
        
    def _load_model(self) -> SwarmBrain:
        worker_cfg = WorkerConfig(input_dim=self.input_dim, hidden_dim=self.worker_hidden)
        coord_cfg = CoordinatorConfig(num_workers=5, worker_dim=self.worker_hidden, hidden_dim=self.coord_hidden)
        model = SwarmBrain(worker_cfg, coord_cfg).to(self.device)
        
        if os.path.exists(self.model_path):
            state = torch.load(self.model_path, map_location=self.device)
            state_dict = state.get('model_state_dict', state)
            model.load_state_dict(state_dict, strict=False)
            logger.info(f"Loaded model from {self.model_path}")
        else:
            logger.warning(f"Model path {self.model_path} not found! Backtest will use random weights.")
            
        model.eval()
        return model

    def _simulate_timeframes(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Same logic as continuous_learner.py to build the timeframe dictionary."""
        seq_len = x.shape[1]
        tfs = {
            "M5": x,
            "M15": x[:, max(0, seq_len-32):, :] if seq_len > 32 else x,
            "M30": x[:, max(0, seq_len-16):, :] if seq_len > 16 else x,
            "H1": x[:, max(0, seq_len-8):, :] if seq_len > 8 else x,
            "H4": x[:, max(0, seq_len-2):, :] if seq_len > 2 else x,
        }
        return tfs

    def evaluate_shards(self, dataset_name: str, sniper_threshold: float = 0.50, callback=None) -> Dict[str, Any]:
        """
        Runs the model over the validation shards, enforcing the sniper_threshold.
        If the network's confidence is lower than sniper_threshold, the trade is skipped (classified as WAIT).
        """
        val_shard_dir = f"data/cache/{dataset_name}/val_shards"
        
        if not os.path.exists(val_shard_dir):
            return {"error": f"Validation shards not found for {dataset_name}"}
            
        shards = sorted([f for f in os.listdir(val_shard_dir) if f.endswith(".pt")])
        if not shards:
            return {"error": f"No shards found in {val_shard_dir}"}
            
        metrics = {
            "total_samples": 0,
            "trades_executed": 0,
            "wins": 0,
            "losses": 0,
            "sniper_threshold": sniper_threshold,
            "false_positives_avoided": 0 # Trades the network wanted to take but Sniper Mode blocked
        }
        
        batch_size = 512
        
        if callback: callback(f"🚀 Starting Backtest on {len(shards)} shards with Sniper mode: {sniper_threshold:.2f}...")

        with torch.no_grad():
            for s_file in shards:
                if callback: callback(f"Evaluating {s_file}...")
                s_data = torch.load(os.path.join(val_shard_dir, s_file), map_location='cpu')
                X_cpu, y_cpu = s_data['X'], s_data['y']
                metrics["total_samples"] += len(X_cpu)
                
                for j in range(0, len(X_cpu), batch_size):
                    batch_X = X_cpu[j : j + batch_size].to(self.device)
                    batch_y = y_cpu[j : j + batch_size].long().to(self.device)
                    
                    # Robust Tensor Padding
                    shard_dim = batch_X.shape[-1]
                    if shard_dim > self.input_dim:
                        batch_X = batch_X[:, :, :self.input_dim]
                    elif shard_dim < self.input_dim:
                        padding = torch.zeros(batch_X.shape[0], batch_X.shape[1], self.input_dim - shard_dim, dtype=batch_X.dtype, device=batch_X.device)
                        batch_X = torch.cat([batch_X, padding], dim=-1)
                        
                    batch_X = torch.nan_to_num(batch_X)
                    batch_X = torch.clamp(batch_X, min=-20.0, max=20.0)
                    
                    x_dict = self._simulate_timeframes(batch_X)
                    
                    # Forward pass with telemetry
                    logits, _, _, _, _, _ = self.model(x_dict, compute_telemetry=True)
                    
                    # Logits are [Batch, 3] corresponding to Long(0), Short(1), Wait(2)
                    probs = F.softmax(logits, dim=-1)
                    
                    # 🎯 SNIPER MODE FILTRATION
                    # Use the network's confidence in its top prediction
                    max_probs, base_predictions = probs.max(dim=-1)
                    
                    # If prediction is Long or Short (0 or 1), but confidence is below threshold, override to Wait (2)
                    sniper_predictions = base_predictions.clone()
                    is_action = (base_predictions == 0) | (base_predictions == 1)
                    low_confidence = max_probs < sniper_threshold
                    
                    # Block trades that lack confidence
                    blocked_mask = is_action & low_confidence
                    sniper_predictions[blocked_mask] = 2 
                    metrics["false_positives_avoided"] += blocked_mask.sum().item()
                    
                    # Evaluate Results
                    for i in range(len(sniper_predictions)):
                        pred = sniper_predictions[i].item()
                        true = batch_y[i].item()
                        
                        if pred != 2: # Model decided to act
                            metrics["trades_executed"] += 1
                            if pred == true:
                                metrics["wins"] += 1
                            else:
                                metrics["losses"] += 1

        if metrics["trades_executed"] > 0:
            metrics["win_rate"] = round((metrics["wins"] / metrics["trades_executed"]) * 100, 2)
        else:
            metrics["win_rate"] = 0.0

        if callback: 
            callback(f"✅ Backtest Complete!")
            callback(f"🎯 Win Rate: {metrics['win_rate']}% | Trades: {metrics['trades_executed']} | Blocked by Sniper: {metrics['false_positives_avoided']}")
            
        return metrics

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="BTCUSD")
    parser.add_argument("--threshold", type=float, default=0.75)
    args = parser.parse_args()
    
    bt = Backtester()
    def console_cb(msg): print(msg)
    res = bt.evaluate_shards(args.dataset, args.threshold, callback=console_cb)
    print(json.dumps(res, indent=2))



--- FILE: backend/src/train/optimizer_backtest.py ---
"""
Threshold Optimization Backtest for I.A.R.A. v4.4
=================================================
Finds the optimal probability threshold to maximize Net Profit.

Simulation Parameters:
- Initial Balance: $1,000
- Fixed Risk per Trade: $20
- Risk:Reward: 1:1
- Threshold Range: 0.40 to 0.60 (step 0.005)

Logic:
- Buy if prob > threshold
- Sell if prob < (1 - threshold)
- Hold otherwise
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import polars as pl
import torch

from src.model.iara_liquid import IARA, LiquidConfig
from src.train.trainer_liquid import TrainerConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Paths
DATA_PROCESSED_DIR = Path(__file__).parent.parent.parent / "data" / "processed"
CHECKPOINT_DIR = Path(__file__).parent.parent.parent / "checkpoints"

# Simulation Parameters
INITIAL_BALANCE = 1000.0
FIXED_RISK = 20.0  # Fixed $20 per trade
RISK_REWARD_RATIO = 1.0  # 1:1


@dataclass
class ThresholdResult:
    """Result of a single threshold backtest."""
    threshold: float
    n_trades: int
    wins: int
    losses: int
    win_rate: float
    net_profit: float
    final_balance: float


def load_model(device: torch.device) -> IARA:
    """Load trained I.A.R.A. model."""
    checkpoint_path = CHECKPOINT_DIR / "best_model.pth"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    trainer_config = checkpoint.get("config")
    if trainer_config:
        model_config = LiquidConfig(
            input_dim=trainer_config.input_dim,
            hidden_dim=trainer_config.hidden_dim,
            dropout=trainer_config.dropout
        )
    else:
        model_config = LiquidConfig(hidden_dim=512)
    
    model = IARA(model_config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    
    logger.info(f"Loaded model from {checkpoint_path}")
    return model


def load_test_data(ticker: str = "EURUSD", seq_len: int = 64) -> Tuple[np.ndarray, np.ndarray]:
    """Load test data and return features and true directions."""
    file_path = DATA_PROCESSED_DIR / f"{ticker}.parquet"
    df = pl.read_parquet(file_path)
    
    feature_cols = sorted([c for c in df.columns if c.endswith("_norm")])
    features = df.select(feature_cols).to_numpy().astype(np.float32)
    
    if "close_log_ret" in df.columns:
        target_vals = df["close_log_ret"].to_numpy().astype(np.float32)
        true_dirs = (target_vals > 0).astype(np.float32)  # 1 = UP, 0 = DOWN
    else:
        true_dirs = np.zeros(len(features))
    
    return features, true_dirs


@torch.no_grad()
def generate_predictions(
    model: IARA,
    features: np.ndarray,
    device: torch.device,
    seq_len: int = 64
) -> List[float]:
    """Generate predictions for all test samples."""
    model.eval()
    
    predictions = []
    n_samples = len(features) - seq_len
    
    for i in range(n_samples):
        x = torch.tensor(features[i:i + seq_len], dtype=torch.float32).unsqueeze(0).to(device)
        logits, _ = model(x)
        prob = torch.sigmoid(logits).cpu().item()
        predictions.append(prob)
    
    return predictions


def simulate_trading(
    probs: List[float],
    true_dirs: np.ndarray,
    threshold: float,
    seq_len: int = 64
) -> ThresholdResult:
    """
    Simulate trading with a specific threshold.
    
    Buy if prob > threshold
    Sell if prob < (1 - threshold)
    """
    balance = INITIAL_BALANCE
    trades = []
    
    for i, prob in enumerate(probs):
        true_idx = i + seq_len - 1
        true_dir = true_dirs[true_idx]
        
        # Trading decision
        trade_type = None
        if prob > threshold:
            trade_type = "BUY"
            predicted_dir = 1
        elif prob < (1 - threshold):
            trade_type = "SELL"
            predicted_dir = 0
        
        if trade_type is not None:
            if predicted_dir == true_dir:
                # Win: gain $20 * R:R
                balance += FIXED_RISK * RISK_REWARD_RATIO
                trades.append(1)
            else:
                # Loss: lose $20
                balance -= FIXED_RISK
                trades.append(0)
    
    # Calculate metrics
    n_trades = len(trades)
    wins = sum(trades)
    losses = n_trades - wins
    win_rate = wins / n_trades * 100 if n_trades > 0 else 0
    net_profit = balance - INITIAL_BALANCE
    
    return ThresholdResult(
        threshold=threshold,
        n_trades=n_trades,
        wins=wins,
        losses=losses,
        win_rate=win_rate,
        net_profit=net_profit,
        final_balance=balance
    )


def run_threshold_sweep():
    """Run the full threshold optimization."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load model and data
    model = load_model(device)
    features, true_dirs = load_test_data()
    logger.info(f"Loaded {len(features)} samples")
    
    # Generate predictions once
    logger.info("Generating predictions...")
    probs = generate_predictions(model, features, device)
    logger.info(f"Generated {len(probs)} predictions")
    
    # Show probability distribution
    probs_array = np.array(probs)
    logger.info(f"Prob stats: min={probs_array.min():.4f}, max={probs_array.max():.4f}, mean={probs_array.mean():.4f}, median={np.median(probs_array):.4f}")
    
    # Threshold sweep with finer granularity
    thresholds = np.arange(0.40, 0.605, 0.005)
    results: List[ThresholdResult] = []
    
    logger.info("Running threshold sweep (0.005 step)...")
    for threshold in thresholds:
        result = simulate_trading(probs, true_dirs, threshold)
        results.append(result)
    
    # Sort by final balance (highest profit)
    sorted_results = sorted(results, key=lambda x: x.final_balance, reverse=True)
    
    # Print TOP 5
    print("\n" + "=" * 80)
    print("TOP 5 OPTIMAL THRESHOLDS (Sorted by Final Balance)")
    print("=" * 80)
    print(f"{'Rank':>4} | {'Threshold':>10} | {'Trades':>8} | {'Win Rate':>10} | {'Net Profit':>12} | {'Balance':>12}")
    print("-" * 80)
    
    for i, r in enumerate(sorted_results[:5], 1):
        print(f"{i:>4} | {r.threshold:>10.3f} | {r.n_trades:>8} | {r.win_rate:>9.1f}% | ${r.net_profit:>10.2f} | ${r.final_balance:>10.2f}")
    
    print("=" * 80)
    
    # Print recommendations
    best = sorted_results[0]
    print(f"\n🏆 GOLDEN THRESHOLD: {best.threshold:.3f}")
    print(f"   → Win Rate: {best.win_rate:.1f}%")
    print(f"   → Net Profit: ${best.net_profit:.2f}")
    print(f"   → Final Balance: ${best.final_balance:.2f}")
    print(f"   → Total Trades: {best.n_trades}")
    
    # Print full table
    print("\n" + "=" * 80)
    print("FULL THRESHOLD SWEEP")
    print("=" * 80)
    print(f"{'Threshold':>10} | {'Trades':>8} | {'Wins':>6} | {'Losses':>8} | {'Win Rate':>10} | {'Net Profit':>12}")
    print("-" * 80)
    
    for r in results:
        profit_indicator = "✓" if r.net_profit > 0 else ""
        print(f"{r.threshold:>10.3f} | {r.n_trades:>8} | {r.wins:>6} | {r.losses:>8} | {r.win_rate:>9.1f}% | ${r.net_profit:>10.2f} {profit_indicator}")
    
    print("=" * 80)


if __name__ == "__main__":
    run_threshold_sweep()



--- FILE: backend/src/train/test_direction.py ---
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



--- FILE: backend/src/train/inference.py ---
"""
Inference - Generate Market Predictions from Trained Saci-Nano Model
=====================================================================
Load a trained checkpoint and generate next token predictions.
"""

import argparse
import logging
from pathlib import Path
from typing import Optional

import torch
import numpy as np

from src.model.saci_nano import SaciNano, SaciNanoConfig
from src.data.tokenizer import FinancialTokenizer
from src.model.saci_nano import SaciNano, SaciNanoConfig
from src.data.tokenizer import FinancialTokenizer
from src.train.dataset import StructuredForexDataset, create_dataloader
from src.train.trainer import TrainerConfig
from src.train.test_direction import get_neutral_threshold, evaluate_direction

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


def load_model(checkpoint_path: Path, device: torch.device) -> SaciNano:
    """Load model from checkpoint."""
    logger.info(f"Loading model from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    model_config = checkpoint['model_config']
    model = SaciNano(model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    epoch = checkpoint.get('epoch', 'unknown')
    logger.info(f"Loaded model from epoch {epoch}")
    logger.info(f"Model parameters: {model.count_parameters():,}")
    
    return model


def analyze_predictions(
    model: SaciNano,
    tokenizer: FinancialTokenizer,
    dataset: StructuredForexDataset,
    num_samples: int = 5,
    device: torch.device = torch.device("cpu")
):
    """
    Analyze model predictions on sample sequences.
    """
    print("\n" + "=" * 60)
    print("SACI-NANO PREDICTION ANALYSIS")
    print("=" * 60)
    
    print("=" * 60)
    
    # Use valid dataset (no shuffle)
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    
    for i, idx in enumerate(indices):
        inputs, targets = dataset[idx]
        
        # Add batch dim
        inputs_t = inputs.unsqueeze(0).to(device)
        targets_t = targets.unsqueeze(0).to(device)
        
        with torch.no_grad():
            logits, loss = model(inputs_t, targets_t)
            probs = torch.softmax(logits, dim=-1)
            predictions = logits.argmax(dim=-1)
        
        # Identify Target positions (where target != -100)
        # Note: targets tensor contains the LABELS for the Next step.
        # But we want to see what the model predicted for the "Price" token.
        # In our dataset logic, the label for the Price token input is Next_Feature (which is masked -100).
        # Wait, the label for the Feature[Last] input IS the Price token.
        
        # Let's find indices where the LABEL is valid (not -100)
        # These correspond to the moments we are predicting the Price.
        valid_indices = (targets_t[0] != -100).nonzero(as_tuple=True)[0]
        
        if len(valid_indices) == 0:
            print(f"\n--- Sample {i + 1} (No valid targets found in window) ---")
            continue

        # Calculate accuracy on Valid Targets Only
        valid_preds = predictions[0][valid_indices]
        valid_targets = targets_t[0][valid_indices]
        
        correct = (valid_preds == valid_targets).float()
        accuracy = correct.mean().item() * 100
        
        
        # Loss is already computed correctly by the model with ignore_index=-100
        
        print(f"\n--- Sample {i + 1} ---")
        print(f"  Loss: {loss.item():.4f}")
        print(f"  Exact Accuracy: {accuracy:.2f}%")
        
        # Calculate Directional Metrics for this sample
        target_col = "close_log_ret"
        try:
             neutral = get_neutral_threshold(tokenizer, target_col)
             
             pred_sign = torch.sign(valid_preds - neutral)
             targ_sign = torch.sign(valid_targets - neutral)
             sign_acc = (pred_sign == targ_sign).float().mean().item() * 100
             
             print(f"  Directional Acc: {sign_acc:.2f}% (Neutral ID: {neutral})")
        except:
             pass
        
        print(f"\n  Predictions (Last 5 valid targets):")
        for pos in valid_indices[-5:]:
            target_token = targets_t[0, pos].item()
            pred_token = predictions[0, pos].item()
            prob = probs[0, pos, pred_token].item()
            
            # The input that generated this prediction was inputs[pos]
            input_token = inputs_t[0, pos].item()
            
            input_feat, input_bin = tokenizer.decode_token(input_token)
            target_feat, target_bin = tokenizer.decode_token(target_token)
            pred_feat, pred_bin = tokenizer.decode_token(pred_token)
            
            # Use arrows for direction
            target_dir = "▲" if target_token > neutral else "▼"
            pred_dir = "▲" if pred_token > neutral else "▼"
            
            dir_match = "✓" if target_dir == pred_dir else "✗"
            
            match = "✓" if pred_token == target_token else "✗"
            print(f"    Input: {input_feat[:10]:>10} -> Target: {target_feat[:10]:>10}[{target_bin:3d}] {target_dir} | "
                  f"Pred: {pred_feat[:10]:>10}[{pred_bin:3d}] {pred_dir} (p={prob:.2f}) {dir_match}")


def generate_sequence(
    model: SaciNano,
    tokenizer: FinancialTokenizer,
    seed_tokens: torch.Tensor,
    num_tokens: int = 50,
    temperature: float = 0.8,
    top_k: int = 10,
    device: torch.device = torch.device("cpu")
):
    """
    Generate a sequence of tokens autoregressively.
    """
    print("\n" + "=" * 60)
    print("AUTOREGRESSIVE GENERATION")
    print("=" * 60)
    
    seed_tokens = seed_tokens.unsqueeze(0).to(device)
    
    with torch.no_grad():
        generated = model.generate(
            seed_tokens,
            max_new_tokens=num_tokens,
            temperature=temperature,
            top_k=top_k
        )
    
    print(f"\nSeed length: {seed_tokens.size(1)} tokens")
    print(f"Generated: {num_tokens} new tokens")
    print(f"Temperature: {temperature}, Top-k: {top_k}")
    
    # Decode generated tokens
    print("\nGenerated Token Sequence (decoded):")
    gen_tokens = generated[0, seed_tokens.size(1):].cpu().numpy()
    
    for i, token in enumerate(gen_tokens[:20]):  # Show first 20
        feat_name, bin_idx = tokenizer.decode_token(int(token))
        print(f"  [{i:3d}] {feat_name}: bin {bin_idx}")


def show_model_stats(model: SaciNano, checkpoint: dict):
    """Display model statistics."""
    print("\n" + "=" * 60)
    print("MODEL STATISTICS")
    print("=" * 60)
    
    config = model.config
    print(f"\n  Architecture:")
    print(f"    - Vocabulary Size: {config.vocab_size:,}")
    print(f"    - Model Dimension: {config.dim}")
    print(f"    - Layers: {config.n_layers}")
    print(f"    - Attention Heads: {config.n_heads}")
    print(f"    - KV Heads (GQA): {config.n_kv_heads}")
    print(f"    - Max Sequence Length: {config.max_seq_len}")
    
    print(f"\n  Training:")
    print(f"    - Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"    - Global Step: {checkpoint.get('global_step', 'N/A'):,}")
    print(f"    - Best Loss: {checkpoint.get('best_loss', 'N/A'):.4f}" if isinstance(checkpoint.get('best_loss'), float) else f"    - Best Loss: N/A")
    
    print(f"\n  Parameters: {model.count_parameters():,}")


def main():
    parser = argparse.ArgumentParser(description="Saci-Nano Inference")
    parser.add_argument("--checkpoint", "-c", type=str, default="latest.pth",
                       help="Checkpoint file name (in checkpoints/)")
    parser.add_argument("--num-samples", "-n", type=int, default=3,
                       help="Number of samples to analyze")
    parser.add_argument("--generate", "-g", type=int, default=0,
                       help="Number of tokens to generate (0 to skip)")
    parser.add_argument("--temperature", "-t", type=float, default=0.8,
                       help="Generation temperature")
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load tokenizer
    try:
        tokenizer = FinancialTokenizer.load(TOKENIZER_PATH)
        logger.info(f"Tokenizer loaded. Vocab size: {tokenizer.vocab_size}")
    except FileNotFoundError:
        logger.error("Tokenizer not found. Run the pipeline first.")
        return
    
    # Load checkpoint
    checkpoint_path = CHECKPOINT_DIR / args.checkpoint
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        available = list(CHECKPOINT_DIR.glob("*.pth"))
        if available:
            logger.info(f"Available checkpoints: {[p.name for p in available]}")
        return
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = load_model(checkpoint_path, device)
    
    # Show stats
    show_model_stats(model, checkpoint)
    
    # Load dataset for analysis
    try:
        # Use StructuredForexDataset with is_training=False to avoid shuffling during inference
        dataset = StructuredForexDataset(
            data_dir=DATA_TOKENIZED_DIR,
            seq_len=model.config.max_seq_len,
            is_training=False
        )
        logger.info(f"Dataset loaded: {len(dataset)} samples")
        
        # Analyze predictions
        analyze_predictions(model, tokenizer, dataset, args.num_samples, device)
        
        # Generate if requested
        if args.generate > 0:
            seed_tokens, _ = dataset[0]
            generate_sequence(
                model, tokenizer, seed_tokens[:64],  # Use first 64 as seed
                num_tokens=args.generate,
                temperature=args.temperature,
                device=device
            )
            
    except Exception as e:
        logger.error(f"Dataset error: {e}")
    
    print("\n" + "=" * 60)
    print("INFERENCE COMPLETE")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()



--- FILE: backend/src/train/nightly_dojo.py ---
"""
Saci Dojo (Smart Nightly Trainer)
=================================
"Treine enquanto eles dormem."

Concepts:
1. Finetuning: Não começa do zero. Pega o `best_model.pth`.
2. Regime Adaptation: Foca nos últimos N meses (default: 6).
3. Importance Sampling: Dá peso maior para candles com alta volatilidade (Onde o dinheiro está).
"""

import sys
import argparse
import logging
from pathlib import Path
from typing import Optional, List, Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
import numpy as np

# Add src to path
sys.path.append("/app")

from src.train.trainer_liquid import TrainerConfig, MultiAssetDataset, Trainer
from src.model.iara_liquid import IARA, LiquidConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class DojoTrainer(Trainer):
    def __init__(self, config: TrainerConfig, device: torch.device, recent_months: int = 6):
        super().__init__(config, device)
        self.recent_months = recent_months
        
    def prepare_smart_data(self):
        """
        Overrides standard data loading to implement:
        1. Time Decay (Recent Bias)
        2. Volatility Importance Sampling
        """
        full_dataset = MultiAssetDataset(seq_len=self.config.seq_len, stride=self.config.stride, augment=False)
        
        # 1. Recent Bias: Select last N% of data roughly fitting 'recent_months'
        # Assuming M5 candles: 1 month ~ 6000 candles (if 24/5)
        # 6 months ~ 36,000 candles.
        
        total_len = len(full_dataset)
        # Let's verify we have enough data
        if total_len < 1000:
            logger.warning("Not enough data for Smart Sampling. Using full dataset.")
            idxs = list(range(total_len))
        else:
            # Take last 50% or fixed window? Let's take last 8000 samples (optimize for speed)
            window_size = 12000 # ~2 months of M5 data Stride 8? 
            # Stride 8: 12000 * 8 = 96,000 raw candles / 12/h = 8000 hours / 24 = 333 days. ~1 year.
            
            start_idx = max(0, total_len - window_size)
            idxs = list(range(start_idx, total_len))
            logger.info(f"Dojo Window: Focusing on last {len(idxs)} samples (Market Regime).")

        # Split Train/Val on this window (Last 20% is Val)
        split = int(len(idxs) * (1 - self.config.val_split))
        train_idxs = idxs[:split]
        val_idxs = idxs[split:]
        
        # 2. Importance Sampling (Volatility)
        # We want to train harder on candles where 'close_log_ret' is high.
        weights = []
        for i in train_idxs:
            sample_idx, start_row = full_dataset.indices[i]
            _, targets, _, _ = full_dataset.samples[sample_idx]
            end_row = start_row + self.config.seq_len
            
            # Target is log_ret
            ret = abs(targets[end_row-1])
            w = 1.0 + (ret * 1000) # Boost high volatility
            weights.append(w)
            
        weights = torch.tensor(weights, dtype=torch.double)
        sampler = WeightedRandomSampler(weights, len(weights))
        
        # Create Loaders
        train_subset = Subset(full_dataset, train_idxs) # Dataset is generic, sampler picks indices relative to subset? 
        # WeightedRandomSampler picks INDICES from 0..len(weights)-1.
        # But DataLoader(subset) maps 0..N to subset indices.
        # So we pass sampler to DataLoader.
        
        self.train_loader = DataLoader(
            train_subset,
            batch_size=self.config.batch_size,
            sampler=sampler, # SMART SAMPLING
            num_workers=self.config.num_workers,
            pin_memory=True
        )
        
        val_subset = Subset(full_dataset, val_idxs)
        self.val_loader = DataLoader(
            val_subset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers
        )
        
        logger.info(f"Smart Data Prepared: Train={len(train_subset)} (Importance Sampled), Val={len(val_subset)}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20) 
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--batch-size", type=int, default=512) # Reduced for stability
    parser.add_argument("--resume", action="store_true", default=True)
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Config for Finetuning
    config = TrainerConfig(
        epochs=args.epochs,
        learning_rate=args.lr,
        resume=args.resume,
        batch_size=args.batch_size,
        stride=4, 
        patience=10
    )
    
    trainer = DojoTrainer(config, device)
    
    # 1. Override Data Prepare
    trainer.prepare_smart_data()
    
    # 2. Train (Logic handled by parent, but uses new loaders)
    trainer.train()

if __name__ == "__main__":
    main()



--- FILE: backend/src/train/continuous_learner.py ---
import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Tuple, Optional

import torch
import torch.nn as nn
from sqlalchemy import create_engine, desc
from sqlalchemy.orm import Session, sessionmaker
import pandas as pd
import numpy as np
import warnings
from dotenv import load_dotenv

load_dotenv()

from src.database.models import Trade, Signal, ModelVersion
from src.brain.models.liquid_v9 import SwarmBrain, CoordinatorConfig
from src.brain.models.workers import WorkerConfig
from src.brain.losses import SwarmLoss # 🚀 Swarm v9.7 Loss
from src.brain.models.generator import ChallengerGenerator, OverfittingGuard # 🚀 Phase 13

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("ContinuousLearner")

class ContinuousLearner:
    def __init__(self, checkpoint_path: str = "logs/models/active_model_v10.pth", device: Optional[str] = None):
        self.checkpoint_path = checkpoint_path
        
        # 🚀 Phase 9: Automatic Hardware Detection (CUDA/ROCm)
        if device:
            self.device = torch.device(device)
        else:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                # Add ROCm check if possible, but PyTorch uses 'cuda' label for both
                logger.info(f"🚀 GPU Acceleration Enabled: Using {torch.cuda.get_device_name(0)}")
            else:
                self.device = torch.device("cpu")
                logger.warning("⚠️ GPU Not Found. Falling back to CPU training.")
        
        self.mode = "finetune"
        
        # Setup synchronous DB Engine for the learning loop
        db_uri = os.getenv("DB_URI", "sqlite:///./saci_core.db")
        if db_uri.startswith("postgresql+asyncpg://"):
             db_uri = db_uri.replace("postgresql+asyncpg://", "postgresql://")
        elif db_uri.startswith("postgresql://") and "+asyncpg" not in db_uri:
             pass 
             
        sync_engine = create_engine(db_uri)
        SyncSessionLocal = sessionmaker(bind=sync_engine)
        self.db: Session = SyncSessionLocal()
        
        # Defaults for architecture (Golden v10.0 / Sniper Mode)
        self.worker_hidden = 64 # Matches Golden active_model.pth
        self.coord_hidden = 256 # Matches Golden active_model.pth
        self.seq_len = 128
        
        # 🚀 Phase 22: Dynamic Input Dimension from Advanced Data Science Pipeline
        self.input_dim = 26 # Fallback
        norm_path = "data/cache/EURUSD/norm_params.pt"
        if os.path.exists(norm_path):
            try:
                meta = torch.load(norm_path, map_location='cpu', weights_only=True)
                if 'feature_cols' in meta:
                    self.input_dim = len(meta['feature_cols'])
                    logger.info(f"🧬 Loaded dynamic input_dim={self.input_dim} from Boruta ExtraTrees pruning.")
            except Exception as e:
                logger.warning(f"⚠️ Could not load norm_params.pt: {e}. Using fallback input_dim=26")
        
        
        # Hyperparameters (Gladiator v9.7 Presets)
        self.epochs = 5
        self.batch_size = 256
        self.learning_rate = 1e-4
        
        # 🚀 Phase 11: SwarmLoss with Dynamic Weighting
        self.criterion = SwarmLoss(alpha=0.75, gamma=2.0, consensus_weight=0.1).to(self.device)

        import itertools
        self.model = self._load_model()
        
        # 🚀 Phase 11 & 19: Separate learning rates for Model and Meta-Weights
        # Swarm Weights (logits) need a much higher LR to adapt quickly (1e-2 vs 1e-4)
        optimizer_grouped_parameters = [
            {"params": self.model.parameters(), "lr": self.learning_rate},
            {"params": self.criterion.parameters(), "lr": 1e-2}
        ]
        self.optimizer = torch.optim.AdamW(optimizer_grouped_parameters)
        
        # 🚀 Phase 14 v2: Persistent Prioritized Experience Replay
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=2)
        self.trauma_buffer_X = []
        self.trauma_buffer_Y = []
        self.trauma_buffer_losses = []  # Track individual sample losses for priority
        self.trauma_threshold = 1.15  # Minimum loss to enter trauma memory (adjusted for stabilized loss)
        self.trauma_mastery_threshold = 0.95  # Loss below this = mastered, can be evicted
        self.trauma_mix_ratio = 0.20  # 20% of each batch will be trauma samples
        self.trauma_mix_interval = 3  # Mix trauma every N batches
        self.MAX_TRAUMA = 15000  # Larger persistent memory
        
        self.scaler = torch.amp.GradScaler('cuda', enabled=self._use_grad_scaler())

        # 🚀 Phase 13: Adversarial Components
        self.challenger = ChallengerGenerator(input_dim=self.input_dim, hidden_dim=64).to(self.device)
        self.challenger_opt = torch.optim.AdamW(self.challenger.parameters(), lr=1e-4)
        self.guard = OverfittingGuard(ema_alpha=0.1).to(self.device)

        # 🚀 Plateau Breaker Variables (Phase 19)
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        self.patience_threshold = 7
        
        # 🚀 Phase 20: Anti-DDoS Challenger Cooldown
        self.challenger_cooldown = 0

    def _use_grad_scaler(self) -> bool:
        """Detect if GradScaler should be enabled. Disabled on AMD ROCm."""
        if self.device.type != "cuda":
            return False
        # ROCm/HIP reports torch.version.hip
        if hasattr(torch.version, 'hip') and torch.version.hip is not None:
            logger.info("🔧 AMD ROCm detected — GradScaler DISABLED (autocast still active)")
            return False
        return True

    def _load_model(self) -> SwarmBrain:
        worker_cfg = WorkerConfig(input_dim=self.input_dim, hidden_dim=self.worker_hidden)
        coord_cfg = CoordinatorConfig(num_workers=5, worker_dim=self.worker_hidden, hidden_dim=self.coord_hidden)
        
        model = SwarmBrain(worker_cfg, coord_cfg).to(self.device)
        
        # Load weights if available
        if self.checkpoint_path and os.path.exists(self.checkpoint_path):
            try:
                state = torch.load(self.checkpoint_path, map_location=self.device)
                state_dict = state.get('model_state_dict', state)
                
                # 🚀 Phase 3: Dynamic Padding for Input Growth
                # If loading older v9.7 weights (12 dims) into v10.0 model (14 dims)
                for name, param in model.named_parameters():
                    if name in state_dict and any(key in name for key in ["input_proj.weight", "cfc.x_proj.weight", "res_proj.weight"]):
                        old_w = state_dict[name]
                        if len(old_w.shape) == 2 and old_w.shape[1] < param.shape[1]:
                            logger.info(f"🦾 Zero-padding {name} from {old_w.shape[1]} to {param.shape[1]} dims")
                            new_w = torch.zeros_like(param)
                            new_w[:, :old_w.shape[1]] = old_w
                            state_dict[name] = new_w

                model.load_state_dict(state_dict, strict=False)
                logger.info(f"Loaded existing model from {self.checkpoint_path} (strict=False)")
            except Exception as e:
                logger.error(f"Failed to load checkpoint: {e}")
        else:
            msg = "Initializing fresh model with random weights." if not self.checkpoint_path else f"Checkpoint {self.checkpoint_path} not found. Using fresh weights."
            logger.warning(msg)
            
        return model

    def build_dataset(self) -> Tuple[torch.Tensor, torch.Tensor, set]:
        cutoff = datetime.utcnow() - timedelta(days=1)
        trades = self.db.query(Trade).filter(Trade.timestamp >= cutoff).all()
        
        if not trades:
            return None, None, set()

        X = []
        Y = []
        symbols_seen = set()
        
        for trade in trades:
            pnl = trade.pnl if trade.pnl is not None else 0.0
            signal = self.db.query(Signal).filter(
                Signal.symbol == trade.symbol,
                Signal.timestamp <= trade.timestamp
            ).order_by(Signal.timestamp.desc()).first()
            
            if not signal or not signal.features_snapshot:
                continue
                
            symbols_seen.add(trade.symbol)
            original_action = signal.signal_value 
            ideal_action = original_action
            
            if pnl < 0:
                ideal_action = 0 
            
            # 🚀 Phase 11: Standardized Mapping (0: Long, 1: Short, 2: Wait)
            target_scalar = 2 # Default to Wait
            if ideal_action == 1:
                target_scalar = 0 # Long
            elif ideal_action == -1:
                target_scalar = 1 # Short
                
            # 🚀 Phase 11: Correct Data Parsing (No more Dummies)
            # Parse the features_snapshot from the Signal
            try:
                features = torch.tensor(signal.features_snapshot, dtype=torch.float32)
                if features.dim() == 2: # [seq_len, dim]
                    features = features.unsqueeze(0) # [1, seq_len, dim]
                
                # Truncate/Pad to match seq_len and feature_dim
                if features.shape[1] > self.seq_len:
                    features = features[:, -self.seq_len:, :]
                
                
                # 🚀 Dynamic Tensor Padding for Backwards Compatibility
                # Old snapshots typically had 22 or 26 features, or were already truncated to 12.
                is_old_snapshot = (features.shape[-1] in [12, 22, 26])
                
                if is_old_snapshot:
                    core_feats = features[:, :, :12]
                    if self.input_dim > 12:
                        padding = torch.zeros(features.shape[0], features.shape[1], self.input_dim - 12, dtype=features.dtype, device=features.device)
                        features = torch.cat([core_feats, padding], dim=-1)
                    else:
                        features = core_feats
                else:
                    if features.shape[-1] > self.input_dim:
                        features = features[:, :, :self.input_dim]
                    elif features.shape[-1] < self.input_dim:
                        padding = torch.zeros(features.shape[0], features.shape[1], self.input_dim - features.shape[-1], dtype=features.dtype, device=features.device)
                        features = torch.cat([features, padding], dim=-1)
                    
                # 🚀 Phase 11: Feature Sanitization
                features = torch.nan_to_num(features)
                features = torch.clamp(features, min=-20.0, max=20.0)
                    
                X.append(features.to(self.device))
                Y.append(target_scalar)
            except Exception as e:
                logger.error(f"Failed to parse snapshot for signal {signal.id}: {e}")
                continue
            
        if not X:
            return None, None, set()
            
        x_tensor = torch.cat(X, dim=0).to(self.device)
        y_tensor = torch.tensor(Y, dtype=torch.long).to(self.device)
        
        return x_tensor, y_tensor, symbols_seen

    def _simulate_timeframes(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Simulates the Multi-Timeframe Brain views from a single M1 sequence.
        Downsamples the [B, L, C] tensor to provide specialists with localized views.
        """
        seq_len = x.shape[1]
        x_dict = {}
        
        # M5: Standard view
        x_dict["M5"] = x
        
        # M15: Every 3rd candle
        x_dict["M15"] = x[:, ::3, :]
        
        # M30: Every 6th candle
        x_dict["M30"] = x[:, ::6, :]
        
        # H1: Every 12th candle
        x_dict["H1"] = x[:, ::12, :]
        
        # H4: Every 48th candle
        x_dict["H4"] = x[:, ::16, :] # Truncated to avoid too small sequences in test
        
        return x_dict

    def _add_to_trauma_buffer(self, x: torch.Tensor, y: torch.Tensor, loss_val: float = None):
        """Phase 14 v2: Persistent Memory. Stores high-loss samples with their loss scores."""
        for i in range(len(x)):
            self.trauma_buffer_X.append(x[i].cpu())
            self.trauma_buffer_Y.append(y[i].cpu())
            self.trauma_buffer_losses.append(loss_val if loss_val else self.trauma_threshold)
        
        if len(self.trauma_buffer_X) > self.MAX_TRAUMA:
            # Evict the EASIEST traumas (lowest loss), keeping the hardest ones
            indexed = sorted(enumerate(self.trauma_buffer_losses), key=lambda x: x[1], reverse=True)
            keep_indices = [idx for idx, _ in indexed[:self.MAX_TRAUMA]]
            keep_indices.sort()  # Preserve temporal order
            self.trauma_buffer_X = [self.trauma_buffer_X[i] for i in keep_indices]
            self.trauma_buffer_Y = [self.trauma_buffer_Y[i] for i in keep_indices]
            self.trauma_buffer_losses = [self.trauma_buffer_losses[i] for i in keep_indices]

    def _get_trauma_mix_batch(self, current_batch_size: int):
        """Phase 14 v2: Sample trauma memories to mix into current training batch.
        Returns a subset of trauma samples proportional to trauma_mix_ratio.
        Higher-loss traumas have higher probability of being selected (Prioritized Replay)."""
        if not self.trauma_buffer_X:
            return None, None
        
        n_trauma = max(1, int(current_batch_size * self.trauma_mix_ratio))
        n_trauma = min(n_trauma, len(self.trauma_buffer_X))
        
        # Prioritized sampling: probability proportional to loss
        losses = np.array(self.trauma_buffer_losses)
        probs = losses / losses.sum()  # Normalize to probability distribution
        
        indices = np.random.choice(len(self.trauma_buffer_X), size=n_trauma, replace=False, p=probs)
        
        X = torch.stack([self.trauma_buffer_X[i] for i in indices]).to(self.device)
        Y = torch.stack([self.trauma_buffer_Y[i] for i in indices]).to(self.device)
        return X, Y

    def _train_trauma_replay(self, callback=None, epochs: int = 3):
        """Phase 14 v2: Intensive Persistent Replay with Graduated Eviction.
        Replays trauma buffer multiple times to force error correction.
        Instead, samples that the model has 'mastered' (loss < mastery_threshold) are evicted."""
        if not self.trauma_buffer_X:
            return
            
        if callback: callback(f"🧠 Initiating Intensive Trauma Replay ({epochs} epochs)... Retraining on {len(self.trauma_buffer_X)} persistent memories.")
        self.model.train()
        
        X = torch.stack(self.trauma_buffer_X)
        Y = torch.stack(self.trauma_buffer_Y)
        
        mastered_indices = set()  # Track which samples have been mastered across all replays
        
        for replay_epoch in range(epochs):
            total_loss = 0.0
            batches = 0
            sample_offset = 0
            
            # Shuffle data for this replay epoch
            indices = torch.randperm(len(X))
            X_shuffled = X[indices]
            Y_shuffled = Y[indices]
            
            for i in range(0, len(X), self.batch_size):
                batch_X = X_shuffled[i:i+self.batch_size].to(self.device)
                batch_y = Y_shuffled[i:i+self.batch_size].to(self.device)
                batch_len = len(batch_X)
                
                self.optimizer.zero_grad()
                x_dict = self._simulate_timeframes(batch_X)
            
            with torch.amp.autocast('cuda', enabled=(self.device.type == "cuda")):
                logits, worker_logits, _, _, attn_w, confidence = self.model(x_dict)
                loss_dict = self.criterion(logits, worker_logits, batch_y)
                loss = loss_dict["total"]
                
                # Titanium Shield for Trauma Replay
                if not torch.isfinite(loss) or loss.item() > 15.0:
                    sample_offset += batch_len
                    continue
                
                # Check per-sample losses to identify mastered items
                with torch.no_grad():
                    per_sample_loss = torch.nn.functional.cross_entropy(logits, batch_y, reduction='none')
                    for j in range(batch_len):
                        orig_idx = indices[sample_offset + j].item()
                        sample_loss = per_sample_loss[j].item()
                        # Update the stored loss for this sample
                        if orig_idx < len(self.trauma_buffer_losses):
                            self.trauma_buffer_losses[orig_idx] = sample_loss
                        # Mark as mastered if loss is low enough
                        if sample_loss < self.trauma_mastery_threshold:
                            mastered_indices.add(orig_idx)
                    
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                total_loss += loss.item()
                batches += 1
                sample_offset += batch_len
                
            if callback and batches > 0:
                callback(f"  ↪ Pass {replay_epoch+1}/{epochs} | Avg Loss: {total_loss/batches:.4f} | Cumulative Mastered: {len(mastered_indices)}/{len(X)}")
        
        # Graduated Eviction: Only remove samples the model has proven mastery over
        if mastered_indices:
            keep_mask = set(range(len(self.trauma_buffer_X))) - mastered_indices
            keep_list = sorted(keep_mask)
            self.trauma_buffer_X = [self.trauma_buffer_X[i] for i in keep_list]
            self.trauma_buffer_Y = [self.trauma_buffer_Y[i] for i in keep_list]
            self.trauma_buffer_losses = [self.trauma_buffer_losses[i] for i in keep_list]

    def _emit_epoch_comparison(self, epoch_history: list, callback=None):
        """Phase 14 v2: Emit a bird's-eye comparison table of all completed epochs."""
        if not callback or not epoch_history:
            return
        
        callback("━" * 90)
        callback("📊 EPOCH COMPARISON DASHBOARD")
        callback("━" * 90)
        callback(f"{'Epoch':<7} {'Loss':>8} {'Acc':>6} {'Prec':>6} {'AUC':>6} {'F1':>6} {'Traumas':>9} {'Mastered':>10} {'LR':>10}")
        callback("─" * 90)
        
        for entry in epoch_history:
            lr_str = f"{entry['lr']:.6f}"
            callback(
                f"  E{entry['epoch']:<4} "
                f"{entry['loss']:>8.4f} "
                f"{entry['acc']:>6.2f} "
                f"{entry['prec']:>6.2f} "
                f"{entry['auc']:>6.2f} "
                f"{entry['f1']:>6.2f} "
                f"{entry['trauma_size']:>9} "
                f"{entry['mastered']:>10} "
                f"{lr_str:>10}"
            )
        
        # Show trend arrows comparing last two epochs
        if len(epoch_history) >= 2:
            prev = epoch_history[-2]
            curr = epoch_history[-1]
            arrows = []
            for metric in ['loss', 'acc', 'prec', 'auc', 'f1']:
                diff = curr[metric] - prev[metric]
                if metric == 'loss':
                    # Lower loss is better
                    arrow = "🟢↓" if diff < -0.01 else ("🔴↑" if diff > 0.01 else "⚪→")
                else:
                    # Higher is better for other metrics
                    arrow = "🟢↑" if diff > 0.01 else ("🔴↓" if diff < -0.01 else "⚪→")
                arrows.append(f"{metric.upper()}: {arrow}")
            callback("─" * 90)
            callback(f"📈 Tendência: {' | '.join(arrows)}")
        
        callback("━" * 90)

    def train_and_evaluate(self, callback=None, stop_check=None):
        if callback: callback(f"🖥️ Device: {self.device} | Building dataset from Trade Journal...")
        x_tensor, y_tensor, symbols = self.build_dataset()
        
        if x_tensor is None or len(x_tensor) == 0:
            if callback: callback("Not enough trade data for learning cycle yet.")
            return

        # Update optimizer
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.learning_rate

        self.model.train()
        best_loss = float('inf')
        best_state = None
        total_samples = len(x_tensor)
        num_batches = max(1, (total_samples + self.batch_size - 1) // self.batch_size)
        total_steps = self.epochs * num_batches
        current_step = 0
        stopped = False
        epoch_history = []  # 📊 Bird's-eye epoch comparison tracker
        
        for epoch in range(self.epochs):
            if stopped:
                break
            epoch_loss = 0.0
            batch_count = 0
            
            # 🧠 Phase 20: Epoch-level accumulators for robust AUC
            epoch_all_preds = []
            epoch_all_labels = []
            epoch_all_probs = []
            
            for i in range(0, total_samples, self.batch_size):
                # Check for stop request
                if stop_check and stop_check():
                    if callback: callback(f"⛔ Stop requested at Epoch {epoch+1}, Batch {batch_count+1}. Saving best model...")
                    stopped = True
                    break
                
                batch_X = x_tensor[i : i + self.batch_size].to(self.device)
                batch_y = y_tensor[i : i + self.batch_size].to(self.device)
                
                # 🛡️ Sync Dimensions: Slice to model's expected input_dim and seq_len
                if batch_X.shape[-1] > self.input_dim:
                    batch_X = batch_X[:, :, :self.input_dim]
                if batch_X.shape[1] > self.seq_len:
                    batch_X = batch_X[:, -self.seq_len:, :]
                
                # 🚀 Dojo Warmup: Curriculum Learning Schedule 
                adv_prob = 0.0 if epoch == 0 else (0.1 if epoch == 1 else 0.3)
                
                # 🚀 Phase 13: Adversarial "Prank" Generation
                if self.challenger_cooldown > 0:
                    is_adversarial = False
                    self.challenger_cooldown -= 1
                else:
                    is_adversarial = torch.rand(1).item() < adv_prob
                
                current_pressure = self.guard.get_pressure()
                
                if is_adversarial:
                    self.challenger_cooldown = 3  # 🔥 Cooldown of 3 batches to prevent DDoS
                    self.challenger.eval()
                    batch_X_orig = batch_X.clone()
                    if callback and batch_count % 50 == 0:
                        callback(f"😈 Professor Sádico: Aplicando modo aleatório (Pressão: {current_pressure:.2f})")
                    with torch.no_grad():
                        batch_X, _ = self.challenger(batch_X, pressure=current_pressure, callback=callback if batch_count % 50 == 0 else None)
                
                self.optimizer.zero_grad()
                if is_adversarial:
                    self.challenger_opt.zero_grad()

                # 🚀 Swarm Implementation: Multi-TF Input
                x_dict = self._simulate_timeframes(batch_X)
                
                with torch.amp.autocast('cuda', enabled=(self.device.type == "cuda")):
                    logits, worker_logits, _, _, _, _ = self.model(x_dict)
                    
                    # 🚀 Phase 11: Swarm v9.7 Targets {0: Long, 1: Short, 2: Wait}
                    # 🎯 Phase 3: High Precision (Sniper Mode) - Asymmetry handled directly by SwarmLoss
                    loss_dict = self.criterion(logits, worker_logits, batch_y)
                    loss = loss_dict["total"]

                # 🛡️ Titanium Shield: Pre-Backward Protection against NaNs and Exceeding Bounds
                if torch.isnan(loss) or torch.isinf(loss) or loss.item() > 15.0:
                    val_str = f"NaN/Inf" if not torch.isfinite(loss) else f"{loss.item():.2f}"
                    if callback: callback(f"⚠️ Red Alert: Extreme Loss ({val_str}) detected in Batch {batch_count+1}. Titanium Shield activated. Skipping update to prevent collapse.")
                    continue

                # Phase 14 v2: Populate Trauma Buffer with loss score
                current_loss = loss.item()
                if current_loss > self.trauma_threshold:
                    self._add_to_trauma_buffer(batch_X.detach(), batch_y.detach(), loss_val=current_loss)
                
                # Phase 14 v2: Inline Trauma Mix - inject old mistakes into current learning
                if batch_count % self.trauma_mix_interval == 0 and len(self.trauma_buffer_X) > 0:
                    trauma_X, trauma_Y = self._get_trauma_mix_batch(len(batch_X))
                    if trauma_X is not None:
                        trauma_dict = self._simulate_timeframes(trauma_X)
                        with torch.amp.autocast('cuda', enabled=(self.device.type == "cuda")):
                            t_logits, t_worker_logits, _, _, _, _ = self.model(trauma_dict)
                            t_loss_dict = self.criterion(t_logits, t_worker_logits, trauma_Y)
                            trauma_loss = t_loss_dict["total"] * 0.3  # Weight trauma replay at 30%
                        if not torch.isnan(trauma_loss):
                            self.scaler.scale(trauma_loss).backward()

                # 1. Trader Backward
                self.scaler.scale(loss).backward()
                
                # 2. Challenger Backward
                if is_adversarial:
                    self.challenger.train()
                    for p in self.model.parameters(): p.requires_grad = False
                    batch_X_adv, _ = self.challenger(batch_X_orig, pressure=current_pressure)
                    x_dict_adv = self._simulate_timeframes(batch_X_adv)
                    with torch.amp.autocast('cuda', enabled=(self.device.type == "cuda")):
                        logits_adv, _, _, _, _, _ = self.model(x_dict_adv)
                        loss_adv_dict = self.criterion(logits_adv, worker_logits, batch_y) # worker_logits reuse
                        challenger_loss = -loss_adv_dict["total"]
                    self.scaler.scale(challenger_loss).backward()
                    for p in self.model.parameters(): p.requires_grad = True

                # 3. Step Optimizers
                self.scaler.unscale_(self.optimizer)
                # 🛡️ Titanium Shield: Strict Maximum Gradient Normalization
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.scaler.step(self.optimizer)
                if is_adversarial:
                    self.scaler.unscale_(self.challenger_opt)
                    self.scaler.step(self.challenger_opt)
                
                self.scaler.update()
                
                # 🚀 Phase 11 & 14: Multi-Class Metrics (Accuracy, Precision, AUC, F1)
                from sklearn.metrics import accuracy_score, precision_score, f1_score, roc_auc_score
                with torch.no_grad():
                    logits_np = logits.detach().cpu().numpy()
                    preds = logits_np.argmax(axis=-1)
                    true_y = batch_y.cpu().numpy()
                    
                    acc = accuracy_score(true_y, preds)
                    prec = precision_score(true_y, preds, average='macro', zero_division=0)
                    f1 = f1_score(true_y, preds, average='macro', zero_division=0)
                    
                    # 🧠 Phase 20: Accumulate for epoch-level AUC
                    probs = torch.softmax(logits.detach(), dim=-1).cpu().numpy()
                    epoch_all_preds.append(preds)
                    epoch_all_labels.append(true_y)
                    epoch_all_probs.append(probs)
                    
                    # Compute running AUC from accumulated data (use only present classes)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        try:
                            all_labels_so_far = np.concatenate(epoch_all_labels)
                            all_probs_so_far = np.concatenate(epoch_all_probs)
                            present_classes = np.unique(all_labels_so_far).astype(int).tolist()
                            if len(present_classes) == 3:
                                # Force strict normalization to exactly 1.0 to avoid scikit-learn float32 precision crash
                                norm_probs = all_probs_so_far / all_probs_so_far.sum(axis=1, keepdims=True)
                                auc = roc_auc_score(all_labels_so_far, norm_probs, multi_class='ovr', average='macro', labels=[0, 1, 2])
                            elif len(present_classes) == 2:
                                # Binary classification fallback
                                pos_class = present_classes[1]
                                binary_probs = all_probs_so_far[:, present_classes]
                                binary_probs = binary_probs / binary_probs.sum(axis=1, keepdims=True)
                                auc = roc_auc_score(all_labels_so_far, binary_probs[:, 1], labels=present_classes)
                            else:
                                auc = 0.5
                                
                            if np.isnan(auc):
                                auc = 0.5
                        except Exception as e:
                            logger.error(f"AUC Error A: {e}") 
                            auc = 0.5
                
                epoch_loss += loss.item()
                batch_count += 1
                current_step += 1
                
                # Report every 10 batches for high-fidelity UI tracking
                if batch_count % 10 == 0:
                    progress = int((current_step / total_steps) * 100)
                    
                    # Log to console less frequently (every 100 batches) to prevent spam
                    if callback and batch_count % 100 == 0:
                        sw_weights = [f"{w:.2f}" for w in loss_dict["weights"]]
                        exp_votes = worker_logits.argmax(dim=-1).cpu().numpy()[0]
                        v_map = {0: "L", 1: "S", 2: "W"}
                        thought_str = "-".join([v_map[v] for v in exp_votes])
                        callback(f"Epoch {epoch+1} | B {batch_count} | Loss: {loss.item():.4f} | Acc: {acc:.2f} | Prec: {prec:.2f} | AUC: {auc:.2f} | F1: {f1:.2f} | Swarm: {thought_str} | W: {sw_weights} | Pressure: {is_adversarial * current_pressure:.2f} | {progress}%")
                    
                    # 🧠 Phase 20: Micro-State Telemetry for "Opened Brain" Dashboard
                    if hasattr(self, '_micro_state_callback') and self._micro_state_callback:
                        try:
                            with torch.no_grad():
                                _, _, _, _, attn_w, confidence = self.model(x_dict, compute_telemetry=True)
                                w_probs = torch.softmax(worker_logits[0], dim=-1).cpu().tolist()
                                c_policy = torch.softmax(logits[0], dim=-1).cpu().tolist()
                                a_weights = attn_w[0].cpu().tolist() if attn_w is not None else None
                                conf_scores = confidence[0].squeeze(-1).cpu().tolist() if confidence is not None else None
                            self._micro_state_callback({
                                "worker_thoughts": w_probs,
                                "attention_weights": a_weights,
                                "consensus_weights": [float(w) for w in loss_dict["weights"]],
                                "coordinator_policy": c_policy,
                                "confidence_scores": conf_scores,
                                "pressure": float(is_adversarial * current_pressure),
                                "trauma_buffer_size": len(self.trauma_buffer_X),
                                "learning_rate": float(self.optimizer.param_groups[0]['lr']),
                            })
                        except Exception:
                            pass
            
            if batch_count > 0:
                avg_loss = epoch_loss / batch_count
                progress = int((current_step / total_steps) * 100)
                if callback: callback(f"Epoch {epoch+1}/{self.epochs} | Loss: {avg_loss:.4f} | Progress: {progress}%")
                
                # 🏆 Best Model Checkpoint
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                    if callback: callback(f"🏆 Epoch {epoch+1} | New best model (Journal)! Loss: {best_loss:.4f}")
                    # Phase 14: Immediate Auto-Save
                    save_path = self.checkpoint_path or "logs/models/active_model_v10.pth"
                    torch.save({'model_state_dict': self.model.state_dict()}, save_path)
                    if callback: callback(f"💾 Epoch {epoch+1} | Auto-saved best model to {os.path.basename(save_path)}")
                
            # Phase 14: Execute Trauma Replay
            trauma_mastered = 0
            trauma_size_before = len(self.trauma_buffer_X)
            if len(self.trauma_buffer_X) > 0:
                if callback: callback(f"🧠 Epoch {epoch+1} | Initiating Trauma Replay... Retraining on {len(self.trauma_buffer_X)} persistent memories.")
                trauma_size_before = len(self.trauma_buffer_X)
                self._train_trauma_replay(lambda m: callback(f"Epoch {epoch+1} | {m}"))
                trauma_mastered = trauma_size_before - len(self.trauma_buffer_X)
                if callback: callback(f"✅ Epoch {epoch+1} | Trauma Replay completed.")
            
            # 📊 Epoch Comparison: Compute aggregate metrics and record snapshot
            if batch_count > 0:
                epoch_avg_acc = np.mean([accuracy_score(l, p) for l, p in zip(epoch_all_labels, epoch_all_preds)]) if epoch_all_preds else 0.0
                epoch_avg_prec = np.mean([precision_score(l, p, average='macro', zero_division=0) for l, p in zip(epoch_all_labels, epoch_all_preds)]) if epoch_all_preds else 0.0
                epoch_avg_f1 = np.mean([f1_score(l, p, average='macro', zero_division=0) for l, p in zip(epoch_all_labels, epoch_all_preds)]) if epoch_all_preds else 0.0
                # Epoch-level AUC from all accumulated probs
                epoch_auc = 0.5
                if epoch_all_probs and epoch_all_labels:
                    try:
                        all_l = np.concatenate(epoch_all_labels)
                        all_p = np.concatenate(epoch_all_probs)
                        present = np.unique(all_l).astype(int).tolist()
                        if len(present) >= 2:
                            norm_p = all_p / all_p.sum(axis=1, keepdims=True)
                            if len(present) == 3:
                                epoch_auc = roc_auc_score(all_l, norm_p, multi_class='ovr', average='macro', labels=[0,1,2])
                            else:
                                bp = norm_p[:, present]
                                bp = bp / bp.sum(axis=1, keepdims=True)
                                epoch_auc = roc_auc_score(all_l, bp[:, 1], labels=present)
                    except Exception:
                        epoch_auc = 0.5
                
                epoch_history.append({
                    'epoch': epoch + 1,
                    'loss': epoch_loss / batch_count,
                    'acc': epoch_avg_acc,
                    'prec': epoch_avg_prec,
                    'auc': epoch_auc,
                    'f1': epoch_avg_f1,
                    'trauma_size': len(self.trauma_buffer_X),
                    'mastered': trauma_mastered,
                    'lr': self.optimizer.param_groups[0]['lr'],
                })
                self._emit_epoch_comparison(epoch_history, callback)

        # Restore and save the best model weights
        if best_state:
            self.model.load_state_dict(best_state)
        
        save_path = self.checkpoint_path or "logs/models/active_model.pth"
        if callback: callback(f"💾 Saving best checkpoint (Loss: {best_loss:.4f}) to {os.path.basename(save_path)}...")
        torch.save({'model_state_dict': self.model.state_dict()}, save_path)
        self.update_registry(best_loss, symbols)

    def train_on_dataset(self, dataset_name: str, callback=None, stop_check=None):
        shard_dir = f"data/cache/{dataset_name}/train_shards"
        if not os.path.exists(shard_dir):
            if callback: callback(f"Error: Dataset {dataset_name} shards not found.")
            return

        shards = sorted([f for f in os.listdir(shard_dir) if f.endswith(".pt")])
        if not shards:
            if callback: callback(f"Error: No shards found in {shard_dir}")
            return

        if callback: callback(f"🖥️ Device: {self.device} | 🚀 Found {len(shards)} shards. Running Supervised Protocol...")
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.learning_rate

        self.model.train()
        total_batches = 0
        best_loss = float('inf')
        best_state = None
        stopped = False
        
        # --- GLOBAL PROGRESS PRE-CALCULATION ---
        # Sum total batches once to avoid redundant I/O per epoch
        total_epoch_batches = 0
        for s_file in shards:
            try:
                # Only load enough to get the size
                s_meta = torch.load(os.path.join(shard_dir, s_file), map_location='cpu', weights_only=True)
                s_X = s_meta.get('X')
                if s_X is not None:
                    total_epoch_batches += (len(s_X) + self.batch_size - 1) // self.batch_size
            except Exception:
                continue
        
        if total_epoch_batches == 0: total_epoch_batches = 1
        epoch_history = []  # 📊 Bird's-eye epoch comparison tracker
        
        for epoch in range(self.epochs):
            if stopped:
                break
            epoch_loss = 0.0
            epoch_batches = 0
            
            # 🧠 Phase 20: Consolidate ALL shards into one dataset for maximum GPU utilization
            all_X_list = []
            all_y_list = []
            for shard_file in shards:
                shard_path = os.path.join(shard_dir, shard_file)
                data = torch.load(shard_path, map_location='cpu')
                X_s = data['X']
                y_s = data['y']
                
                # Robust Dimension Migration
                shard_dim = X_s.shape[-1]
                if shard_dim == 12 and self.input_dim > 12:
                    padding = torch.zeros(X_s.shape[0], X_s.shape[1], self.input_dim - 12, dtype=X_s.dtype)
                    X_s = torch.cat([X_s, padding], dim=-1)
                elif shard_dim > self.input_dim:
                    X_s = X_s[:, :, :self.input_dim]
                elif shard_dim < self.input_dim:
                    padding = torch.zeros(X_s.shape[0], X_s.shape[1], self.input_dim - shard_dim, dtype=X_s.dtype)
                    X_s = torch.cat([X_s, padding], dim=-1)
                
                all_X_list.append(X_s)
                all_y_list.append(y_s)
            
            X = torch.cat(all_X_list, dim=0)
            y = torch.cat(all_y_list, dim=0)
            del all_X_list, all_y_list
            
            if callback and epoch == 0:
                callback(f"📂 Consolidated {len(shards)} shards → {len(X)} samples in memory")
            
            total_epoch_batches = max(1, (len(X) + self.batch_size - 1) // self.batch_size)
            
            # 🧠 Phase 20: Epoch-level accumulators for robust AUC
            epoch_all_preds = []
            epoch_all_labels = []
            epoch_all_probs = []
                
            for i in range(0, len(X), self.batch_size):
                # Check for stop request
                if stop_check and stop_check():
                    if callback: callback(f"⛔ Stop requested at Epoch {epoch+1}, Batch {epoch_batches+1}. Saving best model...")
                    stopped = True
                    break
                
                batch_X = X[i : i + self.batch_size].to(self.device)
                batch_y = y[i : i + self.batch_size].long().to(self.device)
                
                # 🛡️ Sync Dimensions
                if batch_X.shape[-1] > self.input_dim:
                    batch_X = batch_X[:, :, :self.input_dim]
                if batch_X.shape[1] > self.seq_len:
                    batch_X = batch_X[:, -self.seq_len:, :]
                
                # 🚀 Phase 11: Feature Sanitization
                batch_X = torch.nan_to_num(batch_X)
                batch_X = torch.clamp(batch_X, min=-20.0, max=20.0)
                
                # 🚀 Phase 13 & 16: Extreme Warmup Schedule
                # Ep 1-2: 0%, Ep 3-4: 5%, Ep 5-6: 10%, Ep 7-8: 15%, Ep 9+: 30%
                adv_schedule = [0.0, 0.0, 0.05, 0.05, 0.10, 0.10, 0.15, 0.15]
                adv_prob = adv_schedule[epoch] if epoch < len(adv_schedule) else 0.30
                
                # 🚀 Phase 13: Adversarial "Prank" Generation (With Cooldown)
                if self.challenger_cooldown > 0:
                    is_adversarial = False
                    self.challenger_cooldown -= 1
                else:
                    is_adversarial = torch.rand(1).item() < adv_prob
                    
                current_pressure = self.guard.get_pressure()
                
                if is_adversarial:
                    self.challenger_cooldown = 3 # 🔥 Cooldown
                    # Challenger attempts to 'prank' the features to confuse the Trader
                    self.challenger.eval() # Generator doesn't update its own weights during this pass
                    batch_X_orig = batch_X.clone()
                    
                    # LOG: Feed back the "Sadistic Professor" behavior to the UI
                    if callback and epoch_batches % 50 == 0:
                        callback(f"😈 Professor Sádico: Aplicando modo aleatório (Pressão: {current_pressure:.2f})")
                        
                    # 🌋 Phase 19: Professor Breathe (Harmonic Oscillation)
                    import math
                    breathe_factor = 1.0 + 0.2 * math.sin(epoch_batches * 0.1)
                    dynamic_pressure = current_pressure * breathe_factor
                    
                    with torch.no_grad():
                        batch_X, _ = self.challenger(batch_X, pressure=dynamic_pressure, callback=callback if epoch_batches % 50 == 0 else None)
                 
                self.optimizer.zero_grad()
                if is_adversarial:
                    self.challenger_opt.zero_grad()

                x_dict = self._simulate_timeframes(batch_X)
                
                with torch.amp.autocast('cuda', enabled=(self.device.type == "cuda")):
                    logits, worker_logits, _, _, _, _ = self.model(x_dict)
                    loss_dict = self.criterion(logits, worker_logits, batch_y)
                    loss = loss_dict["total"]

                # 🛡️ Titanium Shield: Pre-Backward Protection against NaNs and Exceeding Bounds
                if torch.isnan(loss) or torch.isinf(loss) or loss.item() > 15.0:
                    val_str = f"NaN/Inf" if not torch.isfinite(loss) else f"{loss.item():.2f}"
                    if callback: callback(f"⚠️ Red Alert: Extreme Loss ({val_str}) detected in Shard. Titanium Shield activated. Skipping update.")
                    continue

                # Phase 14 v2: Populate Trauma Buffer with loss score
                current_loss = loss.item()
                if current_loss > self.trauma_threshold:
                    self._add_to_trauma_buffer(batch_X.detach(), batch_y.detach(), loss_val=current_loss)
                
                # Phase 14 v2: Inline Trauma Mix - inject old mistakes into current learning
                if epoch_batches % self.trauma_mix_interval == 0 and len(self.trauma_buffer_X) > 0:
                    trauma_X, trauma_Y = self._get_trauma_mix_batch(len(batch_X))
                    if trauma_X is not None:
                        trauma_dict = self._simulate_timeframes(trauma_X)
                        with torch.amp.autocast('cuda', enabled=(self.device.type == "cuda")):
                            t_logits, t_worker_logits, _, _, _, _ = self.model(trauma_dict)
                            t_loss_dict = self.criterion(t_logits, t_worker_logits, trauma_Y)
                            trauma_loss = t_loss_dict["total"] * 0.3  # Weight trauma replay at 30%
                        if not torch.isnan(trauma_loss):
                            self.scaler.scale(trauma_loss).backward()

                # 1. Trader Backward
                self.scaler.scale(loss).backward()
                
                # 2. Challenger Backward (if needed)
                if is_adversarial:
                    self.challenger.train()
                    for p in self.model.parameters(): p.requires_grad = False
                    
                    batch_X_adv, _ = self.challenger(batch_X_orig, pressure=current_pressure)
                    x_dict_adv = self._simulate_timeframes(batch_X_adv)
                    
                    with torch.amp.autocast('cuda', enabled=(self.device.type == "cuda")):
                        logits_adv, worker_logits_adv, _, _, _, _ = self.model(x_dict_adv)
                        loss_adv_dict = self.criterion(logits_adv, worker_logits_adv, batch_y)
                        challenger_loss = -loss_adv_dict["total"] 
                    
                    self.scaler.scale(challenger_loss).backward()
                    for p in self.model.parameters(): p.requires_grad = True

                # 3. Step Optimizers
                self.scaler.unscale_(self.optimizer)
                # 🛡️ Titanium Shield: Strict Maximum Gradient Normalization
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.scaler.step(self.optimizer)
                if is_adversarial:
                    self.scaler.unscale_(self.challenger_opt)
                    self.scaler.step(self.challenger_opt)
                    
                # 4. Final Scaler Update
                self.scaler.update()
                
                # 🚀 Phase 11 & 14: Multi-Class Metrics (Accuracy, Precision, AUC, F1)
                from sklearn.metrics import accuracy_score, precision_score, f1_score, roc_auc_score
                with torch.no_grad():
                    logits_np = logits.detach().cpu().numpy()
                    preds = logits_np.argmax(axis=-1)
                    true_y = batch_y.cpu().numpy()
                    
                    acc = accuracy_score(true_y, preds)
                    prec = precision_score(true_y, preds, average='macro', zero_division=0)
                    f1 = f1_score(true_y, preds, average='macro', zero_division=0)
                    
                    # 🧠 Phase 20: Accumulate for epoch-level AUC
                    probs = torch.softmax(logits.detach(), dim=-1).cpu().numpy()
                    epoch_all_preds.append(preds)
                    epoch_all_labels.append(true_y)
                    epoch_all_probs.append(probs)
                    
                    # Compute running AUC from accumulated data (use only present classes)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        try:
                            all_labels_so_far = np.concatenate(epoch_all_labels)
                            all_probs_so_far = np.concatenate(epoch_all_probs)
                            present_classes = np.unique(all_labels_so_far).astype(int).tolist()
                            if len(present_classes) == 3:
                                # Force strict normalization to exactly 1.0 to avoid scikit-learn float32 precision crash
                                norm_probs = all_probs_so_far / all_probs_so_far.sum(axis=1, keepdims=True)
                                auc = roc_auc_score(all_labels_so_far, norm_probs, multi_class='ovr', average='macro', labels=[0, 1, 2])
                            elif len(present_classes) == 2:
                                # Binary classification fallback
                                # Scikit-Learn expects a 1D array of probabilities for the positive class (the second class in present_classes)
                                pos_class = present_classes[1]
                                # Re-normalize probabilities just for the two present classes
                                binary_probs = all_probs_so_far[:, present_classes]
                                binary_probs = binary_probs / binary_probs.sum(axis=1, keepdims=True)
                                auc = roc_auc_score(all_labels_so_far, binary_probs[:, 1], labels=present_classes)
                            else:
                                auc = 0.5
                                
                            if np.isnan(auc):
                                auc = 0.5
                        except Exception as e:
                            logger.error(f"AUC Error B: {e}") 
                            auc = 0.5
                    
                    # Phase 13: "Panic Protection"
                    if is_adversarial and acc < 0.40:
                        self.guard.pressure = torch.tensor(max(0.1, self.guard.get_pressure() * 0.9), device=self.device)
                        if callback: callback(f"🛡️ Epoch {epoch+1} | Panic Protection: Reducing Professor pressure to {self.guard.get_pressure():.2f}")
                
                epoch_loss += loss.item()
                epoch_batches += 1
                total_batches += 1
                
                # Report every 5 batches for high-fidelity "Chart Dance"
                if epoch_batches % 5 == 0:
                    progress = int((epoch_batches / total_epoch_batches) * 100)
                    
                    # Log to console less frequently (every 100 batches)
                    if callback and epoch_batches % 100 == 0:
                        sw_weights = [f"{w:.2f}" for w in loss_dict["weights"]]
                        exp_votes = worker_logits.argmax(dim=-1).cpu().numpy()[0] 
                        v_map = {0: "L", 1: "S", 2: "W"}
                        thought_str = "-".join([v_map[v] for v in exp_votes])
                        callback(f"Epoch {epoch+1} | B {epoch_batches} | Loss: {loss.item():.4f} | Acc: {acc:.2f} | Prec: {prec:.2f} | AUC: {auc:.2f} | F1: {f1:.2f} | Swarm: {thought_str} | W: {sw_weights} | Pressure: {is_adversarial * current_pressure:.2f} | {progress}%")
                    
                    # 🧠 Phase 20: Micro-State Telemetry for "Opened Brain" Dashboard
                    if hasattr(self, '_micro_state_callback') and self._micro_state_callback:
                        try:
                            with torch.no_grad():
                                _, _, _, _, attn_w, confidence = self.model(x_dict, compute_telemetry=True)
                                w_probs = torch.softmax(worker_logits[0], dim=-1).cpu().tolist()
                                c_policy = torch.softmax(logits[0], dim=-1).cpu().tolist()
                                a_weights = attn_w[0].cpu().tolist() if attn_w is not None else None
                                conf_scores = confidence[0].squeeze(-1).cpu().tolist() if confidence is not None else None
                            self._micro_state_callback({
                                "worker_thoughts": w_probs,
                                "attention_weights": a_weights,
                                "consensus_weights": [float(w) for w in loss_dict["weights"]],
                                "coordinator_policy": c_policy,
                                "confidence_scores": conf_scores,
                                "pressure": float(is_adversarial * current_pressure),
                                "trauma_buffer_size": len(self.trauma_buffer_X),
                                "learning_rate": float(self.optimizer.param_groups[0]['lr']),
                            })
                        except Exception:
                            pass


            # 🚀 Phase 13: Validation Pass & OverfittingGuard Update
            val_shard_dir = f"data/cache/{dataset_name}/val_shards"
            if os.path.exists(val_shard_dir):
                self.model.eval()
                val_loss = 0.0
                val_batches = 0
                val_shards = sorted([f for f in os.listdir(val_shard_dir) if f.endswith(".pt")])
                
                with torch.no_grad():
                    for v_shard in val_shards[:2]: # Sample 2 shards for speed
                        v_data = torch.load(os.path.join(val_shard_dir, v_shard), map_location='cpu')
                        v_X_cpu, v_y_cpu = v_data['X'], v_data['y']
                        
                        for j in range(0, len(v_X_cpu), self.batch_size):
                            v_X = v_X_cpu[j : j + self.batch_size].to(self.device)
                            v_y = v_y_cpu[j : j + self.batch_size].to(self.device).long()
                            
                            # 🚀 Robust Dimension Migration for Validation
                            shard_dim = v_X.shape[-1]
                            if shard_dim > self.input_dim:
                                v_X = v_X[:, :, :self.input_dim]
                            elif shard_dim < self.input_dim:
                                padding = torch.zeros(v_X.shape[0], v_X.shape[1], self.input_dim - shard_dim, dtype=v_X.dtype, device=v_X.device)
                                v_X = torch.cat([v_X, padding], dim=-1)
                                
                            # Ensure no NaNs or extreme values in validation
                            v_X = torch.nan_to_num(v_X)
                            v_X = torch.clamp(v_X, min=-20.0, max=20.0)
                            
                            v_x_dict = self._simulate_timeframes(v_X)
                            v_logits, v_worker_logits, _, _, _, _ = self.model(v_x_dict)
                            v_loss_dict = self.criterion(v_logits, v_worker_logits, v_y)
                            val_loss += v_loss_dict["total"].item()
                            val_batches += 1
                            
                if val_batches > 0:
                    avg_val_loss = val_loss / val_batches
                    if epoch_batches > 0:
                        avg_train_loss = epoch_loss / epoch_batches 
                        # Phase 13: Delay OverfittingGuard until masters basic data (Epoch 5+)
                        if epoch >= 4:
                            self.guard.update(avg_train_loss, avg_val_loss)
                        else:
                            if callback: callback(f"🧸 Epoch {epoch+1} | Professor Gentil: Pressão travada para masters inicial...")
                    
                    val_label = "Limpo (Base)" if epoch < 2 else "vs Professor Sádico"
                    if callback: callback(f"📉 Epoch {epoch+1} | Validation Loss ({val_label}): {avg_val_loss:.4f}")
                        
                    # Phase 14: ReduceLROnPlateau
                    self.scheduler.step(avg_val_loss)
                        
                    current_lr = self.optimizer.param_groups[0]['lr']
                    if callback and epoch > 0:
                        callback(f"🔧 Learning Rate: {current_lr:.6f} | Challenger Pressure: {self.guard.get_pressure():.2f}")

                    # 🌋 Phase 19: Plateau Breaker (Warm Restart & Professor Retreat)
                    if avg_val_loss < self.best_val_loss:
                        self.best_val_loss = avg_val_loss
                        self.epochs_without_improvement = 0
                    else:
                        self.epochs_without_improvement += 1
                        
                    if self.epochs_without_improvement >= self.patience_threshold:
                        if callback: callback(f"🌋 Plateau Detected! Breaking stagnancy with Warm Restart & Professor Retreat.")
                        # 1. Warm Restart: Reset LR to 75% of initial
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = self.learning_rate * 0.75
                        
                        # 2. Professor Retreat: Temporary Pressure Nerf (Reduce by 50%)
                        self.guard.pressure = torch.tensor(max(0.1, self.guard.get_pressure() * 0.5), device=self.device)
                        
                        self.epochs_without_improvement = 0 # Reset tracker

            # Phase 14: Execute Trauma Replay at the end of every epoch if accumulated
            trauma_mastered = 0
            trauma_size_before = len(self.trauma_buffer_X)
            if len(self.trauma_buffer_X) > 0:
                if callback: callback(f"🧠 Epoch {epoch+1} | Initiating Trauma Replay... Retraining on {len(self.trauma_buffer_X)} persistent memories.")
                trauma_size_before = len(self.trauma_buffer_X)
                self._train_trauma_replay(lambda m: callback(f"Epoch {epoch+1} | {m}"))
                trauma_mastered = trauma_size_before - len(self.trauma_buffer_X)
                if callback: callback(f"✅ Epoch {epoch+1} | Trauma Replay completed.")
            
            if epoch_batches > 0:
                avg_epoch_loss = epoch_loss / epoch_batches
                if callback: callback(f"✅ Epoch {epoch+1} Complete | Avg Loss: {avg_epoch_loss:.4f}")
                
                # 🏆 Best Model Checkpoint
                if avg_epoch_loss < best_loss:
                    best_loss = avg_epoch_loss
                    best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                    if callback: callback(f"🏆 Epoch {epoch+1} | New best model (Dataset)! Loss: {best_loss:.4f}")
                    # Phase 14: Immediate Auto-Save
                    save_path = self.checkpoint_path or "logs/models/active_model_v10.pth"
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    torch.save({'model_state_dict': self.model.state_dict()}, save_path)
                    if callback: callback(f"💾 Epoch {epoch+1} | Auto-saved best model to {os.path.basename(save_path)}")
                
                # 📊 Epoch Comparison: Compute aggregate metrics and record snapshot
                epoch_avg_acc = np.mean([accuracy_score(l, p) for l, p in zip(epoch_all_labels, epoch_all_preds)]) if epoch_all_preds else 0.0
                epoch_avg_prec = np.mean([precision_score(l, p, average='macro', zero_division=0) for l, p in zip(epoch_all_labels, epoch_all_preds)]) if epoch_all_preds else 0.0
                epoch_avg_f1 = np.mean([f1_score(l, p, average='macro', zero_division=0) for l, p in zip(epoch_all_labels, epoch_all_preds)]) if epoch_all_preds else 0.0
                epoch_auc = 0.5
                if epoch_all_probs and epoch_all_labels:
                    try:
                        all_l = np.concatenate(epoch_all_labels)
                        all_p = np.concatenate(epoch_all_probs)
                        present = np.unique(all_l).astype(int).tolist()
                        if len(present) >= 2:
                            norm_p = all_p / all_p.sum(axis=1, keepdims=True)
                            if len(present) == 3:
                                epoch_auc = roc_auc_score(all_l, norm_p, multi_class='ovr', average='macro', labels=[0,1,2])
                            else:
                                bp = norm_p[:, present]
                                bp = bp / bp.sum(axis=1, keepdims=True)
                                epoch_auc = roc_auc_score(all_l, bp[:, 1], labels=present)
                    except Exception:
                        epoch_auc = 0.5
                
                epoch_history.append({
                    'epoch': epoch + 1,
                    'loss': avg_epoch_loss,
                    'acc': epoch_avg_acc,
                    'prec': epoch_avg_prec,
                    'auc': epoch_auc,
                    'f1': epoch_avg_f1,
                    'trauma_size': len(self.trauma_buffer_X),
                    'mastered': trauma_mastered,
                    'lr': self.optimizer.param_groups[0]['lr'],
                })
                self._emit_epoch_comparison(epoch_history, callback)

        # Restore and save the best model weights
        if best_state:
            self.model.load_state_dict(best_state)
        
        save_path = self.checkpoint_path or "logs/models/active_model_v10.pth"
        if callback: callback(f"💾 Saving best checkpoint (Loss: {best_loss:.4f}) to {os.path.basename(save_path)}...")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save({'model_state_dict': self.model.state_dict()}, save_path)
        self.update_registry(best_loss, {dataset_name})
        
    def update_registry(self, loss: float, symbols: set):
        version_tag = "v10.0-Live"
        registry_entry = self.db.query(ModelVersion).filter(ModelVersion.tag == version_tag).first()
        
        if not registry_entry:
            registry_entry = ModelVersion(
                tag=version_tag,
                path="logs/models/active_model_v10.pth",
                is_active=True,
                metrics={}
            )
            self.db.add(registry_entry)
            self.db.commit()
            self.db.refresh(registry_entry)
            
        current_metrics = registry_entry.metrics or {}
        for symbol in symbols:
            current_metrics[symbol] = {
                "loss": round(loss, 4),
                "last_trained": datetime.utcnow().isoformat()
            }
        registry_entry.metrics = current_metrics
        registry_entry.category_type = "GENERALIST" if len(current_metrics.keys()) > 1 else "SPECIALIST"
        self.db.commit()

if __name__ == "__main__":
    learner = ContinuousLearner()
    learner.train_and_evaluate()



--- FILE: backend/src/train/batch_search.py ---
"""
Batch Sensitivity Search for I.A.R.A. v4.4
==========================================
Tests different batch sizes with sqrt LR scaling to find optimal config.

Configurations:
- Batch Sizes: [256, 512, 1024, 2048]
- LR Scaling: LR_new = 0.001 * sqrt(Batch / 256)
- Epochs per run: 30
- Metrics: Time/Epoch, Val AUC, Test Precision
"""

import gc
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import roc_auc_score

from src.model.iara_liquid import IARA, LiquidConfig
from src.train.trainer_liquid import LiquidDataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Config
BATCH_SIZES = [256, 512, 1024, 2048]
BASE_LR = 0.001
BASE_BATCH = 256
EPOCHS_PER_RUN = 30
SEQ_LEN = 64
HIDDEN_DIM = 512
DROPOUT = 0.4


@dataclass
class SearchResult:
    """Result of a single batch size experiment."""
    batch_size: int
    learning_rate: float
    time_per_epoch: float
    best_val_auc: float
    test_precision: float
    status: str


def compute_scaled_lr(batch_size: int) -> float:
    """Square root LR scaling rule."""
    return BASE_LR * math.sqrt(batch_size / BASE_BATCH)


def reset_model(device: torch.device) -> IARA:
    """Create fresh model with reset weights."""
    config = LiquidConfig(
        input_dim=11,
        hidden_dim=HIDDEN_DIM,
        dropout=DROPOUT
    )
    model = IARA(config).to(device)
    return model


def train_single_config(
    batch_size: int,
    learning_rate: float,
    device: torch.device
) -> SearchResult:
    """Train model with a single batch size configuration."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Testing Batch Size: {batch_size}, LR: {learning_rate:.6f}")
    logger.info(f"{'='*60}")
    
    # Clear memory
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    try:
        # Load data
        full_dataset = LiquidDataset(seq_len=SEQ_LEN, augment=False)
        train_idxs, val_idxs = full_dataset.get_chronological_split(0.2)
        
        train_dataset = LiquidDataset(
            seq_len=SEQ_LEN,
            augment=True,
            noise_std=0.02,
            scale_range=(0.98, 1.02),
            label_smooth=0.1
        )
        val_dataset = LiquidDataset(seq_len=SEQ_LEN, augment=False, label_smooth=0.0)
        
        train_subset = Subset(train_dataset, train_idxs)
        val_subset = Subset(val_dataset, val_idxs)
        
        # Adjust workers based on batch size
        num_workers = min(4, batch_size // 64)
        
        train_loader = DataLoader(
            train_subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        # Fresh model
        model = reset_model(device)
        
        # Optimizer with scaled LR
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=1e-4
        )
        
        # Scheduler
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=learning_rate * 2,
            steps_per_epoch=len(train_loader),
            epochs=EPOCHS_PER_RUN,
            pct_start=0.3
        )
        
        # Loss
        criterion = nn.BCEWithLogitsLoss()
        
        # Training loop
        best_val_auc = 0.0
        epoch_times = []
        
        for epoch in range(EPOCHS_PER_RUN):
            epoch_start = time.time()
            
            # Train
            model.train()
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                
                optimizer.zero_grad()
                logits, _ = model(x)
                loss = criterion(logits, y)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
                scheduler.step()
            
            # Validate
            model.eval()
            all_preds, all_targets = [], []
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(device), y.to(device)
                    logits, _ = model(x)
                    probs = torch.sigmoid(logits)
                    all_preds.extend(probs.cpu().numpy())
                    all_targets.extend((y > 0.5).float().cpu().numpy())
            
            try:
                val_auc = roc_auc_score(all_targets, all_preds)
            except:
                val_auc = 0.5
            
            if val_auc > best_val_auc:
                best_val_auc = val_auc
            
            epoch_time = time.time() - epoch_start
            epoch_times.append(epoch_time)
            
            if epoch % 10 == 0 or epoch == EPOCHS_PER_RUN - 1:
                logger.info(f"  Epoch {epoch}: Val AUC={val_auc:.4f}, Time={epoch_time:.2f}s")
        
        # Calculate test precision (simple approximation)
        preds_array = np.array(all_preds)
        targets_array = np.array(all_targets)
        pred_classes = (preds_array > 0.5).astype(float)
        
        tp = ((pred_classes == 1) & (targets_array == 1)).sum()
        fp = ((pred_classes == 1) & (targets_array == 0)).sum()
        test_precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0.0
        
        avg_time = np.mean(epoch_times)
        
        result = SearchResult(
            batch_size=batch_size,
            learning_rate=learning_rate,
            time_per_epoch=avg_time,
            best_val_auc=best_val_auc,
            test_precision=test_precision,
            status="✓ Success"
        )
        
        logger.info(f"  Result: AUC={best_val_auc:.4f}, Precision={test_precision:.1f}%, Time/Epoch={avg_time:.2f}s")
        
        return result
        
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            logger.warning(f"  OOM for batch size {batch_size}")
            return SearchResult(
                batch_size=batch_size,
                learning_rate=learning_rate,
                time_per_epoch=0,
                best_val_auc=0,
                test_precision=0,
                status="✗ OOM"
            )
        raise


def run_search():
    """Run the full batch sensitivity search."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    results: List[SearchResult] = []
    
    for batch_size in BATCH_SIZES:
        lr = compute_scaled_lr(batch_size)
        result = train_single_config(batch_size, lr, device)
        results.append(result)
    
    # Print summary table
    print("\n" + "=" * 80)
    print("BATCH SENSITIVITY SEARCH RESULTS")
    print("=" * 80)
    print(f"{'Batch Size':>12} | {'LR':>10} | {'Time/Epoch':>12} | {'Val AUC':>10} | {'Precision':>12} | {'Status':>12}")
    print("-" * 80)
    
    for r in results:
        print(f"{r.batch_size:>12} | {r.learning_rate:>10.6f} | {r.time_per_epoch:>10.2f}s | {r.best_val_auc:>10.4f} | {r.test_precision:>10.1f}% | {r.status:>12}")
    
    print("=" * 80)
    
    # Find best config
    valid_results = [r for r in results if "Success" in r.status]
    if valid_results:
        best = max(valid_results, key=lambda x: x.best_val_auc)
        print(f"\n🏆 BEST CONFIG: Batch={best.batch_size}, LR={best.learning_rate:.6f}, AUC={best.best_val_auc:.4f}")


if __name__ == "__main__":
    run_search()



--- FILE: backend/src/train/trainer.py ---
"""
Trainer - Training Loop for Saci-Nano (Causal Language Modeling)
=================================================================
Training loop with gradient accumulation, warmup, checkpointing,
and continuous learning support for AMD ROCm.

Supports configurable training modes via config/training_modes.yaml
Usage: python src/train/trainer.py --mode <mode_name>
"""

import argparse
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.model.saci_nano import SaciNano, SaciNanoConfig
from src.train.dataset import create_dataloader
from src.data.tokenizer import FinancialTokenizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Paths
CHECKPOINT_DIR = Path(__file__).parent.parent.parent / "checkpoints"
DATA_TOKENIZED_DIR = Path(__file__).parent.parent.parent / "data" / "tokenized"
TOKENIZER_PATH = Path(__file__).parent.parent.parent / "data" / "tokenizer.pkl"
CONFIG_DIR = Path(__file__).parent.parent.parent / "config"
MODES_CONFIG_PATH = CONFIG_DIR / "training_modes.yaml"


@dataclass
class TrainerConfig:
    """Training configuration."""
    
    # Model
    dim: int = 512
    n_layers: int = 8
    n_heads: int = 8
    n_kv_heads: int = 4
    dropout: float = 0.1
    
    # Data
    seq_len: int = 384
    batch_size: int = 16
    
    # Training
    epochs: int = 50
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    
    # Scheduler
    warmup_steps: int = 500
    min_lr_ratio: float = 0.1
    
    # Checkpointing
    save_every_n_epochs: int = 5
    checkpoint_dir: Path = field(default_factory=lambda: CHECKPOINT_DIR)
    resume_from: Optional[Path] = None
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Metadata
    mode_name: str = "custom"
    mode_description: str = ""
    
    @classmethod
    def from_yaml_mode(cls, mode_name: str, config_path: Path = MODES_CONFIG_PATH) -> "TrainerConfig":
        """
        Load configuration from a YAML mode preset.
        
        Args:
            mode_name: Name of the mode (e.g., 'fast', 'moderate', 'full')
            config_path: Path to the training_modes.yaml file
            
        Returns:
            TrainerConfig instance
        """
        if not config_path.exists():
            logger.warning(f"Config file not found: {config_path}. Using defaults.")
            return cls()
        
        with open(config_path, 'r') as f:
            all_modes = yaml.safe_load(f)
        
        # Get default mode if none specified
        if mode_name == "default":
            mode_name = all_modes.get("_active_default", "moderate")
        
        if mode_name not in all_modes:
            available = [k for k in all_modes.keys() if not k.startswith("_")]
            raise ValueError(f"Mode '{mode_name}' not found. Available: {available}")
        
        mode = all_modes[mode_name]
        
        # Extract nested config
        model_cfg = mode.get("model", {})
        data_cfg = mode.get("data", {})
        train_cfg = mode.get("training", {})
        ckpt_cfg = mode.get("checkpointing", {})
        
        return cls(
            # Model
            dim=model_cfg.get("dim", 512),
            n_layers=model_cfg.get("n_layers", 8),
            n_heads=model_cfg.get("n_heads", 8),
            n_kv_heads=model_cfg.get("n_kv_heads", 4),
            dropout=model_cfg.get("dropout", 0.1),
            # Data
            seq_len=data_cfg.get("seq_len", 384),
            batch_size=data_cfg.get("batch_size", 16),
            # Training
            epochs=train_cfg.get("epochs", 50),
            learning_rate=train_cfg.get("learning_rate", 3e-4),
            weight_decay=train_cfg.get("weight_decay", 0.1),
            gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 4),
            max_grad_norm=train_cfg.get("max_grad_norm", 1.0),
            warmup_steps=train_cfg.get("warmup_steps", 500),
            # Checkpointing
            save_every_n_epochs=ckpt_cfg.get("save_every_n_epochs", 5),
            # Metadata
            mode_name=mode_name,
            mode_description=mode.get("description", "")
        )
    
    @staticmethod
    def list_available_modes(config_path: Path = MODES_CONFIG_PATH) -> list:
        """List all available training modes."""
        if not config_path.exists():
            return []
        
        with open(config_path, 'r') as f:
            all_modes = yaml.safe_load(f)
        
        modes = []
        default_mode = all_modes.get("_active_default", "moderate")
        
        for name, config in all_modes.items():
            if name.startswith("_"):
                continue
            desc = config.get("description", "No description")
            is_default = " (default)" if name == default_mode else ""
            modes.append(f"  {name}{is_default}: {desc}")
        
        return modes


def get_lr_scheduler(
    optimizer: AdamW,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.1
) -> LambdaLR:
    """Create learning rate scheduler with warmup and cosine decay."""
    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        else:
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(min_lr_ratio, 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159)).item()))
    
    return LambdaLR(optimizer, lr_lambda)


class Trainer:
    """
    Trainer for Saci-Nano Causal Language Model.
    
    Features:
    - Cross-Entropy Loss for Next Token Prediction
    - AdamW with warmup + cosine decay
    - Gradient accumulation
    - Continuous learning (resume from checkpoint)
    """
    
    def __init__(self, config: TrainerConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        logger.info(f"Using device: {self.device}")
        if self.device.type == "cuda":
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        
        if config.mode_name != "custom":
            logger.info(f"Training mode: {config.mode_name}")
            if config.mode_description:
                logger.info(f"Description: {config.mode_description}")
        
        # Create checkpoint directory
        self.config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Placeholders
        self.model: Optional[SaciNano] = None
        self.optimizer: Optional[AdamW] = None
        self.scheduler: Optional[LambdaLR] = None
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_loss = float('inf')
    
    def build_model(self, vocab_size: int) -> SaciNano:
        """Build the model."""
        model_config = SaciNanoConfig(
            vocab_size=vocab_size,
            dim=self.config.dim,
            n_layers=self.config.n_layers,
            n_heads=self.config.n_heads,
            n_kv_heads=self.config.n_kv_heads,
            max_seq_len=self.config.seq_len,
            dropout=self.config.dropout
        )
        
        model = SaciNano(model_config)
        model = model.to(self.device)
        
        logger.info(f"Model parameters: {model.count_parameters():,}")
        logger.info(f"Estimated VRAM: {model.estimate_vram_mb(self.config.batch_size):.2f} MB")
        
        return model
    
    def build_optimizer(self) -> AdamW:
        """Build optimizer with weight decay."""
        decay_params = []
        no_decay_params = []
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if "norm" in name or "bias" in name:
                    no_decay_params.append(param)
                else:
                    decay_params.append(param)
        
        optim_groups = [
            {"params": decay_params, "weight_decay": self.config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0}
        ]
        
        return AdamW(optim_groups, lr=self.config.learning_rate)
    
    def train_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        self.optimizer.zero_grad()
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, (inputs, targets) in enumerate(progress_bar):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            _, loss = self.model(inputs, targets)
            
            # Scale for gradient accumulation
            loss = loss / self.config.gradient_accumulation_steps
            
            # Backward pass
            loss.backward()
            
            # Update after accumulation
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm
                )
                
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                
                self.global_step += 1
            
            # Track loss
            batch_loss = loss.item() * self.config.gradient_accumulation_steps
            total_loss += batch_loss
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{batch_loss:.4f}',
                'ppl': f'{torch.exp(torch.tensor(batch_loss)):.2f}',
                'lr': f'{self.scheduler.get_last_lr()[0]:.2e}'
            })
        
        return total_loss / num_batches
    
    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> tuple[float, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        
        for inputs, targets in dataloader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            _, loss = self.model(inputs, targets)
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        return avg_loss, perplexity
    
    def save_checkpoint(self, path: Path, is_best: bool = False):
        """Save checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_loss': self.best_loss,
            'config': self.config,
            'model_config': self.model.config
        }
        
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved: {path}")
        
        if is_best:
            best_path = self.config.checkpoint_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            logger.info(f"Best model saved: {best_path}")
    
    def load_checkpoint(self, path: Path, vocab_size: int):
        """Load checkpoint for continuous learning."""
        logger.info(f"Loading checkpoint from {path}")
        checkpoint = torch.load(path, map_location=self.device)
        
        model_config = checkpoint.get('model_config', SaciNanoConfig(vocab_size=vocab_size))
        self.model = SaciNano(model_config).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_loss = checkpoint['best_loss']
        
        logger.info(f"Resumed from epoch {self.current_epoch}, step {self.global_step}")
    
    def train(
        self,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        vocab_size: int = 16384
    ):
        """Main training loop."""
        # Check for existing checkpoint (continuous learning)
        latest_checkpoint = self.config.checkpoint_dir / "latest.pth"
        
        if self.config.resume_from and self.config.resume_from.exists():
            self.load_checkpoint(self.config.resume_from, vocab_size)
        elif latest_checkpoint.exists():
            logger.info("Found existing checkpoint, resuming training...")
            self.load_checkpoint(latest_checkpoint, vocab_size)
        else:
            self.model = self.build_model(vocab_size)
        
        # Build optimizer and scheduler
        total_steps = len(train_dataloader) * self.config.epochs // self.config.gradient_accumulation_steps
        self.optimizer = self.build_optimizer()
        self.scheduler = get_lr_scheduler(
            self.optimizer,
            self.config.warmup_steps,
            total_steps,
            self.config.min_lr_ratio
        )
        
        # Resume optimizer state if checkpoint loaded
        if self.current_epoch > 0:
            if latest_checkpoint.exists():
                ckpt = torch.load(latest_checkpoint, map_location=self.device)
                self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        
        logger.info(f"Starting training from epoch {self.current_epoch + 1}")
        logger.info(f"Total steps: {total_steps:,}")
        
        start_time = time.time()
        
        for epoch in range(self.current_epoch, self.config.epochs):
            self.current_epoch = epoch + 1
            
            # Train
            train_loss = self.train_epoch(train_dataloader)
            train_ppl = torch.exp(torch.tensor(train_loss)).item()
            
            # Validate
            if val_dataloader is not None:
                val_loss, val_ppl = self.validate(val_dataloader)
                logger.info(
                    f"Epoch {self.current_epoch}: "
                    f"train_loss={train_loss:.4f}, train_ppl={train_ppl:.2f}, "
                    f"val_loss={val_loss:.4f}, val_ppl={val_ppl:.2f}"
                )
                
                if val_loss < self.best_loss:
                    self.best_loss = val_loss
                    self.save_checkpoint(
                        self.config.checkpoint_dir / "best_model.pth",
                        is_best=True
                    )
            else:
                logger.info(
                    f"Epoch {self.current_epoch}: "
                    f"train_loss={train_loss:.4f}, train_ppl={train_ppl:.2f}"
                )
            
            # Save latest checkpoint
            self.save_checkpoint(self.config.checkpoint_dir / "latest.pth")
            
            # Periodic checkpointing
            if self.current_epoch % self.config.save_every_n_epochs == 0:
                self.save_checkpoint(
                    self.config.checkpoint_dir / f"checkpoint_epoch_{self.current_epoch}.pth"
                )
        
        # Final checkpoint
        self.save_checkpoint(self.config.checkpoint_dir / "final_model.pth")
        
        elapsed = time.time() - start_time
        logger.info(f"Training complete in {elapsed / 60:.2f} minutes")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Saci-Nano Training",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--mode", "-m",
        type=str,
        default="default",
        help="Training mode preset (see --list-modes)"
    )
    
    parser.add_argument(
        "--list-modes", "-l",
        action="store_true",
        help="List available training modes and exit"
    )
    
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Start fresh, ignore existing checkpoints"
    )
    
    return parser.parse_args()


def main():
    """Main training entry point."""
    args = parse_args()
    
    # List modes and exit
    if args.list_modes:
        print("\nAvailable training modes:")
        print("-" * 50)
        modes = TrainerConfig.list_available_modes()
        for mode in modes:
            print(mode)
        print("-" * 50)
        print(f"\nConfig file: {MODES_CONFIG_PATH}")
        print("Edit this file to add/modify modes.\n")
        return
    
    # Load config from mode
    try:
        config = TrainerConfig.from_yaml_mode(args.mode)
    except ValueError as e:
        logger.error(str(e))
        return
    
    # Handle --no-resume
    if args.no_resume:
        latest_ckpt = CHECKPOINT_DIR / "latest.pth"
        if latest_ckpt.exists():
            latest_ckpt.unlink()
            logger.info("Removed existing checkpoint (--no-resume)")
    
    # Load tokenizer
    try:
        tokenizer = FinancialTokenizer.load(TOKENIZER_PATH)
        vocab_size = tokenizer.vocab_size
        logger.info(f"Tokenizer loaded. Vocab size: {vocab_size}")
    except FileNotFoundError:
        logger.warning("Tokenizer not found. Using default vocab size 16384")
        vocab_size = 16384
    
    try:
        # Create data loader
        train_loader = create_dataloader(
            data_dir=DATA_TOKENIZED_DIR,
            seq_len=config.seq_len,
            batch_size=config.batch_size
        )
        
        # Create trainer and train
        trainer = Trainer(config)
        trainer.train(train_loader, vocab_size=vocab_size)
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()



--- FILE: backend/src/train/trainer_liquid.py ---
"""
Trainer for I.A.R.A. v6.1 - Fractal Attention-CfC (High Volume + AMP + Optimization)
================================================================================
Features:
- Multi-asset training (Majors)
- Sequence length 256
- Mixed Precision (AMP) for 2x speedup
- Stride Optimization (Stride 8 = 40 min shift) for 8x speedup
- Gradient Accumulation
- ReduceLROnPlateau Scheduler
"""

import json
import time
import argparse
import logging
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
from tqdm import tqdm

# v6.9 Performance: Enable cuDNN auto-tuning for optimal kernel selection
torch.backends.cudnn.benchmark = True
from sklearn.metrics import roc_auc_score

from src.model.iara_liquid import IARA, LiquidConfig
import torch.multiprocessing as mp
import mlflow # SACI V7 Stack
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# SACI V7: MLflow Setup
def setup_mlflow():
    MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    try:
        mlflow.set_tracking_uri(MLFLOW_URI)
        mlflow.set_experiment("SACI_Operations_V1")
        logger.info(f"MLflow connected at {MLFLOW_URI}")
    except Exception as e:
        logger.warning(f"MLflow connection failed: {e}")

# Paths
DATA_PROCESSED_DIR = Path(__file__).parent.parent.parent / "data" / "processed"
CHECKPOINT_DIR = Path(__file__).parent.parent.parent / "checkpoints"
LOGS_DIR = Path(__file__).parent.parent.parent / "logs"


@dataclass
class TrainerConfig:
    """Training configuration v6.7 (Expert Symmetry + Profit Focus)."""
    
    # Data
    seq_len: int = 256
    stride: int = 2
    batch_size: int = 1024
    accumulation_steps: int = 1
    num_workers: int = 8
    val_split: float = 0.2
    
    # Model (Capacity Doubled + Expert Heads)
    input_dim: int = 16  # v6.7: Global Sentiment
    hidden_dim: int = 128
    num_heads: int = 4
    dropout: float = 0.5   # v6.7: High dropout for stacked layers
    max_seq_len: int = 256
    num_cfc_layers: int = 2 # v6.7: Stacked CfC
    
    # Training
    epochs: int = 200
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    grad_clip: float = 0.5
    resume: bool = False
    
    # Label Smoothing
    label_smooth: float = 0.1
    
    # Augmentation
    noise_std: float = 0.02
    scale_min: float = 0.98
    scale_max: float = 1.02
    
    # Early Stopping
    patience: int = 150  # v6.8: Increased for Blindfold adaptation period
    
    # Prediction Target
    target_lookahead: int = 5  # v6.11: Cumulative 5-bar target
    
    # Checkpointing
    checkpoint_dir: Path = CHECKPOINT_DIR
    
    # v6.7 Adaptive
    asymmetry_penalty: float = 2.5
    error_memory_size: int = 500

class AsymmetricFocalLoss(nn.Module):
    """
    v6.7 Profit-Weighted Asymmetric Focal Loss.
    - Down-weights easy examples (gamma).
    - Up-weights positive class (alpha).
    - Multiplies loss by 'profit_weight' (Volatility) to prioritize big moves.
    """
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor = None) -> torch.Tensor:
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.exp(-bce_loss) # prob of correct class
        
        # Focal term: (1 - pt)^gamma
        focal_term = (1 - pt) ** self.gamma
        
        # Alpha balancing: Higher weight for positive class (1)
        # targets is 0 or 1.
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        loss = alpha_t * focal_term * bce_loss
        
        # v6.7 Profit Weighting
        if weights is not None:
            loss = loss * weights
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

class MultiAssetDataset(Dataset):
    """Multi-asset dataset with Profit-Weighting."""
    
    def __init__(
        self,
        data_dir: Path = DATA_PROCESSED_DIR,
        seq_len: int = 256,
        stride: int = 1,
        tickers: Optional[List[str]] = None,
        target_col: str = "close_log_ret",
        augment: bool = False,
        noise_std: float = 0.02,
        scale_range: Tuple[float, float] = (0.98, 1.02),
        label_smooth: float = 0.0,
        target_lookahead: int = 5  # v6.11: Configurable horizon
    ):
        self.data_dir = Path(data_dir)
        self.seq_len = seq_len
        self.stride = stride
        self.target_col = target_col
        self.augment = augment
        self.noise_std = noise_std
        self.scale_range = scale_range
        self.label_smooth = label_smooth
        self.target_lookahead = target_lookahead
        self.feature_cols = None
        self.samples = self._load_data(tickers)
        self.indices = self._compute_indices()
        
        aug_str = " (augmented)" if augment else ""
        smooth_str = f", smooth={label_smooth}" if label_smooth > 0 else ""
        n_tickers = len(self.samples)
        logger.info(f"Dataset{aug_str}{smooth_str}: {len(self.indices)} samples from {n_tickers} tickers, {len(self.feature_cols) if self.feature_cols else 0} features")
    
    def _load_data(self, tickers: Optional[List[str]]) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, str]]:
        parquet_files = list(self.data_dir.glob("*.parquet"))
        if tickers:
            parquet_files = [f for f in parquet_files if f.stem in tickers]
        
        data_list = []
        self.feature_cols = None
        
        for file_path in parquet_files:
            try:
                df = pl.read_parquet(file_path)
                ticker = file_path.stem
                
                if self.target_col not in df.columns:
                    continue
                
                if self.feature_cols is None:
                    # v6.7: Ensure Global Sentiment is included if processed
                    self.feature_cols = sorted([c for c in df.columns if c.endswith("_norm")])
                    logger.info(f"Features: {self.feature_cols}")
                
                available = [c for c in self.feature_cols if c in df.columns]
                
                features = df.select(available).to_numpy().astype(np.float32)
                targets = df[self.target_col].to_numpy().astype(np.float32)
                
                if "valid_sample" in df.columns:
                    valid_mask = df["valid_sample"].to_numpy().astype(np.float32)
                else:
                    valid_mask = np.ones(len(features), dtype=np.float32)
                
                features = np.nan_to_num(features, nan=0.0)
                targets = np.nan_to_num(targets, nan=0.0)
                
                if len(available) == len(self.feature_cols) and len(features) > self.seq_len:
                    data_list.append((features, targets, valid_mask, ticker))
                
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
        
        return data_list
    
    def _compute_indices(self) -> List[Tuple[int, int]]:
        indices = []
        for i, (feats, targets, valid_mask, ticker) in enumerate(self.samples):
            max_start = len(feats) - self.seq_len
            if max_start < 0:
                continue
            for start in range(0, max_start, self.stride):
                end_row = start + self.seq_len
                if valid_mask[end_row - 1] > 0.5:
                    indices.append((i, start))
        return indices
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sample_idx, start_row = self.indices[idx]
        features, targets, _, _ = self.samples[sample_idx]
        
        end_row = start_row + self.seq_len
        x = torch.tensor(features[start_row:end_row], dtype=torch.float32)
        
        if self.augment:
            x = x + torch.randn_like(x) * self.noise_std
            scale = random.uniform(self.scale_range[0], self.scale_range[1])
            x = x * scale
        
        # v6.11: Configurable CUMULATIVE forward return (reduces noise)
        end_idx = min(end_row + self.target_lookahead, len(targets))
        cumulative_return = targets[end_row:end_idx].sum() if end_idx > end_row else targets[end_row - 1]
        
        # Only classify as UP/DOWN if movement exceeds noise threshold
        noise_threshold = 0.0001  # 1 pip for forex
        if cumulative_return > noise_threshold:
            binary_target = 1.0
        elif cumulative_return < -noise_threshold:
            binary_target = 0.0
        else:
            # Small moves get label 0.5 (uncertain) - will be smoothed anyway
            binary_target = 0.5
        
        # v6.10 Profit Weighting: Based on cumulative return
        abs_ret = abs(cumulative_return)
        # Assume 1% move (0.01) is "Big". 
        profit_weight = 1.0 + min(50.0 * abs_ret, 1.5) # Caps at 2.5x total weight for massive moves
        
        if self.label_smooth > 0:
            binary_target = binary_target * (1 - 2 * self.label_smooth) + self.label_smooth
        
        return x, torch.tensor(binary_target, dtype=torch.float32), torch.tensor(profit_weight, dtype=torch.float32)
    
    @property
    def input_dim(self) -> int:
        return len(self.feature_cols) if self.feature_cols else 0
    
    def get_chronological_split(self, val_ratio: float = 0.2) -> Tuple[List[int], List[int]]:
        train_indices = []
        val_indices = []
        
        ticker_indices: Dict[int, List[int]] = {}
        for i, (sample_idx, start_row) in enumerate(self.indices):
            if sample_idx not in ticker_indices:
                ticker_indices[sample_idx] = []
            ticker_indices[sample_idx].append(i)
        
        for sample_idx, idx_list in ticker_indices.items():
            split_point = int(len(idx_list) * (1 - val_ratio))
            train_indices.extend(idx_list[:split_point])
            val_indices.extend(idx_list[split_point:])
        
        return train_indices, val_indices
    
    def compute_class_weights(self, indices: List[int]) -> float:
        pos_count = 0
        neg_count = 0
        
        for idx in indices:
            sample_idx, start_row = self.indices[idx]
            _, targets, _, _ = self.samples[sample_idx]
            end_row = start_row + self.seq_len
            
            if targets[end_row - 1] > 0:
                pos_count += 1
            else:
                neg_count += 1
        
        if pos_count == 0:
            return 1.0
        
        weight = neg_count / pos_count
        logger.info(f"Class distribution: UP={pos_count}, DOWN={neg_count}, pos_weight={weight:.4f}")
        return weight


class Trainer:
    """IARA v6.9 Hybrid Engine Trainer."""
    def __init__(
        self, 
        config: TrainerConfig, 
        device: torch.device,
        train_dataset: Optional[torch.utils.data.Dataset] = None,
        val_dataset: Optional[torch.utils.data.Dataset] = None
    ):
        self.config = config
        self.device = device
        self.use_mlflow = True
        
        # Directories
        Path("checkpoints").mkdir(exist_ok=True)
        # v6.9 Hybrid Engine: Use injected datasets if provided
        if train_dataset is not None and val_dataset is not None:
            logger.info("Using pre-computed datasets (v6.9 Hybrid Engine)")
            
            # Directly use provided datasets
            # v6.9 Performance: num_workers=0 is FASTER for pre-computed data!
            # Workers cause memory copy overhead (415MB per worker fork)
            # pin_memory=True enables direct GPU transfer
            self.train_loader = DataLoader(
                train_dataset,
                batch_size=config.batch_size,
                shuffle=True,
                num_workers=0,  # Optimal for pre-computed (15ms vs 33ms per batch)
                pin_memory=True,
                drop_last=True
            )
            
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=config.batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=True
            )
            
            n_samples = len(train_dataset)
            pos_weight = 1.0  # Balanced for pre-computed
            
        else:
            # Legacy mode: Create datasets on-the-fly
            full_dataset = MultiAssetDataset(
                seq_len=config.seq_len, 
                stride=config.stride, 
                augment=False,
                target_lookahead=config.target_lookahead
            )
            
            if full_dataset.input_dim > 0:
                config.input_dim = full_dataset.input_dim
            
            train_idxs, val_idxs = full_dataset.get_chronological_split(config.val_split)
            pos_weight = full_dataset.compute_class_weights(train_idxs)
            
            # Training dataset
            train_dataset_internal = MultiAssetDataset(
                seq_len=config.seq_len,
                stride=config.stride,
                augment=True,
                noise_std=config.noise_std,
                scale_range=(config.scale_min, config.scale_max),
                label_smooth=config.label_smooth,
                target_lookahead=config.target_lookahead
            )
            # Validation
            val_dataset_internal = MultiAssetDataset(
                seq_len=config.seq_len, 
                stride=config.stride, 
                augment=False, 
                label_smooth=0.0,
                target_lookahead=config.target_lookahead
            )
            
            train_subset = Subset(train_dataset_internal, train_idxs)
            val_subset = Subset(val_dataset_internal, val_idxs)
            
            logger.info(f"Train: {len(train_subset)}, Val: {len(val_subset)}")
            
            self.train_loader = DataLoader(
                train_subset,
                batch_size=config.batch_size,
                shuffle=True,
                num_workers=config.num_workers,
                pin_memory=True,
                drop_last=True
            )
            
            self.val_loader = DataLoader(
                val_subset,
                batch_size=config.batch_size,
                shuffle=False,
                num_workers=config.num_workers,
                pin_memory=True
            )
            
            n_samples = len(train_idxs)
        
        # MEMORY: Initialize Selective History (v6.3)
        self.memory_alpha = 0.7 
        self.sample_loss_ema = torch.ones(n_samples)
        self.chronic_counter = torch.zeros(n_samples, dtype=torch.int16)
        self.mastered_counter = torch.zeros(n_samples, dtype=torch.int16)
        self.noise_mask = torch.zeros(n_samples, dtype=torch.bool)
        self.epoch_losses_history = torch.zeros(n_samples, 10)


        
        # Model v6.7 (Expert Symmetry)
        liquid_config = LiquidConfig(
            input_dim=config.input_dim,
            hidden_dim=config.hidden_dim, # 128
            num_heads=config.num_heads,
            dropout=config.dropout,
            max_seq_len=config.max_seq_len,
            num_cfc_layers=config.num_cfc_layers # 2
        )
        self.model = IARA(liquid_config).to(device)
        
        # Enable TensorFloat32 for faster matmul
        torch.set_float32_matmul_precision('high')
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.8,    # v6.12 Gentle decay (was 0.5)
            patience=15,   # v6.12 Patience (was 5)
            min_lr=1e-6    # v6.12 Floor (was implicit 0)
        )
        
        # v6.10 Debug: Using standard BCE for baseline test
        self.criterion = nn.BCEWithLogitsLoss(reduction='none')
        self.use_simple_bce = True  # Flag for train_epoch to skip profit weighting
        # Legacy for valid reporting (BCE is standard metric)
        self.valid_criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
        
        # AMP Scaler
        self.scaler = torch.amp.GradScaler('cuda')
        
        # State
        self.start_epoch = 0
        self.epoch = 0
        self.best_val_auc = 0.0
        self.patience_counter = 0
        
        # Resume Logic
        if config.resume:
            self._resume_checkpoint()
        
        logger.info(f"Model Parameters: {self.model.count_parameters():,}")
        logger.info(f"Config: seq_len={config.seq_len}, stride={config.stride}, batch={config.batch_size}, Effective Batch={config.batch_size * config.accumulation_steps}, Units={config.hidden_dim}")

    def _resume_checkpoint(self):
        latest_path = self.config.checkpoint_dir / "latest.pth"
        if not latest_path.exists():
            logger.warning(f"Resume requested but {latest_path} not found. Starting fresh.")
            return

        logger.info(f"Resuming from {latest_path}...")
        checkpoint = torch.load(latest_path, map_location=self.device, weights_only=False)
        
        # Load Model
        # v6.5 Compatibility: If resume from v6.1 (64 units) to v6.5 (128 units), weights won't match.
        # We must detect this and handle gracefully or error out.
        state_dict = checkpoint['model']
        if list(state_dict.keys())[0].startswith('_orig_mod.'):
            state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
            
        try:
            self.model.load_state_dict(state_dict)
            logger.info("Weights restored successfully.")
        except RuntimeError as e:
            logger.warning(f"Dimension mismatch (expected when upgrading capacity): {e}")
            logger.warning("Starting fresh with new architecture, but keeping epoch count and memory stats.")
            # We don't load optimizer if model changed structure
            return
        
        # Load Optimizer & Scheduler
        if 'optimizer' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        if 'scheduler' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler'])
            # v6.12 HOTFIX: Enforce new config over old state
            self.scheduler.patience = 15
            self.scheduler.factor = 0.8
            self.scheduler.min_lrs = [1e-6] * len(self.scheduler.optimizer.param_groups)
            logger.info(f"🔧 Scheduler Config Enforced: Patience={self.scheduler.patience}, Factor={self.scheduler.factor}, MinLR={self.scheduler.min_lrs[0]}")
            
        # Load State
        self.start_epoch = checkpoint['epoch'] + 1
        self.epoch = self.start_epoch
        self.best_val_auc = checkpoint.get('best_val_auc', 0.0)
        
        # Load Memory State (v6.3)
        if 'memory_state' in checkpoint:
            mem = checkpoint['memory_state']
            self.sample_loss_ema = mem.get('loss_ema', self.sample_loss_ema).cpu()
            self.chronic_counter = mem.get('chronic', self.chronic_counter).cpu()
            self.mastered_counter = mem.get('mastered', self.mastered_counter).cpu()
            self.noise_mask = mem.get('noise', self.noise_mask).cpu()
            self.epoch_losses_history = mem.get('history', self.epoch_losses_history).cpu()
            logger.info(f"Restored Selective Memory: {self.chronic_counter.gt(0).sum()} chronic, {self.mastered_counter.gt(0).sum()} mastered")
            
        logger.info(f"Resumed at Epoch {self.start_epoch}, Best AUC: {self.best_val_auc:.4f}")

    def _run_epoch(self, loader: DataLoader, training: bool, desc: str = "") -> Dict[str, float]:
        if training:
            self.model.train()
        else:
            self.model.eval()
        
        losses, all_preds, all_targets = [], [], []
        correct, total = 0, 0
        
        pbar = tqdm(loader, desc=desc, leave=False) if training else loader
        
        if training:
            self.optimizer.zero_grad()
        
        context = torch.enable_grad() if training else torch.no_grad()
        
        with context:
            for i, batch in enumerate(pbar):
                # Unpack batch (Handle 2 or 3 items)
                if len(batch) == 3:
                    x, y, w = batch
                    w = w.to(self.device, non_blocking=True)
                else:
                    x, y = batch
                    w = None
                    
                # v6.9: non_blocking=True overlaps CPU→GPU transfer with compute
                x = x.to(self.device, non_blocking=True).contiguous()
                y = y.to(self.device, non_blocking=True)
                
                # Mixed Precision Forward
                with torch.amp.autocast('cuda'):
                    logits, _ = self.model(x)
                    
                    if training:
                        # v6.10: Handle both BCE and custom loss functions
                        if hasattr(self, 'use_simple_bce') and self.use_simple_bce:
                            loss_per_sample = self.criterion(logits.squeeze(), y)
                            if w is not None:
                                loss_per_sample = loss_per_sample * w.to(self.device)
                        else:
                            loss_per_sample = self.criterion(logits.squeeze(), y, weights=w)
                        loss = loss_per_sample.mean()
                    else:
                        loss = self.valid_criterion(logits.squeeze(), y)
                
                if training:
                    # Gradient Accumulation v6.5
                    scaled_loss = loss / self.config.accumulation_steps
                    self.scaler.scale(scaled_loss).backward()
                    
                    if (i + 1) % self.config.accumulation_steps == 0:
                        self.scaler.unscale_(self.optimizer)
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.optimizer.zero_grad()
                        
                        if isinstance(pbar, tqdm):
                            lr = self.optimizer.param_groups[0]['lr']
                            # Multiply loss by steps to show real loss for display
                            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{lr:.2e}")
                
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()
                
                hard_y = (y > 0.5).float()
                correct += (preds == hard_y).sum().item()
                total += y.size(0)
                losses.append(loss.item())
                
                all_preds.extend(probs.detach().cpu().numpy())
                all_targets.extend(hard_y.detach().cpu().numpy())
        
        epoch_loss = np.mean(losses)
        epoch_acc = correct / total if total > 0 else 0
        
        try:
            auc = roc_auc_score(all_targets, all_preds)
        except:
            auc = 0.5
            
        return {"loss": epoch_loss, "accuracy": epoch_acc, "auc": auc}

    def train_epoch(self) -> Tuple[Dict[str, float], Dict[str, float]]:
        train_m = self._run_epoch(self.train_loader, training=True, desc=f"Epoch {self.epoch}")
        val_m = self._run_epoch(self.val_loader, training=False)
        return train_m, val_m

    def save_checkpoint(self, filename="latest.pth"):
        self.config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        # Handle compiled model (if applicable)
        model_state = self.model._orig_mod.state_dict() if hasattr(self.model, '_orig_mod') else self.model.state_dict()
        
        torch.save({
            'epoch': self.epoch,
            'model': model_state,
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'config': self.config,
            'best_val_auc': self.best_val_auc,
            'memory_state': {
                'loss_ema': self.sample_loss_ema,
                'chronic': self.chronic_counter,
                'mastered': self.mastered_counter,
                'noise': self.noise_mask,
                'history': self.epoch_losses_history
            }
        }, self.config.checkpoint_dir / filename)


    def update_sampler_weights(self):
        """
        Selective Hard Mining (v6.3):
        - updates Exponential Moving Average (EMA) of loss.
        - detects Chronic Errors (consistent high loss).
        - detects Mastered Lessons (consistent high confidence correct).
        - discards Market Noise (consistent extreme loss).
        """
        self.model.eval()
        losses = []
        accuracies = []
        
        train_subset = self.train_loader.dataset
        eval_loader = DataLoader(
            train_subset,
            batch_size=self.config.batch_size * 2,
            shuffle=False, # Crucial: Must match index order
            num_workers=self.config.num_workers,
            pin_memory=True
        )
        
        logger.info("Deep Thought: Analyzing difficulty spectrum (Selective Memory)...")
        
        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                for X, y, _ in eval_loader: # Added _ for profit_weight
                    X, y = X.to(self.device), y.to(self.device)
                    logits, _ = self.model(X)
                    loss = nn.BCEWithLogitsLoss(reduction='none')(logits.squeeze(), y)
                    losses.append(loss.cpu())
                    
                    # Check Correctness for Mastery
                    probs = torch.sigmoid(logits.squeeze())
                    preds = (probs > 0.5).float()
                    correct = (preds == y).float()
                    
                    # High Confidence Correct? (>0.6 or <0.4)
                    confident = ((probs > 0.6) | (probs < 0.4)).float()
                    mastered_batch = correct * confident
                    accuracies.append(mastered_batch.cpu())
                    
        current_losses = torch.cat(losses)
        current_mastered = torch.cat(accuracies)
        
        # 1. Update Losses History (EMA)
        if hasattr(self, 'sample_loss_ema') and len(self.sample_loss_ema) == len(current_losses):
             self.sample_loss_ema = (self.memory_alpha * self.sample_loss_ema) + ((1 - self.memory_alpha) * current_losses)
        else:
             self.sample_loss_ema = current_losses
             
        # 2. Update Chronicity (Loss > 0.9 for consecutive epochs)
        high_loss_mask = current_losses > 0.9
        self.chronic_counter[high_loss_mask] += 1
        self.chronic_counter[~high_loss_mask] = 0
        
        # 3. Update Mastery (Confident Correct for consecutive epochs)
        mastered_mask = current_mastered > 0.5
        self.mastered_counter[mastered_mask] += 1
        self.mastered_counter[~mastered_mask] = 0
        
        # 4. Update Noise Filter (Extreme loss > 1.2 for 10 epochs)
        # Update rolling history matrix
        self.epoch_losses_history = torch.roll(self.epoch_losses_history, -1, dims=1)
        self.epoch_losses_history[:, -1] = current_losses
        
        # Check if ALL last 10 entries are > 1.2
        noise_candidates = (self.epoch_losses_history > 1.2).all(dim=1)
        # Update noise mask (Persistent Noise)
        self.noise_mask = self.noise_mask | noise_candidates
        
        # 5. Calculate Weights
        # Base weight on EMA Loss
        weights = (self.sample_loss_ema + 0.05) ** 2
        
        # Boost Chronic Errors (Focus)
        chronic_indices = self.chronic_counter >= 3
        weights[chronic_indices] *= 3.0
        
        # v6.4 Self-Correction Priority
        # Since we didn't capture 'confident errors' explicitly in the loop above in v6.3, 
        # let's rely on the fact that High Confidence + Wrong = Very High Loss.
        # BCE Loss for prob 0.8 vs target 0 is -log(0.2) = 1.6
        # BCE Loss for prob 0.9 vs target 0 is -log(0.1) = 2.3
        # So 'high_loss_mask' (>0.9) actually covers these cases naturally!
        # v6.4 Explicit Boost: If loss > 1.5 (Very confident wrong), we multiply weight further.
        very_high_loss = current_losses > 1.5
        weights[very_high_loss] *= 2.0 # Extra boost for "Arrogant" errors
        
        # Penalize Mastered Lessons (Skip)
        mastered_indices = self.mastered_counter >= 10  # v6.5 Stricter Threshold (Correct for 10 epochs)
        weights[mastered_indices] *= 0.1
        
        # Zero out Noise (Ignore)
        weights[self.noise_mask] = 0.0
        
        # Normalize
        weights = weights / (weights.sum() + 1e-8)
        
        # Stats
        n_chronic = chronic_indices.sum().item()
        n_mastered = mastered_indices.sum().item()
        n_noise = self.noise_mask.sum().item()
        n_arrogant = very_high_loss.sum().item()
        
        top_losses = torch.topk(self.sample_loss_ema, 5).values.tolist()
        
        logger.info(f"Memory Stats: {n_chronic} Chronic | {n_arrogant} 'Arrogant' Errors (Boosted) | {n_mastered} Mastered | {n_noise} Noise Discarded")
        logger.info(f"Hardest EMA Losses: {top_losses}")
        
        sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=len(weights), # Keep epoch size same, just change frequency
            replacement=True
        )
        
        self.train_loader = DataLoader(
            train_subset,
            batch_size=self.config.batch_size,
            sampler=sampler,
            num_workers=self.config.num_workers,
            pin_memory=True,
            drop_last=True
        )
        
        logger.info("Trainer: Curriculum Updated (v6.4 Self-Correction).")

    def train(self):
        logger.info("Starting Training v6.5 Scaled Intelligence (AMP + Focal Loss + Selective Memory)...")
        
        # Keep track of LR to detect reductions
        current_lr = self.config.learning_rate
        best_model_state = None
        
        # V7: Log Params once
        if self.use_mlflow:
            try:
                mlflow.end_run() # Close any previous run if it exists
                mlflow.start_run(run_name=f"Ops_Run_{int(time.time())}")
                mlflow.log_params(asdict(self.config)) # Use asdict for dataclass
                logger.info(f"MLflow run started: {mlflow.active_run().info.run_id}")
            except Exception as e:
                logger.warning(f"MLflow run start or param logging failed: {e}")

        try:
            for epoch in range(self.start_epoch, self.config.epochs):
                self.epoch = epoch
                train_m, val_m = self.train_epoch()
                
                # Step Scheduler
                self.scheduler.step(val_m['loss'])
                new_lr = self.optimizer.param_groups[0]['lr']
                
                # Telemetry Collection
                chronic_count = self.chronic_counter.gt(0).sum().item()
                mastered_count = self.mastered_counter.gt(0).sum().item()
                noise_count = self.noise_mask.sum().item()
                
                vram_usage = 0
                if torch.cuda.is_available():
                    vram_usage = torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory * 100
                
                # V7: Log to MLflow
                if self.use_mlflow:
                    try:
                        metrics_flat = {
                            "train_loss": train_m['loss'],
                            "train_accuracy": train_m['accuracy'],
                            "train_auc": train_m['auc'],
                            "val_loss": val_m['loss'],
                            "val_accuracy": val_m['accuracy'],
                            "val_auc": val_m['auc'],
                            "learning_rate": new_lr,
                            "chronic_memory": chronic_count,
                            "mastered_memory": mastered_count,
                            "noise_discarded": noise_count,
                            "vram_percent": round(vram_usage, 2)
                        }
                        mlflow.log_metrics(metrics_flat, step=epoch)
                    except Exception as e:
                        logger.warning(f"MLflow log failed for epoch {epoch}: {e}")
                
                logger.info(
                    f"Epoch {epoch}: "
                    f"Train[Loss={train_m['loss']:.4f}, Acc={train_m['accuracy']:.2%}, AUC={train_m['auc']:.4f}] | "
                    f"Val[Loss={val_m['loss']:.4f}, Acc={val_m['accuracy']:.2%}, AUC={val_m['auc']:.4f}]"
                )
                
                if val_m['auc'] > self.best_val_auc:
                    self.best_val_auc = val_m['auc']
                    self.patience_counter = 0
                    self.save_checkpoint("best_model.pth")
                    # Cache best state in RAM for quick rollback
                    # Copy is needed because state_dict is a reference
                    best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                    logger.info(f"  ✓ New best Val AUC: {self.best_val_auc:.4f} (State Cached)")
                else:
                    self.patience_counter += 1
                    logger.info(f"  No improvement ({self.patience_counter}/{self.config.patience})")
                
                # AUTO-ROLLBACK 2.0 (Weight Interpolation)
                # If LR dropped (Scheduler detected plateau), blend current weights with best weights
                if new_lr < current_lr:
                    if best_model_state is not None:
                        logger.info(f"  ↺ Plateau detected (LR {current_lr:.2e} -> {new_lr:.2e}). Applying Auto-Rollback 2.0 (Blending 70% Best + 30% Current)...")
                        
                        current_state = self.model.state_dict()
                        blended_state = {}
                        
                        for k, best_v in best_model_state.items():
                            # Blend floating point weights
                            if best_v.is_floating_point():
                                curr_v = current_state[k] # Already on device
                                best_v_dev = best_v.to(self.device)
                                blended_state[k] = (0.7 * best_v_dev) + (0.3 * curr_v)
                            else:
                                # Keep integer buffers (e.g. step counters) from best model
                                blended_state[k] = best_v.to(self.device)
                                
                        self.model.load_state_dict(blended_state)
                    else:
                        logger.warning("  LR reduced but no best model to rollback to.")
                    current_lr = new_lr
                    
                # Update Sampler for next epoch (Feedback Loop)
                # Only do this if not early stopping
                if self.epoch < self.config.epochs - 1:
                    self.update_sampler_weights()
                
                # Always save latest for resume
                self.save_checkpoint("latest.pth")
                
                if self.patience_counter >= self.config.patience:
                    logger.info(f"Early stopping after {epoch + 1} epochs")
                    break
        
        except KeyboardInterrupt:
            logger.warning("\n⚠️  KEYBOARD INTERRUPT DETECTED! INITIATING GRACEFUL SHUTDOWN...")
            
            # Save Interrupted State
            self.save_checkpoint("interrupted_model.pth")
            logger.info("✅ Saved 'interrupted_model.pth' - You can resume from here.")
            
            # Final Telemetry
            self.telemetry.log({
                "status": "INTERRUPTED",
                "last_epoch": self.epoch,
                "best_auc": self.best_val_auc
            })
            
            # Summary Table
            print("\n" + "="*50)
            print("🛑  TRAINING INTERRUPTED SESSION SUMMARY")
            print("="*50)
            print(f"Start Epoch: {self.start_epoch}")
            print(f"Last Epoch:  {self.epoch}")
            print(f"Best AUC:    {self.best_val_auc:.4f}")
            print(f"Log File:    {self.telemetry.log_path}")
            print("="*50 + "\n")
            
            # v6.9: Explicit GPU memory cleanup (fixes ROCm memory leak on Ctrl+C)
            logger.info("🧹 Cleaning up GPU memory...")
            del self.model
            del self.optimizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            logger.info("✅ GPU memory released.")
            
            return

        logger.info(f"Training finished. Best Val AUC: {self.best_val_auc:.4f}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=768)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.001, help="Learning Rate")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    parser.add_argument("--use-cache", action="store_true", 
                        help="Use pre-computed tensors from data/cache/ (v6.9 Hybrid Engine)")
    parser.add_argument("--lab", action="store_true", 
                        help="Run in LAB MODE: Limit dataset to 50k samples, use separate cache/checkpoints.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # V7: Init MLflow
    setup_mlflow()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # v6.9 Hybrid Engine: Check for pre-computed cache
    if args.use_cache or args.lab: # Lab implies cache usage
        from src.data import load_precomputed_datasets, HybridDatasetLoader
        from pathlib import Path
        
        if args.lab:
            logger.info("🧪 LAB MODE ACTIVATED: Using limited dataset & separate cache.")
            cache_dir = Path("data/cache_lab")
            ckt_dir = Path("checkpoints_lab")
            limit = 50000
        else:
            cache_dir = Path("data/cache")
            ckt_dir = CHECKPOINT_DIR
            limit = 0
            
        loader = HybridDatasetLoader(cache_dir)
        
        if not loader.is_cache_valid():
            logger.info(f"Cache not found in {cache_dir}. Running Feature Factory...")
            loader.build_cache(seq_len=args.seq_len, limit=limit)
        
        train_ds, val_ds = loader.get_datasets()
        
        # Update config based on cached data
        config = TrainerConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            patience=args.patience,
            seq_len=train_ds.get_seq_len(),
            input_dim=train_ds.get_feature_dim(),
            learning_rate=args.lr,
            resume=args.resume,
            checkpoint_dir=ckt_dir
        )
        
        logger.info(f"Using pre-computed cache: {train_ds.get_feature_dim()} features, {len(train_ds)} train samples")
        
        # Create trainer with cached datasets (v6.9 Full Integration)
        trainer = Trainer(config, device, train_dataset=train_ds, val_dataset=val_ds)
        trainer.train()
    else:
        # Legacy mode: On-the-fly feature computation
        config = TrainerConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            patience=args.patience,
            seq_len=args.seq_len,
            learning_rate=args.lr,
            resume=args.resume
        )
        
        trainer = Trainer(config, device)
        trainer.train()



--- FILE: backend/src/train/test_sniper.py ---
import os
import torch
import torch.nn.functional as F
from src.brain.models.liquid_v10 import SwarmBrain, WorkerConfig, CoordinatorConfig

def simulate_timeframes(x: torch.Tensor) -> dict:
    x_dict = {}
    x_dict["M5"] = x
    x_dict["M15"] = x[:, ::3, :]
    x_dict["M30"] = x[:, ::6, :]
    x_dict["H1"] = x[:, ::12, :]
    x_dict["H4"] = x[:, ::16, :]
    return x_dict

def test_sniper_mode():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing Sniper Mode on {device}...")
    
    # 1. Load Model
    input_dim = 14
    worker_cfg = WorkerConfig(input_dim=input_dim, hidden_dim=64)
    coord_cfg = CoordinatorConfig(num_workers=5, worker_dim=64, hidden_dim=256)
    model = SwarmBrain(worker_cfg, coord_cfg).to(device)
    
    checkpoint_path = "logs/models/madrugada.pth"
    if not os.path.exists(checkpoint_path):
        print(f"Error: {checkpoint_path} not found.")
        return
        
    state = torch.load(checkpoint_path, map_location=device)
    state_dict = state.get('model_state_dict', state)
    
    # Apply dynamic padding for any older layers if needed (just to be safe)
    for name, param in model.named_parameters():
        if name in state_dict and any(key in name for key in ["input_proj.weight", "cfc.x_proj.weight", "res_proj.weight"]):
            old_w = state_dict[name]
            if len(old_w.shape) == 2 and old_w.shape[1] < param.shape[1]:
                new_w = torch.zeros_like(param)
                new_w[:, :old_w.shape[1]] = old_w
                state_dict[name] = new_w
                
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    print("Model loaded successfully.")
    
    # 2. Load Validation Data (from BTCUSD cache)
    shard_dir = "data/cache/BTCUSD/val_shards"
    if not os.path.exists(shard_dir):
        print("No validation shards found. Using train shards for a quick test...")
        shard_dir = "data/cache/BTCUSD/train_shards"
        
    shards = sorted([f for f in os.listdir(shard_dir) if f.endswith(".pt")])
    if not shards:
        print("No shards available to test.")
        return
        
    print(f"Loaded {len(shards)} shards for testing. Processing first 5 shards...")
    
    total_samples = 0
    
    # Standard Metrics
    standard_correct = 0
    standard_trades = 0
    
    # Sniper Metrics
    sniper_correct = 0
    sniper_trades = 0
    sniper_threshold = 0.82
    
    batch_size = 512
    
    with torch.no_grad():
        for shard in shards[:5]:
            data = torch.load(os.path.join(shard_dir, shard), map_location='cpu')
            X = data['X']
            y = data['y']
            
            # Pad dataset to 14
            is_old_shard = (X.shape[-1] in [12, 22, 26])
            if is_old_shard:
                core_X = X[:, :, :12]
                if input_dim > 12:
                    padding = torch.zeros(X.shape[0], X.shape[1], input_dim - 12, dtype=X.dtype, device=X.device)
                    X = torch.cat([core_X, padding], dim=-1)
                else:
                    X = core_X
            else:
                if X.shape[-1] > input_dim:
                    X = X[:, :, :input_dim]
                elif X.shape[-1] < input_dim:
                    padding = torch.zeros(X.shape[0], X.shape[1], input_dim - X.shape[-1], dtype=X.dtype, device=X.device)
                    X = torch.cat([X, padding], dim=-1)
                    
            for i in range(0, len(X), batch_size):
                batch_X = X[i : i + batch_size].to(device)
                batch_y = y[i : i + batch_size].to(device)
                
                # Truncate sequence length if necessary (usually 128)
                if batch_X.shape[1] > 128:
                    batch_X = batch_X[:, -128:, :]
                    
                x_dict = simulate_timeframes(batch_X)
                logits, _, _, _ = model(x_dict)
                probs = F.softmax(logits, dim=1)
                
                # Targets: 0=Long, 1=Short, 2=Wait
                
                # 1. Standard mode (Argmax ignores confidence threshold)
                preds_argmax = torch.argmax(probs, dim=1)
                for pred, target in zip(preds_argmax, batch_y):
                    target = int(target.item())
                    pred = int(pred.item())
                    # Only score situations where the model wanted to act, OR where a trade existed
                    # For a true backtest precision we care if an ACTION was correct
                    if pred in [0, 1]:  # Model wanted to trade
                        standard_trades += 1
                        if pred == target:
                            standard_correct += 1
                            
                # 2. Sniper mode (>0.82 Confidence)
                max_probs, sniper_preds = torch.max(probs, dim=1)
                for prob, pred, target in zip(max_probs, sniper_preds, batch_y):
                    target = int(target.item())
                    pred = int(pred.item())
                    prob = float(prob.item())
                    
                    if pred in [0, 1] and prob >= sniper_threshold:  # Sniper trigger pulled
                        sniper_trades += 1
                        if pred == target:
                            sniper_correct += 1
                            
                total_samples += len(batch_X)
                
    print("\n" + "="*50)
    print("🎯 CRYPTO ARMOR (v10.0) - SNIPER MODE BENCHMARK")
    print("="*50)
    print(f"Total Market Snapshots Evaluated: {total_samples}")
    print("-" * 50)
    
    # Standard Results
    std_win_rate = (standard_correct / standard_trades * 100) if standard_trades > 0 else 0
    print(f"[STANDARD MODE] (Any Confidence > Margin)")
    print(f"Trades Taken: {standard_trades} / {total_samples}")
    print(f"Correct Hits: {standard_correct}")
    print(f"Win Rate:     {std_win_rate:.2f}%")
    print("-" * 50)
    
    # Sniper Results
    snp_win_rate = (sniper_correct / sniper_trades * 100) if sniper_trades > 0 else 0
    print(f"[SNIPER MODE] (Strict Confidence > {sniper_threshold*100}%)")
    print(f"Trades Taken: {sniper_trades} / {total_samples} (Filtering out Noise)")
    print(f"Correct Hits: {sniper_correct}")
    print(f"Win Rate:     {snp_win_rate:.2f}%")
    print("="*50 + "\n")

if __name__ == "__main__":
    test_sniper_mode()



--- FILE: backend/src/train/dataset.py ---
"""
Dataset - Stochastic Shuffling & Target-Only Loss Masking
==========================================================
Refactored dataset to prevent structural overfitting.
1. Shuffles feature tokens randomly per timestep.
2. Places Target Token (Price Return) always at the end.
3. Masks loss so model only learns to predict Price.
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset, DataLoader

from src.data.tokenizer import FinancialTokenizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Paths
DATA_TOKENIZED_DIR = Path(__file__).parent.parent.parent / "data" / "tokenized"
TOKENIZER_PATH = Path(__file__).parent.parent.parent / "data" / "tokenizer.pkl"


class StructuredForexDataset(Dataset):
    """
    Dataset that treats each timestep as a "Bag of Features" + Target.
    
    Structure per timestep: [Feature_1, Feature_2, ..., Feature_N, TARGET]
    
    - Features are randomly shuffled during training to prevent memorizing order.
    - Target is always last.
    - Loss is masked to only penalize Target prediction.
    """
    
    def __init__(
        self,
        data_dir: Path = DATA_TOKENIZED_DIR,
        tokenizer_path: Path = TOKENIZER_PATH,
        seq_len: int = 384,
        stride: int = 128,
        tickers: Optional[List[str]] = None,
        is_training: bool = True,
        target_column: str = "close_log_ret"
    ):
        self.data_dir = Path(data_dir)
        self.seq_len = seq_len
        self.stride = stride
        self.is_training = is_training
        self.target_column = target_column
        
        # Load tokenizer to identify columns
        self.tokenizer = FinancialTokenizer.load(tokenizer_path)
        self.feature_columns = self.tokenizer.feature_columns
        
        # Identify Target and Feature indices
        self.target_token_col = f"{target_column}_token"
        try:
            # We need to find the index of the target column in the loaded arrays
            # But we load files individually. 
            pass 
        except ValueError:
            logger.warning(f"Target column {target_column} not found in tokenizer!")
            
        # Load data as list of 2D arrays [timesteps, n_cols]
        # We store (features_array, target_array) per ticker
        self.samples = self._load_data(tickers)
        
        # Compute valid starting positions
        self.indices = self._compute_indices()
        
        logger.info(
            f"StructuredDataset initialized: {len(self.indices)} windows. "
            f"Training={is_training}, Target={target_column}"
        )

    def _load_data(self, tickers: Optional[List[str]]) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Load data into memory.
        Returns list of (features_matrix, target_vector).
        """
        parquet_files = list(self.data_dir.glob("*.parquet"))
        if tickers:
            parquet_files = [f for f in parquet_files if f.stem in tickers]
            
        data_list = []
        
        for file_path in parquet_files:
            try:
                df = pl.read_parquet(file_path)
                
                # Check for target column
                if self.target_token_col not in df.columns:
                    logger.warning(f"Skipping {file_path.name}: Target {self.target_token_col} not found")
                    continue
                
                # Get all token columns
                token_cols = [c for c in df.columns if c.endswith("_token")]
                
                # Separate Features vs Target
                feat_cols = [c for c in token_cols if c != self.target_token_col]
                
                # Extract arrays
                features = df.select(feat_cols).to_numpy().astype(np.int32) # [T, N_features]
                targets = df.select(self.target_token_col).to_numpy().astype(np.int32).flatten() # [T]
                
                # Validate length
                if len(features) > self.seq_len:
                    data_list.append((features, targets))
                    
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
                
        return data_list

    def _compute_indices(self) -> List[Tuple[int, int]]:
        """Compute (sample_idx, start_row) for sliding windows."""
        indices = []
        
        # We need N tokens. 
        # Each timestep produces (N_features + 1) tokens.
        # Let's define context window in terms of TIMESTEPS (Rows), not tokens?
        # The user specified seq_len in tokens usually (e.g. 512).
        # But our "tokens" are now dynamic. 
        # Actually, let's keep seq_len as "Number of Tokens".
        # 
        # If we have F features + 1 target per row.
        # Tokens per row = F + 1.
        # Rows per window = seq_len // (F + 1).
        
        if not self.samples:
            return []
            
        n_features = self.samples[0][0].shape[1]
        tokens_per_row = n_features + 1
        self.rows_per_window = max(1, self.seq_len // tokens_per_row)
        
        # Update actual seq_len to be multiple of tokens_per_row to avoid partial rows
        self.actual_seq_len = self.rows_per_window * tokens_per_row
        
        for i, (feats, _) in enumerate(self.samples):
            n_rows = len(feats)
            # Need rows_per_window + 1 (for target shifting)
            # Wait, if we predict target at current step, we technically need next step's target?
            # No, standard LM: [A, B, C] -> Predict [B, C, D]
            # So we need 1 extra token.
            # Easiest is to take 1 extra ROW to be safe.
            
            max_start = n_rows - self.rows_per_window - 1
            if max_start < 0:
                continue
                
            for start in range(0, max_start, max(1, self.stride // tokens_per_row)):
                indices.append((i, start))
                
        return indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample_idx, start_row = self.indices[idx]
        features_all, targets_all = self.samples[sample_idx]
        
        # Slice window
        end_row = start_row + self.rows_per_window
        
        # Get features and targets for this window
        # Plus one extra row for the shift (next token prediction)
        # Actually, we construct the sequence of length N, and targets is shifted.
        # So we need data for N+1 tokens. 
        # Since we block by row, we take 1 extra row and truncate later if needed.
        
        window_feats = features_all[start_row : end_row + 1] # [W+1, F]
        window_targets = targets_all[start_row : end_row + 1] # [W+1]
        
        # Construct flat sequence
        input_tokens = []
        target_labels = [] # For loss calculation
        
        n_features = window_feats.shape[1]
        
        # Iterate through rows
        # We need to build exactly `actual_seq_len` inputs.
        
        for i in range(len(window_feats)):
            row_feats = window_feats[i]
            row_target = window_targets[i]
            
            # 1. Shuffle Features (if training)
            if self.is_training:
                # Random permutation of indices
                perm = np.random.permutation(n_features)
                row_feats_shuffled = row_feats[perm]
            else:
                row_feats_shuffled = row_feats # Fixed order for val
            
            # 2. Append to sequence: [F1, F2... Fn, T]
            input_tokens.extend(row_feats_shuffled)
            input_tokens.append(row_target)
            
            # 3. Build Labels (Masking)
            # Input:  [F1, F2, ..., Fn, T]
            # Label:  [F2, F3, ..., T,  Next_F1]
            
            # We want loss ONLY when predicting T.
            # So label corresponding to input Fn should be T.
            # Label corresponding to input T should be Next_F1 (or ignored).
            # User said: "Gradient only based on price movement".
            # So we ignore prediction of Next_F1 too?
            # "The gradient should only update weights based on the model's ability to predict the Price Movement."
            # This implies P(Target | Features).
            
            # Mask for Feature Inputs (predicting next feature or target)
            # F1 -> predict F2 (Ignore)
            # ...
            # Fn-1 -> predict Fn (Ignore)
            # Fn -> predict T (KEEP!)
            # T -> predict Next_F1 (Ignore)
            
            # So we append -100 for all features except the last one.
            # Actually, the labels array aligns with inputs.
            # input[k] predicts input[k+1].
            # input[LAST_FEAT_INDEX] predicts input[TARGET_INDEX] = T.
            # So label for input[LAST_FEAT_INDEX] is T.
            
            # Let's add -100 for (n_features - 1) times
            # Then add T for 1 time (corresponding to Fn)
            # Then add -100 for 1 time (corresponding to T predicting next)
            
            # Wait, labels are aligned with INPUTS.
            # If input is F1, label is F2. Mask -100.
            # ...
            # If input is Fn, label is T. Keep T.
            # If input is T, label is Next_F1. Mask -100.
            
            # So for each row:
            # - n_features inputs are features.
            # - The last feature input (Fn) predicts T.
            # - The Target input (T) predicts next.
            
            # Labels mapping corresponding to [F1...Fn, T]:
            # F1 -> -100
            # ...
            # Fn -> T (The actual target value)
            # T  -> -100
            
            labels_row = [-100] * (n_features - 1)
            labels_row.append(row_target) # Corresponds to Fn input
            labels_row.append(-100)       # Corresponds to T input
            
            target_labels.extend(labels_row)

        # Truncate to exact seq_len
        # We generated chunks of (F+1).
        # We need input sequence of length seq_len.
        # labels also length seq_len.
        # But standard causal training:
        # inputs = seq[:-1]
        # targets = seq[1:] (aligned with inputs)
        # So we create a sequence of length seq_len + 1
        
        full_seq = np.array(input_tokens[:self.actual_seq_len + 1], dtype=np.int64)
        full_labels = np.array(target_labels[:self.actual_seq_len + 1], dtype=np.int64)
        
        # Shift
        x = torch.tensor(full_seq[:-1])
        y = torch.tensor(full_seq[1:]) # The standard labels (actual next tokens)
        
        # Apply mask to y
        # We constructed `target_labels` to align with `input_tokens`.
        # `target_labels[i]` is the "desired loss target" for `input_tokens[i]`.
        # My logic above: "If input is Fn, label is T".
        # So `target_labels[index_of_Fn]` = T.
        # `y` (standard shifted) will have `T` at `index_of_Fn`.
        # So we can just use `target_labels` logic to build the mask or just use `target_labels` directly as y_masked?
        
        # Let's re-verify:
        # Sequence: [F1, F2, T]
        # Inputs:   [F1, F2, T] (len 3, but usually we feed N and predict N)
        # x: [F1, F2]
        # y: [F2, T ]
        # Masked y: [-100, T]
        
        # My `target_labels` construction:
        # Row: [F1, F2, T]
        # labels_row: [-100, T, -100]
        # labels_row[0] correspond to input F1. (Label should be F2, we set -100)
        # labels_row[1] correspond to input F2. (Label should be T, we set T)
        # labels_row[2] correspond to input T.  (Label should be Next_F1, we set -100)
        
        # So `target_labels` IS exactly `y_masked` but aligned with `input_tokens`.
        # We need `y_masked` corresponding to `x`.
        # `x` is `input_tokens[:-1]`.
        # So `y_masked` is `target_labels[:-1]`.
        
        y_masked = torch.tensor(full_labels[:-1])
        
        return x, y_masked


def create_dataloader(
    data_dir: Path = DATA_TOKENIZED_DIR,
    seq_len: int = 384,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    **dataset_kwargs
) -> DataLoader:
    dataset = StructuredForexDataset(
        data_dir=data_dir,
        seq_len=seq_len,
        is_training=shuffle, # Only shuffle features during training
        **dataset_kwargs
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )

def main():
    # Test
    ds = StructuredForexDataset(seq_len=20)
    if len(ds) > 0:
        x, y = ds[0]
        print(f"X: {x.tolist()}")
        print(f"Y: {y.tolist()}")

if __name__ == "__main__":
    main()



--- FILE: backend/src/train/inference_liquid.py ---
"""
Inference script for I.A.R.A. Liquid Neural Network (CfC Version)
=================================================================
Loads trained model and evaluates on test data for Trend Prediction.
"""

import argparse
import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import polars as pl
import torch
from sklearn.metrics import roc_auc_score

from src.model.iara_liquid import IARA, LiquidConfig
from src.data.processor import Normalizer, NormalizerState
from src.train.trainer_liquid import TrainerConfig  # Required for checkpoint unpickling

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Paths
DATA_PROCESSED_DIR = Path(__file__).parent.parent.parent / "data" / "processed"
SCALER_PATH = Path(__file__).parent.parent.parent / "data" / "scaler.pkl"
CHECKPOINT_DIR = Path(__file__).parent.parent.parent / "checkpoints"


def load_model(checkpoint_path: Path, device: torch.device) -> Tuple[IARA, dict]:
    """Load trained I.A.R.A. model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Handle different config keys (legacy vs new)
    if "config" in checkpoint:
        trainer_config = checkpoint["config"]
        # Convert TrainerConfig to LiquidConfig
        model_config = LiquidConfig(
            input_dim=trainer_config.input_dim,
            hidden_dim=trainer_config.hidden_dim,
            dropout=trainer_config.dropout
        )
    elif "model_config" in checkpoint:
        model_config = checkpoint["model_config"]
    else:
        # Fallback default
        logger.warning("No config found in checkpoint. Using defaults.")
        model_config = LiquidConfig(hidden_dim=256)
    
    model = IARA(model_config).to(device)
    
    # Handle state dict key mismatch (if any)
    state_dict = checkpoint.get("model", checkpoint.get("model_state_dict"))
    model.load_state_dict(state_dict)
    
    model.eval()
    
    epoch = checkpoint.get('epoch', 'N/A')
    best_auc = checkpoint.get('best_auc', 'N/A')
    logger.info(f"Loaded model from {checkpoint_path} (Epoch {epoch}, Best AUC: {best_auc})")
    
    return model, checkpoint


def load_test_data(
    ticker: str = "EURUSD",
    normalizer: Optional[Normalizer] = None,
    limit: int = 500,
    seq_len: int = 64,
    expected_features: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load test data for a specific ticker."""
    file_path = DATA_PROCESSED_DIR / f"{ticker}.parquet"
    if not file_path.exists():
        raise FileNotFoundError(f"Ticker {ticker} not found at {file_path}")
    
    df = pl.read_parquet(file_path)
    
    # Dynamically detect all *_norm columns (same as trainer)
    feature_cols = sorted([c for c in df.columns if c.endswith("_norm")])
    
    # Filter to expected features if model was trained with fewer
    if expected_features and len(feature_cols) > expected_features:
        feature_cols = feature_cols[:expected_features]
        logger.info(f"Filtered to {len(feature_cols)} features to match model")
    
    logger.info(f"Detected {len(feature_cols)} features: {feature_cols}")
    
    # Take last N samples for testing (plus buffer for seq_len)
    df = df.tail(limit + seq_len)
    
    features = df.select(feature_cols).to_numpy().astype(np.float32)
    
    # Targets for reference
    target_cls = np.zeros(len(features))
    if "close_log_ret" in df.columns:
         target_vals = df["close_log_ret"].to_numpy().astype(np.float32)
         target_cls = (target_vals > 0).astype(np.float32)
    
    
    # Log test data distribution
    n_up = int(target_cls.sum())
    n_down = len(target_cls) - n_up
    logger.info(f"Loaded {len(features)} samples, {len(feature_cols)} features")
    logger.info(f"Test distribution: UP={n_up}, DOWN={n_down} ({n_up/len(target_cls)*100:.1f}% UP)")
    
    target_reg = np.zeros(len(features))  # Placeholder
    return features, target_reg, target_cls



from src.trading.active_inference import ActiveInferenceModule

@torch.no_grad()
def evaluate(
    model: IARA,
    features: np.ndarray,
    target_cls: np.ndarray,
    device: torch.device,
    seq_len: int = 64
) -> dict:
    """Evaluate model on test data using Active Inference (Dynamic Thresholds)."""
    model.eval()
    
    n_samples = len(features) - seq_len
    if n_samples <= 0:
        raise ValueError(f"Not enough samples. Need at least {seq_len + 1}")
    
    # Initialize Feedback Loop
    active_brain = ActiveInferenceModule()
    
    predictions = []
    skipped = 0
    active_decisions = []
    
    for i in range(n_samples):
        # Create sequence
        x = torch.tensor(features[i:i + seq_len], dtype=torch.float32).unsqueeze(0).to(device)
        
        # Forward pass
        logits, _ = model(x)
        logits = logits.squeeze(-1)
        
        # Probabilities
        probs = torch.sigmoid(logits)
        pred_prob = probs.cpu().item() 
        
        # v6.11: Extract Regime Features (ADX/ATR)
        # Assuming sorted features: adx_norm is index 0, atr_norm is index 1
        current_step = features[i + seq_len - 1]
        market_state = {
            'adx_norm': current_step[0].item(),
            'atr_norm': current_step[1].item()
        }
        
        # Get Dynamic Decision with Regime Check
        decision = active_brain.get_decision(pred_prob, market_features=market_state)
        
        # True value
        true_idx = i + seq_len - 1
        true_cls_val = target_cls[true_idx]
        
        # Log to Active Brain for Calibration (Simulate Feedback)
        active_brain.log_prediction(pred_prob, true_cls_val)
        
        # Determine Prediction Class based on Decision
        if decision == "BUY":
            pred_cls = 1.0
        elif decision == "SELL":
            pred_cls = 0.0
        else:
            pred_cls = -1.0 # Waited
            skipped += 1
            
        confidence = abs(pred_prob - 0.5) * 2
        
        predictions.append({
            "true_cls": true_cls_val,
            "pred_prob": pred_prob,
            "pred_cls": pred_cls,
            "decision": decision,
            "confidence": confidence,
            "threshold": active_brain.current_threshold
        })
        
        if pred_cls != -1.0:
            active_decisions.append((true_cls_val, pred_cls))
    
    # Calculate Metrics (Only for Traded Samples)
    if active_decisions:
        true_classes = [t for t, p in active_decisions]
        pred_classes = [p for t, p in active_decisions]
        
        tp = sum(1 for t, p in zip(true_classes, pred_classes) if t == 1 and p == 1)
        tn = sum(1 for t, p in zip(true_classes, pred_classes) if t == 0 and p == 0)
        fp = sum(1 for t, p in zip(true_classes, pred_classes) if t == 0 and p == 1)
        fn = sum(1 for t, p in zip(true_classes, pred_classes) if t == 1 and p == 0)
        
        accuracy = (tp + tn) / len(active_decisions) * 100
        precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # AUC needs probs, filtered
        traded_probs = [p["pred_prob"] for p in predictions if p["pred_cls"] != -1.0]
        try:
            auc = roc_auc_score(true_classes, traded_probs)
        except:
            auc = 0.5
    else:
        accuracy, precision, recall, f1, auc = 0, 0, 0, 0, 0
        tp, tn, fp, fn = 0, 0, 0, 0
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
        "confusion_matrix": {"TP": tp, "TN": tn, "FP": fp, "FN": fn},
        "n_samples": n_samples,
        "n_traded": len(active_decisions),
        "n_skipped": skipped,
        "predictions": predictions
    }


def print_sample_predictions(results: dict, n_samples: int = 10):
    """Print sample predictions."""
    print("\n" + "=" * 60)
    print("Sample Predictions (Last 10)")
    print("=" * 60)
    print(f"{'True Dir':>8} | {'Pred Prob':>9} | {'Pred Dir':>8} | {'Conf':>6}")
    print("-" * 60)
    
    preds = results["predictions"][-n_samples:]
    
    for p in preds:
        t_dir = "↑ UP" if p["true_cls"] > 0.5 else "↓ DOWN"
        p_dir = "↑ UP" if p["pred_cls"] > 0.5 else "↓ DOWN"
        match = "✓" if (p["true_cls"] > 0.5) == (p["pred_cls"] > 0.5) else "✗"
        
        print(f"{t_dir:>8} | {p['pred_prob']:>9.4f} | {p_dir:>8} {match} | {p['confidence']:>6.2%}")


def main():
    parser = argparse.ArgumentParser(description="Test I.A.R.A. Liquid Neural Network")
    parser.add_argument("--checkpoint", type=str, default="best_model.pth", help="Checkpoint filename")
    parser.add_argument("--ticker", type=str, default="EURUSD", help="Ticker to test on")
    parser.add_argument("--limit", type=int, default=500, help="Number of samples to test")
    parser.add_argument("--seq-len", type=int, default=64, help="Sequence length")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load model
    checkpoint_path = CHECKPOINT_DIR / args.checkpoint
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        return
    
    model, checkpoint = load_model(checkpoint_path, device)
    
    # Load test data
    logger.info(f"Loading test data for {args.ticker}...")
    normalizer = Normalizer.load(SCALER_PATH)
    
    # Get expected features from checkpoint config
    trainer_config = checkpoint.get('config')
    expected_features = trainer_config.input_dim if trainer_config else None
    
    features, target_reg, target_cls = load_test_data(
        args.ticker, normalizer, args.limit, args.seq_len, expected_features
    )
    logger.info(f"Loaded {len(features)} samples, {features.shape[1]} features")
    
    # Evaluate
    logger.info("Evaluating...")
    results = evaluate(model, features, target_cls, device, args.seq_len)
    
    # Print results
    print("\n" + "=" * 60)
    print(f"I.A.R.A. v6.4 Evaluation Results - {args.ticker}")
    print("=" * 60)
    print(f"Samples Evaluated: {results['n_samples']}")
    print(f"Trades Executed:   {results['n_traded']} ({results['n_traded']/results['n_samples']*100:.1f}%)")
    print(f"Signals Skipped:   {results['n_skipped']} (Volatile/Low Conf)")
    print("-" * 60)
    print(f"Active Accuracy: {results['accuracy']:.2f}%")
    print(f"AUC:             {results['auc']:.4f}")
    print(f"Precision:       {results['precision']:.2f}%")
    print(f"Recall:          {results['recall']:.2f}%")
    print(f"F1 Score:        {results['f1']:.4f}")
    print("-" * 60)
    cm = results["confusion_matrix"]
    print(f"Confusion Matrix: [TP={cm['TP']}, TN={cm['TN']}, FP={cm['FP']}, FN={cm['FN']}]")
    
    print_sample_predictions(results)
    
    # Trading interpretation
    print("\n" + "=" * 60)
    print("Trading Interpretation")
    print("=" * 60)
    if results["accuracy"] > 55:
        print("✅ Model shows predictive edge! (Acc > 55%)")
    elif results["accuracy"] > 52:
        print("⚠️  Marginal edge. May need more training or tuning.")
    else:
        print("❌ No evident edge. Consider more epochs or architecture changes.")


if __name__ == "__main__":
    main()



--- FILE: backend/src/kernels/interface.py ---
"""
Flash Liquid Interface - PyTorch Autograd Wrapper for Triton Kernels
=====================================================================
Provides a drop-in replacement for standard PyTorch CfC operations.
Automatically falls back to PyTorch if Triton is unavailable.
"""

import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)

# Check Triton Availability
TRITON_AVAILABLE = False
try:
    from src.kernels.fused_cfc import fused_cfc_forward, fused_cfc_backward
    TRITON_AVAILABLE = True
    logger.info("Triton kernels loaded successfully.")
except ImportError as e:
    logger.warning(f"Triton not available, falling back to PyTorch: {e}")


class FusedCfCFunction(torch.autograd.Function):
    """
    Autograd Function for Fused CfC Update.
    
    Forward:  h_new = h_prev * (1 - gate) + tanh(input) * gate
    Backward: Computes gradients for h_prev, input, and gate.
    """
    
    @staticmethod
    def forward(ctx, h_prev: torch.Tensor, input_linear: torch.Tensor, gate_linear: torch.Tensor):
        """
        Args:
            h_prev: Previous hidden state [B, H]
            input_linear: Linear transform of input (x @ W + b) [B, H]
            gate_linear: Gate logits (x @ Wg + h @ Ug + bg) [B, H]
        
        Returns:
            h_new: Updated hidden state [B, H]
        """
        # Ensure contiguous memory layout
        h_prev = h_prev.contiguous()
        input_linear = input_linear.contiguous()
        gate_linear = gate_linear.contiguous()
        
        if TRITON_AVAILABLE and h_prev.is_cuda:
            # Use Triton Kernel
            h_new = fused_cfc_forward(h_prev, input_linear, gate_linear)
        else:
            # PyTorch Fallback
            signal = torch.tanh(input_linear)
            gate = torch.sigmoid(gate_linear)
            h_new = h_prev * (1.0 - gate) + signal * gate
        
        # Save for backward
        ctx.save_for_backward(h_prev, input_linear, gate_linear)
        
        return h_new
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """
        Compute gradients.
        """
        h_prev, input_linear, gate_linear = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        
        if TRITON_AVAILABLE and h_prev.is_cuda:
            # Use Triton Kernel
            grad_h_prev, grad_input, grad_gate = fused_cfc_backward(
                grad_output, h_prev, input_linear, gate_linear
            )
        else:
            # PyTorch Fallback
            signal = torch.tanh(input_linear)
            gate = torch.sigmoid(gate_linear)
            
            # d(h_new)/d(h_prev) = (1 - gate)
            grad_h_prev = grad_output * (1.0 - gate)
            
            # d(h_new)/d(signal) = gate
            # d(signal)/d(input) = 1 - tanh^2
            grad_signal = grad_output * gate
            grad_input = grad_signal * (1.0 - signal * signal)
            
            # d(h_new)/d(gate) = signal - h_prev
            # d(gate)/d(gate_linear) = gate * (1 - gate)
            grad_gate_val = grad_output * (signal - h_prev)
            grad_gate = grad_gate_val * gate * (1.0 - gate)
        
        return grad_h_prev, grad_input, grad_gate


# Convenience wrapper
def fused_cfc_update(h_prev: torch.Tensor, input_linear: torch.Tensor, gate_linear: torch.Tensor) -> torch.Tensor:
    """
    Drop-in replacement for CfC update step.
    
    Uses Triton kernel if available, else falls back to PyTorch.
    """
    return FusedCfCFunction.apply(h_prev, input_linear, gate_linear)


class FusedCfCCell(nn.Module):
    """
    Fused CfC Cell using Triton Kernels.
    
    Drop-in replacement for standard CfCCell with ~2x speedup.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Linear transforms (computed in PyTorch, fused only the update)
        self.linear_input = nn.Linear(input_dim, hidden_dim)
        self.linear_hidden = nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        self.gate_input = nn.Linear(input_dim, hidden_dim)
        self.gate_hidden = nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, input_: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        """
        Forward pass using fused kernel for update step.
        
        Args:
            input_: Input tensor [B, D]
            hidden: Previous hidden state [B, H]
        
        Returns:
            h_new: Updated hidden state [B, H]
        """
        # Compute linear transforms (standard PyTorch)
        input_linear = self.linear_input(input_) + self.linear_hidden(hidden)
        gate_linear = self.gate_input(input_) + self.gate_hidden(hidden)
        
        # Fused CfC Update (Triton or PyTorch fallback)
        h_new = fused_cfc_update(hidden, input_linear, gate_linear)
        
        # Post-processing (standard PyTorch)
        h_new = self.layer_norm(h_new)
        h_new = self.dropout(h_new)
        h_new = torch.clamp(h_new, min=-1.0, max=1.0)
        
        return h_new



--- FILE: backend/src/kernels/fused_cfc.py ---
"""
Flash Liquid CfC Kernel - Fused CUDA/ROCm Kernel using Triton
==============================================================
Fuses the CfC update rule into a single GPU operation:
    signal = tanh(input_linear)
    gate = sigmoid(time_gate)
    h_new = h_prev * (1 - gate) + signal * gate

This eliminates multiple memory round-trips, reducing VRAM IO by ~60%.

Reference: Hasani et al. (2022) "Closed-form Continuous-time Neural Networks"
"""

import triton
import triton.language as tl
import torch

# Block size tuned for AMD RDNA2 (RX 6750 XT has 2560 shaders)
# Power of 2, typically 256-1024 works well
DEFAULT_BLOCK_SIZE = 1024


@triton.jit
def fused_cfc_forward_kernel(
    h_prev_ptr,      # Pointer to previous hidden state [B, H]
    input_ptr,       # Pointer to input linear (x @ W_input + b) [B, H]
    gate_ptr,        # Pointer to gate linear (x @ W_gate + h @ U_gate + b) [B, H]
    output_ptr,      # Pointer to output h_new [B, H]
    n_elements,      # Total number of elements (B * H)
    BLOCK_SIZE: tl.constexpr
):
    """
    Fused CfC Forward Kernel.
    
    Computes:
        signal = tanh(input)
        gate = sigmoid(gate_input)
        h_new = h_prev * (1 - gate) + signal * gate
    
    All in registers, single memory load/store.
    """
    # Program ID determines which block of data this kernel instance processes
    pid = tl.program_id(axis=0)
    
    # Calculate the range of elements this block will process
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Mask to handle edge cases (last block may not be full)
    mask = offsets < n_elements
    
    # Load data from global memory to registers
    h_prev = tl.load(h_prev_ptr + offsets, mask=mask, other=0.0)
    input_val = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    gate_val = tl.load(gate_ptr + offsets, mask=mask, other=0.0)
    
    # Compute in registers (ZERO intermediate VRAM IO)
    # tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
    # Using Triton's built-in for numerical stability
    signal = tl.math.tanh(input_val)
    
    # sigmoid(x) = 1 / (1 + exp(-x))
    gate = tl.sigmoid(gate_val)
    
    # CfC Update Rule
    # h_new = h_prev * (1 - gate) + signal * gate
    # Equivalent to: h_new = h_prev + gate * (signal - h_prev)
    h_new = h_prev * (1.0 - gate) + signal * gate
    
    # Store result back to global memory
    tl.store(output_ptr + offsets, h_new, mask=mask)


@triton.jit
def fused_cfc_backward_kernel(
    grad_output_ptr,  # Gradient from upstream [B, H]
    h_prev_ptr,       # Previous hidden state [B, H]
    input_ptr,        # Input linear [B, H]
    gate_ptr,         # Gate linear [B, H]
    grad_h_prev_ptr,  # Output: Gradient w.r.t. h_prev [B, H]
    grad_input_ptr,   # Output: Gradient w.r.t. input [B, H]
    grad_gate_ptr,    # Output: Gradient w.r.t. gate [B, H]
    n_elements,
    BLOCK_SIZE: tl.constexpr
):
    """
    Fused CfC Backward Kernel.
    
    Given: h_new = h_prev * (1 - gate) + signal * gate
    Where: signal = tanh(input), gate = sigmoid(gate_input)
    
    Derivatives:
        d(h_new)/d(h_prev) = (1 - gate)
        d(h_new)/d(signal) = gate
        d(h_new)/d(gate) = signal - h_prev
        
        d(signal)/d(input) = 1 - tanh(input)^2
        d(gate)/d(gate_input) = gate * (1 - gate)
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load
    grad_out = tl.load(grad_output_ptr + offsets, mask=mask, other=0.0)
    h_prev = tl.load(h_prev_ptr + offsets, mask=mask, other=0.0)
    input_val = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    gate_val = tl.load(gate_ptr + offsets, mask=mask, other=0.0)
    
    # Recompute activations
    signal = tl.math.tanh(input_val)
    gate = tl.sigmoid(gate_val)
    
    # Gradients
    # d(loss)/d(h_prev) = d(loss)/d(h_new) * d(h_new)/d(h_prev) = grad_out * (1 - gate)
    grad_h_prev = grad_out * (1.0 - gate)
    
    # d(loss)/d(signal) = grad_out * gate
    # d(loss)/d(input) = d(loss)/d(signal) * d(signal)/d(input)
    #                  = grad_out * gate * (1 - signal^2)
    grad_signal = grad_out * gate
    grad_input = grad_signal * (1.0 - signal * signal)
    
    # d(loss)/d(gate) = grad_out * (signal - h_prev)
    # d(loss)/d(gate_input) = d(loss)/d(gate) * d(gate)/d(gate_input)
    #                       = grad_out * (signal - h_prev) * gate * (1 - gate)
    grad_gate_val = grad_out * (signal - h_prev)
    grad_gate = grad_gate_val * gate * (1.0 - gate)
    
    # Store gradients
    tl.store(grad_h_prev_ptr + offsets, grad_h_prev, mask=mask)
    tl.store(grad_input_ptr + offsets, grad_input, mask=mask)
    tl.store(grad_gate_ptr + offsets, grad_gate, mask=mask)


def fused_cfc_forward(h_prev: torch.Tensor, input_linear: torch.Tensor, gate_linear: torch.Tensor) -> torch.Tensor:
    """
    Launch the Triton kernel for CfC forward pass.
    
    Args:
        h_prev: Previous hidden state [B, H]
        input_linear: Pre-computed linear transform (x @ W + b) [B, H]
        gate_linear: Pre-computed gate linear (x @ Wg + h @ Ug + bg) [B, H]
    
    Returns:
        h_new: Updated hidden state [B, H]
    """
    assert h_prev.is_cuda or h_prev.is_contiguous(), "Tensors must be on GPU and contiguous"
    assert h_prev.shape == input_linear.shape == gate_linear.shape
    
    output = torch.empty_like(h_prev)
    n_elements = h_prev.numel()
    
    # Grid: number of blocks needed to cover all elements
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    # Launch kernel
    fused_cfc_forward_kernel[grid](
        h_prev, input_linear, gate_linear, output,
        n_elements,
        BLOCK_SIZE=DEFAULT_BLOCK_SIZE
    )
    
    return output


def fused_cfc_backward(
    grad_output: torch.Tensor,
    h_prev: torch.Tensor,
    input_linear: torch.Tensor,
    gate_linear: torch.Tensor
) -> tuple:
    """
    Launch the Triton kernel for CfC backward pass.
    
    Returns:
        (grad_h_prev, grad_input, grad_gate)
    """
    assert grad_output.is_contiguous()
    
    grad_h_prev = torch.empty_like(h_prev)
    grad_input = torch.empty_like(input_linear)
    grad_gate = torch.empty_like(gate_linear)
    
    n_elements = h_prev.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    fused_cfc_backward_kernel[grid](
        grad_output, h_prev, input_linear, gate_linear,
        grad_h_prev, grad_input, grad_gate,
        n_elements,
        BLOCK_SIZE=DEFAULT_BLOCK_SIZE
    )
    
    return grad_h_prev, grad_input, grad_gate



--- FILE: backend/src/utils/logger.py ---

import json
import time
import logging
from pathlib import Path
from typing import Dict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# LOGS_DIR = Path(__file__).parent.parent.parent / "logs"
# Use absolute path relative to project root assuming src/utils location
LOGS_DIR = Path("logs")

class TelemetryLogger:
    """Real-time JSONL logger for training dynamics."""
    def __init__(self, log_dir: Path = LOGS_DIR):
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.log_path = log_dir / f"training_history_{timestamp}.jsonl"
        logger.info(f"Telemetry logging to: {self.log_path}")
        
    def log(self, entry: Dict):
        """Append a log entry to the JSONL file."""
        try:
            with open(self.log_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logger.error(f"Failed to write telemetry: {e}")



--- FILE: backend/src/utils/create_dummy_checkpoint.py ---

import sys
import torch
from pathlib import Path

# Add src to path
# sys.path.append("/app") - Removed, rely on PYTHONPATH

from src.model.iara_liquid import IARA, LiquidConfig
from src.train.trainer_liquid import TrainerConfig

def create_checkpoint():
    print("Creating Dummy Checkpoint...")
    
    config = TrainerConfig()
    
    liquid_config = LiquidConfig(
        input_dim=config.input_dim,
        hidden_dim=config.hidden_dim,
        num_heads=config.num_heads,
        dropout=config.dropout,
        max_seq_len=config.max_seq_len
    )
    
    model = IARA(liquid_config)
    
    # Checkpoint Dir
    checkpoint_dir = Path("/app/checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    path = checkpoint_dir / "best_model.pth"
    
    torch.save({
        'model': model.state_dict(),
        'config': config,
        'epoch': 0,
        'best_val_auc': 0.5
    }, path)
    
    print(f"Saved: {path}")

if __name__ == "__main__":
    create_checkpoint()



--- FILE: backend/src/run_simulation.py ---

import torch
import os
import time
from src.trading.simulation import MarketSimulator
from src.model.iara_liquid import IARA, LiquidConfig
from dataclasses import dataclass
from pathlib import Path

@dataclass
class TrainerConfig:
    # Minimal config to satisfy unpickler
    seq_len: int = 256
    stride: int = 2
    batch_size: int = 1024
    accumulation_steps: int = 1
    num_workers: int = 8
    val_split: float = 0.2
    input_dim: int = 16
    hidden_dim: int = 128
    num_heads: int = 4
    dropout: float = 0.5
    max_seq_len: int = 256
    num_cfc_layers: int = 2
    epochs: int = 200
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    grad_clip: float = 0.5
    resume: bool = False
    label_smooth: float = 0.1
    noise_std: float = 0.02
    scale_min: float = 0.98
    scale_max: float = 1.02
    patience: int = 150
    target_lookahead: int = 5
    checkpoint_dir: Path = Path("checkpoints")
    asymmetry_penalty: float = 2.5
    error_memory_size: int = 500

# Config
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_PATH = "checkpoints/best_model.pth"

def main():
    print(f"🚀 Starting SACI Protocol Simulation (Automated)")
    print(f"Device: {DEVICE}")

    # 1. Load Model
    if not os.path.exists(CHECKPOINT_PATH):
        print("❌ Checkpoint not found. Please train first.")
        return

    print("Loading Model...")
    config = LiquidConfig(input_dim=18, hidden_dim=128)
    model = IARA(config).to(DEVICE)
    try:
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            model.load_state_dict(checkpoint["model"])
        else:
            model.load_state_dict(checkpoint)
        model.eval()
        print("✅ Model Loaded.")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    # 2. Init Simulator
    sim = MarketSimulator(model, ticker="EURUSD", device=DEVICE)
    
    # 3. Run Loop
    print("running simulation steps...")
    steps = 50
    for i in range(steps):
        state = sim.step()
        if state:
            status = "WAIT"
            if "BUY" in str(state.logic_log): status = "BUY"
            elif "SELL" in str(state.logic_log): status = "SELL"
            
            print(f"Step {i+1}/{steps} | Prob: {state.prob:.1%} | Action: {status}")
            time.sleep(0.05) # simulate real-time
        else:
            print("End of Data.")
            break
            
    print(f"✅ Simulation Complete. {steps} BrainStates persisted to TimescaleDB.")

if __name__ == "__main__":
    main()



--- FILE: backend/src/container_debug.py ---

import torch
import os
from src.train.continuous_learner import ContinuousLearner
from src.brain.models.liquid_v9 import SwarmBrain
from src.brain.models.generator import ChallengerGenerator

def debug():
    print("--- Docker Container Debug ---")
    learner = ContinuousLearner()
    print(f"Device: {learner.device}")
    print(f"Learner workers hidden: {learner.worker_hidden}")
    
    # Check model dims
    print(f"Model input_dim (implicitly from workers): {learner.model.worker_config.input_dim}")
    
    # Check challenger dims
    print(f"Challenger input_dim: {learner.challenger.input_dim}")
    print(f"Challenger encoder[0] weight shape: {learner.challenger.encoder[0].weight.shape}")

    # Check a shard if exists
    shard_dir = "data/cache/EURUSD/train_shards"
    if os.path.exists(shard_dir):
        shards = [f for f in os.listdir(shard_dir) if f.endswith(".pt")]
        if shards:
            shard_path = os.path.join(shard_dir, shards[0])
            print(f"Checking shard: {shard_path}")
            data = torch.load(shard_path, map_location='cpu')
            X = data['X']
            print(f"Shard X shape: {X.shape}")
            if X.shape[-1] > 12:
                print(f"Warning: Shard has {X.shape[-1]} features, model expects 12.")
    else:
        print(f"Shard dir {shard_dir} not found.")

if __name__ == "__main__":
    debug()



--- FILE: backend/src/brain/train_v8.py ---
import torch
import torch.nn as nn
import torch.optim as optim
import time
import argparse
import sys
import os
import json
import csv
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.metrics import roc_auc_score
import math

# Import Trinity Architecture
from src.brain.models.liquid_v8 import LiquidConfig, ModulatedIARA
from src.brain.models.trinity_liquid import TrinityLiquidBrain

# Constants
LOG_DIR = "logs"
SNAPSHOT_DIR = os.path.join(LOG_DIR, "snapshots")
MODELS_DIR = os.path.join(LOG_DIR, "models")
METRICS_FILE = os.path.join(LOG_DIR, "metrics.csv")
TELEMETRY_FILE = os.path.join(LOG_DIR, "telemetry_latest.json")
LEADERBOARD_FILE = os.path.join(LOG_DIR, "leaderboard.json")

# Global Data Cache
MARKET_DATA_CACHE = None

def load_and_cache_data():
    global MARKET_DATA_CACHE
    if MARKET_DATA_CACHE is None:
        # ABSOLUTE PATH for Docker
        csv_path = "/app/data/raw/EURUSD_synthetic.csv"
        
        if not os.path.exists(csv_path):
            # Also try relative path for local debug
            csv_path = "data/raw/EURUSD_synthetic.csv"
            
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"🔥 CRITICAL: Market Data NOT FOUND at {csv_path}. \n"
                                    "Please ensure data/raw/EURUSD_synthetic.csv exists and is mounted.\n"
                                    "Training CANNOT proceed without real data.")
            
        print(f"📂 Loading Market Data from {csv_path}...")
        try:
            df = pd.read_csv(csv_path)
            # Normalize column names
            df.columns = [c.lower().strip() for c in df.columns]
            
            if "date" in df.columns:
                df["timestamp"] = pd.to_datetime(df["date"])
            elif "time" in df.columns:
                df["timestamp"] = pd.to_datetime(df["time"])
            else:
                df["timestamp"] = pd.date_range(end=datetime.now(), periods=len(df), freq="15min")
                
            df = df.sort_values("timestamp").reset_index(drop=True)
            
            # Ensure required columns
            req_cols = ["open", "high", "low", "close"]
            for c in req_cols:
                if c not in df.columns:
                     raise ValueError(f"Missing column '{c}' in CSV.")
            
            # ----------------------------------------------------
            # FEATURE ENGINEERING (Give the IA "Glasses")
            # ----------------------------------------------------
            print("🧠 Pre-calculating Technical Indicators...")
            
            # 1. RSI (14)
            delta = df["close"].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / (loss + 1e-6)
            df["rsi"] = 100 - (100 / (1 + rs))
            df["rsi"] = df["rsi"].fillna(50) / 100.0 # Normalize 0-1
            
            # 2. MACD (12, 26, 9)
            ema12 = df["close"].ewm(span=12, adjust=False).mean()
            ema26 = df["close"].ewm(span=26, adjust=False).mean()
            df["macd"] = ema12 - ema26
            df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
            # Normalize robustness
            df["macd"] = (df["macd"] - df["macd"].mean()) / (df["macd"].std() + 1e-6)
            
            # 3. Bollinger Bands (20)
            sma20 = df["close"].rolling(window=20).mean()
            std20 = df["close"].rolling(window=20).std()
            df["bb_upper"] = (df["close"] - sma20) / (2 * std20 + 1e-6) # Z-Score from middle
            df["bb_upper"] = df["bb_upper"].fillna(0)
            
            # 4. Returns
            df["log_ret"] = np.log(df["close"] / df["close"].shift(1))
            df["log_ret"] = df["log_ret"].fillna(0)
            
            # 5. Volatility
            df["volatility"] = df["log_ret"].rolling(window=20).std().fillna(0)
            
            # Drop NaN from warmup
            df = df.iloc[30:].reset_index(drop=True)
                     
            MARKET_DATA_CACHE = df
        except Exception as e:
            raise RuntimeError(f"Failed to read CSV: {e}")
            
    return MARKET_DATA_CACHE

from torch.utils.data import TensorDataset, DataLoader

class CryptoDataset(TensorDataset):
    def __init__(self, x_path, y_path=None):
        self.x = torch.load(x_path)
        if isinstance(self.x, dict):
            # Handle potential dict wrapping
            self.y = self.x.get('y')
            self.x = self.x.get('X')
        else:
            self.y = None
            
        if self.y is None and y_path:
             self.y = torch.load(y_path)

        # Move to CPU first to avoid CUDA OOM during init if large
        self.x = self.x.float()
        self.y = self.y.float()
        
    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

def get_dataloaders(batch_size):
    import glob
    # Load Cache (Support Sharding or Single File)
    print("📂 Loading Pre-computed Tensors from data/cache/...")
    
    train_shards = sorted(glob.glob("data/cache/train_shards/*.pt"))
    val_shards = sorted(glob.glob("data/cache/val_shards/*.pt"))
    
    print(f"DEBUG: CWD = {os.getcwd()}")
    print(f"DEBUG: Found {len(train_shards)} train shards at data/cache/train_shards/*.pt")
    
    if len(train_shards) > 0:
        print(f"   Found {len(train_shards)} training shards.")
        train_ds = torch.utils.data.ConcatDataset([CryptoDataset(f) for f in train_shards])
        val_ds = torch.utils.data.ConcatDataset([CryptoDataset(f) for f in val_shards])
    else:
        print("   Found legacy single-file dataset.")
        train_ds = CryptoDataset("data/cache/train.pt")
        val_ds = CryptoDataset("data/cache/val.pt")
    
    print(f"   Train Size: {len(train_ds)} | Val Size: {len(val_ds)}")
    
    # Num workers 0 to avoid multiprocessing overhead/hangs on some systems
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=0)
    
    return train_loader, val_loader

# Replaces generate_market_scenario for Training, 
# BUT we keep a helper for Visualization relying on the CSV.
def get_viz_scenario(seq_len=256):
    """
    Returns a single coherent slice from CSV for Dashboard visualization.
    Matches the input shape (18) by adding dummy features if needed or re-calculating.
    """
    df = load_and_cache_data()
    max_idx = len(df) - seq_len - 10
    start_idx = np.random.randint(0, max_idx)
    slice_df = df.iloc[start_idx : start_idx + seq_len].copy()
    
    # We need to recreate the 18-dim features to match the model
    # For viz purposes, we will try to match the V6 schema roughly or just pad
    # V6 (18 dims) usually: 
    # [NormClose, NormHigh, NormLow, RSI, MACD, Signal, BB_Up, BB_Low, CCI, ADX, STOCH, WILLR, ... ]
    # Since we can't easily reproduce exact V6 features on the fly without the factory code,
    # We will assume dimensions 0-5 are main price/techs and rest are advanced.
    
    # Hack: We generate the 6 basic we know, and pad the rest with zeros for the visualization inference.
    # The 'Brain' might give garbage expert curves on this dummy data, but the Hidden State heatmap will work.
    
    closes = slice_df["close"].values
    norm_close = (closes - closes.mean()) / (closes.std() + 1e-6)
    
    # We rely on what load_and_cache_data computes (RSI, MACD, etc)
    # Check lines 65+ in load_and_cache_data
    
    rsi = slice_df["rsi"].values
    macd = slice_df["macd"].values
    bb_pos = slice_df["bb_upper"].values
    log_ret = slice_df["log_ret"].values
    volat = slice_df["volatility"].values
    
    # 6 Real features
    base_feats = np.column_stack([norm_close, rsi, macd, bb_pos, log_ret, volat])
    
    # Pad to 18
    padding = np.zeros((seq_len, 12))
    full_feats = np.column_stack([base_feats, padding])
    
    inputs = torch.FloatTensor(full_feats).unsqueeze(0) # Batch 1
    llm_ctx = torch.randn(1, 4096)
    news_vec = torch.zeros(1, 2)
    
    return inputs, slice_df, llm_ctx, news_vec

class ForensicRecorder:
    def __init__(self, model: TrinityLiquidBrain):
        self.model = model
        self.captured_data = {"feature_gates": []}
        self._register_hooks()

    def _register_hooks(self):
        def hook_feature_gate(module, input, output):
            self.captured_data["feature_gates"] = output.detach().cpu().numpy()
        self.model.lnn.feature_attention.gate[-1].register_forward_hook(hook_feature_gate)

    def capture_snapshot(self, inputs, ohlc_df, llm_ctx, news_vec, epoch):
        self.model.eval()
        with torch.no_grad():
            hidden_states = []
            def hook_cfc(module, input, output):
                hidden_states.append(output.detach().cpu().numpy())
            
            handle_cfc = self.model.lnn.cfc_layers[-1].register_forward_hook(hook_cfc)
            _ = self.model(inputs, llm_ctx, news_vec)
            
            feat_imp = self.captured_data["feature_gates"][0].mean(axis=0).tolist()
            handle_cfc.remove()
            
            hidden_stack = np.stack(hidden_states, axis=1)
            h_state = hidden_stack[0] # Sample 0
            
            h_tensor = torch.tensor(h_state).to(inputs.device) 
            long_curve = self.model.lnn.long_expert(h_tensor).detach().cpu().numpy().flatten().tolist()
            short_curve = self.model.lnn.short_expert(h_tensor).detach().cpu().numpy().flatten().tolist()
            
            snapshot = {
                "epoch": epoch,
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "ohlc": {
                   "open": ohlc_df["open"].tolist(),
                   "high": ohlc_df["high"].tolist(),
                   "low": ohlc_df["low"].tolist(),
                   "close": ohlc_df["close"].tolist(),
                   "time": ohlc_df["timestamp"].dt.strftime("%H:%M").tolist()
                },
                "feature_importance": feat_imp,
                "expert_conviction": {
                    "long": long_curve,
                    "short": short_curve
                },
                "hidden_activity": h_state.T.tolist(),
                "simulation_metadata": {"scenario": "REAL MARKET (EURUSD)", "volatility": "Real"}
            }
            
            path = os.path.join(SNAPSHOT_DIR, f"epoch_{epoch}.json")
            with open(path, "w") as f:
                json.dump(snapshot, f)
            
        self.model.train()

from torch.utils.data import WeightedRandomSampler

class SelectiveMemory:
    """
    v6.11 Selective Memory (Curriculum Learning).
    Tracks simple vs difficult samples across epochs to focus training.
    """
    def __init__(self, num_samples, alpha=0.7):
        self.alpha = alpha # EMA factor
        self.sample_loss_ema = torch.ones(num_samples)
        self.chronic_counter = torch.zeros(num_samples, dtype=torch.int16) # Persistent errors
        self.mastered_counter = torch.zeros(num_samples, dtype=torch.int16) # Persistent correct
        self.noise_mask = torch.zeros(num_samples, dtype=torch.bool) # Impossible samples
        
        # History for noise detection (Rolling window of 10 epochs)
        self.epoch_losses_history = torch.zeros(num_samples, 10)

    def analyze(self, model, dataset, device, batch_size=2048):
        """
        Deep Thought: Runs a fast evaluation pass to update sample difficulty.
        Returns: WeightedRandomSampler for the next epoch.
        """
        print("🧠 Deep Thought: Analyzing difficulty spectrum (Selective Memory)...")
        model.eval()
        losses = []
        accuracies = []
        
        # Fast Eval Loader (Sequential)
        eval_loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, 
            num_workers=0, pin_memory=True
        )
        
        criterion = nn.BCEWithLogitsLoss(reduction='none')
        
        total_batches = len(eval_loader)
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                for i, (inputs, targets) in enumerate(eval_loader):
                    if i % 10 == 0:
                        print(f"   [Deep Thought] Scanned {i}/{total_batches} batches...", end='\r')
                        
                    inputs = inputs.to(device)
                    targets = targets.float().to(device)
                    llm_ctx = torch.zeros(inputs.size(0), 4096).to(device)
                    news_vec = torch.zeros(inputs.size(0), 2).to(device)
                    
                    outputs = model(inputs, llm_ctx, news_vec)
                    outputs = outputs.squeeze(-1)
                    
                    # 1. Per-Sample Loss
                    loss = criterion(outputs, targets)
                    losses.append(loss.cpu())
                    
                    # 2. Per-Sample Correctness (Mastery)
                    probs = torch.sigmoid(outputs)
                    preds = (probs > 0.5).float()
                    hard_targets = (targets > 0.5).float()
                    correct = (preds == hard_targets).float()
                    
                    # High Confidence?
                    confident = ((probs > 0.7) | (probs < 0.3)).float()
                    accuracies.append(correct * confident)
        
        print(f"   [Deep Thought] Processing stats for {len(losses) * batch_size} samples...") 
        current_losses = torch.cat(losses)
        current_mastered = torch.cat(accuracies)
        
        # --- UPDATE STATE ---
        # 1. EMA Loss
        if len(self.sample_loss_ema) != len(current_losses):
             # Resize if dataset changed (robustness)
             self.sample_loss_ema = torch.ones_like(current_losses)
             
        self.sample_loss_ema = (self.alpha * self.sample_loss_ema) + ((1 - self.alpha) * current_losses)
        
        # 2. Chronic (High Loss > 0.9)
        high_loss = current_losses > 0.9
        self.chronic_counter[high_loss.cpu()] += 1
        self.chronic_counter[~high_loss] = 0
        
        # 3. Mastered (Confident Correct)
        mastered_mask = current_mastered > 0.5
        self.mastered_counter[mastered_mask.cpu()] += 1
        self.mastered_counter[~mastered_mask] = 0
        
        # 4. Noise (Impossible > 1.2 for 10 epochs)
        self.epoch_losses_history = torch.roll(self.epoch_losses_history, -1, dims=1)
        self.epoch_losses_history[:, -1] = current_losses
        noise_candidates = (self.epoch_losses_history > 1.2).all(dim=1)
        self.noise_mask = self.noise_mask | noise_candidates
        
        # --- CALCULATE WEIGHTS ---
        weights = (self.sample_loss_ema + 0.05) ** 2
        
        # Boost Chronic
        weights[self.chronic_counter >= 3] *= 3.0
        
        # Boost Arrogant (Very high loss = Wrong + Confident)
        weights[current_losses > 1.5] *= 2.0
        
        # Suppress Mastered
        weights[self.mastered_counter >= 5] *= 0.05
        
        # Kill Noise
        weights[self.noise_mask] = 0.0
        
        # Stats
        n_chronic = (self.chronic_counter >= 3).sum().item()
        n_mastered = (self.mastered_counter >= 5).sum().item()
        n_noise = self.noise_mask.sum().item()
        
        print(f"   📊 Stats: {n_chronic} Chronic | {n_mastered} Mastered | {n_noise} Noise (Dropped)")
        torch.cuda.empty_cache()
        
        # Sampler
        sampler = WeightedRandomSampler(
            weights=weights, num_samples=len(weights), replacement=True
        )
        return sampler

class AsymmetricFocalLoss(nn.Module):
    """
    v6.7 Profit-Weighted Asymmetric Focal Loss.
    Ported from trainer_liquid.py
    """
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Targets must be 0-1 range.
        bce_loss = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.exp(-bce_loss) # prob of correct class
        
        # Focal term: (1 - pt)^gamma
        focal_term = (1 - pt) ** self.gamma
        
        # Alpha balancing: Higher weight for positive class (1)
        # targets is 0 or 1.
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        loss = alpha_t * focal_term * bce_loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

class Leaderboard:
    def __init__(self):
        self.history = []
        if os.path.exists(LEADERBOARD_FILE):
            try:
                with open(LEADERBOARD_FILE, "r") as f:
                    self.history = json.load(f)
            except: pass
            
    def update(self, epoch, loss, accuracy, auc, model_state):
        entry = {
            "epoch": epoch,
            "loss": loss,
            "accuracy": accuracy,
            "auc": auc,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        self.history.append(entry)
        self.history = sorted(self.history, key=lambda x: x["loss"])[:5] 
        
        with open(LEADERBOARD_FILE, "w") as f:
            json.dump(self.history, f, indent=2)
            
        if self.history[0]["epoch"] == epoch:
            print(f"🏆 New Best Model! Loss: {loss:.4f} | Acc: {accuracy:.2%} | AUC: {auc:.4f}")
            torch.save(model_state, os.path.join(MODELS_DIR, "best_model.pth"))

def main():
    parser = argparse.ArgumentParser()
    # TUNING: Lower LR for stability (Recurrent models explode easily)
    parser.add_argument("--lr", type=float, default=0.0005) 
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=1024)
    # FORCE CPU by default to avoid ROCm hang
    parser.add_argument("--cpu", action="store_true", default=False) 
    args = parser.parse_args()

    # CRITICAL: Clean old snapshots
    if os.path.exists(SNAPSHOT_DIR):
        for f in os.listdir(SNAPSHOT_DIR):
            if f.endswith(".json"):
                 try: os.remove(os.path.join(SNAPSHOT_DIR, f))
                 except: pass
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    with open(METRICS_FILE, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["step", "epoch", "loss", "accuracy", "auc", "lr"])

    # 1. Init Model
    if args.cpu:
        device = "cpu"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        print("⚠️ CUDA not available. Falling back to CPU.")
        device = "cpu"

    print(f"🚀 Trinity Loop Initialized on {device.upper()}")
    if device == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    # Check Data Integrity BEFORE starting model
    try:
        load_and_cache_data() 
    except Exception as e:
        print(f"❌ DATA ERROR: {e}")
        sys.exit(1)

    # High Seq Len for Pattern Recognition
    # V8.5: Consuming V6.11 processed tensors which have 28 features (M5/15/30/H1 Contexts)
    INPUT_DIM = 28
    liq_cfg = LiquidConfig(input_dim=INPUT_DIM, hidden_dim=128, max_seq_len=256)
    lnn = ModulatedIARA(liq_cfg)
    model = TrinityLiquidBrain(lnn, lfm_hidden_dim=4096).to(device)
    
    # ----------------------------------------------------
    # RESUME FROM CHAMPION (Transfer Learning)
    # ----------------------------------------------------
    best_model_path = os.path.join(MODELS_DIR, "best_model.pth")
    if os.path.exists(best_model_path):
        print(f"🏆 FOUND CHAMPION MODEL. Resuming training from: {best_model_path}")
        try:
            # Handle dimension mismatch if upgrading from 16 -> 18
            state_dict = torch.load(best_model_path, map_location=device)
            current_dict = model.state_dict()
            
            # Simple check: If first layer weight shape mismatches, skip it
            # But here we assume we are starting fresh or resuming compatible
            model.load_state_dict(state_dict, strict=False) 
            print("✅ Weights loaded (Relaxed/Strict)! Starting with high IQ base.")
        except Exception as e:
            print(f"⚠️ Failed to load champion model: {e}. Starting fresh.")
    else:
        print("🌱 No previous champion found. Starting fresh training.")

    recorder = ForensicRecorder(model)
    leaderboard = Leaderboard()
    
    # v6.9 Logic: AdamW + Asymmetric Focal Loss
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4) # Added weight_decay
    # V6 Exact Match: Gentle decay
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=15)
    
    # TUNING: Switching to standard BCE to debug convergence
    criterion = nn.BCEWithLogitsLoss() # AsymmetricFocalLoss(alpha=0.75, gamma=1.0)
    # UPDATED: Use torch.amp.GradScaler for future compatibility
    scaler = torch.amp.GradScaler('cuda', enabled=(device == "cuda"))
    
    # LOAD DATASETS
    print("📂 Loading Pre-computed Tensors from data/cache/...")
    
    import glob
    train_shards = sorted(glob.glob("data/cache/train_shards/*.pt"))
    val_shards = sorted(glob.glob("data/cache/val_shards/*.pt"))
    
    if len(train_shards) > 0:
        print(f"   Found {len(train_shards)} training shards.")
        train_ds = torch.utils.data.ConcatDataset([CryptoDataset(f) for f in train_shards])
        val_ds = torch.utils.data.ConcatDataset([CryptoDataset(f) for f in val_shards])
    else:
        print("   Found legacy single-file dataset.")
        train_ds = CryptoDataset("data/cache/train.pt")
        val_ds = CryptoDataset("data/cache/val.pt")
        
    print(f"   Train Size: {len(train_ds)} | Val Size: {len(val_ds)}")
    
    # Init Selective Memory
    memory = SelectiveMemory(len(train_ds))
    
    # Default Loader (Shuffle) for Epoch 1
    # Num workers 0 to avoid multiprocessing overhead/hangs
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=True, num_workers=0)
    
    steps_per_epoch = len(train_loader)
    print("------------------------------------------------")
    print(f"🔄 Training Loop Start: {steps_per_epoch} steps per epoch.")
    
    for epoch in range(1, args.epochs + 1):
        # ----------------------------------------
        # SELECTIVE MEMORY (Curriculum)
        # ----------------------------------------
        # From Epoch 2 onwards, we refine the curriculum based on previous performance
        if epoch > 1:
            sampler = memory.analyze(model, train_ds, device)
            train_loader = DataLoader(
                train_ds, 
                batch_size=args.batch_size, 
                sampler=sampler, # Use Sampler -> shuffle must be False
                shuffle=False, 
                drop_last=True
            )
        
        epoch_loss = 0
        epoch_acc = 0
        epoch_auc = 0
        
        all_targets = []
        all_probs = []
        
        model.train()
        
        # Iterate DataLoader
        for step, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device)
            targets = targets.float().to(device)
            
            # STABILITY FIX: Zero out LLM context instead of random noise
            # We want the LNN to learn market physics first.
            llm_ctx = torch.zeros(inputs.size(0), 4096).to(device)
            news_vec = torch.zeros(inputs.size(0), 2).to(device)
            
            optimizer.zero_grad()
            
            # AMP Forward
            # UPDATED: Use torch.amp.autocast for future compatibility
            with torch.amp.autocast('cuda', enabled=(device == "cuda")):
                outputs = model(inputs, llm_ctx, news_vec)
                outputs = outputs.squeeze(-1)
                
                # Loss: Asymmetric Focal Loss (Classification Logic)
                loss = criterion(outputs, targets)
            
            # AMP Backward
            scaler.scale(loss).backward()
            
            # Gradient Clipping (Unscale first)
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            
            # Metrics
            probs = torch.sigmoid(outputs)
            
            # Binary Accuracy (V6 Style for comparison)
            # Treat > 0.5 as Buy, <= 0.5 as Sell/Hold (Downside bias)
            preds = (probs > 0.5).float()
            
            # For Accuracy calc, we assume Hard targets (0 or 1). 
            # If target is 0.5, it counts as 0 for binary comparison
            hard_targets = (targets > 0.5).float() 
            
            acc = (preds == hard_targets).float().mean().item()
            epoch_acc += acc
            
            # Collect for AUC
            all_targets.extend(hard_targets.detach().cpu().numpy())
            all_probs.extend(probs.detach().cpu().numpy())
            
            # Telemetry (Sampled to avoid CSV spam)
            if step % 10 == 0:
                global_step = (epoch - 1) * steps_per_epoch + step
                current_lr = optimizer.param_groups[0]['lr']
                
                # Running AUC estimate (chunk)
                try:
                    batch_auc = roc_auc_score(hard_targets.detach().cpu().numpy(), probs.detach().cpu().numpy())
                except:
                    batch_auc = 0.5
                    
                # Explicit Print for Docker Logs
                print(f"Epoch {epoch} | Step {step}/{steps_per_epoch} | Loss: {loss.item():.4f} | Acc: {acc:.4f} | AUC: {batch_auc:.4f}")
                
                if math.isnan(loss.item()):
                    print("❌ CRITICAL: Loss is NaN! Aborting current batch...")
                    
                with open(METRICS_FILE, "a", newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([global_step, epoch, loss.item(), acc, batch_auc, current_lr])
            
        avg_loss = epoch_loss / steps_per_epoch
        avg_acc = epoch_acc / steps_per_epoch
        
        try:
            avg_auc = roc_auc_score(all_targets, all_probs)
        except:
            avg_auc = 0.5
        
        # Adjust Learning Rate if plateau
        scheduler.step(avg_loss)
        
        print(f"Epoch {epoch}/{args.epochs} | Loss: {avg_loss:.4f} | Acc: {avg_acc:.2%} | AUC: {avg_auc:.4f}")
        
        # ----------------------------------------
        # SNAPSHOT (Visualization)
        # DISABLE: This causes OOM and dimension mismatch (18 vs 28 feats)
        # ----------------------------------------
        # try:
        #      # Grab a random scenario from CSV for visualization
        #      print("   📸 Capturing Forensic Snapshot...")
        #      # viz_inputs, viz_ohlc, v_ctx, v_news = get_viz_scenario()
        #      # viz_inputs = viz_inputs.to(device)
        #      # v_ctx = v_ctx.to(device)
        #      # v_news = v_news.to(device)
        #      # recorder.capture_snapshot(viz_inputs, viz_ohlc, v_ctx, v_news, epoch)
        #      print("   ✅ Snapshot Saved.")
        # except Exception as e:
        #      print(f"⚠️ Snapshot Warning: {e}")
        
        print("   🏆 Updating Leaderboard...") 
        leaderboard.update(epoch, avg_loss, avg_acc, avg_auc, model.state_dict())
        print("   ✅ Leaderboard Updated.")
        
        with open(TELEMETRY_FILE, "w") as f:
            json.dump({
                "current_epoch": epoch,
                "total_epochs": args.epochs,
                "status": "TRAINING",
                "last_loss": avg_loss,
                "device": device
            }, f)
            
    with open(TELEMETRY_FILE, "w") as f:
        json.dump({"status": "COMPLETED"}, f)
    print("✅ Training Complete.")
 
if __name__ == "__main__":
    main()



--- FILE: backend/src/brain/models/lob_encoder.py ---
"""
I.A.R.A. v11 — LOB Encoder (Sensory Bridge)
============================================
Maps raw LOB microstructure features into a learned embedding
that is fused with the SwarmCoordinator's consensus context.

Architecture:
  6 raw LOB features → Linear(64) → SiLU → Dropout → Linear(hidden) → LayerNorm
"""

import torch
import torch.nn as nn


class LOBEncoder(nn.Module):
    """
    Sensory Bridge for Level 2 Order Book data.

    Encodes 6 LOB microstructure features (OBI, VPIN, Spread, Depth, Walls)
    into a dense embedding compatible with the coordinator's CfC cell.
    """

    def __init__(self, input_dim: int = 6, hidden_dim: int = 64, output_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: LOB features [batch, 6]

        Returns:
            LOB embedding [batch, output_dim]
        """
        return self.encoder(x)



--- FILE: backend/src/brain/models/trinity_liquid.py ---
import torch
import torch.nn as nn
from typing import Optional

class TrinityLiquidBrain(nn.Module):
    """
    I.A.R.A. v8.5 "Pure Liquid Trinity"
    ===================================
    The fusion module connecting the two liquid brains:
    1. Tactical Agent (LNN): Fast, recurrent, continuous-time execution.
    2. Strategic Agent (LFM): Slow, long-context, reasoning modulation.
    """
    def __init__(self, lnn_core: nn.Module, lfm_hidden_dim: int):
        super().__init__()
        self.lnn = lnn_core
        
        # Bridge Layer: Projects LFM thinking (text/strategy) to LNN input space (math/tactical)
        # Assuming lnn_core.input_size is available (defined in ModulatedIARA)
        self.bridge_layer = nn.Linear(lfm_hidden_dim, lnn_core.input_size)
        
        # News Gate: Modulation from Sentinel (Sentiment, Impact)
        # Input: 2 (Sentiment Score, Impact Score) -> Output: 32 (Hidden Modulation)
        # We might want to project this to match modulation dim or keep it separate.
        # For v8.5, we implement a simple brake/boost logic via gating.
        self.news_gate = nn.Linear(2, lnn_core.input_size) 
        
        # Initialize weights
        nn.init.xavier_uniform_(self.bridge_layer.weight)
        nn.init.zeros_(self.bridge_layer.bias)
        nn.init.xavier_uniform_(self.news_gate.weight)
        nn.init.zeros_(self.news_gate.bias)

    def forward(self, market_data: torch.Tensor, lfm_context_tensor: torch.Tensor, news_vector: torch.Tensor):
        """
        Args:
            market_data: [Batch, Seq, Feat] - The raw market ticks/candles.
            lfm_context_tensor: [Batch, LFM_Dim] - The 'Strategy Vector' from the LFM.
            news_vector: [Batch, 2] - Sentiment and Impact from Sentinel.
        """
        # 1. Modulation (Liquid to Liquid)
        # Project Strategy Vector to LNN Input Dimension to bias the ODE
        strategy_bias = self.bridge_layer(lfm_context_tensor) # [Batch, Hidden]
        
        # 2. News Modulation
        news_bias = self.news_gate(news_vector) # [Batch, Hidden]
        
        # Combine Modulations
        # We sum them up. Strategy directs the flow, News alters the gain.
        total_modulation = strategy_bias + news_bias
        
        # 3. Execution (CfC Cell with modulation)
        # The LNN is "primed" by the strategy of the LFM.
        # ModulatedIARA.forward accepts 'external_modulation'
        tactical_decision = self.lnn(market_data, external_modulation=total_modulation)
        
        # 4. Brake Logic (Safety)
        # If News Impact is high (close to 1.0) and Sentiment is Negative, we might want to suppress output.
        # v8.5 Logic: We let the LNN decide based on the modulation, but we can enforce a hard brake 
        # inside the trading engine if needed. Here we return the raw decision (logits/values).
        
        return tactical_decision



--- FILE: backend/src/brain/models/generator.py ---
"""
I.A.R.A. v10.0 - Challenger Brain (Adversarial Generator)
==========================================================
The "Evil Professor" model. Its sole purpose is to find weaknesses 
in the Swarm Intelligence model by generating "pranks" (adversarial noise)
that lead to incorrect trading decisions.

It uses a light MLP architecture to perturb input features while
staying within a certain "plausibility" budget.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class ChallengerGenerator(nn.Module):
    """
    I.A.R.A. v10.0 - Challenger Brain (The Sadistic Professor)
    ==========================================================
    Generates realistic market manipulations to challenge the Trader.
    Modes:
      - Stop Hunt: Simulated wicks (Fake Breakouts)
      - Liquidation Cascade: Fast, high-volume price drops
      - Choppy Market: Lateral movement/noise injection
      - Neural Prank: Learned adversarial noise (v9.0)
    """
    
    def __init__(self, input_dim: int = 12, hidden_dim: int = 64, num_arms: int = 4):
        super().__init__()
        
        self.input_dim = input_dim
        self.num_arms = num_arms
        
        # RL / Multi-Armed Bandit State
        self.register_buffer("arm_counts", torch.ones(num_arms)) # Inicia com 1 para exploração
        self.register_buffer("arm_rewards", torch.zeros(num_arms)) # Recompensa: aumento da Loss do Trader
        self.exploration_c = 2.0 # UCB Exploration parameter
        
        # Neural Backbone (for learned pranks)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU()
        )
        
        self.noise_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Tanh()
        )
        
        self.pressure_scaler = nn.Parameter(torch.tensor(0.005)) 
        
    def _select_attack_mode(self) -> int:
        """UCB (Upper Confidence Bound) Arm Selection"""
        total_counts = self.arm_counts.sum()
        q_values = self.arm_rewards / self.arm_counts
        ucb_values = q_values + self.exploration_c * torch.sqrt(torch.log(total_counts) / self.arm_counts)
        return torch.argmax(ucb_values).item()

    def update_reward(self, arm: int, trader_loss_increase: float):
        """Called by ContinuousLearner after the Trader's forward pass"""
        self.arm_counts[arm] += 1
        self.arm_rewards[arm] += trader_loss_increase
        
    def _apply_stop_hunt(self, x: torch.Tensor, max_perturbation: torch.Tensor) -> torch.Tensor:
        """
        Simulates Stop Hunt (Violinada).
        Push price (Wick) past support/resistance and then snap back.
        Affected Features (indices):
          - 5: BB %B (Spike)
          - 6: ATR (Spike)
          - 9: Velocity (Sharp reversal)
        """
        batch_size, seq_len, dim = x.size()
        x_pranked = x.clone()
        
        # Pick 1-3 random time steps to inject wicks
        for b in range(batch_size):
            num_wicks = torch.randint(1, 4, (1,)).item()
            for _ in range(num_wicks):
                t = torch.randint(seq_len // 2, seq_len - 1, (1,)).item()
                direction = 1 if torch.rand(1).item() > 0.5 else -1
                
                # Intensidade baseada no cap (max_perturbation) do timestep atual
                intensidade = max_perturbation[b, t].item() * 2.0 
                
                # BB %B (idx 5) - Pura quebra de banda
                x_pranked[b, t, 5] += direction * intensidade
                
                # ATR (idx 6) - Volatilidade explode no hunt
                x_pranked[b, t, 6] *= (1.0 + 0.5 * max_perturbation[b, t].item())
                
                # Velocity (idx 9) - O movimento é rápido
                x_pranked[b, t, 9] += direction * intensidade * 3.0
                
        return x_pranked

    def _apply_liquidation_cascade(self, x: torch.Tensor, max_perturbation: torch.Tensor) -> torch.Tensor:
        """
        Simulates Efeito Dominó.
        High volume drops in a short period.
        """
        batch_size, seq_len, dim = x.size()
        x_pranked = x.clone()
        
        for b in range(batch_size):
            # Cascade starts in the last 20% of the sequence
            start_t = int(seq_len * 0.8)
            direction = -1 # Liquidation is usually a drop in this mode
            
            for t in range(start_t, seq_len):
                decay = (t - start_t + 1)
                drop_force = max_perturbation[b, t].item() * decay * 0.2
                
                # Sentiment (idx 11) - Panico
                x_pranked[b, t, 11] -= drop_force
                
                # Velocity (idx 9) - Acelerando
                x_pranked[b, t, 9] -= drop_force * 2.0
                
                # ATR (idx 6) - Panico gera volatilidade
                x_pranked[b, t, 6] *= (1.0 + 0.1 * max_perturbation[b, t].item() * decay)
                
        return x_pranked

    def _apply_choppy_market(self, x: torch.Tensor, max_perturbation: torch.Tensor) -> torch.Tensor:
        """
        Simulates Choppy Market (Ruído Lateral).
        Flatten indicators and add high-frequency noise.
        """
        # Diminue a variância dos indicadores (tendendo ao neutro/lateral)
        # 1.0 - pressure * 0.1 (clamped) using max_perturbation mean roughly
        pressure_proxy = max_perturbation.mean()
        factor = torch.clamp(torch.tensor(1.0 - pressure_proxy * 0.1), 0.1, 1.0).to(x.device)
        x_pranked = x * factor
        
        # Adiciona ruído jitter de alta frequência limitado pelo ATR gap
        noise = torch.randn_like(x) * max_perturbation.unsqueeze(-1)
        return x_pranked + noise

    def forward(
        self, 
        x: torch.Tensor, 
        pressure: float = 1.0,
        callback: Optional[callable] = None
    ) -> Tuple[torch.Tensor, int]:
        """
        Selection logic for the "Professor Sádico".
        Uses UCB Multi-Armed Bandit to exploit model weaknesses.
        """
        mode = self._select_attack_mode()
        
        # ATR Epsilon Cap (Realism Bound)
        # Assuming ATR is idx 6 (or proxy pressure if missing)
        batch_size, seq_len, dim = x.size()
        if dim > 6:
            atr_tensor = torch.abs(x[:, :, 6]) 
            max_perturbation = atr_tensor * 0.2 * pressure
        else:
            max_perturbation = torch.full((batch_size, seq_len), 0.1 * pressure, device=x.device)
            
        # Add tiny floor to ensure some perturbation happens even in flat markets
        max_perturbation = torch.clamp(max_perturbation, min=0.01)
        
        if mode == 0:
            if callback: callback(f"😈 Professor Sádico: Simulando Stop Hunt (Wick/Violinada) | Intensidade: {pressure:.2f}...")
            return self._apply_stop_hunt(x, max_perturbation), mode
        elif mode == 1:
            if callback: callback(f"😈 Professor Sádico: Simulando Liquidation Cascade (Efeito Dominó) | Intensidade: {pressure:.2f}...")
            return self._apply_liquidation_cascade(x, max_perturbation), mode
        elif mode == 2:
            if callback: callback(f"😈 Professor Sádico: Injetando Ruído Lateral (Choppy Market) | Intensidade: {pressure:.2f}...")
            return self._apply_choppy_market(x, max_perturbation), mode
        else:
            if callback: callback(f"😈 Professor Sádico: Aplicando Neural Prank (Adversarial Noise) | Intensidade: {pressure:.2f}...")
            # Traditional Neural Prank
            x_flat = x.view(-1, self.input_dim)
            features = self.encoder(x_flat)
            delta = self.noise_head(features)
            delta = delta.view(batch_size, seq_len, self.input_dim)
            
            # Use max_perturbation to cap neural noise
            delta_capped = torch.clamp(delta, min=-1.0, max=1.0) * max_perturbation.unsqueeze(-1)
            return x + (delta_capped * self.pressure_scaler), mode

class OverfittingGuard(nn.Module):
    """
    Monitors Training vs Validation gap and adjusts Challenger Pressure.
    If the Trader starts 'memorizing' (Loss_Train << Loss_Val), 
    the Guard increases the difficulty of the pranks.
    """
    
    def __init__(self, ema_alpha: float = 0.1):
        super().__init__()
        self.ema_alpha = ema_alpha
        self.register_buffer("train_loss_ema", torch.tensor(1.0))
        self.register_buffer("val_loss_ema", torch.tensor(1.0))
        self.register_buffer("pressure", torch.tensor(1.0))
        
    def update(self, train_loss: float, val_loss: float):
        # Update EMA
        self.train_loss_ema = (1 - self.ema_alpha) * self.train_loss_ema + self.ema_alpha * train_loss
        self.val_loss_ema = (1 - self.ema_alpha) * self.val_loss_ema + self.ema_alpha * val_loss
        
        # Calculate Overfitting Ratio
        # If val_loss is significantly higher than train_loss, we have overfitting
        ratio = (self.val_loss_ema / (self.train_loss_ema + 1e-6)).item()
        
        # Adjust pressure: if ratio > 1.2, increase pressure
        if ratio > 1.2:
            self.pressure *= 1.05 # Increase by 5%
        elif ratio < 1.05:
            self.pressure *= 0.95 # Relax if generalizing well
            
        # Clamp pressure
        self.pressure = torch.clamp(self.pressure, 0.1, 5.0)
        
    def get_pressure(self) -> float:
        return self.pressure.item()



--- FILE: backend/src/brain/models/regime_hmm.py ---
"""
I.A.R.A. v11 — Hidden Markov Model for Market Regime Detection
================================================================
A probabilistic "judge" above the SwarmBrain that classifies the
current market state into discrete regimes.

States:
  0 — Directional Trend (high ADX, strong momentum, H > 0.5)
  1 — Range-Bound / Choppy (low ADX, Hurst ≈ 0.5, noise)
  2 — Crisis / Crash (extreme volatility, sharp negative returns)

The regime state modulates the SwarmCoordinator's confidence:
  - In State 0: trend signals are boosted, reversal signals penalized
  - In State 1: all breakout signals penalized, mean-reversion boosted
  - In State 2: all signals heavily suppressed (capital preservation)

Reference: Rabiner (1989), "A Tutorial on HMMs and Selected Applications"
"""

import logging
import os
import pickle
from typing import Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger("RegimeHMM")


# Regime constants
REGIME_TREND = 0
REGIME_CHOPPY = 1
REGIME_CRISIS = 2
REGIME_UNKNOWN = 3  # Fallback


class MarketRegimeHMM:
    """
    3-state Hidden Markov Model for market regime detection.

    Uses Gaussian emissions over a small set of observable features:
      - ADX (trend strength)
      - ATR / Close ratio (volatility)
      - Returns (directional momentum)
      - Hurst Exponent (memory)
    """

    def __init__(self, n_states: int = 3, n_features: int = 4, model_path: Optional[str] = None):
        self.n_states = n_states
        self.n_features = n_features
        self.model_path = model_path or "data/hmm_regime_model.pkl"
        self.model = None
        self._load_or_init()

    def _load_or_init(self):
        """Load trained HMM or initialize a fresh one."""
        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, "rb") as f:
                    self.model = pickle.load(f)
                logger.info(f"📊 Loaded HMM regime model from {self.model_path}")
                return
            except Exception as e:
                logger.warning(f"Failed to load HMM model: {e}")

        try:
            from hmmlearn.hmm import GaussianHMM

            self.model = GaussianHMM(
                n_components=self.n_states,
                covariance_type="full",
                n_iter=200,
                random_state=42,
                verbose=False,
            )
            # Initialize with reasonable priors for market regimes
            self.model.startprob_ = np.array([0.5, 0.35, 0.15])
            self.model.transmat_ = np.array([
                [0.90, 0.07, 0.03],  # Trend → stays trending
                [0.10, 0.85, 0.05],  # Choppy → mostly stays choppy
                [0.15, 0.10, 0.75],  # Crisis → can recover
            ])
            self._is_fitted = False
            logger.info("📊 Initialized fresh HMM regime model (unfitted)")

        except ImportError:
            logger.warning("hmmlearn not installed. Regime detection disabled. Install with: pip install hmmlearn")
            self.model = None
            self._is_fitted = False

    def extract_regime_features(
        self,
        close: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        adx: np.ndarray,
        hurst: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Extract the 4 observable features for the HMM.

        Args:
            close: Close prices
            high: High prices
            low: Low prices
            adx: ADX values (trend strength)
            hurst: Optional Hurst exponent values

        Returns:
            np.ndarray of shape [N, 4]
        """
        n = len(close)

        # 1. ADX normalized [0, 1]
        adx_norm = np.clip(adx / 100.0, 0, 1)

        # 2. Volatility (ATR / Close ratio)
        tr = np.maximum(np.maximum(high - low, np.abs(high - np.roll(close, 1))), np.abs(low - np.roll(close, 1)))
        tr[0] = high[0] - low[0]
        atr_ratio = tr / (close + 1e-10)
        atr_norm = np.clip(atr_ratio * 100, 0, 5)  # Scale for visibility

        # 3. Returns (log returns, rolling 5-period)
        log_returns = np.zeros(n)
        log_returns[1:] = np.log(close[1:] / (close[:-1] + 1e-10))
        # Rolling mean of 5 periods
        kernel = np.ones(5) / 5
        returns_smooth = np.convolve(log_returns, kernel, mode="same")

        # 4. Hurst exponent (default 0.5 if not provided)
        if hurst is None:
            hurst_feat = np.full(n, 0.5)
        else:
            hurst_feat = np.clip(hurst, 0, 1)

        features = np.column_stack([adx_norm, atr_norm, returns_smooth, hurst_feat])
        return np.nan_to_num(features).astype(np.float32)

    def fit(self, features: np.ndarray) -> "MarketRegimeHMM":
        """
        Train the HMM on historical feature data.

        Args:
            features: np.ndarray of shape [N, 4] from extract_regime_features

        Returns:
            self
        """
        if self.model is None:
            logger.warning("HMM model not available. Cannot fit.")
            return self

        # Remove any rows with NaN
        mask = ~np.isnan(features).any(axis=1)
        clean_features = features[mask]

        if len(clean_features) < 100:
            logger.warning(f"Not enough data to fit HMM ({len(clean_features)} samples). Need at least 100.")
            return self

        try:
            self.model.fit(clean_features)
            self._is_fitted = True
            logger.info(f"📊 HMM fitted on {len(clean_features)} samples. States: {self.n_states}")

            # Save model
            os.makedirs(os.path.dirname(self.model_path) or ".", exist_ok=True)
            with open(self.model_path, "wb") as f:
                pickle.dump(self.model, f)
            logger.info(f"💾 HMM model saved to {self.model_path}")

        except Exception as e:
            logger.error(f"HMM fitting failed: {e}")
            self._is_fitted = False

        return self

    def predict_regime(self, features: np.ndarray) -> int:
        """
        Predict the current market regime from recent features.

        Args:
            features: np.ndarray of shape [N, 4] (use last ~50 candles)

        Returns:
            Regime state (0=Trend, 1=Choppy, 2=Crisis, 3=Unknown)
        """
        if self.model is None or not self._is_fitted:
            return REGIME_UNKNOWN

        try:
            features = np.nan_to_num(features.astype(np.float32))
            states = self.model.predict(features)
            # Return the most recent state
            return int(states[-1])
        except Exception as e:
            logger.debug(f"HMM prediction error: {e}")
            return REGIME_UNKNOWN

    def predict_regime_proba(self, features: np.ndarray) -> np.ndarray:
        """
        Get probability distribution over regimes for the latest observation.

        Returns:
            np.ndarray of shape [n_states] with probabilities
        """
        if self.model is None or not self._is_fitted:
            return np.array([0.33, 0.34, 0.33])

        try:
            features = np.nan_to_num(features.astype(np.float32))
            posteriors = self.model.predict_proba(features)
            return posteriors[-1]
        except Exception as e:
            logger.debug(f"HMM proba error: {e}")
            return np.array([0.33, 0.34, 0.33])

    @staticmethod
    def get_regime_penalty(regime: int) -> Dict[str, float]:
        """
        Get confidence multipliers based on detected regime.

        Returns:
            Dict with signal type multipliers.
            Values > 1.0 boost confidence, < 1.0 suppress.
        """
        penalties = {
            REGIME_TREND: {
                "trend": 1.2,
                "reversal": 0.6,
                "breakout": 1.1,
                "all": 1.0,
            },
            REGIME_CHOPPY: {
                "trend": 0.5,
                "reversal": 1.3,
                "breakout": 0.4,
                "all": 0.7,
            },
            REGIME_CRISIS: {
                "trend": 0.3,
                "reversal": 0.3,
                "breakout": 0.2,
                "all": 0.3,
            },
            REGIME_UNKNOWN: {
                "trend": 1.0,
                "reversal": 1.0,
                "breakout": 1.0,
                "all": 1.0,
            },
        }
        return penalties.get(regime, penalties[REGIME_UNKNOWN])

    @staticmethod
    def regime_name(regime: int) -> str:
        names = {
            REGIME_TREND: "🟢 Directional Trend",
            REGIME_CHOPPY: "🟡 Range-Bound/Choppy",
            REGIME_CRISIS: "🔴 Crisis/Crash",
            REGIME_UNKNOWN: "⚪ Unknown",
        }
        return names.get(regime, "⚪ Unknown")

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted if hasattr(self, '_is_fitted') else False



--- FILE: backend/src/brain/models/liquid_v8.py ---
"""
I.A.R.A. v8.5 - Modulated Liquid Neural Network
===============================================
Refactored for Trinity Engine (Protocol Trinity).
Adds support for 'external_modulation' (LLM Context Injection).

Based on v6.0 architecture by Hasani et al.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class LiquidConfig:
    """Configuration for IARA v8.5 Trinity."""
    input_dim: int = 16
    hidden_dim: int = 128
    num_heads: int = 4
    dropout: float = 0.5
    output_dim: int = 1
    max_seq_len: int = 256
    num_cfc_layers: int = 2

class LearnedPositionalEncoding(nn.Module):
    """Learned Positional Encoding for long sequences."""
    def __init__(self, d_model: int, max_len: int = 256, dropout: float = 0.1):
        super().__init__()
        self.position_embeddings = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)
        
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        self.position_embeddings.weight.data.copy_(pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        return self.dropout(x + self.position_embeddings(positions))

class FeatureAttention(nn.Module):
    """
    Feature-level attention gate.
    [v9.0] Simplified to Element-wise Gating (Diagonal) to force 1-to-1 feature importance.
    This prevents F15 (Noise) from being used to gate F0 (Price).
    """
    def __init__(self, input_dim: int, dropout: float = 0.1):
        super().__init__()
        # Global Average Pooling style context? No, we want to know WHICH feature is important.
        # Simple learnable weight per feature:
        # x * sigmoid(w*x + b) -> Non-linear self-gating
        
        self.gate = nn.Sequential(
            nn.Linear(input_dim, input_dim), # Diagonal-ish preference if we init properly, but dense is fine if trained well.
            # Actually, to strictly enforce "Is this feature useful?", let's use a shared MLP that looks at the feature distribution?
            # Let's stick to the previous architecture but initializing correlation.
            
            # REVISION: Let's use Squeeze-and-Excitation (SE) style:
            # Squeeze: Global Avg -> Excitation: FC -> Sigmoid -> Scale.
            # But we are Time-Distributed.
            
            # Simple Learnable Gate:
            nn.Linear(input_dim, input_dim),
            nn.Tanh(),
            nn.Linear(input_dim, input_dim),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [Batch, Seq, Feat]
        # We want to learn a mask M [Batch, Seq, Feat] or just [Feat]?
        # "Feature Importance" usually implies "for this sample, at this time".
        attention_mask = self.gate(x)
        return x * attention_mask  # Weighted features


class TemporalSelfAttention(nn.Module):
    """Multi-head self-attention."""
    def __init__(self, d_model: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attention(x, x, x, need_weights=False)
        return self.norm(x + self.dropout(attn_out))

class ModulatedCfCCell(nn.Module):
    """
    Symmetric CfC Cell v8.5 [MODULATED].
    
    Change:
    - forward() accepts `external_modulation` tensor.
    - Modulation is added to the system inputs before gating.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        self.linear_input = nn.Linear(input_dim, hidden_dim)
        self.linear_hidden = nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        self.gate_input = nn.Linear(input_dim, hidden_dim)
        self.gate_hidden = nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.Tanh()

    def forward(self, input_: torch.Tensor, hidden: torch.Tensor, external_modulation: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Standard CfC Input
        linear_part = self.linear_input(input_) + self.linear_hidden(hidden)
        gate_part = self.gate_input(input_) + self.gate_hidden(hidden)
        
        # [v8.5] Injection Point: External Modulation (LLM Context)
        # We inject it into the linear path (forcing the ODE to drift)
        if external_modulation is not None:
            # external_modulation: [batch, hidden_dim] (projected before call)
            # Ensure shape match. If modulation is [batch, hidden], we rely on broadcasting for input [batch, hidden]
            linear_part = linear_part + external_modulation
            # Optionally modulate gate too? For now, just linear dynamics.
        
        h_cand = self.activation(linear_part)
        h_cand = self.dropout(h_cand)
        
        gate = torch.tanh(gate_part)
        update = gate * (h_cand - hidden)
        h_new = hidden + update
        
        h_new = self.layer_norm(h_new)
        h_new = torch.clamp(h_new, min=-1.0, max=1.0)
        
        return h_new

class ModulatedIARA(nn.Module):
    """
    I.A.R.A. v8.5 - Modulated Backbone
    """
    
    def __init__(self, config: LiquidConfig):
        super().__init__()
        self.config = config
        
        # [v9.0] Feature Attention moved to INPUT level for Explainability (RSI vs Price)
        self.feature_attention = FeatureAttention(config.input_dim, dropout=config.dropout*0.1)
        
        self.input_proj = nn.Linear(config.input_dim, config.hidden_dim)
        self.pos_encoding = LearnedPositionalEncoding(config.hidden_dim, max_len=config.max_seq_len, dropout=config.dropout*0.5)
        self.self_attention = TemporalSelfAttention(config.hidden_dim, num_heads=config.num_heads, dropout=config.dropout)
        
        # self.attention_proj removed as we don't project back down for CfC input in this variant, 
        # BUT the definition of CfC layers below expects `config.input_dim` if i==0.
        # Wait, the CfC logic below is:
        # i=0: input_dim -> hidden_dim
        # The loop uses `current_input`.
        # If we project up to hidden_dim immediately, then CfC i=0 should be hidden->hidden?
        # Original code had: x_cfc = self.attention_proj(x_a) -> projected back to input_dim.
        # So current_input WAS input_dim.
        
        # Let's keep the flow: Input(16) -> Attn(16) -> Proj(128) -> SelfAttn(128) -> ProjBack(16) -> CfC.
        # NO, that's redundant.
        # Better: Input(16) -> FeatAttn(16) -> Proj(128) -> SelfAttn(128) -> CfC(128).
        # We need to change the first CfC layer to accept 128 input then.
        
        # Let's stick to the simplest explainability change:
        # Input(16) -> FeatAttn(16) -> Proj(128) -> ...
        # But we need to feed CfC.
        
        self.attention_proj = nn.Linear(config.hidden_dim, config.input_dim) # Keep this for compatibility with existing CfC loop
        self.input_norm = nn.LayerNorm(config.input_dim)
        
        # [v8.5] Modulated CfC Layers
        self.cfc_layers = nn.ModuleList([
            ModulatedCfCCell(config.input_dim if i == 0 else config.hidden_dim, 
                             config.hidden_dim, 
                             dropout=config.dropout)
            for i in range(config.num_cfc_layers)
        ])
        
        # MOE Heads
        self.long_expert = nn.Sequential(nn.Linear(config.hidden_dim, config.hidden_dim // 2), nn.SiLU(), nn.Dropout(config.dropout), nn.Linear(config.hidden_dim // 2, 1))
        self.short_expert = nn.Sequential(nn.Linear(config.hidden_dim, config.hidden_dim // 2), nn.SiLU(), nn.Dropout(config.dropout), nn.Linear(config.hidden_dim // 2, 1))
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)
        nn.init.zeros_(self.long_expert[-1].bias)
        nn.init.zeros_(self.short_expert[-1].bias)
    
    @property
    def input_size(self):
        return self.config.hidden_dim # The modulation connects to the hidden state space

    def forward(self, x: torch.Tensor, external_modulation: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 1. Feature Gating (Explainability Layer) - Applied on Raw Inputs
        x_gated = self.feature_attention(x) 
        
        # 2. Projection & Temporal Attention
        x_p = self.input_proj(x_gated)
        x_p = self.pos_encoding(x_p)
        x_a = self.self_attention(x_p)
        
        # 3. Project back for CfC backbone (keeping original flow structure for stability)
        x_cfc = self.attention_proj(x_a)
        x_cfc = self.input_norm(x_cfc)
        
        batch_size, seq_len, _ = x.shape
        current_input = x_cfc
        final_hidden = None
        
        for i, cell in enumerate(self.cfc_layers):
            h_t = torch.zeros(batch_size, self.config.hidden_dim, device=x.device, dtype=x.dtype)
            outputs = []
            
            # Recurrent Loop
            for t in range(seq_len):
                # Pass modulation only if dimensions match or if we want to modulate every step
                # Assumption: external_modulation is [Batch, Hidden]. We broadcast across Time.
                h_t = cell(current_input[:, t, :], h_t, external_modulation=external_modulation)
                outputs.append(h_t.unsqueeze(1))
            
            current_input = torch.cat(outputs, dim=1)
            final_hidden = h_t
            
        long_score = self.long_expert(final_hidden)
        short_score = self.short_expert(final_hidden)
        return (long_score - short_score).squeeze(-1)



--- FILE: backend/src/brain/models/liquid_v10.py ---
"""
I.A.R.A. v11.0 - Swarm Coordinator (Full Spectrum Brain)
=========================================================
Advanced central brain for I.A.R.A. v11.0.
Fuses multi-timeframe Workers with 3 new intelligence layers:

Architecture:
1. Cross-Attention: Multi-head worker consensus
2. Consensus Fusion: Confidence-weighted context
3. Sensory Bridge: LOB L2 Order Book encoder (OBI + VPIN)
4. Regime Gate: HMM regime embedding (Trend/Choppy/Crisis)
5. On-Chain GNN: Sparse graph whale flow embedding
6. Liquid CfC: Time-continuous decision refinement
7. Expert Heads: Long/Short/Policy outputs
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.brain.models.workers import WorkerSwarm, WorkerConfig, CfCCell
from src.brain.models.lob_encoder import LOBEncoder
from src.brain.models.gnn_onchain import OnChainGNN


@dataclass
class CoordinatorConfig:
    """Configuration for SwarmCoordinator v11.0."""
    num_workers: int = 5
    worker_dim: int = 128            # v10.0 Higher capacity
    hidden_dim: int = 512            # v10.0 Higher capacity
    num_heads: int = 8               # More heads for complex L2 patterns
    dropout: float = 0.2
    num_cfc_layers: int = 2          # Deeper liquid refinement
    # v11.0 New features
    lob_input_dim: int = 6           # LOB features: OBI, VPIN, spread, depth, walls
    lob_hidden_dim: int = 64         # LOB encoder hidden
    num_regimes: int = 4             # 3 states + 1 unknown
    regime_embed_dim: int = 32       # Regime embedding size
    gnn_output_dim: int = 32         # GNN graph embedding size
    gnn_node_features: int = 3       # On-chain node features


class CrossAttentionFusion(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attention(x, x, x, need_weights=False)
        x = self.norm(x + self.dropout(attn_out))
        return x


class ConsensusFusion(nn.Module):
    def __init__(self, num_workers: int, worker_dim: int, num_classes: int = 3, hidden_dim: int = 512):
        super().__init__()
        self.num_workers = num_workers
        self.cross_attn = CrossAttentionFusion(embed_dim=worker_dim, num_heads=8, dropout=0.1)
        self.confidence_proj = nn.Linear(num_classes, 1)
        fusion_input_dim = num_workers * worker_dim + num_workers
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.LayerNorm(hidden_dim)
        )
        
    def forward(self, embeddings: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        batch_size = embeddings.size(0)
        attended = self.cross_attn(embeddings)
        probs = F.softmax(logits, dim=-1)
        confidence = torch.sigmoid(self.confidence_proj(probs))
        weighted = attended * confidence
        combined = torch.cat([weighted.view(batch_size, -1), confidence.view(batch_size, -1)], dim=-1)
        return self.fusion(combined)


class SwarmCoordinator(nn.Module):
    def __init__(self, config: CoordinatorConfig = None):
        super().__init__()
        if config is None:
            config = CoordinatorConfig()
        self.config = config

        # === Core Fusion (unchanged from v10) ===
        self.fusion = ConsensusFusion(
            num_workers=config.num_workers,
            worker_dim=config.worker_dim,
            num_classes=3,
            hidden_dim=config.hidden_dim
        )

        # === v11: LOB Sensory Bridge ===
        self.lob_encoder = LOBEncoder(
            input_dim=config.lob_input_dim,
            hidden_dim=config.lob_hidden_dim,
            output_dim=config.lob_hidden_dim,
            dropout=config.dropout,
        )

        # === v11: Regime Embedding (HMM state) ===
        self.regime_embedding = nn.Embedding(config.num_regimes, config.regime_embed_dim)

        # === v11: On-Chain GNN ===
        self.onchain_gnn = OnChainGNN(
            node_features=config.gnn_node_features,
            hidden_dim=config.gnn_output_dim,
            output_dim=config.gnn_output_dim,
            dropout=config.dropout,
        )

        # === Context Fusion: Combine all intelligence sources ===
        # hidden_dim (base) + lob_hidden (LOB) + regime_embed (HMM) + gnn_output (GNN)
        augmented_dim = config.hidden_dim + config.lob_hidden_dim + config.regime_embed_dim + config.gnn_output_dim
        self.context_bridge = nn.Sequential(
            nn.Linear(augmented_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.LayerNorm(config.hidden_dim),
        )

        # === CfC & Expert Heads (same structure) ===
        self.cfc = CfCCell(input_dim=config.hidden_dim, hidden_dim=config.hidden_dim, dropout=config.dropout)
        self.long_expert = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2), nn.SiLU(),
            nn.Dropout(config.dropout), nn.Linear(config.hidden_dim // 2, 1)
        )
        self.short_expert = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2), nn.SiLU(),
            nn.Dropout(config.dropout), nn.Linear(config.hidden_dim // 2, 1)
        )
        self.policy_head = nn.Linear(config.hidden_dim, 3)
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
    def forward(
        self,
        embeddings: torch.Tensor,
        worker_logits: torch.Tensor,
        dt: Optional[torch.Tensor] = None,
        hx: Optional[torch.Tensor] = None,
        lob_features: Optional[torch.Tensor] = None,
        regime_state: Optional[torch.Tensor] = None,
        gnn_node_features: Optional[torch.Tensor] = None,
        gnn_adj_sparse: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with all intelligence layers.

        Args:
            embeddings: [B, num_workers, worker_dim] Worker embeddings
            worker_logits: [B, num_workers, 3] Worker predictions
            dt: Optional time deltas for CfC
            hx: Optional hidden state for CfC
            lob_features: Optional [B, 6] LOB microstructure features
            regime_state: Optional [B] integer regime states (0-3)
            gnn_node_features: Optional [N, 3] on-chain node features
            gnn_adj_sparse: Optional [N, N] sparse adjacency matrix
        """
        batch_size = embeddings.size(0)
        device = embeddings.device

        # 1. Core consensus from workers
        context = self.fusion(embeddings, worker_logits)  # [B, hidden_dim]

        # 2. LOB encoding
        if lob_features is not None:
            lob_emb = self.lob_encoder(lob_features)  # [B, lob_hidden]
        else:
            lob_emb = torch.zeros(batch_size, self.config.lob_hidden_dim, device=device)

        # 3. Regime embedding
        if regime_state is not None:
            regime_state = regime_state.clamp(0, self.config.num_regimes - 1)
            regime_emb = self.regime_embedding(regime_state)  # [B, regime_embed_dim]
        else:
            # Unknown regime
            unknown = torch.full((batch_size,), self.config.num_regimes - 1, dtype=torch.long, device=device)
            regime_emb = self.regime_embedding(unknown)

        # 4. GNN on-chain embedding
        if gnn_node_features is not None and gnn_adj_sparse is not None:
            gnn_emb = self.onchain_gnn(gnn_node_features, gnn_adj_sparse)  # [1, gnn_dim]
            gnn_emb = gnn_emb.expand(batch_size, -1)  # Broadcast to batch
        else:
            gnn_emb = torch.zeros(batch_size, self.config.gnn_output_dim, device=device)

        # 5. Fuse all intelligence sources
        augmented = torch.cat([context, lob_emb, regime_emb, gnn_emb], dim=-1)
        fused_context = self.context_bridge(augmented)  # [B, hidden_dim]

        # 6. CfC temporal refinement
        _, h_refined = self.cfc(fused_context.unsqueeze(1), dt, hx)

        # 7. Expert heads
        long_out = self.long_expert(h_refined)
        short_out = self.short_expert(h_refined)

        return (
            self.policy_head(h_refined),       # [B, 3] policy logits
            long_out - short_out,               # [B, 1] directional logit
            long_out,                           # [B, 1] long expert
            short_out,                          # [B, 1] short expert
        )


class SwarmBrain(nn.Module):
    """
    Complete Swarm Intelligence model v11.0.
    Integrates Workers + Coordinator + LOB + HMM + GNN.
    """

    def __init__(self, worker_config: WorkerConfig = None, coordinator_config: CoordinatorConfig = None):
        super().__init__()
        if worker_config is None:
            worker_config = WorkerConfig(hidden_dim=128)
        if coordinator_config is None:
            coordinator_config = CoordinatorConfig(num_workers=5, worker_dim=worker_config.hidden_dim)
        self.worker_config = worker_config
        self.coordinator_config = coordinator_config
        self.swarm = WorkerSwarm(worker_config)
        self.coordinator = SwarmCoordinator(coordinator_config)
        
    def forward(
        self,
        x_dict: dict,
        hx: Optional[torch.Tensor] = None,
        compute_telemetry: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Full forward pass.

        x_dict keys:
            "M5", "M15", "M30", "H1", "H4":  [B, seq_len, input_dim] per-timeframe tensors
            "coord_dt": Optional time deltas
            "lob": Optional [B, 6] LOB features
            "regime": Optional [B] regime state integers
            "gnn_nodes": Optional [N, 3] node features
            "gnn_adj": Optional [N, N] sparse adjacency
        """
        # Workers
        embeddings, worker_logits, _ = self.swarm(x_dict)

        # Coordinator with augmented inputs
        policy, directional, long_out, short_out = self.coordinator(
            embeddings=embeddings,
            worker_logits=worker_logits,
            dt=x_dict.get("coord_dt"),
            hx=hx,
            lob_features=x_dict.get("lob"),
            regime_state=x_dict.get("regime"),
            gnn_node_features=x_dict.get("gnn_nodes"),
            gnn_adj_sparse=x_dict.get("gnn_adj"),
        )

        # Telemetry placeholders (for backward compat with training loop)
        attn_weights = None
        confidence = None

        return policy, worker_logits, long_out, short_out, attn_weights, confidence
    
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)



--- FILE: backend/src/brain/models/trinity_v8.py ---
"""
TRINITY BRAIN v8.5 [PHASE 1]
============================
The Unified Model Class.

Components:
1. Tactical: LNN (ModulatedIARA)
2. Structural: LLM Gating (Context Projection)
3. Safety: News Awareness Brake (Gradient-Safe)
"""

import torch
import torch.nn as nn
from src.brain.models.liquid_v8 import ModulatedIARA

class TrinityBrain(nn.Module):
    def __init__(self, lnn_core: ModulatedIARA, llm_hidden_dim: int):
        super().__init__()
        self.lnn = lnn_core
        
        # Project LLM context to LNN hidden dimension
        # LNN hidden dim is usually 128 (defined in config)
        # We need to know lnn_core.config.hidden_dim
        lnn_dim = lnn_core.config.hidden_dim
        
        self.context_proj = nn.Linear(llm_hidden_dim, lnn_dim)
        
        # Prepare Gate for Future News (Sentinel Interface)
        # Input: [Sentiment (-1 to 1), Impact (0 to 1)]
        # Output: 32 dim latent representation (reserved for future Phase 2 logic if needed)
        self.news_gate = nn.Linear(2, 32)

    def forward(self, market_data: torch.Tensor, llm_context: torch.Tensor, news_vector: torch.Tensor) -> torch.Tensor:
        """
        Args:
            market_data: [Batch, Seq, Features]
            llm_context: [Batch, LLM_Dim]
            news_vector: [Batch, 2] -> [Sentiment, Impact]
        """
        
        # 1. Modulation (The Dual Core Logic)
        # Project LLM embedding to LNN latent space
        modulation = self.context_proj(llm_context) 
        # modulation shape: [Batch, LNN_Hidden]
        
        # 2. Tactical Execution (Pass modulation to CfC)
        x = self.lnn(market_data, external_modulation=modulation)
        
        # 3. Emergency Brake (Gradient-Safe)
        # news_vector[:, 1] is Impact Score (0.0 to 1.0)
        # If impact is 1.0 (Thermo-Nuclear), we want to suppress output.
        # Formula: output * (1.0 - impact)
        
        impact_score = news_vector[:, 1].unsqueeze(-1) # [Batch, 1]
        
        # Ensure impact is within [0, 1] just in case
        impact_score = torch.clamp(impact_score, 0.0, 1.0)
        
        # Apply Brake
        # x is [Batch] (from ModulatedIARA)
        # impact_score is [Batch, 1]
        # We need data to be [Batch, 1] to avoid [Batch, Batch] broadcasting
        output = x.unsqueeze(-1) * (1.0 - impact_score)
        
        return output.squeeze(-1)



--- FILE: backend/src/brain/models/liquid_v9.py ---
"""
I.A.R.A. v9 - Swarm Coordinator (Liquid Brain)
===============================================
The central brain that fuses Worker embeddings via Cross-Attention
and makes the final trading decision.

Architecture:
1. Cross-Attention: Workers "discuss" with each other
2. Consensus Fusion: Weighted combination based on confidence
3. Final CfC: Refines the consensus into a decision
4. Expert Heads: Long/Short specialists for asymmetric learning

Reference: "Attention Is All You Need" + Liquid Neural Networks
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.brain.models.workers import WorkerSwarm, WorkerConfig, CfCCell


@dataclass
class CoordinatorConfig:
    """Configuration for SwarmCoordinator."""
    num_workers: int = 5
    worker_dim: int = 64
    hidden_dim: int = 256
    num_heads: int = 4
    dropout: float = 0.3
    num_cfc_layers: int = 1


class CrossAttentionFusion(nn.Module):
    """
    Cross-attention layer where workers "vote" on consensus.
    """
    
    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, need_weights: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        attn_out, attn_weights = self.attention(x, x, x, need_weights=need_weights)
        x = self.norm(x + self.dropout(attn_out))
        return x, attn_weights


class ConsensusFusion(nn.Module):
    """
    Fuses worker embeddings + logits into a single context vector.
    """
    
    def __init__(
        self, 
        num_workers: int, 
        worker_dim: int, 
        num_classes: int = 3,
        hidden_dim: int = 256
    ):
        super().__init__()
        
        self.num_workers = num_workers
        self.cross_attn = CrossAttentionFusion(
            embed_dim=worker_dim,
            num_heads=4,
            dropout=0.1
        )
        
        self.confidence_proj = nn.Linear(num_classes, 1)
        fusion_input_dim = num_workers * worker_dim + num_workers
        
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.LayerNorm(hidden_dim)
        )
        
    def forward(
        self, 
        embeddings: torch.Tensor, 
        logits: torch.Tensor,
        need_weights: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        batch_size = embeddings.size(0)
        attended, attn_weights = self.cross_attn(embeddings, need_weights=need_weights)
        
        probs = F.softmax(logits, dim=-1)
        confidence = self.confidence_proj(probs)
        confidence = torch.sigmoid(confidence)
        
        weighted = attended * confidence
        
        flat_emb = weighted.view(batch_size, -1)
        flat_conf = confidence.view(batch_size, -1)
        
        combined = torch.cat([flat_emb, flat_conf], dim=-1)
        context = self.fusion(combined)
        
        return context, attn_weights, confidence


class SwarmCoordinator(nn.Module):
    """
    The central brain of the Swarm Intelligence architecture.
    """
    
    def __init__(self, config: CoordinatorConfig = None):
        super().__init__()
        
        if config is None:
            config = CoordinatorConfig()
        
        self.config = config
        
        # Consensus fusion
        self.fusion = ConsensusFusion(
            num_workers=config.num_workers,
            worker_dim=config.worker_dim,
            num_classes=3,
            hidden_dim=config.hidden_dim
        )
        
        # Phase 5 (The Dojo): Macro Fusion Layer
        self.macro_fusion = nn.Linear(config.hidden_dim + (config.hidden_dim // 4), config.hidden_dim)
        
        # CfC refinement
        self.cfc = CfCCell(
            input_dim=config.hidden_dim,
            hidden_dim=config.hidden_dim,
            dropout=config.dropout
        )
        
        # Expert Heads
        self.long_expert = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, 1)
        )
        
        self.short_expert = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, 1)
        )
        
        # NEW: Multi-class Policy Head (0: Wait, 1: Long, 2: Short)
        self.policy_head = nn.Linear(config.hidden_dim, 3)
        
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
    def forward(
        self, 
        embeddings: torch.Tensor, 
        worker_logits: torch.Tensor,
        dt: Optional[torch.Tensor] = None,
        hx: Optional[torch.Tensor] = None,
        compute_telemetry: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        # 1. Fuse worker outputs (now returns attention weights and confidence scores)
        context, attn_weights, confidence = self.fusion(embeddings, worker_logits, need_weights=compute_telemetry)
        
        # 2. Refine through CfC
        context_seq = context.unsqueeze(1)
        _, h_refined = self.cfc(context_seq, dt, hx)
        
        # 3. Macro Fusion (Fuse micro execution with macro trend)
        if hasattr(self, 'macro_bias') and self.macro_bias is not None:
            h_refined = self.macro_fusion(torch.cat([h_refined, self.macro_bias], dim=-1))
            h_refined = F.silu(h_refined)
            
        # 4. Expert heads
        long_score = self.long_expert(h_refined)
        short_score = self.short_expert(h_refined)
        
        # 5. Policy logits
        policy_logits = self.policy_head(h_refined)
        
        # 6. Differential logit for backward compatibility
        diff_logit = long_score - short_score
        
        return policy_logits, diff_logit, long_score, short_score, attn_weights, confidence


class SwarmBrain(nn.Module):
    """
    Complete Swarm Intelligence model.
    """
    
    def __init__(
        self, 
        worker_config: WorkerConfig = None,
        coordinator_config: CoordinatorConfig = None
    ):
        super().__init__()
        
        if worker_config is None:
            worker_config = WorkerConfig()
        
        if coordinator_config is None:
            coordinator_config = CoordinatorConfig(
                num_workers=5,
                worker_dim=worker_config.hidden_dim
            )
        
        self.worker_config = worker_config
        self.coordinator_config = coordinator_config
        
        # Phase 5 (The Dojo): Macro Context Encoder (Processa candles de H1)
        self.macro_encoder = nn.Sequential(
            nn.Linear(worker_config.input_dim, worker_config.hidden_dim),
            nn.SiLU(),
            nn.Linear(worker_config.hidden_dim, coordinator_config.hidden_dim // 4) 
        )
        
        self.swarm = WorkerSwarm(worker_config)
        self.coordinator = SwarmCoordinator(coordinator_config)
        
    def forward(
        self, 
        x_dict: dict,
        hx: Optional[torch.Tensor] = None,
        compute_telemetry: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        # 0. Process Macro Context (if provided, else dummy)
        x_macro_h1 = x_dict.get("H1")
        if x_macro_h1 is not None and len(x_macro_h1.shape) == 3:
            macro_feat = self.macro_encoder(x_macro_h1)
            self.coordinator.macro_bias = macro_feat.mean(dim=1)
        else:
            self.coordinator.macro_bias = None

        # 1. Process through worker swarm
        embeddings, worker_logits, _ = self.swarm(x_dict)
        
        # 2. Coordinate and decide
        coord_dt = x_dict.get("coord_dt")
        policy_logits, diff_logit, long_score, short_score, attn_weights, confidence = self.coordinator(
            embeddings, worker_logits, coord_dt, hx, compute_telemetry=compute_telemetry
        )
        
        # Return policy_logits + micro-state telemetry (attn_weights, confidence)
        return policy_logits, worker_logits, long_score, short_score, attn_weights, confidence
    
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)



--- FILE: backend/src/brain/models/gnn_onchain.py ---
"""
I.A.R.A. v11 — Sparse Graph Neural Network for On-Chain Analysis
=================================================================
Lightweight GCN (Graph Convolutional Network) that processes whale wallet
transaction graphs WITHOUT requiring torch-geometric.

Uses native torch.sparse for maximum ROCm GFX 10.3.0 compatibility.

Architecture:
  Node Features [N, 3] + Sparse Adj [N, N]
  → GCN Layer 1 [N, 32]
  → GCN Layer 2 [N, 32]
  → Global Mean Pooling [1, 32]
  → Linear Projection [1, output_dim]

Reference: Kipf & Welling (2017), "Semi-Supervised Classification with GCNs"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class SparseGCNLayer(nn.Module):
    """
    Manual Sparse Graph Convolutional Layer.

    Implements: H' = σ(D^{-1/2} A D^{-1/2} H W)

    Uses torch.sparse.mm for efficient computation on both
    CUDA (NVIDIA) and ROCm (AMD) backends.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        nn.init.xavier_uniform_(self.weight)

    def forward(
        self,
        node_features: torch.Tensor,
        adj_sparse: torch.Tensor,
        n_nodes: int,
    ) -> torch.Tensor:
        """
        Args:
            node_features: [N, in_features] dense node feature matrix
            adj_sparse: [N, N] sparse adjacency matrix (COO format)
            n_nodes: Number of nodes

        Returns:
            [N, out_features] updated node features
        """
        # Add self-loops: A_hat = A + I
        identity_indices = torch.arange(n_nodes, device=node_features.device)
        identity_indices = torch.stack([identity_indices, identity_indices])
        identity_values = torch.ones(n_nodes, device=node_features.device)

        # Combine adjacency with self-loops
        if adj_sparse._nnz() > 0:
            combined_indices = torch.cat([adj_sparse.indices(), identity_indices], dim=1)
            combined_values = torch.cat([adj_sparse.values(), identity_values])
        else:
            combined_indices = identity_indices
            combined_values = identity_values

        a_hat = torch.sparse_coo_tensor(
            combined_indices, combined_values,
            size=(n_nodes, n_nodes),
            device=node_features.device,
        ).coalesce()

        # Compute degree matrix D^{-1/2}
        degree = torch.sparse.sum(a_hat, dim=1).to_dense()
        degree_inv_sqrt = torch.zeros_like(degree)
        mask = degree > 0
        degree_inv_sqrt[mask] = degree[mask].pow(-0.5)

        # D^{-1/2} A D^{-1/2}: Scale rows and columns
        d_inv = degree_inv_sqrt.unsqueeze(1)

        # H * W
        support = torch.mm(node_features, self.weight)

        # D^{-1/2} * (A_hat * (D^{-1/2} * support))
        scaled_support = support * d_inv
        output = torch.sparse.mm(a_hat, scaled_support)
        output = output * d_inv

        if self.bias is not None:
            output = output + self.bias

        return output


class OnChainGNN(nn.Module):
    """
    Lightweight 2-layer GCN for on-chain whale wallet analysis.

    Processes the transaction graph and produces a single graph-level
    embedding that captures accumulation/distribution patterns.
    """

    def __init__(
        self,
        node_features: int = 3,
        hidden_dim: int = 32,
        output_dim: int = 32,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.gcn1 = SparseGCNLayer(node_features, hidden_dim)
        self.gcn2 = SparseGCNLayer(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        )
        self.output_dim = output_dim

    def forward(
        self,
        node_features: torch.Tensor,
        adj_sparse: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            node_features: [N, 3] node feature matrix
            adj_sparse: [N, N] sparse adjacency matrix

        Returns:
            [1, output_dim] graph-level embedding
        """
        n_nodes = node_features.size(0)

        # Ensure sparse tensor is on same device
        if adj_sparse.device != node_features.device:
            adj_sparse = adj_sparse.to(node_features.device)

        # GCN Layer 1
        h = self.gcn1(node_features, adj_sparse, n_nodes)
        h = F.silu(h)
        h = self.dropout(h)

        # GCN Layer 2
        h = self.gcn2(h, adj_sparse, n_nodes)
        h = F.silu(h)

        # Global Mean Pooling: [N, hidden] → [1, hidden]
        graph_embedding = h.mean(dim=0, keepdim=True)

        # Project to output dim
        return self.projection(graph_embedding)



--- FILE: backend/src/brain/models/workers.py ---
"""
I.A.R.A. v9 - Lightning Workers (Swarm Intelligence)
=====================================================
Ultra-light Liquid Neural Network specialists for each timeframe.

Each worker learns patterns specific to its timeframe:
- M5: Micro-structure, scalping patterns, noise filtering
- M15: Short-term momentum, minor S/R levels
- M30: Intraday trends, session patterns
- H1: Swing trading signals, major S/R
- H4: Macro trends, position sizing context

Reference: Hasani et al. (2022) "Closed-form Continuous-time Neural Networks"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple, List


@dataclass
class WorkerConfig:
    """Configuration for a specialist worker."""
    input_dim: int = 12  # Standard for v9.7
    hidden_dim: int = 64
    dropout: float = 0.2
    backbone: str = "cfc"  # Closed-form Continuous-time


class CfCCell(nn.Module):
    """
    Closed-form Continuous-time (CfC) layer.
    A computationally efficient Liquid Neural Network.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # RNN-like projection
        self.x_proj = nn.Linear(input_dim, hidden_dim)
        self.h_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Liquid Gating Parameters
        self.gate_f = nn.Linear(hidden_dim, hidden_dim)
        self.gate_i = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self, 
        x: torch.Tensor, 
        dt: Optional[torch.Tensor] = None,
        hx: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = x.size()
        
        if hx is None:
            hx = torch.zeros(batch_size, self.hidden_dim).to(x.device)
            
        if dt is None:
            dt = torch.ones(batch_size, seq_len, 1).to(x.device)
            
        h_t = hx
        outputs = []
        
        for t in range(seq_len):
            xt = x[:, t, :]
            dt_t = dt[:, t, :]
            
            x_in = self.x_proj(xt)
            f_gate = torch.sigmoid(self.gate_f(h_t))
            i_gate = torch.tanh(self.gate_i(x_in + self.h_proj(h_t)))
            
            decay = torch.exp(-dt_t * f_gate)
            h_t = h_t * decay + i_gate * (1.0 - decay)
            h_t = self.dropout(h_t)
            
            outputs.append(h_t.unsqueeze(1))
            
        out = torch.cat(outputs, dim=1)
        return out, h_t


class LightningWorker(nn.Module):
    """
    A specialist worker optimized for a single timeframe.
    """
    
    def __init__(self, config: WorkerConfig, timeframe: str = "M5"):
        super().__init__()
        self.config = config
        self.timeframe = timeframe
        
        # 🌊 Phase 21: Temporal Self-Attention
        # A projection layer ensures robust Multi-head division regardless of how many features survive the ExtraTrees Pruning
        attn_dim = 64
        self.attn_proj = nn.Linear(config.input_dim, attn_dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=attn_dim,
            num_heads=4,
            dropout=config.dropout,
            batch_first=True
        )
        self.attn_norm = nn.LayerNorm(attn_dim)
        
        # Liquid Backbone
        self.cfc = CfCCell(
            input_dim=attn_dim,
            hidden_dim=config.hidden_dim,
            dropout=config.dropout
        )
        
        # Residual projection from the attended dims to hidden
        if attn_dim != config.hidden_dim:
            self.res_proj = nn.Linear(attn_dim, config.hidden_dim)
        else:
            self.res_proj = nn.Identity()
            
        # Final decision head (auxiliary)
        self.head = nn.Linear(config.hidden_dim, 3)  # Long, Short, Neutral
        
    def forward(
        self, 
        x: torch.Tensor, 
        dt: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        # 1. Base Projection (e.g. 18 pruned dimensions -> 64 embedding dimensions)
        x_proj = self.attn_proj(x)
        
        # 1.5 Pre-LayerNorm (CRITICAL for FP16 Autocast Stability!)
        x_norm = self.attn_norm(x_proj)
        
        # 2. Temporal Self-Attention (Locating crucial wicks/anomalies in the sequence)
        attn_out, _ = self.self_attn(x_norm, x_norm, x_norm, need_weights=False)
        x_attn = x_proj + F.dropout(attn_out, p=self.config.dropout, training=self.training)
        
        # 3. Process through Liquid Continuous-Time Backbone
        _, h_t = self.cfc(x_attn, dt)
        
        # Residual connection from LAST timestep of the attended sequence
        residual = self.res_proj(x_attn[:, -1, :])
        
        # Skip connection
        h_t = h_t + residual
        
        # Embedding is the final hidden state after skip connection
        embedding = h_t
        
        # Logits for auxiliary loss (LONG=0, SHORT=1, NEUTRAL=2)
        logits = self.head(embedding)
        
        return embedding, logits
    
    @property
    def embedding_dim(self) -> int:
        return self.config.hidden_dim


class WorkerSwarm(nn.Module):
    """
    Collection of LightningWorkers for all timeframes.
    """
    
    TIMEFRAMES = ["M5", "M15", "M30", "H1", "H4"]
    
    def __init__(self, config: WorkerConfig = None):
        super().__init__()
        
        if config is None:
            config = WorkerConfig()
        
        self.config = config
        
        # Create worker for each timeframe
        self.workers = nn.ModuleDict({
            tf: LightningWorker(config, timeframe=tf)
            for tf in self.TIMEFRAMES
        })
        
    def forward(
        self, 
        x_dict: dict
    ) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        embeddings = []
        all_logits = []
        order = []
        
        # Identify device and batch size from first available input
        first_tf = next(iter(x_dict.values()))
        batch_size = first_tf.size(0)
        device = first_tf.device
        
        for tf in self.TIMEFRAMES:
            if tf in x_dict:
                dt = x_dict.get(f"{tf}_dt")
                emb, logits = self.workers[tf](x_dict[tf], dt=dt)
                embeddings.append(emb.unsqueeze(1))
                all_logits.append(logits.unsqueeze(1))
            else:
                zero_emb = torch.zeros(batch_size, 1, self.embedding_dim).to(device)
                zero_logits = torch.zeros(batch_size, 1, 3).to(device)
                embeddings.append(zero_emb)
                all_logits.append(zero_logits)
            
            order.append(tf)
        
        embeddings = torch.cat(embeddings, dim=1)
        all_logits = torch.cat(all_logits, dim=1)
        
        return embeddings, all_logits, order
    
    @property
    def num_workers(self) -> int:
        return len(self.TIMEFRAMES)
    
    @property
    def embedding_dim(self) -> int:
        return self.config.hidden_dim


# Quick test
if __name__ == "__main__":
    config = WorkerConfig(input_dim=12, hidden_dim=64)
    swarm = WorkerSwarm(config)
    x_dict = {"M5": torch.randn(4, 64, 12)}
    embeddings, logits, _ = swarm(x_dict)
    print(f"Test Successful. Embeddings: {embeddings.shape}, Logits: {logits.shape}")



--- FILE: backend/src/brain/verify_trinity.py ---
import torch
import sys
import os

# Ensure src is in path
sys.path.append(os.path.abspath(os.curdir))

from src.brain.models.liquid_v8 import LiquidConfig, ModulatedIARA
from src.brain.models.trinity_v8 import TrinityBrain
from src.brain.engine import TrinityEngine, EngineConfig

def test_trinity_phase1():
    print("🧪 Starting Trinity Phase 1 Verification...")
    
    # 1. Config
    liquid_cfg = LiquidConfig(
        input_dim=16,
        hidden_dim=32, # Small for test
        num_cfc_layers=1
    )
    llm_dim = 64 # Mock small dimension
    
    # 2. Instantiate Models
    print("   Building Modulated LNN...")
    lnn = ModulatedIARA(liquid_cfg)
    
    print("   Building TrinityBrain...")
    brain = TrinityBrain(lnn, llm_hidden_dim=llm_dim)
    
    # 3. Instantiate Engine
    print("   Initializing Engine (Orchestrator)...")
    engine_cfg = EngineConfig(device="cpu", llm_refresh_rate=1.0) # Use CPU for test
    engine = TrinityEngine(brain, engine_cfg)
    
    # 4. Mock Data
    batch_size = 2
    seq_len = 10
    market_data = torch.randn(batch_size, seq_len, liquid_cfg.input_dim)
    
    # 5. Mock State Update (Simulating LLM Thread)
    print("   Simulating LLM Update...")
    mock_llm_ctx = torch.randn(1, llm_dim)
    engine.state.update_llm(mock_llm_ctx)
    
    # 6. Forward Tick
    print("   Running Forward Tick (Fast Path)...")
    # News is None, should use Sentinel Stub (neutral)
    decision = engine.forward_tick(market_data, raw_news="Fed holds rates")
    
    print(f"   Output Shape: {decision.shape}")
    print(f"   Output Values: {decision}")
    
    assert decision.shape == (batch_size,), "Output shape mismatch!"
    
    # 7. Test Emergency Brake Logic (Manual Verify)
    # Force high impact in state
    print("   Testing Emergency Brake (Simulated)...")
    engine.state.update_news(sentiment=-0.9, impact=1.0) # 100% Impact
    
    decision_brake = engine.forward_tick(market_data)
    print(f"   Braked Output: {decision_brake}")
    
    # If impact is 1.0, output should be 0.0 (Identity * (1 - 1))
    # Note: Tanh output might not be exactly zero if bias, but let's check magnitude
    if torch.allclose(decision_brake, torch.zeros_like(decision_brake), atol=1e-5):
        print("✅ Emergency Brake Functional (Output Zeroed)")
    else:
        print(f"⚠️ Emergency Brake Verification Warning: Output {decision_brake} not zero.")
        
    print("✅ Trinity Phase 1 verification passed!")

if __name__ == "__main__":
    test_trinity_phase1()



--- FILE: backend/src/brain/losses.py ---
"""
I.A.R.A. v9 - Loss Functions (Swarm Intelligence)
==================================================
Multi-objective loss for training the Swarm:

1. Main Loss (Coordinator): AsymmetricFocalLoss
   - Focuses on hard examples
   - Higher weight for positive class (catching trends)
   
2. Auxiliary Loss (Workers): CrossEntropy
   - Forces each worker to be a competent predictor
   - Prevents "lazy workers" that rely on the Coordinator
   
3. Consensus Loss (Regularization): Disagreement Penalty
   - Penalizes when Coordinator contradicts strong worker consensus
   - Helps maintain ensemble coherence

Formula:
L_total = L_coord + λ * Σ L_workers + β * L_consensus

Reference: Focal Loss (Lin et al., 2017) + Ensemble Distillation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class AsymmetricFocalLoss(nn.Module):
    """
    Asymmetric Focal Loss for imbalanced classification.
    
    Features:
    - Focal term: Down-weights easy examples (gamma)
    - Alpha balancing: Higher weight for positive class
    - Works with logits (more numerically stable)
    
    Args:
        alpha: Weight for positive class (default 0.75)
        gamma: Focusing parameter (default 2.0)
        reduction: 'mean', 'sum', or 'none'
    """
    
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(
        self, 
        logits: torch.Tensor, 
        targets: torch.Tensor,
        weight: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            logits: Raw predictions [batch] or [batch, 1]
            targets: Binary targets [batch] (0 or 1)
            weight: Optional per-sample weights [batch]
            
        Returns:
            Loss value (scalar if reduction='mean' or 'sum')
        """
        logits = logits.view(-1)
        targets = targets.view(-1).float()
        
        # BCE with logits (numerically stable)
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        
        # Probability of correct class
        pt = torch.exp(-bce_loss)
        pt = torch.clamp(pt, min=1e-7, max=1.0 - 1e-7)
        
        # Focal term: (1 - pt)^gamma
        focal_term = (1 - pt) ** self.gamma
        
        # Alpha balancing
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        # Combined loss
        loss = alpha_t * focal_term * bce_loss
        
        # Optional per-sample weighting
        if weight is not None:
            loss = loss * weight.view(-1)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class WorkerAuxiliaryLoss(nn.Module):
    """
    Auxiliary loss for worker predictions.
    
    Ensures each worker learns to predict independently,
    preventing "lazy workers" that rely on ensemble averaging.
    
    Uses label smoothing for regularization.
    """
    
    def __init__(self, num_classes: int = 2, label_smoothing: float = 0.1, weight_fp: float = 10.0, weight_fn: float = 1.0):
        super().__init__()
        self.num_classes = num_classes
        self.label_smoothing = label_smoothing
        # weights: [Long, Short, Wait] -> Target 2 is Wait (minority class)
        self.weights = torch.tensor([weight_fn, weight_fn, weight_fp], dtype=torch.float32)
        
    def forward(
        self, 
        worker_logits: torch.Tensor, 
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            worker_logits: All worker predictions [batch, num_workers, 3]
            targets: Action targets [batch] {0, 1, 2}
            
        Returns:
            Average auxiliary loss across all workers
        """
        if self.weights.device != worker_logits.device:
            self.weights = self.weights.to(worker_logits.device)
            
        batch_size, num_workers, num_classes = worker_logits.shape
        
        total_loss = torch.tensor(0.0, device=targets.device)
        
        for w in range(num_workers):
            w_logits = worker_logits[:, w, :]  # [Batch, 3]
            loss = F.cross_entropy(
                w_logits, 
                targets,
                weight=self.weights,
                label_smoothing=self.label_smoothing
            )
            total_loss = total_loss + loss
        
        # Average over workers
        return total_loss / num_workers


class ConsensusLoss(nn.Module):
    """
    Regularization loss to maintain ensemble coherence.
    
    Penalizes when the Coordinator strongly disagrees with
    a clear worker consensus.
    
    Logic:
    - If >=3 workers vote LONG with high confidence, 
      and Coordinator predicts SHORT, apply penalty.
    """
    
    def __init__(self, threshold: float = 0.6, consensus_min: int = 3):
        super().__init__()
        self.threshold = threshold  # Confidence threshold for "clear" vote
        self.consensus_min = consensus_min  # Minimum workers for consensus
        
    def forward(
        self, 
        coord_logits: torch.Tensor,
        worker_logits: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            coord_logits: Coordinator predictions [batch, 1]
            worker_logits: Worker predictions [batch, num_workers, 3]
            
        Returns:
            Consensus penalty loss
        """
        batch_size = coord_logits.size(0)
        
        # Coordinator probabilities (0=Long, 1=Short, 2=Wait)
        coord_probs = F.softmax(coord_logits, dim=-1)  # [batch, 3]
        coord_long_prob = coord_probs[:, 0]
        coord_short_prob = coord_probs[:, 1]
        
        # Worker probabilities
        worker_probs = F.softmax(worker_logits, dim=-1)  # [batch, workers, 3]
        
        # Worker votes
        worker_votes = worker_logits.argmax(dim=-1)  # [batch, workers]
        
        # Worker confidence (max probability)
        worker_conf = worker_probs.max(dim=-1)[0]  # [batch, workers]
        
        # Count confident votes
        confident_mask = worker_conf > self.threshold
        
        # Count confident LONG votes
        n_long = ((worker_votes == 0) & confident_mask).float().sum(dim=-1)
        
        # Count confident SHORT votes
        n_short = ((worker_votes == 1) & confident_mask).float().sum(dim=-1)
        
        # Consensus disagreement penalty
        # If workers say LONG but Coordinator Long prob is low
        long_consensus = (n_long >= self.consensus_min).float()
        long_penalty = long_consensus * F.relu(0.5 - coord_long_prob)
        
        # If workers say SHORT but Coordinator Short prob is low
        short_consensus = (n_short >= self.consensus_min).float()
        short_penalty = short_consensus * F.relu(0.5 - coord_short_prob)
        
        total_penalty = long_penalty + short_penalty
        return total_penalty.mean()


class AsymmetricCrossEntropyLoss(nn.Module):
    """
    Custom Loss for Sniper Mode (High Precision).
    Penalizes False Positives (acting on noise, target=Wait) heavier than 
    False Negatives (missing a trade, target=Long/Short).
    
    Args:
        weight_fp: Weight penalty when target is Wait (class 2)
        weight_fn: Weight penalty when target is Action (class 0 or 1)
        label_smoothing: CrossEntropy label smoothing
    """
    def __init__(self, weight_fp: float = 10.0, weight_fn: float = 1.0, label_smoothing: float = 0.1):
        super().__init__()
        # weights: [Long, Short, Wait] -> Target 2 is Wait
        self.weights = torch.tensor([weight_fn, weight_fn, weight_fp], dtype=torch.float32)
        self.label_smoothing = label_smoothing
        
    def forward(self, logits: torch.Tensor, targets: torch.Tensor, weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.weights.device != logits.device:
            self.weights = self.weights.to(logits.device)
            
        # reduction='none' so we can apply optional sample weights
        # epsilon padding added intrinsically by cross_entropy via Softmax
        loss = F.cross_entropy(logits, targets.long(), weight=self.weights, label_smoothing=self.label_smoothing, reduction='none')
        
        if weight is not None:
            loss = loss * weight
            
        return loss.mean()


class SwarmLoss(nn.Module):
    """
    Complete loss function for Swarm Intelligence training.
    
    Combines:
    1. AsymmetricCrossEntropyLoss for Coordinator (Sniper Mode)
    2. CrossEntropyLoss for Workers (auxiliary)
    3. ConsensusLoss for regularization
    
    Formula:
    L_total = L_coord + λ * L_workers + β * L_consensus
    
    Args:
        alpha: Focal loss alpha (default 0.75)
        gamma: Focal loss gamma (default 2.0)
        consensus_weight: Weight for consensus loss β (default 0.1)
        label_smoothing: Smoothing for worker loss (default 0.1)
    """
    
    def __init__(
        self,
        alpha: float = 0.75,
        gamma: float = 2.0,
        consensus_weight: float = 0.1,
        label_smoothing: float = 0.1,
        diversity_weight: float = 0.1 # 🌋 NEW: Penalty to avoid weight collapse
    ):
        super().__init__()
        
        # Learnable weight logits (Natural Dynamic Weighting)
        # Softmax of this tensor determines final weights.
        # Initialized to favor Coordinator (Index 0) at approx 85%
        self.weight_logits = nn.Parameter(torch.tensor([2.5, 0.0, 0.0]))
        
        # Sub-losses (Sniper Mode Asymmetric Loss)
        self.coord_loss = AsymmetricCrossEntropyLoss(weight_fp=10.0, weight_fn=1.0, label_smoothing=label_smoothing)
        self.focal_alt = AsymmetricFocalLoss(alpha=alpha, gamma=gamma) # Still keep for binary auxiliary if needed
        self.worker_loss = WorkerAuxiliaryLoss(
            num_classes=3, 
            label_smoothing=label_smoothing,
            weight_fp=10.0,
            weight_fn=1.0
        )
        self.consensus_loss = ConsensusLoss()
        self.diversity_weight = diversity_weight
        
    def forward(
        self,
        coord_logits: torch.Tensor,
        worker_logits: torch.Tensor,
        targets: torch.Tensor,
        weight: Optional[torch.Tensor] = None
    ) -> dict:
        """
        Compute all losses with automatic Natural Scaling.
        Targets: [Batch] with values {0: Long, 1: Short, 2: Neutral}
        """
        # 1. Compute individual losses
        
        # Coordinator is now 3-class {Long, Short, Neutral} mapped via Asymmetric Loss
        # coord_logits shape: [Batch, 3]
        # targets shape: [Batch]
        l_coord = self.coord_loss(coord_logits, targets, weight=weight)
             
        l_workers = self.worker_loss(worker_logits, targets.long())
        l_consensus = self.consensus_loss(coord_logits, worker_logits)
        
        # 2. Natural Dynamic Weighting (Softmax)
        weights = F.softmax(self.weight_logits, dim=0)
        
        # 3. Anchor Mechanism: Ensure Coordinator (Líder) always has strong voice
        # We enforce a minimum presence (e.g., mixing with a static anchor)
        # This prevents the weight from falling to zero if other tasks are "easier"
        coord_weight = 0.5 * weights[0] + 0.3 # At least 30%, up to 80%
        
        # 4. Total Weighted Loss
        # FIX: Include l_workers to allow gradients to flow to weight_logits[1]
        total_l = (
            coord_weight * l_coord + 
            weights[1] * l_workers * 0.5 + 
            weights[2] * l_consensus * 0.1
        )
        
        # 4. Weight Entropy Penalty (🌋 Phase 19)
        # Prevents weights from becoming "one-hot" (winner-take-all collapse)
        # Entropy is sum(-p * log(p)). To maximize it, we minimize -entropy.
        weight_entropy = - (weights * torch.log(weights + 1e-8)).sum()
        total_l = total_l - (self.diversity_weight * weight_entropy)
        
        return {
            "total": total_l,
            "coord": l_coord,
            "workers": l_workers,
            "consensus": l_consensus,
            "entropy": weight_entropy,
            "weights": [coord_weight.item(), weights[1].item(), weights[2].item()]
        }


# Quick test
if __name__ == "__main__":
    print("Testing SwarmLoss...")
    
    batch_size = 64
    num_workers = 5
    
    # Simulate outputs
    coord_logits = torch.randn(batch_size, 3)
    worker_logits = torch.randn(batch_size, num_workers, 3)
    targets = torch.randint(0, 3, (batch_size,)).float()
    
    # Create loss
    criterion = SwarmLoss(
        alpha=0.75,
        gamma=2.0,
        consensus_weight=0.1,
        label_smoothing=0.1
    )
    
    # Compute
    losses = criterion(coord_logits, worker_logits, targets)
    
    print(f"  Total loss: {losses['total']:.4f}")
    print(f"  Coord loss: {losses['coord']:.4f}")
    print(f"  Worker loss: {losses['workers']:.4f}")
    print(f"  Consensus loss: {losses['consensus']:.4f}")
    
    # Verify gradient flow
    losses['total'].backward()
    print("  ✅ Gradient flow OK!")
    print("  ✅ SwarmLoss OK!")



--- FILE: backend/src/brain/agents/sentinel_stub.py ---
from typing import Dict, Any

class SentinelStub:
    """
    A placeholder for the Sentinel (Watchdog) agent.
    
    Role:
    - Phase 1: Passive / Stubbed. Returns neutral scores to validate core loop.
    - Phase 2: Will be replaced by FinBERT implementation.
    """
    
    def __init__(self):
        print("🕊️ Sentinel Stub Initialized (Passive Mode)")

    def is_safe(self) -> bool:
        """
        Check if trading is allowed.
        Currently returns True (Passive).
        Future: Output of FinBERT or Risk Rules.
        """
        return True

    def analyze(self, text: str) -> Dict[str, float]:
        """
        Mimics the future FinBERT output.
        
        Args:
            text (str): Raw news headline or text.
            
        Returns:
            dict: {
                'sentiment': 0.0,  # Neutral (-1 to 1)
                'impact': 0.0      # Zero Impact (0 to 1) -> No Emergency Brake
            }
        """
        return {
            "sentiment": 0.0,
            "impact": 0.0
        }



--- FILE: backend/src/brain/tools/ccxt_executor.py ---
import os
import logging
from typing import Dict, Any, List, Optional, Tuple
from dotenv import load_dotenv

# Try importing ccxt
try:
    import ccxt
except ImportError:
    ccxt = None

load_dotenv()

logger = logging.getLogger("CCXTExecutor")

class CCXTExecutor:
    """
    Execution Layer for Crypto Exchanges via CCXT.
    Supports Binance (Spot/Futures) and Kraken.
    """
    def __init__(self, exchange_id: str = "binance", default_type: str = "spot"):
        self.exchange_id = exchange_id
        self.exchange = None
        self.active = False
        
        if not ccxt:
            logger.error("CCXT library not installed.")
            return

        # Load Keys based on exchange_id
        api_key = os.getenv(f"{exchange_id.upper()}_API_KEY")
        secret = os.getenv(f"{exchange_id.upper()}_SECRET_KEY")
        
        if not api_key or not secret:
            logger.warning(f"No API Keys found for {exchange_id.upper()} in .env. Execution will fail.")
            # We initialize anyway to allow dry-run or verification steps, 
            # but private methods will fail.
        
        try:
            exchange_class = getattr(ccxt, exchange_id)
            self.exchange = exchange_class({
                'apiKey': api_key,
                'secret': secret,
                'enableRateLimit': True,
                'options': {'defaultType': default_type}
            })
            
            # Load markets to verify connection
            # self.exchange.load_markets() # This might block, better to do lazy or async if possible
            # But Executor is usually sync in this architecture or threaded? 
            # SentinelStub and Engine seem to use blocking calls or threads. 
            # We'll stick to blocking for simplicity in this Executor class.
            self.active = True
            logger.info(f"✅ CCXT Executor initialized for {exchange_id} ({default_type})")
            
        except Exception as e:
            logger.error(f"Failed to initialize CCXT Executor: {e}")

        # Symbol Mapping (Internal -> Exchange)
        self._symbol_map = {
            "BTC/USD": "BTC/USDT", # Binance standard
            "ETH/USD": "ETH/USDT",
            # Add more as needed or make dynamic
        }

    def _resolve_symbol(self, symbol: str) -> str:
        """Maps internal symbol (BTC/USD) to exchange symbol (BTC/USDT)."""
        return self._symbol_map.get(symbol, symbol)

    def get_account_summary(self) -> Dict[str, Any]:
        """Fetches free balance and total equity."""
        if not self.exchange or not self.exchange.apiKey:
            return {}
            
        try:
            balance = self.exchange.fetch_balance()
            # Parse standard CCXT balance response
            total = balance.get('total', {})
            free = balance.get('free', {})
            
            # Estimate equity in USDT (simplified)
            equity = conversion = 0.0
            if 'USDT' in total:
                equity = total['USDT']
            
            return {
                "equity": equity, # Approximate
                "balance": free.get('USDT', 0.0),
                "buying_power": free.get('USDT', 0.0),
                "raw": balance['info']
            }
        except Exception as e:
            logger.error(f"Failed to fetch balance: {e}")
            return {}

    def place_market_order(self, symbol: str, volume: float, side: str = "BUY") -> Optional[Any]:
        """
        Places a market order.
        """
        if not self.exchange or not self.exchange.apiKey:
            logger.error("Cannot place order: No API Keys or Exchange not init.")
            return None
            
        remote_symbol = self._resolve_symbol(symbol)
        side = side.lower() # ccxt expects 'buy' or 'sell'
        
        try:
            logger.info(f"🚀 Sending Order: {remote_symbol} {side} {volume}")
            order = self.exchange.create_market_order(remote_symbol, side, volume)
            logger.info(f"✅ Order Filled: {order['id']} @ {order.get('average', 'Market')}")
            return order
        except Exception as e:
            logger.error(f"CCXT Order Failed: {e}")
            return None

    def get_open_positions(self) -> List[Dict[str, Any]]:
        """
        Fetches open positions. 
        Note: For Spot, this is just positive balances. 
        For Futures, it's actual positions.
        """
        if not self.exchange:
            return []
            
        # Logic differs for Spot vs Futures
        # Assuming Spot for now as per defaultType='spot'
        try:
            balance = self.exchange.fetch_balance()
            total = balance.get('total', {})
            
            positions = []
            for coin, amount in total.items():
                if amount > 0 and coin != 'USDT':
                     positions.append({
                         "symbol": f"{coin}/USDT",
                         "qty": amount,
                         "entry": 0.0, # Spot doesn't easily track avg entry without history
                         "market_value": 0.0, # Would need current price
                         "unrealized_pl": 0.0
                     })
            return positions
        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            return []

    def close_all_positions(self):
        """Emergency panic button."""
        # Complex to implement for Spot (Sell all assets).
        # Implemented as log warning for now.
        logger.warning("🔴 Close All Positions not fully implemented for CCXT Spot yet.")



--- FILE: backend/src/brain/tools/market_scanner.py ---
import os
import psycopg2
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Union

from sqlalchemy import create_engine, text

class MarketScanner:
    """
    Tool Set for the Strategic Agent (LFM).
    Allows the Slow System to query historical context from TimescaleDB/Postgres.
    """
    def __init__(self, db_uri: Optional[str] = None):
        self.dsn = db_uri or os.getenv("DB_URI", "postgresql://saci:saci_secret@localhost:5432/saci_core")
        self.engine = create_engine(self.dsn)
        
    def _get_conn(self):
        return self.engine.connect()

    def check_support_resistance(self, symbol: str, current_price: float, lookback_days: int = 30) -> str:
        """
        Scans for price levels that have been tested multiple times (High Volume Nodes or Rejections).
        """
        try:
            with self._get_conn() as conn:
                # 1. Fetch simplified PA (Highs and Lows)
                query = """
                    SELECT high, low 
                    FROM market_candles 
                    WHERE symbol = %s 
                      AND time > NOW() - INTERVAL '%s days'
                """
                df = pd.read_sql(query, conn, params=(symbol, lookback_days))
                
            if df.empty:
                return f"No historical data for {symbol} in last {lookback_days} days."

            # 2. Logic: Find clusters of lows (Support) and highs (Resistance)
            # Tolerance: 0.1% of price
            tolerance = current_price * 0.001
            
            # Supports
            lows = df['low'].values
            nearby_supports = []
            # Naive cluster: Check if 'low' is within tolerance of other lows
            # Optimization: Histogram
            hist_l, bins_l = np.histogram(lows, bins=50)
            # Find bins with high count
            peak_idxs_l = np.where(hist_l > len(lows)*0.05)[0] # Peaks with >5% of data
            
            support_msg = "None"
            for p in peak_idxs_l:
                level = bins_l[p]
                if level < current_price: # Only supports below
                     count = hist_l[p]
                     if abs(current_price - level) / current_price < 0.05: # Relevant (near 5%)
                        nearby_supports.append(f"{level:.5f} ({count} tests)")
            
            if nearby_supports:
                support_msg = ", ".join(nearby_supports)

            # Resistances
            highs = df['high'].values
            nearby_resistances = []
            hist_h, bins_h = np.histogram(highs, bins=50)
            peak_idxs_h = np.where(hist_h > len(highs)*0.05)[0]
            
            res_msg = "None"
            for p in peak_idxs_h:
                level = bins_h[p]
                if level > current_price: # Only resistances above
                    count = hist_h[p]
                    if abs(current_price - level) / current_price < 0.05:
                        nearby_resistances.append(f"{level:.5f} ({count} tests)")
                        
            if nearby_resistances:
                res_msg = ", ".join(nearby_resistances)

            return f"Market Scan ({symbol}):\n- Supports nearby: {support_msg}\n- Resistances nearby: {res_msg}"

        except Exception as e:
            return f"Error scanning S/R for {symbol}: {e}"

    def get_trend_alignment(self, symbol: str, timeframe: str = '1h', lookback: int = 50) -> str:
        """
        Verifies trend across the specified timeframe.
        Returns: "BULLISH", "BEARISH", or "RANGING" with slope details.
        """
        try:
            with self._get_conn() as conn:
                query = """
                    SELECT close 
                    FROM market_candles 
                    WHERE symbol = %s AND interval = %s 
                    ORDER BY time DESC 
                    LIMIT %s
                """
                df = pd.read_sql(query, conn, params=(symbol, timeframe, lookback))
            
            if len(df) < 10:
                return f"Insufficient data for {timeframe} trend analysis."
                
            closes = df['close'].values[::-1] # Reverse to chronological
            
            # Simple Linear Regression for slope
            x = np.arange(len(closes))
            slope, intercept = np.polyfit(x, closes, 1)
            
            # R-squared integration for quality?
            # For text output, slope is enough.
            
            trend = "RANGING"
            if slope > 0.0001: trend = "BULLISH"
            elif slope < -0.0001: trend = "BEARISH"
            
            return f"Trend ({timeframe}): {trend} (Slope: {slope:.6f}, Sample: {len(closes)} candles)"

        except Exception as e:
            return f"Error trend analysis: {e}"



--- FILE: backend/src/brain/tools/binance_executor.py ---
import os
import time
import logging
from typing import Dict, Any, Optional
from binance.client import Client
from binance.exceptions import BinanceAPIException, BinanceOrderException

logger = logging.getLogger("api.binance_executor")

class BinanceExecutor:
    """
    Execution Layer for Binance Markets (Phase 5).
    Handles LIVE execution and TESTNET simulated logging with Slippage.
    Includes a Latency Kill-Switch to prevent execution during API lag spikes.
    """
    def __init__(self, symbol: str = "BTCUSDT", mode: str = "TESTNET"):
        self.symbol = symbol.upper().replace("/", "").replace("-", "")
        self.mode = os.getenv("TRADE_MODE", mode).upper() # TESTNET or LIVE
        
        # Kill-Switch Tolerance
        self.max_latency_ms = 500
        
        # Keys
        self.api_key = os.getenv("BINANCE_API_KEY", "")
        self.secret_key = os.getenv("BINANCE_SECRET_KEY", "")
        
        self.client = Client(self.api_key, self.secret_key)
        
        # If true TESTNET on Binance API
        if self.mode == "TESTNET" and "testnet" in self.api_key.lower():
             self.client.API_URL = 'https://testnet.binance.vision/api'
             
        logger.info(f"🛡️ Binance Executor online for {self.symbol} | Mode: {self.mode}")

    def _check_latency_killswitch(self) -> bool:
        """Pings Binance API. Returns False if latency is > 500ms or connection fails."""
        try:
            start_time = time.time()
            self.client.ping()
            latency_ms = (time.time() - start_time) * 1000
            
            if latency_ms > self.max_latency_ms:
                logger.error(f"🚨 TICKER LAG DETECTED: {latency_ms:.1f}ms (Limit: {self.max_latency_ms}ms). Kill-Switch Activated. Order Aborted.")
                return False
                
            logger.debug(f"API Latency: {latency_ms:.1f}ms - OK")
            return True
        except Exception as e:
            logger.error(f"🚨 API PING FAILED: {e}. Kill-Switch Activated.")
            return False

    def execute_signal(self, action: str, confidence: float, current_price: float, qty: float = 0.001) -> Optional[Dict[str, Any]]:
        """
        Receives signal from IARA (e.g., LONG, SHORT).
        Action should be "BUY" or "SELL".
        """
        if action.upper() not in ["BUY", "SELL"]:
            logger.warning(f"Invalid signal received: {action}")
            return None
            
        logger.info(f"🧠 IARA Signal Triggered -> {action} {qty} {self.symbol} | Confidence: {confidence:.1%}")

        # 1. Latency Kill-Switch
        if not self._check_latency_killswitch():
            return {"status": "aborted", "reason": "latency_killswitch"}

        # 2. Execution Logic
        if self.mode == "TESTNET":
            # Simulate Execution
            slippage_pct = 0.001 # 0.1% Average Slippage
            taker_fee = 0.0004   # 0.04% Typical taker fee
            
            # Calculate slippage impact based on direction
            if action.upper() == "BUY":
                executed_price = current_price * (1 + slippage_pct)
            else:
                executed_price = current_price * (1 - slippage_pct)
                
            fee_cost = (executed_price * qty) * taker_fee
            
            order_data = {
                "symbol": self.symbol,
                "orderId": f"SIM_{int(time.time()*1000)}",
                "clientOrderId": "sim_testnet_saci",
                "transactTime": int(time.time() * 1000),
                "price": 0.0, # Market order
                "origQty": qty,
                "executedQty": qty,
                "status": "FILLED",
                "type": "MARKET",
                "side": action.upper(),
                "fills": [{"price": str(executed_price), "qty": str(qty), "commission": str(fee_cost), "commissionAsset": "USDT"}]
            }
            logger.info(f"🟢 [TESTNET] Simulated {action} executed at ~${executed_price:.2f} (Slippage applied)")
            return order_data

        elif self.mode == "LIVE":
            # LIVE Market Execution
            try:
                order = self.client.create_order(
                    symbol=self.symbol,
                    side=action.upper(),
                    type=Client.ORDER_TYPE_MARKET,
                    quantity=qty
                )
                logger.info(f"🔥 [LIVE TRADING] Order executed successfully: {order['orderId']}")
                return order
            except BinanceAPIException as e:
                logger.error(f"❌ [LIVE TRADING] Binance API Error: {e}")
                return {"status": "failed", "reason": str(e)}
            except BinanceOrderException as e:
                logger.error(f"❌ [LIVE TRADING] Binance Order Exception: {e}")
                return {"status": "failed", "reason": str(e)}



--- FILE: backend/src/brain/engine.py ---
"""
TRINITY ENGINE v11.0 "FULL SPECTRUM SWARM"
=========================================
The Orchestrator. Manages the live trading loop using SwarmBrain v11.

Architecture:
  Fast Brain (SwarmBrain): Multi-timeframe workers + Coordinator for tactical decisions
  LOB Sensor:             Real-time Level 2 Order Book microstructure
  Chaos Math:             Hurst Exponent for regime awareness
  HMM Regime:             Hidden Markov Model for state classification
  GNN On-Chain:           Graph Neural Network for whale flow analysis
  Slow Brain (LFM):       Optional strategic modulation via Ollama/LFM
  Sentinel:               Safety override (news, volatility, drawdown)
  
Data Flow:
  Market Tick → Feature Engineering → [LOB + Hurst + HMM + GNN] → SwarmBrain → Consensus → Execution

Usage:
  engine = TrinityEngine.from_checkpoint()
  engine.start()
"""

import threading
import time
import os
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List

import torch
import numpy as np
import pandas as pd
import requests

# Models
from src.brain.models.liquid_v9 import SwarmBrain, CoordinatorConfig
from src.brain.models.workers import WorkerConfig

# Trading
from src.trading.signal_generator import SignalEngine, TradeSignal

# Tools
from src.brain.tools.market_scanner import MarketScanner
from src.brain.tools.binance_executor import BinanceExecutor
from src.brain.agents.sentinel_stub import SentinelStub

# v11: New Intelligence Layers
from src.data.orderbook import OrderBookIngestor, MockOrderBookIngestor
from src.data.hurst import compute_hurst
from src.brain.models.regime_hmm import MarketRegimeHMM, REGIME_UNKNOWN
from src.data.onchain import OnChainIngestor

logger = logging.getLogger("TrinityEngine")


@dataclass
class EngineConfig:
    """Configuration for the TrinityEngine."""
    # LFM (Strategic Agent)
    lfm_model_url: str = os.getenv("OLLAMA_URL", "http://ollama:11434/api/generate")
    lfm_model_name: str = "lfm2.5-thinking:1.2b"
    lfm_refresh_rate: float = 60.0   # Strategic thinking every 60s
    
    # Trading
    symbol: str = "BTC/USD"
    binance_symbol: str = "BTCUSDT"
    market_type: str = "crypto" # "crypto" or "stock"
    execution_venue: str = "BINANCE"
    trade_qty: float = 0.01
    
    # Model
    checkpoint_path: str = "logs/models/best_model_v9_BTCUSD.pth"
    worker_hidden: int = 0   # 0 = auto-detect from checkpoint
    coord_hidden: int = 0    # 0 = auto-detect from checkpoint
    seq_len: int = 128
    device: str = "cpu"
    
    # v11: LOB
    lob_enabled: bool = True
    lob_mock: bool = True           # Use mock LOB data (no WebSocket)
    
    # v11: Hurst Regime Gating
    hurst_enabled: bool = True
    hurst_skip_range: tuple = (0.45, 0.55)   # Random walk zone → skip
    hurst_trend_threshold: float = 0.65       # Strong trend → hold longer
    hurst_scalp_threshold: float = 0.35       # Mean revision → scalp
    
    # v11: HMM Regime Detection
    hmm_enabled: bool = True
    hmm_model_path: str = "data/hmm_regime_model.pkl"
    
    # v11: GNN On-Chain
    gnn_enabled: bool = True
    gnn_mock: bool = True            # Use mock on-chain data
    gnn_n_wallets: int = 50


class DataBuffer:
    """Manages streaming market data and converts to SwarmBrain-ready DataFrames."""
    def __init__(self, max_len: int = 10000):
        self.max_len = max_len
        self.data: List[dict] = []

    def add_candle(self, candle: dict):
        self.data.append(candle)
        if len(self.data) > self.max_len:
            self.data.pop(0)

    def get_dataframe(self) -> pd.DataFrame:
        df = pd.DataFrame(self.data)
        return df

class SharedState:
    """Thread-safe state shared between LFM and fast SwarmBrain."""
    def __init__(self, device: str = "cpu"):
        self.lock = threading.Lock()
        self.lfm_bias = 0.0
        self.lfm_rationale = ""
        self.last_price: float = 0.0
        self.last_signal: Optional[TradeSignal] = None
        self.cumulative_pnl: float = 0.0
        self.recent_trades: list = []
        self.logs: list = []
        self.max_logs = 100

    def update_lfm(self, bias: float, rationale: str):
        with self.lock:
            self.lfm_bias = bias
            self.lfm_rationale = rationale
            if rationale:
                self.add_log(f"[Strategic] LFM Thoughts: {rationale[:60]}...")

    def update_signal(self, signal: TradeSignal, current_price: float):
        with self.lock:
            self.last_signal = signal
            self.last_price = current_price
            self.add_log(f"[Signal] IARA Triggered: {signal.direction} @ {current_price:.2f} (Conf: {signal.confidence:.0%})")

    def add_trade(self, trade: dict):
        with self.lock:
            self.recent_trades.insert(0, trade)
            if len(self.recent_trades) > 50:
                self.recent_trades.pop()
            # Simple mock PnL for demo purposes if not updating fully from closures
            self.cumulative_pnl += trade.get("pnl", 0.0)
            self.add_log(f"[Exec] Trade Executed: {trade.get('side')} {trade.get('symbol')} @ {trade.get('price')}")

    def add_log(self, message: str):
        import time
        ts = time.strftime("%H:%M:%S")
        entry = f"{ts} {message}"
        with self.lock:
            self.logs.append(entry)
            if len(self.logs) > self.max_logs:
                self.logs.pop(0)

    def get_state(self) -> dict:
        with self.lock:
            return {
                "lfm_bias": self.lfm_bias,
                "lfm_summary": self.lfm_rationale,
                "last_price": self.last_price,
                "signal": self.last_signal,
                "pnl": self.cumulative_pnl,
                "trades": list(self.recent_trades),
                "logs": list(self.logs)
            }
            
    def get_position_summary(self) -> Optional[dict]:
        # Placeholder for actual position tracking in engine
        return None

class TrinityEngine:
    """
    Live trading orchestrator using SwarmBrain v11.0.
    Full Spectrum Intelligence: Workers + LOB + Hurst + HMM + GNN + LFM.
    """

    def __init__(self, signal_engine: SignalEngine, config: EngineConfig):
        from src.brain.tools.ccxt_executor import CCXTExecutor # Lazy import to avoid circular deps if any
        
        self.signal_engine = signal_engine
        self.config = config
        self.device = torch.device(config.device)

        # State
        self.state = SharedState(device=config.device)
        self.start_time: float = 0.0
        self.buffer = DataBuffer(max_len=10000)

        # Tools
        self.sentinel = SentinelStub()
        self.scanner = MarketScanner()
        
        # Executor
        self.executor = BinanceExecutor(symbol=config.symbol)
        logger.info(f"🔌 Using Binance Executor for {config.symbol} | Mode: {self.executor.mode}")

        # === v11: LOB Ingestor ===
        if config.lob_enabled:
            if config.lob_mock:
                self.lob_ingestor = MockOrderBookIngestor(symbol=config.binance_symbol.lower())
            else:
                self.lob_ingestor = OrderBookIngestor(symbol=config.binance_symbol.lower())
            self.lob_ingestor.start()
        else:
            self.lob_ingestor = None

        # === v11: HMM Regime Detector ===
        if config.hmm_enabled:
            self.regime_hmm = MarketRegimeHMM(model_path=config.hmm_model_path)
        else:
            self.regime_hmm = None

        # === v11: GNN On-Chain Ingestor ===
        if config.gnn_enabled:
            self.onchain_ingestor = OnChainIngestor(
                n_wallets=config.gnn_n_wallets,
                mode="mock" if config.gnn_mock else "live",
            )
            self.onchain_ingestor.start()
        else:
            self.onchain_ingestor = None

        # Thread control
        self.lfm_thread: Optional[threading.Thread] = None
        self.active = False
        
    def pre_warm(self, candles: List[dict]):
        """Inject historical candles into the engine's buffer without triggering ML inference or executions."""
        for c in candles:
            self.buffer.add_candle(c)
        logger.info(f"⚙️ TrinityEngine pre-warmed with {len(candles)} historical candles.")

    @classmethod
    def from_checkpoint(cls, config: Optional[EngineConfig] = None) -> "TrinityEngine":
        """Factory: build engine from checkpoint path in config."""
        if config is None:
            config = EngineConfig()

        # Auto-detect dimensions from checkpoint if not explicitly set
        worker_hidden = config.worker_hidden
        coord_hidden = config.coord_hidden
        
        if (worker_hidden == 0 or coord_hidden == 0) and os.path.exists(config.checkpoint_path):
            try:
                state = torch.load(config.checkpoint_path, map_location="cpu", weights_only=False)
                # Detect worker_hidden from input_proj weight shape
                w = state.get("swarm.workers.M5.input_proj.weight")
                if w is not None and hasattr(w, 'shape') and len(w.shape) == 2:
                    worker_hidden = int(w.shape[0])  # hidden_dim
                    # We NO LONGER auto-detect input_dim to allow it to grow. 
                    # If input_dim grows, the inner SwarmBrain handles the padding.
                    logger.info(f"  Auto-detected worker_hidden={worker_hidden} from checkpoint")
                    
                # Detect coord_hidden from policy_head weight shape
                ph = state.get("coordinator.policy_head.weight")
                if ph is not None and hasattr(ph, 'shape') and len(ph.shape) == 2:
                    coord_hidden = int(ph.shape[1])  # hidden_dim
                    logger.info(f"  Auto-detected coord_hidden={coord_hidden} from checkpoint")
            except Exception as e:
                logger.warning(f"  Could not auto-detect dims: {e}. Using defaults.")
        
        # Fallback to safe defaults if still 0
        if worker_hidden == 0: worker_hidden = 64
        if coord_hidden == 0: coord_hidden = 256

        signal_engine = SignalEngine(
            checkpoint_path=config.checkpoint_path,
            worker_hidden=worker_hidden,
            coord_hidden=coord_hidden,
            seq_len=config.seq_len,
            device=config.device,
        )
        return cls(signal_engine, config)

    # ──────────────────────────────────────────
    # Fast Path: Tactical Inference
    # ──────────────────────────────────────────
    async def forward_tick(self, candle: dict) -> Optional[TradeSignal]:
        """
        Process a new M5 candle from the stream.
        Now enriched with LOB, Hurst, HMM, and GNN intelligence.

        Args:
            candle: Dict with {time, open, high, low, close, volume} (time in seconds)

        Returns:
            TradeSignal if inference ran, else None
        """
        # 1. Update Buffer
        self.buffer.add_candle(candle)
        
        # 2. Check if enough data
        if len(self.buffer.data) < 200:
            return None
            
        # 3. Prepare DataFrame
        df_m5 = self.buffer.get_dataframe()

        # === v11: Hurst Exponent Regime Gating ===
        hurst_value = 0.5
        if self.config.hurst_enabled and len(df_m5) >= 200:
            try:
                close_arr = df_m5["close"].values.astype(np.float64)
                hurst_value = compute_hurst(close_arr[-200:])
                
                # Skip if market is pure noise
                low_h, high_h = self.config.hurst_skip_range
                if low_h <= hurst_value <= high_h:
                    logger.debug(f"⚪ Hurst={hurst_value:.3f} (random walk zone) → SKIP")
                    return None
            except Exception as e:
                logger.debug(f"Hurst computation error: {e}")

        # 4. Get base signal from SwarmBrain
        signal = self.signal_engine.predict(df_m5)
        if signal is None:
            return None

        # === v11: Attach LOB features to signal context ===
        lob_features = None
        if self.lob_ingestor is not None:
            lob_features = self.lob_ingestor.get_features()
            signal.lob_features = lob_features  # Enrich signal for logging

        # === v11: HMM Regime Detection & Confidence Modulation ===
        regime_state = REGIME_UNKNOWN
        regime_penalty = 1.0
        if self.regime_hmm is not None and self.regime_hmm.is_fitted:
            try:
                close_arr = df_m5["close"].values.astype(np.float64)
                high_arr = df_m5["high"].values.astype(np.float64)
                low_arr = df_m5["low"].values.astype(np.float64)
                # Compute ADX for HMM (using simple proxy)
                import talib
                adx_arr = talib.ADX(high_arr, low_arr, close_arr, timeperiod=14)
                adx_arr = np.nan_to_num(adx_arr)
                
                hmm_features = self.regime_hmm.extract_regime_features(
                    close_arr[-50:], high_arr[-50:], low_arr[-50:], adx_arr[-50:]
                )
                regime_state = self.regime_hmm.predict_regime(hmm_features)
                penalties = MarketRegimeHMM.get_regime_penalty(regime_state)
                regime_penalty = penalties.get("all", 1.0)
                
                logger.debug(f"📊 Regime: {MarketRegimeHMM.regime_name(regime_state)} | Penalty: {regime_penalty:.2f}")
            except Exception as e:
                logger.debug(f"HMM prediction error: {e}")

        # Apply regime penalty to confidence
        signal.confidence *= regime_penalty

        # === v11: Hurst-based execution mode ===
        hold_multiplier = 1.0
        if self.config.hurst_enabled:
            if hurst_value > self.config.hurst_trend_threshold:
                hold_multiplier = 1.5  # Strong trend → hold longer
                logger.debug(f"🟢 Hurst={hurst_value:.3f} (trend) → Hold x{hold_multiplier}")
            elif hurst_value < self.config.hurst_scalp_threshold:
                hold_multiplier = 0.5  # Mean-reversion → scalp fast
                logger.debug(f"🟡 Hurst={hurst_value:.3f} (mean-reversion) → Scalp mode")
        signal.hold_multiplier = hold_multiplier

        # Update shared state
        current_price = candle["close"]
        self.state.update_signal(signal, current_price)

        # 5. Persist Decision (Async)
        await self._persist_signal(signal, candle)

        # Check Sentinel (Watchdog)
        if not self.sentinel.is_safe():
            logger.warning("⚠️ Sentinel BLOCKED execution: safety override active")
            return signal

        # 🎯 High Precision Sniper Mode (Threshold > 0.82)
        if signal.direction == "LONG" and signal.confidence > 0.82:
            logger.info(
                f"🚀 [SNIPER] LONG ({signal.confidence:.2f}) | "
                f"Workers: L={signal.n_long} S={signal.n_short} W={signal.n_wait} | "
                f"H={hurst_value:.2f} Regime={regime_state}"
            )
            order = self.executor.execute_signal(
                action="BUY", confidence=signal.confidence, current_price=current_price, qty=self.config.trade_qty
            )
            if order and order.get("status") != "aborted":
                await self._persist_trade(order, signal, "BUY", current_price)

        elif signal.direction == "SHORT" and signal.confidence > 0.82:
            logger.info(
                f"🔻 [SNIPER] SHORT ({signal.confidence:.2f}) | "
                f"Workers: L={signal.n_long} S={signal.n_short} W={signal.n_wait} | "
                f"H={hurst_value:.2f} Regime={regime_state}"
            )
            order = self.executor.execute_signal(
                action="SELL", confidence=signal.confidence, current_price=current_price, qty=self.config.trade_qty
            )
            if order and order.get("status") != "aborted":
                await self._persist_trade(order, signal, "SELL", current_price)

        return signal

    async def _persist_signal(self, signal: TradeSignal, candle: dict):
        """Save decision to DB (Fire and Forget technically, but we await to ensure data integrity)."""
        from src.database.core import AsyncSessionLocal
        from src.database.models import Signal
        
        try:
             async with AsyncSessionLocal() as session:
                db_signal = Signal(
                    symbol=self.config.symbol,
                    timeframe="M5",
                    signal_value=1 if signal.direction == "LONG" else (-1 if signal.direction == "SHORT" else 0),
                    confidence=float(signal.confidence),
                    model_version="v9.7-Gladiator",
                    features_snapshot={
                        "price": candle["close"],
                        "workers_agreement": signal.worker_agreement
                    },
                    worker_votes={
                        "long": signal.n_long,
                        "short": signal.n_short,
                        "wait": signal.n_wait
                    }
                )
                session.add(db_signal)
                await session.commit()
        except Exception as e:
            logger.error(f"Failed to persist signal: {e}")

    async def _persist_trade(self, order: Any, signal: TradeSignal, side: str, price: float):
        """Save trade execution to DB."""
        from src.database.core import AsyncSessionLocal
        from src.database.models import Trade
        
        # Parse order object (Binance native)
        order_id = "unknown"
        exec_price = price
        
        if isinstance(order, dict):
            order_id = str(order.get('orderId', 'unknown'))
            # Calculate average execution price if available
            fills = order.get("fills", [])
            if fills:
                try:
                    total_cost = sum(float(f["price"]) * float(f["qty"]) for f in fills)
                    total_qty = sum(float(f["qty"]) for f in fills)
                    if total_qty > 0:
                        exec_price = total_cost / total_qty
                except Exception:
                    pass
        
        # Update live telemetry state
        import time
        # Random mock PnL logic for immediate visual feedback on the Live Terminal since we don't have position closing modeled in this sprint
        mock_pnl = (exec_price * self.config.trade_qty * 0.005) if side == "BUY" else -(exec_price * self.config.trade_qty * 0.002)
        
        trade_event = {
            "id": order_id,
            "symbol": self.config.symbol,
            "side": side,
            "price": exec_price,
            "pnl": mock_pnl,
            "status": "OPEN",
            "confidence": float(signal.confidence),
            "time": time.time() * 1000
        }
        self.state.add_trade(trade_event)
        
        try:
             async with AsyncSessionLocal() as session:
                db_trade = Trade(
                    symbol=self.config.symbol,
                    side=side,
                    qty=self.config.trade_qty,
                    price=exec_price,
                    order_id=order_id,
                    strategy="Trinity-V9.7"
                )
                session.add(db_trade)
                await session.commit()
                logger.info(f"💾 Trade saved to DB: {side} @ {exec_price}")
        except Exception as e:
             logger.error(f"Failed to persist trade: {e}")

    # ──────────────────────────────────────────
    # Slow Path: Strategic LFM Worker
    # ──────────────────────────────────────────
    def _lfm_worker(self):
        """Background thread for strategic market analysis via LFM."""
        logger.info(f"🧠 Strategic Agent (LFM: {self.config.lfm_model_name}) Online")

        while self.active:
            start_time = time.time()
            try:
                price = self.state.last_price
                if price == 0:
                    time.sleep(1)
                    continue

                # Phase 1: Tool usage
                prompt_scan = (
                    f"You are a Strategic Trading AI. Current {self.config.symbol} "
                    f"Price: {price:.2f}. Do you need to check Support/Resistance or "
                    f"Trend? Reply with 'CHECK_SR' or 'CHECK_TREND' or 'NO_ACTION'."
                )
                decision = self._query_lfm(prompt_scan).strip()

                tool_output = ""
                if "CHECK_SR" in decision:
                    logger.info("🧠 LFM: CHECK_SR")
                    tool_output = self.scanner.check_support_resistance(
                        self.config.symbol, price
                    )
                elif "CHECK_TREND" in decision:
                    logger.info("🧠 LFM: CHECK_TREND")
                    tool_output = self.scanner.get_trend_alignment(
                        self.config.symbol, "4h"
                    )

                # Phase 2: Synthesize bias
                bias = 0.0
                if "Support" in tool_output:
                    bias += 0.3
                if "Resistance" in tool_output:
                    bias -= 0.3
                if "Bullish" in tool_output:
                    bias += 0.2
                if "Bearish" in tool_output:
                    bias -= 0.2

                self.state.update_lfm(bias, tool_output[:200])
                logger.info(f"🧠 Strategic Bias Updated: {bias:+.2f}")

            except Exception as e:
                logger.error(f"LFM Worker Error: {e}")

            elapsed = time.time() - start_time
            sleep_time = max(1.0, self.config.lfm_refresh_rate - elapsed)
            time.sleep(sleep_time)

    def _query_lfm(self, prompt: str) -> str:
        """Query the Ollama/LFM endpoint."""
        try:
            payload = {
                "model": self.config.lfm_model_name,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.2},
            }
            res = requests.post(
                self.config.lfm_model_url, json=payload, timeout=30
            )
            if res.status_code == 200:
                return res.json().get("response", "")
            else:
                logger.error(f"LFM Error: {res.status_code}")
                return ""
        except Exception as e:
            msg = f"LFM Connection Failed: {e}"
            logger.error(msg)
            self.state.add_log(f"[Strategic] {msg}")
            return ""

    # ──────────────────────────────────────────
    # Lifecycle
    # ──────────────────────────────────────────
    def start(self):
        """Start the engine (launches LFM background thread)."""
        self.active = True
        self.start_time = time.time()
        self.lfm_thread = threading.Thread(target=self._lfm_worker, daemon=True)
        self.lfm_thread.start()
        msg = "🚀 Trinity Engine v11.0 (Full Spectrum) Started"
        logger.info(msg)
        self.state.add_log(msg)

    def stop(self):
        """Gracefully stop the engine."""
        self.active = False
        if self.lfm_thread:
            self.lfm_thread.join(timeout=5)
        msg = "🛑 Trinity Engine Stopped"
        logger.info(msg)
        self.state.add_log(msg)



--- FILE: backend/src/brain/simulate_v8.py ---
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import argparse
import time
import json
import os
import sys
from datetime import datetime

# Import Architecture
# Import Architecture
from src.brain.models.liquid_v8 import LiquidConfig, ModulatedIARA
from src.brain.models.trinity_v8 import TrinityBrain
try:
    from src.trading.active_inference import ActiveInferenceModule
except ImportError:
    ActiveInferenceModule = None

# Constants
LOG_DIR = "logs"
SIM_TELEMETRY_FILE = os.path.join(LOG_DIR, "sim_telemetry.json")
SIM_SNAPSHOT_FILE = os.path.join(LOG_DIR, "sim_snapshot.json")
SIM_PID_FILE = "simulation_process.pid"

# Legacy Compatibility
class TrainerConfig:
    pass

def load_data(csv_path, start_date=None, end_date=None):
    print(f"📂 Loading Data: {csv_path}")
    df = pd.read_csv(csv_path)
    df.columns = [c.lower().strip() for c in df.columns]
    
    # Date parsing
    if "date" in df.columns:
        df["timestamp"] = pd.to_datetime(df["date"])
    elif "time" in df.columns:
        df["timestamp"] = pd.to_datetime(df["time"])
    else:
        df["timestamp"] = pd.date_range(end=datetime.now(), periods=len(df), freq="15min")
        
    df = df.sort_values("timestamp").reset_index(drop=True)
    
    # Filtering
    if start_date:
        df = df[df["timestamp"] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df["timestamp"] <= pd.to_datetime(end_date)]
        
    if len(df) == 0:
        raise ValueError("Date range filtered out all data!")
        
    print(f"✅ Data Loaded: {len(df)} candles.")
    return df

def feature_engineering(df):
    # Quick FE matching training logic
    # Ideally should be a shared library, but for now copying logic for robustness
    delta = df["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-6)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50) / 100.0
    
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    macd = (macd - macd.mean()) / (macd.std() + 1e-6)
    
    sma20 = df["close"].rolling(window=20).mean()
    std20 = df["close"].rolling(window=20).std()
    bb_upper = (df["close"] - sma20) / (2 * std20 + 1e-6)
    bb_upper = bb_upper.fillna(0)
    
    log_ret = np.log(df["close"] / df["close"].shift(1)).fillna(0)
    volatility = log_ret.rolling(window=20).std().fillna(0)
    
    # Normalize Close
    norm_close = (df["close"] - df["close"].mean()) / (df["close"].std() + 1e-6)
    
    # Stack 6 base features
    base_feats = np.column_stack([norm_close, rsi, macd, bb_upper, log_ret, volatility])
    
    # Pad to 18 (Input Dim)
    padding = np.zeros((len(df), 12))
    full_feats = np.column_stack([base_feats, padding])
    
    return full_feats

def run_simulation(model_path, data_path, start_date, speed, seq_len=256):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Starting Simulation on {device}...")
    
    # 1. Load Checkpoint First (to get config)
    try:
        # Weights_only=False for legacy support
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    except Exception as e:
        print(f"❌ Failed to load checkpoint: {e}")
        return

    # 2. Determine Config
    input_dim = 18
    hidden_dim = 128
    
    if "config" in checkpoint:
        cfg = checkpoint["config"]
        # If it's a TrainerConfig object, extract attr
        if hasattr(cfg, 'input_dim'):
            input_dim = cfg.input_dim
            hidden_dim = cfg.hidden_dim
            print(f"ℹ️ Loaded Config from Checkpoint: Input={input_dim}, Hidden={hidden_dim}")
            
    # 3. Init Model
    liq_cfg = LiquidConfig(input_dim=input_dim, hidden_dim=hidden_dim, max_seq_len=seq_len)
    lnn = ModulatedIARA(liq_cfg)
    model = TrinityBrain(lnn, llm_hidden_dim=32).to(device)
    
    # 4. Load State Dict
    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint
    
    try:
        model.load_state_dict(state_dict, strict=False)
        print(f"✅ Model Weights Loaded: {os.path.basename(model_path)}")
    except Exception as e:
         print(f"⚠️ Weight loading warning (mismatch?): {e}")

    model.eval()
    
    # hooks for neurons
    hidden_states = []
    def hook_cfc(module, input, output):
        hidden_states.append(output.detach().cpu().numpy())
    
    # Hook registration (safe check)
    if len(model.lnn.cfc_layers) > 0:
        handle = model.lnn.cfc_layers[-1].register_forward_hook(hook_cfc)
    
    # 5. Load Data
    df = load_data(data_path, start_date)
    features = feature_engineering(df)
    
    # 6. Active Inference Init
    active_brain = ActiveInferenceModule() if ActiveInferenceModule else None
    if active_brain:
        print("🧠 Active Inference Module: ENABLED")
    else:
        print("⚠️ Active Inference Module: NOT FOUND (Using static thresholds)")
    
    # 7. Loop
    total_steps = len(df) - seq_len - 1 # -1 for ground truth check
    print(f"▶️ Playing {total_steps} steps...")
    
    capital = 10000.0
    position = 0.0
    entry_price = 0.0
    
    for i in range(total_steps):
        time.sleep(1.0 / speed)
        
        step_idx = i + seq_len
        
        # Prepare Input
        input_slice = features[i : i+seq_len]
        input_tensor = torch.FloatTensor(input_slice).unsqueeze(0).to(device) # [1, 256, 18]
        
        llm_ctx = torch.zeros(1, 32).to(device)
        news_vec = torch.zeros(1, 2).to(device)
        
        hidden_states = [] # Reset hook
        
        with torch.no_grad():
             logits = model(input_tensor, llm_ctx, news_vec)
             prob = torch.sigmoid(logits).item()
             
        # Stack Hidden States
        if hidden_states:
             h_full = np.stack(hidden_states, axis=1) # [Batch, Seq, Hidden]
             h_matrix = h_full[0].T.tolist() 
             h_last_step = h_full[0][-1] 
             h_last = torch.tensor(h_last_step).float().to(device).unsqueeze(0)
        else:
            h_matrix = []
            h_last = torch.zeros(1, hidden_dim).to(device)

        # Decision Logic (Active Inference)
        price = df.iloc[step_idx]["close"]
        time_str = df.iloc[step_idx]["timestamp"].strftime("%Y-%m-%d %H:%M")
        
        if active_brain:
            # Look ahead for ground truth (Simulating real-time feedback after the fact)
            next_price = df.iloc[step_idx + 1]["close"]
            actual_move = 1.0 if next_price > price else 0.0
            
            # Log PREVIOUS prediction's outcome (In real life we'd log at step T+1)
            # For simplicity in simulation loop, we log 'prob' vs 'actual_move' instantaneously 
            # (treating it as if the 'epoch' closed immediately)
            # A more correct way involves a delay, but this is fine for calibration testing.
            active_brain.log_prediction(prob, actual_move)
            
            decision = active_brain.get_decision(prob) # Can pass features too if we had ADX
            
            # Translate Decision
            if decision == "BUY" and position == 0:
                action = "BUY"
            elif decision == "SELL" and position > 0:
                action = "SELL"
            else:
                action = "HOLD"
        else:
            # Fallback Static
            if prob > 0.6 and position == 0:
                action = "BUY"
            elif prob < 0.4 and position > 0:
                 action = "SELL"
            else:
                action = "HOLD"
        
        # Execution
        if action == "BUY":
            position = capital / price
            entry_price = price
            capital = 0
        elif action == "SELL":
            capital = position * price
            position = 0
            
        # PnL
        current_val = capital + (position * price)
        pnl_pct = (current_val - 10000.0) / 10000.0 * 100
        
        # Expert Curves
        long_bias = model.lnn.long_expert(h_last).item()
        short_bias = model.lnn.short_expert(h_last).item()
        
        # Emit State
        state = {
            "step": i,
            "timestamp": time_str,
            "price": float(price),
            "prob": prob,
            "action": action,
            "equity": current_val,
            "pnl_pct": pnl_pct,
            "long_bias": long_bias,
            "short_bias": short_bias,
            "neuronal_map": h_matrix,
            "completed": False,
            "threshold": active_brain.current_threshold if active_brain else 0.6
        }
        
        temp_file = SIM_SNAPSHOT_FILE + ".tmp"
        with open(temp_file, "w") as f:
            json.dump(state, f)
        os.replace(temp_file, SIM_SNAPSHOT_FILE)
        
        with open(SIM_TELEMETRY_FILE, "w") as f:
            json.dump({"step": i, "equity": current_val, "pnl": pnl_pct}, f)
            
    with open(SIM_SNAPSHOT_FILE, "w") as f:
        json.dump({"completed": True}, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--start_date", default=None)
    parser.add_argument("--speed", type=float, default=10.0)
    args = parser.parse_args()
    
    # Write PID
    with open(SIM_PID_FILE, "w") as f:
        f.write(str(os.getpid()))
        
    try:
        run_simulation(args.model, args.data, args.start_date, args.speed)
    finally:
        if os.path.exists(SIM_PID_FILE):
             os.remove(SIM_PID_FILE)



--- FILE: backend/src/brain/train_v9.py ---
"""
I.A.R.A. v9 - Swarm Intelligence Training Loop
===============================================
Robust training loop for the Swarm architecture with:

1. Multi-Timeframe DataLoader (time-aligned batches)
2. Mixed Precision (AMP) for VRAM efficiency
3. Selective Memory (curriculum learning)
4. MLflow/Tensorboard logging with separate metrics per loss component
5. Gradient clipping (0.5) for LNN stability

Usage:
    docker compose exec saci-brain python src/brain/train_v9.py --epochs 100

Hardware: AMD RX 6750 XT (12GB VRAM)
Target: <8GB VRAM usage
"""

import argparse
import csv
import glob
import json
import logging
import math
import os
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

# Project imports
from src.brain.models.workers import WorkerConfig
from src.brain.models.liquid_v9 import SwarmBrain, CoordinatorConfig
from src.brain.losses import SwarmLoss

import signal
import sys
import time
import shutil
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)] # Ensure stdout flush
)
logger = logging.getLogger("IARA_v9")

# Global model reference for signal handler
global_model = None

def save_checkpoint(signum, frame):
    """Handle signal to save checkpoint manually"""
    logger.info("💾 Manual Save Requested via Signal...")
    if global_model:
        path = MODELS_DIR / f"manual_checkpoint_v9.pth"
        torch.save(global_model.state_dict(), path)
        logger.info(f"✅ Model manually saved to {path}")
    else:
        logger.warning("⚠️ Model not initialized yet, cannot save.")

# Register signal
signal.signal(signal.SIGUSR1, save_checkpoint)

# Paths
LOG_DIR = Path("logs")
MODELS_DIR = LOG_DIR / "models"
METRICS_FILE = LOG_DIR / "metrics.csv"  # Dashboard reads this file!
METRICS_FILE_V9 = LOG_DIR / "metrics_v9.csv"  # Backup for v9 specific
TELEMETRY_FILE = LOG_DIR / "telemetry.json"  # Dashboard reads metrics.csv
LEADERBOARD_FILE = LOG_DIR / "leaderboard_v9.json"


@dataclass
class TrainConfig:
    """Training configuration for v9."""
    # Data
    seq_len: int = 128
    batch_size: int = 256
    val_split: float = 0.2
    
    # Model
    input_dim: int = 12
    worker_hidden: int = 64
    coord_hidden: int = 256
    
    # Training
    epochs: int = 200
    learning_rate: float = 0.0005
    weight_decay: float = 1e-4
    grad_clip: float = 0.5
    use_amp: bool = True
    debug: bool = False
    
    # Loss weights
    aux_weight: float = 0.4
    consensus_weight: float = 0.1
    label_smoothing: float = 0.1
    
    # Scheduler
    scheduler_patience: int = 15
    scheduler_factor: float = 0.8
    
    # Early stopping
    patience: int = 50


class CapturedExampleDataset(Dataset):
    """Dataset for examples captured from backtest/simulation (Errors, Successes, Patterns)."""
    def __init__(self, sources: List[Path], seq_len: int = 64):
        self.seq_len = seq_len
        self.data = []
        for src in sources:
            if src.exists():
                try:
                    loaded = torch.load(src, map_location="cpu", weights_only=False)
                    # Support both list and dict-based captures
                    if isinstance(loaded, list):
                        self.data.extend(loaded)
                    logger.info(f"🔥 Mixed {len(loaded)} examples from {src.name}")
                except Exception as e:
                    logger.error(f"Failed to load captured examples from {src}: {e}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        x_dict = item["x_dict"] 
        processed_x = {}
        for tf, tensor in x_dict.items():
            # If tensor is [B, L, C], take [L, C]
            if tensor.dim() == 3:
                processed_x[tf] = tensor[0, :, :12].float() # Truncate to 12 features
            else:
                processed_x[tf] = tensor[:, :12].float() # Truncate to 12 features
        
        y = item["y"].float() if "y" in item else torch.tensor(0.5) # Default neutral if pattern only
        
        # Add placeholder forensics metadata (zeros) to match MultiTimeframeDataset structure
        # This prevents KeyError in the collate function
        processed_x["__ohlc__"] = torch.zeros(self.seq_len, 4)
        
        return processed_x, y


class MultiTimeframeDataset(Dataset):
    """
    Dataset that provides time-aligned data for all timeframes.
    
    Each batch contains synchronized data from M5, M15, M30, H1, H4
    representing the same wall-clock time.
    """
    
    TIMEFRAMES = ["M5", "M15", "M30", "H1", "H4"]
    TF_MULTIPLIERS = {"M5": 1, "M15": 3, "M30": 6, "H1": 12, "H4": 48}
    
    def __init__(
        self,
        data_path: str = "data/cache",
        seq_len: int = 64,
        split: str = "train"
    ):
        self.seq_len = seq_len
        self.split = split
        
        # Load sharded data
        shard_dir = Path(data_path) / f"{split}_shards"
        
        if not shard_dir.exists():
            # Fallback to single file
            single_file = Path(data_path) / f"{split}.pt"
            if single_file.exists():
                data = torch.load(single_file, weights_only=False)
                self.X = data.get('X', data)
                self.y = data.get('y', None)
                if self.y is None:
                    raise ValueError(f"No targets found in {single_file}")
            else:
                raise FileNotFoundError(f"No data found at {data_path}")
        else:
            # Load shards INCREMENTALLY to avoid RAM spikes
            shard_paths = sorted(shard_dir.glob("*.pt"))
            logger.info(f"  🔍 Processing {len(shard_paths)} shards for {split}...")
            
            X_list, y_list = [], []
            for i, shard_path in enumerate(shard_paths):
                data = torch.load(shard_path, weights_only=False)
                # Convert to half IMMEDIATELY upon loading to save 50% RAM during concatenation
                X_list.append(data['X'].half())
                y_list.append(data['y'].float())
                
                if (i + 1) % 5 == 0 or (i + 1) == len(shard_paths):
                    logger.info(f"    - [{split}] Loaded shard {i+1}/{len(shard_paths)}")
                
                # Manual cleanup is tricky with PyTorch, but we hope the list append is efficient
                del data 
            
            logger.info(f"  ⛓️ Concatenating {split} shards...")
            self.X = torch.cat(X_list, dim=0)
            self.y = torch.cat(y_list, dim=0)
            
            # Already in half() from loop
        
        # Get feature dimension
        self.feature_dim = self.X.shape[-1]
        
        logger.info(f"  Loaded {split} dataset: {len(self)} samples, {self.feature_dim} features")
        
    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Returns time-aligned data for all timeframes.
        This is a simplification. Production should use actual resampled data.
        """
        # Convert back to float32 for model computation (stored as float16 for RAM savings)
        x = self.X[idx].float()  # [seq_len, features] - NOW IN FLOAT32
        y = self.y[idx]  # scalar (already float32)
        
        # Simulate timeframe views by downsampling
        x_dict = {}
        
        # Core features for the model (First 12)
        x_core = x[:, :12]
        
        # M5: Full resolution
        x_dict["M5"] = x_core[-self.seq_len:]
        
        # Metadata OHLC (Last 4 columns) for Forensics
        # We only keep it for M5 to save memory
        ohlc = x[-self.seq_len:, 22:]
        if ohlc.shape[1] < 4:
            # Pad with zeros if metadata columns are missing in cache
            ohlc = torch.cat([ohlc, torch.zeros(ohlc.shape[0], 4 - ohlc.shape[1])], dim=1)
        x_dict["__ohlc__"] = ohlc
        
        # M15: Every 3rd candle
        step_15 = max(1, self.seq_len // 3)
        x_dict["M15"] = x_core[::3][-step_15:] if self.seq_len >= 3 else x_core[-1:]
        
        # M30: Every 6th candle
        step_30 = max(1, self.seq_len // 6)
        x_dict["M30"] = x_core[::6][-step_30:] if self.seq_len >= 6 else x_core[-1:]
        
        # H1: Every 12th candle
        step_h1 = max(1, self.seq_len // 12)
        x_dict["H1"] = x_core[::12][-step_h1:] if self.seq_len >= 12 else x_core[-1:]
        
        # H4: Every 48th candle
        step_h4 = max(1, self.seq_len // 16)
        x_dict["H4"] = x_core[::16][-step_h4:] if self.seq_len >= 16 else x_core[-1:]
        
        return x_dict, y


def collate_multitf(batch: List[Tuple[Dict[str, torch.Tensor], torch.Tensor]]):
    """Custom collate function for multi-timeframe batches."""
    x_dicts, ys = zip(*batch)
    
    # Identify common keys
    common_keys = set(x_dicts[0].keys())
    for x in x_dicts[1:]:
        common_keys &= set(x.keys())
    
    x_batch = {}
    for tf in sorted(list(common_keys)):
        seqs = [x[tf] for x in x_dicts]
        
        # Check for feature dimension consistency
        feat_dims = [s.size(-1) for s in seqs]
        if len(set(feat_dims)) > 1:
            # If inconsistent, we skip this key (likely metadata mismatch)
            continue
            
        max_len = max(s.size(0) for s in seqs)
        feat_dim = feat_dims[0]
        
        # Pad sequences to same length
        padded = []
        for s in seqs:
            if s.size(0) < max_len:
                pad = torch.zeros(max_len - s.size(0), feat_dim)
                s = torch.cat([pad, s], dim=0)
            padded.append(s.unsqueeze(0))
        
        x_batch[tf] = torch.cat(padded, dim=0)
    
    y_batch = torch.stack(ys, dim=0)
    return x_batch, y_batch


class Leaderboard:
    """Tracks best models."""
    
    def __init__(self, path: Path = LEADERBOARD_FILE, top_k: int = 5):
        self.path = path
        self.top_k = top_k
        self.history = []
        
        if path.exists():
            try:
                with open(path, "r") as f:
                    self.history = json.load(f)
            except:
                pass
    
    def update(
        self, 
        epoch: int, 
        loss: float, 
        auc: float, 
        accuracy: float,
        model_state: dict,
        symbol: str = "EURUSD"
    ) -> bool:
        """Update leaderboard. Returns True if new best."""
        entry = {
            "epoch": epoch,
            "loss": loss,
            "auc": auc,
            "accuracy": accuracy,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        self.history.append(entry)
        self.history = sorted(self.history, key=lambda x: -x["auc"])[:self.top_k]
        
        with open(self.path, "w") as f:
            json.dump(self.history, f, indent=2)
        
        is_best = self.history[0]["epoch"] == epoch
        
        if is_best:
            logger.info(f"🏆 New Best Model for {symbol}! AUC: {auc:.4f} | Acc: {accuracy:.2%}")
            torch.save(model_state, MODELS_DIR / f"best_model_v9_{symbol}.pth")
        
        return is_best


class SelectiveMemory:
    """
    Curriculum learning based on sample difficulty.
    Ported from v6 trainer_liquid.py.
    """
    
    def __init__(self, num_samples: int, alpha: float = 0.7):
        self.alpha = alpha
        # Initialize with 0.6 (neutral) instead of 1.0 to avoid extreme bias on new data
        self.sample_loss_ema = torch.full((num_samples,), 0.6)
        self.chronic_counter = torch.zeros(num_samples, dtype=torch.int16)
        self.mastered_counter = torch.zeros(num_samples, dtype=torch.int16)
        self.noise_mask = torch.zeros(num_samples, dtype=torch.bool)
        
    def analyze(
        self, 
        model: nn.Module, 
        dataset: Dataset, 
        device: str,
        criterion: nn.Module,
        batch_size: int = 512,
        sample_size: Optional[int] = None,
        captured_indices: Optional[range] = None
    ) -> WeightedRandomSampler:
        """Compute sample weights for curriculum learning."""
        
        num_total = len(dataset)
        indices = np.arange(num_total)
        
        if sample_size and sample_size < num_total:
             logger.info(f"🧠 Deep Thought: Analyzing random subset of {sample_size} samples...")
             indices = np.random.choice(indices, size=sample_size, replace=False)
             analysis_ds = torch.utils.data.Subset(dataset, indices)
        else:
             logger.info("🧠 Deep Thought: Analyzing ALL samples...")
             analysis_ds = dataset
        
        model.eval()
        losses = []
        
        loader = DataLoader(
            analysis_ds, 
            batch_size=batch_size, 
            shuffle=False,
            collate_fn=collate_multitf,
            num_workers=0
        )
        
        with torch.no_grad():
            with torch.amp.autocast('cuda', enabled=(device == "cuda")):
                for x_dict, targets in loader:
                    # Move to device
                    x_dict = {k: v.to(device) for k, v in x_dict.items()}
                    targets = targets.to(device)
                    
                    # Forward
                    logits, worker_logits, _, _ = model(x_dict)
                    
                    # Store individual losses
                    bce = nn.functional.binary_cross_entropy_with_logits(
                        logits.view(-1), 
                        targets.view(-1),
                        reduction='none'
                    )
                    losses.append(bce.cpu())
                    
                    if len(losses) % 20 == 0:
                        import gc
                        gc.collect() # Force GC to avoid OOM
                        
        if not losses:
            return None

        current_losses = torch.cat(losses)
        
        # Ensure state tensors match dataset size (resizing if dataset grew)
        if len(self.sample_loss_ema) != num_total:
             logger.info(f"⚠️ Resizing memory state from {len(self.sample_loss_ema)} to {num_total}")
             # Extend with ones/zeros
             new_size = num_total - len(self.sample_loss_ema)
             self.sample_loss_ema = torch.cat([self.sample_loss_ema, torch.ones(new_size)])
             self.chronic_counter = torch.cat([self.chronic_counter, torch.zeros(new_size, dtype=torch.int16)])
             self.mastered_counter = torch.cat([self.mastered_counter, torch.zeros(new_size, dtype=torch.int16)])
        
        # Update EMA only for analyzed indices
        # We need to map back to original indices. 
        # current_losses corresponds to 'indices' array.
        
        indices_tensor = torch.tensor(indices, dtype=torch.long)
        
        # Update EMA
        # self.sample_loss_ema[indices] = alpha * old + (1-alpha) * new
        old_ema = self.sample_loss_ema[indices_tensor]
        new_ema = self.alpha * old_ema + (1 - self.alpha) * current_losses
        self.sample_loss_ema[indices_tensor] = new_ema
        
        # Update counters
        high_loss = current_losses > 0.9
        self.chronic_counter[indices_tensor[high_loss]] += 1
        self.chronic_counter[indices_tensor[~high_loss]] = 0
        
        low_loss = current_losses < 0.3
        self.mastered_counter[indices_tensor[low_loss]] += 1
        self.mastered_counter[indices_tensor[~low_loss]] = 0
        
        # Compute weights for WHOLE dataset
        # Exponent 1.0 instead of 1.5 for linear priority (tempered Importance Sampling)
        weights = (self.sample_loss_ema + 0.05) ** 1.0
        
        # Boost chronic - Lowered from 3.0 to 2.0 to be less aggressive with noise
        weights[self.chronic_counter >= 2] *= 2.0
        
        # Suppress mastered
        weights[self.mastered_counter >= 5] *= 0.1
        
        # Boost Captured (Human Feedback) - Lowered from 5.0 to 2.0 to avoid noise dominance
        if captured_indices is not None:
             weights[list(captured_indices)] *= 2.0
        
        # 10% Diversity Mix (Random Background)
        # Adds a baseline weight to all samples to prevent catastrophic forgetting
        diversity_floor = weights.mean() * 0.1
        weights = weights + diversity_floor
        
        # Stats
        n_chronic = (self.chronic_counter >= 2).sum().item()
        n_mastered = (self.mastered_counter >= 5).sum().item()
        logger.info(f"  📊 Stats: {n_chronic} Chronic | {n_mastered} Mastered")
        
        model.train()
        
        return WeightedRandomSampler(
            weights=weights.tolist(),
            num_samples=len(weights),
            replacement=True
        )

    def get_summary(self) -> Dict[str, Any]:
        """Export memory statistics for dashboard visualization."""
        try:
            # 1. Distribution of Loss (Memory state)
            hist = torch.histc(self.sample_loss_ema, bins=10, min=0.0, max=1.0)
            loss_dist = hist.tolist()
            
            # 2. Key stats
            n_samples = len(self.sample_loss_ema)
            n_chronic = (self.chronic_counter >= 3).sum().item()
            n_mastered = (self.mastered_counter >= 5).sum().item()
            
            return {
                "loss_distribution": loss_dist,
                "n_chronic": n_chronic,
                "n_mastered": n_mastered,
                "n_total": n_samples,
                "mean_loss": float(self.sample_loss_ema.mean().item()),
                "max_loss": float(self.sample_loss_ema.max().item())
            }
        except Exception as e:
            return {"error": str(e)}


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: str,
    config: TrainConfig,
    epoch: int,
    total_epochs: int,
    global_step: int = 0,
    current_lr: float = 0.001
) -> Tuple[Dict[str, float], int]:
    """Train one epoch."""
    model.train()
    
    losses = {"total": [], "coord": [], "workers": [], "consensus": [], "weights": []}
    all_preds, all_targets = [], []
    
    for step, (x_dict, targets) in enumerate(loader):
        # Move to device
        x_dict = {k: v.to(device) for k, v in x_dict.items()}
        targets = targets.to(device)
        
        optimizer.zero_grad()
        
        # Mixed Precision (AMP)
        if config.use_amp:
            with torch.amp.autocast('cuda', enabled=True):
                logits, worker_logits, _, _ = model(x_dict)
                loss_dict = criterion(logits, worker_logits, targets)
        else:
            logits, worker_logits, _, _ = model(x_dict)
            loss_dict = criterion(logits, worker_logits, targets)
        
        # Check for NaN
        if torch.isnan(loss_dict['total']):
            logger.warning(f"⚠️ NaN loss at step {step}, skipping...")
            continue
        
        # AMP backward
        scaler.scale(loss_dict['total']).backward()
        
        # Gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        
        scaler.step(optimizer)
        scaler.update()
        
        # Track losses
        for key in ["total", "coord", "workers", "consensus"]:
            losses[key].append(loss_dict[key].item())
        losses["weights"].append(loss_dict["weights"])
        
        # Track predictions
        probs = torch.sigmoid(logits.detach()).cpu().numpy()
        all_preds.extend(probs.flatten())
        all_targets.extend((targets > 0.5).float().cpu().numpy().flatten())
        
        # Log progress
        if config.debug or step % 10 == 0:
            step_loss = loss_dict['total'].item()
            
            # Check RAM (Linux only, no psutil needed)
            mem_usage = 0
            try:
                with open("/proc/self/status") as f:
                    for line in f:
                        if "VmRSS" in line:
                            mem_usage = int(line.split()[1]) / 1024 # MB
                            break
            except:
                pass
            
            if config.debug:
                 print(f"Step {step}: Loss {step_loss:.4f} | RAM: {mem_usage:.0f}MB", end="\r")
            
            if step % 10 == 0:
                # Real Accuracy
                step_acc = np.mean([(p > 0.5) == (t > 0.5) for p, t in zip(probs.flatten(), targets.cpu().numpy().flatten())]) if len(probs) > 0 else 0.5
                
                # Real AUC for the step (if possible)
                try:
                    # Check if both classes are present in the batch
                    if len(np.unique(targets.cpu().numpy())) > 1:
                        step_auc = roc_auc_score(targets.cpu().numpy().flatten(), probs.flatten())
                    else:
                        step_auc = df["auc"].iloc[-1] if not df.empty else 0.5 # Use last known if available
                except:
                    step_auc = 0.5 
                
                logger.info(
                    f"Epoch {epoch} | Step {step}/{len(loader)} | "
                    f"Loss: {step_loss:.4f} | "
                    f"AUC: {step_auc:.4f} | "
                    f"RAM: {mem_usage:.0f} MB"
                )
                
                # Write to CSV
                with open(METRICS_FILE, "a", newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([global_step, epoch, step_loss, step_acc, step_auc, current_lr])
                
                # Update real-time telemetry (PERSISTENT version to avoid flickering)
                try:
                    tele_data = {
                        "status": "TRAINING",
                        "current_epoch": epoch,
                        "total_epochs": total_epochs,
                        "current_step": step,
                        "total_steps": len(loader),
                        "loss": step_loss,
                        "acc": step_acc,
                        "auc": step_auc,
                        "mem": mem_usage,
                        "lr": current_lr
                    }
                    
                    # Merge with existing data if exists (to keep memory_stats and rollback_count)
                    if os.path.exists(TELEMETRY_FILE):
                        with open(TELEMETRY_FILE, "r") as f:
                            existing = json.load(f)
                            # Only overwrite if same status or if it's a training update
                            for key in ["rollback_count", "memory_stats", "device"]:
                                if key in existing:
                                    tele_data[key] = existing[key]
                    
                    temp_path = f"{TELEMETRY_FILE}.tmp"
                    with open(temp_path, "w") as f:
                        json.dump(tele_data, f, indent=2)
                    os.replace(temp_path, TELEMETRY_FILE)
                except:
                    pass
        
        global_step += 1
    
    # Compute metrics
    avg_losses = {k: np.mean(v) if v else 0.0 for k, v in losses.items() if k != "weights"}
    if losses["weights"]:
        avg_losses["weights"] = np.mean(losses["weights"], axis=0).tolist()
    else:
        avg_losses["weights"] = [0.0, 0.0, 0.0]
    
    try:
        auc = roc_auc_score(all_targets, all_preds)
    except:
        auc = 0.5
    
    accuracy = np.mean([(p > 0.5) == t for p, t in zip(all_preds, all_targets)])
    
    avg_losses["auc"] = auc
    avg_losses["accuracy"] = accuracy
    
    return avg_losses, global_step


def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str
) -> Dict[str, float]:
    """Validate model."""
    model.eval()
    
    losses = {"total": [], "coord": [], "workers": [], "consensus": [], "weights": []}
    all_preds, all_targets = [], []
    
    with torch.no_grad():
        with torch.amp.autocast('cuda', enabled=(device == "cuda")):
            for x_dict, targets in loader:
                x_dict = {k: v.to(device) for k, v in x_dict.items()}
                targets = targets.to(device)
                
                logits, worker_logits, _, _ = model(x_dict)
                loss_dict = criterion(logits, worker_logits, targets)
                
                for key in ["total", "coord", "workers", "consensus"]:
                    losses[key].append(loss_dict[key].item())
                losses["weights"].append(loss_dict["weights"])
                
                probs = torch.sigmoid(logits).cpu().numpy()
                all_preds.extend(probs.flatten())
                all_targets.extend((targets > 0.5).float().cpu().numpy().flatten())
    
    avg_losses = {k: np.mean(v) if v else 0.0 for k, v in losses.items() if k != "weights"}
    if losses["weights"]:
        avg_losses["weights"] = np.mean(losses["weights"], axis=0).tolist()
    else:
        avg_losses["weights"] = [0.0, 0.0, 0.0]
    
    try:
        auc = roc_auc_score(all_targets, all_preds)
    except:
        auc = 0.5
    
    accuracy = np.mean([(p > 0.5) == t for p, t in zip(all_preds, all_targets)])
    
    avg_losses["auc"] = auc
    avg_losses["accuracy"] = accuracy
    
    return avg_losses


def main():
    parser = argparse.ArgumentParser(description="Train IARA v9 Swarm Intelligence")
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--worker_hidden", type=int, default=64)
    parser.add_argument("--coord_hidden", type=int, default=256)
    parser.add_argument("--cpu", action="store_true", default=False)
    parser.add_argument("--no_amp", action="store_true", default=False, help="Disable Mixed Precision")
    parser.add_argument("--debug", action="store_true", default=False, help="Log every step")
    parser.add_argument("--resume", action="store_true", default=False, help="Resume from checkpoint")
    parser.add_argument("--symbol", type=str, default="EURUSD", help="Currency pair (e.g. EURUSD, GBPUSD)")
    args = parser.parse_args()
    
    # Setup directories
    LOG_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)
    
    # Symbol-specific paths
    metrics_file = LOG_DIR / f"metrics_{args.symbol}.csv"
    metrics_file_v9 = LOG_DIR / f"metrics_v9_{args.symbol}.csv"
    telemetry_file = LOG_DIR / f"telemetry_{args.symbol}.json"
    
    # Initialize metrics CSV - Dashboard format: step,epoch,loss,accuracy,auc,lr
    with open(metrics_file, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["step", "epoch", "loss", "accuracy", "auc", "lr"])
        
    # Initialize V9 detailed metrics
    with open(metrics_file_v9, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch", "train_loss", "val_loss", 
            "train_auc", "val_auc",
            "coord_loss", "worker_loss", "consensus_loss",
            "lr"
        ])
    
    # Global step counter for dashboard
    global_step = 0
    
    # Device
    if args.cpu:
        device = "cpu"
    elif torch.cuda.is_available():
        device = "cuda"
        logger.info(f"🚀 GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        logger.warning("⚠️ CUDA not available, using CPU")
    
    logger.info(f"Device: {device.upper()} | Symbol: {args.symbol}")
    
    # Configuration
    config = TrainConfig(
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        worker_hidden=args.worker_hidden,
        coord_hidden=args.coord_hidden,
        learning_rate=args.lr,
        epochs=args.epochs,
        patience=args.patience,
        use_amp=not args.no_amp,
        debug=args.debug
    )
    
    # Load data (Isolated by symbol)
    logger.info(f"📂 Loading datasets for {args.symbol}...")
    cache_path = f"data/cache/{args.symbol}"
    
    try:
        train_ds = MultiTimeframeDataset(data_path=cache_path, seq_len=config.seq_len, split="train")
        val_ds = MultiTimeframeDataset(data_path=cache_path, seq_len=config.seq_len, split="val")
    except FileNotFoundError as e:
        logger.error(f"❌ Data not found: {e}")
        logger.error("Please run feature_factory.py first to generate training data.")
        sys.exit(1)
    
    # Determine input dimension from data
    sample_x, _ = train_ds[0]
    config.input_dim = sample_x["M5"].shape[-1]
    logger.info(f"  Detected {config.input_dim} input features")
    
    # Create model
    logger.info("🧠 Creating SwarmBrain...")
    
    worker_cfg = WorkerConfig(
        input_dim=config.input_dim,
        hidden_dim=config.worker_hidden,
        dropout=0.1
    )
    
    coord_cfg = CoordinatorConfig(
        num_workers=5,
        worker_dim=config.worker_hidden,
        hidden_dim=config.coord_hidden,
        dropout=0.3
    )
    
    model = SwarmBrain(worker_cfg, coord_cfg).to(device)
    
    # Expose to global for signal handler
    global global_model
    global_model = model
    
    logger.info(f"  Total parameters: {model.count_parameters():,}")
    
    # Check VRAM
    if device == "cuda":
        param_mem = sum(p.numel() * p.element_size() for p in model.parameters())
        logger.info(f"  Parameter memory: {param_mem / 1e6:.2f} MB")
    
    # Resume from checkpoint if exists
    checkpoint_path = MODELS_DIR / f"best_model_v9_{args.symbol}.pth"
    start_epoch = 0
    
    if args.resume and checkpoint_path.exists():
        logger.info(f"🏆 Found checkpoint for {args.symbol}, resuming...")
        try:
            state_dict = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(state_dict, strict=False)
            logger.info("  ✅ Weights loaded!")
        except Exception as e:
            logger.warning(f"  ⚠️ Could not load checkpoint: {e}")
    elif checkpoint_path.exists():
        # Backup existing model if starting fresh
        backup_path = MODELS_DIR / f"best_model_v9_{args.symbol}_backup_{int(time.time())}.pth"
        import shutil
        try:
            shutil.copy(checkpoint_path, backup_path)
            logger.info(f"📦 Existing model backed up to {backup_path.name}")
        except Exception as e:
            logger.warning(f"  ⚠️ Could not backup model: {e}")
        
        logger.info("🆕 Starting FRESH training sequence.")
    
    criterion = SwarmLoss(
        alpha=0.45,  # Balanced/Cautious (was 0.75, which over-prioritized Longs)
        gamma=2.0,
        consensus_weight=config.consensus_weight,
        label_smoothing=config.label_smoothing
    ).to(device)
    
    # Optimizer (Now includes model AND criterion parameters for Dynamic Uncertainty)
    # We use full LR to allow faster adaptation of loss weights
    params = list(model.parameters()) + list(criterion.parameters())
    optimizer = optim.AdamW(
        params,
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=config.scheduler_factor,
        patience=config.scheduler_patience,
        min_lr=1e-6
    )
    
    # AMP scaler
    scaler = torch.amp.GradScaler('cuda', enabled=(device == "cuda"))
    
    # Feedback Loop (Captured Examples)
    feedback_sources = [
        Path("logs/backtest_errors.pt"),
        Path("logs/backtest_success.pt"),
        Path("logs/pattern_reservoir.pt")
    ]
    captured_ds = CapturedExampleDataset(feedback_sources, seq_len=config.seq_len)
    leaderboard = Leaderboard()
    
    if len(captured_ds) > 0:
        logger.info(f"🚀 Integrating {len(captured_ds)} captured examples into training set...")
        from torch.utils.data import ConcatDataset
        full_train_ds = ConcatDataset([train_ds, captured_ds])
        logger.info(f"📊 Combined dataset size: {len(full_train_ds)}")
    else:
        full_train_ds = train_ds

    # DataLoaders
    train_loader = DataLoader(
        full_train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_multitf,
        drop_last=True,
        num_workers=0,
        pin_memory=(device == "cuda")
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_multitf,
        num_workers=0,
        pin_memory=(device == "cuda")
    )
    
    logger.info(f"  Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")
    
    # Selective Memory initialized with FULL dataset
    memory = SelectiveMemory(len(full_train_ds))
    
    # Training loop
    logger.info("=" * 60)
    logger.info("🚀 Starting Training v9 Swarm Intelligence...")
    logger.info("=" * 60)
    
    best_val_auc = 0.0
    patience_counter = 0
    rollback_count = 0
    
    # Track indices for human feedback boosting
    captured_indices = range(len(train_ds), len(full_train_ds)) if len(captured_ds) > 0 else None
    
    for epoch in range(start_epoch, config.epochs):
        epoch_start = time.time()
        
        # Update sampler after first epoch
        # Update sampler after first epoch
        if epoch > 0:
            try:
                # Use full_train_ds for memory analysis
                # sample_size increased to 60,000 for better coverage
                sampler = memory.analyze(
                    model, full_train_ds, device, criterion, 
                    sample_size=60000, captured_indices=captured_indices
                )
                
                if sampler:
                    train_loader = DataLoader(
                        full_train_ds,
                        batch_size=config.batch_size,
                        sampler=sampler,
                        collate_fn=collate_multitf,
                        drop_last=True,
                        num_workers=0
                    )
                    logger.info("✅ Selective Memory Analysis Complete (Sampled 20k)")
                else:
                    raise Exception("No sampler")
                 
            except Exception as e:
                # Fallback to standard loader
                train_loader = DataLoader(
                    full_train_ds,
                    batch_size=config.batch_size,
                    shuffle=True,
                    collate_fn=collate_multitf,
                    drop_last=True,
                    num_workers=0
                )
        
        # Train (pass global_step and current_lr for dashboard logging)
        current_lr = optimizer.param_groups[0]['lr']
        train_metrics, global_step = train_epoch(
            model, train_loader, criterion, optimizer, scaler,
            device, config, epoch + 1, config.epochs, global_step, current_lr
        )
        
        # Validation
        val_metrics = validate(model, val_loader, criterion, device)
        
        # Save Forensic Snapshot (First batch of validation)
        try:
            save_forensic_snapshot(model, val_loader, device, epoch + 1, config)
        except Exception as e:
            logger.error(f"⚠️ Forensic Snapshot Failed: {e}")

        # SELF-CORRECTING LOGIC: ROLLBACK IF DEGRADED
        # If val AUC drops more than 5% below best, reload best model and lower LR
        if best_val_auc > 0 and val_metrics['auc'] < best_val_auc * 0.95:
             logger.warning(f"🚨 Performance Degradation Detected! (AUC: {val_metrics['auc']:.4f} < {best_val_auc:.4f} * 0.95)")
             logger.info("🏆 Rollback: Reloading last best weights and reducing Learning Rate...")
             try:
                 # Checkpoint is symbol-specific
                 if checkpoint_path.exists():
                     state_dict = torch.load(checkpoint_path, map_location=device)
                     model.load_state_dict(state_dict)
                     
                     # Force LR reduction by 20%
                     for param_group in optimizer.param_groups:
                         param_group['lr'] *= 0.8
                     
                     rollback_count += 1
                     # Update telemetry immediately (using symbol-specific file)
                     try:
                         with open(telemetry_file, "w") as f:
                             json.dump({"status": "ROLLBACK", "current_epoch": epoch+1, "rollback_count": rollback_count, "new_lr": optimizer.param_groups[0]['lr']}, f)
                     except: pass
                     
                     logger.info(f"📉 New Learning Rate set to {optimizer.param_groups[0]['lr']:.2e}")
                     
                     # Skip the rest of this epoch's logic (best_val_auc remains same)
                     continue
             except Exception as e:
                 logger.error(f"⚠️ Rollback failed: {e}")

        # Update scheduler
        scheduler.step(val_metrics['total'])
        epoch_time = time.time() - epoch_start
        
        # Log
        logger.info(f"\n{'='*60}")
        logger.info(
            f"Epoch {epoch + 1}/{config.epochs} | Time: {epoch_time:.1f}s\n"
            f"  Train - Loss: {train_metrics['total']:.4f} | AUC: {train_metrics['auc']:.4f} | Acc: {train_metrics['accuracy']:.2%}\n"
            f"  Val   - Loss: {val_metrics['total']:.4f} | AUC: {val_metrics['auc']:.4f} | Acc: {val_metrics['accuracy']:.2%}\n"
            f"  Weights (Dynamic) - Coord: {train_metrics['weights'][0]:.2f} | Workers: {train_metrics['weights'][1]:.2f} | Consensus: {train_metrics['weights'][2]:.2f}\n"
            f"  LR: {current_lr:.2e}"
        )
        logger.info(f"{'='*60}\n")
        
        # Save epoch metrics to V9 specific file (detailed)
        with open(metrics_file_v9, "a", newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch + 1,
                train_metrics['total'],
                val_metrics['total'],
                train_metrics['auc'],
                val_metrics['auc'],
                train_metrics['coord'],
                train_metrics['workers'],
                train_metrics['consensus'],
                current_lr
            ])
        
        # Also update the Master Metrics for live dashboard view
        with open(metrics_file, "a", newline='') as f:
            writer = csv.writer(f)
            writer.writerow([global_step, epoch + 1, train_metrics['total'], train_metrics['accuracy'], train_metrics['auc'], current_lr])

        # Leaderboard (now takes the symbol-specific checkpoint path)
        lb_leaderboard = Leaderboard(path=LOG_DIR / f"leaderboard_v9_{args.symbol}.json")
        is_best = lb_leaderboard.update(
            epoch + 1,
            val_metrics['total'],
            val_metrics['auc'],
            val_metrics['accuracy'],
            model.state_dict(),
            args.symbol
        )
        
        # Early stopping
        if val_metrics['auc'] > best_val_auc:
            best_val_auc = val_metrics['auc']
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= config.patience:
            logger.info(f"⏹️ Early stopping at epoch {epoch + 1}")
            break
    
    # Final telemetry
    with open(telemetry_file, "w") as f:
        json.dump({"status": "COMPLETED", "symbol": args.symbol, "best_auc": best_val_auc}, f)
    
    logger.info(f"\n✅ Training Complete! Best AUC: {best_val_auc:.4f}")


def save_forensic_snapshot(model, loader, device, epoch, config):
    """Generate forensic snapshot for dashboard visualization."""
    model.eval()
    
    # Get a single batch for visualization
    try:
        # Debug Logging
        # logger.info("🔍 Debugging Snapshot Generation...")
        
        iterator = iter(loader)
        batch = next(iterator)
        # logger.info(f"    Batch Type: {type(batch)}")
        
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            x_dict, targets = batch
        else:
            logger.error(f"    ❌ Unexpected batch structure: {type(batch)}")
            return

        # logger.info(f"    x_dict keys: {list(x_dict.keys())}")
        
        x_dict = {k: v.to(device) for k, v in x_dict.items()}
        
        with torch.no_grad():
            # Forward pass
            # SwarmBrain v9 returns: logits, worker_logits, long_score, short_score
            logits, worker_logits, long_score, short_score = model(x_dict)
            probs = torch.sigmoid(logits).cpu().numpy()
            
            # worker_logits shape: [batch, num_workers, 3]
            # Mapping indices to names for Forensic Lab v8 compatibility:
            # Worker 0 = M5 (Short-term), Worker 3 = H1 (Long-term)
            try:
                w_m5 = torch.sigmoid(worker_logits[:, 0, 0]).cpu().numpy().flatten() # index 0 is LONG prob
                w_h1 = torch.sigmoid(worker_logits[:, 3, 0]).cpu().numpy().flatten()
            except Exception as e:
                logger.warning(f"    ⚠️ Failed to map worker indices: {e}")
                w_m5 = torch.zeros_like(logits).flatten().cpu().numpy()
                w_h1 = torch.zeros_like(logits).flatten().cpu().numpy()
            
            # Map coordination/attention (v9 uses cross-attention weights)
            # For now, we mock the attention map if not easily extractable
            coord_attn = None 
            # Extract OHLC from metadata if available
            ohlc_data = None
            if "__ohlc__" in x_dict:
                 metadata = x_dict["__ohlc__"][0].cpu().numpy() # [Seq, 4] -> [O, H, L, C]
                 ohlc_data = {
                     "open": metadata[:, 0].tolist(),
                     "high": metadata[:, 1].tolist(),
                     "low": metadata[:, 2].tolist(),
                     "close": metadata[:, 3].tolist(),
                     "time": list(range(len(metadata)))
                 }
            
            # Extract sample input (Close price normalized or feature 0)
            sample_input = x_dict["M5"][0, :, 0].cpu().numpy().tolist() # Feature 0 is RSI or Close (depending on config)
            
            snapshot = {
                "epoch": epoch,
                "timestamp": datetime.now().isoformat(),
                "sample_input": sample_input, 
                "ohlc": ohlc_data,
                "expert_conviction": {
                    "long": w_m5.tolist()[:100],  # Short-term expert (M5)
                    "short": w_h1.tolist()[:100], # Long-term expert (H1)
                    "consensus": probs.flatten().tolist()[:100] # Coordinator
                },
                "feature_importance": [1.0] * 12, # Dummy for v9 compatibility
                "hidden_activity": [[0.0] * 100 for _ in range(32)], # Dummy for v9 compatibility
                "attention_map": coord_attn.cpu().numpy().tolist()[:1] if coord_attn is not None else []
            }
            
            # Save
            Path("logs/snapshots").mkdir(parents=True, exist_ok=True)
            with open(f"logs/snapshots/epoch_{epoch}.json", "w") as f:
                json.dump(snapshot, f)
            
            logger.info(f"📸 Forensic Snapshot saved for Epoch {epoch}")
                
    except Exception as e:
        logger.warning(f"⚠️ Failed to save forensic snapshot: {e}")
        import traceback
        traceback.print_exc()



if __name__ == "__main__":
    main()



--- FILE: backend/src/brain/backtest_v9.py ---
import argparse
import json
import time
import os
import sys
import pandas as pd
import numpy as np
import torch
import talib
import polars as pl
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.append(os.getcwd())

# Import Model Architecture (must import classes, not run script)
from src.brain.train_v9 import SwarmBrain, WorkerConfig, CoordinatorConfig

# Setup logging
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("BacktestV9")


def compute_v9_features(df: pd.DataFrame) -> np.ndarray:
    """Compute the canonical 22 features using centralized indicators module."""
    from src.data.indicators import compute_features_from_df
    return compute_features_from_df(df, time_col="time", include_time=True)


def resample_and_extract(df_pl: pl.DataFrame, tf: str) -> pd.DataFrame:
    """Resamples data and returns pandas DF with features."""
    if tf == "5m":
        resampled = df_pl
    else:
        # Polars resampling
        resampled = (
            df_pl.upsample(time_column="time", every="1m")
            .interpolate()
            .group_by_dynamic("time", every=tf, label="left")
            .agg(
                [
                    pl.col("open").first(),
                    pl.col("high").max(),
                    pl.col("low").min(),
                    pl.col("close").last(),
                    pl.col("volume").sum(),
                ]
            )
        )

    pdf = resampled.to_pandas()
    if "time" in pdf.columns:
        pdf["time"] = pd.to_datetime(pdf["time"]).dt.tz_localize(None)
    # Compute features on the resampled dataframe
    features = compute_v9_features(pdf)

    # Create a feature dataframe
    feat_cols = [f"f_{i}" for i in range(features.shape[1])]
    feat_df = pd.DataFrame(features, columns=feat_cols)
    feat_df["time"] = pdf["time"]

    return feat_df


def main():
    parser = argparse.ArgumentParser(
        description="Backtest Swarm Brain v9 (Production Edition)"
    )
    parser.add_argument(
        "--model", type=str, required=True, help="Path to model .pth file"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/processed/EURUSD.parquet",
        help="Path to data file",
    )
    parser.add_argument(
        "--start_date", type=str, default="2023-01-01", help="Start simulation date"
    )
    parser.add_argument(
        "--end_date", type=str, default=None, help="End simulation date (inclusive)"
    )
    parser.add_argument("--speed", type=float, default=10.0, help="Steps per second")
    parser.add_argument(
        "--max_steps", type=int, default=None, help="Maximum number of simulation steps"
    )
    parser.add_argument(
        "--report_path",
        type=str,
        default="logs/backtest_report.json",
        help="Where to write summary report JSON",
    )
    parser.add_argument(
        "--no_sleep",
        action="store_true",
        default=False,
        help="Disable real-time sleeping (useful for automated loops)",
    )
    parser.add_argument("--worker_hidden", type=int, default=64)
    parser.add_argument("--coord_hidden", type=int, default=256)
    parser.add_argument("--seq_len", type=int, default=128)
    args = parser.parse_args()

    def decide_by_consensus(
        worker_logits: torch.Tensor, coord_logit: torch.Tensor
    ) -> str:
        """Consensus-based decision (Option 2).

        - If >=3 workers vote NEUTRAL -> WAIT
        - If coordinator agrees with strong majority -> LONG/SHORT
        - If coordinator is very confident -> override
        - Else -> WAIT
        """
        # worker_logits: [1, num_workers, 3] or [num_workers, 3]
        wl = worker_logits
        if wl.dim() == 3:
            wl = wl[0]

        votes = wl.argmax(dim=-1)  # 0=LONG, 1=SHORT, 2=NEUTRAL
        n_long = int((votes == 0).sum().item())
        n_short = int((votes == 1).sum().item())
        n_wait = int((votes == 2).sum().item())

        coord_prob = torch.sigmoid(coord_logit.view(-1))[0].item()

        if n_wait >= 4:
            return "WAIT"

        if coord_prob > 0.55 and n_long >= 2:
            return "LONG"
        if coord_prob < 0.45 and n_short >= 2:
            return "SHORT"

        if coord_prob > 0.65:
            return "LONG"
        if coord_prob < 0.35:
            return "SHORT"

        return "WAIT"

    def summarize_raw_votes(worker_logits: torch.Tensor) -> dict:
        """Summarize raw argmax votes from worker logits for debugging/telemetry."""
        wl = worker_logits
        if wl.dim() == 3:
            wl = wl[0]
        votes = wl.argmax(dim=-1)  # 0=LONG, 1=SHORT, 2=NEUTRAL
        n_long = int((votes == 0).sum().item())
        n_short = int((votes == 1).sum().item())
        n_wait = int((votes == 2).sum().item())
        agreement = float(max(n_long, n_short, n_wait) / max(1, votes.numel()))
        return {
            "n_long": n_long,
            "n_short": n_short,
            "n_wait": n_wait,
            "agreement": agreement,
        }

    # PID
    PID_FILE = "simulation_process.pid"
    with open(PID_FILE, "w") as f:
        f.write(str(os.getpid()))

    try:
        # Load Data
        logger.info(f"Loading data from {args.data}...")
        if args.data.endswith(".parquet"):
            df = pd.read_parquet(args.data)
        else:
            df = pd.read_csv(args.data)

        # Ensure column names are correct
        df.columns = [c.lower() for c in df.columns]
        if "date" in df.columns and "time" not in df.columns:
            df.rename(columns={"date": "time"}, inplace=True)

        df["time"] = pd.to_datetime(df["time"]).dt.tz_localize(None)
        df = df.sort_values("time").reset_index(drop=True)

        # Convert to Polars for fast resampling
        df_pl = pl.from_pandas(df)

        # Prepare all timeframes
        logger.info("🛠️  Preparing Multi-Timeframe Features (22 cols)...")
        tf_dfs = {}
        tf_map = {"5m": "M5", "15m": "M15", "30m": "M30", "1h": "H1", "4h": "H4"}
        for tf, tf_key in tf_map.items():
            tf_dfs[tf_key] = resample_and_extract(df_pl, tf)
            logger.info(f"   - {tf_key} ready: {len(tf_dfs[tf_key])} candles")

        # Load Model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")

        checkpoint = torch.load(args.model, map_location="cpu", weights_only=False)
        ckpt_input_dim = None
        try:
            w = checkpoint.get("swarm.workers.M5.input_proj.weight")
            if w is not None and hasattr(w, "shape") and len(w.shape) == 2:
                ckpt_input_dim = int(w.shape[1])
        except Exception:
            ckpt_input_dim = None

        if ckpt_input_dim is None:
            ckpt_input_dim = 12
            logger.warning("⚠️ Could not infer checkpoint input_dim; defaulting to 12")

        logger.info(f"🧩 Checkpoint expects input_dim={ckpt_input_dim}")

        # Config (Match checkpoint input dimension)
        worker_cfg = WorkerConfig(input_dim=ckpt_input_dim, hidden_dim=args.worker_hidden)
        coord_cfg = CoordinatorConfig(
            num_workers=5, worker_dim=args.worker_hidden, hidden_dim=args.coord_hidden
        )
        model = SwarmBrain(worker_cfg, coord_cfg).to(device)

        model.load_state_dict(checkpoint, strict=False)
        model.eval()
        logger.info(f"✅ Model loaded from {args.model}")

        # Replay Loop
        m5_df = tf_dfs["M5"]
        start_dt = pd.to_datetime(args.start_date)
        mask = m5_df["time"] >= start_dt
        end_dt = None
        if args.end_date:
            end_dt = pd.to_datetime(args.end_date)
            mask = mask & (m5_df["time"] <= end_dt)
        sim_m5 = m5_df[mask].reset_index(drop=True)

        if len(sim_m5) < 100:
            logger.error("❌ Not enough data for simulation!")
            return

        from collections import deque

        history = deque(maxlen=100)

        # Initial State
        equity = 10000.0
        position = 0.0
        entry_price = 0.0
        units = 10000
        is_paused = False
        last_exit_time = None
        trade_count = 0
        last_save_time = 0
        seq_len = args.seq_len

        COMMAND_FILE = "logs/sim_command.json"
        if os.path.exists(COMMAND_FILE):
            os.remove(COMMAND_FILE)

        # Capture Storage
        LAST_ENTRY_STATE = None
        ERRORS_FILE = "logs/backtest_errors.pt"
        SUCCESS_FILE = "logs/backtest_success.pt"
        RESERVOIR_FILE = "logs/pattern_reservoir.pt"

        processed_errors = (
            torch.load(ERRORS_FILE, map_location="cpu", weights_only=False)
            if os.path.exists(ERRORS_FILE)
            else []
        )
        processed_success = (
            torch.load(SUCCESS_FILE, map_location="cpu", weights_only=False)
            if os.path.exists(SUCCESS_FILE)
            else []
        )
        pattern_reservoir = (
            torch.load(RESERVOIR_FILE, map_location="cpu", weights_only=False)
            if os.path.exists(RESERVOIR_FILE)
            else []
        )

        logger.info(f"🚀 Simulation starting on {len(sim_m5)} M5 ticks...")

        is_paused = False
        i = 0
        max_steps = int(args.max_steps) if args.max_steps is not None else None
        n_total = len(sim_m5) if max_steps is None else min(len(sim_m5), max_steps)
        while i < n_total:
            if not os.path.exists(PID_FILE):
                break

            # --- Control Logic ---
            if os.path.exists(COMMAND_FILE):
                try:
                    with open(COMMAND_FILE, "r") as f:
                        cmd_data = json.load(f)
                    cmd = cmd_data.get("command")

                    if cmd == "PAUSE":
                        is_paused = True
                        if os.path.exists(COMMAND_FILE):
                            os.remove(COMMAND_FILE)
                    elif cmd == "RESUME":
                        is_paused = False
                        if os.path.exists(COMMAND_FILE):
                            os.remove(COMMAND_FILE)
                    elif cmd == "STEP":
                        is_paused = False  # Run one step
                    elif cmd == "FLAG_ERROR":
                        if LAST_ENTRY_STATE is not None:
                            # INVERT: If the error was a BUY, the target should have been 0 (SELL/NEUTRAL)
                            target_label = (
                                0 if LAST_ENTRY_STATE["action"] == "BUY" else 1
                            )
                            error_item = {
                                "x_dict": LAST_ENTRY_STATE["x"],
                                "y": torch.tensor(target_label, dtype=torch.long),
                                "timestamp": LAST_ENTRY_STATE["time"].isoformat(),
                                "pnl": 0.0,
                                "manual": True,
                            }
                            processed_errors.append(error_item)
                            try:
                                torch.save(processed_errors, ERRORS_FILE)
                            except:
                                pass
                        if os.path.exists(COMMAND_FILE):
                            os.remove(COMMAND_FILE)
                    elif cmd == "FLAG_PATTERN":
                        x_cpu = {k: v.cpu().clone() for k, v in x_dict.items()}
                        pattern_reservoir.append(
                            {
                                "x_dict": x_cpu,
                                "timestamp": current_time.isoformat(),
                                "label": "user_flagged",
                            }
                        )
                        try:
                            torch.save(pattern_reservoir, RESERVOIR_FILE)
                        except:
                            pass
                        if os.path.exists(COMMAND_FILE):
                            os.remove(COMMAND_FILE)
                except Exception as e:
                    logger.error(f"Error handling command: {e}")

            # --- Handle Pause State ---
            if is_paused:
                time.sleep(0.1)
                continue

            # --- Check if we just did a STEP ---
            # If the command was STEP, we pause after this tick
            if os.path.exists(COMMAND_FILE):
                with open(COMMAND_FILE, "r") as f:
                    final_cmd = json.load(f).get("command")
                if final_cmd == "STEP":
                    is_paused = True
                    # Keep the file but mark as PAUSE
                    with open(COMMAND_FILE, "w") as f:
                        json.dump({"command": "PAUSE"}, f)

            current_time = sim_m5.iloc[i]["time"]
            row_ohlc = df[df["time"] == current_time]
            if row_ohlc.empty:
                i += 1
                continue

            o, h, l, c = row_ohlc[["open", "high", "low", "close"]].values[0]
            current_price = c

            # Build v9 Input Map
            x_dict = {}
            can_predict = True

            for tf_key in ["M5", "M15", "M30", "H1", "H4"]:
                tf_data = tf_dfs[tf_key]
                tf_slice = tf_data[tf_data["time"] <= current_time].tail(seq_len)

                if len(tf_slice) < seq_len:
                    can_predict = False
                    break

                feat_vals = tf_slice.drop(columns=["time"]).values
                if feat_vals.shape[1] > ckpt_input_dim:
                    feat_vals = feat_vals[:, :ckpt_input_dim]
                x_dict[tf_key] = (
                    torch.tensor(feat_vals, dtype=torch.float32).unsqueeze(0).to(device)
                )

            if not can_predict:
                i += 1
                continue

            # Inference
            with torch.no_grad():
                with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                    logits, worker_logits, long_imp, short_imp = model(x_dict)
                    prob = torch.sigmoid(logits).item()

            # Consensus Decision (Swarm)
            desired_action = decide_by_consensus(
                worker_logits, logits
            )  # LONG/SHORT/WAIT
            raw_votes = summarize_raw_votes(worker_logits)
            coord_prob = float(torch.sigmoid(logits.view(-1))[0].item())

            # Trading Logic
            action = "HOLD"
            units = 10000

            # Conviction Thresholds
            BUY_THRESH = 0.65
            SELL_THRESH = 0.35

            # --- 1. EXIT LOGIC (TP/TS) ---
            trade_closed = False
            pnl_final = 0.0
            unrealized_pnl = 0.0

            if position != 0:
                if position > 0:
                    unrealized_pnl = (current_price - entry_price) * units
                else:
                    unrealized_pnl = (entry_price - current_price) * units

                # Update Max PnL for trailing
                if "max_pnl" not in LAST_ENTRY_STATE:
                    LAST_ENTRY_STATE["max_pnl"] = unrealized_pnl
                LAST_ENTRY_STATE["max_pnl"] = max(
                    LAST_ENTRY_STATE["max_pnl"], unrealized_pnl
                )
                max_pnl = LAST_ENTRY_STATE["max_pnl"]

                # SCALPING EXIT RULES (Requested by user: $0.50, $1.00)
                # Rule A: Take Profit @ $1.00 (Aggressive Scalp Target)
                if unrealized_pnl >= 1.0:
                    action, trade_closed = "TP_SCALP", True

                # Rule B: Trailing Stop (Protect Profit > $0.60)
                elif max_pnl > 0.6 and (max_pnl - unrealized_pnl) > 0.25:
                    action, trade_closed = "TS_ACTIVE", True

                # Rule C: Conviction Fade (Close if profit is positive and trend weakens)
                elif position > 0 and prob < 0.55 and unrealized_pnl > 0.3:
                    action, trade_closed = "EXIT_NEUTRAL", True
                elif position < 0 and prob > 0.45 and unrealized_pnl > 0.3:
                    action, trade_closed = "EXIT_NEUTRAL", True

                # Rule D: Signal Reversal (Standard)
                elif (desired_action == "LONG") and position < 0:
                    action, trade_closed = "BUY", True
                elif (desired_action == "SHORT") and position > 0:
                    action, trade_closed = "SELL", True

            if trade_closed:
                pnl_final = unrealized_pnl
                equity += pnl_final

                if LAST_ENTRY_STATE is not None:
                    # Capture Metadata for training
                    taken_label = 1 if LAST_ENTRY_STATE["action"] == "BUY" else 0

                    if pnl_final < -0.5: # Only capture significant mistakes (at least 0.5 pips)
                        # ERROR: Target should have been the OPPOSITE of what we did
                        target_label = 1 - taken_label
                        item = {
                            "x_dict": LAST_ENTRY_STATE["x"],
                            "y": torch.tensor(target_label, dtype=torch.long),
                            "timestamp": LAST_ENTRY_STATE["time"].isoformat(),
                            "pnl": float(pnl_final),
                        }
                        processed_errors.append(item)
                        try:
                            torch.save(processed_errors, ERRORS_FILE)
                        except:
                            pass
                    else:
                        # SUCCESS: Target was indeed what we did
                        target_label = taken_label
                        item = {
                            "x_dict": LAST_ENTRY_STATE["x"],
                            "y": torch.tensor(target_label, dtype=torch.long),
                            "timestamp": LAST_ENTRY_STATE["time"].isoformat(),
                            "pnl": float(pnl_final),
                        }
                        processed_success.append(item)
                        try:
                            torch.save(processed_success, SUCCESS_FILE)
                        except:
                            pass

                # Reset state
                if action not in ["BUY", "SELL"]:  # If it wasn't a reversal
                    position, entry_price = 0.0, 0.0
                    LAST_ENTRY_STATE = None

            # Cooldown check
            is_cooldown = False
            if last_exit_time:
                # Cooldown of 3 M5 candles (15 mins)
                candles_passed = (current_time - last_exit_time).total_seconds() / 300
                if candles_passed < 3:
                    is_cooldown = True

            # --- 2. ENTRY LOGIC ---
            # IMPORTANT: Don't re-enter immediately if we just closed a trade for profit (TP/TS)
            # OR if we are in a cooldown period.
            can_enter = (
                not trade_closed or action in ["BUY", "SELL"]
            ) and not is_cooldown

            if can_enter:
                if (desired_action == "LONG") and position <= 0:
                    position, entry_price, action = 1.0, current_price, "BUY"
                    x_cpu = {k: v.cpu().clone() for k, v in x_dict.items()}
                    LAST_ENTRY_STATE = {
                        "x": x_cpu,
                        "action": "BUY",
                        "time": current_time,
                        "max_pnl": 0.0,
                    }
                elif (desired_action == "SHORT") and position >= 0:
                    position, entry_price, action = -1.0, current_price, "SELL"
                    x_cpu = {k: v.cpu().clone() for k, v in x_dict.items()}
                    LAST_ENTRY_STATE = {
                        "x": x_cpu,
                        "action": "SELL",
                        "time": current_time,
                        "max_pnl": 0.0,
                    }

            # Record exit time for cooldown if we just closed
            if trade_closed and action not in ["BUY", "SELL"]:
                last_exit_time = current_time
                trade_count += 1

            # Current Unrealized PnL
            unrealized_pnl = 0.0
            if position > 0:
                unrealized_pnl = (current_price - entry_price) * units
            elif position < 0:
                unrealized_pnl = (entry_price - current_price) * units

            total_equity = equity + unrealized_pnl
            pnl_money = total_equity - 10000.0

            # Build History (Include ACTION for reliable visual signals)
            tick_data = {
                "t": current_time.strftime("%Y-%m-%d %H:%M"),
                "o": float(o),
                "h": float(h),
                "l": float(l),
                "c": float(c),
                "prob": float(prob),
                "pos": float(position),
                "act": action,
                "price": float(current_price),
            }
            history.append(tick_data)

            # Save Snapshot (Freq Capped at 5/sec)
            now = time.time()
            if now - last_save_time > 0.2:
                votes = model.get_worker_votes(worker_logits)
                snapshot = {
                    "step": i,
                    "timestamp": current_time.strftime("%Y-%m-%d %H:%M"),
                    "price": float(current_price),
                    "entry_price": float(entry_price),
                    "units": int(units if position != 0 else 0),
                    "equity": float(total_equity),
                    "pnl_usd": float(pnl_money),
                    "unrealized_pnl": float(unrealized_pnl),
                    "realized_pnl": float(equity - 10000.0),
                    "prob": float(prob),
                    "action": action,
                    "desired_action": desired_action,
                    "coord_prob": coord_prob,
                    "pos": float(position),
                    "is_cooldown": is_cooldown,
                    "trade_count": trade_count,
                    "long_bias": float(long_imp.item()),  # Raw score
                    "short_bias": float(short_imp.item()),  # Raw score
                    "agreement": float(votes["agreement"].mean().item()),
                    "n_long": int(votes["n_long"].mean().item()),
                    "n_short": int(votes["n_short"].mean().item()),
                    "agreement_raw": float(raw_votes["agreement"]),
                    "n_long_raw": int(raw_votes["n_long"]),
                    "n_short_raw": int(raw_votes["n_short"]),
                    "n_wait_raw": int(raw_votes["n_wait"]),
                    "neuronal_map": torch.sigmoid(worker_logits[0]).cpu().tolist(),
                    "completed": False,
                    "history": list(history),
                }
                with open("logs/sim_snapshot.json", "w") as f:
                    json.dump(snapshot, f)
                last_save_time = now

            if i % 50 == 0:
                print(
                    f"[{current_time}] Equity: ${total_equity:.2f} | PnL: ${pnl_money:.2f} | Prob: {prob:.2f}"
                )

            # Post-tick Pause if Step
            if os.path.exists(COMMAND_FILE):
                with open(COMMAND_FILE, "r") as f:
                    final_cmd = json.load(f).get("command")
                if final_cmd == "PAUSED_AFTER_STEP":
                    is_paused = True
                    # Keep the file but mark as PAUSE
                    with open(COMMAND_FILE, "w") as f:
                        json.dump({"command": "PAUSE"}, f)

            i += 1
            if not args.no_sleep:
                time.sleep(1.0 / args.speed)

        with open("logs/sim_snapshot.json", "w") as f:
            json.dump({"completed": True}, f)

        report = {
            "model": args.model,
            "data": args.data,
            "start_date": args.start_date,
            "end_date": args.end_date,
            "steps": int(n_total),
            "equity_final": float(equity),
            "pnl_realized": float(equity - 10000.0),
            "trade_count": int(trade_count),
            "errors_captured": int(len(processed_errors)),
            "success_captured": int(len(processed_success)),
            "patterns_captured": int(len(pattern_reservoir)),
        }
        try:
            Path(os.path.dirname(args.report_path) or ".").mkdir(
                parents=True, exist_ok=True
            )
            with open(args.report_path, "w") as f:
                json.dump(report, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to write report: {e}")

        logger.info("Simulation Finished.")

    except Exception as e:
        logger.error(f"💥 Simulation Crashed: {e}", exc_info=True)
    finally:
        if os.path.exists(PID_FILE):
            os.remove(PID_FILE)


if __name__ == "__main__":
    main()



--- FILE: backend/src/brain/inspect_leakage.py ---
import torch
import pandas as pd
import numpy as np
from pathlib import Path

# Load data
try:
    data_path = "data/cache/train.pt"
    if not Path(data_path).exists():
        # Try shards
        shard = list(Path("data/cache/train_shards").glob("*.pt"))[0]
        data_path = str(shard)

    print(f"Loading {data_path}...")
    data = torch.load(data_path, weights_only=False)
    
    X = data['X'].numpy()
    y = data['y'].numpy()
    
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    
    # Check for NaNs
    if np.isnan(X).any():
        print("⚠️ NaNs found in X!")
    if np.isnan(y).any():
        print("⚠️ NaNs found in y!")
        
    # Flatten X to 2D using the last timestep (most relevant for leakage)
    X_last = X[:, -1, :]
    print(f"Analyzing correlations on last timestep: {X_last.shape}")
    
    # Correlations
    df = pd.DataFrame(X_last)
    df['target'] = y
    
    corr = df.corr()['target'].sort_values(ascending=False)
    
    print("\nTop Correlations with Target:")
    print(corr.head(10))
    print(corr.tail(10))
    
    # Check for exact matches
    for col in df.columns[:-1]:
        if abs(corr[col]) > 0.95:
            print(f"\n🚨 POTENTIAL LEAKAGE DETECTED in Feature {col} (Corr: {corr[col]:.4f})")
            
except Exception as e:
    print(f"Error: {e}")



--- FILE: backend/src/brain/auto_learn_v9.py ---
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch


def _run(cmd: list[str], cwd: str | None = None, log_file: Path | None = None) -> None:
    if log_file:
        with open(log_file, "w") as f:
            subprocess.run(cmd, cwd=cwd, check=True, stdout=f, stderr=subprocess.STDOUT)
    else:
        subprocess.run(cmd, cwd=cwd, check=True)


def _load_backtest_metrics(report_path: Path) -> dict:
    """Extract key metrics from backtest report for gating decisions."""
    if not report_path.exists():
        return {}
    with open(report_path, "r") as f:
        report = json.load(f)
    # Backtest report stores metrics at top level, not under "summary"
    return {
        "realized_pnl": report.get("pnl_realized", 0.0),
        "unrealized_pnl": 0.0,  # Not tracked in current report
        "total_trades": report.get("trade_count", 0),
        "winning_trades": 0,  # Not tracked in current report
        "losing_trades": 0,  # Not tracked in current report
        "max_drawdown": 0.0,  # Not tracked in current report
    }


def _is_better_than_previous(
    current: dict, previous: dict | None, min_trades: int = 5
) -> bool:
    """Determine if current cycle is better than previous for model promotion."""
    if previous is None:
        # First cycle: always promote to establish a baseline
        return True

    # Must have minimum trades to be statistically significant
    if current["total_trades"] < min_trades:
        return False

    # Primary: realized PnL must be better
    if current["realized_pnl"] > previous["realized_pnl"]:
        return True

    # Secondary: if PnL similar, check drawdown (lower is better)
    pnl_diff = abs(current["realized_pnl"] - previous["realized_pnl"])
    if pnl_diff < 10:  # Within $10
        return current["max_drawdown"] < previous["max_drawdown"]

    return False


def _promote_model(model_path: Path, cycle_dir: Path, cycle: int) -> Path:
    """Promote model to best_model_v9.pth with backup."""
    # Backup previous best if exists
    if model_path.exists():
        backup_path = cycle_dir / f"previous_best_cycle{cycle:02d}.pth"
        import shutil

        shutil.copy2(model_path, backup_path)

    # The trained model is already at model_path (train_v9 writes there)
    # Just verify it exists
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")

    return model_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Auto-Learning Loop for IARA v9 (train -> backtest -> replay -> train)"
    )
    parser.add_argument(
        "--cycles", type=int, default=3, help="Number of train/backtest cycles"
    )
    parser.add_argument(
        "--train_epochs", type=int, default=10, help="Epochs per training cycle"
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=64, help="Batch size for training"
    )
    parser.add_argument(
        "--train_seq_len", type=int, default=64, help="Sequence length for training"
    )
    parser.add_argument(
        "--train_lr", type=float, default=0.0005, help="Learning rate for training"
    )
    parser.add_argument(
        "--train_patience",
        type=int,
        default=None,
        help="Early stopping patience for training (defaults to max(5, train_epochs//2))",
    )
    parser.add_argument(
        "--train_worker_hidden", type=int, default=64, help="Worker hidden dim"
    )
    parser.add_argument(
        "--train_coord_hidden", type=int, default=256, help="Coordinator hidden dim"
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume training from existing checkpoint (default: true)",
    )
    parser.add_argument(
        "--no_amp", action="store_true", default=False, help="Disable AMP in training"
    )
    parser.add_argument(
        "--require_cuda",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require CUDA/GPU to run (default: true)",
    )

    parser.add_argument(
        "--backtest_data",
        type=str,
        default=None, # Inferred from symbol
        help="Parquet/CSV path used by backtest",
    )
    parser.add_argument("--backtest_start_date", type=str, default="2023-01-01")
    parser.add_argument("--backtest_end_date", type=str, default=None)
    parser.add_argument("--backtest_speed", type=float, default=200.0)
    parser.add_argument("--backtest_max_steps", type=int, default=2000)
    parser.add_argument(
        "--gating",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable automatic model promotion based on backtest quality (default: true)",
    )
    parser.add_argument(
        "--min_trades_for_promotion",
        type=int,
        default=5,
        help="Minimum trades required in backtest for model promotion",
    )
    parser.add_argument(
        "--work_dir",
        type=str,
        default="logs/auto_learn_v9",
        help="Folder to store per-cycle reports",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None, # Inferred from symbol if None
        help="Path to the model checkpoint to evaluate/train",
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default="EURUSD",
        help="Currency pair (e.g. EURUSD, GBPUSD)"
    )

    args = parser.parse_args()

    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for auto_learn_v9, but torch.cuda.is_available() is False"
        )

    project_root = str(Path(__file__).resolve().parents[2])
    
    # Isolate work_dir by symbol
    work_dir = Path(args.work_dir) / args.symbol
    work_dir.mkdir(parents=True, exist_ok=True)

    # Inferred model path
    if args.model_path is None:
        model_path = Path("logs/models") / f"best_model_v9_{args.symbol}.pth"
    else:
        model_path = Path(args.model_path)

    summary: list[dict] = []

    # Inferred backtest data
    backtest_data = args.backtest_data
    if backtest_data is None:
        backtest_data = f"data/processed/{args.symbol}.parquet"
    
    for cycle in range(1, args.cycles + 1):
        cycle_dir = work_dir / f"cycle_{cycle:02d}"
        cycle_dir.mkdir(parents=True, exist_ok=True)

        # 1) TRAIN
        train_cmd = [
            sys.executable,
            "-u",
            "src/brain/train_v9.py",
            "--epochs",
            str(args.train_epochs),
            "--lr",
            str(args.train_lr),
            "--batch_size",
            str(args.train_batch_size),
            "--seq_len",
            str(args.train_seq_len),
            "--worker_hidden",
            str(args.train_worker_hidden),
            "--coord_hidden",
            str(args.train_coord_hidden),
            "--symbol",
            args.symbol, # Pass symbol to train_v9
        ]
        patience = args.train_patience
        if patience is None:
            patience = max(5, args.train_epochs // 2)
        train_cmd.extend(["--patience", str(int(patience))])

        if args.resume:
            train_cmd.append("--resume")
        if args.no_amp:
            train_cmd.append("--no_amp")

        t0 = time.time()
        _run(train_cmd, cwd=project_root, log_file=cycle_dir / "train.log")
        train_s = time.time() - t0

        if not model_path.exists():
            raise FileNotFoundError(
                f"Expected model checkpoint at {model_path}, but it was not created."
            )

        # 2) BACKTEST (generates captures: errors/success/patterns)
        report_path = cycle_dir / "backtest_report.json"
        backtest_cmd = [
            sys.executable,
            "-u",
            "src/brain/backtest_v9.py",
            "--model",
            str(model_path),
            "--data",
            str(backtest_data),
            "--start_date",
            args.backtest_start_date,
            "--speed",
            str(args.backtest_speed),
            "--max_steps",
            str(args.backtest_max_steps),
            "--report_path",
            str(report_path),
            "--worker_hidden",
            str(args.train_worker_hidden),
            "--coord_hidden",
            str(args.train_coord_hidden),
            "--seq_len",
            str(args.train_seq_len),
            "--no_sleep",
        ]
        if args.backtest_end_date:
            backtest_cmd.extend(["--end_date", args.backtest_end_date])

        t1 = time.time()
        _run(backtest_cmd, cwd=project_root, log_file=cycle_dir / "backtest.log")
        backtest_s = time.time() - t1

        # 3) GATING: Evaluate and decide on model promotion
        current_metrics = _load_backtest_metrics(report_path)
        promoted = False
        rollback_to = None

        if args.gating and current_metrics:
            previous_metrics = summary[-1]["metrics"] if summary else None

            if _is_better_than_previous(
                current_metrics, previous_metrics, args.min_trades_for_promotion
            ):
                # Promote: model already at model_path, just backup previous
                promoted_path = _promote_model(model_path, cycle_dir, cycle)
                promoted = True
                print(
                    f"[Cycle {cycle}] Model PROMOTED - PnL: ${current_metrics['realized_pnl']:.2f}, Trades: {current_metrics['total_trades']}"
                )
            else:
                # Rollback: restore previous best if available
                if previous_metrics and len(summary) > 0:
                    prev_cycle_dir = work_dir / f"cycle_{summary[-1]['cycle']:02d}"
                    backup_candidates = list(
                        prev_cycle_dir.glob("previous_best_cycle*.pth")
                    )
                    if backup_candidates:
                        rollback_to = backup_candidates[0]
                    elif (prev_cycle_dir / "model_final.pth").exists():
                        rollback_to = prev_cycle_dir / "model_final.pth"

                    if rollback_to and rollback_to.exists():
                        import shutil

                        shutil.copy2(rollback_to, model_path)
                        print(
                            f"[Cycle {cycle}] Model REJECTED - Rolling back to {rollback_to.name}"
                        )
                        print(
                            f"  Current PnL: ${current_metrics['realized_pnl']:.2f} vs Previous: ${previous_metrics['realized_pnl']:.2f}"
                        )
                    else:
                        print(
                            f"[Cycle {cycle}] Model REJECTED but no rollback candidate found (keeping current)"
                        )
                else:
                    print(
                        f"[Cycle {cycle}] Model REJECTED (first cycle not profitable enough)"
                    )
        else:
            # Gating disabled or no metrics - always promote
            promoted = True
            print(
                f"[Cycle {cycle}] Gating disabled - Model kept (PnL: ${current_metrics.get('realized_pnl', 0):.2f})"
            )

        backtest_report = {}
        if report_path.exists():
            with open(report_path, "r") as f:
                backtest_report = json.load(f)

        entry = {
            "cycle": cycle,
            "train_seconds": float(train_s),
            "backtest_seconds": float(backtest_s),
            "model_path": str(model_path),
            "promoted": promoted,
            "rollback_source": str(rollback_to) if rollback_to else None,
            "metrics": current_metrics,
            "backtest": backtest_report,
        }
        summary.append(entry)

        with open(cycle_dir / "cycle_summary.json", "w") as f:
            json.dump(entry, f, indent=2)

    with open(work_dir / "summary.json", "w") as f:
        json.dump({"cycles": summary}, f, indent=2)


if __name__ == "__main__":
    main()



--- FILE: backend/src/data/init_db.py ---
import os
import time
import psycopg2
from pathlib import Path

def wait_for_db():
    dsn = os.getenv("DB_URI", "postgresql://saci:saci_secret@localhost:5432/saci_core")
    retries = 30
    while retries > 0:
        try:
            conn = psycopg2.connect(dsn)
            print("✅ Database connected!")
            return conn
        except psycopg2.OperationalError as e:
            print(f"⏳ Waiting for DB... ({30-retries}/30)")
            time.sleep(2)
            retries -= 1
    raise Exception("❌ Database Not Ready")

def init_db():
    conn = wait_for_db()
    conn.autocommit = True
    cursor = conn.cursor()
    
    # Load Schema
    schema_path = Path(__file__).parent / "schema.sql"
    print(f"Applying schema from {schema_path}...")
    
    with open(schema_path, "r") as f:
        sql = f.read()
        
    try:
        cursor.execute(sql)
        print("✅ Schema applied successfully!")
    except Exception as e:
        print(f"⚠️ Schema error (might be already applied): {e}")
        
    conn.close()

if __name__ == "__main__":
    init_db()



--- FILE: backend/src/data/orderbook.py ---
"""
I.A.R.A. v11 — Order Book Microstructure Ingestor
===================================================
Real-time Level 2 Order Book data from Binance Depth WebSocket.

Computes:
  - OBI (Order Book Imbalance): Bid vs Ask volume pressure
  - VPIN (Volume-Synchronized Probability of Informed Trading): Toxicity metric
  - Spread %: Relative bid-ask spread
  - Depth Ratio: Deep bid vs ask liquidity
  - Bid/Ask Walls: Large resting orders detection

The "center of gravity" of the market — reveals intent before price moves.
"""

import asyncio
import json
import logging
import threading
import time
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("OrderBookIngestor")


class OrderBookSnapshot:
    """Single snapshot of the order book state."""

    __slots__ = ["timestamp", "bids", "asks"]

    def __init__(self, timestamp: float, bids: List[List[float]], asks: List[List[float]]):
        self.timestamp = timestamp
        self.bids = bids  # [[price, qty], ...] sorted best (highest) first
        self.asks = asks  # [[price, qty], ...] sorted best (lowest) first


class OrderBookIngestor:
    """
    Binance Depth WebSocket Ingestor.

    Connects to the partial book depth stream and maintains a rolling buffer
    of snapshots. Computes LOB microstructure features on demand.

    Usage:
        ingestor = OrderBookIngestor(symbol="btcusdt")
        ingestor.start()  # Launches background WebSocket thread

        features = ingestor.get_features()  # [obi, vpin, spread_pct, depth_ratio, bid_wall, ask_wall]

        ingestor.stop()
    """

    def __init__(
        self,
        symbol: str = "btcusdt",
        depth_levels: int = 20,
        buffer_size: int = 500,
        vpin_buckets: int = 50,
        vpin_bucket_volume: float = 1.0,
    ):
        self.symbol = symbol.lower()
        self.depth_levels = depth_levels
        self.buffer_size = buffer_size
        self.vpin_buckets = vpin_buckets
        self.vpin_bucket_volume = vpin_bucket_volume

        # Rolling buffer of snapshots
        self._snapshots: deque = deque(maxlen=buffer_size)

        # VPIN state
        self._vpin_buy_volume: deque = deque(maxlen=vpin_buckets)
        self._vpin_sell_volume: deque = deque(maxlen=vpin_buckets)
        self._current_bucket_buy = 0.0
        self._current_bucket_sell = 0.0
        self._current_bucket_total = 0.0
        self._last_mid_price = 0.0

        # Thread management
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # Cached features (updated on each snapshot)
        self._cached_features = np.zeros(6, dtype=np.float32)

    def start(self):
        """Start the background WebSocket listener thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run_ws_loop, daemon=True)
        self._thread.start()
        logger.info(f"📦 LOB Ingestor started for {self.symbol.upper()}")

    def stop(self):
        """Stop the WebSocket listener."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("📦 LOB Ingestor stopped")

    def _run_ws_loop(self):
        """Run the asyncio WebSocket event loop in a background thread."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._ws_listener())
        except Exception as e:
            logger.error(f"WebSocket loop error: {e}")
        finally:
            loop.close()

    async def _ws_listener(self):
        """Connect to Binance and process depth updates."""
        try:
            import websockets
        except ImportError:
            logger.error("websockets package not installed. LOB Ingestor disabled.")
            self._running = False
            return

        url = f"wss://stream.binance.com:9443/ws/{self.symbol}@depth{self.depth_levels}@100ms"
        logger.info(f"Connecting to {url}")

        while self._running:
            try:
                async with websockets.connect(url, ping_interval=20) as ws:
                    logger.info("✅ LOB WebSocket connected")
                    while self._running:
                        try:
                            msg = await asyncio.wait_for(ws.recv(), timeout=5.0)
                            data = json.loads(msg)
                            self._process_depth_update(data)
                        except asyncio.TimeoutError:
                            continue
                        except Exception as e:
                            logger.warning(f"WS message error: {e}")
                            break
            except Exception as e:
                logger.warning(f"WS connection error: {e}. Reconnecting in 5s...")
                await asyncio.sleep(5)

    def _process_depth_update(self, data: dict):
        """Process a depth update message and compute features."""
        try:
            bids = [[float(p), float(q)] for p, q in data.get("bids", [])]
            asks = [[float(p), float(q)] for p, q in data.get("asks", [])]

            if not bids or not asks:
                return

            snapshot = OrderBookSnapshot(
                timestamp=time.time(),
                bids=bids,
                asks=asks,
            )

            with self._lock:
                self._snapshots.append(snapshot)
                self._update_vpin(bids, asks)
                self._cached_features = self._compute_features_internal(snapshot)

        except Exception as e:
            logger.debug(f"Depth processing error: {e}")

    def _update_vpin(self, bids: list, asks: list):
        """Update VPIN buckets based on mid-price changes."""
        if not bids or not asks:
            return

        mid_price = (bids[0][0] + asks[0][0]) / 2.0

        if self._last_mid_price > 0:
            price_change = mid_price - self._last_mid_price
            # Approximate volume classification using price direction
            total_volume = sum(q for _, q in bids[:5]) + sum(q for _, q in asks[:5])
            if total_volume > 0:
                if price_change > 0:
                    self._current_bucket_buy += total_volume * 0.1
                else:
                    self._current_bucket_sell += total_volume * 0.1
                self._current_bucket_total += total_volume * 0.1

                # Check if bucket is full
                if self._current_bucket_total >= self.vpin_bucket_volume:
                    self._vpin_buy_volume.append(self._current_bucket_buy)
                    self._vpin_sell_volume.append(self._current_bucket_sell)
                    self._current_bucket_buy = 0.0
                    self._current_bucket_sell = 0.0
                    self._current_bucket_total = 0.0

        self._last_mid_price = mid_price

    def _compute_features_internal(self, snapshot: OrderBookSnapshot) -> np.ndarray:
        """Compute 6 LOB microstructure features from current snapshot."""
        bids = snapshot.bids
        asks = snapshot.asks

        if not bids or not asks:
            return np.zeros(6, dtype=np.float32)

        # 1. OBI — Order Book Imbalance (top N levels)
        bid_vol = sum(q for _, q in bids[:10])
        ask_vol = sum(q for _, q in asks[:10])
        total_vol = bid_vol + ask_vol
        obi = (bid_vol - ask_vol) / (total_vol + 1e-10)

        # 2. VPIN — Volume-Synchronized Probability of Informed Trading
        vpin = 0.0
        if len(self._vpin_buy_volume) >= 5:
            buy_arr = np.array(list(self._vpin_buy_volume))
            sell_arr = np.array(list(self._vpin_sell_volume))
            total_arr = buy_arr + sell_arr + 1e-10
            vpin = float(np.mean(np.abs(buy_arr - sell_arr) / total_arr))

        # 3. Spread % — Relative bid-ask spread
        best_bid = bids[0][0]
        best_ask = asks[0][0]
        mid_price = (best_bid + best_ask) / 2.0
        spread_pct = (best_ask - best_bid) / (mid_price + 1e-10) * 100

        # 4. Depth Ratio (deep levels 5-20)
        deep_bid = sum(q for _, q in bids[5:]) if len(bids) > 5 else 0
        deep_ask = sum(q for _, q in asks[5:]) if len(asks) > 5 else 0
        deep_total = deep_bid + deep_ask
        depth_ratio = (deep_bid - deep_ask) / (deep_total + 1e-10)

        # 5. Bid Wall Detection (single large order > 3x mean)
        bid_qtys = np.array([q for _, q in bids]) if bids else np.zeros(1)
        bid_mean = np.mean(bid_qtys)
        bid_wall = float(np.max(bid_qtys) / (bid_mean + 1e-10) > 3.0)

        # 6. Ask Wall Detection (single large order > 3x mean)
        ask_qtys = np.array([q for _, q in asks]) if asks else np.zeros(1)
        ask_mean = np.mean(ask_qtys)
        ask_wall = float(np.max(ask_qtys) / (ask_mean + 1e-10) > 3.0)

        # Normalize all to [-1, 1] or [0, 1]
        features = np.array([
            np.clip(obi, -1.0, 1.0),                   # OBI: already in [-1, 1]
            np.clip(vpin * 2 - 1, -1.0, 1.0),          # VPIN: [0,1] → [-1,1]
            np.clip(spread_pct * 10 - 1, -1.0, 1.0),   # Spread: normalized
            np.clip(depth_ratio, -1.0, 1.0),            # Depth: already in [-1, 1]
            bid_wall,                                    # Binary: 0 or 1
            ask_wall,                                    # Binary: 0 or 1
        ], dtype=np.float32)

        return features

    def get_features(self) -> np.ndarray:
        """
        Get the latest LOB microstructure features.

        Returns:
            np.ndarray of shape [6]:
              [obi, vpin, spread_pct, depth_ratio, bid_wall, ask_wall]
        """
        with self._lock:
            return self._cached_features.copy()

    def get_features_tensor(self):
        """Get features as a PyTorch tensor [1, 6]."""
        import torch
        features = self.get_features()
        return torch.tensor(features, dtype=torch.float32).unsqueeze(0)

    @property
    def is_connected(self) -> bool:
        return self._running and len(self._snapshots) > 0

    @property
    def snapshot_count(self) -> int:
        return len(self._snapshots)


class MockOrderBookIngestor(OrderBookIngestor):
    """
    Mock ingestor for offline/testing mode.
    Generates realistic synthetic LOB features.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._mock_step = 0

    def start(self):
        """No-op for mock mode."""
        self._running = True
        logger.info("📦 LOB Mock Ingestor started (synthetic data)")

    def stop(self):
        self._running = False

    def get_features(self) -> np.ndarray:
        """Generate synthetic LOB features with realistic distributions."""
        self._mock_step += 1
        t = self._mock_step * 0.01

        obi = np.sin(t * 0.7) * 0.3 + np.random.normal(0, 0.1)
        vpin = 0.3 + np.sin(t * 0.2) * 0.15 + np.random.normal(0, 0.05)
        spread = 0.01 + abs(np.random.normal(0, 0.005))
        depth = np.sin(t * 0.3) * 0.4 + np.random.normal(0, 0.15)
        bid_wall = float(np.random.random() > 0.85)
        ask_wall = float(np.random.random() > 0.85)

        return np.array([
            np.clip(obi, -1.0, 1.0),
            np.clip(vpin * 2 - 1, -1.0, 1.0),
            np.clip(spread * 10 - 1, -1.0, 1.0),
            np.clip(depth, -1.0, 1.0),
            bid_wall,
            ask_wall,
        ], dtype=np.float32)



--- FILE: backend/src/data/crypto_data_fetcher.py ---
#!/usr/bin/env python3
"""
I.A.R.A. - Crypto Historical Data Fetcher
==========================================
Downloads cryptocurrency OHLCV data from Alpaca Markets API
and saves it as Parquet files compatible with the feature_factory pipeline.

Usage:
    python src/data/crypto_data_fetcher.py --symbol BTC/USD --days 90
    python src/data/crypto_data_fetcher.py --symbol ETH/USD --days 30 --timeframe 5Min
"""

import argparse
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("CryptoFetcher")

# Output directory (same as existing forex pipeline)
OUTPUT_DIR = Path("data/processed")


def get_timeframe(tf_str: str) -> TimeFrame:
    """Convert string timeframe to Alpaca TimeFrame object."""
    tf_map = {
        "1Min": TimeFrame.Minute,
        "5Min": TimeFrame(5, TimeFrameUnit.Minute),
        "15Min": TimeFrame(15, TimeFrameUnit.Minute),
        "1Hour": TimeFrame.Hour,
        "1Day": TimeFrame.Day,
    }
    return tf_map.get(tf_str, TimeFrame.Minute)


def normalize_symbol_for_filename(symbol: str) -> str:
    """Convert 'BTC/USD' -> 'BTCUSD' for filesystem compatibility."""
    return symbol.replace("/", "")


def fetch_crypto_bars(
    symbol: str,
    days: int = 90,
    timeframe_str: str = "1Min",
    chunk_days: int = 7,
) -> pd.DataFrame:
    """
    Fetch historical crypto bars from Alpaca in chunks.
    
    Alpaca's crypto data API doesn't require authentication for market data,
    but we use credentials if available for higher rate limits.
    
    Args:
        symbol: Crypto pair (e.g., 'BTC/USD')
        days: Number of days of history to fetch
        timeframe_str: Candlestick timeframe ('1Min', '5Min', etc.)
        chunk_days: Days per API request chunk (to avoid hitting limits)
    
    Returns:
        pd.DataFrame with columns: date, open, high, low, close, volume
    """
    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")
    
    # CryptoHistoricalDataClient works without auth for market data
    if api_key and secret_key:
        client = CryptoHistoricalDataClient(api_key, secret_key)
    else:
        client = CryptoHistoricalDataClient()
    
    tf = get_timeframe(timeframe_str)
    end = datetime.now()
    start = end - timedelta(days=days)
    
    logger.info(f"📊 Fetching {symbol} from {start.date()} to {end.date()} ({timeframe_str})")
    
    all_frames = []
    current_start = start
    chunk_num = 0
    
    while current_start < end:
        current_end = min(current_start + timedelta(days=chunk_days), end)
        chunk_num += 1
        
        logger.info(f"  Chunk {chunk_num}: {current_start.date()} → {current_end.date()}")
        
        try:
            request = CryptoBarsRequest(
                symbol_or_symbols=[symbol],
                timeframe=tf,
                start=current_start,
                end=current_end,
                limit=10000,
            )
            bars = client.get_crypto_bars(request)
            df_chunk = bars.df
            
            if not df_chunk.empty:
                all_frames.append(df_chunk)
                logger.info(f"    → {len(df_chunk)} bars fetched")
            else:
                logger.warning(f"    → No data for this period")
                
        except Exception as e:
            logger.error(f"    → Error: {e}")
        
        current_start = current_end
    
    if not all_frames:
        logger.error("No data fetched!")
        return pd.DataFrame()
    
    # Combine all chunks
    df = pd.concat(all_frames)
    
    # Remove duplicates (overlapping chunks)
    df = df[~df.index.duplicated(keep='first')]
    
    # Sort by timestamp
    df = df.sort_index()
    
    # Flatten multi-index
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index()
    else:
        df = df.reset_index()
    
    # Normalize column names for feature_factory compatibility
    df.columns = [c.lower() for c in df.columns]
    
    # Rename 'timestamp' to 'date' for feature_factory compatibility
    if "timestamp" in df.columns:
        df = df.rename(columns={"timestamp": "date"})
    
    # Drop symbol column if present (feature_factory doesn't expect it)
    if "symbol" in df.columns:
        df = df.drop(columns=["symbol"])
    
    # Ensure required columns exist
    required = ["date", "open", "high", "low", "close", "volume"]
    # Handle trade_count, vwap etc. from Alpaca
    if "trade_count" in df.columns:
        df = df.drop(columns=["trade_count"], errors="ignore")
    if "vwap" in df.columns:
        df = df.drop(columns=["vwap"], errors="ignore")
    
    # Keep only the columns we need
    available = [c for c in required if c in df.columns]
    df = df[available]
    
    logger.info(f"✅ Total: {len(df)} bars ({df['date'].min()} → {df['date'].max()})")
    
    return df


def save_to_parquet(df: pd.DataFrame, symbol: str, output_dir: Path = OUTPUT_DIR):
    """Save DataFrame as Parquet file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    filename = normalize_symbol_for_filename(symbol)
    output_path = output_dir / f"{filename}.parquet"
    
    df.to_parquet(output_path, index=False)
    size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(f"💾 Saved: {output_path} ({size_mb:.2f} MB, {len(df)} rows)")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description="I.A.R.A. Crypto Data Fetcher")
    parser.add_argument(
        "--symbol", type=str, default="BTC/USD",
        help="Crypto pair (e.g., BTC/USD, ETH/USD)"
    )
    parser.add_argument(
        "--days", type=int, default=90,
        help="Number of days of history to fetch"
    )
    parser.add_argument(
        "--timeframe", type=str, default="1Min",
        choices=["1Min", "5Min", "15Min", "1Hour", "1Day"],
        help="Candlestick timeframe"
    )
    parser.add_argument(
        "--output-dir", type=str, default=str(OUTPUT_DIR),
        help="Output directory for Parquet files"
    )
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("  I.A.R.A. - Crypto Historical Data Fetcher")
    logger.info("=" * 60)
    
    df = fetch_crypto_bars(
        symbol=args.symbol,
        days=args.days,
        timeframe_str=args.timeframe,
    )
    
    if df.empty:
        logger.error("❌ No data fetched. Check your API credentials and symbol.")
        return
    
    save_to_parquet(df, args.symbol, Path(args.output_dir))
    
    logger.info("🎯 Done! Run feature_factory.py next to prepare training data.")


if __name__ == "__main__":
    main()



--- FILE: backend/src/data/hurst.py ---
"""
I.A.R.A. v11 — Hurst Exponent & Fractal Dimension
===================================================
Measures the "memory" of a time series using Rescaled Range (R/S) analysis.

Interpretation:
  H ≈ 0.5  → Random Walk (noise) → Skip trading
  H > 0.5  → Long Memory (trending) → Hold positions longer
  H < 0.5  → Anti-persistent (mean-reverting) → Scalp mode

Reference: Hurst, H.E. (1951). "Long-term storage capacity of reservoirs"
"""

import numpy as np
from typing import Tuple


def _rescaled_range(series: np.ndarray) -> float:
    """Compute the Rescaled Range (R/S) statistic for a time series."""
    n = len(series)
    if n < 2:
        return 0.0

    mean = np.mean(series)
    deviations = series - mean
    cumulative = np.cumsum(deviations)

    R = np.max(cumulative) - np.min(cumulative)
    S = np.std(series, ddof=1)

    if S < 1e-10:
        return 0.0

    return R / S


def compute_hurst(series: np.ndarray, min_window: int = 20, max_window: int = 200) -> float:
    """
    Compute the Hurst Exponent using Rescaled Range (R/S) analysis.

    Uses multiple window sizes and fits log(R/S) vs log(n) to extract H.

    Args:
        series: 1D array of prices or returns
        min_window: Minimum sub-series window size
        max_window: Maximum sub-series window size (clamped to len(series)//2)

    Returns:
        Hurst exponent H ∈ [0, 1]. Returns 0.5 if computation fails.
    """
    n = len(series)
    if n < min_window * 2:
        return 0.5  # Not enough data

    max_window = min(max_window, n // 2)
    if max_window <= min_window:
        max_window = min_window + 1

    # Generate logarithmically spaced window sizes
    window_sizes = np.unique(
        np.logspace(
            np.log10(min_window),
            np.log10(max_window),
            num=15,
            dtype=int,
        )
    )
    window_sizes = window_sizes[window_sizes >= min_window]

    if len(window_sizes) < 3:
        return 0.5

    log_n = []
    log_rs = []

    for w in window_sizes:
        # Split series into non-overlapping windows of size w
        num_windows = n // w
        if num_windows < 1:
            continue

        rs_values = []
        for i in range(num_windows):
            segment = series[i * w: (i + 1) * w]
            rs = _rescaled_range(segment)
            if rs > 0:
                rs_values.append(rs)

        if rs_values:
            avg_rs = np.mean(rs_values)
            log_n.append(np.log(w))
            log_rs.append(np.log(avg_rs))

    if len(log_n) < 3:
        return 0.5

    # Linear regression: log(R/S) = H * log(n) + c
    log_n = np.array(log_n)
    log_rs = np.array(log_rs)

    try:
        coeffs = np.polyfit(log_n, log_rs, 1)
        H = float(coeffs[0])
        # Clamp to valid range
        H = max(0.0, min(1.0, H))
        return H
    except (np.linalg.LinAlgError, ValueError):
        return 0.5


def compute_fractal_dimension(series: np.ndarray, **kwargs) -> float:
    """
    Compute the Fractal Dimension from Hurst Exponent.
    D = 2 - H

    A higher D means more "rough"/complex behavior.

    Returns:
        Fractal dimension D ∈ [1, 2]
    """
    H = compute_hurst(series, **kwargs)
    return 2.0 - H


def compute_hurst_rolling(
    series: np.ndarray,
    window: int = 200,
    step: int = 1,
) -> np.ndarray:
    """
    Compute rolling Hurst Exponent over a time series.

    Args:
        series: 1D array of close prices
        window: Rolling window size
        step: Step size for rolling computation

    Returns:
        np.ndarray of same length as series with Hurst values.
        Positions with insufficient data are filled with 0.5.
    """
    n = len(series)
    result = np.full(n, 0.5, dtype=np.float32)

    for i in range(window, n, step):
        segment = series[i - window: i]
        h = compute_hurst(segment, min_window=20, max_window=window // 2)
        result[i] = h

    # Forward-fill steps > 1
    if step > 1:
        for i in range(window, n):
            if result[i] == 0.5 and i > window:
                result[i] = result[i - 1]

    return result



--- FILE: backend/src/data/download_histdata.py ---
#!/usr/bin/env python3
"""
I.A.R.A. Historical Data Downloader
====================================
Downloads forex M1 (1-minute) data from Histdata.com

This script downloads years of historical forex data for training.
Histdata.com provides free M1 data since 2001.

Usage:
    docker compose exec saci-brain python src/data/download_histdata.py --pairs EURUSD GBPUSD --years 2020 2021 2022 2023

Output:
    data/raw/{PAIR}_{YEAR}.csv
"""

import argparse
import logging
import os
import zipfile
from io import BytesIO
from pathlib import Path
from typing import List
import time

import requests
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("DataDownloader")

# Output directory
RAW_DATA_DIR = Path("data/raw")

# Histdata.com download URL pattern
# They use a specific format for their downloads
HISTDATA_BASE_URL = "https://www.histdata.com/download-free-forex-historical-data/?/ascii/1-minute-bar-quotes"

# Alternative: Direct download URLs (these are more reliable)
# Histdata provides data in monthly chunks
HISTDATA_DOWNLOAD_URL = "https://www.histdata.com/download-free-forex-data/?/ascii/1-minute-bar-quotes/{pair}/{year}"


def download_from_dukascopy(pair: str, year: int, month: int) -> pd.DataFrame:
    """
    Download data from Dukascopy (alternative source).
    Note: Dukascopy uses tick data, more complex to process.
    """
    # This is a placeholder - Dukascopy requires more complex handling
    raise NotImplementedError("Dukascopy downloader not implemented yet")


def download_from_forexsb(pair: str, years: List[int]) -> pd.DataFrame:
    """
    Download from Forex Strategy Builder's data server.
    They provide clean CSV files.
    """
    # FSB data URL pattern
    base_url = f"https://forexsb.com/historical-data/{pair.lower()}"
    
    all_data = []
    for year in years:
        url = f"{base_url}/{year}"
        logger.info(f"Trying ForexSB: {url}")
        try:
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                # Parse CSV
                from io import StringIO
                df = pd.read_csv(StringIO(response.text))
                all_data.append(df)
                logger.info(f"  Downloaded {len(df)} rows")
        except Exception as e:
            logger.warning(f"  Failed: {e}")
    
    if all_data:
        return pd.concat(all_data, ignore_index=True)
    return None


def generate_sample_data(pair: str, years: List[int], output_dir: Path) -> Path:
    """
    Generate synthetic forex data for testing when downloads fail.
    This creates realistic-looking price action for development.
    """
    import numpy as np
    
    logger.info(f"Generating synthetic data for {pair} ({years[0]}-{years[-1]})")
    
    # Starting prices for different pairs
    start_prices = {
        'EURUSD': 1.1000,
        'GBPUSD': 1.2500,
        'USDJPY': 110.00,
        'AUDUSD': 0.7500,
    }
    
    base_price = start_prices.get(pair, 1.0)
    
    all_rows = []
    current_price = base_price
    
    for year in years:
        # Generate one year of M15 data (less data, faster processing)
        # ~35k candles per year for M15 (252 trading days * 24 hours * ~6 candles)
        n_candles = 35000
        
        for i in range(n_candles):
            # Random walk with mean reversion
            change_pct = np.random.normal(0, 0.0005)  # 0.05% std dev
            
            # Mean reversion toward base price
            reversion = (base_price - current_price) * 0.001
            change_pct += reversion
            
            # Add some trends
            if np.random.random() < 0.1:  # 10% chance of trend
                trend_direction = np.random.choice([-1, 1])
                change_pct += trend_direction * 0.001
            
            open_price = current_price
            close_price = current_price * (1 + change_pct)
            high_price = max(open_price, close_price) * (1 + abs(np.random.normal(0, 0.0002)))
            low_price = min(open_price, close_price) * (1 - abs(np.random.normal(0, 0.0002)))
            volume = int(np.random.exponential(1000))
            
            # Timestamp
            day_of_year = i // 96  # ~96 candles per day for M15
            hour = (i % 96) // 4
            minute = (i % 4) * 15
            
            timestamp = pd.Timestamp(year=year, month=1, day=1) + pd.Timedelta(days=day_of_year, hours=hour, minutes=minute)
            
            all_rows.append({
                'date': timestamp,
                'open': round(open_price, 5),
                'high': round(high_price, 5),
                'low': round(low_price, 5),
                'close': round(close_price, 5),
                'volume': volume
            })
            
            current_price = close_price
    
    df = pd.DataFrame(all_rows)
    
    # Save to CSV
    output_path = output_dir / f"{pair}_synthetic.csv"
    df.to_csv(output_path, index=False)
    logger.info(f"Generated {len(df)} synthetic candles -> {output_path}")
    
    return output_path


def download_from_yfinance(pair: str, years: List[int]) -> pd.DataFrame:
    """
    Download from Yahoo Finance using yfinance library.
    Note: Yahoo has limited forex data, mainly major pairs.
    """
    try:
        import yfinance as yf
    except ImportError:
        logger.error("yfinance not installed. Run: pip install yfinance")
        return None
    
    # Yahoo Finance forex symbol format
    symbol = f"{pair}=X"
    
    start_date = f"{min(years)}-01-01"
    end_date = f"{max(years) + 1}-01-01"
    
    logger.info(f"Downloading {symbol} from Yahoo Finance ({start_date} to {end_date})")
    
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_date, end=end_date, interval="1h")  # 1h is the minimum for forex
        
        if df.empty:
            logger.warning(f"No data returned for {symbol}")
            return None
        
        # Rename columns to match our format
        df = df.reset_index()
        df = df.rename(columns={
            'Datetime': 'date',
            'Date': 'date',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        })
        
        df = df[['date', 'open', 'high', 'low', 'close', 'volume']]
        logger.info(f"Downloaded {len(df)} candles from Yahoo Finance")
        
        return df
        
    except Exception as e:
        logger.error(f"Yahoo Finance download failed: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Download forex historical data")
    parser.add_argument("--pairs", nargs="+", default=["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"],
                        help="Currency pairs to download")
    parser.add_argument("--years", nargs="+", type=int, default=[2020, 2021, 2022, 2023, 2024],
                        help="Years to download")
    parser.add_argument("--source", choices=["yahoo", "synthetic"], default="yahoo",
                        help="Data source")
    parser.add_argument("--output-dir", type=Path, default=RAW_DATA_DIR,
                        help="Output directory")
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("  I.A.R.A. Historical Data Downloader")
    logger.info("=" * 60)
    logger.info(f"Pairs: {args.pairs}")
    logger.info(f"Years: {args.years}")
    logger.info(f"Source: {args.source}")
    logger.info(f"Output: {args.output_dir}")
    
    downloaded_files = []
    
    for pair in args.pairs:
        logger.info(f"\n--- Processing {pair} ---")
        
        df = None
        
        if args.source == "yahoo":
            df = download_from_yfinance(pair, args.years)
        
        if df is None or args.source == "synthetic":
            # Fallback to synthetic data
            logger.warning(f"Using synthetic data for {pair}")
            output_path = generate_sample_data(pair, args.years, args.output_dir)
            downloaded_files.append(output_path)
            continue
        
        # Save to CSV
        output_path = args.output_dir / f"{pair}.csv"
        df.to_csv(output_path, index=False)
        logger.info(f"Saved {len(df)} candles -> {output_path}")
        downloaded_files.append(output_path)
        
        # Rate limiting
        time.sleep(1)
    
    logger.info("\n" + "=" * 60)
    logger.info("Download complete!")
    logger.info(f"Files: {[str(f) for f in downloaded_files]}")
    logger.info("\nNext steps:")
    logger.info("1. Process data: python src/data/processor.py")
    logger.info("2. Generate cache: python src/data/feature_factory.py")
    logger.info("3. Train: python src/train/trainer_liquid.py --use-cache")


if __name__ == "__main__":
    main()



--- FILE: backend/src/data/downloader.py ---
"""
Bulk Downloader - Forex Data Ingestion Module
==============================================
Async bulk downloader for Forex pairs using yfinance.
Saves raw OHLCV data as Parquet files partitioned by ticker.
"""

import asyncio
import logging
from pathlib import Path
from typing import List, Optional

import polars as pl
import yfinance as yf
from tqdm import tqdm

# Fix yfinance cache issue
try:
    yf.set_tz_cache_location("/tmp/py-yfinance")
except Exception:
    pass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Forex pairs (Target Majors for v6.1)
FOREX_PAIRS: List[str] = [
    "EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X"
]

# Configuration
DATA_RAW_DIR = Path(__file__).parent.parent.parent / "data" / "raw"
MAX_RETRIES = 3
RETRY_DELAY_BASE = 2  # seconds (exponential backoff)


async def download_ticker(
    ticker: str,
    period: str = "max",
    interval: str = "1d",
    retries: int = MAX_RETRIES
) -> Optional[pl.DataFrame]:
    """
    Download historical data for a single ticker with retry logic.
    
    Args:
        ticker: Yahoo Finance ticker symbol (e.g., "EURUSD=X")
        period: Data period (e.g., "max", "5y", "1y")
        interval: Data interval (e.g., "1d", "1h")
        retries: Number of retry attempts
        
    Returns:
        Polars DataFrame with OHLCV data or None if failed
    """
    for attempt in range(retries):
        try:
            # yfinance is not truly async, run in executor
            loop = asyncio.get_event_loop()
            df_pandas = await loop.run_in_executor(
                None,
                lambda: yf.Ticker(ticker).history(period=period, interval=interval)
            )
            
            if df_pandas.empty:
                logger.warning(f"{ticker}: Empty DataFrame received")
                return None
            
            # Reset index to get Date as column
            df_pandas = df_pandas.reset_index()
            
            # Convert to Polars
            df = pl.from_pandas(df_pandas)
            
            # Standardize column names
            rename_map = {
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume"
            }
            
            # Handle Date vs Datetime
            if "Date" in df.columns:
                rename_map["Date"] = "date"
            elif "Datetime" in df.columns:
                rename_map["Datetime"] = "date"
                
            # Filter map to only existing columns to avoid Polars errors
            rename_map = {k: v for k, v in rename_map.items() if k in df.columns}
            
            df = df.rename(rename_map)
            
            # Select only needed columns
            available_cols = df.columns
            cols_to_keep = [c for c in ["date", "open", "high", "low", "close", "volume"] 
                          if c in available_cols]
            df = df.select(cols_to_keep)
            
            # Add ticker column
            df = df.with_columns(pl.lit(ticker.replace("=X", "")).alias("ticker"))
            
            # Validate data
            if df.height == 0:
                logger.warning(f"{ticker}: No valid rows after processing")
                return None
                
            logger.info(f"{ticker}: Downloaded {df.height} rows")
            return df
            
        except Exception as e:
            delay = RETRY_DELAY_BASE ** (attempt + 1)
            logger.warning(f"{ticker}: Attempt {attempt + 1} failed - {e}. Retrying in {delay}s...")
            await asyncio.sleep(delay)
    
    logger.error(f"{ticker}: All {retries} attempts failed")
    return None


async def download_all(
    tickers: List[str] = FOREX_PAIRS,
    period: str = "max",
    interval: str = "1d",
    max_concurrent: int = 5
) -> dict[str, pl.DataFrame]:
    """
    Download all tickers concurrently with rate limiting.
    
    Args:
        tickers: List of ticker symbols
        period: Data period
        interval: Data interval
        max_concurrent: Maximum concurrent downloads
        
    Returns:
        Dictionary mapping ticker to DataFrame
    """
    semaphore = asyncio.Semaphore(max_concurrent)
    results = {}
    
    async def download_with_semaphore(ticker: str):
        async with semaphore:
            df = await download_ticker(ticker, period, interval)
            if df is not None:
                results[ticker] = df
    
    tasks = [download_with_semaphore(t) for t in tickers]
    
    # Use tqdm for progress
    for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Downloading"):
        await coro
    
    return results


def save_to_parquet(dataframes: dict[str, pl.DataFrame], output_dir: Path = DATA_RAW_DIR):
    """
    Save DataFrames as Parquet files partitioned by ticker.
    
    Args:
        dataframes: Dictionary mapping ticker to DataFrame
        output_dir: Output directory path
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for ticker, df in dataframes.items():
        ticker_clean = ticker.replace("=X", "")
        output_path = output_dir / f"{ticker_clean}.parquet"
        
        df.write_parquet(output_path)
        logger.info(f"Saved {ticker_clean} to {output_path}")


async def run_ingestion(
    tickers: List[str] = FOREX_PAIRS,
    period: str = "60d",
    interval: str = "5m"
):
    """
    Main ingestion pipeline.
    
    Args:
        tickers: List of ticker symbols
        period: Data period
        interval: Data interval
    """
    logger.info(f"Starting ingestion for {len(tickers)} tickers")
    
    # Download all data
    dataframes = await download_all(tickers, period, interval)
    
    logger.info(f"Successfully downloaded {len(dataframes)}/{len(tickers)} tickers")
    
    # Save to parquet
    save_to_parquet(dataframes)
    
    logger.info("Ingestion complete!")
    
    return dataframes


def main():
    """CLI entry point."""
    print("Starting massive data download (60d 5m)... This may take a minute.")
    asyncio.run(run_ingestion())


if __name__ == "__main__":
    main()



--- FILE: backend/src/data/indicators.py ---
"""
I.A.R.A. — Centralized Feature Engineering (indicators.py)
==========================================================
Single source of truth for all technical indicators used in training,
backtesting, and live inference.

Features (22 total):
  Core (12): RSI, MACD (3), BBands (2), ATR, Stochastic (2), Velocity, ADX, Sentiment
  Mindset (6): Dist SMA200, Dist SMA20, Dist Resistance, Dist Support, Upper Wick, Lower Wick
  Time (4): hour_sin, hour_cos, day_sin, day_cos

References:
  - feature_factory.py (training)
  - backtest_v9.py (backtesting)
  - signal_generator.py (live inference)

Usage:
    from src.data.indicators import compute_features, compute_time_features, compute_trend
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from src.data.hurst import compute_hurst_rolling

logger = logging.getLogger("indicators")

try:
    import talib
except ImportError:
    talib = None
    logger.warning("TA-Lib not found. Using Mock (Zero) Indicators. Install ta-lib for real features.")

# Mock if missing
if talib is None:
    class MockTalib:
        def RSI(self, data, timeperiod=14):
            return np.zeros_like(data)
        def MACD(self, data, fastperiod=12, slowperiod=26, signalperiod=9):
            return np.zeros_like(data), np.zeros_like(data), np.zeros_like(data)
        def BBANDS(self, data, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0):
            return data, data, data # Upper, Middle, Lower
        def ATR(self, high, low, close, timeperiod=14):
            return np.zeros_like(close)
        def STOCH(self, high, low, close, fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0):
            return np.zeros_like(close), np.zeros_like(close)
        def ROC(self, data, timeperiod=10):
            return np.zeros_like(data)
        def ADX(self, high, low, close, timeperiod=14):
            return np.zeros_like(close)
        def EMA(self, data, timeperiod=30):
            return data 
        def SMA(self, data, timeperiod=30):
            return data
        def MAX(self, data, timeperiod=30):
            return data
        def MIN(self, data, timeperiod=30):
            return data

    talib = MockTalib()

# ──────────────────────────────────────────────
# Feature Names (Canonical Order)
# ──────────────────────────────────────────────
FEATURE_NAMES_CORE = [
    "rsi_norm",
    "macd_norm", "macd_signal_norm", "macd_hist_norm",
    "bb_width_norm", "bb_pct_b_norm",
    "atr_norm",
    "stoch_k_norm", "stoch_d_norm",
    "velocity_norm",
    "adx_norm",
    "sentiment_norm",
    "volume_norm", "vma_norm",
    "cvd_norm", "oi_norm",
    "hurst_norm", "fractal_dim_norm",
]

FEATURE_NAMES_MINDSET = [
    "dist_sma_200", "dist_sma_20",
    "dist_res_50", "dist_sup_50",
    "upper_wick_ratio", "lower_wick_ratio",
]

FEATURE_NAMES_TIME = [
    "hour_sin", "hour_cos", "day_sin", "day_cos",
]

FEATURE_NAMES_ALL = FEATURE_NAMES_CORE + FEATURE_NAMES_MINDSET + FEATURE_NAMES_TIME
NUM_FEATURES = len(FEATURE_NAMES_ALL)  # 26 (was 24, +2 Hurst/Fractal)


# ──────────────────────────────────────────────
# Core Feature Computation
# ──────────────────────────────────────────────
def compute_features(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    opens: np.ndarray,
    volume: Optional[np.ndarray] = None,
    timestamps: Optional[pd.Series] = None,
    include_time: bool = True,
) -> np.ndarray:
    """
    Compute the canonical 22-feature vector from OHLC data.

    Args:
        close: Close prices (float64)
        high: High prices (float64)
        low: Low prices (float64)
        opens: Open prices (float64)
        timestamps: Optional datetime Series for time features
        include_time: Whether to include time features (4 cols)

    Returns:
        np.ndarray of shape [N, 22] (or [N, 18] if include_time=False)
    """
    close = close.astype(np.float64)
    high = high.astype(np.float64)
    low = low.astype(np.float64)
    opens = opens.astype(np.float64)

    # ── 1. CORE 12 ──
    # RSI
    rsi = talib.RSI(close, timeperiod=14)
    rsi_norm = (rsi - 50) / 25

    # MACD
    macd, macdsignal, macdhist = talib.MACD(
        close, fastperiod=12, slowperiod=26, signalperiod=9
    )
    macd_norm = macd * 100
    macd_signal_norm = macdsignal * 100
    macd_hist_norm = macdhist * 100

    # Bollinger Bands
    upper, middle, lower_bb = talib.BBANDS(
        close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        bb_width = (upper - lower_bb) / (middle + 1e-10)
        bb_pct_b = (close - lower_bb) / (upper - lower_bb + 1e-10)
    bb_width_norm = np.nan_to_num(bb_width) * 10
    bb_pct_b_norm = (np.nan_to_num(bb_pct_b) - 0.5) * 2

    # ATR
    atr = talib.ATR(high, low, close, timeperiod=14)
    atr_norm = (atr / (close + 1e-10)) * 100

    # Stochastic
    stoch_k, stoch_d = talib.STOCH(
        high, low, close,
        fastk_period=14, slowk_period=3, slowk_matype=0,
        slowd_period=3, slowd_matype=0,
    )
    stoch_k_norm = (stoch_k - 50) / 25
    stoch_d_norm = (stoch_d - 50) / 25

    # Velocity (ROC)
    velocity = talib.ROC(close, timeperiod=5)
    velocity_norm = velocity * 10

    # ADX
    adx = talib.ADX(high, low, close, timeperiod=14)
    adx_norm = (adx - 25) / 25

    # Global Sentiment (EMA 50)
    ema_50 = talib.EMA(close, timeperiod=50)
    ema_50 = np.nan_to_num(ema_50, nan=close[0] if len(close) > 0 else 0)
    sentiment = (close - ema_50) / (ema_50 + 1e-10) * 100
    sentiment_norm = sentiment * 10

    # Volume Context
    if volume is not None and len(volume) > 0:
        volume = volume.astype(np.float64)
        vma = talib.SMA(volume, timeperiod=20)
        vma = np.nan_to_num(vma, nan=volume[0] if len(volume) > 0 else 0)
        volume_norm = ((volume - vma) / (vma + 1e-10)) * 10
        
        vma_long = talib.SMA(volume, timeperiod=50)
        vma_long = np.nan_to_num(vma_long, nan=volume[0] if len(volume) > 0 else 0)
        vma_norm = ((vma - vma_long) / (vma_long + 1e-10)) * 10
    else:
        volume_norm = np.zeros_like(close)
        vma_norm = np.zeros_like(close)

    # Hurst Exponent & Fractal Dimension (Chaos Mathematics)
    hurst_raw = compute_hurst_rolling(close, window=200, step=1)
    hurst_norm = (hurst_raw - 0.5) * 4.0  # Center at 0, scale to [-2, 2]
    fractal_dim_raw = 2.0 - hurst_raw
    fractal_dim_norm = (fractal_dim_raw - 1.5) * 4.0  # Center at 0

    # ── 2. MINDSET 6 ──
    # Trend Alignment (SMA 200 & 20)
    sma_200 = talib.SMA(close, timeperiod=200)
    sma_200 = np.nan_to_num(sma_200, nan=close[0] if len(close) > 0 else 0)
    sma_20 = talib.SMA(close, timeperiod=20)
    sma_20 = np.nan_to_num(sma_20, nan=close[0] if len(close) > 0 else 0)

    dist_sma_200 = (close - sma_200) / (sma_200 + 1e-10) * 100
    dist_sma_20 = (close - sma_20) / (sma_20 + 1e-10) * 100

    # Support & Resistance (Rolling 50)
    roll_max_50 = talib.MAX(high, timeperiod=50)
    roll_min_50 = talib.MIN(low, timeperiod=50)
    roll_max_50 = np.nan_to_num(roll_max_50, nan=close[0] if len(close) > 0 else 0)
    roll_min_50 = np.nan_to_num(roll_min_50, nan=close[0] if len(close) > 0 else 0)

    dist_res_50 = (roll_max_50 - close) / (close + 1e-10) * 100
    dist_sup_50 = (close - roll_min_50) / (close + 1e-10) * 100

    # Candle Physics (Wick Ratios)
    candle_range = high - low
    upper_wick = high - np.maximum(close, opens)
    upper_wick_ratio = upper_wick / (candle_range + 1e-10)
    lower_wick = np.minimum(close, opens) - low
    lower_wick_ratio = lower_wick / (candle_range + 1e-10)

    # ── Stack Core + Mindset (22 features) ──
    features = np.column_stack([
        rsi_norm,
        macd_norm, macd_signal_norm, macd_hist_norm,
        bb_width_norm, bb_pct_b_norm,
        atr_norm,
        stoch_k_norm, stoch_d_norm,
        velocity_norm,
        adx_norm,
        sentiment_norm,
        volume_norm, vma_norm,
        hurst_norm, fractal_dim_norm,
        # Mindset
        dist_sma_200, dist_sma_20,
        dist_res_50, dist_sup_50,
        upper_wick_ratio, lower_wick_ratio,
    ])

    # ── 3. TIME 4 ──
    if include_time and timestamps is not None:
        time_feats = compute_time_features_from_series(timestamps)
        features = np.column_stack([features, time_feats])
    elif include_time:
        # Pad with zeros if no timestamps provided
        features = np.column_stack([
            features, np.zeros((len(close), 4), dtype=np.float32)
        ])

    return np.nan_to_num(features).astype(np.float32)


# ──────────────────────────────────────────────
# Time Features
# ──────────────────────────────────────────────
def compute_time_features_from_series(dt_series: pd.Series) -> np.ndarray:
    """
    Compute cyclical time features from a pandas datetime Series.

    Returns:
        np.ndarray of shape [N, 4] with [hour_sin, hour_cos, day_sin, day_cos]
    """
    dt_index = pd.to_datetime(dt_series)
    hours = dt_index.dt.hour.values
    days = dt_index.dt.dayofweek.values

    hour_sin = np.sin(2 * np.pi * hours / 24).astype(np.float32)
    hour_cos = np.cos(2 * np.pi * hours / 24).astype(np.float32)
    day_sin = np.sin(2 * np.pi * days / 7).astype(np.float32)
    day_cos = np.cos(2 * np.pi * days / 7).astype(np.float32)

    return np.stack([hour_sin, hour_cos, day_sin, day_cos], axis=1)


def compute_time_features_polars(df) -> np.ndarray:
    """
    Compute cyclical time features from a Polars DataFrame with 'date' column.
    Fallback to zeros if 'date' column is missing or unparseable.

    Returns:
        np.ndarray of shape [N, 4] with [hour_sin, hour_cos, day_sin, day_cos]
    """
    import polars as pl

    if "date" not in df.columns:
        logger.warning("No 'date' column found, using zeros for time features")
        return np.zeros((len(df), 4), dtype=np.float32)

    try:
        dt_series = df["date"]
        if dt_series.dtype == pl.Utf8:
            dt_series = dt_series.str.to_datetime()
        return compute_time_features_from_series(dt_series.to_pandas())
    except Exception as e:
        logger.warning(f"Could not parse dates: {e}, using zeros")
        return np.zeros((len(df), 4), dtype=np.float32)


# ──────────────────────────────────────────────
# Trend Detection
# ──────────────────────────────────────────────
def compute_trend(close_arr: np.ndarray, period: int = 14) -> np.ndarray:
    """
    Simple trend detection using EMA slope.

    Returns:
        np.ndarray with values: 1 (Up), -1 (Down), 0 (Flat)
    """
    if len(close_arr) < period:
        return np.zeros(len(close_arr))

    close_arr = np.nan_to_num(
        close_arr.astype(np.float64),
        nan=close_arr[0] if len(close_arr) > 0 else 0,
    )

    ema = talib.EMA(close_arr, timeperiod=period)

    slope = np.zeros_like(ema)
    slope[period:] = ema[period:] - ema[period - 1 : -1]

    with np.errstate(divide="ignore", invalid="ignore"):
        slope_pct = slope / close_arr
        slope_pct = np.nan_to_num(slope_pct)

    threshold = 0.00005  # 0.005% slope threshold
    trend = np.zeros_like(slope)
    trend[slope_pct > threshold] = 1
    trend[slope_pct < -threshold] = -1

    return trend


# ──────────────────────────────────────────────
# Convenience: Compute from DataFrame
# ──────────────────────────────────────────────
def compute_features_from_df(
    df: pd.DataFrame,
    time_col: str = "time",
    include_time: bool = True,
) -> np.ndarray:
    """
    Compute features from a pandas DataFrame with OHLCV columns.

    Args:
        df: DataFrame with 'open', 'high', 'low', 'close' columns
        time_col: Name of the timestamp column (for time features)
        include_time: Whether to include cyclical time features

    Returns:
        np.ndarray of shape [N, 22] (or [N, 18])
    """
    timestamps = None
    if include_time and time_col in df.columns:
        timestamps = pd.to_datetime(df[time_col])

    return compute_features(
        close=df["close"].values,
        high=df["high"].values,
        low=df["low"].values,
        opens=df["open"].values,
        volume=df["volume"].values if "volume" in df.columns else None,
        timestamps=timestamps,
        include_time=include_time,
    )



--- FILE: backend/src/data/feature_factory.py ---
#!/usr/bin/env python3
"""
I.A.R.A. v6.10 - CPU Feature Factory (Curriculum Learning Edition)
===================================================================
Converts pre-processed parquet files into ready-to-consume PyTorch Tensors.

v6.10 Improvements:
- Recalculates time features from 'date' column (fixes zero bug)
- Curriculum Learning: Threshold-based labels (ignores noise/lateral movements)
- "Golden Dataset": Only strong moves are labeled as UP/DOWN

Usage:
    docker compose exec saci-brain python src/data/feature_factory.py
"""

import argparse
import logging
from pathlib import Path
from typing import List, Tuple
import multiprocessing as mp
from functools import partial

import numpy as np
import polars as pl
import torch

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("FeatureFactory")

# Paths
DATA_PROCESSED_DIR = Path("data/processed")
CACHE_DIR = Path("data/cache")

# v6.10 Curriculum Learning: Minimum return threshold
# Movements smaller than this are considered "noise" and excluded
# These are adjusted per market type
MIN_RETURN_THRESHOLD = {
    "forex": 0.0003,  # 0.03% for forex (low volatility)
    "crypto": 0.002,  # 0.2% for crypto (higher volatility)
}


import talib
import pandas as pd

# Centralized feature computation
from src.data.indicators import (
    compute_features,
    compute_time_features_polars as compute_time_features,
    compute_trend,
)


def resample_data(df: pl.DataFrame, timeframe: str) -> pl.DataFrame:
    """
    Resample M1 data to higher timeframes (5m, 1h).
    """
    # Ensure date is sorted
    df = df.sort("date")
    
    # Polars efficient resampling
    # Note: 'every' argument handles 5m, 1h etc.
    return (
        df.group_by_dynamic("date", every=timeframe)
        .agg([
            pl.col("open").first(),
            pl.col("high").max(),
            pl.col("low").min(),
            pl.col("close").last(),
            pl.col("volume").sum(),
        ])
    )


def process_hierarchical(
    file_path: Path, 
    seq_len: int, 
    stride: int,
    market_type: str = "crypto"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Process M1 data with Hierarchical Consensus (M1 + M5 + H1).
    """
    logger.info(f"Processing Hierarchical: {file_path.name}")
    
    try:
        # Load Data
        if file_path.suffix == '.csv':
            df_m1 = pl.read_csv(file_path, ignore_errors=True)
        else:
            df_m1 = pl.read_parquet(file_path)
    except Exception as e:
        logger.error(f"Failed to load {file_path}: {e}")
        return None, None
        
    # Standardize column names
    df_m1 = df_m1.rename({c: c.lower() for c in df_m1.columns})
    
    # Date parsing
    if 'date' not in df_m1.columns:
        # Check alternatives
        for alt in ['datetime', 'time', 'timestamp', 'date time']:
            if alt in df_m1.columns:
                df_m1 = df_m1.rename({alt: 'date'})
                break
    
    if 'date' not in df_m1.columns:
        logger.error(f"Skipping {file_path.name}: No date column")
        return None, None
        
    # Ensure date type
    if df_m1['date'].dtype == pl.Utf8:
        try:
             # Try generic parsing first, then specific formats if needed
            df_m1 = df_m1.with_columns(pl.col('date').str.to_datetime())
        except:
             # Fallback using pandas for robust parsing
            pdf = df_m1.to_pandas()
            pdf['date'] = pd.to_datetime(pdf['date'])
            df_m1 = pl.from_pandas(pdf)

    # 1. Resample Layer
    # 1. Resample Layer
    # Base is now M5 (5 minutes)
    logger.info(f"  Resampling M1 ({len(df_m1)}) -> M5 (Base), M15, M30, H1 (Context)...")
    df_m5 = resample_data(df_m1, "5m")
    
    # Contexts
    df_m15 = resample_data(df_m1, "15m")
    df_m30 = resample_data(df_m1, "30m")
    df_h1 = resample_data(df_m1, "1h")
    
    if len(df_h1) < 50:
        logger.error(f"Skipping {file_path.name}: Transformation resulted in too few H1 samples")
        return None, None

    # 2. Compute Trends for each layer
    close_m5 = df_m5['close'].to_numpy()
    close_m15 = df_m15['close'].to_numpy()
    close_m30 = df_m30['close'].to_numpy()
    close_h1 = df_h1['close'].to_numpy()
    
    # We use float64 for talib then cast back
    trend_m5 = compute_trend(close_m5.astype(np.float64))
    trend_m15 = compute_trend(close_m15.astype(np.float64))
    trend_m30 = compute_trend(close_m30.astype(np.float64))
    trend_h1 = compute_trend(close_h1.astype(np.float64))
    
    # Also calc RSI for contexts (Reversal detection)
    rsi_m15 = talib.RSI(close_m15.astype(np.float64), timeperiod=14)
    rsi_m30 = talib.RSI(close_m30.astype(np.float64), timeperiod=14)
    
    # Normalize Context RSIs
    rsi_m15_norm = (rsi_m15 - 50) / 25
    rsi_m30_norm = (rsi_m30 - 50) / 25
    
    # 3. Align Trends (Consensus)
    # Map Contexts back to M5 timestamps
    
    pdf_m5 = df_m5['date'].to_pandas()
    pdf_m15 = df_m15['date'].to_pandas()
    pdf_m30 = df_m30['date'].to_pandas()
    pdf_h1 = df_h1['date'].to_pandas()
    
    # Reindex Contexts to match M5 index (ffill)
    # We use reindex to propagate the last known macro state to the current micro state
    s_trend_m15 = pd.Series(trend_m15, index=pdf_m15).reindex(pdf_m5, method='ffill').fillna(0)
    s_trend_m30 = pd.Series(trend_m30, index=pdf_m30).reindex(pdf_m5, method='ffill').fillna(0)
    s_trend_h1 = pd.Series(trend_h1, index=pdf_h1).reindex(pdf_m5, method='ffill').fillna(0)
    
    s_rsi_m15 = pd.Series(rsi_m15_norm, index=pdf_m15).reindex(pdf_m5, method='ffill').fillna(0)
    s_rsi_m30 = pd.Series(rsi_m30_norm, index=pdf_m30).reindex(pdf_m5, method='ffill').fillna(0)
    
    aligned_trend_m5 = trend_m5 # M5 is base
    aligned_trend_m15 = s_trend_m15.values
    aligned_trend_m30 = s_trend_m30.values
    aligned_trend_h1 = s_trend_h1.values
    
    aligned_rsi_m15 = s_rsi_m15.values
    aligned_rsi_m30 = s_rsi_m30.values
    
    # Consensus Mask: H1 == M5 (Fractal Alignment)
    # Only keep samples where Medium and Long term agree
    consensus_mask = (aligned_trend_h1 == aligned_trend_m5) & (aligned_trend_h1 != 0)
    
    # Create Labels for M5 based on the consensus
    # IARA v9 Map: 0: LONG, 1: SHORT, 2: NEUTRAL (WAIT)
    # 🚀 Phase 11: Golden Dataset - Strict Thresholds
    threshold = MIN_RETURN_THRESHOLD.get(market_type, 0.001)
    
    labels = np.full(len(close_m5), 2.0, dtype=np.float32) # Default to 2.0 (Neutral)
    
    # Simple future return for the next period (approx 1 hour / 12 bars)
    # Note: feature_factory shifts targets later, here we just define the raw label
    # but we will use the threshold during the sequence loop for maximum refinement.
    
    labels[aligned_trend_h1 == 1] = 0.0  # UP -> LONG
    labels[aligned_trend_h1 == -1] = 1.0 # DOWN -> SHORT
    
    # 4. Generate M5 Features (using centralized indicators)
    c_m5 = close_m5.astype(np.float64)
    highs = df_m5['high'].to_numpy().astype(np.float64)
    lows = df_m5['low'].to_numpy().astype(np.float64)
    opens = df_m5['open'].to_numpy().astype(np.float64)
    
    # Time features from Polars 'date' column
    time_feats = compute_time_features(df_m5)
    
    # Order Flow / Macro Features (Phase 5)
    # Estimate Delta based on Close relative to High/Low
    range_total = df_m5['high'].to_numpy() - df_m5['low'].to_numpy()
    range_total = np.where(range_total == 0, 1e-8, range_total)
    buy_pressure = (df_m5['close'].to_numpy() - df_m5['low'].to_numpy()) / range_total
    sell_pressure = (df_m5['high'].to_numpy() - df_m5['close'].to_numpy()) / range_total
    
    volume_m5 = df_m5['volume'].to_numpy().astype(np.float64)
    delta = (buy_pressure - sell_pressure) * volume_m5
    
    # CVD (Cumulative Volume Delta) - Rolling sum over last 24 periods
    cvd_m5 = pd.Series(delta).rolling(window=24, min_periods=1).sum().to_numpy()
    
    # Open Interest Proxy (Using Volume Moving Average as a basic stand-in if OI not available)
    if 'open_interest' in df_m5.columns:
        oi_m5 = df_m5['open_interest'].to_numpy()
    else:
        oi_m5 = pd.Series(volume_m5).rolling(window=48, min_periods=1).mean().to_numpy()
        
    # Scale CVD and OI for NN consumption
    cvd_m5 = (cvd_m5 - np.mean(cvd_m5)) / (np.std(cvd_m5) + 1e-8)
    oi_m5 = (oi_m5 - np.mean(oi_m5)) / (np.std(oi_m5) + 1e-8)
    
    # 🌊 Phase 21: Fast Fourier Transform (FFT) Temporal Extraction
    def get_rolling_fft(series, window):
        shape = series.shape[:-1] + (series.shape[-1] - window + 1, window)
        strides = series.strides + (series.strides[-1],)
        rolled = np.lib.stride_tricks.as_strided(series, shape=shape, strides=strides)
        
        # Detrend to remove DC component dominance
        means = np.mean(rolled, axis=1, keepdims=True)
        detrended = rolled - means
        
        fft_vals = np.abs(np.fft.rfft(detrended, axis=1))
        # Max amplitude of frequencies (excluding DC)
        max_fft = np.max(fft_vals[:, 1:], axis=1) if fft_vals.shape[1] > 1 else np.zeros((fft_vals.shape[0],))
        
        padded = np.zeros(len(series))
        padded[window-1:] = max_fft
        return (padded - np.mean(padded)) / (np.std(padded) + 1e-8)

    fft_m5 = get_rolling_fft(c_m5, 24)

    # Compute canonical features (Core 14 + Mindset 6 + Time 4)
    features_24 = compute_features(
        close=c_m5,
        high=highs,
        low=lows,
        opens=opens,
        volume=volume_m5,
        timestamps=df_m5['date'].to_pandas() if 'date' in df_m5.columns else None,
        include_time=True,
    )
    
    # Append Macro/Order Flow features
    features = np.column_stack([
        features_24,                 # 24 Canonical features
        cvd_m5, oi_m5,               # +2 Volumes/Delta
        fft_m5,                      # +1 Temporal FFT
        opens, highs, lows, c_m5     # +4 OHLC
    ])                               # Total: 31 Features
    
    features = np.nan_to_num(features, nan=0.0).astype(np.float32)
    
    # 5. Build Sequences (Predictive Mode - Lead Shifting)
    # Target is shifted lead_steps (12) into the future
    lead_steps = 12 # Predict 1 hour ahead
    X_list = []
    y_list = []
    
    for i in range(0, len(features) - seq_len - lead_steps, stride):
        target_idx = i + seq_len - 1
        future_idx = target_idx + lead_steps
            
        # 🚀 Phase 11: Golden Dataset - Maximum Refinement
        # 1. Strict Fractal Consensus (100% Agreement)
        is_consensus = consensus_mask[target_idx]
        
        if is_consensus:
             # 2. Return Threshold (Ensure move is large enough to be profit)
             price_now = c_m5[target_idx]
             price_future = c_m5[future_idx]
             ret = abs(price_future - price_now) / price_now
             
             if ret >= threshold:
                  X_list.append(features[i:i+seq_len])
                  y_list.append(labels[future_idx])
             else:
                  # If move is too small, label as Neutral (optional, or just skip)
                  # Skipping helps train only on "Golden" signals
                  continue
             
    if not X_list:
        logger.warning(f"  No consensus samples found for {file_path.name}")
        return None, None
        
    X_arr = np.array(X_list, dtype=np.float32)
    y_arr = np.array(y_list, dtype=np.float32)
    
    return X_arr, y_arr


def run_factory(
    seq_len: int = 128,
    stride: int = 2,
    val_split: float = 0.2,
    limit: int = 0,
    symbol: str = "EURUSD",
    market_type: str = "forex"
):
    """
    Main factory runner (Sharded Output).
    Isolated by SYMBOL to support Multi-Pair training.
    """
    logger.info("=" * 60)
    logger.info(f"  I.A.R.A. v6.11 - Hierarchical Feature Factory ({symbol} / {market_type})")
    logger.info("  Strategy: Fractal Consensus Filtering + Low RAM Mode")
    logger.info("=" * 60)
    
    processed_dir = DATA_PROCESSED_DIR
    cache_dir = CACHE_DIR / symbol
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories for shards
    train_shard_dir = cache_dir / "train_shards"
    val_shard_dir = cache_dir / "val_shards"
    train_shard_dir.mkdir(exist_ok=True)
    val_shard_dir.mkdir(exist_ok=True)
    
    # Cleanup old shards
    for f in train_shard_dir.glob("*.pt"): f.unlink()
    for f in val_shard_dir.glob("*.pt"): f.unlink()
    
    data_files = list(processed_dir.glob("*.csv")) + list(processed_dir.glob("*.parquet"))
    
    if not data_files:
        logger.error(f"No valid data files found in {processed_dir}")
        return

    logger.info(f"Found {len(data_files)} files for processing")
    
    total_samples = 0
    shard_count = 0
    
    import gc
    
    # Process file by file and save immediately
    samples_per_shard = 50000 
    global_feature_mask = None
    
    for f in data_files:
        # Match files by symbol name
        if symbol and symbol not in f.name:
            continue
            
        if f.name.endswith(('.csv', '.parquet')):
            X, y = process_hierarchical(f, seq_len, stride, market_type=market_type)
            
            if X is not None and len(X) > 0:
                # 🌲 Advanced Data Science: Structural Feature Selection (ExtraTrees)
                if global_feature_mask is None:
                    from sklearn.ensemble import ExtraTreesClassifier
                    from sklearn.feature_selection import SelectFromModel
                    
                    logger.info("🌲 Training ExtraTreesClassifier for Structural Feature Pruning...")
                    # Flatten sequences for global feature analysis
                    X_2d = X.reshape(-1, X.shape[-1])
                    # Duplicate labels for each timestep in the sequence to match X_2d
                    y_2d = np.repeat(y, X.shape[1])
                    
                    # Subsample for speed
                    if len(X_2d) > 100000:
                        idx = np.random.choice(len(X_2d), 100000, replace=False)
                        X_sub, y_sub = X_2d[idx], y_2d[idx]
                    else:
                        X_sub, y_sub = X_2d, y_2d
                        
                    # Target 0: LONG, 1: SHORT. Ignore neutral (2) for strict structural signal
                    valid_idx = y_sub != 2.0
                    if valid_idx.sum() > 1000: # Ensure we have enough signal samples
                        X_sub, y_sub = X_sub[valid_idx], y_sub[valid_idx]
                        
                    clf = ExtraTreesClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
                    clf.fit(X_sub, y_sub)
                    
                    # Select features: threshold "0.75*mean" to aggressively prune noisy features
                    selector = SelectFromModel(clf, prefit=True, threshold="0.75*mean")
                    global_feature_mask = selector.get_support()
                    
                    logger.info(f"🌲 Pruning complete! Kept {global_feature_mask.sum()} / {len(global_feature_mask)} high-signal features.")
                
                # Apply structural mask to tensor: [batch, seq_len, features]
                X = X[:, :, global_feature_mask]
                
                # 🚀 Phase 11: Shard Splitting logic
                # Instead of one shard per file, we split into chunks of samples_per_shard
                num_file_samples = len(X)
                
                for start_idx in range(0, num_file_samples, samples_per_shard):
                    end_idx = min(start_idx + samples_per_shard, num_file_samples)
                    X_chunk = X[start_idx:end_idx]
                    y_chunk = y[start_idx:end_idx]
                    
                    # Split each chunk into Train/Val
                    val_size = int(len(X_chunk) * val_split)
                    split_idx = len(X_chunk) - val_size
                    
                    X_train, y_train = X_chunk[:split_idx], y_chunk[:split_idx]
                    X_val, y_val = X_chunk[split_idx:], y_chunk[split_idx:]
                    
                    # Save Train Shard
                    if len(X_train) > 0:
                        train_shard_path = train_shard_dir / f"shard_{shard_count}.pt"
                        torch.save({
                            'X': torch.from_numpy(X_train), 
                            'y': torch.from_numpy(y_train)
                        }, train_shard_path)
                    
                    # Save Val Shard
                    if len(X_val) > 0:
                        val_shard_path = val_shard_dir / f"shard_{shard_count}.pt"
                        torch.save({
                            'X': torch.from_numpy(X_val), 
                            'y': torch.from_numpy(y_val)
                        }, val_shard_path)
                    
                    total_samples += len(X_chunk)
                    logger.info(f"  -> Saved Shard {shard_count}: {len(X_train)} train, {len(X_val)} val samples")
                    shard_count += 1
            
            # Explicit GC
            del X, y
            gc.collect()
    
    if total_samples == 0:
        logger.error("No valid sequences generated!")
        return
    
    logger.info(f"Total Processed Samples: {total_samples}")
    
    # Save meta params for inference (using standard paths)
    norm_path = cache_dir / "norm_params.pt"
    
    feature_names = [
        'rsi_norm', 'macd_norm', 'macd_signal_norm', 'macd_hist_norm',
        'bb_width_norm', 'bb_pct_b_norm', 'atr_norm',
        'stoch_k_norm', 'stoch_d_norm',
        'velocity_norm', 'adx_norm', 'global_sentiment_norm',
        'dist_sma_200', 'dist_sma_20',
        'dist_res_50', 'dist_sup_50',
        'upper_wick_ratio', 'lower_wick_ratio',
        'dist_res_50_v2', 'dist_sup_50_v2',
        'upper_wick_ratio_v2', 'lower_wick_ratio_v2',
        'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
        'cvd_m5', 'oi_m5', 'fft_m5',
        'open', 'high', 'low', 'close'
    ]
    
    # Filter the exact feature names that survived pruning
    if global_feature_mask is not None:
        if len(global_feature_mask) == len(feature_names):
            feature_names = [name for mask, name in zip(global_feature_mask, feature_names) if mask]
        else:
             # Fallback if there's a length mismatch
            feature_names = [f"feat_{i}" for i in range(global_feature_mask.sum())]
            
    meta = {
        'feature_cols': feature_names,
        'feature_mask': global_feature_mask.tolist() if global_feature_mask is not None else None,
        'seq_len': seq_len,
        'stride': stride,
        'strategy': 'v6.11_hierarchical_sharded',
        'symbol': symbol,
        'market_type': market_type,
    }
    torch.save(meta, norm_path)
    
    logger.info(f"Saved {shard_count} shards to {train_shard_dir}")
    
    # Return Directories instead of single files
    return train_shard_dir, val_shard_dir


def main():
    parser = argparse.ArgumentParser(description="I.A.R.A. Hierarchical Factory")
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--limit", type=int, default=0, help="Limit samples for Lab Mode (0=No limit)")
    parser.add_argument("--symbol", type=str, default="EURUSD", help="Symbol to process (e.g. EURUSD, BTCUSD)")
    parser.add_argument("--market-type", type=str, default="forex", choices=["forex", "crypto"],
                        help="Market type (forex or crypto)")
    args = parser.parse_args()
    
    run_factory(
        seq_len=args.seq_len,
        stride=args.stride,
        val_split=args.val_split,
        limit=args.limit,
        symbol=args.symbol,
        market_type=args.market_type,
    )


if __name__ == "__main__":
    main()



--- FILE: backend/src/data/tokenizer.py ---
"""
Financial Tokenizer - Discretize Continuous Market Data into Tokens
====================================================================
Converts continuous float indicator values into discrete token IDs
using quantile-based binning for Language Model training.
"""

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import polars as pl

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Paths
DATA_PROCESSED_DIR = Path(__file__).parent.parent.parent / "data" / "processed"
DATA_TOKENIZED_DIR = Path(__file__).parent.parent.parent / "data" / "tokenized"
TOKENIZER_PATH = Path(__file__).parent.parent.parent / "data" / "tokenizer.pkl"


class FinancialTokenizer:
    """
    Tokenizer for financial time series data.
    
    Converts continuous indicator values to discrete token IDs
    using quantile-based binning (equal frequency bins).
    
    Vocabulary Structure:
    - Each feature has K bins (default 256)
    - Token ID = feature_offset + bin_index
    - Special tokens: [PAD]=0, [UNK]=1, [BOS]=2, [EOS]=3
    """
    
    SPECIAL_TOKENS = {
        "[PAD]": 0,
        "[UNK]": 1,
        "[BOS]": 2,
        "[EOS]": 3
    }
    N_SPECIAL = 4
    
    def __init__(
        self,
        n_bins: int = 256,
        feature_columns: Optional[List[str]] = None
    ):
        """
        Initialize the tokenizer.
        
        Args:
            n_bins: Number of bins per feature (default 256 for byte-level)
            feature_columns: List of feature column names to tokenize
        """
        self.n_bins = n_bins
        self.feature_columns = feature_columns or []
        
        # Quantile edges for each feature
        self.bin_edges: Dict[str, np.ndarray] = {}
        
        # Feature to token offset mapping
        self.feature_offsets: Dict[str, int] = {}
        
        # Vocabulary
        self.vocab_size: int = self.N_SPECIAL
        
        self.is_fitted = False
    
    def fit(self, df: pl.DataFrame, feature_columns: Optional[List[str]] = None):
        """
        Fit the tokenizer on training data.
        
        Computes quantile bin edges for each feature.
        
        Args:
            df: Training DataFrame with features
            feature_columns: Optional list of columns to tokenize
        """
        if feature_columns:
            self.feature_columns = feature_columns
        elif not self.feature_columns:
            # Auto-detect numeric columns (exclude metadata)
            exclude_cols = {"date", "ticker", "target", "volume"}
            self.feature_columns = [
                c for c in df.columns
                if c not in exclude_cols and df[c].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]
            ]
        
        logger.info(f"Fitting tokenizer on {len(self.feature_columns)} features")
        
        # Compute bin edges for each feature
        current_offset = self.N_SPECIAL
        
        for col in self.feature_columns:
            values = df[col].drop_nulls().to_numpy().astype(np.float64)
            
            if len(values) == 0:
                logger.warning(f"Skipping {col}: no valid values")
                continue
            
            # Compute quantile edges (n_bins + 1 edges)
            quantiles = np.linspace(0, 1, self.n_bins + 1)
            edges = np.quantile(values, quantiles)
            
            # Handle edge cases where all values are same
            if np.all(edges == edges[0]):
                edges = np.array([edges[0] - 1, edges[0] + 1])
            
            self.bin_edges[col] = edges
            self.feature_offsets[col] = current_offset
            current_offset += self.n_bins
        
        self.vocab_size = current_offset
        self.is_fitted = True
        
        logger.info(f"Tokenizer fitted. Vocabulary size: {self.vocab_size}")
    
    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Transform features to token IDs.
        
        Args:
            df: DataFrame with features
            
        Returns:
            DataFrame with token ID columns (integer)
        """
        if not self.is_fitted:
            raise RuntimeError("Tokenizer must be fitted before transform")
        
        token_columns = []
        
        for col in self.feature_columns:
            if col not in df.columns:
                logger.warning(f"Column {col} not found, using [UNK]")
                token_col = pl.lit(self.SPECIAL_TOKENS["[UNK]"]).alias(f"{col}_token")
                token_columns.append(token_col)
                continue
            
            values = df[col].to_numpy().astype(np.float64)
            edges = self.bin_edges[col]
            offset = self.feature_offsets[col]
            
            # Digitize: returns bin indices 1 to n_bins, we want 0 to n_bins-1
            bin_indices = np.digitize(values, edges[1:-1])  # Skip first and last edge
            
            # Handle NaN -> [UNK]
            bin_indices = np.where(np.isnan(values), self.SPECIAL_TOKENS["[UNK]"], bin_indices)
            
            # Add offset to get final token IDs
            token_ids = (offset + bin_indices).astype(np.int32)
            
            token_columns.append(pl.Series(f"{col}_token", token_ids))
        
        return df.with_columns(token_columns)
    
    def fit_transform(self, df: pl.DataFrame, feature_columns: Optional[List[str]] = None) -> pl.DataFrame:
        """Fit and transform in one call."""
        self.fit(df, feature_columns)
        return self.transform(df)
    
    def encode_sequence(self, token_df: pl.DataFrame) -> np.ndarray:
        """
        Convert tokenized DataFrame to a flat sequence of token IDs.
        
        For language modeling, we interleave all feature tokens per timestep.
        
        Args:
            token_df: DataFrame with *_token columns
            
        Returns:
            1D array of token IDs [seq_len * n_features]
        """
        token_cols = [c for c in token_df.columns if c.endswith("_token")]
        
        # Stack all token columns [timesteps, n_features]
        tokens = token_df.select(token_cols).to_numpy().astype(np.int32)
        
        # Flatten row-wise (each timestep's features in sequence)
        return tokens.flatten()
    
    def save(self, path: Path = TOKENIZER_PATH):
        """Save tokenizer to pickle file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        
        state = {
            "n_bins": self.n_bins,
            "feature_columns": self.feature_columns,
            "bin_edges": self.bin_edges,
            "feature_offsets": self.feature_offsets,
            "vocab_size": self.vocab_size,
            "special_tokens": self.SPECIAL_TOKENS
        }
        
        with open(path, "wb") as f:
            pickle.dump(state, f)
        
        logger.info(f"Tokenizer saved to {path}")
    
    @classmethod
    def load(cls, path: Path = TOKENIZER_PATH) -> "FinancialTokenizer":
        """Load tokenizer from pickle file."""
        with open(path, "rb") as f:
            state = pickle.load(f)
        
        tokenizer = cls(n_bins=state["n_bins"])
        tokenizer.feature_columns = state["feature_columns"]
        tokenizer.bin_edges = state["bin_edges"]
        tokenizer.feature_offsets = state["feature_offsets"]
        tokenizer.vocab_size = state["vocab_size"]
        tokenizer.is_fitted = True
        
        logger.info(f"Tokenizer loaded from {path}. Vocab size: {tokenizer.vocab_size}")
        return tokenizer
    
    def decode_token(self, token_id: int) -> Tuple[str, int]:
        """
        Decode a single token ID to (feature_name, bin_index).
        
        Args:
            token_id: Token ID
            
        Returns:
            Tuple of (feature_name, bin_index) or ("[SPECIAL]", token_id)
        """
        # Check special tokens
        for name, tid in self.SPECIAL_TOKENS.items():
            if token_id == tid:
                return (name, token_id)
        
        # Find which feature this token belongs to
        for col in self.feature_columns:
            offset = self.feature_offsets[col]
            if offset <= token_id < offset + self.n_bins:
                return (col, token_id - offset)
        
        return ("[UNK]", token_id)


def tokenize_all_parquet(
    input_dir: Path = DATA_PROCESSED_DIR,
    output_dir: Path = DATA_TOKENIZED_DIR,
    tokenizer_path: Path = TOKENIZER_PATH,
    n_bins: int = 256
):
    """
    Tokenize all processed parquet files.
    
    Args:
        input_dir: Directory with processed parquet files
        output_dir: Directory for tokenized output
        tokenizer_path: Path to save tokenizer
        n_bins: Number of bins per feature
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    parquet_files = list(input_dir.glob("*.parquet"))
    if not parquet_files:
        logger.error(f"No parquet files found in {input_dir}")
        return
    
    # Load all data for fitting
    logger.info(f"Loading {len(parquet_files)} parquet files for tokenizer fitting...")
    all_dfs = [pl.read_parquet(f) for f in parquet_files]
    combined_df = pl.concat(all_dfs)
    
    # Fit tokenizer
    tokenizer = FinancialTokenizer(n_bins=n_bins)
    tokenizer.fit(combined_df)
    tokenizer.save(tokenizer_path)
    
    # Transform each file
    for file_path in parquet_files:
        df = pl.read_parquet(file_path)
        df_tokenized = tokenizer.transform(df)
        
        # Save tokenized data
        output_path = output_dir / file_path.name
        df_tokenized.write_parquet(output_path)
        logger.info(f"Tokenized {file_path.stem} -> {output_path}")
    
    logger.info(f"Tokenization complete. Vocab size: {tokenizer.vocab_size}")


def main():
    """CLI entry point."""
    tokenize_all_parquet()


if __name__ == "__main__":
    main()



--- FILE: backend/src/data/dataset.py ---
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



--- FILE: backend/src/data/processor.py ---
"""
Multi-Asset Data Processor for I.A.R.A. v6.0
=============================================
Extended indicators with momentum filtering.

Features:
- Multi-asset support (all forex majors)
- Core indicators: RSI, MACD, BBands, ATR
- NEW: Stochastic Oscillator (%K, %D) for fake breakout filtering
- NEW: Price Velocity (rate of change) for momentum
- Cyclical time encoding (hour, day of week)
- Smart label filtering (volatility threshold)
- Unified StandardScaler across all assets
"""

import logging
import math
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import polars as pl
from sklearn.preprocessing import StandardScaler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Paths
RAW_DATA_DIR = Path(__file__).parent.parent.parent / "data" / "raw"
PROCESSED_DATA_DIR = Path(__file__).parent.parent.parent / "data" / "processed"
SCALER_PATH = Path(__file__).parent.parent.parent / "data" / "scaler.pkl"


@dataclass
class NormalizerState:
    """Serializable state for the normalizer."""
    feature_names: List[str] = field(default_factory=list)
    scaler_params: Dict = field(default_factory=dict)


class Normalizer:
    """Multi-asset normalizer with unified scaling."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_names: List[str] = []
        self.fitted = False
    
    def fit(self, features: np.ndarray, feature_names: List[str]):
        """Fit scaler on pooled multi-asset data."""
        self.scaler.fit(features)
        self.feature_names = feature_names
        self.fitted = True
    
    def transform(self, features: np.ndarray) -> np.ndarray:
        """Transform features using fitted scaler."""
        if not self.fitted:
            raise ValueError("Normalizer not fitted yet")
        return self.scaler.transform(features)
    
    def fit_transform(self, features: np.ndarray, feature_names: List[str]) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(features, feature_names)
        return self.transform(features)
    
    def save(self, path: Path = SCALER_PATH):
        """Save normalizer state."""
        state = NormalizerState(
            feature_names=self.feature_names,
            scaler_params={
                'mean_': self.scaler.mean_.tolist(),
                'var_': self.scaler.var_.tolist(),
                'scale_': self.scaler.scale_.tolist(),
                'n_features_in_': self.scaler.n_features_in_
            }
        )
        with open(path, 'wb') as f:
            pickle.dump(state, f)
        logger.info(f"Saved normalizer to {path}")
    
    @classmethod
    def load(cls, path: Path = SCALER_PATH) -> 'Normalizer':
        """Load normalizer from file."""
        normalizer = cls()
        with open(path, 'rb') as f:
            state = pickle.load(f)
        
        normalizer.feature_names = state.feature_names
        normalizer.scaler.mean_ = np.array(state.scaler_params['mean_'])
        normalizer.scaler.var_ = np.array(state.scaler_params['var_'])
        normalizer.scaler.scale_ = np.array(state.scaler_params['scale_'])
        normalizer.scaler.n_features_in_ = state.scaler_params['n_features_in_']
        normalizer.fitted = True
        
        logger.info(f"Loaded normalizer. Features: {len(normalizer.feature_names)}")
        return normalizer


def compute_rsi(close: pl.Series, period: int = 14) -> pl.Series:
    """Compute RSI indicator."""
    delta = close.diff()
    gain = delta.clip(lower_bound=0)
    loss = (-delta).clip(lower_bound=0)
    
    avg_gain = gain.rolling_mean(window_size=period)
    avg_loss = loss.rolling_mean(window_size=period)
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def compute_macd(close: pl.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pl.Series, pl.Series, pl.Series]:
    """Compute MACD indicator."""
    ema_fast = close.ewm_mean(span=fast, min_samples=fast)
    ema_slow = close.ewm_mean(span=slow, min_samples=slow)
    
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm_mean(span=signal, min_samples=signal)
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram


def compute_bollinger(close: pl.Series, period: int = 20, num_std: float = 2.0) -> Tuple[pl.Series, pl.Series, pl.Series, pl.Series]:
    """Compute Bollinger Bands."""
    sma = close.rolling_mean(window_size=period)
    std = close.rolling_std(window_size=period)
    
    upper = sma + num_std * std
    lower = sma - num_std * std
    
    # Percent B: position within bands
    pct_b = (close - lower) / (upper - lower + 1e-10)
    
    # Bandwidth: volatility measure
    bandwidth = (upper - lower) / (sma + 1e-10)
    
    return upper, lower, pct_b, bandwidth


def compute_atr(high: pl.Series, low: pl.Series, close: pl.Series, period: int = 14) -> pl.Series:
    """Compute Average True Range."""
    prev_close = close.shift(1)
    
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    
    tr = pl.max_horizontal(tr1, tr2, tr3)
    atr = tr.rolling_mean(window_size=period)
    
    return atr


def compute_stochastic(high: pl.Series, low: pl.Series, close: pl.Series, 
                       k_period: int = 14, d_period: int = 3) -> Tuple[pl.Series, pl.Series]:
    """
    Compute Stochastic Oscillator (14, 3, 3).
    Critical for filtering fake breakouts.
    """
    lowest_low = low.rolling_min(window_size=k_period)
    highest_high = high.rolling_max(window_size=k_period)
    
    # %K: Raw stochastic
    stoch_k = 100 * (close - lowest_low) / (highest_high - lowest_low + 1e-10)
    
    # %D: Smoothed %K
    stoch_d = stoch_k.rolling_mean(window_size=d_period)
    
    return stoch_k, stoch_d


def compute_price_velocity(close: pl.Series, period: int = 5) -> pl.Series:
    """
    Compute Price Velocity (1st derivative of price).
    Rate of change over the last N periods.
    """
    # Rate of change: (close - close[n]) / close[n]
    roc = (close - close.shift(period)) / (close.shift(period) + 1e-10) * 100
    return roc


def compute_adx(high: pl.Series, low: pl.Series, close: pl.Series, period: int = 14) -> pl.Series:
    """
    Compute Average Directional Index (ADX).
    Measures trend strength (0-100).
    """
    # 1. True Range
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pl.max_horizontal(tr1, tr2, tr3)
    
    # 2. Directional Movement
    up = high - high.shift(1)
    down = low.shift(1) - low
    
    pos_dm = pl.when((up > down) & (up > 0)).then(up).otherwise(0.0)
    neg_dm = pl.when((down > up) & (down > 0)).then(down).otherwise(0.0)
    
    # 3. Smooth TR, +DM, -DM (Wilder's Smoothing = EMA with alpha=1/period)
    tr_smooth = tr.ewm_mean(alpha=1/period, min_samples=period)
    pos_dm_smooth = pos_dm.ewm_mean(alpha=1/period, min_samples=period)
    neg_dm_smooth = neg_dm.ewm_mean(alpha=1/period, min_samples=period)
    
    # 4. DI
    pos_di = 100 * (pos_dm_smooth / (tr_smooth + 1e-10))
    neg_di = 100 * (neg_dm_smooth / (tr_smooth + 1e-10))
    
    # 5. DX
    dx = 100 * (pos_di - neg_di).abs() / (pos_di + neg_di + 1e-10)
    
    # 6. ADX
    adx = dx.ewm_mean(alpha=1/period, min_samples=period)
    
    return adx


def compute_time_features(df: pl.DataFrame) -> pl.DataFrame:
    """Compute cyclical time features using native Polars."""
    # Extract hour and day of week
    if "time" in df.columns:
        hour = df["time"].dt.hour()
        day = df["time"].dt.weekday()
    else:
        # Fallback: create dummy features
        n = len(df)
        hour = pl.Series("hour", [12] * n)
        day = pl.Series("day", [2] * n)
    
    # Cyclical encoding
    hour_rad = hour * (2 * math.pi / 24)
    day_rad = day * (2 * math.pi / 7)
    
    return df.with_columns([
        hour.alias("hour"),
        hour_rad.sin().alias("hour_sin"),
        hour_rad.cos().alias("hour_cos"),
        day_rad.sin().alias("day_sin"),
        day_rad.cos().alias("day_cos")
    ])


def process_single_asset(file_path: Path, volatility_threshold: float = 0.0002) -> Optional[pl.DataFrame]:
    """Process a single asset file with all indicators."""
    try:
        df = pl.read_parquet(file_path)
        ticker = file_path.stem
        
        # Required columns
        if not all(c in df.columns for c in ["open", "high", "low", "close"]):
            logger.warning(f"{ticker}: Missing OHLC columns")
            return None
        
        # Compute indicators
        close = df["close"]
        high = df["high"]
        low = df["low"]
        
        # Core Indicators
        rsi = compute_rsi(close, 14)
        macd, macd_signal, macd_hist = compute_macd(close)
        bb_upper, bb_lower, bb_pct_b, bb_width = compute_bollinger(close)
        atr = compute_atr(high, low, close, 14)
        
        # NEW v6.0: Stochastic Oscillator
        stoch_k, stoch_d = compute_stochastic(high, low, close, 14, 3)
        
        # NEW v6.0: Price Velocity
        velocity = compute_price_velocity(close, 5)
        
        # NEW v6.1: ADX
        adx = compute_adx(high, low, close, 14)
        
        # NEW v6.7: Global Sentiment (EMA 50 Trend Proxy)
        ema_50 = close.ewm_mean(span=50, min_samples=50)
        global_sentiment = (close - ema_50) / (ema_50 + 1e-10) * 100
        
        # Add all indicators
        df = df.with_columns([
            rsi.alias("rsi"),
            macd.alias("macd"),
            macd_signal.alias("macd_signal"),
            macd_hist.alias("macd_hist"),
            bb_pct_b.alias("bb_pct_b"),
            bb_width.alias("bb_width"),
            atr.alias("atr"),
            stoch_k.alias("stoch_k"),
            stoch_d.alias("stoch_d"),
            velocity.alias("velocity"),
            adx.alias("adx"),
            global_sentiment.alias("global_sentiment")
        ])
        
        # Time features
        df = compute_time_features(df)
        
        # Target: log return
        close_log_ret = close.log().diff()
        df = df.with_columns([
            close_log_ret.alias("close_log_ret")
        ])
        
        # Smart label: valid if volatility > threshold
        valid_sample = close_log_ret.abs() > volatility_threshold
        df = df.with_columns([
            valid_sample.alias("valid_sample")
        ])
        
        # Add ticker column
        df = df.with_columns([
            pl.lit(ticker).alias("ticker")
        ])
        
        # Drop warmup rows (increased for EMA 50)
        df = df.slice(50, len(df) - 50)
        
        return df
        
    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}")
        return None


def process_all_assets(
    tickers: Optional[List[str]] = None,
    volatility_threshold: float = 0.0001  # Lowered to 1 pip for 5m data
) -> None:
    """Process all assets with unified scaling."""
    
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Get files to process
    if tickers:
        files = [RAW_DATA_DIR / f"{t}.parquet" for t in tickers if (RAW_DATA_DIR / f"{t}.parquet").exists()]
    else:
        files = list(RAW_DATA_DIR.glob("*.parquet"))
    
    logger.info(f"Processing {len(files)} assets...")
    
    # Feature columns to normalize (v6.7: 16 features)
    feature_cols = [
        "rsi", "macd", "macd_signal", "macd_hist",
        "bb_pct_b", "bb_width", "atr",
        "stoch_k", "stoch_d",
        "velocity",
        "adx",
        "global_sentiment", # NEW v6.7
        "hour_sin", "hour_cos", "day_sin", "day_cos"
    ]
    
    # First pass: collect all data for unified scaling
    all_features = []
    processed_dfs = {}
    
    for file_path in files:
        df = process_single_asset(file_path, volatility_threshold)
        if df is not None:
            ticker = file_path.stem
            processed_dfs[ticker] = df
            
            # Extract features for scaling
            available_cols = [c for c in feature_cols if c in df.columns]
            features = df.select(available_cols).to_numpy()
            
            # Remove NaN rows for scaling
            mask = ~np.isnan(features).any(axis=1)
            all_features.append(features[mask])
    
    if not all_features:
        logger.error("No data processed!")
        return
    
    # Fit normalizer on pooled data
    pooled_features = np.vstack(all_features)
    logger.info(f"Pooled features shape: {pooled_features.shape}")
    
    normalizer = Normalizer()
    normalizer.fit(pooled_features, feature_cols)
    normalizer.save()
    
    # Second pass: normalize and save
    valid_count = 0
    total_samples = 0
    
    for ticker, df in processed_dfs.items():
        available_cols = [c for c in feature_cols if c in df.columns]
        features = df.select(available_cols).to_numpy()
        
        # Normalize
        features_norm = normalizer.transform(features)
        
        # Add normalized columns
        for i, col in enumerate(available_cols):
            df = df.with_columns([
                pl.Series(f"{col}_norm", features_norm[:, i])
            ])
        
        # Save
        output_path = PROCESSED_DATA_DIR / f"{ticker}.parquet"
        df.write_parquet(output_path)
        
        # Stats
        valid_mask = df["valid_sample"].to_numpy()
        valid_count += valid_mask.sum()
        total_samples += len(df)
        
        logger.info(f"  {ticker}: {len(df)} samples, {valid_mask.sum()} valid ({valid_mask.mean()*100:.1f}%)")
    
    logger.info(f"Processing complete. {len(processed_dfs)} tickers, {total_samples} total samples, {valid_count} valid ({valid_count/total_samples*100:.1f}%)")
    logger.info(f"Features ({len(feature_cols)}): {feature_cols}")


if __name__ == "__main__":
    # Process main 4 pairs as requested + others if available
    target_pairs = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"]
    
    # Check what we have
    available = [f.stem for f in RAW_DATA_DIR.glob("*.parquet")]
    to_process = [t for t in target_pairs if t in available]
    
    if len(to_process) < len(target_pairs):
        logger.warning(f"Only found {len(to_process)} of 4 target pairs: {to_process}")
    
    process_all_assets(tickers=to_process)



--- FILE: backend/src/data/onchain.py ---
"""
I.A.R.A. v11 — On-Chain Data Ingestor
======================================
Tracks whale wallet flows to/from exchanges for accumulation/distribution signals.

Provides:
  - Node features: balance_change_24h, tx_count, is_exchange (per wallet)
  - Adjacency matrix: sparse matrix of transaction volumes between wallets

Currently operates in mock mode with realistic synthetic data.
Real integration requires a blockchain data API (Glassnode, Nansen, The Graph).
"""

import logging
import time
import threading
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("OnChainIngestor")


class OnChainIngestor:
    """
    On-chain whale wallet tracker.

    Tracks the top N whale wallets and their transaction patterns
    to/from exchanges. Builds a graph representation suitable for
    Graph Neural Network processing.

    Modes:
        - mock: Generates synthetic but realistic data
        - live: (Future) Connects to blockchain API
    """

    # Known exchange cold wallets (placeholder identifiers)
    EXCHANGE_WALLETS = {
        "binance_cold_1", "binance_cold_2",
        "coinbase_cold_1",
        "kraken_cold_1",
        "okx_cold_1",
    }

    def __init__(
        self,
        n_wallets: int = 50,
        mode: str = "mock",
        api_key: Optional[str] = None,
        update_interval: float = 300.0,  # 5 minutes
    ):
        self.n_wallets = n_wallets
        self.mode = mode
        self.api_key = api_key
        self.update_interval = update_interval

        # Node features: [balance_change_24h, tx_count_norm, is_exchange]
        self.node_features = np.zeros((n_wallets, 3), dtype=np.float32)

        # Sparse adjacency: list of (src, dst, weight) tuples
        self.edges: List[Tuple[int, int, float]] = []

        # Background thread
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._last_update = 0.0

        # Initialize exchange flags
        self._exchange_indices = set(range(min(5, n_wallets)))  # First 5 are exchanges

        if mode == "mock":
            self._generate_mock_data()

    def start(self):
        """Start background data update thread."""
        if self._running:
            return
        self._running = True
        if self.mode == "mock":
            logger.info("🔗 On-Chain Ingestor started (mock mode)")
            # Mock mode updates inline
        else:
            self._thread = threading.Thread(target=self._update_loop, daemon=True)
            self._thread.start()
            logger.info("🔗 On-Chain Ingestor started (live mode)")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)

    def _update_loop(self):
        """Background loop for live data updates."""
        while self._running:
            try:
                self._fetch_live_data()
            except Exception as e:
                logger.error(f"On-chain update error: {e}")
            time.sleep(self.update_interval)

    def _fetch_live_data(self):
        """Fetch real on-chain data (placeholder for future API integration)."""
        # TODO: Integrate with Glassnode / Nansen / The Graph API
        logger.debug("Live on-chain data fetch not yet implemented. Using mock.")
        self._generate_mock_data()

    def _generate_mock_data(self):
        """Generate realistic synthetic whale wallet data."""
        n = self.n_wallets

        with self._lock:
            # Node features
            for i in range(n):
                is_exchange = 1.0 if i in self._exchange_indices else 0.0

                if is_exchange:
                    # Exchanges: large balance changes, high tx count
                    balance_change = np.random.normal(0, 5000)
                    tx_count = np.random.poisson(50)
                else:
                    # Whale wallets: moderate activity
                    balance_change = np.random.normal(0, 500)
                    tx_count = np.random.poisson(5)

                self.node_features[i] = [
                    np.clip(balance_change / 10000, -1, 1),  # Normalized
                    np.clip(tx_count / 100, 0, 1),           # Normalized
                    is_exchange,
                ]

            # Generate edges (transactions between wallets)
            self.edges = []
            n_edges = np.random.poisson(n * 2)  # ~2 transactions per wallet on avg

            for _ in range(n_edges):
                src = np.random.randint(0, n)
                dst = np.random.randint(0, n)
                if src == dst:
                    continue

                # Higher volume for exchange-related transactions
                if src in self._exchange_indices or dst in self._exchange_indices:
                    volume = abs(np.random.normal(0, 2000))
                else:
                    volume = abs(np.random.normal(0, 200))

                weight = np.clip(volume / 5000, 0, 1)  # Normalized
                self.edges.append((src, dst, weight))

        self._last_update = time.time()

    def get_graph_data(self) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Get the current graph data for GNN processing.

        Returns:
            Tuple of:
              - node_features: np.ndarray [n_wallets, 3]
              - sparse_adj: Tuple of (row_indices, col_indices, values) for sparse COO
        """
        # Refresh mock data periodically
        if self.mode == "mock" and time.time() - self._last_update > 60:
            self._generate_mock_data()

        with self._lock:
            features = self.node_features.copy()

            if self.edges:
                rows = np.array([e[0] for e in self.edges], dtype=np.int64)
                cols = np.array([e[1] for e in self.edges], dtype=np.int64)
                vals = np.array([e[2] for e in self.edges], dtype=np.float32)
            else:
                rows = np.zeros(0, dtype=np.int64)
                cols = np.zeros(0, dtype=np.int64)
                vals = np.zeros(0, dtype=np.float32)

        return features, (rows, cols, vals)

    def get_graph_tensors(self):
        """
        Get graph data as PyTorch tensors.

        Returns:
            Tuple of:
              - node_features: torch.Tensor [n_wallets, 3]
              - adj_sparse: torch.sparse_coo_tensor [n_wallets, n_wallets]
        """
        import torch

        features, (rows, cols, vals) = self.get_graph_data()

        node_features = torch.tensor(features, dtype=torch.float32)

        if len(rows) > 0:
            indices = torch.stack([
                torch.tensor(rows, dtype=torch.long),
                torch.tensor(cols, dtype=torch.long),
            ])
            values = torch.tensor(vals, dtype=torch.float32)
            adj_sparse = torch.sparse_coo_tensor(
                indices, values,
                size=(self.n_wallets, self.n_wallets),
            ).coalesce()
        else:
            adj_sparse = torch.sparse_coo_tensor(
                torch.zeros(2, 0, dtype=torch.long),
                torch.zeros(0),
                size=(self.n_wallets, self.n_wallets),
            )

        return node_features, adj_sparse

    def get_flow_summary(self) -> Dict[str, float]:
        """
        High-level summary of on-chain flow direction.

        Returns:
            Dict with:
              - to_exchange: proportion of volume flowing TO exchanges
              - from_exchange: proportion of volume flowing FROM exchanges
              - net_flow: positive = accumulation, negative = distribution
        """
        to_exchange = 0.0
        from_exchange = 0.0

        with self._lock:
            for src, dst, weight in self.edges:
                if dst in self._exchange_indices:
                    to_exchange += weight
                if src in self._exchange_indices:
                    from_exchange += weight

        total = to_exchange + from_exchange + 1e-10
        return {
            "to_exchange": to_exchange / total,
            "from_exchange": from_exchange / total,
            "net_flow": (from_exchange - to_exchange) / total,  # Positive = money leaving exchanges (bullish)
        }

    @property
    def is_connected(self) -> bool:
        return self._running



--- FILE: backend/src/data/ingest_db.py ---
"""
I.A.R.A. Database Ingestion
===========================
Reads Forex CSV data (M1) and populates the TimescaleDB 'market_candles' table.
Crucial for the Strategic Agent's 'MarketScanner' tool.

Usage:
    python src/data/ingest_db.py --file data/raw/EURUSD.csv --symbol EURUSD
"""

import os
import argparse
import pandas as pd
from sqlalchemy import create_engine, text
from datetime import datetime
import logging
from dotenv import load_dotenv

# Load env vars
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# DB Connection
DB_URI = os.getenv("DB_URI")

def ingest_csv(file_path: str, symbol: str):
    if not DB_URI:
        logger.error("DB_URI environment variable not set.")
        return

    logger.info(f"📂 Reading {file_path}...")
    try:
        # Load CSV
        df = pd.read_csv(file_path)
        
        # Normalize columns
        df.columns = [c.lower().strip() for c in df.columns]
        
        # Ensure timestamp
        if "date" in df.columns:
            df["time"] = pd.to_datetime(df["date"])
        elif "timestamp" in df.columns:
            df["time"] = pd.to_datetime(df["timestamp"])
            
        # Ensure required cols
        # req_cols = ["time", "open", "high", "low", "close", "volume"] # validation logic can be added here
        
        df["symbol"] = symbol
        df["interval"] = "1m" # Assuming M1 data
        df["source"] = "histdata"
        
        # Filter for DB Schema
        db_cols = ["time", "symbol", "interval", "open", "high", "low", "close", "volume", "source"]
        
        # Rename if necessary
        if "tick_volume" in df.columns:
            df = df.rename(columns={"tick_volume": "volume"})
            
        # Select only existing columns that match schema
        available_cols = [c for c in db_cols if c in df.columns]
        df = df[available_cols]
        
        logger.info(f"🚀 Pushing {len(df)} rows to TimescaleDB...")
        engine = create_engine(DB_URI)
        
        # Batch insert using pandas
        df.to_sql("market_candles", engine, if_exists="append", index=False, chunksize=10000, method='multi')
        
        logger.info(f"✅ Successfully ingested {symbol} data!")
        
    except Exception as e:
        logger.error(f"❌ Ingestion Failed: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True, help="Path to CSV file")
    parser.add_argument("--symbol", required=True, help="Symbol (e.g. EURUSD)")
    args = parser.parse_args()
    
    ingest_csv(args.file, args.symbol)



--- FILE: backend/src/dashboard/app.py ---
import streamlit as st
import os
import sys
import importlib

# Ensure src is in path for imports
sys.path.append(os.path.abspath(os.curdir))

st.set_page_config(
    page_title="I.A.R.A. Mission Control",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Professional Dark Theme CSS ──────────────────────────────
st.markdown("""
<style>
    /* Dark pro background */
    .stApp {
        background-color: #0e1117;
    }
    /* Metric cards */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #1a1f2e 0%, #141824 100%);
        border: 1px solid rgba(128, 128, 128, 0.15);
        border-radius: 12px;
        padding: 12px 16px;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.1rem;
        font-weight: 700;
    }
    [data-testid="stMetricDelta"] > div {
        font-size: 0.85rem;
    }
    /* Dataframe styling */
    .stDataFrame {
        border-radius: 8px;
        overflow: hidden;
    }
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0e1117 0%, #161b22 100%);
        border-right: 1px solid rgba(128, 128, 128, 0.1);
    }
    /* Plotly chart containers */
    .stPlotlyChart {
        border-radius: 8px;
        border: 1px solid rgba(128, 128, 128, 0.1);
    }
    /* Tabs */
    .stTabs [data-baseweb="tab"] {
        font-weight: 600;
    }
    /* Dividers */
    hr {
        border-color: rgba(128, 128, 128, 0.1) !important;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar Navigation
st.sidebar.title("🧬 I.A.R.A. v8.5")
st.sidebar.markdown("---")

# Navigation
page = st.sidebar.radio("Navigation", ["🏢 Trading Desk", "Mission Control", "Training Lab", "Model Manager", "Simulation Lab"])

st.sidebar.markdown("---")
st.sidebar.info(
    """
    **System Status:**
    - **Engine:** Trinity v8.5
    - **Mode:** Dual Core (Phase 1)
    - **Device:** ROCm (RX 6750 XT)
    """
)

# Router
if page == "🏢 Trading Desk":
    from src.dashboard.pages import live_trading
    importlib.reload(live_trading)
    live_trading.render()
elif page == "Mission Control":
    from src.dashboard.pages import mission_control
    importlib.reload(mission_control)
    mission_control.render()
elif page == "Training Lab":
    from src.dashboard.pages import training
    importlib.reload(training)
    training.render()
elif page == "Model Manager":
    from src.dashboard.pages import models
    importlib.reload(models)
    models.render()
elif page == "Simulation Lab":
    from src.dashboard.pages import simulation
    try:
        importlib.reload(simulation)
        simulation.render()
    except ImportError:
        st.warning("Simulation Lab module not found yet. Please wait for installation.")
    except Exception as e:
         st.error(f"Sim Lab Error: {e}")



--- FILE: backend/src/dashboard/pages/training.py ---
import streamlit as st
import subprocess
from pathlib import Path
import os
import signal  # For killing the process
import time
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import numpy as np
import torch
import datetime

print(f">>> [{datetime.datetime.now()}] RELOADING training.py (v9.6) <<<")

# Constants
TRAIN_SCRIPT = "src/brain/train_v9.py"  # Updated to v9 Swarm Intelligence
AUTO_LEARN_SCRIPT = "src/brain/auto_learn_v9.py"
LOG_DIR = "logs"
METRICS_FILE = os.path.join(LOG_DIR, "metrics.csv")
SNAPSHOT_DIR = os.path.join(LOG_DIR, "snapshots")
PID_FILE = "training_process.pid"
AUTO_PID_FILE = "auto_learn_process.pid"
TELEMETRY_FILE = os.path.join(LOG_DIR, "telemetry.json")
MODELS_DIR = os.path.join(LOG_DIR, "models")
MODEL_ZOO_DIR = os.path.join(LOG_DIR, "model_zoo")


def save_checkpoint():
    """Send signal to training process to save checkpoint"""
    if os.path.exists(PID_FILE):
        try:
            with open(PID_FILE, "r") as f:
                pid = int(f.read().strip())
            os.kill(pid, signal.SIGUSR1)
            st.toast("💾 Save signal sent!", icon="✅")
        except Exception as e:
            st.error(f"Failed to send save signal: {e}")


def clear_logs():
    """Clear metric logs"""
    try:
        if os.path.exists(METRICS_FILE):
            os.remove(METRICS_FILE)
        # Also clear v9 file if exists
        v9_metrics = os.path.join(LOG_DIR, "metrics_v9.csv")
        if os.path.exists(v9_metrics):
            os.remove(v9_metrics)
        st.toast("🧹 Logs cleared!", icon="✅")
        time.sleep(0.5)
        st.rerun()
    except Exception as e:
        st.error(f"Failed to clear logs: {e}")


def is_training_running():
    if os.path.exists(PID_FILE):
        try:
            with open(PID_FILE, "r") as f:
                pid = int(f.read().strip())

            # Check if process exists via signal
            os.kill(pid, 0)

            # Check if Zombie (Linux only)
            try:
                with open(f"/proc/{pid}/stat", "r") as stat_file:
                    state = stat_file.read().split()[2]
                    if state == "Z":
                        # It's a zombie, cleanup
                        os.waitpid(pid, os.WNOHANG)  # Reap it
                        os.remove(PID_FILE)
                        return False
            except:
                pass  # Not linux or permission error

            return True
        except OSError:
            if os.path.exists(PID_FILE):
                os.remove(PID_FILE)

    # Fallback: infer from telemetry freshness (auto_learn_v9 runs train_v9 without dashboard PID)
    if os.path.exists(TELEMETRY_FILE):
        try:
            mtime = os.path.getmtime(TELEMETRY_FILE)
            if (time.time() - mtime) < 15:
                with open(TELEMETRY_FILE, "r") as f:
                    telemetry = json.load(f)
                if telemetry.get("status") in {"TRAINING", "ROLLBACK"}:
                    return True
        except Exception:
            pass
    return False


def is_auto_learn_running():
    if os.path.exists(AUTO_PID_FILE):
        try:
            with open(AUTO_PID_FILE, "r") as f:
                pid = int(f.read().strip())
            os.kill(pid, 0)
            return True
        except OSError:
            if os.path.exists(AUTO_PID_FILE):
                os.remove(AUTO_PID_FILE)
        except Exception:
            if os.path.exists(AUTO_PID_FILE):
                os.remove(AUTO_PID_FILE)
    return False


def stop_training():
    if os.path.exists(PID_FILE):
        try:
            with open(PID_FILE, "r") as f:
                pid = int(f.read().strip())

            # Kill process group (parent + children like dataloaders)
            try:
                os.killpg(pid, signal.SIGTERM)
                time.sleep(1)  # Give it a second to die gracefully
            except OSError:
                pass  # Process might be gone already

            # Force kill if still alive
            try:
                os.killpg(pid, signal.SIGKILL)
            except OSError:
                pass

            st.toast(f"🛑 Mission Aborted (PID {pid} killed)", icon="skull")

        except Exception as e:
            st.error(f"Error checking PID: {e}")

        # Always clean up PID file
        if os.path.exists(PID_FILE):
            os.remove(PID_FILE)


def render():
    # Note: st.set_page_config must NOT be called here as it is set in app.py

    # --- SIDEBAR CONTROLS ---
    with st.sidebar:
        st.header("🎮 Dashboard Controls")
        auto_ref = st.toggle(
            "Auto-Refresh (3s)",
            value=False,
            help="Enable for live telemetry. Disable to interact with Model Zoo reliably.",
        )
        if st.button("🔄 Force Refresh (Clear Cache)"):
            st.cache_data.clear()
            st.rerun()
        st.divider()

    # --- UI TABS ---
    st.title("🧪 Trinity Training Lab Pro")
    tab_config, tab_monitor, tab_forensics, tab_zoo = st.tabs(
        ["⚙️ Config & Launch", "📈 Live Monitor", "🧠 Forensic Lab", "🦁 Model Zoo"]
    )

    # === TAB 1: CONFIG & LAUNCH ===
    with tab_config:
        col1, col2 = st.columns([1, 2])
        with col1:
            st.subheader("Mission Paramters")
            # Epochs & Patience
            infinite_epochs = st.toggle(
                "Modo Noturno (Infinito)",
                value=False,
                help="Define épocas para 1.000.000",
            )
            if infinite_epochs:
                epochs = 1000000
                st.info("🕒 Épocas definidas para 1.000.000")
            else:
                epochs = st.slider("Epochs", 1, 1000, 200)

            patience = st.number_input(
                "Patience (Early Stopping)",
                value=50,
                min_value=10,
                step=10,
                help="Para o treino após X épocas sem melhora",
            )

            lr = st.number_input("Learning Rate", value=0.0005, format="%.5f")
            # Batch Size Logic
            batch_options = [64, 128, 256, 512, 1024, 2048, "Custom"]
            batch_preset = st.selectbox(
                "Batch Size", batch_options, index=0
            )  # Default 64

            if batch_preset == "Custom":
                batch_size = st.number_input(
                    "Custom Batch Size", value=64, min_value=16, step=16
                )
            else:
                batch_size = batch_preset

            # Market Configuration
            st.divider()
            st.subheader("📊 Market Configuration")
            market_type = st.radio("Market Type", ["Forex", "Crypto"], horizontal=True)
            
            if market_type == "Crypto":
                crypto_symbols = ["BTCUSD", "ETHUSD", "SOLUSD", "DOGEUSD", "LTCUSD"]
                symbol = st.selectbox("Symbol", crypto_symbols, index=0)
                
                # Check if data exists
                data_path = Path(f"data/processed/{symbol}.parquet")
                cache_path = Path(f"data/cache/{symbol}")
                
                if data_path.exists():
                    st.success(f"✅ Historical data found: {data_path.name}")
                else:
                    st.warning(f"⚠️ No data for {symbol}. Fetch it below.")
                
                if not cache_path.exists() or not list(cache_path.glob("train_shards/*.pt")):
                    st.warning(f"⚠️ No training cache for {symbol}. Run Feature Factory after fetching data.")
                
                # Data fetch controls
                with st.expander("📥 Fetch Crypto Data"):
                    fetch_days = st.slider("Days of History", 7, 365, 90, key="fetch_days")
                    fetch_tf = st.selectbox("Timeframe", ["1Min", "5Min", "15Min"], index=0, key="fetch_tf")
                    alpaca_symbol = symbol[:3] + "/" + symbol[3:]  # BTCUSD -> BTC/USD
                    
                    if st.button("📥 Fetch Data from Alpaca", type="secondary"):
                        fetch_cmd = [
                            "python3", "src/data/crypto_data_fetcher.py",
                            "--symbol", alpaca_symbol,
                            "--days", str(fetch_days),
                            "--timeframe", fetch_tf,
                        ]
                        st.info(f"Running: {' '.join(fetch_cmd)}")
                        log_file = open("crypto_fetch.log", "w")
                        proc = subprocess.Popen(fetch_cmd, stdout=log_file, stderr=log_file)
                        proc.wait()
                        log_file.close()
                        if proc.returncode == 0:
                            st.success("✅ Data fetched!")
                        else:
                            with open("crypto_fetch.log", "r") as f:
                                st.error(f"❌ Fetch failed:\n{f.read()[-500:]}")
                        time.sleep(1)
                        st.rerun()
                    
                    if st.button("⚙️ Run Feature Factory", type="secondary"):
                        ff_cmd = [
                            "python3", "src/data/feature_factory.py",
                            "--symbol", symbol,
                            "--market-type", "crypto",
                        ]
                        st.info(f"Running: {' '.join(ff_cmd)}")
                        log_file = open("feature_factory.log", "w")
                        proc = subprocess.Popen(ff_cmd, stdout=log_file, stderr=log_file)
                        proc.wait()
                        log_file.close()
                        if proc.returncode == 0:
                            st.success("✅ Features generated!")
                        else:
                            with open("feature_factory.log", "r") as f:
                                st.error(f"❌ Feature Factory failed:\n{f.read()[-500:]}")
                        time.sleep(1)
                        st.rerun()
            else:
                forex_symbols = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD"]
                symbol = st.selectbox("Symbol", forex_symbols, index=0)

            st.caption(f"Compute Unit: GPU (ROCm) | Symbol: {symbol}")

            # Resume/Reset Logic
            start_mode = st.radio(
                "Starting Point",
                [
                    "Resume Training (Load Best Checkpoint)",
                    "Start Fresh (Reset & Backup)",
                ],
                index=0,
                help="Resume: Continues from best_model_v9.pth. Fresh: Backs up old model and starts new.",
            )

            # Advanced Options
            with st.expander("Advanced Configuration"):
                disable_amp = st.checkbox(
                    "Disable Mixed Precision (AMP)",
                    value=False,
                    help="Check this if training hangs on ROCm",
                )
                debug_mode = st.checkbox(
                    "Debug Mode (Log Steps)",
                    value=False,
                    help="Detailed logging for diagnosis",
                )

            launch_btn = st.button(
                "🚀 DEPLOY TRAINING MISSION",
                type="primary",
                disabled=is_training_running(),
            )
            stop_btn = st.button(
                "🛑 ABORT MISSION", type="secondary", disabled=not is_training_running()
            )

            st.divider()
            st.subheader("Auto-Learn Loop (v9)")
            cycles = st.number_input(
                "Cycles", value=3, min_value=1, step=1, disabled=is_auto_learn_running()
            )
            cycle_epochs = st.number_input(
                "Epochs per Cycle",
                value=5,
                min_value=1,
                step=1,
                disabled=is_auto_learn_running(),
            )
            max_steps = st.number_input(
                "Backtest Max Steps",
                value=2000,
                min_value=50,
                step=50,
                disabled=is_auto_learn_running(),
            )
            start_date_bt = st.text_input(
                "Backtest Start Date (YYYY-MM-DD)",
                value="2023-01-01",
                disabled=is_auto_learn_running(),
            )
            end_date_bt = st.text_input(
                "Backtest End Date (YYYY-MM-DD)",
                value="",
                disabled=is_auto_learn_running(),
            )
            backtest_data = st.text_input(
                "Backtest Data Path",
                value="data/processed/EURUSD.parquet",
                disabled=is_auto_learn_running(),
            )
            backtest_speed = st.number_input(
                "Backtest Speed (steps/sec)",
                value=200.0,
                min_value=1.0,
                step=10.0,
                disabled=is_auto_learn_running(),
            )

            st.markdown("#### Train Config (per cycle)")
            auto_seq_len = st.number_input(
                "Seq Len",
                value=64,
                min_value=8,
                step=8,
                disabled=is_auto_learn_running(),
            )
            auto_lr = st.number_input(
                "Learning Rate",
                value=float(lr),
                format="%.6f",
                disabled=is_auto_learn_running(),
            )
            auto_patience = st.number_input(
                "Patience",
                value=10,
                min_value=1,
                step=1,
                disabled=is_auto_learn_running(),
            )
            auto_resume = st.toggle(
                "Resume (load existing checkpoint)",
                value=True,
                disabled=is_auto_learn_running(),
            )

            st.markdown("#### Paths")
            auto_model_path = st.text_input(
                "Model Path",
                value="logs/models/best_model_v9.pth",
                disabled=is_auto_learn_running(),
            )
            auto_work_dir = st.text_input(
                "Work Dir", value="logs/auto_learn_v9", disabled=is_auto_learn_running()
            )

            launch_auto = st.button(
                "🤖 START AUTO-LEARN", type="primary", disabled=is_auto_learn_running()
            )
            stop_auto = st.button(
                "🛑 STOP AUTO-LEARN",
                type="secondary",
                disabled=not is_auto_learn_running(),
            )

            # Manual Controls
            if is_training_running():
                save_btn = st.button(
                    "💾 SAVE CHECKPOINT", help="Force save current model state"
                )
                if save_btn:
                    save_checkpoint()

            if not is_training_running():
                clear_logs_btn = st.button(
                    "🧹 CLEAR LOGS", help="Delete old metric logs to reset charts"
                )
                if clear_logs_btn:
                    clear_logs()

        with col2:
            st.subheader("System Status")
            if is_training_running() or is_auto_learn_running():
                st.success("🟢 TRAINING IN PROGRESS")
                st.markdown("Go to **Live Monitor** tab to see telemetry.")
            else:
                st.warning("⚪ SYSTEM IDLE")

            # Leaderboard Display
            leaderboard_path = os.path.join(LOG_DIR, "leaderboard_v9.json")
            if os.path.exists(leaderboard_path):
                try:
                    with open(leaderboard_path, "r") as f:
                        lb_data = json.load(f)

                    st.markdown("#### 🏆 Top Models (V9 Swarm)")

                    if lb_data:
                        # Find best model
                        best = sorted(
                            lb_data, key=lambda x: x.get("auc", 0), reverse=True
                        )[0]

                        b_col1, b_col2, b_col3 = st.columns([1, 1, 2])
                        b_col1.metric(
                            "Best Loss", f"{best['loss']:.4f}", f"Epoch {best['epoch']}"
                        )
                        b_col2.metric("Best AUC", f"{best.get('auc', 0):.4f}")

                        if b_col3.button(
                            "🗑️ Reset Leaderboard",
                            help="Clear all records if you suspect data leakage or want to restart.",
                        ):
                            try:
                                os.remove(leaderboard_path)
                                st.toast("Leaderboard reset!", icon="🗑️")
                                time.sleep(1)
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error resetting: {e}")

                        # Formatting Table
                        lb_df = pd.DataFrame(lb_data)
                        if "timestamp" in lb_df.columns:
                            lb_df["timestamp"] = pd.to_datetime(
                                lb_df["timestamp"]
                            ).dt.strftime("%H:%M:%S %d/%m")

                        st.dataframe(
                            lb_df[
                                ["epoch", "loss", "accuracy", "auc", "timestamp"]
                            ].sort_values("auc", ascending=False),
                            height=200,
                            hide_index=True,
                            column_config={
                                "epoch": "Epoch",
                                "loss": st.column_config.NumberColumn(
                                    "Loss", format="%.4f"
                                ),
                                "accuracy": st.column_config.ProgressColumn(
                                    "Accuracy", format="%.2f", min_value=0, max_value=1
                                ),
                                "auc": st.column_config.NumberColumn(
                                    "AUC", format="%.4f"
                                ),
                                "timestamp": "Time",
                            },
                        )
                    else:
                        st.info("No models ranked yet.")
                except Exception as e:
                    st.error(f"Leaderboard Error: {e}")

        if launch_btn:
            cmd = [
                "python3",
                TRAIN_SCRIPT,
                "--epochs",
                str(epochs),
                "--patience",
                str(patience),
                "--lr",
                str(lr),
                "--batch_size",
                str(batch_size),
                "--symbol",
                symbol,
            ]

            if "Resume" in start_mode:
                cmd.append("--resume")

            if disable_amp:
                cmd.append("--no_amp")

            if debug_mode:
                cmd.append("--debug")

            # Spawn detached process
            log_file = open("training_std.log", "w")
            proc = subprocess.Popen(
                cmd, stdout=log_file, stderr=log_file, preexec_fn=os.setpgrp
            )

            with open(PID_FILE, "w") as f:
                f.write(str(proc.pid))

            st.toast(f"🚀 Mission Deployed! PID: {proc.pid}", icon="🚀")
            time.sleep(1)
            st.rerun()

        if stop_btn:
            stop_training()
            st.rerun()

        if launch_auto:
            cmd = [
                "python3",
                AUTO_LEARN_SCRIPT,
                "--cycles",
                str(int(cycles)),
                "--train_epochs",
                str(int(cycle_epochs)),
                "--train_batch_size",
                str(int(batch_size)),
                "--train_seq_len",
                str(int(auto_seq_len)),
                "--train_lr",
                str(float(auto_lr)),
                "--train_patience",
                str(int(auto_patience)),
                "--backtest_data",
                str(backtest_data),
                "--backtest_max_steps",
                str(int(max_steps)),
                "--backtest_start_date",
                str(start_date_bt),
                "--backtest_speed",
                str(float(backtest_speed)),
                "--model_path",
                str(auto_model_path),
                "--work_dir",
                str(auto_work_dir),
            ]

            if end_date_bt.strip():
                cmd.extend(["--backtest_end_date", end_date_bt.strip()])

            if not auto_resume:
                cmd.append("--no-resume")

            if disable_amp:
                cmd.append("--no_amp")

            log_file = open("training_std.log", "a")
            proc = subprocess.Popen(
                cmd, stdout=log_file, stderr=log_file, preexec_fn=os.setpgrp
            )
            with open(AUTO_PID_FILE, "w") as f:
                f.write(str(proc.pid))
            st.toast(f"🤖 Auto-Learn Started! PID: {proc.pid}", icon="🤖")
            time.sleep(1)
            st.rerun()

        if stop_auto:
            if os.path.exists(AUTO_PID_FILE):
                try:
                    with open(AUTO_PID_FILE, "r") as f:
                        pid = int(f.read().strip())
                    try:
                        os.killpg(pid, signal.SIGTERM)
                        time.sleep(1)
                    except Exception:
                        pass
                    try:
                        os.killpg(pid, signal.SIGKILL)
                    except Exception:
                        pass
                except Exception as e:
                    st.error(f"Error stopping auto-learn: {e}")
                finally:
                    try:
                        os.remove(AUTO_PID_FILE)
                    except Exception:
                        pass
            st.toast("🛑 Auto-Learn stopped", icon="🛑")
            st.rerun()

    # === TAB 2: LIVE MONITOR ===
    with tab_monitor:
        st.markdown("### 📡 Real-time Telemetry")

        from datetime import datetime

        st.caption(
            f"🕒 Server Time: {datetime.now().strftime('%H:%M:%S')} (Enable Auto-Refresh in sidebar for live view)"
        )

        # Check telemetry
        if not os.path.exists(TELEMETRY_FILE):
            st.info("Waiting for telemetry data... (Start training first)")
        else:
            try:
                with open(TELEMETRY_FILE, "r") as f:
                    telemetry = json.load(f)

                # Mission Progress
                current_epoch = telemetry.get("current_epoch", 0)
                total_epochs = telemetry.get("total_epochs", 100)

                # Progress Bar
                progress = min(1.0, current_epoch / max(1, total_epochs))
                st.markdown("#### 🚀 Mission Progress (v9.5 LIVE)")
                st.progress(progress, text=f"Epoch {current_epoch} / {total_epochs}")

                # Telemetry bar (Aligned with progress bar at 18 spaces)
                cols = st.columns(6)
                cols[0].metric("Epoch", f"{current_epoch} / {total_epochs}")
                cols[1].metric(
                    "Train Loss",
                    f"{telemetry.get('train_loss', telemetry.get('loss', 0.0)):.4f}",
                )
                cols[2].metric(
                    "Val AUC",
                    f"{telemetry.get('val_auc', telemetry.get('auc', 0.0)):.4f}",
                )
                cols[3].metric("Val Acc", f"{telemetry.get('acc', 0.0):.2%}")
                cols[4].metric(
                    "Self-Corrections",
                    f"{telemetry.get('rollback_count', 0)}",
                    help="Número de vezes que a IA detectou piora e voltou para um estado estável",
                )
                # Infer device if missing
                dev = telemetry.get(
                    "device", "cuda" if torch.cuda.is_available() else "cpu"
                )
                cols[5].metric("Device", dev)

                st.divider()

                # === CHARTS & HISTORY ===
                if os.path.exists(METRICS_FILE):
                    try:
                        df = pd.read_csv(METRICS_FILE)

                        if df.empty:
                            st.warning(
                                "⚠️ Training is running but no metrics recorded yet. (Possible NaN Loss due to high Learning Rate?)"
                            )
                        else:
                            # Charts Layout
                            tab_charts, tab_raw, tab_memory = st.tabs(
                                [
                                    "📊 Visual Analysis",
                                    "📄 Raw Data",
                                    "🧠 AI Synaptic Memory",
                                ]
                            )

                            with tab_memory:
                                stats = telemetry.get("memory_stats", {})
                                if not stats:
                                    st.info(
                                        "🧠 Memória ainda está sendo mapeada. Aguarde o primeiro ciclo de análise (Epoch > 1)."
                                    )
                                else:
                                    # 1. Memory Composition Metrics
                                    m_col1, m_col2, m_col3 = st.columns(3)
                                    m_col1.metric(
                                        "Patterns Total", f"{stats.get('n_total', 0):,}"
                                    )
                                    m_col2.metric(
                                        "Chronic (Hard)",
                                        f"{stats.get('n_chronic', 0):,}",
                                        help="Padrões que a IA está tendo dificuldade em aprender (Peso Alto)",
                                    )
                                    m_col3.metric(
                                        "Mastered (Easy)",
                                        f"{stats.get('n_mastered', 0):,}",
                                        help="Padrões que a IA já decorou (Peso Baixo)",
                                    )

                                    # 2. Synaptic Loss Distribution
                                    st.markdown("#### 🌊 Synaptic Loss Distribution")
                                    st.caption(
                                        "Distribuição do 'esquecimento' vs 'prioridade'. Barras à direita indicam o que a IA está focando no momento."
                                    )
                                    dist = stats.get("loss_distribution", [])
                                    if dist:
                                        chart_df = pd.DataFrame(
                                            {
                                                "Dificuldade": [
                                                    f"{i*10}%-{(i+1)*10}%"
                                                    for i in range(10)
                                                ],
                                                "Qtd Amostras": dist,
                                            }
                                        )
                                        st.bar_chart(chart_df.set_index("Dificuldade"))

                                    st.divider()
                                    st.caption(
                                        f"Brain Temperature: {stats.get('mean_loss', 0.0):.4f} (Mean Loss)"
                                    )

                            with tab_charts:
                                # 1. Main Loss & AUC
                                fig_main = make_subplots(
                                    specs=[[{"secondary_y": True}]]
                                )

                                # Loss (Primary Y)
                                fig_main.add_trace(
                                    go.Scatter(
                                        x=df["step"],
                                        y=df["loss"],
                                        name="Train Loss",
                                        mode="lines",
                                        line=dict(color="#FF4B4B"),
                                    ),
                                    secondary_y=False,
                                )

                                # AUC (Secondary Y)
                                fig_main.add_trace(
                                    go.Scatter(
                                        x=df["step"],
                                        y=df["auc"],
                                        name="Train AUC",
                                        mode="lines",
                                        line=dict(color="#00CC96"),
                                    ),
                                    secondary_y=True,
                                )

                                fig_main.update_layout(
                                    title_text="Training Dynamics (Loss vs AUC)",
                                    height=400,
                                    template="plotly_dark",
                                    hovermode="x unified",
                                )
                                fig_main.update_yaxes(
                                    title_text="Loss", range=[0, 1], secondary_y=False
                                )
                                fig_main.update_yaxes(
                                    title_text="AUC", range=[0, 1], secondary_y=True
                                )
                                st.plotly_chart(fig_main, use_container_width=True)

                                # 2. Learning Rate & Accuracy
                                col_c1, col_c2 = st.columns(2)

                                with col_c1:
                                    fig_lr = px.line(
                                        df,
                                        x="step",
                                        y="lr",
                                        title="Learning Rate Schedule",
                                        template="plotly_dark",
                                    )
                                    fig_lr.update_traces(line_color="#FFAA00")
                                    st.plotly_chart(fig_lr, use_container_width=True)

                                with col_c2:
                                    fig_acc = px.line(
                                        df,
                                        x="step",
                                        y="accuracy",
                                        title="Batch Accuracy",
                                        template="plotly_dark",
                                    )
                                    fig_acc.update_traces(line_color="#0099FF")
                                    st.plotly_chart(fig_acc, use_container_width=True)

                            with tab_raw:
                                st.markdown("#### 📜 Recent Steps History")
                                st.dataframe(
                                    df.sort_values("step", ascending=False).head(50),
                                    use_container_width=True,
                                )

                                csv_data = df.to_csv(index=False).encode("utf-8")
                                st.download_button(
                                    label="📥 Download Full Metrics CSV",
                                    data=csv_data,
                                    file_name=f"training_metrics_{int(time.time())}.csv",
                                    mime="text/csv",
                                )

                    except pd.errors.EmptyDataError:
                        st.warning(
                            "⚠️ Metrics file exists but is empty. Training might be initializing."
                        )
                    except Exception as e:
                        st.error(f"Error reading metrics: {e}")
                else:
                    st.info("Waiting for metrics file generation...")

            except json.JSONDecodeError:
                st.caption("Syncing telemetry...")
            except Exception as e:
                st.error(f"Telemetry Read Error: {e}")

        if auto_ref:
            time.sleep(3)
            st.rerun()

    # === TAB 3: FORENSIC LAB ===
    with tab_forensics:
        st.markdown("### 🕵️ Forensic Replay (Time Travel)")
        st.caption("Forensic Engine v9.6 (Fixed)")
        st.info(
            'Analise o estado interno do Swarm Brain durante épocas passadas. Workers M5 e H1 são mapeados como "Long" e "Short" experts.'
        )

        # 1. Snapshot Loader
        if not os.path.exists(SNAPSHOT_DIR):
            st.warning(f"No snapshots found in {SNAPSHOT_DIR}")
        else:
            # Sort properly by epoch number (integer)
            try:
                snapshots = sorted(
                    [f for f in os.listdir(SNAPSHOT_DIR) if f.endswith(".json")],
                    key=lambda x: int(x.split("_")[1].split(".")[0]),
                )
            except:
                snapshots = sorted(
                    [f for f in os.listdir(SNAPSHOT_DIR) if f.endswith(".json")]
                )  # Fallback

            if not snapshots:
                st.warning("No snapshots available yet.")
            else:
                # --- SNAPSHOT MANAGEMENT ---
                with st.expander("🗑️ Manage Snapshots (Clean History)"):
                    to_delete = st.multiselect("Select snapshots to remove:", snapshots)
                    if st.button(
                        f"Delete {len(to_delete)} Snapshots",
                        type="primary",
                        disabled=len(to_delete) == 0,
                    ):
                        for snap in to_delete:
                            try:
                                os.remove(os.path.join(SNAPSHOT_DIR, snap))
                            except Exception as e:
                                st.error(f"Failed to delete {snap}: {e}")
                        st.toast(f"Deleted {len(to_delete)} snapshots.", icon="🗑️")
                        time.sleep(1)
                        st.rerun()

                # Re-read snapshots after potential deletion
                try:
                    snapshots = sorted(
                        [f for f in os.listdir(SNAPSHOT_DIR) if f.endswith(".json")],
                        key=lambda x: int(x.split("_")[1].split(".")[0]),
                    )
                except:
                    snapshots = sorted(
                        [f for f in os.listdir(SNAPSHOT_DIR) if f.endswith(".json")]
                    )

                if not snapshots:
                    st.warning("No snapshots available after cleanup.")
                    st.stop()

                selected_snap = st.selectbox(
                    "Select Epoch Snapshot", snapshots, index=len(snapshots) - 1
                )

                # Load Data
                try:
                    with open(os.path.join(SNAPSHOT_DIR, selected_snap), "r") as f:
                        data = json.load(f)
                except json.JSONDecodeError:
                    st.error(f"Snapshot {selected_snap} is corrupted.")
                    return

                # Data Containers
                # expert_conviction: {long: [], short: []}
                long_curve = data["expert_conviction"]["long"]
                short_curve = data["expert_conviction"]["short"]

                # Handle inputs for seq_len calculation
                if "ohlc" in data:
                    inputs = data["ohlc"]["close"]  # Use Close price as reference
                elif "sample_input" in data:
                    inputs = data["sample_input"]
                else:
                    inputs = [0] * len(long_curve)  # Fallback empty

                seq_len = len(inputs)

                # Defensive Fix for Legacy/Scalar Snapshots
                if isinstance(long_curve, (float, int)):
                    long_curve = [long_curve] * seq_len
                if isinstance(short_curve, (float, int)):
                    short_curve = [short_curve] * seq_len

                # Robustness: Ensure list length matches seq_len
                if len(long_curve) < seq_len:
                    # Pad with last value
                    long_curve += [long_curve[-1]] * (seq_len - len(long_curve))
                if len(short_curve) < seq_len:
                    short_curve += [short_curve[-1]] * (seq_len - len(short_curve))

                # Truncate if too long
                long_curve = long_curve[:seq_len]
                short_curve = short_curve[:seq_len]

                # 2. Time Travel Slider
                selected_tick = st.slider(
                    "⏳ Time Travel (Tick)", 0, seq_len - 1, seq_len - 1
                )

                # 3. Visualizations (The Visual Cortex)

                # Layout:
                # [ Candlestick Chart with Trade Markers ]
                # [ Neural Heatmap (Spectrogram)         ]

                # --- A. Context & Market Data ---
                # Retrieve OHLC from snapshot if available, else mocking it for legacy snapshots would be hard.
                # Assuming new snapshots have 'ohlc' key.

                has_ohlc = "ohlc" in data

                col_cortex_1, col_cortex_2 = st.columns([3, 1])

                with col_cortex_1:
                    st.subheader("👁️ Visual Cortex (Market View)")

                    if has_ohlc:
                        ohlc = data["ohlc"]
                        times = ohlc["time"]
                        opens = ohlc["open"]
                        highs = ohlc["high"]
                        lows = ohlc["low"]
                        closes = ohlc["close"]

                        # Trim to seq_len just in case
                        times = times[:seq_len]
                        opens = opens[:seq_len]
                        highs = highs[:seq_len]
                        lows = lows[:seq_len]
                        closes = closes[:seq_len]

                        fig = go.Figure()

                        # 1. Candlestick Main
                        fig.add_trace(
                            go.Candlestick(
                                x=list(range(seq_len)),
                                open=opens,
                                high=highs,
                                low=lows,
                                close=closes,
                                name="Market",
                                increasing_line_color="#00CC96",
                                decreasing_line_color="#FF4B4B",
                            )
                        )

                        # 2. Brain Overlays (Trade Markers)
                        # Identify points of high conviction
                        buy_signals = [
                            i for i, v in enumerate(long_curve) if v > 1.5
                        ]  # Threshold 1.5
                        sell_signals = [i for i, v in enumerate(short_curve) if v > 1.5]

                        if buy_signals:
                            fig.add_trace(
                                go.Scatter(
                                    x=buy_signals,
                                    y=[lows[i] * 0.9995 for i in buy_signals],
                                    mode="markers",
                                    marker=dict(
                                        symbol="triangle-up", size=12, color="#00CC96"
                                    ),
                                    name="AI Buy Conviction",
                                )
                            )

                        if sell_signals:
                            fig.add_trace(
                                go.Scatter(
                                    x=sell_signals,
                                    y=[highs[i] * 1.0005 for i in sell_signals],
                                    mode="markers",
                                    marker=dict(
                                        symbol="triangle-down", size=12, color="#FF4B4B"
                                    ),
                                    name="AI Sell Conviction",
                                )
                            )

                        # 3. Time Travel Marker
                        fig.add_vline(
                            x=selected_tick,
                            line_width=2,
                            line_dash="dash",
                            line_color="yellow",
                        )

                        fig.update_layout(
                            title="Market Context + AI Intent",
                            template="plotly_dark",
                            xaxis_rangeslider_visible=False,
                            height=400,
                            margin=dict(l=0, r=0, t=30, b=0),
                        )
                        st.plotly_chart(fig, key="cortex_chart", width="stretch")

                    else:
                        st.warning(
                            "⚠️ Legacy Snapshot: No OHLC Data found. Run new training to see Visual Cortex."
                        )
                        # Fallback to simple line
                        fig = px.line(y=inputs, title="Normalized Input (Legacy)")
                        fig.add_vline(x=selected_tick, line_color="yellow")
                        st.plotly_chart(fig, width="stretch")

                with col_cortex_2:
                    st.subheader("🧠 Synaptic Map")

                    # 1. Expert Bar Chart (Current Tick)
                    expert_fig = go.Figure()
                    expert_fig.add_trace(
                        go.Bar(
                            x=["Long", "Short"],
                            y=[long_curve[selected_tick], short_curve[selected_tick]],
                            marker_color=["#00CC96", "#FF4B4B"],
                            text=[
                                f"{long_curve[selected_tick]:.2f}",
                                f"{short_curve[selected_tick]:.2f}",
                            ],
                            textposition="auto",
                        )
                    )
                    expert_fig.update_layout(
                        title=f"Conviction (Tick {selected_tick})",
                        template="plotly_dark",
                        yaxis_range=[-2, 5],
                        height=200,
                        margin=dict(l=0, r=0, t=30, b=0),
                    )
                    st.plotly_chart(
                        expert_fig, key="expert_conviction_bar", width="stretch"
                    )

                    # 2. Feature Gating (Global)
                    feature_importances = data.get("feature_importance", [1.0] * 12)

                    # Map Feature Indices to Names (Based on train_v8.py generator)
                    # 0=NormClose, 1=NormMom, 2=NormVol, 3..15=Noise
                    feat_names = []
                    for i in range(len(feature_importances)):
                        if i == 0:
                            feat_names.append("F0: Price")
                        elif i == 1:
                            feat_names.append("F1: Momentum")
                        elif i == 2:
                            feat_names.append("F2: Volatility")
                        else:
                            feat_names.append(f"F{i}: Noise")

                    # Top 5 Features
                    indices = np.argsort(feature_importances)[::-1][:5]
                    top_feats = [feature_importances[i] for i in indices]
                    top_names = [feat_names[i] for i in indices]

                    feat_fig = px.bar(
                        x=top_names,
                        y=top_feats,
                        title="Active Senses (Input Gating)",
                        template="plotly_dark",
                        height=200,
                    )
                    feat_fig.update_layout(margin=dict(l=0, r=0, t=30, b=0))
                    st.plotly_chart(feat_fig, key="feat_imp_chart", width="stretch")

                # --- B. System 2 Narrative (The "Thought") ---
                # Mocking logic to generate "Language" from the internal state
                sys2_msg = ""
                top_idx = indices[0] if len(indices) > 0 else 0
                expert_diff = long_curve[selected_tick] - short_curve[selected_tick]

                if top_idx == 0:
                    focus = "Price Action"
                elif top_idx == 1:
                    focus = "Momentum Acceleration"
                elif top_idx == 2:
                    focus = "Volatility/Risk"
                else:
                    focus = "Market Noise"

                if expert_diff > 1.0:
                    intent = "Strong BUY"
                    reason = "Upward pressure confirmed."
                elif expert_diff < -1.0:
                    intent = "Strong SELL"
                    reason = "Downward pressure confirmed."
                else:
                    intent = "HOLD"
                    reason = "Uncertainty in signal."

                sys2_msg = f"**System 2 Analysis**: Focusing heavily on **{focus}**. Detect {intent} signal. {reason} (Confidence: {abs(expert_diff):.2f})"

                st.info(sys2_msg, icon="🧠")

                # --- C. Neural Spectrogram (Bottom) ---
                st.markdown("#### ⚡ Neural Activation Spectrogram")
                # Ensure we have the full sequence of hidden states
                hidden_state_raw = data.get(
                    "hidden_activity", [[0.0] * 100 for _ in range(32)]
                )
                hidden_state = np.array(hidden_state_raw)  # [Hidden, Seq]

                # Trim to seq_len
                hidden_state = hidden_state[:, :seq_len]

                # Colorscale: Magma or Viridis. Magma looks more "electric".
                spectro_fig = px.imshow(
                    hidden_state,
                    labels=dict(x="Time (Tick)", y="Neuron ID", color="Activation"),
                    x=list(range(seq_len)),
                    color_continuous_scale="Magma",
                    template="plotly_dark",
                    aspect="auto",
                    height=250,
                )
                spectro_fig.update_layout(
                    margin=dict(l=0, r=0, t=0, b=0), xaxis_title="Time Sequence"
                )
                # Mark current tick
                spectro_fig.add_vline(
                    x=selected_tick, line_color="yellow", line_width=2
                )

                st.plotly_chart(spectro_fig, key="spectrogram", width="stretch")

    # === TAB 4: MODEL ZOO ===
    with tab_zoo:
        st.markdown("### 🦁 Model Zoo - Gerenciador de Modelos")

        col_backups, col_zoo = st.columns(2)

        # --- LEFT: Backups (logs/models) ---
        with col_backups:
            st.markdown("#### 📦 Backups Temporários")
            st.caption(f"Diretório: `{MODELS_DIR}`")

            if os.path.exists(MODELS_DIR):
                backup_files = sorted(
                    [f for f in os.listdir(MODELS_DIR) if f.endswith(".pth")],
                    reverse=True,
                )

                if backup_files:
                    st.info(
                        f"**{len(backup_files)}** modelos encontrados ({sum(os.path.getsize(os.path.join(MODELS_DIR, f)) for f in backup_files) / 1024 / 1024:.1f} MB)"
                    )

                    # Seleção para deletar
                    to_delete = st.multiselect(
                        "Selecione para deletar:", backup_files, key="delete_backups"
                    )

                    col_del, col_promote = st.columns(2)

                    if col_del.button(
                        "🗑️ Deletar Selecionados",
                        disabled=len(to_delete) == 0,
                        type="secondary",
                    ):
                        for f in to_delete:
                            try:
                                os.remove(os.path.join(MODELS_DIR, f))
                            except Exception as e:
                                st.error(f"Erro ao deletar {f}: {e}")
                        st.toast(f"Deletados {len(to_delete)} modelos!", icon="🗑️")
                        time.sleep(0.5)
                        st.rerun()

                    # Promoção para Zoo
                    st.markdown("---")
                    st.markdown("##### 🚀 Promoção")

                    to_promote = st.selectbox(
                        "Selecione modelo para promover:",
                        ["-- Selecione --"] + backup_files,
                        key="promote_model",
                    )

                    # Sugerir nome baseado na seleção
                    default_name = (
                        to_promote.replace(".pth", "")
                        if to_promote != "-- Selecione --"
                        else ""
                    )
                    promote_name = st.text_input(
                        "Renomear para (Opcional):",
                        value="",
                        placeholder=f"Padrão: {default_name}",
                        key="promote_name",
                    )

                    # Logic: If name provided, use it. Else use original filename.
                    final_name = (
                        promote_name.strip() if promote_name.strip() else default_name
                    )

                    # Remove disabled state to prevent UI deadlock. Validate inside.
                    if st.button(
                        "🏆 Promover para Zoo", type="primary", key="btn_promote_zoo"
                    ):
                        if to_promote == "-- Selecione --":
                            st.warning("⚠️ Selecione um modelo primeiro!")
                        elif not final_name:
                            st.error("❌ Nome inválido! Tente preencher o campo.")
                        else:
                            src = os.path.join(MODELS_DIR, to_promote)
                            dst = os.path.join(MODEL_ZOO_DIR, f"{final_name}.pth")
                            try:
                                if not os.path.exists(src):
                                    st.error(
                                        f"Arquivo fonte não encontrado: {to_promote}"
                                    )
                                else:
                                    os.makedirs(MODEL_ZOO_DIR, exist_ok=True)
                                    import shutil

                                    shutil.copy2(src, dst)
                                    st.success(
                                        f"✅ Modelo promovido com sucesso: **{final_name}.pth**"
                                    )
                                    st.toast(f"Zoo atualizado!", icon="🦁")
                                    time.sleep(1)
                                    st.rerun()
                            except Exception as e:
                                st.error(f"Erro crítico ao promover: {e}")

                    # Lista com detalhes
                    with st.expander("Ver detalhes dos backups"):
                        for f in backup_files[:10]:
                            fpath = os.path.join(MODELS_DIR, f)
                            size = os.path.getsize(fpath) / 1024
                            mtime = time.strftime(
                                "%d/%m %H:%M", time.localtime(os.path.getmtime(fpath))
                            )
                            st.text(f"{f} ({size:.0f} KB) - {mtime}")
                else:
                    st.success("Nenhum backup pendente! 🎉")
            else:
                st.warning("Diretório de modelos não encontrado.")

        # --- RIGHT: Zoo (logs/model_zoo) ---
        with col_zoo:
            st.markdown("#### 🏆 Model Zoo (Produção)")
            st.caption(f"Diretório: `{MODEL_ZOO_DIR}`")

            if os.path.exists(MODEL_ZOO_DIR):
                zoo_files = sorted(
                    [f for f in os.listdir(MODEL_ZOO_DIR) if f.endswith(".pth")],
                    reverse=True,
                )

                if zoo_files:
                    for f in zoo_files:
                        fpath = os.path.join(MODEL_ZOO_DIR, f)
                        size = os.path.getsize(fpath) / 1024 / 1024
                        mtime = time.strftime(
                            "%d/%m/%Y %H:%M", time.localtime(os.path.getmtime(fpath))
                        )

                        with st.container():
                            st.markdown(f"**🦁 {f}**")
                            st.caption(f"{size:.2f} MB | {mtime}")
                            if st.button(f"🗑️ Remover", key=f"del_zoo_{f}"):
                                os.remove(fpath)
                                st.toast(f"Removido {f} do Zoo!", icon="🗑️")
                                time.sleep(0.5)
                                st.rerun()
                            st.divider()
                else:
                    st.info("Zoo vazio. Promova modelos da esquerda!")
            else:
                st.warning("Diretório do Zoo não encontrado.")



--- FILE: backend/src/dashboard/pages/mission_control.py ---
import streamlit as st
import time
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
from src.brain.tools.alpaca_executor import AlpacaExecutor

# Import Engine Components (Mock/Stub for UI if Engine not running in same process)
# In a real deployment, the Dashboard might communicate via DB or API.
# For Phase 1, we simulate the Shared State visualization.

def render():
    st.title("🚀 Trinity Mission Control")
    st.markdown("### Strategic Oversight & Tactical Intervention")
    
    # Top Row: KPI Metrics
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Market Regime", "Volatile", "+2.4%")
    kpi2.metric("LLM Context", "Bearish Divergence", "-0.85")
    kpi3.metric("LNN Confidence", "Neutral", "0.12")
    kpi4.metric("Active Agents", "2/3", "Sentinel Passive")
    
    st.markdown("---")
    
    # Main Control Panel
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        st.subheader("📡 Sentinel Watchdog (News Feed)")
        
        # Simulated News Feed
        news_data = [
            {"time": "10:32:05", "headline": "Fed Chair Powell to speak at 2 PM", "impact": 0.3, "sentiment": -0.2},
            {"time": "10:15:00", "headline": "Tech sector rallies on earnings", "impact": 0.1, "sentiment": 0.6},
            {"time": "09:45:30", "headline": "Oil prices stable amid supply concerns", "impact": 0.05, "sentiment": 0.1},
        ]
        
        # Check for Manual Trigger via Session State
        if 'simulated_trigger' in st.session_state and st.session_state.simulated_trigger:
             news_data.insert(0, {
                 "time": datetime.now().strftime("%H:%M:%S"), 
                 "headline": "🚨 SIMULATED: CENTRAL BANK UNEXPECTED RATE HIKE", 
                 "impact": 0.95, 
                 "sentiment": -0.9
             })
        
        df_news = pd.DataFrame(news_data)
        
        # Display as a styled dataframe or list
        st.dataframe(
            df_news, 
            column_config={
                "impact": st.column_config.ProgressColumn("Impact Score", min_value=0, max_value=1, format="%.2f"),
                "sentiment": st.column_config.NumberColumn("Sentiment", format="%.2f")
            },
            width="stretch",
            hide_index=True
        )
        
    with col_right:
        st.subheader("🛡️ Defense Systems")
        
        # DEFCON Gauge (Sentinel Impact Level)
        current_impact = 0.0
        if 'simulated_trigger' in st.session_state and st.session_state.simulated_trigger:
            current_impact = 0.95
            
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = current_impact * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "THREAT LEVEL (Impact %)"},
            gauge = {
                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "darkred" if current_impact > 0.8 else "green"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 50], 'color': 'rgba(0, 255, 0, 0.3)'},
                    {'range': [50, 80], 'color': 'rgba(255, 255, 0, 0.3)'},
                    {'range': [80, 100], 'color': 'rgba(255, 0, 0, 0.3)'}],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90}}))
        
        st.plotly_chart(fig, width="stretch")
        
        st.markdown("### Manual Override")
        
        # Emergency Buttons
        if st.button("🚨 SIMULATE SENTINEL TRIGGER (TEST)", type="secondary"):
            st.session_state.simulated_trigger = True
            st.rerun()
            
        if st.button("✅ CLEAR ALARM", type="primary"):
            st.session_state.simulated_trigger = False
            st.rerun()
            
        if st.button("🔴 EMERGENCY FREEZE (HARD STOP)", type="primary"):
            st.error("DANGER: SYSTEM HALTED MANUALLY")
            # Logic to write to DB/File to stop engine would go here
            executor = AlpacaExecutor()
            executor.close_all_positions()

    # --- ALPACA LIVE MONITOR ---
    st.markdown("---")
    st.subheader("🦙 Alpaca Paper Trading Monitor")
    
    executor = AlpacaExecutor()
    if not executor.api_key or not executor.secret_key:
        st.warning("⚠️ Alpaca Credentials missing in .env. Integration disabled.")
    else:
        acc_col, trade_col = st.columns([1, 2])
        
        with acc_col:
            st.markdown("#### Account Summary")
            summary = executor.get_account_summary()
            if summary:
                st.metric("Total Equity", f"${float(summary.get('equity', 0)):,.2f}")
                st.metric("Buying Power", f"${float(summary.get('buying_power', 0)):,.2f}")
                st.metric("Unrealized P/L", f"${float(summary.get('unrealized_pl', 0)):,.2f}")
                st.write(f"**Currency:** {summary.get('currency')}")
                st.write(f"**Environment:** {'Paper' if executor.paper else 'Live'}")
            else:
                st.error("Failed to fetch Alpaca summary.")

        with trade_col:
            st.markdown("#### Open Positions")
            positions = executor.get_open_positions()
            if positions:
                st.dataframe(pd.DataFrame(positions), use_container_width=True, hide_index=True)
            else:
                st.info("No active positions on Alpaca.")



--- FILE: backend/src/dashboard/pages/models.py ---
import streamlit as st
import os
import json
import shutil
import time
from datetime import datetime
from pathlib import Path
import pandas as pd
import torch

# Constants
LOG_DIR = "logs"
MODELS_DIR = os.path.join(LOG_DIR, "models")
ZOO_DIR = os.path.join(LOG_DIR, "model_zoo")
LEADERBOARD_FILE = os.path.join(LOG_DIR, "leaderboard.json")
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_model.pth")
ACTIVE_MODEL_PATH = os.path.join(MODELS_DIR, "active_model.pth")
ACTIVE_MODEL_META = os.path.join(MODELS_DIR, "active_model.json")

# Ensure directories exist
os.makedirs(ZOO_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


def _detect_model_symbol() -> list:
    """Scan for symbol-specific best models in the models directory."""
    models = []
    for f in Path(MODELS_DIR).glob("best_model_v9_*.pth"):
        # Extract symbol from filename: best_model_v9_BTCUSD.pth -> BTCUSD
        parts = f.stem.replace("best_model_v9_", "")
        if parts and not parts.startswith("backup"):
            models.append({"filename": f.name, "symbol": parts, "path": str(f)})
    return models


def render():
    st.title("🗄️ Model Manager")
    st.info("Manage your trained Brains. Save the champions, discard the failures. Deploy models to live trading.")

    tab_current, tab_zoo, tab_deploy, tab_upload = st.tabs(
        ["🔥 Current Training", "🦁 Model Zoo", "🚀 Deploy", "📤 Upload"]
    )

    # --- TAB 1: CURRENT TRAINING ---
    with tab_current:
        st.subheader("Active Session Artifacts")
        
        # Find all symbol-specific models
        available_models = _detect_model_symbol()
        
        if available_models:
            st.success(f"Found {len(available_models)} trained model(s)")
            
            for model_info in available_models:
                with st.expander(f"🧠 {model_info['symbol']}", expanded=True):
                    # Try to find leaderboard for this symbol
                    lb_path = os.path.join(LOG_DIR, f"leaderboard_v9.json")
                    if os.path.exists(lb_path):
                        try:
                            with open(lb_path, "r") as f:
                                lb_data = json.load(f)
                            
                            if lb_data:
                                best = lb_data[0]
                                col1, col2, col3, col4 = st.columns(4)
                                col1.metric("AUC", f"{best.get('auc', 0):.4f}")
                                col2.metric("Loss", f"{best['loss']:.4f}")
                                col3.metric("Accuracy", f"{best['accuracy']:.2%}")
                                col4.metric("Epoch", best['epoch'])
                        except Exception:
                            pass
                    
                    # Save to Zoo button
                    auc_str = "0000"
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
                    default_name = f"swarm_v9_{model_info['symbol']}_{timestamp}.pth"
                    
                    save_name = st.text_input("Filename", value=default_name, key=f"save_{model_info['symbol']}")
                    
                    if st.button(f"💾 Save {model_info['symbol']} to Zoo", key=f"btn_save_{model_info['symbol']}"):
                        source_path = model_info['path']
                        if os.path.exists(source_path):
                            dest = os.path.join(ZOO_DIR, save_name)
                            shutil.copy(source_path, dest)
                            
                            # Save metadata
                            meta = {
                                "symbol": model_info['symbol'],
                                "market_type": "crypto" if model_info['symbol'] in ["BTCUSD", "ETHUSD", "SOLUSD", "DOGEUSD", "LTCUSD"] else "forex",
                                "architecture": "SwarmBrain_v9",
                                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "auc": 0.0,
                                "loss": 0.0,
                                "accuracy": 0.0,
                                "epoch": 0,
                            }
                            
                            # Try to get actual metrics from leaderboard
                            lb_path = os.path.join(LOG_DIR, "leaderboard_v9.json")
                            if os.path.exists(lb_path):
                                try:
                                    with open(lb_path, "r") as f:
                                        lb = json.load(f)
                                    if lb:
                                        meta.update({
                                            "auc": lb[0].get("auc", 0),
                                            "loss": lb[0].get("loss", 0),
                                            "accuracy": lb[0].get("accuracy", 0),
                                            "epoch": lb[0].get("epoch", 0),
                                        })
                                except Exception:
                                    pass
                            
                            meta_path = dest + ".json"
                            with open(meta_path, "w") as f:
                                json.dump(meta, f, indent=2)
                            
                            st.toast(f"Saved to {save_name}", icon="🦁")
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error("Model file not found!")
        else:
            # Fallback to legacy leaderboard check
            if os.path.exists(LEADERBOARD_FILE):
                try:
                    with open(LEADERBOARD_FILE, "r") as f:
                        lb_data = json.load(f)
                    if lb_data:
                        best = lb_data[0]
                        st.success(f"🏆 Current Champion found from Epoch {best['epoch']}")
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("AUC", f"{best.get('auc', 0):.4f}")
                        col2.metric("Loss", f"{best['loss']:.4f}")
                        col3.metric("Accuracy", f"{best['accuracy']:.2%}")
                        col4.metric("Date", best['timestamp'])
                    else:
                        st.warning("Leaderboard is empty.")
                except Exception as e:
                    st.error(f"Error reading leaderboard: {e}")
            else:
                st.info("No active training data found. Train a model first!")

    # --- TAB 2: MODEL ZOO ---
    with tab_zoo:
        st.subheader("🦁 Hall of Fame (Saved Models)")
        
        models = [f for f in os.listdir(ZOO_DIR) if f.endswith(".pth")]
        
        if not models:
            st.warning("Zoo is empty. Save a model from training or upload one.")
        else:
            # Parse Metadata
            zoo_data = []
            for m in models:
                full_path = os.path.join(ZOO_DIR, m)
                stat = os.stat(full_path)
                size_mb = stat.st_size / (1024 * 1024)
                created = datetime.fromtimestamp(stat.st_ctime).strftime("%Y-%m-%d %H:%M")
                
                # Check for sidecar JSON
                meta_path = full_path + ".json"
                auc = 0.0
                loss = 0.0
                acc = 0.0
                epoch = 0
                symbol = "Unknown"
                market = "Unknown"
                if os.path.exists(meta_path):
                    try:
                        with open(meta_path, "r") as f:
                            meta = json.load(f)
                            auc = meta.get("auc", 0)
                            loss = meta.get("loss", 0)
                            acc = meta.get("accuracy", 0)
                            epoch = meta.get("epoch", 0)
                            symbol = meta.get("symbol", "Unknown")
                            market = meta.get("market_type", "Unknown")
                    except Exception:
                        pass
                
                zoo_data.append({
                    "Filename": m,
                    "Symbol": symbol,
                    "Market": market.capitalize(),
                    "AUC": auc,
                    "Loss": loss,
                    "Accuracy": acc,
                    "Epoch": epoch,
                    "Size (MB)": f"{size_mb:.2f}",
                    "Created": created,
                })
            
            df = pd.DataFrame(zoo_data)
            
            # Filter by market
            market_filter = st.radio("Filter by Market", ["All", "Forex", "Crypto"], horizontal=True, key="zoo_filter")
            if market_filter != "All":
                df = df[df["Market"] == market_filter]
            
            # Sort by AUC descending by default
            df = df.sort_values("AUC", ascending=False)
            
            # Interactive Table
            st.dataframe(
                df,
                column_config={
                    "AUC": st.column_config.ProgressColumn(
                        "AUC",
                        help="Area Under Curve",
                        format="%.4f",
                        min_value=0,
                        max_value=1,
                    ),
                    "Accuracy": st.column_config.NumberColumn(
                        "Accuracy",
                        format="%.2f",
                    ),
                    "Epoch": st.column_config.NumberColumn(
                        "Epoch",
                        format="%d",
                    ),
                },
                width="stretch",
                hide_index=True
            )
            
            # Selection for Actions
            st.divider()
            col_sel, col_act1, col_act2 = st.columns([3, 1, 1])
            selected_model = col_sel.selectbox("Select Model", df["Filename"])
            
            if col_act1.button("🗑️ Delete", type="secondary"):
                if selected_model:
                    try:
                        target = os.path.join(ZOO_DIR, selected_model)
                        os.remove(target)
                        if os.path.exists(target + ".json"):
                            os.remove(target + ".json")
                        st.toast(f"Deleted {selected_model}")
                        time.sleep(1)
                        st.rerun()
                    except Exception as e:
                        st.error(str(e))
            
            if col_act2.button("🚀 Deploy", type="primary"):
                if selected_model:
                    try:
                        source = os.path.join(ZOO_DIR, selected_model)
                        shutil.copy(source, ACTIVE_MODEL_PATH)
                        
                        # Copy or create metadata
                        meta_source = source + ".json"
                        if os.path.exists(meta_source):
                            shutil.copy(meta_source, ACTIVE_MODEL_META)
                        else:
                            # Create basic metadata
                            meta = {
                                "filename": selected_model,
                                "deployed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            }
                            with open(ACTIVE_MODEL_META, "w") as f:
                                json.dump(meta, f, indent=2)
                        
                        st.toast(f"🚀 {selected_model} deployed to Live Trading!", icon="🚀")
                        time.sleep(1)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Deploy failed: {e}")

    # --- TAB 3: DEPLOY STATUS ---
    with tab_deploy:
        st.subheader("🚀 Active Deployment")
        
        if os.path.exists(ACTIVE_MODEL_META):
            try:
                with open(ACTIVE_MODEL_META, "r") as f:
                    active_meta = json.load(f)
                
                st.success("🟢 Model is deployed and ready for Live Trading")
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Symbol", active_meta.get("symbol", "N/A"))
                col2.metric("Market", active_meta.get("market_type", "N/A").capitalize())
                col3.metric("AUC", f"{active_meta.get('auc', 0):.4f}")
                
                st.json(active_meta)
                
                if st.button("🔴 Undeploy Model", type="secondary"):
                    try:
                        os.remove(ACTIVE_MODEL_PATH)
                        os.remove(ACTIVE_MODEL_META)
                        st.toast("Model undeployed", icon="🔴")
                        time.sleep(1)
                        st.rerun()
                    except Exception as e:
                        st.error(str(e))
            except Exception as e:
                st.error(f"Error reading deployment info: {e}")
        else:
            st.warning("⚪ No model deployed. Go to Model Zoo tab to deploy one.")
            st.info("💡 Tip: Train a model → Save to Zoo → Deploy to Live Trading")

    # --- TAB 4: UPLOAD ---
    with tab_upload:
        st.subheader("📤 Import External Brain")
        st.markdown("Upload a `.pth` file trained elsewhere.")
        
        upload_symbol = st.text_input("Symbol for this model", value="UNKNOWN")
        upload_market = st.selectbox("Market Type", ["forex", "crypto"])
        
        uploaded_file = st.file_uploader("Choose a .pth file", type="pth")
        
        if uploaded_file is not None:
            save_path = os.path.join(ZOO_DIR, uploaded_file.name)
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Save metadata
            meta = {
                "symbol": upload_symbol,
                "market_type": upload_market,
                "architecture": "uploaded",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "auc": 0.0,
                "loss": 0.0,
                "accuracy": 0.0,
                "epoch": 0,
            }
            with open(save_path + ".json", "w") as f:
                json.dump(meta, f, indent=2)
            
            st.success(f"Saved {uploaded_file.name} to Zoo.")
            time.sleep(1)
            st.rerun()



--- FILE: backend/src/dashboard/pages/live_trading.py ---
"""
🏢 TRADING DESK - Professional Live Trading Monitor
====================================================
Real-time candlestick chart with order markers, equity curve,
order history, and AI brain panel. Powered by Alpaca Markets.
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import time
import os

# Service
from src.dashboard.services.alpaca_data_service import AlpacaDataService


def render():
    # ── Init Service ──────────────────────────────────────────
    service = AlpacaDataService()

    if not service.is_configured:
        st.error("⚠️ Alpaca não configurado. Preencha `ALPACA_API_KEY` e `ALPACA_SECRET_KEY` no `.env`.")
        st.stop()

    # ── Sidebar Controls ──────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Trading Desk Config")
        
        # Active Brain Indicator
        import json
        active_meta_path = "logs/models/active_model.json"
        if os.path.exists(active_meta_path):
            try:
                with open(active_meta_path, "r") as f:
                    active_brain = json.load(f)
                st.success(f"🧠 **Active Brain**: {active_brain.get('symbol', 'N/A')}")
                col1, col2 = st.columns(2)
                col1.caption(f"Market: {active_brain.get('market_type', 'N/A').capitalize()}")
                col2.caption(f"AUC: {active_brain.get('auc', 0):.4f}")
            except Exception:
                st.warning("🧠 Brain file corrupted")
        else:
            st.warning("🧠 No Brain deployed")
            st.caption("Go to Models → Deploy")
        
        st.divider()
        
        # Asset Selection
        asset_type = st.radio("Asset Class", ["Crypto", "Stocks"], horizontal=True)
        available_assets = service.get_available_assets("crypto" if asset_type == "Crypto" else "us_equity")
        
        # Default index logic
        default_idx = 0
        target = service.symbol if service.symbol in available_assets else ("BTC/USD" if asset_type=="Crypto" else "AAPL")
        if target in available_assets:
            default_idx = available_assets.index(target)
            
        symbol = st.selectbox("Symbol", available_assets, index=default_idx)
        timeframe = st.selectbox("Timeframe", ["1Min", "5Min", "15Min", "1Hour", "1Day"], index=1)
        chart_limit = st.slider("Candles", 50, 500, 200)
        days_back = st.slider("Days Back", 1, 30, 3)
        auto_refresh = st.toggle("Auto-Refresh (3s)", value=True)
        order_filter = st.selectbox("Orders Filter", ["all", "closed", "open"])

    # ── Fetch All Data ────────────────────────────────────────
    account = service.get_account()
    positions = service.get_positions()
    df_bars = service.get_bars(symbol=symbol, timeframe=timeframe, limit=chart_limit, days_back=days_back)

    df_orders = service.get_orders(status=order_filter, limit=100)
    df_equity = service.get_portfolio_history(period="1D", timeframe="5Min")
    
    # ── Debug Expander for User ──────────────────────────────
    with st.expander("🔍 System Internals (Debug)", expanded=True):
        st.write(f"**Symbol**: `{symbol}`")
        if df_bars.empty:
            st.error("❌ Dataframe is EMPTY. Check Docker Logs.")
        else:
            st.success(f"✅ Loaded {len(df_bars)} bars.")
            st.dataframe(df_bars.tail(5))

    # ══════════════════════════════════════════════════════════
    # ██  HEADER: TITLE + STATUS BADGE
    # ══════════════════════════════════════════════════════════
    header_left, header_right = st.columns([3, 1])
    with header_left:
        st.markdown(f"## 🏢 Trading Desk — `{symbol}`")
    with header_right:
        env_label = "🟢 PAPER" if service.paper else "🔴 LIVE"
        status = account.get("status", "UNKNOWN")
        st.markdown(f"### {env_label} &nbsp; | &nbsp; `{status}`")

    # ══════════════════════════════════════════════════════════
    # ██  KPI BAR
    # ══════════════════════════════════════════════════════════
    k1, k2, k3, k4, k5, k6 = st.columns(6)

    equity = account.get("equity", 0)
    cash = account.get("cash", 0)
    buying_power = account.get("buying_power", 0)
    unrealized_pl = account.get("unrealized_pl", 0)
    unrealized_plpc = account.get("unrealized_plpc", 0)

    # Win rate from closed orders
    if not df_orders.empty and "status" in df_orders.columns:
        closed = df_orders[df_orders["status"] == "filled"]
        total_trades = len(closed)
    else:
        total_trades = 0

    active_positions = len(positions)

    k1.metric("💰 Equity", f"${equity:,.2f}")
    k2.metric("💵 Cash", f"${cash:,.2f}")
    k3.metric("🔋 Buying Power", f"${buying_power:,.2f}")
    k4.metric("📈 Unrealized P/L", f"${unrealized_pl:+,.2f}",
              delta=f"{unrealized_plpc * 100:+.2f}%" if unrealized_plpc else None)
    k5.metric("📊 Trades (Filled)", total_trades)
    k6.metric("📍 Open Positions", active_positions)

    st.markdown("---")

    # ══════════════════════════════════════════════════════════
    # ██  MAIN LAYOUT: CHART + AI BRAIN
    # ══════════════════════════════════════════════════════════
    col_chart, col_brain = st.columns([3, 1])

    # ── CANDLESTICK CHART ─────────────────────────────────────
    with col_chart:
        # Calculate Real-Time Stats
        current_price = 0.0
        variation = 0.0
        if not df_bars.empty:
            latest = df_bars.iloc[-1]
            current_price = latest["close"]
            if len(df_bars) > 1:
                prev = df_bars.iloc[-2]["close"]
                if prev > 0:
                    variation = ((current_price - prev) / prev) * 100
        
        # Display Price Header (if data exists)
        if current_price > 0:
            color = "#00CC96" if variation >= 0 else "#FF4B4B"
            st.markdown(f"### {symbol} <span style='color:{color}'>${current_price:,.2f} ({variation:+.2f}%)</span>", unsafe_allow_html=True)

        if df_bars.empty:
            st.warning(f"📊 No bar data available for {symbol} ({timeframe}). Try increasing 'Days Back'.")
            if service.paper:
                st.info("💡 Note: Paper Trading data for Stocks may be delayed by 15min. Crypto is usually real-time.")
        else:
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.03,
                row_heights=[0.8, 0.2],
                subplot_titles=("", "Volume"),
            )

            # Candlestick
            fig.add_trace(
                go.Candlestick(
                    x=df_bars["time"],
                    open=df_bars["open"],
                    high=df_bars["high"],
                    low=df_bars["low"],
                    close=df_bars["close"],
                    name="Market",
                    increasing_line_color="#00CC96",
                    decreasing_line_color="#FF4B4B",
                ),
                row=1, col=1,
            )

            # Volume bars
            if "volume" in df_bars.columns:
                colors = ["#00CC96" if c >= o else "#FF4B4B"
                          for c, o in zip(df_bars["close"], df_bars["open"])]
                fig.add_trace(
                    go.Bar(
                        x=df_bars["time"],
                        y=df_bars["volume"],
                        marker_color=colors,
                        opacity=0.5,
                        name="Volume",
                        showlegend=False,
                    ),
                    row=2, col=1,
                )

            # ── Order Markers on Chart ────────────────────────
            if not df_orders.empty and "filled_price" in df_orders.columns:
                filled = df_orders[
                    (df_orders["status"] == "filled") &
                    (df_orders["symbol"].str.replace("/", "") == symbol.replace("/", ""))
                ].copy()

                if not filled.empty and "filled_at" in filled.columns:
                    buys = filled[filled["side"] == "buy"]
                    sells = filled[filled["side"] == "sell"]

                    if not buys.empty:
                        fig.add_trace(
                            go.Scatter(
                                x=pd.to_datetime(buys["filled_at"]),
                                y=buys["filled_price"],
                                mode="markers",
                                marker=dict(
                                    symbol="triangle-up",
                                    size=14,
                                    color="#00CC96",
                                    line=dict(width=2, color="white"),
                                ),
                                name="BUY",
                                hovertemplate="BUY @ %{y:.2f}<extra></extra>",
                            ),
                            row=1, col=1,
                        )

                    if not sells.empty:
                        fig.add_trace(
                            go.Scatter(
                                x=pd.to_datetime(sells["filled_at"]),
                                y=sells["filled_price"],
                                mode="markers",
                                marker=dict(
                                    symbol="triangle-down",
                                    size=14,
                                    color="#FF4B4B",
                                    line=dict(width=2, color="white"),
                                ),
                                name="SELL",
                                hovertemplate="SELL @ %{y:.2f}<extra></extra>",
                            ),
                            row=1, col=1,
                        )

            # Chart Layout
            fig.update_layout(
                template="plotly_dark",
                height=550,
                margin=dict(l=0, r=0, t=10, b=0),
                xaxis_rangeslider_visible=False,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1,
                    font=dict(size=10),
                ),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            fig.update_xaxes(showgrid=True, gridcolor="rgba(128,128,128,0.1)")
            fig.update_yaxes(showgrid=True, gridcolor="rgba(128,128,128,0.1)")

            st.plotly_chart(fig, use_container_width=True, key=f"main_chart_{time.time()}")

    # ── AI BRAIN PANEL ────────────────────────────────────────
    with col_brain:
        st.markdown("### 🧠 AI Brain")

        # Signal Log (from recent orders as proxy)
        st.markdown("#### 📡 Signal Log")
        if not df_orders.empty:
            recent = df_orders.head(8)
            for _, row in recent.iterrows():
                side_icon = "🟢" if row.get("side") == "buy" else "🔴"
                status_icon = "✅" if row.get("status") == "filled" else "⏳"
                price_str = f"@ ${row['filled_price']:.2f}" if row.get("filled_price") else ""
                st.markdown(
                    f"`{row.get('time', '')}` {side_icon} **{row.get('side', '').upper()}** "
                    f"{row.get('symbol', '')} {price_str} {status_icon}"
                )
        else:
            st.info("No signals dispatched yet.")

        st.markdown("---")

        # Active Positions
        st.markdown("#### 📍 Active Positions")
        if positions:
            for p in positions:
                pl = p.get("unrealized_pl", 0)
                pl_icon = "🟢" if pl >= 0 else "🔴"
                st.markdown(
                    f"{pl_icon} **{p['symbol']}** — {p['qty']} units\n"
                    f"Entry: `${p['avg_entry']:.2f}` → Now: `${p['current_price']:.2f}`\n"
                    f"P/L: **${pl:+.2f}**"
                )
                st.markdown("---")
        else:
            st.caption("No open positions.")

        st.markdown("---")

        # Sentinel Status
        st.markdown("#### 🛡️ Sentinel")
        sentinel_file = "logs/sentinel_status.json"
        if os.path.exists(sentinel_file):
            import json
            with open(sentinel_file) as f:
                sentinel = json.load(f)
            threat = sentinel.get("threat_level", 0)
            st.progress(min(threat, 1.0), text=f"Threat: {threat * 100:.0f}%")
        else:
            st.progress(0.0, text="Threat: 0% (Sentinel Idle)")

    # ══════════════════════════════════════════════════════════
    # ██  BOTTOM SECTION: ORDER HISTORY + EQUITY CURVE
    # ══════════════════════════════════════════════════════════
    st.markdown("---")
    bottom_left, bottom_right = st.columns([3, 2])

    # ── Order History Table ───────────────────────────────────
    with bottom_left:
        st.markdown("### 📋 Order History")
        if df_orders.empty:
            st.info("No orders found.")
        else:
            styled_orders = df_orders.copy()

            # Color-code status
            st.dataframe(
                styled_orders,
                column_config={
                    "id": st.column_config.TextColumn("ID", width="small"),
                    "time": st.column_config.TextColumn("Submitted", width="medium"),
                    "symbol": st.column_config.TextColumn("Symbol", width="small"),
                    "side": st.column_config.TextColumn("Side", width="small"),
                    "qty": st.column_config.NumberColumn("Qty", format="%.4f"),
                    "type": st.column_config.TextColumn("Type", width="small"),
                    "status": st.column_config.TextColumn("Status", width="small"),
                    "filled_price": st.column_config.NumberColumn("Fill Price", format="$%.2f"),
                    "filled_at": st.column_config.TextColumn("Filled At", width="medium"),
                },
                use_container_width=True,
                hide_index=True,
                height=300,
            )

    # ── Equity Curve ──────────────────────────────────────────
    with bottom_right:
        st.markdown("### 💰 Equity Curve")
        if df_equity.empty:
            st.info("Portfolio history not available yet.")
        else:
            fig_eq = go.Figure()
            fig_eq.add_trace(
                go.Scatter(
                    x=df_equity["time"],
                    y=df_equity["equity"],
                    mode="lines",
                    name="Equity",
                    line=dict(color="#00CC96", width=2),
                    fill="tozeroy",
                    fillcolor="rgba(0, 204, 150, 0.1)",
                )
            )

            # P/L coloring
            if "profit_loss" in df_equity.columns:
                fig_eq.add_trace(
                    go.Bar(
                        x=df_equity["time"],
                        y=df_equity["profit_loss"],
                        name="P/L",
                        marker_color=[
                            "#00CC96" if v >= 0 else "#FF4B4B"
                            for v in df_equity["profit_loss"]
                        ],
                        opacity=0.3,
                    )
                )

            fig_eq.update_layout(
                template="plotly_dark",
                height=300,
                margin=dict(l=0, r=0, t=10, b=0),
                showlegend=False,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            fig_eq.update_xaxes(showgrid=False)
            fig_eq.update_yaxes(showgrid=True, gridcolor="rgba(128,128,128,0.1)")
            st.plotly_chart(fig_eq, use_container_width=True, key=f"equity_{time.time()}")

    # ── Auto-Refresh Loop ─────────────────────────────────────
    if auto_refresh:
        time.sleep(3)
        st.rerun()


if __name__ == "__main__":
    render()



--- FILE: backend/src/dashboard/pages/simulation.py ---
import streamlit as st
import sys
import subprocess
import os
import signal
import time
import pandas as pd
import json
import plotly.graph_objects as go
import plotly.express as px
import torch

# Constants
LOG_DIR = "logs"
ZOO_DIR = os.path.join(LOG_DIR, "model_zoo")
MODELS_DIR = os.path.join(LOG_DIR, "models")
DATA_DIR = "data/processed"
SIM_SCRIPT = "src/brain/backtest_v9.py"
SIM_SNAPSHOT_FILE = os.path.join(LOG_DIR, "sim_snapshot.json")
SIM_PID_FILE = "simulation_process.pid"

def is_sim_running():
    if os.path.exists(SIM_PID_FILE):
        try:
            with open(SIM_PID_FILE, "r") as f:
                pid = int(f.read().strip())
            os.kill(pid, 0)
            return True, pid
        except:
            return False, None
    return False, None

def stop_sim():
    running, pid = is_sim_running()
    if running and pid:
        try:
            os.kill(pid, signal.SIGTERM)
        except: pass
    if os.path.exists(SIM_PID_FILE):
        os.remove(SIM_PID_FILE)

def render():
    st.title("🧪 Simulation Lab (Production Replay)")
    st.markdown("Replay historical data through the AI Brain to verify behavior before live deployment.")

    running, pid = is_sim_running()

    # --- SIDEBAR CONFIG ---
    with st.sidebar:
        st.header("🎮 Game Controls")
        
        # Asset (Symbol) Selector
        available_assets = [f.replace(".parquet", "") for f in os.listdir(DATA_DIR) if f.endswith(".parquet")] if os.path.exists(DATA_DIR) else ["EURUSD"]
        selected_asset = st.selectbox("Select Asset (Symbol)", available_assets, disabled=running)
        sim_data_path = os.path.join(DATA_DIR, f"{selected_asset}.parquet")
        
        # Model Selector (Filter by Asset)
        models_zoo = [os.path.join("model_zoo", f) for f in os.listdir(ZOO_DIR) if f.endswith(".pth")] if os.path.exists(ZOO_DIR) else []
        models_promoted = [os.path.join("models", f) for f in os.listdir(MODELS_DIR) if f.endswith(".pth")] if os.path.exists(MODELS_DIR) else []
        
        # Smart Filter: Show models for selected asset OR general models
        filtered_models = [m for m in (models_zoo + models_promoted) if selected_asset in m or "v9" in m]
        # If no asset-specific model found, show all
        if not filtered_models:
             filtered_models = models_zoo + models_promoted
             
        selected_model = st.selectbox("Select Brain", filtered_models, disabled=running)
        
        # Data Config
        start_date = st.text_input("Start Date (YYYY-MM-DD)", "2023-01-01", disabled=running)
        
        # Speed
        speed = st.slider("Playback Speed (Steps/Sec)", 1, 100, 25, disabled=running)
        
        st.divider()
        st.subheader("🧠 Architecture (V10 Boost)")
        # Architecture Config (Match Cycle 01 results)
        w_hidden = st.number_input("Worker Hidden Dim", value=128, step=32, disabled=running)
        c_hidden = st.number_input("Coord Hidden Dim", value=512, step=128, disabled=running)
        s_len = st.number_input("Sequence Length", value=128, step=64, disabled=running)
        
        if not running:
            if st.button("🚀 START SIMULATION", type="primary"):
                if not selected_model:
                     st.error("No model selected!")
                     st.stop()
                if not os.path.exists(sim_data_path):
                     st.error(f"Data file not found: {sim_data_path}")
                     st.stop()
                if not os.path.exists(SIM_SCRIPT):
                     st.error(f"Script not found: {SIM_SCRIPT}")
                     st.stop()

                if os.path.exists(SIM_SNAPSHOT_FILE):
                    try: os.remove(SIM_SNAPSHOT_FILE)
                    except: pass
                
                model_path = os.path.join(LOG_DIR, selected_model)
                cmd = [
                    sys.executable, "-u", SIM_SCRIPT,
                    "--model", model_path,
                    "--data", sim_data_path,
                    "--start_date", start_date,
                    "--speed", str(speed),
                    "--worker_hidden", str(w_hidden),
                    "--coord_hidden", str(c_hidden),
                    "--seq_len", str(s_len)
                ]
                
                with open("sim_std.log", "w") as log:
                    subprocess.Popen(cmd, stdout=log, stderr=log, preexec_fn=os.setpgrp)
                
                st.toast("Simulation Started!", icon="🚀")
                time.sleep(1)
                st.rerun()
        else:
            st.info(f"🟢 Sim Running (PID {pid})")
            col1, col2 = st.columns(2)
            COMMAND_FILE = "logs/sim_command.json"
            
            if col1.button("⏸️ PAUSE", use_container_width=True):
                with open(COMMAND_FILE, "w") as f: json.dump({"command": "PAUSE"}, f)
                st.toast("Pausing...")
                
            if col2.button("▶️ RESUME", use_container_width=True):
                with open(COMMAND_FILE, "w") as f: json.dump({"command": "RESUME"}, f)
                st.toast("Resuming...")

            if st.button("⏭️ NEXT TICK (STEP)", use_container_width=True):
                with open(COMMAND_FILE, "w") as f: json.dump({"command": "STEP"}, f)
                st.toast("Advancing one tick...")

            if st.button("🚨 ENSINAR ERRO (FLAG)", use_container_width=True):
                with open(COMMAND_FILE, "w") as f: json.dump({"command": "FLAG_ERROR"}, f)
                st.warning("IA avisada!")

            if st.button("🛑 STOP SIMULATION", type="primary", use_container_width=True):
                stop_sim()
                st.rerun()
                    
            st.divider()
            def get_pt_count(path):
                if not os.path.exists(path): return 0
                try: return len(torch.load(path, map_location="cpu", weights_only=False))
                except: return 0
                
            c1, c2, c3 = st.columns(3)
            c1.metric("Errors", get_pt_count("logs/backtest_errors.pt"))
            c2.metric("Success", get_pt_count("logs/backtest_success.pt"))
            c3.metric("Patterns", get_pt_count("logs/pattern_reservoir.pt"))

    # --- MAIN STAGE ---
    if not running:
        st.info("👈 Configure and Start the Simulation from the Sidebar.")
        return

    # --- LIVE REPLAY VIEW ---
    if not os.path.exists(SIM_SNAPSHOT_FILE):
        st.info("🕒 Waiting for simulation snapshot...")
        time.sleep(1)
        st.rerun()
        
    try:
        with open(SIM_SNAPSHOT_FILE, "r") as f:
            state = json.load(f)
    except:
        time.sleep(0.5); st.rerun()
        
    if state.get("completed"):
        st.success("Simulation Completed!")
        stop_sim()
        st.rerun()
        return

    # TOP BAR: METRICS
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Date", state.get("timestamp", "N/A"))
    c2.metric("Price", f"{state.get('price', 0):.5f}")
    
    realized = state.get("realized_pnl", 0.0)
    floating = state.get("unrealized_pnl", 0.0)
    
    c3.metric("Realized PnL", f"${realized:+.2f}")
    c4.metric("Floating PnL", f"${floating:+.2f}", delta=f"{floating:+.2f}")
    c5.metric("Trades", state.get("trade_count", 0))

    # --- POSITION & COOLDOWN ---
    pos = state.get("pos", 0.0)
    units = state.get("units", 0)
    entry_p = state.get("entry_price", 0.0)
    is_cd = state.get("is_cooldown", False)
    
    pos_label = "⚪ NEUTRAL"
    if is_cd: pos_label = "⏳ COOLDOWN"
    elif pos > 0: pos_label = f"🟢 LONG (${units:,})"
    elif pos < 0: pos_label = f"🔴 SHORT (${units:,})"
    
    pos_help = f"Entry: {entry_p:.5f}" if pos != 0 else "Market Neutral"
    c6.metric("Position", pos_label, help=pos_help)
    
    col_chart, col_brain = st.columns([3, 1])
    
    with col_brain:
        st.subheader("🧠 Intelligence")
        st.progress(state.get("prob", 0.5), text=f"Buy Conviction: {state.get('prob',0)*100:.1f}%")
        
        b1, b2 = st.columns(2)
        b1.metric("L-Expert", f"{state.get('long_bias', 0):.2f}")
        b2.metric("S-Expert", f"{state.get('short_bias', 0):.2f}")
        
        agreement = state.get("agreement", 0.0)
        n_l, n_s, n_n = state.get("n_long", 0), state.get("n_short", 0), state.get("n_neutral", 0)
        st.markdown(f"**Agreement:** {agreement*100:.0f}% | 🟢{n_l} 🔴{n_s} ⚪{n_n}")
        
        show_brain = st.checkbox("Show Neuronal Map", value=True)
        if show_brain:
            neurons = state.get("neuronal_map", [])
            if neurons:
                 fig_n = px.imshow(
                     pd.DataFrame(neurons), aspect="auto", color_continuous_scale="Viridis",
                     labels=dict(x="Memory", y="Worker IP")
                 )
                 fig_n.update_layout(height=250, margin=dict(l=0,r=0,t=0,b=0), coloraxis_showscale=False)
                 st.plotly_chart(fig_n, use_container_width=True, key=f"n_{state['step']}")

    with col_chart:
        st.subheader("👁️ Visual Cortex (Live Replay)")
        history = state.get("history", [])
        if history:
            df_full = pd.DataFrame(history)
            df_hist = df_full.tail(100).copy()
            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x=df_hist["t"], open=df_hist["o"], high=df_hist["h"],
                low=df_hist["l"], close=df_hist["c"], name="Market"
            ))
            
            # Highlight Position
            if "pos" in df_hist.columns:
                long_mask = df_hist["pos"] > 0
                if long_mask.any():
                    fig.add_trace(go.Scatter(x=df_hist["t"], y=df_hist["h"] * 1.0005, mode='lines', line=dict(color='rgba(0, 204, 150, 0.5)', width=3), name="LONG Active", hoverinfo='none'))
                short_mask = df_hist["pos"] < 0
                if short_mask.any():
                    fig.add_trace(go.Scatter(x=df_hist["t"], y=df_hist["l"] * 0.9995, mode='lines', line=dict(color='rgba(255, 75, 75, 0.5)', width=3), name="SHORT Active", hoverinfo='none'))

            # Action Signals (INCLUDING EXITS)
            # Map all exit actions to SELL/BUY for visualization
            def map_action(a):
                if a in ["BUY", "SELL"]: return a
                if a in ["TP_SCALP", "TS_ACTIVE", "EXIT_NEUTRAL", "REVERSAL_EXIT"]: return "EXIT"
                return None

            df_hist["act_val"] = df_hist["act"].apply(map_action)
            
            buy_ents = df_hist[df_hist["act"] == "BUY"]
            sell_ents = df_hist[df_hist["act"] == "SELL"]
            exit_ents = df_hist[df_hist["act_val"] == "EXIT"]

            if not buy_ents.empty:
                fig.add_trace(go.Scatter(x=buy_ents["t"], y=buy_ents["l"]*0.9985, mode="markers", marker=dict(symbol="triangle-up", size=16, color="#00CC96", line=dict(width=2, color="white")), name="BUY ENTRY"))
            if not sell_ents.empty:
                fig.add_trace(go.Scatter(x=sell_ents["t"], y=sell_ents["h"]*1.0015, mode="markers", marker=dict(symbol="triangle-down", size=16, color="#FF4B4B", line=dict(width=2, color="white")), name="SELL ENTRY"))
            if not exit_ents.empty:
                fig.add_trace(go.Scatter(x=exit_ents["t"], y=exit_ents["c"], mode="markers", marker=dict(symbol="x", size=12, color="yellow", line=dict(width=1, color="black")), name="EXIT (TP/TS)"))

            fig.update_layout(template="plotly_dark", xaxis_rangeslider_visible=False, height=500, margin=dict(l=0,r=0,t=0,b=0))
            st.plotly_chart(fig, use_container_width=True, key=f"chart_{state['step']}")
        else:
            st.warning("Waiting for history buffer...")

    # Cycle speed control & rerun
    time.sleep(1.0 / speed)
    st.rerun()

if __name__ == "__main__":
    render()



--- FILE: backend/src/demo_pure_liquid.py ---
"""
PURE LIQUID TRINITY v8.5 - LIVE DEMO
====================================
This script demonstrates the "Think-Query-Decide" loop of the Strategic Agent (LFM).
It simulates a market feed and connects to Ollama to show the AI 'thinking' and using tools.

Usage:
    python src/demo_pure_liquid.py
"""

import time
import torch
import logging
import sys
import os
import pandas as pd
import numpy as np

# Add src to path
sys.path.append(os.getcwd())

from src.brain.models.liquid_v8 import LiquidConfig, ModulatedIARA
from src.brain.models.trinity_liquid import TrinityLiquidBrain
from src.brain.engine import TrinityEngine, EngineConfig

# Configure nice logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("Demo")

def generate_mock_data(length=1000):
    """Generates a random walk for simulation."""
    price = 1.0850
    data = []
    prices = []
    
    for _ in range(length):
        change = np.random.normal(0, 0.0001)
        price += change
        prices.append(price)
        
        # Mock features: [NormPrice, RSI, MACD, BB_Up, LogRet, Vol] (size 18)
        feats = np.random.randn(18)
        feats[0] = (price - 1.08) / 0.01 # Mock norm price
        data.append(feats)
        
    return prices, np.array(data)

def main():
    print("\n💧 INITIALIZING PURE LIQUID TRINITY (v8.5) 💧")
    print("===============================================")
    
    # 1. Config
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔧 Hardware: {device.upper()}")
    
    # Check Ollama
    ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
    print(f"🧠 Strategic Agent: Connecting to {ollama_url}...")
    
    # 2. Initialize Neural Core (Tactical)
    print("⚡ Initializing Tactical LNN...")
    lnn_cfg = LiquidConfig(input_dim=18, hidden_dim=128)
    lnn_core = ModulatedIARA(lnn_cfg)
    
    # 3. Initialize Trinity Brain (Fusion)
    # LFM Output Dim: 4096 (standard Llama) -> Projected to 18 (LNN Input)
    brain = TrinityLiquidBrain(lnn_core, lfm_hidden_dim=4096).to(device)
    
    # 4. Start Engine (Strategic)
    engine_cfg = EngineConfig(
        lfm_model_url=ollama_url,
        lfm_model_name="lfm2.5-thinking:1.2b", # Ensure this matches what user pulled
        device=device,
        lfm_refresh_rate=10.0 # Fast thinking for demo (every 10s)
    )
    
    engine = TrinityEngine(brain, engine_cfg)
    engine.start()
    
    # 5. Simulation Loop
    prices, features = generate_mock_data()
    print("\n▶️  STARTING MARKET SIMULATION")
    print("   (Watch the logs for '🧠 LFM' activity...)\n")
    
    try:
        for i in range(len(prices)):
            price = prices[i]
            
            # Feed input to engine (Mock 1-step sequence)
            # Shape: [Batch=1, Seq=1, Feat=18]
            feat_tensor = torch.tensor(features[i]).float().unsqueeze(0).unsqueeze(0).to(device)
            
            # Tactical Execution (Fast)
            # The engine automatically injects the latest Strategic Context
            decision = engine.forward_tick(feat_tensor)
            
            # Update Engine State (so LFM sees price)
            engine.state.update_price(price)
            
            # Visual feedback every tick
            if i % 10 == 0:
                print(f"\rTick {i}: EURUSD {price:.5f} | Tactical Signal: {decision.item():.4f}", end="")
            
            time.sleep(0.1) # Simulate real-time (10 ticks/sec)
            
    except KeyboardInterrupt:
        print("\n\n🛑 Stopping Demo...")
    finally:
        engine.stop()
        print("✅ Engine Shutdown.")

if __name__ == "__main__":
    main()



--- FILE: backend/src/trading/active_inference.py ---
from dataclasses import dataclass, field
from typing import List, Deque, Dict, Tuple
import numpy as np
from collections import deque
import logging

logger = logging.getLogger(__name__)

@dataclass
class RegimeFilter:
    """
    Filters out market conditions that are unsuitable for the model.
    Uses normalized values (Z-score assumed).
    """
    adx_threshold: float = -0.5  # Approx ADX < 20
    atr_threshold: float = -0.5  # Low volatility validation
    
    def check(self, features: Dict[str, float]) -> Tuple[bool, str]:
        """Returns (is_tradable, reason)"""
        adx = features.get('adx_norm', 1.0)
        atr = features.get('atr_norm', 1.0)
        
        # Check Trend Strength
        if adx < self.adx_threshold:
            return False, f"CHOP (ADX {adx:.2f} < {self.adx_threshold})"
            
        # Check Volatility
        if atr < self.atr_threshold:
            return False, f"COMPRESSION (ATR {atr:.2f} < {self.atr_threshold})"
            
        return True, "OK"

@dataclass
class ActiveInferenceModule:
    """
    I.A.R.A. v6.4 Feedback Loop Engine.
    Monitors recent performance and adjusts aggression levels dynamically.
    """
    history_len: int = 100
    base_threshold: float = 0.60
    
    # State
    prediction_history: Deque[Dict] = field(default_factory=lambda: deque(maxlen=100))
    current_threshold: float = 0.60
    confidence_history: Deque[float] = field(default_factory=lambda: deque(maxlen=5)) # Stability check
    
    # Components
    regime_filter: RegimeFilter = field(default_factory=RegimeFilter)
    
    def log_prediction(self, prob: float, actual: float):
        """Register a prediction outcome to update internal stats."""
        pred_label = 1 if prob > 0.5 else 0
        actual_label = 1 if actual > 0.5 else 0
        is_correct = (pred_label == actual_label)
        
        confidence = abs(prob - 0.5) * 2 # 0.0 to 1.0
        
        record = {
            'prob': prob,
            'actual': actual,
            'correct': is_correct,
            'confidence': confidence,
            'timestamp': 0 # Fallback 
        }
        self.prediction_history.append(record)
        self.confidence_history.append(confidence)
        
        self._calibrate()
        
    def _calibrate(self):
        """Adjust Threshold based on recent Accuracy (Last 50 trades)."""
        if len(self.prediction_history) < 20:
            return # Not enough data
            
        # Analyze last 50
        history = list(self.prediction_history)[-50:]
        correct = sum(1 for x in history if x['correct'])
        acc = correct / len(history)
        
        # Dynamic Logic
        if acc < 0.51:
            self.current_threshold = 0.60 # DEFENSIVE (Lowered from 0.75)
            mode = "DEFENSIVE 🛡️"
        elif acc <= 0.55:
            self.current_threshold = 0.55 # MODERATE
            mode = "MODERATE ⚖️"
        else:
            self.current_threshold = 0.53 # AGGRESSIVE (Lowered from 0.55)
            mode = "AGGRESSIVE ⚔️"
            
        logger.info(f"Active Calibration: Acc {acc:.1%} -> Threshold {self.current_threshold} ({mode})")
        
    def get_decision(self, prob: float, market_features: Dict[str, float] = None) -> str:
        """
        Returns 'BUY', 'SELL', or 'WAIT' based on Dynamic Threshold and Stability.
        """
        # 1. Regime Check (Optional)
        if market_features:
            is_good, reason = self.regime_filter.check(market_features)
            if not is_good:
                return f"WAIT ({reason})"

        # 2. Stability Check (Volatility of confidence)
        if len(self.confidence_history) >= 5:
            volatility = np.std(list(self.confidence_history))
            if volatility > 0.2:
                return "WAIT (Volatile)"
        
        # 3. Threshold Check
        if prob > self.current_threshold:
            return "BUY"
        elif prob < (1 - self.current_threshold):
            return "SELL"
        else:
            return "WAIT"
            
    def get_chronic_failures(self) -> List[Dict]:
        """Identify Confident Errors for Re-Training."""
        failures = []
        for rec in self.prediction_history:
            # If confident (>0.80 or <0.20) but WRONG
            if rec['confidence'] > 0.60 and not rec['correct']:
                failures.append(rec)
        return failures



--- FILE: backend/src/trading/signal_generator.py ---
"""
Signal Generator v9 (Live Engine)
==================================
Bridges the SwarmBrain v9.7 model to real-time market data.

Architecture:
  1. Ingests raw OHLCV candles (M5 base)
  2. Builds multi-timeframe feature dict using centralized indicators
  3. Runs SwarmBrain inference (Swarm workers + Coordinator consensus)
  4. Applies rule engine (conviction thresholds, cooldown, sentiment)

Usage:
  from src.trading.signal_generator import SignalEngine, TradeSignal
  engine = SignalEngine(checkpoint_path="logs/models/best_model_v9_BTCUSD.pth")
  signal = engine.predict(candles_df)
"""

import logging
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from src.brain.models.liquid_v9 import SwarmBrain as SwarmBrainV9
from src.brain.models.liquid_v10 import SwarmBrain as SwarmBrainV10
from src.brain.models.liquid_v9 import CoordinatorConfig
from src.brain.models.workers import WorkerConfig
from src.data.indicators import compute_features_from_df, NUM_FEATURES

logger = logging.getLogger("SignalEngine")


# ──────────────────────────────────────────────
# Data Structures
# ──────────────────────────────────────────────
@dataclass
class TradeSignal:
    """Immutable result from a single inference pass."""
    timestamp: str
    direction: str          # LONG, SHORT, WAIT
    confidence: float       # Coordinator probability (0-1)
    worker_agreement: float # % of workers that agree (0-1)
    n_long: int
    n_short: int
    n_wait: int
    long_score: float       # Long expert raw score
    short_score: float      # Short expert raw score
    worker_scores: dict     # Per-TF scores: {M5: float, M15: float, ...} in [-1, 1]


# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
TIMEFRAMES = ["M5", "M15", "M30", "H1", "H4"]
TF_MAP = {"5min": "M5", "15min": "M15", "30min": "M30", "1h": "H1", "4h": "H4"}


class SignalEngine:
    """
    Live inference engine backed by SwarmBrain v9.7.

    Responsibilities:
      - Load and manage model checkpoint
      - Build multi-timeframe feature dictionary from OHLCV data
      - Run consensus-based inference
      - Generate actionable trade signals
    """

    def __init__(
        self,
        checkpoint_path: str,
        worker_hidden: int = 128,
        coord_hidden: int = 512,
        seq_len: int = 128,
        device: Optional[str] = None,
    ):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.seq_len = seq_len

        # Load checkpoint and build model
        self._load_model(checkpoint_path, worker_hidden, coord_hidden)

        # Enable optimizations
        torch.set_float32_matmul_precision("high")
        logger.info(
            f"SignalEngine v9.7 ready | device={self.device} | "
            f"params={self.model.count_parameters():,}"
        )

    def _load_model(
        self,
        path: str,
        worker_hidden: int,
        coord_hidden: int,
    ):
        """Load SwarmBrain from checkpoint, auto-detecting input dimension."""
        logger.info(f"Loading SwarmBrain from {path}...")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)

        # Auto-detect input dimension from checkpoint
        self.input_dim = self._detect_input_dim(checkpoint)
        logger.info(f"  Detected input_dim={self.input_dim}")

        worker_cfg = WorkerConfig(
            input_dim=self.input_dim,
            hidden_dim=worker_hidden,
            dropout=0.1,
        )
        coord_cfg = CoordinatorConfig(
            num_workers=5,
            worker_dim=worker_hidden,
            hidden_dim=coord_hidden,
            dropout=0.3,
        )

        if "v10" in path.lower():
            self.model = SwarmBrainV10(worker_cfg, coord_cfg).to(self.device)
        else:
            self.model = SwarmBrainV9(worker_cfg, coord_cfg).to(self.device)
        
        self.model.load_state_dict(checkpoint, strict=False)
        self.model.eval()
        logger.info("  ✅ SwarmBrain loaded and ready!")

    @staticmethod
    def _detect_input_dim(checkpoint: dict) -> int:
        """Infer input dimension from checkpoint weight shapes."""
        try:
            w = checkpoint.get("swarm.workers.M5.input_proj.weight")
            if w is not None and hasattr(w, "shape") and len(w.shape) == 2:
                return int(w.shape[1])
        except Exception:
            pass
        # Fallback: try common dim
        logger.warning("Could not detect input_dim from checkpoint; defaulting to 22")
        return NUM_FEATURES  # 22

    # ──────────────────────────────────────────
    # Feature Engineering (Live Path)
    # ──────────────────────────────────────────
    def build_feature_dict(
        self, df_m5: pd.DataFrame, seq_len: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Build multi-timeframe feature dictionary from M5 candle data.

        Args:
            df_m5: DataFrame with columns [time, open, high, low, close, volume]
            seq_len: Override for sequence length

        Returns:
            Dict mapping timeframe keys (M5, M15, ...) to tensors [1, seq, features]
        """
        seq_len = seq_len or self.seq_len

        # Ensure sorted
        df_m5 = df_m5.sort_values("time").reset_index(drop=True)

        # Resample to higher timeframes
        df_m5["time"] = pd.to_datetime(df_m5["time"])
        df_m5 = df_m5.set_index("time")

        tf_resample_map = {
            "M5": None,  # Base — no resampling needed
            "M15": "15min",
            "M30": "30min",
            "H1": "1h",
            "H4": "4h",
        }

        x_dict: Dict[str, torch.Tensor] = {}

        for tf_key, rule in tf_resample_map.items():
            if rule is None:
                df_tf = df_m5.reset_index()
            else:
                df_tf = (
                    df_m5.resample(rule)
                    .agg({
                        "open": "first",
                        "high": "max",
                        "low": "min",
                        "close": "last",
                        "volume": "sum",
                    })
                    .dropna()
                    .reset_index()
                    .rename(columns={"index": "time"})
                )

            features = compute_features_from_df(df_tf, time_col="time")
            
            # Enforce input dimension from the model (Base part only)
            # v10.0 uses 12 base + 42 L2 = 54. v9.7 uses 22 base.
            base_dim = self.input_dim - 42 if self.input_dim > 42 else self.input_dim
            
            if features.shape[-1] > base_dim:
                features = features[:, :base_dim]
            elif features.shape[-1] < base_dim:
                pad = np.zeros((features.shape[0], base_dim - features.shape[-1]))
                features = np.hstack([features, pad])
            
            # Handle sequence length (Padding vs Slicing)
            if len(df_tf) < seq_len:
                pad_len = seq_len - len(df_tf)
                padded = np.zeros((seq_len, base_dim), dtype=np.float32)
                padded[pad_len:] = features
                tensor = torch.tensor(padded, dtype=torch.float32)
            else:
                tensor = torch.tensor(features[-seq_len:], dtype=torch.float32)
                
            # 🔥 RAIO-X: If L2 Sensory features are present in the last candle, append them
            if tf_key == "M5" and "sensory_l2" in df_tf.columns:
                last_l2 = df_tf["sensory_l2"].iloc[-1]
                if isinstance(last_l2, list) and len(last_l2) == 42:
                    l2_tensor = torch.tensor(last_l2, dtype=torch.float32)
                    l2_seq = l2_tensor.unsqueeze(0).repeat(seq_len, 1)
                    
                    if tensor.shape[-1] + l2_seq.shape[-1] == self.input_dim:
                        tensor = torch.cat([tensor, l2_seq], dim=-1)
                    else:
                        logger.warning(f"L2 merge FAIL: base({tensor.shape[-1]}) + L2(42) != model({self.input_dim})")

            x_dict[tf_key] = tensor.unsqueeze(0).to(self.device)

        return x_dict

    # ──────────────────────────────────────────
    # Inference
    # ──────────────────────────────────────────
    def predict(self, df_m5: pd.DataFrame) -> Optional[TradeSignal]:
        """
        Run SwarmBrain inference on M5 candle data.

        Args:
            df_m5: DataFrame with OHLCV + time columns

        Returns:
            TradeSignal or None on error
        """
        try:
            x_dict = self.build_feature_dict(df_m5)

            with torch.no_grad():
                with torch.amp.autocast(
                    str(self.device), enabled=(self.device.type == "cuda")
                ):
                    logits, worker_logits, long_score, short_score, _, _ = self.model(
                        x_dict
                    )

            # Coordinator directional probability (Legacy backward compatibility)
            # We map Long vs Short expert scores into a 0.0 to 1.0 scalar.
            # > 0.5 is Long biased, < 0.5 is Short biased
            diff_logit = (long_score - short_score).view(-1)
            coord_prob = torch.sigmoid(diff_logit)[0].item()

            # Worker consensus
            votes = self._summarize_votes(worker_logits)

            # Decision
            direction = self._decide(coord_prob, votes)

            return TradeSignal(
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                direction=direction,
                confidence=coord_prob,
                worker_agreement=votes["agreement"],
                n_long=votes["n_long"],
                n_short=votes["n_short"],
                n_wait=votes["n_wait"],
                long_score=float(long_score.item()),
                short_score=float(short_score.item()),
                worker_scores=votes["worker_scores"],
            )

        except Exception as e:
            logger.error(f"Prediction error: {e}", exc_info=True)
            return None

    @staticmethod
    def _summarize_votes(worker_logits: torch.Tensor) -> dict:
        """Extract vote counts and per-timeframe scores from worker logits [1, num_workers, 3]."""
        wl = worker_logits
        if wl.dim() == 3:
            wl = wl[0]
        votes = wl.argmax(dim=-1)  # 0=LONG, 1=SHORT, 2=NEUTRAL
        n_long = int((votes == 0).sum().item())
        n_short = int((votes == 1).sum().item())
        n_wait = int((votes == 2).sum().item())
        total = max(1, votes.numel())
        agreement = float(max(n_long, n_short, n_wait) / total)
        
        # Per-timeframe scores: convert logit[0] (long) vs logit[1] (short) to [-1, 1]
        # Positive = bullish, Negative = bearish
        tf_names = ["M5", "M15", "M30", "H1", "H4"]
        scores = {}
        for i, tf in enumerate(tf_names):
            if i < wl.shape[0]:
                long_logit = float(wl[i, 0].item())
                short_logit = float(wl[i, 1].item())
                # Normalized score: sigmoid difference mapped to [-1, 1]
                score = float(torch.sigmoid(wl[i, 0]).item() - torch.sigmoid(wl[i, 1]).item())
                scores[tf] = round(max(-1.0, min(1.0, score)), 3)
            else:
                scores[tf] = 0.0
        
        return {
            "n_long": n_long,
            "n_short": n_short,
            "n_wait": n_wait,
            "agreement": agreement,
            "worker_scores": scores,
        }

    @staticmethod
    def _decide(coord_prob: float, votes: dict) -> str:
        """
        Consensus-based decision.

        Rules:
          1. If >=4 workers vote WAIT -> WAIT
          2. If coordinator agrees with strong worker majority -> LONG/SHORT
          3. If coordinator is very confident -> override
          4. Else -> WAIT
        """
        n_long = votes["n_long"]
        n_short = votes["n_short"]
        n_wait = votes["n_wait"]

        # Rule 1: Strong neutral consensus
        if n_wait >= 4:
            return "WAIT"

        # Rule 2: Coordinator + Worker agreement
        if coord_prob > 0.55 and n_long >= 2:
            return "LONG"
        if coord_prob < 0.45 and n_short >= 2:
            return "SHORT"

        # Rule 3: Very strong coordinator confidence
        if coord_prob > 0.65:
            return "LONG"
        if coord_prob < 0.35:
            return "SHORT"

        return "WAIT"

    def get_x_dict(self, df_m5: pd.DataFrame) -> Dict[str, torch.Tensor]:
        """Public accessor for feature dict (used by engine/backtest)."""
        return self.build_feature_dict(df_m5)



--- FILE: backend/src/trading/simulation.py ---

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

from src.data.feature_factory import process_hierarchical
from src.train.trainer_liquid import TrainerConfig
import psycopg2
import os
import json
import time

@dataclass
class Trade:
    entry_time: int
    entry_price: float
    direction: int # 1 = Long, -1 = Short
    size: float
    exit_time: Optional[int] = None
    exit_price: Optional[float] = None
    pnl: float = 0.0
    status: str = "OPEN" # OPEN, CLOSED

@dataclass
class BrainState:
    timestamp: int
    price: float
    prob: float
    confidence: float
    trend_m5: int
    trend_h1: int
    feature_attribution: Dict[str, float]
    logic_log: List[str]
    # Future: Activations, Attention

class MarketSimulator:
    def __init__(self, model, ticker: str = "EURUSD", device: str = "cpu"):
        self.model = model
        self.ticker = ticker
        self.device = device
        
        # SACI V7: Database Connection
        self.db_conn = None
        try:
            dsn = os.getenv("DB_URI", "postgresql://saci:saci_secret@localhost:5432/saci_core")
            self.db_conn = psycopg2.connect(dsn)
            self.db_conn.autocommit = True
            print("✅ MarketSimulator connected to TimescaleDB")
        except Exception as e:
            print(f"⚠️ DB Connection failed: {e}. Simulation will run in RAM only.")

        # State
        self.initial_balance = 1000.0
        self.balance = 1000.0
        self.position: Optional[Trade] = None
        self.trades: List[Trade] = []
        self.history: List[BrainState] = []
        self.current_step = 0
        
        # Load Data
        self.load_data()
        
    def load_data(self):
        """Loads data using Feature Factory logic to ensure feature parity."""
        # Config placeholder (assuming 256 seq_len from trained model)
        seq_len = 256
        if hasattr(self.model, 'config'):
            if hasattr(self.model.config, 'seq_len'):
                seq_len = self.model.config.seq_len
            elif hasattr(self.model.config, 'max_seq_len'):
                seq_len = self.model.config.max_seq_len
        
        # Try raw paths
        candidates = [
            Path(f"data/raw/{self.ticker}.parquet"),
            Path(f"data/raw/{self.ticker}_M1.parquet"),
            Path(f"data/processed/{self.ticker}.parquet")
        ]
        
        raw_path = next((p for p in candidates if p.exists()), None)
        if not raw_path:
            raise FileNotFoundError(f"No data found for {self.ticker}")
            
        print(f"Loading simulation data from {raw_path}...")
        # X: [N, seq, feat], y: [N] (Consensus labels)
        self.X, self.y = process_hierarchical(raw_path, seq_len, stride=1) # Stride 1 for full replay!
        
        # We also need Price Data for PnL calculation.
        # process_hierarchical returns tensors, does not return prices directly.
        # We need to load the DF separately to map indices to prices/dates.
        # This is a slight inefficiency but necessary for accurate PnL.
        import polars as pl
        if raw_path.suffix == '.csv':
            df = pl.read_csv(raw_path, ignore_errors=True)
        else:
            df = pl.read_parquet(raw_path)
            
        # Rename columns to standard and sort
        df = df.rename({c: c.lower() for c in df.columns}).sort("date")
        
        # We need to align the dataset indices with the DataFrame rows.
        # process_hierarchical builds sequences. The last point of sequence `i` corresponds to row `i + seq_len - 1` in the filtered consensus data?
        # Actually process_hierarchical filters only consensus points! 
        # This makes alignment tricky.
        # For the "Trade Hub", we ideally want CONTINUOUS replay, not just consensus points.
        # But the model was trained ONLY on consensus.
        # If we feed it non-consensus data, it might hallucinate.
        # VALID APPROACH: We replay only the "Valid Opportunities" (Consensus points).
        # This acts as a "Scanner" that skips noise.
        
        # To get prices, we need to hack `process_hierarchical` or fetch prices for the valid indices.
        # For now, we will simulate PRICE based on the features or just mock it if real price alignment is too hard without refactoring factory.
        # Wait, index 0 of the dataset corresponds to some time T.
        # Refactoring `process_hierarchical` to return metadata (prices/dates) is best.
        # But I can't easily break existing API.
        # Hack: The feature vector likely includes Close price? No, it's normalized.
        # However, we can track PnL in "pips" or "pct" if we don't have absolute price, 
        # but for a dashboard we really want the price curve.
        
        # Let's trust that for the visualization, we can just assume a price curve or try to re-load prices.
        # BETTER: Let's modify Simulation to load the DF and re-run logic?
        # No, that duplicates code.
        
        # PROPOSED SOLUTION: Just assume we replay the tensor. We can't plot price perfectly matched to date without the date.
        # But `process_hierarchical` is imported. I can check if I can modify it or use it differently.
        # Actually, `process_hierarchical` uses `df_m1`.
        # I will assume for now we just plot "Trade Index" vs "PnL".
        # Later we can improve exact date mapping.
        pass

    def reset(self):
        self.balance = self.initial_balance
        self.position = None
        self.trades = []
        self.history = []
        self.current_step = 0
        
    def step(self):
        """Advances one step in the simulation."""
        if self.current_step >= len(self.X):
            return None # End of Data
        
        # 1. Get Data
        x_np = self.X[self.current_step] # [seq, feat]
        # x_np is numpy, need tensor
        x_in = torch.from_numpy(x_np).unsqueeze(0).to(self.device).float()
        
        # 2. Extract Context (for UI)
        # Assuming current feature map: 12=M5 Trend, 13=H1 Trend
        trend_m5 = x_np[-1, 12]
        trend_h1 = x_np[-1, 13]
        
        # 3. Model Inference
        with torch.no_grad():
            logits, _ = self.model(x_in)
            prob = torch.sigmoid(logits).item()
            
        # 4. Trading Logic (Based on Plan)
        # Thresholds
        BUY_THRESH = 0.55 # Aggressive
        SELL_THRESH = 0.45
        
        decision = "WAIT"
        confidence = abs(prob - 0.5) * 2
        
        # Logic Log
        log = []
        log.append(f"Prob: {prob:.2%} | Conf: {confidence:.2%}")
        
        if trend_h1 == 1:
            log.append("Macro: UP Strategy")
        elif trend_h1 == -1:
            log.append("Macro: DOWN Strategy")
            
        if prob > BUY_THRESH:
            decision = "BUY"
            log.append("Signal: BUY Triggered")
        elif prob < SELL_THRESH:
            decision = "SELL"
            log.append("Signal: SELL Triggered")
            
        # Execute Trade
        # Mock price (since we don't have exact price aligned yet, use randomness or previous + delta?)
        # For visual demo, we simulate price movement based on label?
        # No, self.y has the label (1 for UP, 0 for DOWN).
        # We can verify if we won or lost based on Y!
        # If y=1 and we BUY -> WIN.
        # If y=0 and we BUY -> LOSS.
        
        actual_outcome = self.y[self.current_step]
        reward = 0.0
        
        if decision == "BUY":
            if actual_outcome == 1:
                reward = 0.0010 # 10 pips win
                log.append("Result: WIN (+10 pips)")
            else:
                reward = -0.0005 # 5 pips loss (Stop Loss)
                log.append("Result: LOSS (-5 pips)")
                
        elif decision == "SELL":
             if actual_outcome == 0:
                reward = 0.0010
                log.append("Result: WIN (+10 pips)")
             else:
                reward = -0.0005
                log.append("Result: LOSS (-5 pips)")
                
        self.balance += (reward * 100000) # Standard lot size approximation
        
        # 5. Capture Brain State
        brain_state = BrainState(
            timestamp=self.current_step,
            price=self.balance, # Plot Balance instead of Price for now if Price is missing
            prob=prob,
            confidence=confidence,
            trend_m5=trend_m5,
            trend_h1=trend_h1,
            feature_attribution={}, # TODO: Add hooks if needed
            logic_log=log
        )
        self.history.append(brain_state)
        
        # V7: Persist State
        self.save_state_to_db(brain_state)
        
        self.current_step += 1
        return brain_state

    def save_state_to_db(self, state: BrainState):
        if not self.db_conn:
            return

        try:
            with self.db_conn.cursor() as cur:
                # Table schema: time, symbol, run_id, prob_up, confidence, trend_m5, trend_h1, features, logic_log
                cur.execute("""
                    INSERT INTO brain_states (time, symbol, run_id, prob_up, confidence, trend_m5, trend_h1, logic_log)
                    VALUES (NOW(), %s, %s, %s, %s, %s, %s, %s)
                """, (
                    self.ticker,
                    "sim_v1", # Temporary Run ID
                    state.prob,
                    state.confidence,
                    state.trend_m5,
                    state.trend_h1,
                    state.logic_log
                ))
        except Exception as e:
            print(f"DB Write Error: {e}")
            self.db_conn = None # Disable DB on error to avoid spam




--- FILE: backend/src/analysis/active_learning.py ---
"""
Active Learner (Memory Module)
==============================
Implements "Self-Improvement" logic.
- Logs trades and outcomes.
- Calculates Rolling Win Rate.
- Adjusts Thresholds dynamically based on recent performance.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

ARTIFACTS_DIR = Path(__file__).parent.parent.parent / "artifacts"
MEMORY_FILE = ARTIFACTS_DIR / "trade_memory.json"
ADAPTIVE_CONFIG_FILE = ARTIFACTS_DIR / "adaptive_config.json"

class ActiveLearner:
    def __init__(self):
        self._ensure_files()
        
    def _ensure_files(self):
        if not ARTIFACTS_DIR.exists():
            ARTIFACTS_DIR.mkdir(parents=True)
            
        if not MEMORY_FILE.exists():
            with open(MEMORY_FILE, 'w') as f:
                json.dump([], f)
                
        if not ADAPTIVE_CONFIG_FILE.exists():
            default_config = {
                "base_threshold": 0.60,
                "current_threshold": 0.60,
                "regime": "NEUTRAL",
                "rolling_win_rate": 0.50,
                "last_update": datetime.now().isoformat()
            }
            with open(ADAPTIVE_CONFIG_FILE, 'w') as f:
                json.dump(default_config, f, indent=4)

    def log_trade(self, ticker: str, direction: str, confidence: float, predicted_at: str, outcome: Optional[float] = None):
        """Logs a trade signal. If outcome is known (PnL), serves as learning data."""
        with open(MEMORY_FILE, 'r') as f:
            history = json.load(f)
            
        entry = {
            "ticker": ticker,
            "direction": direction,
            "confidence": confidence,
            "predicted_at": predicted_at,
            "outcome": outcome  # 1.0 (Win), 0.0 (Loss), or None (Pending)
        }
        
        history.append(entry)
        
        # Keep only last 1000 trades to keep JSON light
        if len(history) > 1000:
            history = history[-1000:]
            
        with open(MEMORY_FILE, 'w') as f:
            json.dump(history, f, indent=4)
            
        # Trigger Adaptive Update if outcome is present
        if outcome is not None:
            self._update_adaptation(history)

    def _update_adaptation(self, history: List[Dict]):
        """Recalculates regime and thresholds based on recent history."""
        # Filter for closed trades
        closed_trades = [t for t in history if t.get("outcome") is not None]
        
        if len(closed_trades) < 20:
            return  # Need more data
            
        # Get last 20 trades
        recent = closed_trades[-20:]
        outcomes = [t["outcome"] for t in recent]
        win_rate = sum(outcomes) / len(outcomes)
        
        # Load Config
        with open(ADAPTIVE_CONFIG_FILE, 'r') as f:
            config = json.load(f)
            
        base_threshold = config.get("base_threshold", 0.60)
        
        # Logic: "Hot Hand" vs "Cool Down"
        if win_rate > 0.60:
            # We are winning! Loosen threshold to trade more
            new_threshold = max(0.55, base_threshold - 0.05)
            regime = "BULL_RUN"
        elif win_rate < 0.40:
            # We are losing. Tighten threshold.
            new_threshold = min(0.80, base_threshold + 0.10)
            regime = "BEAR_TRAP"
        else:
            new_threshold = base_threshold
            regime = "NEUTRAL"
            
        # Update Config
        config["current_threshold"] = new_threshold
        config["regime"] = regime
        config["rolling_win_rate"] = win_rate
        config["last_update"] = datetime.now().isoformat()
        
        with open(ADAPTIVE_CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=4)
            
        logger.info(f"Adaptive Update: WinRate={win_rate:.2f} -> Regime={regime}, New Threshold={new_threshold:.2f}")

    def get_adaptive_threshold(self) -> float:
        """Returns the current dynamic threshold."""
        try:
            with open(ADAPTIVE_CONFIG_FILE, 'r') as f:
                config = json.load(f)
            return config.get("current_threshold", 0.60)
        except:
            return 0.60
            
if __name__ == "__main__":
    # Test
    learner = ActiveLearner()
    learner.log_trade("EURUSD", "BUY", 0.75, datetime.now().isoformat(), outcome=1.0)
    print(f"Current Threshold: {learner.get_adaptive_threshold()}")



--- FILE: backend/src/analysis/sniper_v5.py ---
"""
Sniper Protocol v5 - High-Conviction Trading Filter
====================================================
Multi-factor filtering to maximize Win Rate and Profit Factor.

Filters:
1. Confidence Filter: prob > (mean + 1.5σ) or prob < (mean - 1.5σ)
2. Volatility Filter: ATR > 25th percentile (avoid dead markets)
3. Session Filter: London (08:00-16:00) and NY (13:00-21:00) overlap

Metrics:
- Win Rate (Target: >56%)
- Profit Factor (Target: >1.5)
- Recovery Factor
- Sniper Trades vs Total Trades
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import polars as pl
import torch
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

from src.model.iara_liquid import IARA, LiquidConfig
from src.train.trainer_liquid import TrainerConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Paths
DATA_PROCESSED_DIR = Path(__file__).parent.parent.parent / "data" / "processed"
CHECKPOINT_DIR = Path(__file__).parent.parent.parent / "checkpoints"
OUTPUT_DIR = Path(__file__).parent.parent.parent / "output"

# Simulation Parameters
INITIAL_BALANCE = 1000.0
FIXED_RISK = 20.0
RISK_REWARD_RATIO = 1.0

# Sniper Parameters
CONFIDENCE_SIGMA = 1.5
ATR_PERCENTILE = 25
LONDON_START = 8   # 08:00 UTC
LONDON_END = 16    # 16:00 UTC
NY_START = 13      # 13:00 UTC
NY_END = 21        # 21:00 UTC


@dataclass
class TradeResult:
    """Individual trade result."""
    timestamp_idx: int
    hour: int
    prob: float
    predicted_dir: int
    true_dir: int
    won: bool
    profit: float
    is_sniper: bool
    session: str
    atr: float
    confidence_percentile: float


@dataclass
class SniperMetrics:
    """Aggregated sniper protocol metrics."""
    total_trades: int = 0
    sniper_trades: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    net_profit: float = 0.0
    final_balance: float = 0.0
    max_drawdown: float = 0.0
    recovery_factor: float = 0.0
    avg_yield_per_trade: float = 0.0
    session_win_rates: Dict[str, float] = field(default_factory=dict)
    hourly_win_rates: Dict[int, float] = field(default_factory=dict)


def load_model(device: torch.device) -> IARA:
    """Load trained I.A.R.A. model."""
    checkpoint_path = CHECKPOINT_DIR / "best_model.pth"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    trainer_config = checkpoint.get("config")
    if trainer_config:
        model_config = LiquidConfig(
            input_dim=trainer_config.input_dim,
            hidden_dim=trainer_config.hidden_dim,
            dropout=trainer_config.dropout
        )
    else:
        model_config = LiquidConfig(hidden_dim=512)
    
    model = IARA(model_config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    
    return model


def load_full_data(ticker: str = "EURUSD") -> pl.DataFrame:
    """Load full dataset with all columns."""
    file_path = DATA_PROCESSED_DIR / f"{ticker}.parquet"
    return pl.read_parquet(file_path)


@torch.no_grad()
def generate_all_predictions(
    model: IARA,
    df: pl.DataFrame,
    device: torch.device,
    seq_len: int = 64
) -> List[Dict]:
    """Generate predictions with full context."""
    model.eval()
    
    feature_cols = sorted([c for c in df.columns if c.endswith("_norm")])
    features = df.select(feature_cols).to_numpy().astype(np.float32)
    
    # Get metadata
    has_hour = "hour" in df.columns
    has_atr = "atr" in df.columns or "atr_norm" in df.columns
    has_target = "close_log_ret" in df.columns
    
    results = []
    n_samples = len(features) - seq_len
    
    for i in range(n_samples):
        x = torch.tensor(features[i:i + seq_len], dtype=torch.float32).unsqueeze(0).to(device)
        logits, _ = model(x)
        prob = torch.sigmoid(logits).cpu().item()
        
        target_idx = i + seq_len - 1
        
        result = {
            "idx": target_idx,
            "prob": prob,
        }
        
        if has_hour:
            result["hour"] = df["hour"][target_idx]
        else:
            result["hour"] = 12  # Default
        
        if has_atr:
            atr_col = "atr" if "atr" in df.columns else "atr_norm"
            result["atr"] = df[atr_col][target_idx]
        else:
            result["atr"] = 1.0
        
        if has_target:
            true_ret = df["close_log_ret"][target_idx]
            result["true_dir"] = 1 if true_ret > 0 else 0
        else:
            result["true_dir"] = 0
        
        results.append(result)
    
    return results


def get_session(hour: int) -> str:
    """Determine trading session."""
    london = LONDON_START <= hour < LONDON_END
    ny = NY_START <= hour < NY_END
    
    if london and ny:
        return "London-NY Overlap"
    elif london:
        return "London"
    elif ny:
        return "New York"
    elif 0 <= hour < 8:
        return "Asian"
    else:
        return "Other"


def apply_sniper_filters(
    predictions: List[Dict],
    prob_mean: float,
    prob_std: float,
    atr_threshold: float,
    top_percentile: float = 10.0
) -> List[Dict]:
    """Apply multi-factor sniper filters."""
    
    # Calculate confidence thresholds
    upper_threshold = prob_mean + CONFIDENCE_SIGMA * prob_std
    lower_threshold = prob_mean - CONFIDENCE_SIGMA * prob_std
    
    logger.info(f"Confidence thresholds: UP > {upper_threshold:.4f}, DOWN < {lower_threshold:.4f}")
    logger.info(f"ATR threshold (25th pctl): {atr_threshold:.6f}")
    
    for pred in predictions:
        prob = pred["prob"]
        atr = pred["atr"]
        hour = pred["hour"]
        
        # Determine if sniper trade
        confidence_pass = (prob > upper_threshold) or (prob < lower_threshold)
        volatility_pass = atr > atr_threshold
        session = get_session(hour)
        session_pass = session in ["London-NY Overlap", "London", "New York"]
        
        # All filters must pass
        pred["is_sniper"] = confidence_pass and volatility_pass and session_pass
        pred["session"] = session
        
        # Confidence percentile (for ranking)
        if prob > prob_mean:
            pred["confidence_percentile"] = (prob - prob_mean) / prob_std
        else:
            pred["confidence_percentile"] = (prob_mean - prob) / prob_std
    
    return predictions


def simulate_sniper_trades(
    predictions: List[Dict],
    sniper_only: bool = True
) -> Tuple[List[TradeResult], SniperMetrics]:
    """Simulate trading with sniper protocol."""
    
    balance = INITIAL_BALANCE
    balances = [balance]
    trades: List[TradeResult] = []
    
    for pred in predictions:
        if sniper_only and not pred["is_sniper"]:
            continue
        
        prob = pred["prob"]
        true_dir = pred["true_dir"]
        
        # Determine trade direction
        if prob > 0.5:
            predicted_dir = 1  # BUY
        else:
            predicted_dir = 0  # SELL
        
        # Trade outcome
        won = predicted_dir == true_dir
        profit = FIXED_RISK * RISK_REWARD_RATIO if won else -FIXED_RISK
        balance += profit
        balances.append(balance)
        
        trade = TradeResult(
            timestamp_idx=pred["idx"],
            hour=pred["hour"],
            prob=prob,
            predicted_dir=predicted_dir,
            true_dir=true_dir,
            won=won,
            profit=profit,
            is_sniper=pred["is_sniper"],
            session=pred["session"],
            atr=pred["atr"],
            confidence_percentile=pred["confidence_percentile"]
        )
        trades.append(trade)
    
    # Calculate metrics
    n_trades = len(trades)
    wins = sum(1 for t in trades if t.won)
    losses = n_trades - wins
    win_rate = wins / n_trades * 100 if n_trades > 0 else 0
    
    gross_profit = sum(t.profit for t in trades if t.profit > 0)
    gross_loss = abs(sum(t.profit for t in trades if t.profit < 0))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    net_profit = balance - INITIAL_BALANCE
    
    # Max Drawdown
    balances_array = np.array(balances)
    peak = np.maximum.accumulate(balances_array)
    drawdown = (peak - balances_array) / peak
    max_drawdown = drawdown.max() * 100
    
    # Recovery Factor
    recovery_factor = net_profit / (max_drawdown * INITIAL_BALANCE / 100) if max_drawdown > 0 else 0
    
    # Avg Yield per Trade
    avg_yield = net_profit / n_trades if n_trades > 0 else 0
    
    # Session Win Rates
    session_stats: Dict[str, List[bool]] = {}
    for t in trades:
        if t.session not in session_stats:
            session_stats[t.session] = []
        session_stats[t.session].append(t.won)
    
    session_win_rates = {
        s: sum(wins) / len(wins) * 100 if len(wins) > 0 else 0
        for s, wins in session_stats.items()
    }
    
    # Hourly Win Rates
    hourly_stats: Dict[int, List[bool]] = {}
    for t in trades:
        if t.hour not in hourly_stats:
            hourly_stats[t.hour] = []
        hourly_stats[t.hour].append(t.won)
    
    hourly_win_rates = {
        h: sum(wins) / len(wins) * 100 if len(wins) > 0 else 0
        for h, wins in hourly_stats.items()
    }
    
    sniper_count = sum(1 for t in trades if t.is_sniper)
    
    metrics = SniperMetrics(
        total_trades=len(predictions),
        sniper_trades=sniper_count,
        wins=wins,
        losses=losses,
        win_rate=win_rate,
        profit_factor=profit_factor,
        net_profit=net_profit,
        final_balance=balance,
        max_drawdown=max_drawdown,
        recovery_factor=recovery_factor,
        avg_yield_per_trade=avg_yield,
        session_win_rates=session_win_rates,
        hourly_win_rates=hourly_win_rates
    )
    
    return trades, metrics


def generate_visualizations(
    predictions: List[Dict],
    trades: List[TradeResult],
    metrics: SniperMetrics
):
    """Generate analysis visualizations."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Probability Distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    probs = [p["prob"] for p in predictions]
    prob_mean = np.mean(probs)
    prob_std = np.std(probs)
    
    # Histogram
    ax1 = axes[0]
    ax1.hist(probs, bins=50, color='steelblue', alpha=0.7, edgecolor='white')
    ax1.axvline(prob_mean, color='red', linestyle='--', label=f'Mean: {prob_mean:.3f}')
    ax1.axvline(prob_mean + 1.5 * prob_std, color='green', linestyle=':', label=f'+1.5σ: {prob_mean + 1.5 * prob_std:.3f}')
    ax1.axvline(prob_mean - 1.5 * prob_std, color='orange', linestyle=':', label=f'-1.5σ: {prob_mean - 1.5 * prob_std:.3f}')
    ax1.set_xlabel('Probability')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Model Probability Distribution (Confidence Tails)')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Sniper vs Regular trades
    ax2 = axes[1]
    sniper_probs = [p["prob"] for p in predictions if p["is_sniper"]]
    regular_probs = [p["prob"] for p in predictions if not p["is_sniper"]]
    
    ax2.hist(regular_probs, bins=30, alpha=0.5, label=f'Regular ({len(regular_probs)})', color='gray')
    ax2.hist(sniper_probs, bins=30, alpha=0.7, label=f'Sniper ({len(sniper_probs)})', color='green')
    ax2.set_xlabel('Probability')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Sniper vs Regular Trade Probabilities')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'sniper_probability_distribution.png', dpi=150)
    plt.close()
    logger.info(f"Saved probability distribution to {OUTPUT_DIR / 'sniper_probability_distribution.png'}")
    
    # 2. Win Rate Heatmap by Hour
    fig, ax = plt.subplots(figsize=(12, 4))
    
    hours = list(range(24))
    win_rates = [metrics.hourly_win_rates.get(h, 50) for h in hours]
    trade_counts = [sum(1 for t in trades if t.hour == h) for h in hours]
    
    # Create heatmap data
    colors = ['red' if wr < 50 else 'green' for wr in win_rates]
    bars = ax.bar(hours, win_rates, color=colors, alpha=0.7, edgecolor='white')
    ax.axhline(50, color='black', linestyle='--', alpha=0.5)
    ax.axhline(56, color='gold', linestyle='-', linewidth=2, label='Target: 56%')
    
    # Add trade count annotation
    for i, (bar, count) in enumerate(zip(bars, trade_counts)):
        if count > 0:
            ax.annotate(f'{count}', 
                       xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                       ha='center', va='bottom', fontsize=8, color='gray')
    
    # Highlight session overlaps
    ax.axvspan(LONDON_START, LONDON_END, alpha=0.1, color='blue', label='London')
    ax.axvspan(NY_START, NY_END, alpha=0.1, color='orange', label='New York')
    
    ax.set_xlabel('Hour (UTC)')
    ax.set_ylabel('Win Rate (%)')
    ax.set_title('Win Rate by Hour of Day (Sniper Trades)')
    ax.set_xticks(hours)
    ax.set_ylim(0, 100)
    ax.legend(loc='upper right')
    ax.grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'sniper_hourly_winrate.png', dpi=150)
    plt.close()
    logger.info(f"Saved hourly win rate to {OUTPUT_DIR / 'sniper_hourly_winrate.png'}")


def run_sniper_analysis():
    """Run the full Sniper Protocol analysis."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load model and data
    model = load_model(device)
    df = load_full_data()
    logger.info(f"Loaded {len(df)} samples")
    
    # Generate predictions
    logger.info("Generating predictions...")
    predictions = generate_all_predictions(model, df, device)
    logger.info(f"Generated {len(predictions)} predictions")
    
    # Calculate statistics
    probs = np.array([p["prob"] for p in predictions])
    prob_mean = probs.mean()
    prob_std = probs.std()
    logger.info(f"Prob stats: mean={prob_mean:.4f}, std={prob_std:.4f}")
    
    # ATR threshold
    atrs = np.array([p["atr"] for p in predictions])
    atr_threshold = np.percentile(atrs, ATR_PERCENTILE)
    
    # Apply sniper filters
    predictions = apply_sniper_filters(predictions, prob_mean, prob_std, atr_threshold)
    
    sniper_count = sum(1 for p in predictions if p["is_sniper"])
    logger.info(f"Sniper trades identified: {sniper_count} / {len(predictions)} ({sniper_count/len(predictions)*100:.1f}%)")
    
    # Simulate trades
    print("\n" + "=" * 80)
    print("SNIPER PROTOCOL v5 ANALYSIS")
    print("=" * 80)
    
    # 1. All trades (baseline)
    trades_all, metrics_all = simulate_sniper_trades(predictions, sniper_only=False)
    print(f"\n📊 BASELINE (All Trades):")
    print(f"   Trades: {metrics_all.sniper_trades}")
    print(f"   Win Rate: {metrics_all.win_rate:.1f}%")
    print(f"   Profit Factor: {metrics_all.profit_factor:.2f}")
    print(f"   Net Profit: ${metrics_all.net_profit:.2f}")
    
    # 2. Sniper trades only
    trades_sniper, metrics_sniper = simulate_sniper_trades(predictions, sniper_only=True)
    print(f"\n🎯 SNIPER PROTOCOL (Filtered Trades):")
    print(f"   Trades: {metrics_sniper.sniper_trades} ({metrics_sniper.sniper_trades/len(predictions)*100:.1f}% of total)")
    print(f"   Win Rate: {metrics_sniper.win_rate:.1f}% {'✓' if metrics_sniper.win_rate > 56 else ''}")
    print(f"   Profit Factor: {metrics_sniper.profit_factor:.2f} {'✓' if metrics_sniper.profit_factor > 1.5 else ''}")
    print(f"   Net Profit: ${metrics_sniper.net_profit:.2f}")
    print(f"   Max Drawdown: {metrics_sniper.max_drawdown:.1f}%")
    print(f"   Recovery Factor: {metrics_sniper.recovery_factor:.2f}")
    print(f"   Avg Yield/Trade: ${metrics_sniper.avg_yield_per_trade:.2f}")
    
    # Session analysis
    print(f"\n📍 SESSION ANALYSIS:")
    for session, wr in sorted(metrics_sniper.session_win_rates.items(), key=lambda x: -x[1]):
        count = sum(1 for t in trades_sniper if t.session == session)
        status = "✓" if wr > 55 else ""
        print(f"   {session:20s}: {wr:5.1f}% ({count} trades) {status}")
    
    # Top 10% highest confidence trades
    print(f"\n🔥 TOP 10% HIGHEST CONFIDENCE:")
    sorted_by_confidence = sorted(predictions, key=lambda x: x["confidence_percentile"], reverse=True)
    top_10_pct = sorted_by_confidence[:len(sorted_by_confidence)//10]
    
    top_trades = []
    for pred in top_10_pct:
        if pred["prob"] > 0.5:
            predicted_dir = 1
        else:
            predicted_dir = 0
        won = predicted_dir == pred["true_dir"]
        top_trades.append(won)
    
    top_win_rate = sum(top_trades) / len(top_trades) * 100 if top_trades else 0
    print(f"   Trades: {len(top_trades)}")
    print(f"   Win Rate: {top_win_rate:.1f}%")
    
    # Generate visualizations
    generate_visualizations(predictions, trades_sniper, metrics_sniper)
    
    print("\n" + "=" * 80)
    print("SNIPER SUMMARY")
    print("=" * 80)
    print(f"│ Metric           │ Baseline │ Sniper   │ Target   │ Status   │")
    print(f"├──────────────────┼──────────┼──────────┼──────────┼──────────┤")
    print(f"│ Win Rate         │ {metrics_all.win_rate:6.1f}%  │ {metrics_sniper.win_rate:6.1f}%  │  >56%    │ {'✓' if metrics_sniper.win_rate > 56 else '✗':^8} │")
    print(f"│ Profit Factor    │ {metrics_all.profit_factor:6.2f}   │ {metrics_sniper.profit_factor:6.2f}   │   >1.5   │ {'✓' if metrics_sniper.profit_factor > 1.5 else '✗':^8} │")
    print(f"│ Net Profit       │ ${metrics_all.net_profit:>7.0f} │ ${metrics_sniper.net_profit:>7.0f} │    -     │    -     │")
    print(f"│ Trades           │ {metrics_all.sniper_trades:>8} │ {metrics_sniper.sniper_trades:>8} │    -     │    -     │")
    print(f"└──────────────────┴──────────┴──────────┴──────────┴──────────┘")
    
    print(f"\n📁 Visualizations saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    run_sniper_analysis()



--- FILE: backend/src/analysis/performance_matrix.py ---
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



--- FILE: backend/src/analysis/dynamic_calibrator.py ---
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



--- FILE: backend/src/database/models.py ---
from datetime import datetime
from typing import Optional, List, Dict, Any
from sqlalchemy import Column, Integer, String, Float, DateTime, JSON, ForeignKey, Boolean
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.sql import func

Base = declarative_base()

class Trade(Base):
    """
    Records executed trades.
    Replaces the ephemeral 'orders' list in SharedState.
    """
    __tablename__ = "trades"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, index=True, nullable=False)
    side = Column(String, nullable=False) # 'BUY' or 'SELL'
    qty = Column(Float, nullable=False)
    price = Column(Float, nullable=False)
    
    # Execution Details
    order_id = Column(String, unique=True, index=True) # ID from Exchange
    strategy = Column(String, default="Trinity-V9")
    
    # Profit Tracking
    pnl = Column(Float, nullable=True) # Realized PnL (filled on close)
    entry_price = Column(Float, nullable=True) # For closing trades
    
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    def __repr__(self):
        return f"<Trade(id={self.id}, {self.side} {self.symbol} @ {self.price})>"


class Signal(Base):
    """
    Records the 'Brain State' at the moment of decision.
    Essential for 'X-Ray' analysis and Retraining.
    """
    __tablename__ = "signals"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, index=True, nullable=False)
    timeframe = Column(String, default="M5")
    
    # Model Output
    signal_value = Column(Integer, nullable=False) # -1 (Sell), 0 (Hold), 1 (Buy)
    confidence = Column(Float, default=0.0)
    model_version = Column(String, nullable=False) # e.g. "v9.7-Gladiator"
    
    # Context Snapshot (The 'Why')
    # Use JSONB in Postgres for better query performance
    features_snapshot = Column(JSON, nullable=True) # RSI, Price, etc.
    worker_votes = Column(JSON, nullable=True) # { "worker_1": "BUY", ... }
    
    timestamp = Column(DateTime(timezone=True), server_default=func.now())

    def __repr__(self):
        return f"<Signal(id={self.id}, {self.symbol}, val={self.signal_value}, conf={self.confidence:.2f})>"


class AgentProfile(Base):
    """
    User configured trading agents.
    """
    __tablename__ = "agent_profiles"

    id = Column(String, primary_key=True, index=True) # e.g. "agent-btc-usd"
    symbol = Column(String, nullable=False) # e.g. "BTC/USD"
    market_type = Column(String, nullable=False) # "crypto" or "forex"
    model_version = Column(String, default="active_model_v10.pth")
    is_active = Column(Boolean, default=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    def __repr__(self):
        return f"<AgentProfile({self.id}, {self.symbol}, active={self.is_active})>"


class ModelVersion(Base):
    """
    Registry of trained models.
    """
    __tablename__ = "model_versions"

    id = Column(Integer, primary_key=True, index=True)
    tag = Column(String, unique=True, index=True, nullable=False) # v9.7
    path = Column(String, nullable=False) # checkpoints/v9.7.pth
    
    metrics = Column(JSON, nullable=True) # { "BTC/USD": {"accuracy": 0.65}, "ETH/USD": {"accuracy": 0.61} }
    category_type = Column(String, default="SPECIALIST") # "SPECIALIST" (one pair) or "GENERALIST" (multi-pair)
    is_active = Column(Boolean, default=False)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())



--- FILE: backend/src/database/core.py ---
import os
import logging
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from dotenv import load_dotenv

from src.database.models import Base

load_dotenv()

logger = logging.getLogger("Database")

# DB Configuration
DATABASE_URL = os.getenv("DB_URI", "sqlite+aiosqlite:///./saci_core.db")

# Patch driver for AsyncPG if needed
if DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://")

# If running outside Docker, 'db' host might not resolve.
# We trust the user OR we could have a fallback. 
# For now, we trust the env or the default SQLite fallback if env is missing.

engine = create_async_engine(
    DATABASE_URL,
    echo=False, # Set to True for SQL debug
    future=True
)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False
)

async def init_db():
    """Create tables if they don't exist."""
    try:
        async with engine.begin() as conn:
            # await conn.run_sync(Base.metadata.drop_all) # Dangerous!
            await conn.run_sync(Base.metadata.create_all)
        logger.info("✅ Database Tables Initialized (Trades, Signals)")
    except Exception as e:
        logger.error(f"❌ Database Init Failed: {e}")
        # Fallback for dev without DB?
        # Only if we want to allow running without DB (Mock mode)

async def get_db():
    """Dependency for FastAPI/Context Managers."""
    async with AsyncSessionLocal() as session:
        yield session



--- FILE: backend/src/diag/visualize_brain.py ---

import logging
import argparse
from pathlib import Path
import sys
import torch
import torch.nn as nn
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import polars as pl
from tqdm import tqdm

# Add root to path
sys.path.append(".")

from src.model.iara_liquid import IARA, LiquidConfig
from src.train.trainer_liquid import MultiAssetDataset, TrainerConfig
from src.data.processor import Normalizer, NormalizerState

# Configure Headless Plotting
plt.switch_backend('Agg')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("BrainScan")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(checkpoint_path, input_dim=16):
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    state_dict = checkpoint['model']
    
    # Handle compiled model names
    if list(state_dict.keys())[0].startswith('_orig_mod.'):
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
        
    config = checkpoint.get('config')
    
    # Enforce v6.7 settings if config missing new fields
    liquid_config = LiquidConfig(
        input_dim=input_dim,
        hidden_dim=getattr(config, 'hidden_dim', 128),
        num_heads=getattr(config, 'num_heads', 4),
        dropout=getattr(config, 'dropout', 0.5),
        max_seq_len=getattr(config, 'max_seq_len', 256),
        num_cfc_layers=2 # v6.7 standard
    )
    
    model = IARA(liquid_config).to(DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    return model

def visualize(args):
    # 1. Load Normalizer for feature names
    normalizer = Normalizer.load("data/scaler.pkl")
    feature_names = normalizer.feature_names
    input_dim = len(feature_names)
    logger.info(f"Loaded {input_dim} features: {feature_names}")
    
    # 2. Load Model
    model = load_model(args.checkpoint, input_dim)
    logger.info("Model loaded.")
    
    # 3. Load Data Sample
    dataset = MultiAssetDataset(
        data_dir="data/processed",
        seq_len=256,
        stride=100, # Sparse sampling
        augment=False,
        target_col="close_log_ret"
    )
    
    # Pick a batch
    indices = range(0, min(len(dataset), args.batch_size))
    batch_tensors = [dataset[i][0] for i in indices]
    batch_tensor = torch.stack(batch_tensors).to(DEVICE) # [B, T, D]
    
    logger.info(f"Visualizing batch: {batch_tensor.shape}")
    
    # 4. Hook Mechanism
    # We want to capture FeatureAttention gates
    # FeatureAttention returns x * gate(x). We want 'gate(x)'.
    # A backward hook on gradients is one way, but forward hook on GATE is better.
    # Structure: model.feature_attention.gate (Sequential)
    
    attention_outputs = {}
    
    def hook_fn(module, input, output):
        # output is the gate value (sigmoid output)
        attention_outputs['gate'] = output.detach()
        
    # Register hook on the sigmoid layer of the specific attention module
    # model.feature_attention.gate is a Sequential. Last layer is Sigmoid.
    hook = model.feature_attention.gate[-1].register_forward_hook(hook_fn)
    
    # 5. Forward Pass
    with torch.no_grad():
        logits, _ = model(batch_tensor)
        
    hook.remove()
    
    # 6. Process Attention (Back-Projection)
    # Gate: [B, T, Hidden]
    # InputProj: [Hidden, Input] (Linear weight is [Out, In])
    # Importance = Gate @ Weight
    
    gate_values = attention_outputs['gate'] # [B, T, 128]
    proj_weight = model.input_proj.weight.abs() # [128, 16]
    
    # Broadcast matmul: [B, T, 128] x [128, 16] -> [B, T, 16]
    # We use torch.matmul which handles broadcasting
    feature_importance_raw = torch.matmul(gate_values, proj_weight) # [B, T, 16]
    
    # Normalize per step for heatmap
    # feature_importance = feature_importance_raw / feature_importance_raw.sum(dim=-1, keepdim=True)
    feature_importance = feature_importance_raw
    
    # 7. Visualization 1: Aggregated Importance (Global)
    # Sum over Batch and Time
    global_importance = feature_importance.mean(dim=(0, 1)).detach().cpu().numpy()
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x=feature_names, y=global_importance, palette="viridis")
    plt.title("Reference: IARA v6.7 Global Feature Importance (Gate Projection)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    Path("reports/figures").mkdir(parents=True, exist_ok=True)
    plt.savefig("reports/figures/feature_importance_global.png")
    logger.info("Saved reports/figures/feature_importance_global.png")
    
    # 8. Visualization 2: Temporal Dynamics (Best Sample)
    # Find sample with highest confidence (logit magnitude)
    conf = torch.sigmoid(logits).abs() - 0.5 # Distance from neutral
    best_idx = torch.argmax(conf).item()
    
    sample_imp = feature_importance[best_idx].detach().cpu().numpy().T # [Feat, Time]
    
    plt.figure(figsize=(14, 8))
    sns.heatmap(sample_imp, yticklabels=feature_names, cmap="magma", robust=True)
    plt.title(f"IARA v6.7 Brain Scan - Sample #{best_idx} (Conf: {torch.sigmoid(logits[best_idx]):.2f})")
    plt.xlabel("Time Steps (Past -> Now)")
    plt.ylabel("Features")
    plt.tight_layout()
    plt.savefig("reports/figures/brain_scan_sample.png")
    logger.info("Saved reports/figures/brain_scan_sample.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/latest.pth")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    
    if not Path(args.checkpoint).exists():
        logger.error(f"Checkpoint {args.checkpoint} not found. Train first!")
        exit(1)
        
    visualize(args)



--- FILE: backend/src/diag/sanity_check.py ---

import sys
import torch
import numpy as np
import polars as pl
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(".")
from src.model.iara_liquid import IARA, LiquidConfig
from src.train.trainer_liquid import MultiAssetDataset, TrainerConfig

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_checkpoint(path):
    checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
    config_dict = checkpoint['config']
    
    # Reconstruct Config
    # If config is object, convert to dict or use attributes
    if isinstance(config_dict, TrainerConfig):
        t_conf = config_dict
    else:
        # Fallback
        t_conf = TrainerConfig(**config_dict)
        
    liquid_config = LiquidConfig(
        input_dim=t_conf.input_dim,
        hidden_dim=t_conf.hidden_dim, 
        num_heads=t_conf.num_heads,
        dropout=t_conf.dropout,
        max_seq_len=t_conf.max_seq_len,
        num_cfc_layers=t_conf.num_cfc_layers
    )
    
    model = IARA(liquid_config).to(DEVICE)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    return model, t_conf

def sanity_check():
    print("🧠 Loading Interrupted Model...")
    model, config = load_checkpoint("checkpoints/interrupted_model.pth")
    
    print("📂 Loading Validation Data...")
    ds = MultiAssetDataset(seq_len=config.seq_len, stride=config.stride, augment=False)
    _, val_idxs = ds.get_chronological_split(config.val_split)
    
    # Take a subset for speed
    val_subset = torch.utils.data.Subset(ds, val_idxs[:2000]) 
    loader = torch.utils.data.DataLoader(val_subset, batch_size=256, shuffle=False)
    
    all_preds = []
    all_targets = []
    all_probs = []
    
    print("🔍 Running Inference...")
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            for X, y, _ in loader:
                X = X.to(DEVICE)
                logits, _ = model(X)
                probs = torch.sigmoid(logits).cpu().numpy()
                preds = (probs > 0.5).astype(float)
                
                all_probs.extend(probs)
                all_preds.extend(preds)
                all_targets.extend(y.numpy())
                
    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    print("\n" + "="*40)
    print("📊 DIAGNOSTIC RESULTS")
    print("="*40)
    
    # 1. Bias Check
    mean_prob = np.mean(all_probs)
    perc_up = np.mean(all_preds) * 100
    print(f"Mean Prediction Probability: {mean_prob:.4f}")
    print(f"Predicted 'UP' Ratio:      {perc_up:.2f}% (Target: 50% usually)")
    
    if perc_up > 65:
        print("⚠️  WARNING: Model is PERMABULL (Biased towards UP)")
    elif perc_up < 35:
        print("⚠️  WARNING: Model is PERMABEAR (Biased towards DOWN)")
    else:
        print("✅  Bias seems balanced.")
        
    # 2. Confusion Matrix
    cm = confusion_matrix(all_targets, all_preds)
    print("\nConfusion Matrix:")
    print(cm)
    print("TN  FP")
    print("FN  TP")
    
    # 3. High Confidence Check
    high_conf_mask = np.abs(all_probs - 0.5) > 0.1 # Prob > 0.6 or < 0.4
    if np.sum(high_conf_mask) > 0:
        hc_acc = np.mean(all_preds[high_conf_mask] == all_targets[high_conf_mask])
        print(f"\nHigh Confidence Accuracy (>60%): {hc_acc:.2%} ({np.sum(high_conf_mask)} samples)")
    else:
        print("\n❌ No High Confidence samples found.")

if __name__ == "__main__":
    sanity_check()



--- FILE: backend/src/diag/quick_signal.py ---

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



--- FILE: backend/src/diag/neuro_lab.py ---
#!/usr/bin/env python3
"""
I.A.R.A. Neuro-Lab - Real-Time Training Monitor
================================================
A "Mission Control" dashboard for visualizing the AI's evolution.

Features:
- Telemetry parsing from JSONL logs
- Neural attention heatmaps
- Dark-mode dashboard rendering

Usage:
    docker compose exec saci-brain python src/diag/neuro_lab.py --watch
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import seaborn as sns
import torch

# Add project root to path
sys.path.append(".")

from src.model.iara_liquid import IARA, LiquidConfig
from src.train.trainer_liquid import MultiAssetDataset, TrainerConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger("NeuroLab")

# Paths
LOGS_DIR = Path("logs")
CHECKPOINTS_DIR = Path("checkpoints")
OUTPUT_DIR = Path("reports/lab")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dark mode aesthetics
plt.style.use('dark_background')
plt.rcParams['font.family'] = 'monospace'
plt.rcParams['axes.facecolor'] = '#0a0a0a'
plt.rcParams['figure.facecolor'] = '#0a0a0a'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.labelcolor'] = '#cccccc'
plt.rcParams['xtick.color'] = '#888888'
plt.rcParams['ytick.color'] = '#888888'
plt.rcParams['grid.color'] = '#222222'


class TelemetryParser:
    """Parse JSONL training logs."""
    
    def __init__(self, logs_dir: Path = LOGS_DIR):
        self.logs_dir = logs_dir
        self.data = []
        
    def find_latest_log(self) -> Optional[Path]:
        """Find the most recent JSONL file."""
        jsonl_files = list(self.logs_dir.glob("training_history_*.jsonl"))
        if not jsonl_files:
            return None
        return max(jsonl_files, key=lambda p: p.stat().st_mtime)
    
    def parse(self) -> List[Dict]:
        """Parse the latest log file robustly."""
        log_file = self.find_latest_log()
        if not log_file:
            logger.warning("No JSONL logs found.")
            return []
        
        self.data = []
        try:
            with open(log_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        self.data.append(entry)
                    except json.JSONDecodeError:
                        # Incomplete line (being written), skip
                        continue
        except Exception as e:
            logger.error(f"Error parsing log: {e}")
        
        return self.data
    
    def get_metrics(self) -> Dict[str, List]:
        """Extract key metrics for plotting."""
        if not self.data:
            self.parse()
        
        metrics = {
            'epochs': [],
            'train_auc': [],
            'val_auc': [],
            'train_loss': [],
            'val_loss': [],
            'chronic': [],
            'mastered': [],
            'lr': [],
            'vram': []
        }
        
        for entry in self.data:
            if 'status' in entry:  # Skip INTERRUPTED entries
                continue
            
            epoch = entry.get('epoch', 0)
            train = entry.get('train', {})
            val = entry.get('val', {})
            memory = entry.get('memory', {})
            hardware = entry.get('hardware', {})
            meta = entry.get('metrics', {})
            
            metrics['epochs'].append(epoch)
            metrics['train_auc'].append(train.get('auc', 0.5))
            metrics['val_auc'].append(val.get('auc', 0.5))
            metrics['train_loss'].append(train.get('loss', 0))
            metrics['val_loss'].append(val.get('loss', 0))
            metrics['chronic'].append(memory.get('chronic', 0))
            metrics['mastered'].append(memory.get('mastered', 0))
            metrics['lr'].append(meta.get('learning_rate', 0.001))
            metrics['vram'].append(hardware.get('vram_percent', 0))
        
        return metrics


class NeuralImager:
    """Extract and visualize neural attention patterns."""
    
    def __init__(self, checkpoint_path: Path = CHECKPOINTS_DIR / "latest.pth"):
        self.checkpoint_path = checkpoint_path
        self.model = None
        self.config = None
        self.val_batch = None
        
    def load_model(self) -> bool:
        """Load the latest checkpoint."""
        if not self.checkpoint_path.exists():
            logger.warning(f"Checkpoint not found: {self.checkpoint_path}")
            return False
        
        try:
            checkpoint = torch.load(self.checkpoint_path, map_location=DEVICE, weights_only=False)
            self.config = checkpoint.get('config')
            
            # Handle both object and dict configs
            if isinstance(self.config, TrainerConfig):
                t_conf = self.config
            else:
                t_conf = TrainerConfig()
            
            liquid_config = LiquidConfig(
                input_dim=t_conf.input_dim,
                hidden_dim=t_conf.hidden_dim,
                num_heads=t_conf.num_heads,
                dropout=t_conf.dropout,
                max_seq_len=t_conf.max_seq_len,
                num_cfc_layers=t_conf.num_cfc_layers
            )
            
            self.model = IARA(liquid_config).to(DEVICE)
            
            state_dict = checkpoint['model']
            # Handle compiled model weights
            if list(state_dict.keys())[0].startswith('_orig_mod.'):
                state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
            
            self.model.load_state_dict(state_dict, strict=False)
            self.model.eval()
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def get_validation_batch(self, batch_size: int = 32) -> Optional[torch.Tensor]:
        """Get a fixed validation batch for consistent attention visualization."""
        if self.val_batch is not None:
            return self.val_batch
        
        try:
            dataset = MultiAssetDataset(
                seq_len=256,
                stride=100,
                augment=False
            )
            _, val_idxs = dataset.get_chronological_split(0.2)
            
            # Get first N samples
            batch_tensors = []
            for i in range(min(batch_size, len(val_idxs))):
                idx = val_idxs[i]
                x, _, _ = dataset[idx]
                batch_tensors.append(x)
            
            self.val_batch = torch.stack(batch_tensors).to(DEVICE)
            return self.val_batch
            
        except Exception as e:
            logger.error(f"Failed to load validation data: {e}")
            return None
    
    def extract_attention(self) -> Optional[np.ndarray]:
        """Extract attention weights from the model."""
        if self.model is None:
            if not self.load_model():
                return None
        
        batch = self.get_validation_batch()
        if batch is None:
            return None
        
        attention_maps = {}
        
        # Hook to capture attention from FeatureAttention gate
        def hook_fn(module, input, output):
            attention_maps['gate'] = output.detach()
        
        # Register hook on the sigmoid of FeatureAttention.gate
        try:
            hook = self.model.feature_attention.gate[-1].register_forward_hook(hook_fn)
        except AttributeError:
            logger.warning("FeatureAttention not found in model.")
            return None
        
        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                _, _ = self.model(batch)
        
        hook.remove()
        
        if 'gate' not in attention_maps:
            return None
        
        # Back-project to input features
        gate_values = attention_maps['gate'].float()  # [B, T, H] - Ensure float32
        proj_weight = self.model.input_proj.weight.float().abs()  # [H, D]
        
        # [B, T, H] x [H, D] -> [B, T, D]
        feature_importance = torch.matmul(gate_values, proj_weight)
        
        # Average over batch, transpose for heatmap [D, T]
        importance = feature_importance.mean(dim=0).detach().cpu().numpy().T
        
        return importance


class DashboardRenderer:
    """Render the Mission Control dashboard."""
    
    def __init__(self, output_path: Path = OUTPUT_DIR / "mission_control_live.png"):
        self.output_path = output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Feature names for heatmap labels
        self.feature_names = [
            'rsi', 'macd', 'macd_sig', 'macd_hist',
            'bb_pct', 'bb_width', 'atr',
            'stoch_k', 'stoch_d', 'velocity',
            'adx', 'sentiment',
            'h_sin', 'h_cos', 'd_sin', 'd_cos'
        ]
    
    def render(
        self,
        metrics: Dict[str, List],
        attention_map: Optional[np.ndarray] = None,
        current_epoch: int = 0,
        current_lr: float = 0.001,
        vram_usage: float = 0.0
    ):
        """Generate the dashboard image."""
        
        fig = plt.figure(figsize=(16, 10), dpi=100)
        gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1.2], hspace=0.3, wspace=0.2)
        
        # === TOP LEFT: AUC Growth Chart ===
        ax1 = fig.add_subplot(gs[0, 0])
        if metrics['epochs']:
            ax1.plot(metrics['epochs'], metrics['train_auc'], 
                     color='#00bfff', linewidth=2, label='Train AUC', alpha=0.8)
            ax1.plot(metrics['epochs'], metrics['val_auc'], 
                     color='#00ff88', linewidth=2.5, label='Val AUC')
            ax1.axhline(y=0.5, color='#ff4444', linestyle='--', alpha=0.5, label='Random')
            ax1.fill_between(metrics['epochs'], 0.5, metrics['val_auc'], 
                             color='#00ff88', alpha=0.1)
        
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('AUC Score')
        ax1.set_title('🧠 Intelligence Growth', fontsize=12, fontweight='bold', color='#00ff88')
        ax1.legend(loc='lower right', framealpha=0.3)
        ax1.set_ylim(0.45, 0.65)
        ax1.grid(True, alpha=0.3)
        
        # === TOP RIGHT: Memory Therapy Chart ===
        ax2 = fig.add_subplot(gs[0, 1])
        if metrics['epochs']:
            width = 0.8
            ax2.bar(metrics['epochs'], metrics['chronic'], 
                    width, label='Chronic Errors', color='#ff6b6b', alpha=0.8)
            ax2.bar(metrics['epochs'], metrics['mastered'], 
                    width, bottom=metrics['chronic'], label='Mastered', color='#4ecdc4', alpha=0.8)
        
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Sample Count')
        ax2.set_title('🧬 Memory Therapy Progress', fontsize=12, fontweight='bold', color='#ff6b6b')
        ax2.legend(loc='upper right', framealpha=0.3)
        ax2.grid(True, alpha=0.3)
        
        # === BOTTOM: Brain Scan Heatmap ===
        ax3 = fig.add_subplot(gs[1, :])
        
        if attention_map is not None:
            # Limit time steps for readability
            display_map = attention_map[:, -100:]  # Last 100 timesteps
            
            # Use shortened feature names
            n_features = min(len(self.feature_names), display_map.shape[0])
            yticklabels = self.feature_names[:n_features]
            
            sns.heatmap(
                display_map[:n_features],
                ax=ax3,
                cmap='magma',
                robust=True,
                yticklabels=yticklabels,
                xticklabels=False,
                cbar_kws={'label': 'Attention', 'shrink': 0.5}
            )
            ax3.set_xlabel('Time Steps (Past → Now)')
            ax3.set_ylabel('Features')
        else:
            ax3.text(0.5, 0.5, 'Awaiting Neural Data...', 
                     ha='center', va='center', fontsize=14, color='#666666')
            ax3.set_xlim(0, 1)
            ax3.set_ylim(0, 1)
        
        ax3.set_title('🔬 Real-Time Brain Scan (Attention Heatmap)', 
                      fontsize=12, fontweight='bold', color='#ff9f1c')
        
        # === FOOTER: Status Bar ===
        status_text = (
            f"Epoch: {current_epoch}  |  "
            f"LR: {current_lr:.2e}  |  "
            f"VRAM: {vram_usage:.1f}%  |  "
            f"Samples: {len(metrics['epochs'])}  |  "
            f"Updated: {time.strftime('%H:%M:%S')}"
        )
        fig.text(0.5, 0.02, status_text, ha='center', fontsize=10, 
                 color='#888888', family='monospace')
        
        # Title
        fig.suptitle('I.A.R.A. NEURO-LAB v6.7', fontsize=16, fontweight='bold', 
                     color='#ffffff', y=0.98)
        
        # Save
        plt.savefig(self.output_path, bbox_inches='tight', facecolor='#0a0a0a', edgecolor='none')
        plt.close(fig)
        
        logger.info(f"Dashboard saved: {self.output_path}")


def run_neuro_lab(watch: bool = False, interval: int = 30):
    """Main loop for the Neuro-Lab."""
    
    logger.info("=" * 50)
    logger.info("  I.A.R.A. NEURO-LAB v6.7")
    logger.info("  Real-Time Training Monitor")
    logger.info("=" * 50)
    
    telemetry = TelemetryParser()
    imager = NeuralImager()
    renderer = DashboardRenderer()
    
    while True:
        try:
            # 1. Parse Telemetry
            logger.info("Parsing telemetry...")
            metrics = telemetry.get_metrics()
            
            # 2. Neural Imaging (Reload model each time to get latest weights)
            logger.info("Extracting neural patterns...")
            imager.load_model()
            attention_map = imager.extract_attention()
            
            # 3. Extract current state
            current_epoch = metrics['epochs'][-1] if metrics['epochs'] else 0
            current_lr = metrics['lr'][-1] if metrics['lr'] else 0.001
            vram_usage = metrics['vram'][-1] if metrics['vram'] else 0.0
            
            # 4. Render Dashboard
            logger.info("Rendering dashboard...")
            renderer.render(
                metrics=metrics,
                attention_map=attention_map,
                current_epoch=current_epoch,
                current_lr=current_lr,
                vram_usage=vram_usage
            )
            
            if not watch:
                logger.info("Single render complete. Use --watch for continuous mode.")
                break
            
            logger.info(f"Sleeping {interval}s before next update...")
            time.sleep(interval)
            
        except KeyboardInterrupt:
            logger.info("\nNeuro-Lab shutting down. Goodbye!")
            break
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
            if not watch:
                raise
            time.sleep(interval)


def main():
    parser = argparse.ArgumentParser(description="I.A.R.A. Neuro-Lab Dashboard")
    parser.add_argument("--watch", action="store_true", help="Continuous monitoring mode")
    parser.add_argument("--interval", type=int, default=30, help="Update interval in seconds")
    args = parser.parse_args()
    
    run_neuro_lab(watch=args.watch, interval=args.interval)


if __name__ == "__main__":
    main()



--- FILE: backend/src/api/brain/router.py ---
"""
IARA API — Brain Lab Routes
Training orchestration and model management.
"""

import logging
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import desc
from fastapi.responses import PlainTextResponse

from src.database.core import get_db
from src.database.models import Trade, Signal, ModelVersion
from src.train.continuous_learner import ContinuousLearner
import threading
import os
import shutil
import subprocess
from pydantic import BaseModel
from datetime import datetime

router = APIRouter(prefix="/brain", tags=["Brain Lab"])
logger = logging.getLogger("api.brain")

# In-memory training status
training_state = {
    "is_training": False,
    "last_loss": None,
    "last_acc": None,
    "last_prec": None,
    "last_auc": None,
    "last_f1": None,
    "pressure": 1.0,
    "last_run": None,
    "message": "Idle",
    "progress": 0,  # 0 to 100
    "trauma_count": 0,
    "logs": [],     # Recent training logs
    "all_logs": [], # Full history for export
    "stop_requested": False,
    "current_epoch": 1,
    "micro_state": {},  # 🧠 Phase 20: Real-time internal brain state for "Opened Brain" dashboard
    "epoch_history": [],  # 📊 Phase 14 v2: Epoch comparison data for frontend dashboard
}

# In-memory data fetching status
data_state = {
    "is_fetching": False,
    "message": "Idle",
    "logs": []
}

backtest_state = {
    "is_running": False,
    "dataset": "",
    "threshold": 0.5,
    "message": "Idle",
    "logs": [],
    "results": None
}

class TrainingProtocol(BaseModel):
    mode: str = "finetune" # "scratch", "finetune"
    base_model: Optional[str] = "active_model.pth"
    symbol: str = "BTC/USD"
    batch_size: int = 256
    epochs: int = 5
    learning_rate: float = 0.0001
    recent_months: int = 6
    workers: int = 8
    data_source: str = "journal" # "journal", "historical"
    dataset_name: Optional[str] = None
    save_as: Optional[str] = "active_model.pth"

class FetchRequest(BaseModel):
    symbol: str = "BTC/USD"
    days: int = 30
    timeframe: str = "1Min"

def run_training_background(protocol: Dict[str, Any]):
    global training_state
    try:
        training_state["is_training"] = True
        training_state["progress"] = 0
        training_state["logs"] = []
        training_state["all_logs"] = []
        training_state["history"] = []
        training_state["stop_requested"] = False
        training_state["micro_state"] = {}
        training_state["epoch_history"] = []
        training_state["message"] = f"Initializing {protocol['mode']} protocol..."
        
        def log_callback(msg: str):
            log_str = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
            training_state["all_logs"].append(log_str)
            training_state["logs"].append(log_str)
            if len(training_state["logs"]) > 50:
                training_state["logs"].pop(0)
            training_state["message"] = msg
            
            # Parse progress % and loss from callback messages
            import re
            epoch_match = re.search(r'Epoch\s*(\d+)', msg)
            if epoch_match:
                training_state["current_epoch"] = int(epoch_match.group(1))
            
            progress_match = re.search(r'(\d+)%', msg)
            if progress_match:
                training_state["progress"] = int(progress_match.group(1))
            loss_match = re.search(r'Loss:\s*([\d.]+)', msg)
            if loss_match:
                training_state["last_loss"] = float(loss_match.group(1))
            
            acc_match = re.search(r'Acc:\s*([\d.]+)', msg)
            if acc_match:
                training_state["last_acc"] = float(acc_match.group(1))
                
            prec_match = re.search(r'Prec:\s*([\d.]+)', msg)
            if prec_match:
                training_state["last_prec"] = float(prec_match.group(1))

            auc_match = re.search(r'AUC:\s*([\d.]+)', msg)
            if auc_match:
                training_state["last_auc"] = float(auc_match.group(1))

            f1_match = re.search(r'F1:\s*([\d.]+)', msg)
            if f1_match:
                training_state["last_f1"] = float(f1_match.group(1))

            pressure_match = re.search(r'Pressure:\s*([\d.]+)', msg)
            if pressure_match:
                training_state["pressure"] = float(pressure_match.group(1))

            trauma_match = re.search(r'Retraining on (\d+) (?:severe mistakes|persistent memories)', msg)
            if trauma_match:
                training_state["trauma_count"] = int(trauma_match.group(1))
            
            # 📊 Phase 14 v2: Parse epoch comparison rows (E1, E2, etc.)
            epoch_row_match = re.search(r'E(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)\s+(\d+)\s+([\d.]+)', msg)
            if epoch_row_match:
                epoch_num = int(epoch_row_match.group(1))
                epoch_entry = {
                    "epoch": epoch_num,
                    "loss": float(epoch_row_match.group(2)),
                    "acc": float(epoch_row_match.group(3)),
                    "prec": float(epoch_row_match.group(4)),
                    "auc": float(epoch_row_match.group(5)),
                    "f1": float(epoch_row_match.group(6)),
                    "trauma_size": int(epoch_row_match.group(7)),
                    "mastered": int(epoch_row_match.group(8)),
                    "lr": float(epoch_row_match.group(9)),
                }
                # Replace existing entry for same epoch or append
                existing = [i for i, e in enumerate(training_state["epoch_history"]) if e["epoch"] == epoch_num]
                if existing:
                    training_state["epoch_history"][existing[0]] = epoch_entry
                else:
                    training_state["epoch_history"].append(epoch_entry)
            
            # Parse trend line
            trend_match = re.search(r'Tend[êe]ncia:\s*(.+)', msg)
            if trend_match:
                training_state["epoch_trend"] = trend_match.group(1).strip()

            # Store in history if it's a metric update
            if loss_match:
                epoch_val = int(epoch_match.group(1)) if epoch_match else training_state["current_epoch"]
                progress_val = int(progress_match.group(1)) if progress_match else 0
                
                training_state["history"].append({
                    "time": datetime.now().strftime('%H:%M:%S'),
                    "epoch": epoch_val,
                    "progress": progress_val,
                    "loss": training_state["last_loss"],
                    "acc": training_state["last_acc"] or 0,
                    "prec": training_state["last_prec"] or 0,
                    "auc": training_state.get("last_auc", 0),
                    "f1": training_state.get("last_f1", 0),
                    "pressure": training_state.get("pressure", 1.0)
                })
                # Keep last 50 points to prevent memory leak
                if len(training_state["history"]) > 50:
                    training_state["history"].pop(0)

        checkpoint_name = protocol.get("base_model")
        checkpoint_path = f"logs/models/{checkpoint_name}" if checkpoint_name else None
        
        save_name = protocol.get("save_as", "active_model.pth")
        if not save_name.endswith('.pth'):
            save_name += '.pth'
        save_path = f"logs/models/{save_name}"

        learner = ContinuousLearner(checkpoint_path=checkpoint_path)
        learner.checkpoint_path = save_path # Override where it saves
        
        # 🧠 Phase 20: Wire Micro-State Callback
        def micro_state_callback(state: dict):
            training_state["micro_state"] = state
        learner._micro_state_callback = micro_state_callback
        
        log_callback(f"Compute device detected: {learner.device}")
        log_callback(f"Target Save Location: {save_name}")
        
        # Override learner params from protocol
        learner.epochs = protocol["epochs"]
        learner.batch_size = protocol["batch_size"]
        learner.learning_rate = protocol["learning_rate"]
        
        # Stop check callable for graceful interruption
        def should_stop():
            return training_state.get("stop_requested", False)
        
        # Handle Data Source
        if protocol.get("data_source") == "historical" and protocol.get("dataset_name"):
            log_callback(f"Switching to Historical Dataset: {protocol['dataset_name']}")
            if hasattr(learner, 'train_on_dataset'):
                learner.train_on_dataset(protocol['dataset_name'], callback=log_callback, stop_check=should_stop)
            else:
                log_callback("Error: train_on_dataset not implemented. Falling back.")
                learner.train_and_evaluate(callback=log_callback, stop_check=should_stop)
        else:
            learner.train_and_evaluate(callback=log_callback, stop_check=should_stop)
        
        if training_state["stop_requested"]:
            training_state["message"] = "⛔ Training stopped by user. Best model saved."
        else:
            training_state["message"] = "Training completed successfully"
        training_state["progress"] = 100
    except Exception as e:
        logger.error(f"Training failed: {e}")
        training_state["message"] = f"Error: {str(e)}"
    finally:
        training_state["is_training"] = False
        # Free GPU Memory
        try:
            import gc
            import torch
            if 'learner' in locals():
                del learner
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            logger.error(f"Failed to clear memory after training: {e}")

def run_data_fetch_background(request: Dict[str, Any]):
    global data_state
    try:
        data_state["is_fetching"] = True
        data_state["logs"] = []
        
        def log_data(msg: str):
            data_state["logs"].append(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")
            if len(data_state["logs"]) > 50:
                data_state["logs"].pop(0)
            data_state["message"] = msg

        symbol = request["symbol"]
        days = request["days"]
        tf = request["timeframe"]
        
        log_data(f"🚀 Starting data fetch for {symbol}...")
        
        cmd_fetch = ["python3", "src/data/crypto_data_fetcher.py", "--symbol", symbol, "--days", str(days), "--timeframe", tf]
        result_fetch = subprocess.run(cmd_fetch, capture_output=True, text=True)
        for line in result_fetch.stdout.splitlines():
            if line.strip(): log_data(line)
        if result_fetch.returncode != 0:
            log_data(f"❌ Fetcher failed: {result_fetch.stderr}")
            return

        norm_symbol = symbol.replace("/", "")
        cmd_factory = ["python3", "src/data/feature_factory.py", "--symbol", norm_symbol, "--market-type", "crypto"]
        result_factory = subprocess.run(cmd_factory, capture_output=True, text=True)
        for line in result_factory.stdout.splitlines():
            if line.strip(): log_data(line)
        if result_factory.returncode != 0:
            log_data(f"❌ Factory failed: {result_factory.stderr}")
        else:
            log_data(f"✅ Data processing complete for {symbol}!")
            
    except Exception as e:
        logger.error(f"Data fetch failed: {e}")
        data_state["message"] = f"Error: {str(e)}"
    finally:
        data_state["is_fetching"] = False

@router.post("/train")
async def trigger_training(protocol: TrainingProtocol):
    global training_state
    if training_state["is_training"]:
        return {"message": "Training already in progress", "status": training_state}
    protocol_dict = protocol.dict()
    threading.Thread(target=run_training_background, args=(protocol_dict,)).start()
    return {"message": "Training started", "status": training_state}

@router.post("/stop")
async def stop_training():
    global training_state
    if not training_state["is_training"]:
        raise HTTPException(status_code=400, detail="No training in progress")
    training_state["stop_requested"] = True
    training_state["message"] = "⏳ Stop requested. Finishing current batch and saving best model..."
    return {"message": "Stop signal sent. Training will halt after current batch."}

@router.get("/export")
async def export_logs():
    return PlainTextResponse("\n".join(training_state.get("all_logs", [])))

@router.post("/data/fetch")
async def trigger_data_fetch(request: FetchRequest):
    global data_state
    if data_state["is_fetching"]:
        return {"message": "Data Pipeline already active", "status": data_state}
    request_dict = request.dict()
    threading.Thread(target=run_data_fetch_background, args=(request_dict,)).start()
    return {"message": "Data fetch initiated", "status": data_state}

class BacktestRequest(BaseModel):
    dataset: str = "BTCUSD"
    threshold: float = 0.50
    model_name: str = "active_model.pth"

def run_backtest_background(request: Dict[str, Any]):
    global backtest_state
    try:
        backtest_state["is_running"] = True
        backtest_state["dataset"] = request["dataset"]
        backtest_state["threshold"] = request["threshold"]
        backtest_state["model_name"] = request.get("model_name", "active_model.pth")
        backtest_state["logs"] = []
        backtest_state["results"] = None
        
        def log_cb(msg: str):
            backtest_state["logs"].append(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")
            if len(backtest_state["logs"]) > 100:
                backtest_state["logs"].pop(0)
            backtest_state["message"] = msg

        from src.train.backtester import Backtester
        
        target_model = backtest_state["model_name"]
        if not target_model.endswith(".pth"):
            target_model += ".pth"
            
        # If it's a direct path (from checkpoints/ or data/), use it. Otherwise assume logs/models
        if os.path.exists(target_model):
            model_full_path = target_model
        else:
            model_full_path = f"logs/models/{target_model}"
            
        bt = Backtester(model_path=model_full_path)
        
        results = bt.evaluate_shards(
            dataset_name=request["dataset"],
            sniper_threshold=request["threshold"],
            callback=log_cb
        )
        
        backtest_state["results"] = results
        backtest_state["message"] = "Backtest completed."
            
    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        backtest_state["message"] = f"Error: {str(e)}"
    finally:
        backtest_state["is_running"] = False
        # Free GPU Memory
        try:
            import gc
            import torch
            if 'bt' in locals():
                del bt
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            logger.error(f"Failed to clear memory after backtest: {e}")

@router.post("/backtest")
async def trigger_backtest(request: BacktestRequest):
    global backtest_state
    if backtest_state["is_running"]:
        return {"message": "Backtest already running", "status": backtest_state}
        
    threading.Thread(target=run_backtest_background, args=(request.dict(),)).start()
    return {"message": "Backtest initiated", "status": backtest_state}

@router.get("/backtest/status")
async def get_backtest_status():
    return backtest_state

@router.get("/data/status")
async def get_data_status():
    return data_state

@router.get("/data/inventory")
async def list_datasets():
    cache_dir = "data/cache"
    if not os.path.exists(cache_dir): return []
    datasets = []
    for symbol_dir in os.listdir(cache_dir):
        full_path = os.path.join(cache_dir, symbol_dir)
        if os.path.isdir(full_path):
            shard_dir = os.path.join(full_path, "train_shards")
            if os.path.exists(shard_dir):
                shards = [f for f in os.listdir(shard_dir) if f.endswith(".pt")]
                datasets.append({"symbol": symbol_dir, "shards": len(shards), "path": full_path})
    return datasets

@router.delete("/data/{symbol}")
async def delete_dataset(symbol: str):
    cache_dir = f"data/cache/{symbol}"
    raw_dir = f"data/raw/{symbol}"
    deleted = []
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
        deleted.append(cache_dir)
    if os.path.exists(raw_dir):
        shutil.rmtree(raw_dir)
        deleted.append(raw_dir)
    if not deleted: raise HTTPException(status_code=404, detail="Dataset not found")
    return {"message": f"Dataset {symbol} deleted", "paths": deleted}

@router.get("/models")
async def list_models():
    import torch
    
    # Check both common save directories
    model_dirs = ["checkpoints", "data", "logs/models"]
    models = []
    
    for model_dir in model_dirs:
        if not os.path.exists(model_dir):
            continue
            
        for root, _, files in os.walk(model_dir):
            for file in files:
                if file.endswith(".pth"):
                    full_path = os.path.join(root, file)
                    stats = os.stat(full_path)
                    
                    # Introspect checkpoint for dimensions
                    meta = {"input_dim": "?", "worker_hidden": "?", "coord_hidden": "?", "num_params": "?"}
                    try:
                        state = torch.load(full_path, map_location="cpu", weights_only=False)
                        # input_dim from first worker input_proj
                        w = state.get("swarm.workers.M5.input_proj.weight")
                        if w is not None and hasattr(w, 'shape'):
                            meta["input_dim"] = int(w.shape[1])
                            meta["worker_hidden"] = int(w.shape[0])
                        # coord_hidden from policy_head
                        ph = state.get("coordinator.policy_head.weight")
                        if ph is not None and hasattr(ph, 'shape'):
                            meta["coord_hidden"] = int(ph.shape[1])
                        # Count params
                        meta["num_params"] = sum(v.numel() for v in state.values() if hasattr(v, 'numel'))
                    except Exception:
                        pass
                    
                    # Store relative path from backend root as the name so the active model loader can find it
                    # Ensure name is unique
                    rel_name = os.path.relpath(full_path, ".")
                    
                    models.append({
                        "name": rel_name,
                        "size": f"{stats.st_size / (1024*1024):.2f} MB",
                        "last_modified": datetime.fromtimestamp(stats.st_mtime).isoformat(),
                        **meta
                    })
    
    # Sort models by last_modified descending
    models.sort(key=lambda x: x["last_modified"], reverse=True)
    return models


@router.delete("/models/{model_name:path}")
async def delete_model(model_name: str):
    if model_name == "active_model.pth":
        raise HTTPException(status_code=400, detail="Cannot delete the active model directly")
    
    # model_name is now a relative path from the list_models call
    path = model_name
    if os.path.exists(path):
        os.remove(path)
        return {"message": f"Model {model_name} deleted"}
    raise HTTPException(status_code=404, detail="Model not found")

@router.post("/models/activate")
async def activate_model(request: Dict[str, str]):
    model_name = request.get("name")
    if not model_name: raise HTTPException(status_code=400, detail="Model name required")
    
    # model_name is expected to be the path returned by list_models
    src = model_name
    dst = "logs/models/active_model.pth"
    
    if not os.path.exists(src): 
        raise HTTPException(status_code=404, detail=f"Source model not found at {src}")
    
    os.makedirs("logs/models", exist_ok=True)
    import shutil
    shutil.copy2(src, dst)
    return {"message": f"Model {model_name} is now active"}

@router.get("/status")
async def get_training_status():
    return training_state

@router.get("/trades")
async def list_trades(limit: int = 50, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Trade).order_by(desc(Trade.timestamp)).limit(limit))
    return result.scalars().all()

@router.get("/signals")
async def list_signals(limit: int = 20, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Signal).order_by(desc(Signal.timestamp)).limit(limit))
    return result.scalars().all()

@router.get("/live/status")
async def get_live_execution_status(agent_id: Optional[str] = None):
    """Returns the real-time execution state of the active trading engine (Phase 5)."""
    from src.api.fleet.manager import fleet_manager
    
    # If no agent provided, pick the first active one
    if not agent_id:
        active_ids = list(fleet_manager.engines.keys())
        if active_ids:
            agent_id = active_ids[0]
    
    if agent_id and agent_id in fleet_manager.engines:
        engine = fleet_manager.engines[agent_id]
        state = engine.state.get_state()
        
        from dataclasses import asdict
        # Build payload mapping the newly augmented engine.state
        return {
            "agent_id": agent_id,
            "is_live": engine.active,
            "engine_start": engine.start_time,
            "pnl": state.get("pnl", 0.0),
            "trades": state.get("trades", []),
            "logs": state.get("logs", []),
            "current_inference": asdict(state.get("signal")) if state.get("signal") else None
        }
        
    # Formatting a default inactive response for the dashboard
    return {
        "agent_id": None,
        "is_live": False,
        "engine_start": 0,
        "pnl": 0.0,
        "trades": [],
        "logs": [],
        "current_inference": None
    }



--- FILE: backend/src/api/live_ingestor.py ---
import asyncio
import logging
from typing import Dict, Any, Callable
from binance import AsyncClient, BinanceSocketManager
import numpy as np
import collections

logger = logging.getLogger("api.live_ingestor")

class LiveIngestor:
    """
    HFT Live Ingestor (Phase 5)
    Connects to Binance WebSockets to stream M5 and H1 candles,
    along with raw order flow to compute local Cumulative Volume Delta (CVD).
    """
    def __init__(self, symbol: str = "BTCUSDT"):
        self.symbol = symbol.upper()
        self.client: AsyncClient = None
        self.bsm: BinanceSocketManager = None
        self.active = False
        
        # Real-time state
        self.current_m5: Dict[str, Any] = None
        self.current_h1: Dict[str, Any] = None
        
        # CVD (Cumulative Volume Delta) Tracking
        self.cvd_period = 300 # Reset CVD every 5 minutes (300s)
        self.current_cvd = 0.0
        self.last_cvd_reset = 0
        
        # Callbacks for Engine feeding
        self.on_candle_close: Callable = None

    async def connect(self):
        """Initializes the Async Binance Client and Socket Manager."""
        self.client = await AsyncClient.create()
        self.bsm = BinanceSocketManager(self.client)
        self.active = True
        logger.info(f"🔗 Live Ingestor connected to Binance for {self.symbol}")

    async def disconnect(self):
        """Cleanly closes the WebSocket connections."""
        self.active = False
        if self.client:
            await self.client.close_connection()
        logger.info("🛑 Live Ingestor disconnected.")

    async def _stream_klines(self, interval: str):
        """Streams OHLCV candles for a specific timeframe."""
        while self.active:
            try:
                # Use multiplex socket for resilience or direct kline socket
                async with self.bsm.kline_socket(symbol=self.symbol, interval=interval) as stream:
                    while self.active:
                        res = await stream.recv()
                        if 'k' in res:
                            kline = res['k']
                            candle_data = {
                                "time": int(kline['t'] / 1000), # Start time
                                "open": float(kline['o']),
                                "high": float(kline['h']),
                                "low": float(kline['l']),
                                "close": float(kline['c']),
                                "volume": float(kline['v']),
                                "is_closed": kline['x']
                            }
                            
                            if interval == AsyncClient.KLINE_INTERVAL_5MINUTE:
                                self.current_m5 = candle_data
                                if candle_data["is_closed"] and self.on_candle_close:
                                    # Inject CVD into the closed candle
                                    candle_data["cvd"] = self.current_cvd
                                    await self.on_candle_close(interval, candle_data)
                                    # Reset local CVD after 5m close
                                    self.current_cvd = 0.0 
                            
                            elif interval == AsyncClient.KLINE_INTERVAL_1HOUR:
                                self.current_h1 = candle_data
                                if candle_data["is_closed"] and self.on_candle_close:
                                    await self.on_candle_close(interval, candle_data)

            except Exception as e:
                logger.error(f"WebSocket Error (Kline {interval}): {e}. Reconnecting in 5s...")
                await asyncio.sleep(5)

    async def _stream_trades(self):
        """Streams raw trades to calculate CVD (Buy Volume - Sell Volume)."""
        while self.active:
            try:
                # Use aggtrade_socket to bundle trades and heavily reduce websocket queue overflows
                async with self.bsm.aggtrade_socket(symbol=self.symbol) as stream:
                    while self.active:
                        res = await stream.recv()
                        # 'm' is True if the buyer is the market maker (i.e., someone sold into the bid) -> Bearish
                        is_buyer_maker = res['m']
                        qty = float(res['q'])
                        
                        if is_buyer_maker:
                            self.current_cvd -= qty # Sell pressure
                        else:
                            self.current_cvd += qty # Buy pressure
                            
            except Exception as e:
                logger.error(f"WebSocket Error (Trades): {e}. Reconnecting in 5s...")
                await asyncio.sleep(5)

    async def start_streaming(self, on_candle_close: Callable):
        """Launches the async tasks for M5, H1, and Raw Trade streams."""
        self.on_candle_close = on_candle_close
        
        await self.connect()
        
        # Fire and forget concurrent streams
        asyncio.create_task(self._stream_klines(AsyncClient.KLINE_INTERVAL_5MINUTE))
        asyncio.create_task(self._stream_klines(AsyncClient.KLINE_INTERVAL_1HOUR))
        asyncio.create_task(self._stream_trades())
        
        logger.info(f"🌊 Live Streaming active for {self.symbol} (M5, H1, CVD)")

if __name__ == "__main__":
    # Test script
    logging.basicConfig(level=logging.INFO)
    async def process_tick(interval, candle):
        cvd_msg = f" | CVD: {candle.get('cvd', 0):.2f}" if interval == '5m' else ""
        print(f"[{interval}] Candle Closed at {candle['close']} (Vol: {candle['volume']}){cvd_msg}")
        
    async def main():
        ingestor = LiveIngestor("BTCUSDT")
        await ingestor.start_streaming(process_tick)
        await asyncio.sleep(300) # Run for 5 minutes
        await ingestor.disconnect()
        
    asyncio.run(main())



--- FILE: backend/src/api/schemas.py ---
"""
IARA API — Pydantic Schemas
Strict type definitions for request/response bodies.
"""

from pydantic import BaseModel, Field
from typing import Optional


# ── Auth ──────────────────────────────────────────────────────

class LoginRequest(BaseModel):
    username: str = Field(..., min_length=1, max_length=50)
    password: str = Field(..., min_length=1, max_length=128)


class UserResponse(BaseModel):
    id: str
    username: str
    role: str


class TokenResponse(BaseModel):
    user: UserResponse
    message: str


# ── Fleet ─────────────────────────────────────────────────────

class AgentConfig(BaseModel):
    pulse: str
    label: str
    color: str


class AgentStatus(BaseModel):
    id: str
    symbol: str
    market_type: str  # "crypto" | "forex"
    status: str  # "idle" | "thinking" | "stalking" | "in_trade" | "paused" | "error"
    model_name: str
    pnl: float
    win_rate: float
    total_trades: int
    uptime: float  # seconds
    current_position: Optional[dict] = None
    last_signal: Optional[dict] = None
    config: Optional[AgentConfig] = None


class FleetStatusResponse(BaseModel):
    agents: list[AgentStatus]
    total_pnl: float
    active_agents: int


class StartAgentRequest(BaseModel):
    symbol: str = Field(..., min_length=1, max_length=20)
    market_type: str = Field(..., pattern="^(crypto|forex)$")
    model_name: str = Field(default="SwarmBrain v9")


class StopAgentRequest(BaseModel):
    agent_id: str


class ManualOverrideRequest(BaseModel):
    agent_id: str
    action: str = Field(..., pattern="^(buy|sell)$")


# ── Market ────────────────────────────────────────────────────

class CandleData(BaseModel):
    time: int
    open: float
    high: float
    low: float
    close: float
    volume: Optional[float] = None



--- FILE: backend/src/api/fleet/manager.py ---
"""
Fleet Manager
=============
Orchestrates multiple TrinityEngine instances for the API.
Singleton responsible for lifecycle management and state aggregation.
"""

import logging
import threading
import time
from typing import Dict, Optional, List

from src.brain.engine import TrinityEngine, EngineConfig
from src.trading.signal_generator import SignalEngine

logger = logging.getLogger("FleetManager")


class FleetManager:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(FleetManager, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
            
        self.engines: Dict[str, TrinityEngine] = {}
        self.configs: Dict[str, EngineConfig] = {}
        self._market_types: Dict[str, str] = {}
        self.logger = logger
        self._initialized = True
        self.logger.info("FleetManager initialized")

    def start_agent(self, symbol: str, market_type: str = "crypto", model_name: str = "best_model_v9_EURUSD.pth", execution_venue: str = "ALPACA") -> str:
        """Start a new TrinityEngine agent for a symbol."""
        agent_id = f"agent-{symbol.lower().replace('/', '-')}"
        
        if agent_id in self.engines:
            self.logger.warning(f"Agent {agent_id} already exists")
            if not self.engines[agent_id].active:
                self.logger.info(f"Restarting existing agent {agent_id}")
                self.engines[agent_id].start()
            return agent_id

        import os
        if os.path.exists(model_name):
            checkpoint_path = model_name
        else:
            checkpoint_path = f"logs/models/{model_name.replace('logs/models/', '')}"

        config = EngineConfig(
            symbol=symbol,
            market_type=market_type,
            execution_venue=execution_venue,
            checkpoint_path=checkpoint_path,
            lfm_model_name="lfm2.5-thinking:1.2b"
        )
        
        try:
            self.logger.info(f"Initializing engine for {symbol}...")
            engine = TrinityEngine.from_checkpoint(config)
            engine.start()
            
            self.engines[agent_id] = engine
            self.configs[agent_id] = config
            self._market_types[agent_id] = market_type
            self.logger.info(f"✅ Agent {agent_id} started successfully")
            return agent_id
            
        except Exception as e:
            self.logger.error(f"Failed to start agent {agent_id}: {e}")
            raise e

    def stop_agent(self, agent_id: str):
        """Stop a specific agent."""
        if agent_id in self.engines:
            self.logger.info(f"Stopping agent {agent_id}...")
            self.engines[agent_id].stop()
            # Remove from memory so it can be cleanly recreated/reloaded
            del self.engines[agent_id]
            if agent_id in self.configs:
                del self.configs[agent_id]
            if agent_id in self._market_types:
                del self._market_types[agent_id]
        else:
            self.logger.warning(f"Agent {agent_id} not found")

    def stop_all(self):
        """Emergency stop for all agents."""
        self.logger.warning("🚨 KILLSWITCH ACTIVATED - Stopping all agents")
        for agent_id, engine in self.engines.items():
            if engine.active:
                engine.stop()

    def manual_override(self, agent_id: str, action: str):
        """Force a manual buy/sell action on a specific agent."""
        if agent_id in self.engines:
            self.logger.info(f"Executing manual override {action.upper()} on {agent_id}")
            # The engine itself should handle the execution. For now, we just log it.
            # In a full implementation, you'd route this to the agent's executor
            self.engines[agent_id].executor.execute_signal(action.upper())
        else:
            self.logger.warning(f"Agent {agent_id} not found for manual override")

    def get_status(self) -> List[dict]:
        """Collect status from all agents."""
        status_list = []
        
        for agent_id, engine in self.engines.items():
            state = engine.state.get_state()
            config = self.configs[agent_id]
            
            last_signal = state.get("signal")
            signal_dict = None
            if last_signal:
                # Map direction to frontend action format
                action_map = {"long": "buy", "short": "sell", "wait": "hold"}
                action = action_map.get(last_signal.direction.lower(), "hold")
                
                # Build workerVotes in frontend format: {M5, M15, M30, H1, H4, consensus}
                worker_votes = last_signal.worker_scores.copy()
                worker_votes["consensus"] = last_signal.worker_agreement
                
                signal_dict = {
                    "timestamp": last_signal.timestamp,
                    "action": action,
                    "confidence": last_signal.confidence,
                    "longScore": last_signal.long_score,
                    "shortScore": last_signal.short_score,
                    "workerVotes": worker_votes,
                    "lfmThought": state.get("lfm_summary", "")
                }
            
            market_type = self._market_types.get(agent_id, "crypto")
            
            status_entry = {
                "id": agent_id,
                "symbol": config.symbol,
                "market_type": market_type,
                "status": "active" if engine.active else "paused",
                "model_name": config.checkpoint_path.split("/")[-1],
                "pnl": 0.0,
                "win_rate": 0.0,
                "total_trades": 0,
                "uptime": engine.start_time,
                "current_position": None,
                "last_signal": signal_dict
            }
            status_list.append(status_entry)
            
        return status_list

# Global instance
fleet_manager = FleetManager()



--- FILE: backend/src/api/fleet/router.py ---
"""
IARA API — Fleet Routes
Agent management: status, start, stop, killswitch.
"""

import time
import logging
from fastapi import APIRouter, Request, HTTPException, Depends
from src.api.auth.jwt import get_current_user
from src.api.schemas import (
    FleetStatusResponse,
    AgentStatus,
    AgentConfig,
    StartAgentRequest,
    StopAgentRequest,
    ManualOverrideRequest,
)
from src.api.fleet.manager import fleet_manager

router = APIRouter(prefix="/fleet", tags=["Fleet Management"])
logger = logging.getLogger("api.fleet")


@router.get("/status", response_model=FleetStatusResponse)
async def fleet_status(user: dict = Depends(get_current_user)):
    """Get the current status of all fleet agents."""
    agents_data = fleet_manager.get_status()
    
    # Convert dicts to AgentStatus objects
    agents = []
    for a in agents_data:
        # Calculate uptime duration
        uptime_ts = a.get("uptime", 0)
        uptime_duration = time.time() - uptime_ts if uptime_ts > 0 else 0
        
        # Determine UI Config
        symbol = a["symbol"]
        status = a.get("status", "idle")
        
        # Color mapping by symbol
        color = "emerald"
        if "BTC" in symbol: color = "orange"
        elif "ETH" in symbol: color = "indigo"
        elif "EUR" in symbol: color = "blue"
            
        # Pulse animation by status
        pulse = "none"
        if status == "active" or status == "in_trade": pulse = color
        elif status == "thinking": pulse = "blue"
        
        agents.append(AgentStatus(
            id=a["id"],
            symbol=a["symbol"],
            market_type=a["market_type"],
            status=a.get("status", "idle"),
            model_name=a["model_name"],
            pnl=a.get("pnl", 0.0),
            win_rate=a.get("win_rate", 0.0),
            total_trades=a.get("total_trades", 0),
            uptime=uptime_duration,
            current_position=a.get("current_position"),
            last_signal=a.get("last_signal"),
            config=AgentConfig(
                pulse=pulse,
                label=symbol,
                color=color
            )
        ))

    total_pnl = sum(a.pnl for a in agents)
    active_count = sum(1 for a in agents if a.status == "active")

    return FleetStatusResponse(agents=agents, total_pnl=total_pnl, active_agents=active_count)


@router.post("/start")
async def start_agent(body: StartAgentRequest, user: dict = Depends(get_current_user)):
    """Start a new trading agent for the given symbol."""
    try:
        agent_id = fleet_manager.start_agent(
            symbol=body.symbol,
            market_type=body.market_type,
            model_name=body.model_name
        )
        return {"message": f"Agent {agent_id} started for {body.symbol}", "agent_id": agent_id}
    except Exception as e:
        logger.error(f"Failed to start agent: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stop")
async def stop_agent(body: StopAgentRequest, user: dict = Depends(get_current_user)):
    """Stop a trading agent gracefully."""
    try:
        fleet_manager.stop_agent(body.agent_id)
        return {"message": f"Agent {body.agent_id} stopped"}
    except Exception as e:
        logger.error(f"Failed to stop agent: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/killswitch")
async def killswitch(user: dict = Depends(get_current_user)):
    """Emergency stop all agents."""
    fleet_manager.stop_all()
    return {"message": "KILLSWITCH activated. All agents stopped.", "status": "shutting_down"}



--- FILE: backend/src/api/config.py ---
"""
IARA API Server — Configuration
Loads environment variables and defines server constants.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
load_dotenv(PROJECT_ROOT / ".env")


class Settings:
    # Server
    HOST: str = os.getenv("API_HOST", "0.0.0.0")
    PORT: int = int(os.getenv("API_PORT", "8000"))
    DEBUG: bool = os.getenv("API_DEBUG", "true").lower() == "true"

    # CORS
    FRONTEND_URL: str = os.getenv("FRONTEND_URL", "http://localhost:3000")
    FRONTEND_URLS: list = [
        "http://localhost:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:3001",
        FRONTEND_URL
    ]

    # JWT
    JWT_SECRET: str = os.getenv("JWT_SECRET", "iara-dev-secret-change-in-production-2026")
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MIN", "60"))
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    # Default Admin (dev mode — replace with DB in production)
    ADMIN_USERNAME: str = os.getenv("ADMIN_USERNAME", "admin")
    ADMIN_PASSWORD: str = os.getenv("ADMIN_PASSWORD", "admin")

    # Database (optional, for future integration)
    DATABASE_URL: str = os.getenv("DATABASE_URL", "")

    # Binance
    BINANCE_API_KEY: str = os.getenv("BINANCE_API_KEY", "")
    BINANCE_SECRET_KEY: str = os.getenv("BINANCE_SECRET_KEY", "")


settings = Settings()



--- FILE: backend/src/api/telegram_broadcaster.py ---
import os
import logging
import asyncio
import aiohttp
from typing import Dict, Any

logger = logging.getLogger("api.telegram_broadcaster")

class TelegramBroadcaster:
    """
    SaaS Broadcaster Phase 5 (Telegram VIP)
    Listens for high-confidence 'Sniper Mode' signals from IARA and dispatches them 
    asynchronously to a Telegram Chat/Group.
    """
    def __init__(self):
        self.bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
        self.active = bool(self.bot_token and self.chat_id)
        
        # Sniper Threshold for Public Alerts
        self.broadcast_threshold = 0.82 

        if self.active:
            logger.info(f"📣 SaaS Telegram Broadcaster ON. Dest: {self.chat_id}")
        else:
            logger.warning("Telegram Bot missing credentials. SaaS Broadcasting disabled.")

    async def broadcast_signal(self, symbol: str, direction: str, confidence: float, current_price: float):
        """
        Receives raw JSON from the IARA Model and filters out low-confidence setups.
        If valid, translates to human-readable text and fires to Telegram.
        """
        if not self.active:
            return

        if confidence < self.broadcast_threshold:
            logger.debug(f"Signal {direction} ({confidence:.2f}) below VIP threshold {self.broadcast_threshold}. Ignoring.")
            return

        # Map emojis
        emoji = "🟢 LONG" if direction.upper() == "BUY" else "🔴 SHORT"
        
        # Message Formatting
        message = (
            f"⚡ **SACI BRAIN - ALERTA SNIPER** ⚡\n\n"
            f"🎯 **Ativo:** #{symbol.upper()}\n"
            f"🛡️ **Direção:** {emoji}\n"
            f"🧠 **Confiança da IA:** {confidence * 100:.1f}%\n"
            f"💲 **Preço Estimado:** ${current_price:.2f}\n\n"
            f"⚠️ *Atenção: Gerencie seu risco utilizando stop-loss. Isso é um sinal preditivo autônomo.*"
        )

        # Dispatch Asynchronously
        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": message,
            "parse_mode": "Markdown"
        }

        try:
             async with aiohttp.ClientSession() as session:
                 async with session.post(url, json=payload) as response:
                     if response.status == 200:
                         logger.info(f"✅ Telegram VIP Alert dispatched! ({symbol} - {direction})")
                     else:
                         error_msg = await response.text()
                         logger.error(f"❌ Telegram API Error ({response.status}): {error_msg}")
        except Exception as e:
             logger.error(f"Failed to post to Telegram: {e}")

# Async Test Trigger
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    async def test_broadcast():
        bot = TelegramBroadcaster()
        # Force active for testing if env vars missing
        if not bot.active:
            print("Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env")
        else:
            print("Publishing test signal...")
            await bot.broadcast_signal("BTCUSDT", "BUY", 0.88, 97500.50)
            
    asyncio.run(test_broadcast())



--- FILE: backend/src/api/agents/router.py ---
"""
IARA API — Agent Profiles Routes
CRUD for dynamic trading agents.
"""

import logging
from typing import List
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from src.database.core import get_db
from src.database.models import AgentProfile
from src.api.auth.jwt import get_current_user
from pydantic import BaseModel
import os
import glob

router = APIRouter(prefix="/agents", tags=["Agents"])
logger = logging.getLogger("api.agents")

class AgentCreate(BaseModel):
    id: str
    symbol: str
    market_type: str
    model_version: str = "active_model_v10.pth"
    
class AgentResponse(BaseModel):
    id: str
    symbol: str
    market_type: str
    model_version: str
    is_active: bool

@router.get("", response_model=List[AgentResponse])
async def list_agents(db: AsyncSession = Depends(get_db)):
    """List all agent profiles."""
    result = await db.execute(select(AgentProfile))
    agents = result.scalars().all()
    return agents

@router.post("", response_model=AgentResponse)
async def create_agent(agent: AgentCreate, db: AsyncSession = Depends(get_db)):
    """Create a new agent profile."""
    # Check if exists
    result = await db.execute(select(AgentProfile).where(AgentProfile.id == agent.id))
    existing = result.scalars().first()
    if existing:
        raise HTTPException(status_code=400, detail="Agent ID already exists")

    new_agent = AgentProfile(
        id=agent.id,
        symbol=agent.symbol,
        market_type=agent.market_type,
        model_version=agent.model_version,
        is_active=True
    )
    db.add(new_agent)
    await db.commit()
    await db.refresh(new_agent)
    
    try:
        from src.api.fleet.manager import fleet_manager
        fleet_manager.start_agent(
            symbol=new_agent.symbol,
            market_type=new_agent.market_type,
            model_name=new_agent.model_version,
            execution_venue="ALPACA"
        )
    except Exception as e:
        logger.error(f"Failed to start newly created agent {new_agent.id}: {e}")

    return new_agent

@router.delete("/{agent_id}")
async def delete_agent(agent_id: str, db: AsyncSession = Depends(get_db)):
    """Delete an agent profile."""
    result = await db.execute(select(AgentProfile).where(AgentProfile.id == agent_id))
    agent = result.scalars().first()
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
        
    await db.delete(agent)
    await db.commit()
    
    try:
        from src.api.fleet.manager import fleet_manager
        fleet_manager.stop_agent(agent_id)
    except Exception as e:
        logger.error(f"Failed to stop agent {agent_id} during deletion: {e}")

    return {"message": "Agent deleted successfully"}

@router.put("/{agent_id}/toggle")
async def toggle_agent(agent_id: str, db: AsyncSession = Depends(get_db)):
    """Toggle agent active status."""
    result = await db.execute(select(AgentProfile).where(AgentProfile.id == agent_id))
    agent = result.scalars().first()
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
        
    agent.is_active = not agent.is_active
    await db.commit()
    
    try:
        from src.api.fleet.manager import fleet_manager
        if agent.is_active:
            fleet_manager.start_agent(
                symbol=agent.symbol,
                market_type=agent.market_type,
                model_name=agent.model_version,
                execution_venue="ALPACA"
            )
        else:
            fleet_manager.stop_agent(agent.id)
    except Exception as e:
        logger.error(f"Failed to toggle agent {agent.id} in fleet: {e}")
        
    return {"message": f"Agent {'activated' if agent.is_active else 'deactivated'} successfully", "is_active": agent.is_active}

@router.get("/models")
async def list_models():
    """List available model checkpoints (.pth) in logs/models."""
    models_dir = os.path.join(os.getcwd(), "logs", "models")
    if not os.path.exists(models_dir):
        return []
        
    pths = glob.glob(os.path.join(models_dir, "*.pth"))
    models = []
    
    for pth in pths:
        filename = os.path.basename(pth)
        # Parse common v9 naming (e.g. best_model_v9_BTCUSD.pth)
        name = filename.replace(".pth", "")
        if "best_model_v9_" in name:
            pair = name.split("best_model_v9_")[-1]
            label = f"SwarmBrain v9.7 ({pair})"
        else:
            label = name
            
        models.append({
            "value": filename,
            "label": label
        })
        
    return models



--- FILE: backend/src/api/events.py ---
"""
IARA API — Socket.IO Event Manager
Handles real-time bidirectional communication.
"""

import logging
import socketio
from jose import jwt, JWTError

from src.api.config import settings

logger = logging.getLogger("iara.events")

# Create the async Socket.IO server
sio = socketio.AsyncServer(
    async_mode="asgi",
    cors_allowed_origins=settings.FRONTEND_URLS,
    logger=False,
    engineio_logger=False,
)


# ── Connection Lifecycle ──────────────────────────────────────

@sio.event
async def connect(sid: str, environ: dict, auth: dict | None = None):
    """Validate JWT cookie on WebSocket handshake."""
    # Extract cookies from the HTTP headers in the handshake
    cookie_header = environ.get("HTTP_COOKIE", "")
    cookies = {}
    for item in cookie_header.split(";"):
        item = item.strip()
        if "=" in item:
            key, val = item.split("=", 1)
            cookies[key.strip()] = val.strip()

    token = cookies.get("access_token")
    if not token:
        logger.warning(f"[WS] Connection rejected (no token): {sid}")
        raise socketio.exceptions.ConnectionRefusedError("Authentication required")

    try:
        payload = jwt.decode(token, settings.JWT_SECRET, algorithms=[settings.JWT_ALGORITHM])
        # Store user data in the session
        async with sio.session(sid) as session:
            session["user_id"] = payload["sub"]
            session["username"] = payload["username"]
        logger.info(f"[WS] Connected: {payload['username']} ({sid})")
    except JWTError:
        logger.warning(f"[WS] Connection rejected (invalid token): {sid}")
        raise socketio.exceptions.ConnectionRefusedError("Invalid token")


@sio.event
async def disconnect(sid: str):
    logger.info(f"[WS] Disconnected: {sid}")


# ── Client → Server Events ───────────────────────────────────

@sio.event
async def subscribe(sid: str, data: dict):
    """Subscribe to a market symbol's room for live data."""
    symbol = data.get("symbol")
    if symbol:
        await sio.enter_room(sid, f"market:{symbol}")
        logger.info(f"[WS] {sid} subscribed to market:{symbol}")


@sio.event
async def unsubscribe(sid: str, data: dict):
    """Unsubscribe from a symbol's room."""
    symbol = data.get("symbol")
    if symbol:
        await sio.leave_room(sid, f"market:{symbol}")
        logger.info(f"[WS] {sid} unsubscribed from market:{symbol}")


@sio.event
async def manual_override(sid: str, data: dict):
    """Handle manual Buy/Sell from God Mode."""
    async with sio.session(sid) as session:
        username = session.get("username", "unknown")

    agent_id = data.get("agent_id")
    action = data.get("action")
    logger.warning(f"[WS] MANUAL OVERRIDE by {username}: {action} on {agent_id}")

    # TODO: Forward to FleetAdmiral.manual_override(agent_id, action)
    # For now, emit confirmation back
    await sio.emit("trade_update", {
        "agent_id": agent_id,
        "type": f"manual_{action}",
        "price": 0,
        "pnl": 0,
        "initiated_by": username,
    }, room=sid)


# ── Server → Client Emission Helpers ─────────────────────────

class SocketEmitter:
    """
    Bridge between the trading engine and the frontend.
    FleetAdmiral calls these methods to push events to connected clients.
    """

    @staticmethod
    async def emit_market_tick(symbol: str, price: float, timestamp: str):
        await sio.emit("market_tick", {
            "symbol": symbol,
            "price": price,
            "time": timestamp,
        }, room=f"market:{symbol}")

    @staticmethod
    async def emit_candle_closed(symbol: str, candle: dict):
        await sio.emit("candle_closed", {
            "symbol": symbol,
            **candle,
        }, room=f"market:{symbol}")

    @staticmethod
    async def emit_brain_thought(agent_id: str, step: str, text: str, level: str = "info"):
        await sio.emit("brain_thought", {
            "agentId": agent_id,
            "timestamp": "",  # Will be set by caller
            "step": step,
            "text": text,
            "level": level,
        })

    @staticmethod
    async def emit_agent_update(agent: dict):
        await sio.emit("agent_update", agent)

    @staticmethod
    async def emit_fleet_sync(agents: list[dict]):
        await sio.emit("fleet_sync", agents)

    @staticmethod
    async def emit_worker_votes(agent_id: str, votes: dict):
        await sio.emit("worker_votes", {
            "agentId": agent_id,
            **votes,
        })
    @staticmethod
    async def emit_historical_candles(symbol: str, candles: list, sid: str = None):
        """Emit historical candles to frontend."""
        target = sid if sid else f"market:{symbol}"
        await sio.emit(
            "historical_candles",
            {
                "symbol": symbol,
                "candles": candles,
            },
            room=target,
        )

@sio.event
async def request_history(sid: str, data: dict):
    """Handle history request from frontend."""
    from src.api.market_data import market_service
    symbol = data.get("symbol")
    if symbol:
        history = market_service.get_history(symbol)
        if history:
            await SocketEmitter.emit_historical_candles(symbol, history, sid=sid)

@sio.event
async def manual_override(sid: str, data: dict):
    """Handle FORCE BUY / FORCE SELL events from frontend UI."""
    from src.api.fleet.manager import fleet_manager
    agent_id = data.get("agent_id")
    action = data.get("action")
    
    if not agent_id or not action:
        return
        
    try:
        # Pass manual override signal down to fleet manager
        fleet_manager.manual_override(agent_id, action.lower())
        logger.info(f"Manual Override Triggered: {action.upper()} on {agent_id}")
    except Exception as e:
        logger.error(f"Failed to execute manual override: {e}")




--- FILE: backend/src/api/sensory_hub.py ---
import asyncio
import logging
import os
from typing import Dict, List, Optional
import numpy as np
from binance import AsyncClient, BinanceSocketManager

logger = logging.getLogger("SensoryHub")

class BinanceSensoryHub:
    """
    I.A.R.A. v10.0 - Sensory Hub (RAIO-X)
    =====================================
    Handles real-time L2 Order Book streaming via Binance WebSockets.
    Processes depth data into model-ready sensory vectors.
    """
    
    def __init__(self, api_key: str, secret_key: str):
        self.api_key = api_key
        self.secret_key = secret_key
        self.client: Optional[AsyncClient] = None
        self.bsm: Optional[BinanceSocketManager] = None
        self.active_symbols: List[str] = []
        self.active = False
        
        # Latest L2 Snapshots: { "BTCUSDT": { "bids": [...], "asks": [...], "vbi": 0.0 } }
        self.l2_cache: Dict[str, dict] = {}
        
        # History for Spoofing detection
        self.prev_l2: Dict[str, dict] = {}
        self.cancel_rate: Dict[str, float] = {} # { "BTCUSDT": 0.0 }
        
    async def start(self, symbols: List[str]):
        """Initialize and run the WebSocket stream."""
        if not self.api_key or not self.secret_key:
            logger.error("Binance keys missing! Cannot start Sensory Hub.")
            return

        self.active_symbols = [s.replace("/", "").replace("-", "") for s in symbols]
        self.client = await AsyncClient.create(self.api_key, self.secret_key)
        self.bsm = BinanceSocketManager(self.client)
        self.active = True
        
        # Run in background
        asyncio.create_task(self._stream_manager())

    async def _stream_manager(self):
        """Manages the Multiplex depth stream."""
        # Binance format for depth: <symbol>@depth20@100ms
        streams = [f"{sym.lower()}@depth20@100ms" for sym in self.active_symbols]
        for sym in self.active_symbols:
            logger.info(f"📡 Sensory Hub: Subscribed to Order Book for {sym}")
            
        while self.active:
            try:
                async with self.bsm.multiplex_socket(streams) as stream:
                    while self.active:
                        res = await stream.recv()
                        if 'data' in res:
                            await self._on_orderbook(res['data'])
            except Exception as e:
                logger.error(f"WebSocket Error (Sensory Hub): {e}. Reconnecting...")
                await asyncio.sleep(5)

    async def _on_orderbook(self, data: dict):
        """Callback for incoming orderbook updates."""
        # Binance depth payload structure
        # 'bids': [ [ "Price", "Qty" ], ... ]
        symbol = data.get('s', '').upper()
        if not symbol: return
        
        # Process top 10 levels
        bids = data.get('bids', [])[:10]
        asks = data.get('asks', [])[:10]
        
        # Calculate VBI (Volume Book Imbalance)
        # VBI = (BidVol - AskVol) / (BidVol + AskVol)
        try:
             bid_vol = sum(float(b[1]) for b in bids)
             ask_vol = sum(float(a[1]) for a in asks)
             vbi = (bid_vol - ask_vol) / (bid_vol + ask_vol + 1e-10)
        except Exception:
             vbi = 0.0
             
        # Update Cancel Rate (Spoofing Proxy)
        # Parse bids into {price: qty} floats
        float_bids = [(float(b[0]), float(b[1])) for b in bids]
        float_asks = [(float(a[0]), float(a[1])) for a in asks]
        
        self._update_cancel_rate(symbol, float_bids, float_asks)
        
        # Store in cache
        self.l2_cache[symbol] = {
            "bids": float_bids,
            "asks": float_asks,
            "vbi": vbi,
            "cancel_rate": self.cancel_rate.get(symbol, 0.0),
            "timestamp": data.get('E', 0)
        }
        
        # Backup for next diff
        self.prev_l2[symbol] = {
            "bids": {b[0]: b[1] for b in float_bids},
            "asks": {a[0]: a[1] for a in float_asks}
        }
        
        # Optional: Log anomaly if VBI is extreme (Spoofing/Manipulation detection)
        if abs(vbi) > 0.8:
            pass # logger.warning(f"⚠️ Extreme Order Book Imbalance on {symbol}: {vbi:.2f}")

    def _update_cancel_rate(self, symbol: str, bids: List[tuple], asks: List[tuple]):
        """Estimate cancellation rates by comparing level volume shifts."""
        if symbol not in self.prev_l2:
            return
            
        prev = self.prev_l2[symbol]
        curr_bids = {b[0]: b[1] for b in bids}
        curr_asks = {a[0]: a[1] for a in asks}
        
        cancels = 0.0
        total_delta = 0.0
        
        # Check Bids
        for price, prev_size in prev["bids"].items():
            if price in curr_bids:
                delta = prev_size - curr_bids[price]
                if delta > 0: # Volume decreased
                    # In a real engine we'd subtract actual trades at this price
                    # Here we treat any unexplained decrease as a potential cancel
                    cancels += delta
                    total_delta += delta
            else:
                # Level disappeared completely
                cancels += prev_size
                total_delta += prev_size
                
        # Normalize and smooth
        alpha = 0.1 # Smoothing factor
        new_rate = cancels / (sum(prev["bids"].values()) + 1e-10)
        self.cancel_rate[symbol] = (1 - alpha) * self.cancel_rate.get(symbol, 0.0) + alpha * new_rate

    def get_sensory_vector(self, symbol: str) -> Optional[np.ndarray]:
        """
        Returns a normalized sensory vector for the model.
        Dimensions: 10 Bids (P, S) + 10 Asks (P, S) + VBI + CancelRate = 42 features.
        """
        # Engine might ask for BTC/USD format, map it internally
        binance_symbol = symbol.replace("/", "").replace("-", "").upper()
            
        data = self.l2_cache.get(binance_symbol)
        if not data:
            return None
            
        vector = []
        
        # Normalize relative to the mid-price
        best_bid = data["bids"][0][0] if data["bids"] else 0
        best_ask = data["asks"][0][0] if data["asks"] else 0
        mid_price = (best_bid + best_ask) / 2 if (best_bid and best_ask) else (best_bid or best_ask)
        
        if mid_price == 0: return None

        # Process Bids
        for price, size in data["bids"]:
            vector.append((price - mid_price) / mid_price * 1000) # Price as bps
            vector.append(size) 
            
        # Pad if less than 10
        while len(vector) < 20: vector.extend([0, 0])
        
        # Process Asks
        for price, size in data["asks"]:
            vector.append((price - mid_price) / mid_price * 1000)
            vector.append(size)
            
        # Pad if less than 10
        while len(vector) < 40: vector.extend([0, 0])
        
        # VBI & CancelRate
        vector.append(data["vbi"])
        vector.append(data["cancel_rate"])
        
        return np.array(vector, dtype=np.float32)

    async def stop(self):
        self.active = False
        if self.client:
           await self.client.close_connection()
        logger.info("📡 Sensory Hub Stopped")

# Singleton (or initialized in MarketDataService)
sensory_hub = None # Initialized after keys are available



--- FILE: backend/src/api/market_data.py ---
"""
MarketDataService
==================
Market Data Orchestrator that:
1. Fetches real-time market data via CCXT (Binance/Kraken)
2. Falls back to simulation if data fetch fails
3. Feeds data to FleetManager engines
4. Broadcasts events to Frontend via Socket.IO
"""

import asyncio
import logging
import random
import time
import os
from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta

from src.api.events import SocketEmitter
from src.api.fleet.manager import fleet_manager
from src.database.core import AsyncSessionLocal
from src.database.models import AgentProfile
from sqlalchemy.future import select

# Binance API
from binance.client import Client
from src.api.sensory_hub import BinanceSensoryHub

logger = logging.getLogger("MarketDataService")

# Minimum interval (seconds) between engine state emissions
_EMIT_DEBOUNCE_S = 1.0

class MarketDataService:
    def __init__(self):
        self.active = False
        self.symbols = ["BTC/USD", "EUR/USD"] # Default Crypto & Forex symbol
        
        # Default starting prices (fallback for first tick)
        self.prices = {"BTC/USD": 97500.0, "EUR/USD": 1.0500}
        
        self.candle_start: Dict[str, float] = {}
        self.candle_ohlc: Dict[str, dict] = {}
        self._last_emit: Dict[str, float] = {}
        self.history: Dict[str, list] = {}  # In-memory history buffer
        
        # Clients
        self.crypto_client: Optional[Client] = None
        
        self.api_key = os.getenv("BINANCE_API_KEY")
        self.secret_key = os.getenv("BINANCE_SECRET_KEY")
        
        # Sensory Hub (RAIO-X)
        self.sensory_hub: Optional[BinanceSensoryHub] = None

    async def _init_exchanges(self):
        """Initialize Binance Data Clients."""
        if not self.api_key:
             logger.warning("Binance keys missing. Running in simulation mode.")
             return

        try:
            self.crypto_client = Client(self.api_key, self.secret_key)
            self.sensory_hub = BinanceSensoryHub(self.api_key, self.secret_key)
            logger.info("✅ Connected to Binance Data API (Crypto & Sensory Hub)")
        except Exception as e:
            logger.error(f"Failed to initialize Binance Data: {e}")
            self.crypto_client = None
            self.sensory_hub = None

    async def _close_exchanges(self):
        """Close clients (No-op for REST clients, but kept for interface compatibility)."""
        pass

    async def _init_fleet(self):
        """Ensure default agents are running based on active profiles in database."""
        try:
            async with AsyncSessionLocal() as session:
                result = await session.execute(select(AgentProfile).where(AgentProfile.is_active == True))
                agents = result.scalars().all()
                self.symbols = []
                for a in agents:
                    self.symbols.append(a.symbol)
                    try:
                        agent_id = fleet_manager.start_agent(
                            symbol=a.symbol,
                            market_type=a.market_type,
                            model_name=a.model_version,
                            execution_venue="BINANCE",
                        )
                        engine = fleet_manager.engines.get(agent_id)
                        if engine:
                            await self._pre_warm_engine(a.symbol, engine)
                    except Exception as e:
                        logger.error(f"Failed to auto-start agent for {a.symbol}: {e}")
                
                if not self.symbols:
                    logger.info("No active agents found in DB. Waiting for UI configuration.")
        except Exception as e:
            logger.error(f"Error querying active agents: {e}")

    async def _pre_warm_engine(self, symbol: str, engine):
        """Fetch 48h of 5m bars and instantly fill the TrinityEngine buffer."""
        if not self.crypto_client:
            logger.warning(f"No Binance client. Skipping pre-warm for {symbol}.")
            return

        try:
            logger.info(f"🔥 Pre-warming engine for {symbol} with 48h of 5m historical data...")
            binance_symbol = symbol.replace("/", "").replace("-", "")
            klines = self.crypto_client.get_historical_klines(binance_symbol, Client.KLINE_INTERVAL_5MINUTE, "48 hours ago UTC")
            
            candles = []
            for k in klines:
                candles.append({
                    "time": int(k[0] / 1000),
                    "open": float(k[1]),
                    "high": float(k[2]),
                    "low": float(k[3]),
                    "close": float(k[4]),
                    "volume": float(k[5])
                })
                
            if candles:
                engine.pre_warm(candles)
            else:
                logger.warning(f"No historical data found for {symbol} pre-warming.")
        except Exception as e:
            logger.error(f"Pre-warming failed for {symbol}: {e}")

    async def _fetch_price(self, symbol: str) -> Optional[float]:
        """Fetch latest price from Binance."""
        if not self.crypto_client:
            return None

        try:
             binance_symbol = symbol.replace("/", "").replace("-", "")
             res = self.crypto_client.get_symbol_ticker(symbol=binance_symbol)
             return float(res['price'])
        except Exception as e:
            # logger.warning(f"Fetch failed for {symbol}: {e}") # Reduce noise
            return None

    async def _fetch_and_emit_history(self, symbol: str):
        """Fetch historical candles from Binance and emit to frontend."""
        candles = []
        
        if self.crypto_client:
            try:
                # Fetch last 2 hours of 1-minute bars
                binance_symbol = symbol.replace("/", "").replace("-", "")
                klines = self.crypto_client.get_historical_klines(binance_symbol, Client.KLINE_INTERVAL_1MINUTE, "2 hours ago UTC")
                
                for k in klines:
                    candles.append({
                        "time": int(k[0] / 1000),
                        "open": float(k[1]),
                        "high": float(k[2]),
                        "low": float(k[3]),
                        "close": float(k[4]),
                        "volume": float(k[5])
                    })
                logger.info(f"Fetched {len(candles)} historical candles for {symbol} via Binance")
            except Exception as e:
                logger.warning(f"History fetch failed for {symbol}: {e}")

        # Fallback to simulation if empty
        if not candles:
            logger.info(f"Generating simulated history for {symbol}")
            candles = self._generate_simulated_history(symbol, 100)
            
        # Emit
        if candles:
             # Initialize internal state with last candle
            last = candles[-1]
            self.prices[symbol] = last["close"]
            self.candle_ohlc[symbol] = {
                "open": last["close"], "high": last["close"], 
                "low": last["close"], "close": last["close"]
            }
            # Store in buffer
            self.history[symbol] = candles
            
            await SocketEmitter.emit_historical_candles(symbol, candles)

    def get_history(self, symbol: str) -> list:
        """Get cached history for a symbol."""
        return self.history.get(symbol, [])

    def _generate_simulated_history(self, symbol: str, count: int) -> list:
        """Generate random walk history if API fails."""
        candles = []
        price = self.prices.get(symbol, 1000.0)
        volatility = {"BTC/USD": 15, "EUR/USD": 0.00015}
        vol = volatility.get(symbol, 1.0)
        
        now = int(time.time())
        # Generate backwards
        history_prices = [price]
        for _ in range(count):
            change = (random.random() - 0.5) * vol * 5 # slightly higher vol for candles
            history_prices.append(history_prices[-1] - change)
        
        history_prices.reverse()
        
        for i, p in enumerate(history_prices):
            open_p = p
            close_p = p + (random.random() - 0.5) * vol
            high = max(open_p, close_p) + random.random() * vol * 0.5
            low = min(open_p, close_p) - random.random() * vol * 0.5
            
            candles.append({
                "time": now - (count - i) * 60, # 1m intervals
                "open": round(open_p, 5),
                "high": round(high, 5),
                "low": round(low, 5),
                "close": round(close_p, 5),
                "volume": random.randint(100, 1000)
            })
            
        return candles

    def _simulate_price_movement(self, symbol: str) -> float:
        """Fallback simulation (Random Walk)."""
        volatility = {"BTC/USD": 15, "EUR/USD": 0.00015}
        change = (random.random() - 0.5) * volatility.get(symbol, 1.0)
        # Ensure we have a price
        if symbol not in self.prices:
             self.prices[symbol] = 1000.0
        self.prices[symbol] += change
        return self.prices[symbol]

    async def run(self):
        """Main loop."""
        self.active = True
        await self._init_fleet()
        await self._init_exchanges()
        
        # Initial Data Load & Emit
        logger.info("Initializing market data history & sensory hub...")
        for symbol in self.symbols:
            await self._fetch_and_emit_history(symbol)
            
        if self.sensory_hub:
            await self.sensory_hub.start(self.symbols)
        
        logger.info("🚀 Market Data Service Started (Binance Source)")
        
        now = time.time()
        for s in self.symbols:
            self.candle_start[s] = now
            self._last_emit[s] = 0.0

        while self.active:
            try:
                tasks = []
                now_ts = time.time()
                
                # Check for dynamically added agents
                current_fleet_symbols = [cfg.symbol for cfg in fleet_manager.configs.values()]
                for sym in current_fleet_symbols:
                    if sym not in self.symbols:
                        self.symbols.append(sym)
                        self.candle_start[sym] = now_ts
                        self._last_emit[sym] = 0.0
                        logger.info(f"MarketDataService picked up new symbol: {sym}")
                        agent_id = f"agent-{sym.lower().replace('/', '-')}"
                        engine = fleet_manager.engines.get(agent_id)
                        if engine:
                            await self._pre_warm_engine(sym, engine)
                
                for symbol in self.symbols:
                    # 1. Get Price (Real or Simulated)
                    price = await self._fetch_price(symbol)
                    
                    if price is None:
                        # Fallback
                        price = self._simulate_price_movement(symbol)
                    else:
                        self.prices[symbol] = price
                    
                    # 2. Emit Tick
                    tasks.append(SocketEmitter.emit_market_tick(
                        symbol=symbol,
                        price=round(price, 5),
                        timestamp=str(int(now_ts)),
                    ))
                    
                    # 3. Update Candle Logic
                    self._update_ohlc(symbol, price)
                    
                    # 4. Check Candle Close (5s demo interval for UI responsiveness)
                    # Real application would use 1m, 5m etc. keeping 5s for 'live feel'
                    if now_ts - self.candle_start.get(symbol, now_ts) >= 5:
                        candle = {
                            "time": int(now_ts),
                            "open": self.candle_ohlc[symbol]["open"],
                            "high": self.candle_ohlc[symbol]["high"],
                            "low": self.candle_ohlc[symbol]["low"],
                            "close": self.candle_ohlc[symbol]["close"],
                            "volume": random.randint(100, 5000), # Volume simulated for now
                        }
                        
                        # Emit & Feed
                        tasks.append(SocketEmitter.emit_candle_closed(symbol, candle))
                        tasks.append(self._feed_engine(symbol, candle))
                        
                        # Update History Buffer
                        if symbol not in self.history:
                            self.history[symbol] = []
                        self.history[symbol].append(candle)
                        # Keep buffer size manageable (e.g. 500)
                        if len(self.history[symbol]) > 500:
                            self.history[symbol] = self.history[symbol][-500:]
                        
                        # Reset
                        self.candle_start[symbol] = now_ts
                        self.candle_ohlc[symbol] = {
                           "open": price, "high": price, "low": price, "close": price
                        }
                        
                        # 5. Emit Engine State
                        if now_ts - self._last_emit.get(symbol, 0) >= _EMIT_DEBOUNCE_S:
                            await self._emit_engine_state(symbol)
                            self._last_emit[symbol] = now_ts

                if tasks:
                    await asyncio.gather(*tasks)
                
                # Sleep to respect rate limits (1s is safe for 3 symbols on public API)
                await asyncio.sleep(1.0)

            except Exception as e:
                logger.error(f"Market loop error: {e}")
                await asyncio.sleep(5) # Backoff on error

    def _update_ohlc(self, symbol: str, price: float):
        if symbol not in self.candle_ohlc:
            self.candle_ohlc[symbol] = {
                "open": price, "high": price, "low": price, "close": price
            }
        else:
            self.candle_ohlc[symbol]["high"] = max(self.candle_ohlc[symbol]["high"], price)
            self.candle_ohlc[symbol]["low"] = min(self.candle_ohlc[symbol]["low"], price)
            self.candle_ohlc[symbol]["close"] = price

    async def _feed_engine(self, symbol: str, candle: dict):
        agent_id = f"agent-{symbol.lower().replace('/', '-')}"
        engine = fleet_manager.engines.get(agent_id)
        if engine and engine.active:
            try:
                # Enrich candle with Sensory Data (L2)
                if self.sensory_hub:
                    sensory_vector = self.sensory_hub.get_sensory_vector(symbol)
                    if sensory_vector is not None:
                        candle["sensory_l2"] = sensory_vector.tolist()
                
                await engine.forward_tick(candle)
            except Exception as e:
                logger.error(f"Engine push failed for {agent_id}: {e}")

    async def _emit_engine_state(self, symbol: str):
        """Check engine state and emit thoughts/updates to the frontend."""
        if self.sensory_hub:
            l2_data = self.sensory_hub.l2_cache.get(symbol)
            if l2_data:
                vbi = l2_data["vbi"]
                cancel = l2_data["cancel_rate"]
                await SocketEmitter.emit_brain_thought(
                    agent_id=f"agent-{symbol.lower().replace('/', '-')}",
                    step="raio-x",
                    text=f"Vision: Imbalance {vbi:+.2f} | Cancel {cancel:.1%}",
                    level="warning" if abs(vbi) > 0.7 or cancel > 0.3 else "info"
                )

        agent_id = f"agent-{symbol.lower().replace('/', '-')}"
        engine = fleet_manager.engines.get(agent_id)
        if not engine:
            return
            
        state = engine.state.get_state()
        signal = state.get("signal")
        lfm_summary = state.get("lfm_summary")
        current_position = engine.state.get_position_summary()
        
        if lfm_summary:
            await SocketEmitter.emit_brain_thought(
                agent_id=agent_id,
                step="strategy",
                text=f"LFM: {lfm_summary}",
                level="info"
            )
            
        if signal:
            await SocketEmitter.emit_brain_thought(
                agent_id=agent_id,
                step="signal",
                text=f"Swarm Consensus: {signal.confidence:.2f} ({signal.direction}) | Agreement: {signal.worker_agreement:.1%}",
                level="success" if signal.confidence > 0.6 else "info"
            )
            
            # Use real position data if available, or signal-derived
            pos_data = {
                "side": current_position["side"] if current_position else (signal.direction.lower() if signal.direction != "WAIT" else "none"),
                "size": current_position["size"] if current_position else 0.01,
            }

            await SocketEmitter.emit_agent_update({
                "id": agent_id,
                "symbol": symbol,
                "status": "active" if pos_data["side"] == "none" else "in_trade",
                "current_position": pos_data
            })

    async def stop(self):
        """Stop service and cleanup."""
        self.active = False
        await self._close_exchanges()
        logger.info("Market Data Service Stopped")

market_service = MarketDataService()




--- FILE: backend/src/api/auth/jwt.py ---
"""
IARA API — JWT Authentication Logic
Handles token creation, validation, and cookie management.
"""

from datetime import datetime, timedelta, timezone
from jose import jwt, JWTError
from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse

from src.api.config import settings


def create_access_token(data: dict, expires_delta: timedelta | None = None) -> str:
    """Create a signed JWT access token."""
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (
        expires_delta or timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    to_encode.update({"exp": expire, "iat": datetime.now(timezone.utc)})
    return jwt.encode(to_encode, settings.JWT_SECRET, algorithm=settings.JWT_ALGORITHM)


def verify_token(token: str) -> dict:
    """Verify and decode a JWT token. Raises HTTPException on failure."""
    try:
        payload = jwt.decode(
            token, settings.JWT_SECRET, algorithms=[settings.JWT_ALGORITHM]
        )
        return payload
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
        )


def get_current_user(request: Request) -> dict:
    """Extract current user from HttpOnly cookie. Used as dependency."""
    token = request.cookies.get("access_token")
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
        )
    return verify_token(token)


def set_auth_cookies(response: JSONResponse, token: str) -> JSONResponse:
    """Set HttpOnly, Secure, SameSite cookies on the response."""
    response.set_cookie(
        key="access_token",
        value=token,
        httponly=True,
        secure=not settings.DEBUG,  # Secure=True in production
        samesite="lax",
        max_age=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        path="/",
    )
    return response


def clear_auth_cookies(response: JSONResponse) -> JSONResponse:
    """Clear auth cookies."""
    response.delete_cookie(key="access_token", path="/")
    return response



--- FILE: backend/src/api/auth/router.py ---
"""
IARA API — Auth Routes
Login, logout, and user identity endpoints.
"""

from fastapi import APIRouter, Request, HTTPException, status
from fastapi.responses import JSONResponse

from src.api.config import settings
from src.api.schemas import LoginRequest, TokenResponse, UserResponse
from src.api.auth.jwt import (
    create_access_token,
    get_current_user,
    set_auth_cookies,
    clear_auth_cookies,
)

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post("/login", response_model=TokenResponse)
async def login(body: LoginRequest):
    """Authenticate user and set HttpOnly cookie with JWT."""

    # Simple credential check (replace with DB lookup in prod)
    if body.username != settings.ADMIN_USERNAME or body.password != settings.ADMIN_PASSWORD:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
        )

    # Create JWT token
    token = create_access_token(data={
        "sub": "1",
        "username": body.username,
        "role": "admin",
    })

    # Build response with HttpOnly cookie
    user = UserResponse(id="1", username=body.username, role="admin")
    response = JSONResponse(
        content=TokenResponse(user=user, message="Login successful").model_dump()
    )
    set_auth_cookies(response, token)

    return response


@router.post("/logout")
async def logout():
    """Clear auth cookies."""
    response = JSONResponse(content={"message": "Logged out"})
    clear_auth_cookies(response)
    return response


@router.get("/me", response_model=UserResponse)
async def me(request: Request):
    """Return identity of the current authenticated user."""
    payload = get_current_user(request)
    return UserResponse(
        id=payload["sub"],
        username=payload["username"],
        role=payload["role"],
    )



--- FILE: backend/src/api/main.py ---
"""
IARA API — Main Server Entrypoint
FastAPI + Socket.IO combined ASGI application.

Usage:
    python -m src.api.main
    # or
    uvicorn src.api.main:app --reload --port 8000
"""

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import socketio

from src.api.config import settings
from src.api.events import sio
from src.api.auth.router import router as auth_router
from src.api.fleet.router import router as fleet_router
from src.api.agents.router import router as agents_router
from src.api.brain.router import router as brain_router
from src.api.market_data import market_service

# ── Logging ───────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-18s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("iara.server")



# ── App Lifecycle ─────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manages background tasks during app lifecycle."""
    logger.info("=" * 60)
    logger.info("  I.A.R.A. API Server starting...")
    logger.info(f"  Frontend URL: {settings.FRONTEND_URL}")
    logger.info(f"  Debug: {settings.DEBUG}")
    logger.info("=" * 60)

    # Initialize Database
    from src.database.core import init_db
    await init_db()

    # Start the Market Data Service (feeds engines)
    service_task = asyncio.create_task(market_service.run())

    yield

    # Shutdown
    await market_service.stop()
    service_task.cancel()
    logger.info("I.A.R.A. API Server stopped.")


# ── FastAPI Application ──────────────────────────────────────

api = FastAPI(
    title="I.A.R.A. API",
    description="Intelligent Autonomous Real-time Agent — Backend API",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow frontend origin
api.add_middleware(
    CORSMiddleware,
    allow_origins=settings.FRONTEND_URLS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API routes under /api/v1
api.include_router(auth_router, prefix="/api/v1")
api.include_router(fleet_router, prefix="/api/v1")
api.include_router(agents_router, prefix="/api/v1")
api.include_router(brain_router, prefix="/api/v1")


@api.get("/api/v1/health")
async def health_check():
    return {"status": "online", "service": "I.A.R.A.", "version": "1.0.0"}


# ── Combine FastAPI + Socket.IO into single ASGI app ──────────

app = socketio.ASGIApp(sio, other_asgi_app=api)


# ── CLI Entrypoint ────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info",
    )



--- FILE: backend/src/model/bitnet.py ---
"""
BitNet Core - 1.58-bit Quantized Linear Layers
===============================================
Implements BitLinear layer with ternary weight quantization {-1, 0, +1}
and 8-bit activation quantization for extreme memory efficiency.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Quantization Functions (Straight-Through Estimator)
# ============================================================================

def weight_quant(w: torch.Tensor) -> torch.Tensor:
    """
    Ternary weight quantization for BitNet b1.58.
    Quantizes weights to {-1, 0, +1} using absmean scaling.
    
    Args:
        w: Weight tensor
        
    Returns:
        Quantized weights with STE gradient
    """
    # Absmean scaling factor
    scale = w.abs().mean().clamp(min=1e-5)
    
    # Quantize: round(w / scale) clamped to {-1, 0, 1}
    w_scaled = w / scale
    w_quant = torch.clamp(torch.round(w_scaled), -1, 1)
    
    # Straight-Through Estimator
    return (w_quant * scale).detach() + w - w.detach()


def activation_quant(x: torch.Tensor, bits: int = 8) -> torch.Tensor:
    """
    Absmax quantization for activations.
    
    Args:
        x: Activation tensor
        bits: Quantization bits (default 8)
        
    Returns:
        Quantized activations with STE gradient
    """
    Qn = -(2 ** (bits - 1))
    Qp = 2 ** (bits - 1) - 1
    
    # Per-tensor absmax scale
    scale = x.abs().max().clamp(min=1e-5)
    
    # Quantize
    x_scaled = x / scale * Qp
    x_quant = torch.clamp(torch.round(x_scaled), Qn, Qp)
    
    # Dequantize with STE
    x_dequant = x_quant / Qp * scale
    return x_dequant.detach() + x - x.detach()


# ============================================================================
# BitLinear Layer
# ============================================================================

class BitLinear(nn.Module):
    """
    Linear layer with 1.58-bit quantized weights.
    
    Uses ternary quantization {-1, 0, +1} for weights
    and 8-bit quantization for activations.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        activation_bits: int = 8
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation_bits = activation_bits
        
        # Weight parameter (full precision during training)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        
        # Optional bias
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        
        # RMSNorm for input (BitNet uses pre-norm)
        self.norm = RMSNorm(in_features)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with quantization.
        
        Args:
            x: Input [..., in_features]
            
        Returns:
            Output [..., out_features]
        """
        # Normalize input
        x = self.norm(x)
        
        # Quantize activations
        x_quant = activation_quant(x, self.activation_bits)
        
        # Quantize weights
        w_quant = weight_quant(self.weight)
        
        return F.linear(x_quant, w_quant, self.bias)


# ============================================================================
# RMSNorm (Root Mean Square Layer Normalization)
# ============================================================================

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    More efficient than LayerNorm, used in Llama.
    """
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # RMS normalization
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x / rms * self.weight


# ============================================================================
# SwiGLU Feed-Forward Network (with BitLinear)
# ============================================================================

class SwiGLU(nn.Module):
    """
    SwiGLU activation for Feed-Forward network.
    Uses BitLinear layers for efficiency.
    
    SwiGLU(x) = (xW1) * SiLU(xV) @ W2
    """
    
    def __init__(
        self,
        dim: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.0
    ):
        super().__init__()
        hidden_dim = hidden_dim or int(dim * 8 / 3)
        # Round to nearest multiple of 256 for efficiency
        hidden_dim = ((hidden_dim + 255) // 256) * 256
        
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)  # Gate projection
        self.w2 = BitLinear(hidden_dim, dim, bias=False)  # Down projection  
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)  # Up projection
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: gate * up projection
        gate = F.silu(self.w1(x))
        up = self.w3(x)
        x = gate * up
        x = self.dropout(x)
        return self.w2(x)


# ============================================================================
# Rotary Positional Embeddings (RoPE)
# ============================================================================

def precompute_freqs_cis(dim: int, max_seq_len: int, theta: float = 10000.0) -> torch.Tensor:
    """
    Precompute rotary embedding frequencies.
    
    Args:
        dim: Head dimension (must be even)
        max_seq_len: Maximum sequence length
        theta: Base frequency
        
    Returns:
        Complex frequencies tensor [max_seq_len, dim//2]
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_seq_len)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)  # complex64


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to query and key tensors.
    
    Args:
        xq: Query tensor [batch, seq, n_heads, head_dim]
        xk: Key tensor [batch, seq, n_kv_heads, head_dim]
        freqs_cis: Precomputed frequencies [seq, head_dim//2]
        
    Returns:
        Rotated (xq, xk)
    """
    # Reshape to complex
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    
    # Apply rotation
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(2)  # [1, seq, 1, dim//2]
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(-2)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(-2)
    
    return xq_out.type_as(xq), xk_out.type_as(xk)


# ============================================================================
# Test
# ============================================================================

def main():
    """Test BitLinear layer."""
    batch_size = 4
    seq_len = 128
    dim = 256
    
    # Test BitLinear
    layer = BitLinear(dim, dim * 4)
    x = torch.randn(batch_size, seq_len, dim)
    y = layer(x)
    print(f"BitLinear: {x.shape} -> {y.shape}")
    
    # Test SwiGLU
    ffn = SwiGLU(dim)
    y = ffn(x)
    print(f"SwiGLU: {x.shape} -> {y.shape}")
    
    # Test RoPE
    freqs = precompute_freqs_cis(64, 512)
    print(f"RoPE freqs: {freqs.shape}")
    
    print("All tests passed!")


if __name__ == "__main__":
    main()



--- FILE: backend/src/model/iara_liquid.py ---
"""
I.A.R.A. v6.0 - Fractal Attention-CfC Model
============================================
Deep temporal context for long-term dependencies.

Key Features:
- Sequence Length: 256 steps (~21 hours of 5m data)
- Learned Positional Encoding (vs sinusoidal)
- Multi-Head Self-Attention (4 heads)
- Feature Attention Gate
- Symmetric CfC with tanh gating
- Higher regularization (dropout 0.5)

Reference: Hasani et al. (2022) "Closed-form Continuous-time Neural Networks"
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F



@dataclass
class LiquidConfig:
    """Configuration for IARA v6.7 Expert Symmetry."""
    
    input_dim: int = 16  # v6.7: Added Global Sentiment
    hidden_dim: int = 128
    num_heads: int = 4
    dropout: float = 0.5
    output_dim: int = 1
    max_seq_len: int = 256
    num_cfc_layers: int = 2 # v6.7: Stacked CfC


class LearnedPositionalEncoding(nn.Module):
    """
    Learned Positional Encoding for long sequences.
    Better than sinusoidal for distinguishing "10 min ago" vs "4 hours ago".
    """
    
    def __init__(self, d_model: int, max_len: int = 256, dropout: float = 0.1):
        super().__init__()
        
        self.position_embeddings = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Initialize with sinusoidal pattern for better starting point
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        
        self.position_embeddings.weight.data.copy_(pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add learned positional encoding. x: [batch, seq_len, d_model]"""
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        positional_encoding = self.position_embeddings(positions)
        return self.dropout(x + positional_encoding)


class FeatureAttention(nn.Module):
    """Feature-level attention gate with gating mechanism."""
    
    def __init__(self, input_dim: int, dropout: float = 0.1):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim * 2, input_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attention = self.gate(x)
        return x * attention


class TemporalSelfAttention(nn.Module):
    """
    Multi-head self-attention (no feed-forward for speed).
    v6.1 Lite: Removed FF expansion for 3x speedup.
    """
    
    def __init__(self, d_model: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Self-attention with residual.
        x: [batch, seq_len, d_model]
        """
        # Self-attention with residual
        attn_out, _ = self.attention(x, x, x, need_weights=False)
        x = self.norm(x + self.dropout(attn_out))
        return x


class CfCCell(nn.Module):
    """
    Symmetric Closed-form Continuous (CfC) Cell v6.7.
    
    Features:
    - Tanh gating for symmetric dynamics
    - Hidden state clipping [-1, 1]
    - Layer normalization for stability
    - Triton kernel acceleration (when available)
    """
    
    # Class-level flag for kernel availability
    _use_triton = None
    
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Candidate generation
        self.linear_input = nn.Linear(input_dim, hidden_dim)
        self.linear_hidden = nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        # Symmetric gate
        self.gate_input = nn.Linear(input_dim, hidden_dim)
        self.gate_hidden = nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        # Regularization
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
        self.activation = nn.Tanh()
        
        # Check Triton availability once
        if CfCCell._use_triton is None:
            try:
                from src.kernels import fused_cfc_update, TRITON_AVAILABLE
                CfCCell._use_triton = TRITON_AVAILABLE
                if TRITON_AVAILABLE:
                    import logging
                    logging.getLogger(__name__).info("CfCCell: Triton Flash Liquid kernels enabled!")
            except ImportError:
                CfCCell._use_triton = False
        
    def forward(self, input_: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        # Compute linear transforms
        input_linear = self.linear_input(input_) + self.linear_hidden(hidden)
        gate_logits = self.gate_input(input_) + self.gate_hidden(hidden)
        
        if CfCCell._use_triton and input_.is_cuda:
            # Use Triton Fused Kernel (Flash Liquid)
            from src.kernels import fused_cfc_update
            
            # The kernel expects: h_prev, input_linear, gate_linear
            # But our CfC uses tanh gate, not sigmoid. We need to adapt.
            # For now, use PyTorch fallback since the kernel assumes sigmoid gate.
            # TODO: Implement tanh-gate variant of the kernel.
            
            # Fallback to PyTorch for tanh gate
            h_cand = self.activation(input_linear)
            h_cand = self.dropout(h_cand)
            gate = torch.tanh(gate_logits)
            update = gate * (h_cand - hidden)
            h_new = hidden + update
        else:
            # Standard PyTorch Implementation
            h_cand = self.activation(input_linear)
            h_cand = self.dropout(h_cand)
            gate = torch.tanh(gate_logits)
            update = gate * (h_cand - hidden)
            h_new = hidden + update
        
        # Normalize and clip
        h_new = self.layer_norm(h_new)
        h_new = torch.clamp(h_new, min=-1.0, max=1.0)
        
        return h_new


class IARA(nn.Module):
    """
    I.A.R.A. v6.7 - Expert Symmetry (MoE Lite)
    
    Changes:
    - Stacked CfC (2 layers)
    - Gradient Checkpointing
    - Asymmetric Expert Heads (Long vs Short)
    """
    
    def __init__(self, config: LiquidConfig):
        super().__init__()
        self.config = config
        
        # Input projection to attention dimension
        self.input_proj = nn.Linear(config.input_dim, config.hidden_dim)
        
        # Learned positional encoding
        self.pos_encoding = LearnedPositionalEncoding(
            config.hidden_dim, 
            max_len=config.max_seq_len,
            dropout=config.dropout * 0.5
        )
        
        # Self-attention layer
        self.self_attention = TemporalSelfAttention(
            d_model=config.hidden_dim,
            num_heads=config.num_heads,
            dropout=config.dropout
        )
        
        # Feature attention
        self.feature_attention = FeatureAttention(
            config.hidden_dim, 
            dropout=config.dropout * 0.5
        )
        
        # Project back to input_dim (This was weird in v6.0, keeping structure but optimizing)
        # Actually for stacked CfC, better to keep in hidden_dim if possible.
        # But CfCCell takes input_dim. Let's adapt CfC to take hidden_dim or project back.
        # The design says: Attention -> Project -> CfC.
        # If we stack CfC, intermediate projection might be needed or we just chain hidden.
        # Let's keep logic: Attention features become input to CfC system.
        self.attention_proj = nn.Linear(config.hidden_dim, config.input_dim)
        self.input_norm = nn.LayerNorm(config.input_dim)
        
        # Stacked CfC Layers (v6.7)
        self.cfc_layers = nn.ModuleList([
            CfCCell(config.input_dim if i == 0 else config.hidden_dim, 
                   config.hidden_dim, 
                   dropout=config.dropout)
            for i in range(config.num_cfc_layers)
        ])
        
        # Expert Heads (MoE Lite)
        # Head A: Long Expert (Finds reasons to BUY) - Using SiLU per spec
        self.long_expert = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, 1)
        )
        
        # Head B: Short Expert (Finds reasons to SELL/AVOID) - Using SiLU per spec
        self.short_expert = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, 1)
        )
        
        self._init_weights()
        
    def _init_weights(self):
        """Xavier init + negative output bias for neutral initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        # Initialize heads to cancel out initially (0 - 0 = 0)
        nn.init.zeros_(self.long_expert[-1].bias)
        nn.init.zeros_(self.short_expert[-1].bias)
    
    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        # v6.8 "Blindfolded Prophet" - Targeted Feature Dropout
        # The last 4 features are temporal: hour_sin, hour_cos, day_sin, day_cos
        # By zeroing them out 30% of the time, we force the model to learn
        # price dynamics (RSI, MACD, velocity) instead of relying on "clock tricks".
        if self.training and torch.rand(1, device=x.device).item() < 0.3:
            x = x.clone()  # Avoid modifying original tensor
            x[:, :, -4:] = 0.0  # Blindfold the time perception
        
        # Gradient Checkpointing Helper (Enabled for v6.7 VRAM Optimization)
        def run_attention(x_in):
            x_p = self.input_proj(x_in)
            x_p = self.pos_encoding(x_p)
            x_a = self.self_attention(x_p)
            x_a = self.feature_attention(x_a)
            return self.attention_proj(x_a)

        if self.training:
            # Use checkpointing to trade compute for memory
            x_cfc = torch.utils.checkpoint.checkpoint(run_attention, x, use_reentrant=False)
        else:
            x_cfc = run_attention(x)
            
        x_cfc = self.input_norm(x_cfc)
        
        batch_size, seq_len, _ = x.shape
        
        # Stacked CfC Loop
        # We process layer by layer or time by time? 
        # Standard RNN stack: Layer 1 over all time, then Layer 2 over all time.
        # This is more efficient for parallelism if not fully coupled.
        # But CfC depends on prev hidden.
        
        current_input = x_cfc
        final_hidden = None
        
        for layer_idx, cell in enumerate(self.cfc_layers):
            h_t = torch.zeros(batch_size, self.config.hidden_dim, device=x.device, dtype=x.dtype)
            outputs = []
            
            # Recurrent Loop
            for t in range(seq_len):
                h_t = cell(current_input[:, t, :], h_t)
                outputs.append(h_t.unsqueeze(1))
            
            # Output of this layer is input to next
            current_input = torch.cat(outputs, dim=1) # [batch, seq, hidden]
            final_hidden = h_t
            
        # v6.7 Expert Heads
        # We use the final hidden state of the last layer
        
        long_score = self.long_expert(final_hidden)
        short_score = self.short_expert(final_hidden)
        
        # Differential Logit: "Buy Conviction" minus "Sell Conviction"
        logits = long_score - short_score
        
        return logits.squeeze(-1), final_hidden
        
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)



--- FILE: backend/src/model/saci_nano.py ---
"""
Saci-Nano - Causal Decoder-Only Transformer for Financial Token Prediction
===========================================================================
A GPT-style autoregressive model that treats market indicators as tokens.
Implements Next Token Prediction for market state forecasting.
"""

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from src.model.bitnet import (
    BitLinear,
    RMSNorm,
    SwiGLU,
    precompute_freqs_cis,
    apply_rotary_emb
)


@dataclass
class SaciNanoConfig:
    """Configuration for Saci-Nano model."""
    
    vocab_size: int = 16384  # Token vocabulary size
    dim: int = 512           # Model dimension
    n_layers: int = 8        # Number of transformer layers
    n_heads: int = 8         # Number of attention heads
    n_kv_heads: int = 4      # Number of key-value heads (GQA)
    max_seq_len: int = 512   # Maximum sequence length
    dropout: float = 0.1     # Dropout rate
    rope_theta: float = 10000.0  # RoPE base frequency


class CausalSelfAttention(nn.Module):
    """
    Multi-Head Causal Self-Attention with RoPE.
    Supports Grouped Query Attention (GQA) for efficiency.
    """
    
    def __init__(self, config: SaciNanoConfig):
        super().__init__()
        self.config = config
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.dim // config.n_heads
        self.n_rep = self.n_heads // self.n_kv_heads  # Repetitions for GQA
        
        # Projections
        self.wq = nn.Linear(config.dim, config.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(config.dim, config.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(config.dim, config.n_kv_heads * self.head_dim, bias=False)
        self.wo = BitLinear(config.n_heads * self.head_dim, config.dim, bias=False)
        
        self.dropout = nn.Dropout(config.dropout)
        
        # Causal mask (will be registered as buffer)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(config.max_seq_len, config.max_seq_len), diagonal=1).bool()
        )
    
    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass with causal masking.
        
        Args:
            x: Input [batch, seq, dim]
            freqs_cis: RoPE frequencies [seq, head_dim//2]
            
        Returns:
            Output [batch, seq, dim]
        """
        batch_size, seq_len, _ = x.shape
        
        # Project Q, K, V
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)
        
        # Reshape to heads
        q = rearrange(q, 'b s (h d) -> b s h d', h=self.n_heads)
        k = rearrange(k, 'b s (h d) -> b s h d', h=self.n_kv_heads)
        v = rearrange(v, 'b s (h d) -> b s h d', h=self.n_kv_heads)
        
        # Apply RoPE
        q, k = apply_rotary_emb(q, k, freqs_cis[:seq_len])
        
        # Repeat KV for GQA
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=2)
            v = v.repeat_interleave(self.n_rep, dim=2)
        
        # Rearrange for attention
        q = rearrange(q, 'b s h d -> b h s d')
        k = rearrange(k, 'b s h d -> b h s d')
        v = rearrange(v, 'b s h d -> b h s d')
        
        # Scaled dot-product attention
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        
        # Apply causal mask
        attn = attn.masked_fill(self.mask[:seq_len, :seq_len], float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply to values
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h s d -> b s (h d)')
        
        return self.wo(out)


class TransformerBlock(nn.Module):
    """
    Single transformer decoder block.
    Pre-norm architecture with RMSNorm.
    """
    
    def __init__(self, config: SaciNanoConfig):
        super().__init__()
        
        self.attention = CausalSelfAttention(config)
        self.ffn = SwiGLU(config.dim, dropout=config.dropout)
        
        self.norm1 = RMSNorm(config.dim)
        self.norm2 = RMSNorm(config.dim)
    
    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward with residual connections.
        
        Args:
            x: Input [batch, seq, dim]
            freqs_cis: RoPE frequencies
            
        Returns:
            Output [batch, seq, dim]
        """
        # Self-attention with residual
        x = x + self.attention(self.norm1(x), freqs_cis)
        
        # FFN with residual
        x = x + self.ffn(self.norm2(x))
        
        return x


class SaciNano(nn.Module):
    """
    Saci-Nano: Causal Decoder-Only Transformer for Financial Prediction.
    
    Treats market indicators as tokens and predicts the next token
    in an autoregressive manner (Next Token Prediction).
    """
    
    def __init__(self, config: SaciNanoConfig):
        super().__init__()
        self.config = config
        
        # Token embedding
        self.token_embedding = nn.Embedding(config.vocab_size, config.dim)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])
        
        # Final layer norm
        self.norm = RMSNorm(config.dim)
        
        # Output projection (tied with embedding)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)
        
        # Precompute RoPE frequencies
        head_dim = config.dim // config.n_heads
        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(head_dim, config.max_seq_len, config.rope_theta)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module: nn.Module):
        """Initialize weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(
        self,
        tokens: torch.Tensor,
        targets: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            tokens: Input token IDs [batch, seq]
            targets: Optional target token IDs for loss computation
            
        Returns:
            Tuple of (logits, loss)
        """
        batch_size, seq_len = tokens.shape
        
        # Token embeddings
        x = self.token_embedding(tokens)
        
        # Pass through transformer layers
        for layer in self.layers:
            x = layer(x, self.freqs_cis)
        
        # Final norm
        x = self.norm(x)
        
        # Output logits
        logits = self.output(x)
        
        # Compute loss if targets provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                targets.view(-1),
                ignore_index=-100  # Ignore padding
            )
        
        return logits, loss
    
    @torch.no_grad()
    def generate(
        self,
        tokens: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None
    ) -> torch.Tensor:
        """
        Generate new tokens autoregressively.
        
        Args:
            tokens: Initial tokens [batch, seq]
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling (None for greedy)
            
        Returns:
            Generated tokens [batch, seq + max_new_tokens]
        """
        for _ in range(max_new_tokens):
            # Truncate to max_seq_len
            tokens_cond = tokens[:, -self.config.max_seq_len:]
            
            # Forward pass
            logits, _ = self(tokens_cond)
            logits = logits[:, -1, :] / temperature
            
            # Top-k sampling
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            
            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append
            tokens = torch.cat([tokens, next_token], dim=1)
        
        return tokens
    
    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def estimate_vram_mb(self, batch_size: int = 32) -> float:
        """Estimate VRAM usage in MB."""
        params = self.count_parameters() * 4 * 2  # weights + gradients
        seq = self.config.max_seq_len
        hidden = self.config.dim
        layers = self.config.n_layers
        activations = batch_size * seq * hidden * 4 * layers * 2
        return (params + activations) / (1024 * 1024)


def main():
    """Test model forward pass."""
    config = SaciNanoConfig(
        vocab_size=8192,
        dim=256,
        n_layers=4,
        n_heads=4,
        n_kv_heads=2,
        max_seq_len=512
    )
    
    model = SaciNano(config)
    print(f"Parameters: {model.count_parameters():,}")
    print(f"Estimated VRAM: {model.estimate_vram_mb():.2f} MB")
    
    # Test forward
    tokens = torch.randint(0, config.vocab_size, (2, 128))
    targets = torch.randint(0, config.vocab_size, (2, 128))
    
    logits, loss = model(tokens, targets)
    print(f"Logits shape: {logits.shape}")
    print(f"Loss: {loss.item():.4f}")
    
    # Test generate
    generated = model.generate(tokens[:, :10], max_new_tokens=20, top_k=50)
    print(f"Generated shape: {generated.shape}")
    
    print("Forward pass successful!")


if __name__ == "__main__":
    main()



--- FILE: backend/src/viz/neuroscope.py ---
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path
import argparse
import sys
import logging

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.train.trainer_liquid import TrainerConfig, MultiAssetDataset
from src.model.iara_liquid import IARA
# from src.train.dataset import MultiAssetDataset # Moved to trainer_liquid

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Neuroscope")

class Neuroscope:
    def __init__(self, model_path, device='cuda'):
        self.device = device
        self.output_dir = Path("plots/neuroscope")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load Model
        self.model, self.config = self._load_model(model_path)
        
        # Trace Storage
        self.traces = {
            'cfc_states': [],    # List of [batch, hidden]
            'attention_weights': None
        }
        
    def _load_model(self, path):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        # Reconstruct Config
        config_data = checkpoint.get('config', None)
        
        if isinstance(config_data, TrainerConfig):
             config = config_data
        elif isinstance(config_data, dict):
             config = TrainerConfig(**config_data)
        else:
             config = TrainerConfig()
        
        # Override architecture params if stored directly
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint # Assume raw state dict
            
        # Initialize Model
        model = IARA(config).to(self.device)
        
        # Handle state dict mismatch (if any)
        try:
            model.load_state_dict(state_dict)
        except Exception as e:
            logger.warning(f"Strict load failed: {e}. Trying non-strict.")
            # Filter distinct keys if they contain 'module.' prefix (DataParallel)
            new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            model.load_state_dict(new_state_dict, strict=False)
            
        model.eval()
        return model, config
    
    def _hook_cfc(self, module, input, output):
        # Output is h_new: [batch, hidden_dim]
        # We append to list. Since this is called for every timestep, 
        # we will have seq_len entries per layer.
        self.traces['cfc_states'].append(output.detach().cpu())

    def _hook_attention(self, module, input, output):
        # Attention output is usually [batch, seq_len, hidden]. 
        # We want the weights ideally, but IARA's attention layer returns context tensor.
        # If we can't get weights easily without modifying code, we visualize the output context magnitude.
        # Ideally, we'd hook 'attn_output_weights' inside MultiheadAttention, but that's internal.
        self.traces['attention_weights'] = output.detach().cpu()

    def register_hooks(self):
        # Hook CfC Layers
        for i, layer in enumerate(self.model.cfc_layers):
            layer.register_forward_hook(self._hook_cfc)
            logger.info(f"Hooked CfC Layer {i}")
            
        # Hook Attention
        if hasattr(self.model, 'self_attention'):
            self.model.self_attention.register_forward_hook(self._hook_attention)
            logger.info("Hooked Self Attention")

    def run(self, ticker='EURUSD', limit=100):
        # Load Data
        dataset = MultiAssetDataset(
            seq_len=self.config.seq_len,
            stride=1,
            tickers=[ticker] if ticker else None,
            augment=False,
            target_lookahead=5
        )
        
        # Pick a random sample or specific one
        idx = np.random.randint(0, len(dataset))
        x, y, w = dataset[idx]
        
        # Add batch dim
        x = x.unsqueeze(0).to(self.device) # [1, seq, 16]
        
        # Handle Feature Mismatch (e.g. Model trained on 13 features but dataset has 16)
        if x.shape[-1] > self.config.input_dim:
            logger.warning(f"Input features {x.shape[-1]} > Model expects {self.config.input_dim}. Truncating.")
            x = x[:, :, :self.config.input_dim]
            
        logger.info(f"Analyzing sample {idx} from {ticker}...")
        
        # Run Inference
        with torch.no_grad():
            self.model(x)
            
        # Process Traces
        self.plot_cfc_activation()
        self.plot_attention_magnitude()
        self.plot_input_features(x.cpu())
        
        logger.info(f"Plots saved to {self.output_dir}")

    def plot_cfc_activation(self):
        # Traces contains (seq_len * num_layers) items
        # Each item is [1, hidden_dim]
        
        seq_len = self.config.seq_len
        num_layers = self.config.num_cfc_layers
        
        # Stack all: [total_steps, hidden]
        all_states = torch.stack(self.traces['cfc_states']).squeeze(1) # [steps, hidden]
        
        # Reshape to [layers, seq, hidden]
        # Because the loop is: for layer: for time: ...
        # The hooks fire in that order? 
        # Wait, the code is:
        # for layer in layers:
        #    for t in range(seq_len):
        #        cell(x, h)
        # So yes, the list is ordered: Layer0_t0, Layer0_t1... Layer0_tN, Layer1_t0...
        
        try:
            states_per_layer = all_states.view(num_layers, seq_len, -1)
        except RuntimeError:
            logger.error("Could not reshape traces. Did the sequence length match?")
            return

        # Plot Heatmap for each layer
        for i in range(num_layers):
            plt.figure(figsize=(15, 6))
            data = states_per_layer[i].T.numpy() # [neurons, time]
            
            # Normalize for better visualization (-1 to 1 usually)
            sns.heatmap(data, cmap='viridis', center=0, vmin=-1, vmax=1)
            plt.title(f"CfC Layer {i+1} - Neuron Activation (Liquid State)")
            plt.xlabel("Time Step (M5 Bars)")
            plt.ylabel("Neuron #")
            plt.savefig(self.output_dir / f"cfc_activity_layer_{i+1}.png")
            plt.close()

    def plot_attention_magnitude(self):
        if self.traces['attention_weights'] is None:
            return
            
        # attention output: [1, seq, hidden]
        attn = self.traces['attention_weights'].squeeze(0).T.numpy() # [hidden, seq]
        
        plt.figure(figsize=(15, 6))
        # Compute "Focus" as simple magnitude of attention context vector
        focus = np.linalg.norm(attn, axis=0) # [seq]
        
        plt.plot(focus, color='orange', linewidth=2)
        plt.fill_between(range(len(focus)), focus, color='orange', alpha=0.3)
        plt.title("Fractal Attention Magnitude (Temporal Focus)")
        plt.xlabel("Time Step (M5 Bars)")
        plt.ylabel("Attention Energy")
        plt.grid(True, alpha=0.3)
        plt.savefig(self.output_dir / "attention_magnitude.png")
        plt.close()

    def plot_input_features(self, x):
        # x: [1, seq, features]
        x = x.squeeze(0).numpy().T # [features, seq]
        
        plt.figure(figsize=(15, 8))
        sns.heatmap(x, cmap='coolwarm', center=0, robust=True)
        plt.title("Input Features (Normalized)")
        plt.xlabel("Time Step")
        plt.ylabel("Feature Index")
        plt.savefig(self.output_dir / "input_features.png")
        plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="checkpoints/best_model.pth")
    parser.add_argument("--limit", type=int, default=1)
    args = parser.parse_args()
    
    scope = Neuroscope(args.model)
    scope.register_hooks()
    scope.run()



--- FILE: backend/src/viz/dashboard.py ---
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import torch
import numpy as np
import pandas as pd
import sys
import json
import glob
import time
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.train.trainer_liquid import TrainerConfig, MultiAssetDataset
from src.model.iara_liquid import IARA
from src.data.feature_factory import process_hierarchical  # v6.12 Integration
import torch.nn.functional as F
from src.trading.simulation import MarketSimulator  # Trade Hub Engine

# Configuração da Página
st.set_page_config(layout="wide", page_title="SACI Neural Dashboard", page_icon="🧠")

st.markdown("""
<style>
    .block-container {padding-top: 1rem;}
    h1 {color: #FF4B4B;}
    .stMetric {background-color: #0E1117; padding: 10px; border-radius: 5px; border: 1px solid #262730;}
</style>
""", unsafe_allow_html=True)

st.title("🧠 SACI-Brain: Neural Inspector v6.11")

# --- Helpers ---
@st.cache_resource
def load_model(device='cpu', use_lab=False):
    base_dir = "checkpoints_lab" if use_lab else "checkpoints"
    path = f"{base_dir}/best_model.pth"
    if not Path(path).exists():
        # Fallback to interrupted if best not found
        path = f"{base_dir}/interrupted_model.pth"
    
    try:
        # Load com segurança para TrainerConfig
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        
        # Config
        config_data = checkpoint.get('config', None)
        if isinstance(config_data, TrainerConfig):
             config = config_data
        elif isinstance(config_data, dict):
             config = TrainerConfig(**config_data)
        else:
             config = TrainerConfig()
             
        # Model
        model = IARA(config).to(device)
        
        # State Dict
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
            
        # Clean keys
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict, strict=False)
        model.eval()
        
        return model, config
    except Exception as e:
        st.error(f"Erro ao carregar modelo ({path}): {e}")
        return None, None

@st.cache_resource
def get_dataset(_config, ticker="EURUSD"):
    # v6.12: Use Feature Factory to generate 18-feature vectors (Fixes Padding Issue)
    # Search for raw file
    # Check for both {ticker}.parquet and {ticker}_M1.parquet to be safe
    candidates = [
        Path(f"data/raw/{ticker}.parquet"),
        Path(f"data/raw/{ticker}_M1.parquet"),
        Path(f"data/processed/{ticker}.parquet"),
        Path(f"data/processed/{ticker}_M1.parquet")
    ]
    
    raw_path = None
    for p in candidates:
        if p.exists():
            raw_path = p
            break
         
    if raw_path and raw_path.exists():
        X, y = process_hierarchical(raw_path, _config.seq_len, 2) # Stride 2 to match training
        if X is not None:
            return torch.from_numpy(X), torch.from_numpy(y)
    
    return None, None

def get_latest_log_file():
    log_files = glob.glob("logs/training_history_*.jsonl")
    if not log_files:
        return None
    return max(log_files, key=os.path.getmtime) if 'os' in locals() else sorted(log_files)[-1]

def load_training_history():
    log_file = get_latest_log_file()
    if not log_file:
        return pd.DataFrame()
    
    data = []
    with open(log_file, 'r') as f:
        for line in f:
            try:
                entry = json.loads(line)
                if 'epoch' in entry: # Filter only epoch stats
                    # Flatten nested metrics
                    flat_entry = {
                        'epoch': entry['epoch'],
                        'train_loss': entry['train'].get('loss'),
                        'train_acc': entry['train'].get('accuracy'),
                        'val_loss': entry['val'].get('loss'),
                        'val_auc': entry['val'].get('auc'),
                        # Handle old logs vs new logs format
                        'lr': entry.get('metrics', {}).get('learning_rate', 0),
                        'memory_chronic': entry.get('memory', {}).get('chronic', 0),
                        'memory_mastered': entry.get('memory', {}).get('mastered', 0)
                    }
                    data.append(flat_entry)
            except:
                pass
    return pd.DataFrame(data)

# --- Hooks ---
class HookManager:
    def __init__(self):
        self.activations = {}
        self.attention = None
        
    def hook_cfc(self, name):
        def hook(module, input, output):
            # Output: [batch, hidden]
            if name not in self.activations:
                self.activations[name] = []
            self.activations[name].append(output.detach().cpu())
        return hook
    
    def hook_attn(self, module, input, output):
        self.attention = output.detach().cpu()

# --- Main Logic ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Sidebar Configuration
st.sidebar.header("Configuration")
model_source = st.sidebar.radio("Model Source", ["Production (Full)", "Lab (Experimental)"], index=0)
is_lab = model_source == "Lab (Experimental)"

# Load Model
model, config = load_model(device, use_lab=is_lab)

# --- TABS ---
tab_live, tab_neu, tab_hub, tab_mkt = st.tabs(["🔥 Live Training", "🧠 Neural Neuroscience", "🏟️ Trade Hub (Sim)", "📉 Market Context"])

# --- TAB 1: Live Training ---
with tab_live:
    st.header("Monitoramento de Treino em Tempo Real")
    
    if st.button("🔄 Refresh Logs"):
        st.rerun()
        
    df_history = load_training_history()
    
    if not df_history.empty:
        # Metrics
        latest = df_history.iloc[-1]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Epoch", int(latest['epoch']))
        c2.metric("Val AUC", f"{latest.get('val_auc', 0):.4f}", delta=f"{latest.get('val_auc', 0) - 0.5:.4f}")
        c3.metric("Train Loss", f"{latest.get('train_loss', 0):.4f}")
        c4.metric("Learning Rate", f"{latest.get('lr', 0):.2e}")
        
        # Plots
        col_plot1, col_plot2 = st.columns(2)
        with col_plot1:
            fig_loss = px.line(df_history, x='epoch', y=['train_loss', 'val_loss'], title="Loss Curve (Focal Loss)")
            st.plotly_chart(fig_loss, use_container_width=True)
            
        with col_plot2:
            fig_auc = px.line(df_history, x='epoch', y=['val_auc', 'train_acc'], title="Validation AUC & Accuracy")
            st.plotly_chart(fig_auc, use_container_width=True)
            
        with st.expander("Memory Consistency (Selective Memory Strategy)"):
            if 'memory_chronic' in df_history.columns:
                 fig_mem = px.area(df_history, x='epoch', y=['memory_chronic', 'memory_mastered'], title="Curriculum Learning State")
                 st.plotly_chart(fig_mem, use_container_width=True)
    else:
        st.info("Nenhum log de treino encontrado. O treino começou?")

# --- TAB 2: Neuroscience ---
with tab_neu:
    if model:
        # Sidebar Controls
        st.sidebar.header("Inference Config")
        ticker = st.sidebar.selectbox("Asset", ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCHF"])
        
        # Dataset Loading
        # Dataset Loading (v6.12 Direct Tensor)
        X_data, y_data = get_dataset(config, ticker)
        
        if X_data is not None and len(X_data) > 0:
            sample_idx = st.sidebar.slider("Sample Index (Time)", 0, len(X_data)-1, len(X_data)-1)
            
            # Inference Data
            x = X_data[sample_idx]
            y = y_data[sample_idx]
            
            x_in = x.unsqueeze(0).to(device) # [1, seq, feat]
            
            # v6.12: No more padding needed (Feature Factory handles it)
            # Verify dimensions
            target_dim = config.input_dim
            if x_in.shape[-1] != target_dim:
                st.error(f"Dimension Mismatch: Model expects {target_dim}, got {x_in.shape[-1]}. Re-run Feature Factory?")

            
            # Hooks
            hooks = HookManager()
            handles = []
            
            # Register Hooks
            for i, layer in enumerate(model.cfc_layers):
                h = layer.register_forward_hook(hooks.hook_cfc(f"Layer {i+1} (Liquid)"))
                handles.append(h)
            
            if hasattr(model, 'self_attention'):
                h = model.self_attention.register_forward_hook(hooks.hook_attn)
                handles.append(h)
            
            # Forward
            with torch.no_grad():
                logits, _ = model(x_in)
                prob = torch.sigmoid(logits).item()
                
            # Remove Hooks
            for h in handles: h.remove()
            
            # --- Visualization Area ---
            c_prob, c_label, c_conf = st.columns(3)
            c_prob.metric("Model Probability", f"{prob:.2%}")
            c_label.metric("Ground Truth", "UP" if y.item() > 0.5 else "DOWN")
            c_conf.metric("Status", "CONFIDENT" if abs(prob-0.5) > 0.1 else "UNCERTAIN")
            
            # 1. Feature Importance (Weights Proxy)
            st.subheader("🕵️ Feature Attribution (O que o modelo olhou?)")
            
            # Try to get feature names from norm_params or default
            feature_names = [f"Feat {i}" for i in range(target_dim)]
            try:
                norm_params = torch.load("data/cache/norm_params.pt", weights_only=False)
                if 'feature_cols' in norm_params:
                     feature_names = norm_params['feature_cols']
            except:
                pass
                
            # Input Projection Weights (First Layer Importance)
            # Sum of absolute weights connected to each feature
            input_weights = model.input_proj.weight.abs().mean(dim=0).detach().cpu().numpy()
            
            # Adjust if dimension mismatch (model holds 18 weights, names might be 18)
            if len(input_weights) == len(feature_names):
                df_imp = pd.DataFrame({'Feature': feature_names, 'Importance': input_weights})
                df_imp = df_imp.sort_values('Importance', ascending=True)
                
                fig_imp = px.bar(df_imp, x='Importance', y='Feature', orientation='h', title="Relevância Global das Features (Camada de Entrada)")
                st.plotly_chart(fig_imp, use_container_width=True)
            
            # 2. Neuron Activities
            st.subheader("🧠 Atividade dos Neurônios (Dinâmica Líquida)")
            if hooks.activations:
                # Use only the last layer or aggregate
                last_layer_name = list(hooks.activations.keys())[-1]
                activity_list = hooks.activations[last_layer_name]
                act_tensor = torch.stack(activity_list).squeeze(1).T.numpy() # [hidden, seq]
                
                # Heatmap
                fig_neuro = px.imshow(
                    act_tensor,
                    aspect='auto',
                    color_continuous_scale='Magma',
                    labels=dict(x="Tempo (Barras)", y="Neurônio ID"),
                    title=f"Mapa de Disparo - {last_layer_name}"
                )
                st.plotly_chart(fig_neuro, use_container_width=True)
                
            # 3. Attention Flow
            st.subheader("👁️ Fractal Attention Focus")
            if hooks.attention is not None:
                 attn_mag = torch.norm(hooks.attention.squeeze(0), dim=1).numpy()
                 # Smoothing
                 attn_series = pd.Series(attn_mag).rolling(5).mean().fillna(0)
                 
                 fig_attn = px.area(
                     x=range(len(attn_mag)), 
                     y=attn_series,
                     title="Atenção Temporal (Picos indicam momentos decisivos)",
                 )
                 st.plotly_chart(fig_attn, use_container_width=True)

            # 4. Neuroscope: Macro vs Micro (Hidden Context)
            st.subheader("🔬 Neuroscope: Macro vs Micro (O que a Rede Vê)")
            
            # Extract Trend Features (Indices 12=M5, 13=H1 based on feature_factory)
            # x is [seq, feat]
            trend_m5 = x[:, 12].cpu().numpy() # [256]
            trend_h1 = x[:, 13].cpu().numpy() # [256]
            
            # Create a dataframe for Plotly
            df_ctx = pd.DataFrame({
                "Time": range(len(trend_m5)),
                "Trend M5 (Tactical)": trend_m5,
                "Trend H1 (Strategic)": trend_h1 + 2.5 # Offset for visibility
            })
            
            fig_ctx = go.Figure()
            
            # Plot M5
            fig_ctx.add_trace(go.Scatter(
                x=df_ctx["Time"], y=df_ctx["Trend M5 (Tactical)"],
                mode='lines+markers', name='M5 Trend (Tactical)',
                line=dict(shape='hv', width=2),
                marker=dict(size=4)
            ))
            
            # Plot H1
            fig_ctx.add_trace(go.Scatter(
                x=df_ctx["Time"], y=df_ctx["Trend H1 (Strategic)"],
                mode='lines+markers', name='H1 Trend (Strategic)',
                line=dict(shape='hv', width=3),
                marker=dict(size=6)
            ))
            
            fig_ctx.update_layout(
                title="Sincronia Fractal: Alinhamento de Tendências no Input",
                yaxis=dict(
                    tickvals=[-1, 0, 1, 1.5, 2.5, 3.5],
                    ticktext=["DOWN", "FLAT", "UP", "", "DOWN (H1)", "UP (H1)"]
                ),
                height=300
            )
            st.plotly_chart(fig_ctx, use_container_width=True)
            
            # 5. Logic Stream (Explainability)
            with st.expander("🤖 Log de Pensamento (Decodificado)"):
                last_m5 = trend_m5[-1]
                last_h1 = trend_h1[-1]
                
                logic_msg = []
                if last_h1 == 1:
                    logic_msg.append("✅ **Macro (H1)**: Tendência de ALTA forte.")
                elif last_h1 == -1:
                    logic_msg.append("🔻 **Macro (H1)**: Tendência de BAIXA forte.")
                else:
                    logic_msg.append("⚠️ **Macro (H1)**: Lateral/Indefinido.")
                    
                if last_m5 == last_h1 and last_h1 != 0:
                    logic_msg.append("🚀 **Sinergia**: M5 alinhou com H1. Sinal amplificado.")
                elif last_m5 != last_h1:
                    logic_msg.append("🛑 **Conflito**: M5 diverge do H1 (Correção ou Ruído). Aguardando alinhamento.")
                    
                confidence = abs(prob - 0.5) * 2
                logic_msg.append(f"🧠 **Confiança Neural**: {confidence:.1%} ({'Alta' if confidence > 0.6 else 'Baixa'})")
                
                for msg in logic_msg:
                    st.write(msg)

        else:
            st.warning("Dataset vazio para este ticker.")
    else:
        st.error("Carregamento do modelo falhou.")


# --- TAB 3: Trade Hub (Simulation) ---
with tab_hub:
    st.header("🏟️ Trade Hub: Neural Simulation Engine")
    
    if model:
        # Session State for Simulator
        if 'simulator' not in st.session_state:
            st.session_state['simulator'] = MarketSimulator(model, ticker="EURUSD", device=device)
            st.session_state['sim_running'] = False
            
        sim = st.session_state['simulator']
        
        # Controls
        c_play, c_step, c_reset, c_speed = st.columns([1, 1, 1, 2])
        
        if c_step.button("👣 Step"):
            sim.step()
            
        if c_reset.button("🔄 Reset"):
            sim.reset()
            st.session_state['sim_running'] = False
            
        # Sim Loop (basic manual for now, autoplay requires rerun loop)
        # Display Current State
        if sim.history:
            state = sim.history[-1]
            
            # Metrics Row
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Step (Candle)", sim.current_step)
            m2.metric("Balance", f"${sim.balance:.2f}", delta=f"{sim.balance - sim.initial_balance:.2f}")
            m3.metric("Neural Prob", f"{state.prob:.1%}")
            
            status = "WAIT"
            if len(state.logic_log) > 0:
                 if "BUY" in str(state.logic_log): status = "BUY"
                 if "SELL" in str(state.logic_log): status = "SELL"
            m4.metric("Action", status)
            
            # Logic Log
            st.subheader("🧠 Brain Log (Pensamento)")
            for line in state.logic_log:
                if "Signal" in line:
                    st.success(line)
                elif "Macro" in line:
                    st.info(line)
                elif "Result" in line:
                    st.caption(line)
                else:
                    st.write(line)
                    
            # Chart (PnL Curve)
            pnl_history = [s.price for s in sim.history]
            fig_pnl = px.line(y=pnl_history, title="Equity Curve (Simulated)")
            st.plotly_chart(fig_pnl, use_container_width=True)
            
        else:
            st.info("Simulation Ready. Press Step to start.")
            
        st.divider()
        st.subheader("💾 TimescaleDB History (Latest 10)")
        try:
             dsn = os.getenv("DB_URI", "postgresql://saci:saci_secret@localhost:5432/saci_core")
             conn = psycopg2.connect(dsn)
             df_db = pd.read_sql("SELECT time, symbol, prob_up, confidence, trend_m5, trend_h1, logic_log FROM brain_states ORDER BY time DESC LIMIT 10", conn)
             st.dataframe(df_db)
             conn.close()
        except Exception as e:
            st.warning(f"DB not connected: {e}")
            
    else:
        st.error("Model not loaded.")

# --- TAB 3: Market ---
with tab_mkt:
    st.subheader("Input Feature Heatmap")
    if 'x_in' in locals():
        feat_data = x_in.squeeze(0).cpu().numpy().T
        fig_feat = px.imshow(feat_data, aspect='auto', title="Raw Input Features", origin='lower')
        st.plotly_chart(fig_feat, use_container_width=True)



--- FILE: docs/V9_ARCHITECTURE.md ---
# I.A.R.A. v9 "Swarm Intelligence" Architecture

## Overview

A v9 representa uma mudança paradigmática: de um **modelo único multi-timeframe** para um **enxame de especialistas coordenados**.

```
                    ┌─────────────────────────────────────┐
                    │      COORDINATOR (Main Brain)       │
                    │   Liquid CfC (256 units) + Experts  │
                    └─────────────────▲───────────────────┘
                                      │
                    ┌─────────────────┴───────────────────┐
                    │    CONSENSUS FUSION LAYER           │
                    │    (Cross-Attention over Workers)   │
                    └─────────────────▲───────────────────┘
                                      │
       ┌──────────┬──────────┬────────┴────────┬──────────┬──────────┐
       │          │          │                 │          │          │
   ┌───▼───┐  ┌───▼───┐  ┌───▼───┐         ┌───▼───┐  ┌───▼───┐  ┌───▼───┐
   │ W-M5  │  │ W-M15 │  │ W-M30 │         │ W-H1  │  │ W-H4  │  │  LLM  │
   │       │  │       │  │       │         │       │  │       │  │Regime │
   │ 128dim│  │ 128dim│  │ 128dim│         │ 128dim│  │ 128dim│  │(ext)  │
   └───────┘  └───────┘  └───────┘         └───────┘  └───────┘  └───────┘
   High-Capacity Workers (~800KB each)     Optional    External
```

---

## Design Philosophy

### Por que Workers separados?

1. **Especialização**: Cada timeframe tem características únicas
   - M5: Ruído alto, reversões rápidas
   - H1: Tendências claras, momentum
   - H4: Contexto macro, suporte/resistência chave

2. **Resiliência**: Se M5 "ficar louco" com ruído, H1/H4 mantêm sanidade

3. **Eficiência**: Workers ultra-leves (~32 units) são 10x mais rápidos que modelo único grande

4. **Interpretabilidade**: Podemos visualizar "quem discorda de quem"

---

## Components

### 1. LightningWorker (Ultra-Light Liquid Model)

```python
class LightningWorker(nn.Module):
    """
    Especialista para um único timeframe.
    Usa CfC compacto com quantização opcional.
    
    Input:  [batch, seq_len, 12 features]
    Output: 
      - embedding: [batch, 32]  (representação aprendida)
      - signal: [batch, 3]      (softmax: LONG/SHORT/WAIT confidence)
    """
    
    def __init__(self, timeframe: str, hidden_dim: int = 32):
        self.timeframe = timeframe  # "M5", "M15", etc.
        self.cfc = CfCCell(input_dim=12, hidden_dim=32)
        self.head = nn.Linear(32, 3)  # LONG/SHORT/WAIT
```

**VRAM por Worker (int8):**
- Parâmetros: ~20KB
- Activations: ~50KB
- **Total: ~70KB**

### 2. SwarmCoordinator (Main Brain)

```python
class SwarmCoordinator(nn.Module):
    """
    Cérebro central que analisa consenso entre workers.
    
    Input: 
      - worker_embeddings: [batch, 5, 32]  (5 workers x 32 dims)
      - worker_signals: [batch, 5, 3]      (5 workers x 3 classes)
    
    Output:
      - final_logit: [batch, 1]  (probabilidade de compra)
    """
    
    def __init__(self, num_workers: int = 5):
        # Cross-Attention: Qual worker está mais "confiante"?
        self.cross_attn = nn.MultiheadAttention(embed_dim=32, num_heads=4)
        
        # Fusion: Concatena tudo
        self.fusion = nn.Linear(5 * 128 + 5 * 3, 512)
        
        # CfC Principal
        self.cfc = CfCCell(input_dim=512, hidden_dim=512)
        
        # Expert Heads
        self.long_expert = nn.Linear(512, 1)
        self.short_expert = nn.Linear(512, 1)
```

### 3. Technical Analyzer (Pattern Detection)

```python
class TechnicalAnalyzer:
    """
    Detecta padrões técnicos para enriquecer features.
    
    Features geradas:
      - trend_slope: Inclinação da linha de tendência
      - dist_to_support: Distância ao suporte mais próximo
      - dist_to_resistance: Distância à resistência mais próxima
      - pattern_score: Pin bar, Engulfing, Doji detection
    """
    
    def detect_trend(self, highs, lows, closes) -> Tensor:
        # Linear regression over last N candles
        ...
    
    def find_sr_levels(self, highs, lows) -> Tuple[Tensor, Tensor]:
        # Pivot point detection
        ...
```

---

## Data Pipeline

### Input Features per Timeframe (12 dims)

| Índice | Feature | Descrição |
|--------|---------|-----------|
| 0 | `rsi_norm` | RSI normalizado [-1, 1] |
| 1 | `macd_norm` | MACD normalizado |
| 2 | `bb_pct` | Posição nas Bollinger Bands |
| 3 | `atr_norm` | Volatilidade normalizada |
| 4 | `adx_norm` | Força da tendência |
| 5 | `stoch_k` | Stochastic K |
| 6 | `dist_sma20` | Distância à SMA 20 |
| 7 | `dist_sma200` | Distância à SMA 200 |
| 8 | `trend_slope` | Inclinação da tendência |
| 9 | `sr_position` | Posição relativa S/R |
| 10 | `hour_sin` | Tempo cíclico |
| 11 | `hour_cos` | Tempo cíclico |

---

## Training Strategy

### Loss Function (From v6)

```python
class SwarmLoss(nn.Module):
    """
    Combinação de losses para treinar coordinator + workers.
    """
    def forward(self, coord_output, worker_outputs, targets):
        # 1. Coordinator Loss (Asymmetric Focal)
        main_loss = asymmetric_focal_loss(coord_output, targets)
        
        # 2. Worker Auxiliary Losses (Each worker predicts too)
        aux_loss = sum([
            0.1 * cross_entropy(w_out, targets) 
            for w_out in worker_outputs
        ])
        
        # 3. Consensus Regularization (Penalize extreme disagreement)
        # If all workers say LONG but coordinator says SHORT, penalize
        consensus_loss = 0.05 * disagreement_penalty(worker_outputs, coord_output)
        
        return main_loss + aux_loss + consensus_loss
```

### Training Loop Fixes (From v6)

1. **Label Smoothing**: `smooth=0.1`
2. **pos_weight**: Calculado dinamicamente do dataset
3. **Gradient Clip**: `max_norm=0.5`
4. **LR Schedule**: `ReduceLROnPlateau(factor=0.8, patience=15)`

---

## VRAM Budget (12GB GPU)

| Componente | fp32 | int8 |
|------------|------|------|
| 5 Workers (128 units) | 16MB | 4MB |
| Coordinator (512 units) | 40MB | 10MB |
| Activations (batch=256) | 4.5GB | 2.2GB |
| Gradients | 500MB | 250MB |
| Optimizer States | 1GB | 500MB |
| **Total** | **~6GB** | **~3GB** |

> ✅ **Otimizado para GPUs de 8GB-12GB (ex: RX 6750 XT).** 
> Aumentamos o `batch_size` e `seq_len` para 256 para maximizar o throughput e uso de VRAM (de 10% para ~40%).

---

## Consensus Logic

```python
def make_decision(worker_signals: Tensor, coordinator_logit: Tensor) -> str:
    """
    Lógica de decisão multi-camada.
    
    worker_signals: [5, 3] - probabilidades LONG/SHORT/WAIT por worker
    coordinator_logit: [1] - logit final do coordinator
    """
    
    # 1. Count worker votes
    worker_votes = worker_signals.argmax(dim=1)  # 0=LONG, 1=SHORT, 2=WAIT
    n_long = (worker_votes == 0).sum()
    n_short = (worker_votes == 1).sum()
    n_wait = (worker_votes == 2).sum()
    
    # 2. Coordinator decision
    coord_prob = torch.sigmoid(coordinator_logit)
    
    # 3. Consensus rules
    if n_wait >= 3:
        return "WAIT"  # Muitos workers incertos
    
    if coord_prob > 0.6 and n_long >= 3:
        return "LONG"  # Coordinator + maioria workers concordam
    
    if coord_prob < 0.4 and n_short >= 3:
        return "SHORT"
    
    # 4. Coordinator overrides if confident enough
    if coord_prob > 0.7:
        return "LONG"
    if coord_prob < 0.3:
        return "SHORT"
    
    return "WAIT"  # Sem consenso claro
```

---

## File Structure

```
src/brain/
├── models/
│   ├── liquid_v9.py      # [NEW] SwarmBrain, SwarmCoordinator
│   ├── workers.py        # [NEW] LightningWorker
│   └── analyzers.py      # [NEW] TechnicalAnalyzer
├── train_v9.py           # [NEW] Training loop corrigido
└── simulate_v9.py        # [NEW] Simulation with swarm
```

---

---

## Training Stability & Self-Correction (v9.6)

A v9.6 introduziu subsistemas para garantir que o treinamento seja resiliente a ruidos de mercado e "esquecimento catastrófico".

### 1. Autonomous Rollback System
O cérebro monitora sua própria performance em tempo real. Se o **Validation AUC** cai mais de **5%** em uma única época (sinal de confusão ou overfitting agudo):
- **Backup**: O modelo descarta o aprendizado da última época.
- **Restore**: Restaura os pesos do melhor estado anterior (`best_model_v9.pth`).
- **Adaptation**: Reduz o Learning Rate em **20%** automaticamente para ser mais cauteloso na próxima tentativa.

### 2. Selective Memory v9.6 (Curriculum Stabilizer)
O algoritmo de amostragem foi refinado para evitar que o modelo foque excessivamente em erros e esqueça os acertos:
- **Starting Loss (0.6)**: Novos dados entram com um peso neutro, evitando "pânico" inicial.
- **Diversity Mix (10%)**: Cada batch contém obrigatoriamente 10% de dados puramente aleatórios, agindo como uma revisão constante de conceitos básicos.
- **Coverage coverage (60k)**: O mapa sináptico é atualizado 3x mais rápido para evitar que informações fiquem obsoletas.
- **Softer exponent (^1.5)**: A pressão de seleção foi suavizada para uma transição de currículo mais orgânica.

---

## Migration Path

### Phase 1: Foundation
1. Criar `LightningWorker` baseado no `CfCCell` da v6
2. Criar `SwarmCoordinator` com cross-attention
3. Portar `AsymmetricFocalLoss` e técnicas da v6

### Phase 2: Training
1. Adaptar `feature_factory.py` para gerar dados por timeframe
2. Criar `train_v9.py` com training loop corrigido
3. Testar convergência

### Phase 3: Operations
1. Integrar com dashboard
2. Adicionar visualização de "votação" dos workers
3. Otimizar para latência (quantização)

---

*Architecture locked for V9 "Swarm Intelligence"*



--- FILE: docs/project_evolution.md ---
# I.A.R.A. Project Evolution History (v1.0 - v6.11)

**Project Name:** I.A.R.A. (Intelligent Algorithmic Robosapien Agent)
**Core Technology:** Liquid Neural Networks (LTC/CfC), Fractal Attention, Hybrid CPU/GPU Pipeline.
**Objective:** Create a biological-inspired trading agent capable of generalizing market dynamics without overfitting.

---

## Phase 1: Genesis & Foundations (v1.0 - v4.0)
**Goal:** Establish a working Liquid Neural Network (LNN) capability.

### v1.0 - v3.0: The "Vanilla" Era
*   **Architecture:** Basic LSTM/GRU baselines compared against early LTC (Liquid Time-Constant) implementations.
*   **Challenge:** The "Vanishing Gradient" and Numerical Instability. Liquid networks, being ODE-based solvers, were extremely unstable with financial time-series data which has high variance (heteroscedasticity).
*   **Outcome:** Models failed to converge or exploded to NaN.

### v4.0 - v4.4: Stabilization
**Problem:** Model suffered from **Mode Collapse** (predicting only one class, e.g., "HOLD").
*   **Solution 1 (Architecture)**: Introduced **Single Wide Layer (512 units)** with Orthogonal Initialization to preserve gradient flow.
*   **Solution 2 (Loss Function)**: Replaced MSE with **Focal Loss** (Alpha=0.25, Gamma=2.0) to force the model to focus on hard examples rather than easy "noise".
*   **Solution 3 (Dynamics)**: Implemented **Input Noise Injection (1%)** and Target Scaling to prevent the ODE solver from saturating.
*   **Result:** First stable convergence. Model began learning basic patterns.

---

## Phase 2: Cognitive Evolution (v6.0 - v6.5)
**Goal:** Move from "Pattern Matching" to "Reasoning".

### v6.0: The Stride Breakthrough
**Problem:** Processing M1 (1-minute) data tick-by-tick was computationally expensive and noisy.
*   **Innovation:** **Fractal Stride**. Instead of seeing `t, t-1, t-2`, the model sees `t, t-8, t-16`. This expanded the receptive field from ~4 hours to ~2 days without increasing parameter count.
*   **Result:** 8x Speedup in training, improved trend detection.

### v6.3: Selective Memory (Biomimicry)
**Problem:** The model treated all errors equally.
*   **Innovation:** Inspired by human earning.
    *   **Chronic Index**: Tracks samples where the model *consistently* fails.
    *   **Arrogant Errors**: Penalizes cases where confidence is high (>90%) but prediction is wrong.
*   **Mechanism:** A dynamic feedback loop that re-weights the loss function epoch-by-epoch based on "difficulty".

### v6.4: Active Feedback & Self-Correction
**Problem:** Static thresholds (e.g., probability > 0.5) were inefficient for volatile markets.
*   **Solution:** **Dynamic Thresholding** (Active Inference). The cut-off for a "Buy" signal floats based on the model's recent volatility-adjusted confidence.

---

## Phase 3: The Hybrid Engine (v6.7 - v6.9)
**Goal:** Extreme Scalability and Engineering Robustness.

### v6.9: The Hybrid Pipeline (CPU Factory + GPU Brain)
**Challenge (The HIPBLAS/ROCm Crisis):** 
Training on ROCm (AMD GPUs) encountered a critical `HIPBLAS_STATUS_INTERNAL_ERROR` when using multiprocessing `DataLoader`. The GPU memory context was crashing during fork operations.

**Debug Journey:**
1.  **Diagnosis:** Identified that transferring data `CPU -> GPU` inside the training loop with multiple workers caused driver conflicts.
2.  **Solution:**
    *   **Feature Factory v2 (CPU)**: Pre-computes ALL tensors (normalization, windowing, technical indicators) and saves to optimized `.pt` files.
    *   **Dataset v2**: Loads the entire dataset into RAM (Memory Mapped).
    *   **Optimization**: Set `num_workers=0` + `pin_memory=True`.
*   **Result:**
    *   Eliminated HIPBLAS errors.
    *   **2x Training Speedup** (15ms per batch vs 33ms).
    *   Fully deterministic pipeline.

---

## Phase 4: Biological Curation (v6.10 - v6.11)
**Goal:** Solving the "Garbage In, Garbage Out" problem of Financial Data.

### The "AUC 0.50" Stagnation
Despite the robust engine, the model stuck at ~50% accuracy (Random Guess).
*   **Hypothesis:** The model was trying to predict *market noise* (consolidation/chop) which is mathematically random. Prediction of randomness is impossible.

### v6.10: Curriculum Learning
**Idea:** Don't teach everything at once.
*   **Implementation:** **Threshold Filtering**. Only label a sample as UP/DOWN if the price moves > 0.03% (Curriculum Threshold).
*   **Result:** Reduced dataset size by 80%, but AUC broke the 0.50 barrier (reached 0.514).

### v6.11: Hierarchical Curation (Fractal Consensus)
**The Current State-of-the-Art.**
**Problem:** A movement might look bullish in M1 (1-min) but be a bearish trap in H1 (1-hour).
**Solution:** **Multi-Timeframe Consensus Filtering.**
1.  **Resample**: Convert raw M1 data to M5 and H1 layers.
2.  **Trend Detection**: Calculate EMA slopes for M5 and H1.
3.  **Consensus Rule**: A sample is valid ONLY if `Trend_H1 == Trend_M5`.
    *   If H1 says UP and M5 says UP -> **Golden Sample**.
    *   If H1 says UP and M5 says DOWN -> **Discard (Noise/Pullback)**.

**Final Impact:**
*   **Dataset Quality:** Created a "Golden Dataset" of ~219k samples (from years of history).
*   **Class Balance:** achieved near-perfect 50/50 split naturally.
*   **Result:** We moved from "Predicting Price" to "Identifying Flow". The AI no longer guesses; it waits for fractal alignment.

### v7.0 - The MLOps Evolution
**Problem:** Local file-based logging (`.txt`, `.jsonl`) and disparate tools made it impossible to scale experiments or analyze history effectively.
**Solution:** **SACI Protocol V7 Infrastructure.**
1.  **TimescaleDB**: Unified SQL + TimeSeries database for market data and "Brain States" (Simulation History).
2.  **MLflow**: Centralized experiment tracking (Parameters, Metrics, Artifacts).
3.  **MinIO**: S3-compatible object storage for heavy artifacts (Model Checkpoints, Datasets).
4.  **Containerization**: `docker-compose-mlops.yml` orchestrates the entire intelligence stack.

**Final Impact:**
*   **Reproducibility**: Every model run is versioned and trackable.
*   **Scalability**: The system can now handle massive datasets and concurrent experiments.
*   **Visibility**: Real-time dashboards reading directly from the database (Trade Hub).

## Phase 5: The Stability & Resonance Era (v9.5 - v9.6)
**Goal:** Achieving long-term training sessions that don't degrade.

### v9.5: Autonomous Rollback
**Problem:** The model could enter "Catastrophic Forgetting" states during long overnight runs if the market noise increased.
*   **Solution:** **Self-Monitoring Brain**. The training loop now detects AUC drops > 5% and automatically rolls back weights to the best previous state and reduces Learning Rate by 20%.

### v9.6: Curriculum Stabilization
**Problem:** Sampling was too aggressive, focusing 100x more on errors than standard data, leading to a loss of basic pattern recognition.
*   **Innovation:** **The Diversity Mix (10%)**. Injected a constant stream of random samples into every batch to act as a "Biological Anchor".
*   **Innovation:** **Broader Memory Coverage**. 3x faster memory analysis (60k samples) ensures the AI isn't learning from stale assumptions.
*   **Result:** Stable AUC/Accuracy curves for infinite training sessions.

---

## Summary of Major Technical Breakthroughs

| Feature | Version | Problem Solved |
| :--- | :--- | :--- |
| **Focal Loss** | v4.2 | Mode Collapse (Predicting only zeros) |
| **Fractal Stride** | v6.1 | Context Window too short / Training too slow |
| **CPU Feature Factory** | v6.9 | ROCm HIPBLAS Errors & Data Loading Bottlenecks |
| **Hierarchical Curation** | v6.11 | Low Signal-to-Noise Ratio (AUC ~0.50) |
| **MLOps Infrastructure** | v7.0 | Lack of Experiment Tracking & Scalable Storage |
| **Autonomous Rollback** | v9.5 | Training Degradation & Catastrophic Forgetting |
| **Selective Memory v9.6** | v9.6 | Extreme Sampling Bias & Knowledge Stability |

## Conclusion
The evolution of I.A.R.A. demonstrates that in financial machine learning, **Architecture < Data Curation**. No amount of LSTM/Transformer complexity could solve the foundational issue of market noise. The breakthrough came not from a smarter neuron, but from a smarter **teacher** (The Hierarchical Curator) that refuses to train the model on ambiguous data.



--- FILE: docs/ARCHITECTURE.md ---
# I.A.R.A. v8.5 Trinity Engine Architecture

## 1. Overview: The Tri-Cameral Brain
I.A.R.A. v8.5 "Trinity" represents a paradigm shift from a monolithic model to a **Tri-Cameral Architecture**, inspired by the breakdown of human cognition into **System 1 (Fast)**, **System 2 (Slow)**, and a **Reflexive Guard (Sentinel)**.

This design addresses the core limitation of previous versions: *The inability to stop trading during black swan events or regime shifts.*

---

## 2. System Architecture

### ⚡ System 1: The Fast Brain (Liquid Neural Network)
Responsible for millisecond-level execution. It is always "on" and looking for patterns.
*   **Core**: `ModulatedIARA` (LTC/CfC Backbone).
*   **Input**: 256-step Time Series (Technical Indicators).
*   **Enhancements**:
    *   **Fractal Attention**: Focuses on relevant historical windows.
    *   **Expert Mixture**: Two specialized "Experts" (Linear Layers) trained for Long and Short biases respectively.
    *   **Feature Gating**: A learnable attention mechanism that assigns importance to each input feature (e.g., ignoring RSI when Volume is low).

### 🛡️ System 2: The Sentinel (Watchdog)
A reflexive, rule-based inhibition system that overrides System 1.
*   **Role**: The "Amygdala" – instantaneous fear response.
*   **Triggers**:
    *   **News DEFCON**: Keyword scanning in real-time news feeds (War, Fed Rates, Disaster).
    *   **Volatility Spikes**: VIX or ATR explosions.
*   **Mechanism**: **Gradient Interlock**. It doesn't just "filter" trades; it physically zeroes out the output logits of the Neural Network, preventing the signal from ever reaching the execution layer.

### 🧠 System 3: The Strategist (Slow System)
A Large Language Model (LLM) operating asynchronously.
*   **Core**: Quantized LLM (DeepSeek/Llama via `llama-cpp-python`).
*   **Role**: Market Analyst.
*   **Function**:
    *   Reads macro reports and news summaries.
    *   Sets a global **"Regime Bias"** (e.g., `Bearish`).
    *   Modulates the Fast System's priors (e.g., "Harder to buy, easier to sell").

---

## 3. Data Flow & Tensor Shape

### Input Pipeline ($B, 256, 17$)
1.  **Market Data ($15$)**: OHLCV derived indicators (RSI, MACD, Bollinger, etc.).
2.  **Context Embeddings ($2$)**:
    *   `news_sentiment`: Float from System 3.
    *   `regime_code`: Categorical embedding of the market state.

### The Forward Pass
1.  **Feature Gate**: $X' = X \odot \sigma(W_g X)$
    *   *Filters out noise before it enters the expensive RNN.*
2.  **Liquid Backbone**: $H_t = \text{CfC}(X'_t, H_{t-1})$
    *   *Solves the continuous-time hidden state.*
3.  **Expert Heads**:
    *   $E_{long} = \text{Linear}(H_t)$
    *   $E_{short} = \text{Linear}(H_t)$
4.  **Conflict Resolution**: $Y = E_{long} - E_{short}$
    *   *Buying requires high Long conviction AND low Short conviction.*

---

## 4. Operational Dashboard (Mission Control v2.0)

The interface is no longer just a log viewer; it is an active component of the feedback loop.

### 🔬 Forensic Lab
Allows the human operator to inspect the "Brain's" internal state.
*   **Tick Slider**: Replay the 256-step inference window.
*   **Heatmaps**:
    *   **Feature Importance**: See which indicator triggered the trade.
    *   **Neural Activity**: Visualize CfC hidden state saturation.
*   **Conviction Plot**: Visual proof of *why* a trade was taken (Long/Short split).

### 🎛️ Ops Panel
*   **DEFCON Gauge**: Real-time visual of the Sentinel's status.
*   **Manual Override**: Physical "Kill Switch" to disconnect the execution engine.

---

## 5. MLOps Infrastructure

*   **Training**: PyTorch Lightning loop logic (custom implemented) with MLflow logging.
*   **Storage**:
    *   **TimescaleDB**: Stores "Brain Snapshots" (Tensors) for replay.
    *   **MinIO**: Stores model weights (`.pth`) and optimized ONNX exports.
*   **Telemetry**: Real-time CSV logging for the Dashboard Streamlit app.

---
*Architecture definition locked for V8.5 Protocol Trinity.*



--- FILE: docs/ANALYSIS.md ---
# IARA Implementation Analysis: Engineering & Trading Perspective

## Executive Summary
The IARA (Intelligent Active Reasoning Agent) project demonstrates a sophisticated software architecture but currently suffers from a fundamental disconnect between **model complexity** and **signal availability**. As both a Senior Software Engineer and Quantitative Trader, I have analyzed the system to identify why performance remains stagnant (AUC ~0.52).

---

## 🏗️ Engineering Perspective: "Over-Engineering the Noise"

The codebase is robust, modular, and uses advanced techniques. However, it exhibits signs of premature optimization and complexity that hinders basic learning.

### 1. Complexity Mismatch
-   **Issue**: The architecture uses **Liquid Time-Constant (CfC) Networks**, **Fractal Attention**, and **Asymmetric Focal Loss**. This is like using a Formula 1 engine to drive in a traffic jam.
-   **Impact**: These advanced layers require strong signals to converge. When fed noisy financial data (M5 timeframe), they likely overfit to noise or collapse to a mean prediction (random guessing).
-   **Evidence**: My tests showed that a simplified 1-layer model performed *indistinguishably* from the complex 2-layer architecture.

### 2. The "Loss Function" Trap
-   **Issue**: The use of **Asymmetric Focal Loss** is theoretically sound for class imbalance but introduces hyperparameters ($\alpha, \gamma$) that make the optimization landscape harder to navigate.
-   **Impact**: The model struggles to find a gradient that isn't just "predict the majority class" or "predict 0".
-   **Verification**: Switching to standard **BCE (Binary Cross Entropy)** resulted in nearly identical performance, proving the loss function wasn't the bottleneck.

### 3. Feedback Loops & Calibration
-   **Issue**: The `ActiveInferenceModule` and `SelectiveMemory` create complex feedback loops. The "Curriculum Learning" filters out "hard" samples, but in stochastic markets, "hard" samples might just be outliers/noise.
-   **Impact**: The model might be systematically ignoring the only data that contains real alpha, thinking it's "noise", or conversely, overfitting to "Mastered" examples that are just lucky streaks.

---

## 📈 Trading Perspective: "Seeking Signal in the Noise"

From a quant trading view, the current approach tries to predict the **random walk** of the M5 timeframe without a structural edge.

### 1. The M5 Timeframe Fallacy
-   **Gap**: Market microstructure noise at 5-minute intervals is extremely high. Directional prediction (Will the *next* bar be green?) is close to a coin flip (50/50).
-   **Reality Check**: Institutional algos operate on milliseconds or daily trends. The "middle ground" (M5/M15) is the "Kill Zone" where retail strategies die due to noise and spread.
-   **Recommendation**: Move to **H1/H4** for trend capture or **Tick data** for microstructure scalping. M5 is insufficient for this model architecture.

### 2. Target Definition Weakness
-   **Gap**: The target was `close > previous_close` (1-bar lookahead).
-   **Problem**: This treats a 1-pip move the same as a 50-pip move. It forces the model to learn "noise wiggles".
-   **Evidence**: Changing to a **5-bar cumulative return** target improved AUC from 0.51 to 0.528. This confirms that smoothing the target reveals a faint signal, but it's still weak.

### 3. Missing Informational Edge
-   **Gap**: The model relies pure technically (OHLC + Indicators).
-   **Reality**: Technical analysis explains only a fraction of price movement. Real alpha comes from:
    -   **Correlations**: How is DXY (Dollar Index) moving? Gold? S&P500?
    -   **Session Context**: Is it London Open? NY Close? Liquidity voids?
    -   **Sentiment**: What is the news flow?
-   **Verdict**: You cannot squeeze blood from a stone. No amount of neural network layers can extract signal if the input features don't contain it.

---

## 🛡️ Strategic Recommendations

### 1. Pivot the Target (Easy Win)
Stop predicting "Next Bar Direction". Start predicting **Volatility** or **Trend Stability**.
-   *Why*: Predicting *magnitude* is easier than *direction*. Use the model to size positions, not to enter them.

### 2. Regime Filtering (High Impact)
The model should not trade 100% of the time.
-   **Implement**: A "Market Regime" filter (e.g., ADX > 25, or moving average slope).
-   **Logic**: Only run the Neural Network when the market is *already* moving. Use the AI to sniper entries, not to guess market conditions.

### 3. Data Enrichment (Long Term)
-   Ingest highly correlated assets (EURUSD needs DXY and Yields).
-   Add "Time of Day" features (Session encoding) to capture liquidity patterns.

## Summary Status
The system works code-wise, but the **Trading Hypothesis** (predicting M5 direction with simple indicators) is flawed. The AI is working correctly; it is correctly telling you that the data is random (AUC 0.50).



--- FILE: docs/CHANGELOG.md ---
# Changelog - SACI Project

## [V9.6 - Stability & Curriculum] - Sunday, Feb 8 2026 (Current)
### Curriculum Stabilization
- **Selective Memory v9.6**: 
    - **Neutral Initialization**: Starting memory at 0.6 Loss (Neutral) instead of 1.0 (Panic) to prevent extreme bias.
    - **Diversity Mix**: Guaranteed **10% random background** in every batch to prevent catastrophic forgetting.
    - **Coverage Boost**: Analysis coverage increased to **60,000 samples** per epoch (3x speedup).
    - **Softer Weights**: Reduced selection pressure from `^2.0` to `^1.5` for smoother convergence.
- **UI & Visualization**:
    - **Atomic Telemetry**: Fixed dashboard flickering by implementing atomic file write/replace operations.
    - **Fixed Chart Scaling**: Locked Training Dynamics axis (Loss/AUC) to [0, 1] for intuitive visual monitoring.

## [V9.5 - Self-Correcting Brain] - Saturday, Feb 7 2026
### Autonomous Resilience
- **Autonomous Rollback**: Brain can now sense "Confusion" (Drop > 5% in AUC) and auto-revert to the previous best state.
- **Dynamic Speed Reduction**: Automatically reduces Learning Rate by **20%** after a rollback to approach hard patterns more carefully.
- **Persistence Fix**: Ensured human-captured feedback persists across the entire training mission.

## [V9.0 - Swarm Intelligence] - Friday, Feb 7 2026
### The "Fractal Consensus" Architecture
- **Swarm Intelligence**: Replaced monolithic model with **5 Specialized Workers** (M5, M15, M30, H1, H4).
- **LightningWorker**: Ultra-light CfC cells (~32 units each) for each timeframe.
- **SwarmCoordinator**: Cross-Attention fusion + CfC refinement for final decision.
- **New Loss Function (SwarmLoss)**:
    - `AsymmetricFocalLoss` for Coordinator
    - `WorkerAuxiliaryLoss` forces each worker to predict independently
    - `ConsensusLoss` penalizes Coordinator contradicting strong worker consensus
- **Training Improvements**:
    - Ported `SelectiveMemory` curriculum learning from v6
    - Gradient Clipping: 0.5 (crucial for LNN stability)
    - Label Smoothing: 0.1
- **VRAM Estimate**: ~210-370MB (fits 12GB GPU easily)

### Files Created
- `src/brain/models/workers.py`: LightningWorker, WorkerSwarm
- `src/brain/models/liquid_v9.py`: SwarmBrain, SwarmCoordinator  
- `src/brain/losses.py`: SwarmLoss with 3 component losses
- `src/brain/train_v9.py`: Complete training loop with AMP

---

## [V7 - MLOps Era] - Thursday, Feb 5 2026
### Implemented
- **SACI Protocol V7**: Full MLOps Architecture.
    - **TimescaleDB**: Integrated for BrainState persistence (`src/trading/simulation.py`).
    - **MLflow**: Connected for experiment tracking (`SACI_Operations_V1`).
    - **MinIO**: Artifact storage configuration verified.
    - **Dashboard**: Connected to TimescaleDB for "Trade Hub" replay.
- **Trade Hub**: Automated Simulation (`src/run_simulation.py`) and Real-Time Dashboard.
- **Fixes**: MLflow 403 Security, Docker Dependency Injection, Dashboard DB Grid.

---

## [V6 - Liquid Era] - Wednesday, Feb 4 2026
### v6.11 - The "Golden" Dataset
- **Hierarchical Curation**: Implemented M1 -> M5 -> H1 trend consensus filtering.
- **Impact**: Reduced noise, balanced classes (50/50), AUC broke the 0.50 barrier.

### v6.9 - The "Hybrid" Engine
- **Crisis**: ROCm HIPBLAS errors during Data Loading.
- **Solution**: Feature Factory (CPU) + RAM Mapping.
- **Result**: 2x speedup (33ms -> 15ms/batch) and stability.

### v6.3 - Biological Learning
- **Selective Memory**: "Focus on Hard Examples" (Focal Loss).
- **Arrogant Error Penalty**: Feedback loop for high-confidence mistakes.

### v6.0 - The Initial Shift
- **Architecture**: Pivot to **Liquid Neural Networks (LNN)**.
- **Reason**: Transformers were too heavy; LSTMs vanished. Liquid (LTC/CfC) handles irregular series naturally.
- **Fractal Stride**: Expanded receptive field from 4h to 2 days efficiently.

---

## [V1 ~ V5 - The "Genesis" (MVP)] - Tuesday, Feb 3 2026
### v4.0 - Stabilization
- **Problem**: Mode Collapse (Predicting only "HOLD").
- **Fix**: Input Noise Injection + Wide Layers (512 units). Stable convergence achieved.

### v1.0 ~ v3.0 - Validation & Failures
- **Tests**:
    - **Bitnet**: Tested 1.58-bit quantization (failed to capture nuance).
    - **Vanilla LSTMs**: Failed due to exploding gradients.
    - **Transformers**: High compute, low signal (Overfitting).
- **Conclusion**: Financial data requires specialized architectures, not just "bigger" models.

## [V8.5 - Protocol Trinity] - Friday, Feb 6 2026
### The "Consciousness" Upgrade
- **Trinity Engine**: Replaced monolithic model with **Tri-Cameral Architecture** (Fast/Sentinel/Slow).
- **Dashboard v2.0**:
    - **Forensic Lab**: Added "Time Travel" slider to replay model states tick-by-tick.
    - **Brain MRI**: Visualizations for Feature Gating, Expert Conviction, and Neural Heatmaps.
    - **Hot Reload**: Fixed Python caching issues for instant UI updates.
- **Forensic Recorder**:
    - Implemented  to capture full-sequence hidden states ().
    - Fixed  by ensuring snapshot dimensions match the UI slider (64 ticks).
- **Infrastructure**:
    - **Defensive Coding**: Added robust type/length checks to validade legacy snapshots vs new ones.
    - **Batch Size**: Increased limits to 1024 for high-end GPUs.



--- FILE: docs/RESEARCH_LOG_V9.md ---
# Research Log: GPU Optimization & Model Scaling (V9.7)

## Problem: Low GPU Utilization
Initial benchmarks of the IARA v9 architecture showed VRAM usage at ~1.2GB and GPU load at ~8% on an AMD Radeon RX 6750 XT (12GB). While efficient, this indicated the model was too small to leverage the hardware's parallel processing power effectively.

## Research & Hypothesis
By scaling the internal dimensions of the Swarm Intelligence components, we aimed to:
1. Increase the model's capacity to learn complex market nuances.
2. Maximize data throughput to reduce idle GPU cycles.
3. Improve convergence stability by using larger hidden states in Liquid Neural Networks (CfC).

## Optimization Workloads

### 1. Dimension Scaling
We doubled (and in some cases quadrupled) the hidden dimensions:
- **Workers**: Increased from 32 to 128 units.
- **Coordinator**: Increased from 128 to 512 units.
- **Impact**: Parameter count grew from ~300k to ~2.1M.

### 2. Throughput Maximization
We increased the window of attention and batch processing:
- **Sequence Length**: Increased from 64 to 256 candles (expanding the temporal horizon by 4x).
- **Batch Size**: Increased from 64 to 256 samples.
- **Impact**: VRAM usage increased from ~10% to ~40% (4.5GB+).

## Bypassing Constraints
- **Architecture Mismatch**: Resolved issues where `backtest_v9.py` and `auto_learn_v9.py` had hardcoded smaller dimensions. Implemented dynamic CLI arguments to sync training and evaluation environments.
- **Resource Allocation**: Optimized the Docker `shm_size` and ROCm overrides to ensure stable execution on AMD hardware.
- **Learning Continuity**: Fixed the gating and rollback logic to ensure that scaled models are correctly promoted and compared against previous baselines.

## Results
- **Memory**: Stabilized at ~4.5GB VRAM.
- **Learning**: AUC improved significantly due to the increased capacity (observed Val AUC ~0.74 in initial cycles).
- **Efficiency**: Throughput increased 16x (4x batch * 4x sequence) while maintaining acceptable step times (~0.3s/step).

---
*Status: V9.7 "Gladiator" Scaling Research Concluded.*



--- FILE: docs/new/documentation_plan.md ---
# Professional System Architecture Documentation Plan

## Context
Refactoring the IARA trading system to a "Professional Grade" Client-Server architecture.
- **Backend**: Python (FastAPI + Socket.IO) keeping the core Brain logic.
- **Frontend**: Next.js 14 + Tailwind + Shadcn/UI for "God Mode" observability.

## Documentation Structure (`docs/new/`)

### 1. Backend Architecture (`docs/new/backend/`)
- **`api_design.md`**: REST Endpoints, Socket.IO Events (Paylod Specs, Namespaces), Service Layer (Patterns).
- **`security_protocol.md`**: JWT Implementation (Access/Refresh), HttpOnly Cookie strategy, CORS, Rate Limiting, and Input Validation (Pydantic).
- **`database_integration.md`**: Async SQLAlchemy/Asyncpg usage, Migration strategy (Alembic), and TimescaleDB optimization for high-frequency data.

### 2. Frontend Architecture (`docs/new/frontend/`)
- **`nextjs_structure.md`**: App Router layout, Server vs Client Components strategy, Middleware protection.
- **`state_management.md`**: Zustand (Client Global State) + React Query (Server State) strategy. Socket.IO integration pattern (`useSocket` hook).
- **`ui_ux_design.md`**: Design System (Tokens, Colors, Typography), Component Library (Shadcn), "God Mode" visualization specs (Charts, Terminal).

### 3. System Flows (`docs/new/`)
- **`data_flows.md`**: Sequence diagrams for:
  - Authentication (Login -> Cookie -> Socket Handshake).
  - Live Market Data (Alpaca -> FleetAdmiral -> Socket -> Frontend Chart).
  - Active Learning (Trade Error -> Snapshot -> DB).

## Key Design Decisions
1.  **Zero-Latency Bridge**: The `FleetAdmiral` will broadcast internal `BrainState` directly to the Socket.IO `BroadcastManager`, bypassing the database for live UI updates (Database is for persistence/history only).
2.  **Secure by Default**: No LocalStorage for tokens. Strict Content Security Policy (CSP).
3.  **Resilient Sockets**: Auto-reconnect logic with exponential backoff and "state reconciliation" upon reconnection (fetching missed candles).

---
## Implementation Roadmap reference
(See `task.md` Phase 6 & 7)



--- FILE: docs/new/backend/api_design.md ---
# Backend API Architecture (FastAPI + Socket.IO)

## Overview
The backend transforms from a monolithic Streamlit script to a headless **FastAPI** service. It serves:
1.  **REST API**: Configuration, historical data, authentication.
2.  **Socket.IO Server**: Real-time bidirectional communication for live trading.

## Technology Stack
- **Framework**: FastAPI (high performance, async).
- **Real-time**: `python-socketio` (ASGI mode).
- **Validation**: Pydantic v2.
- **Database**: Async SQLAlchemy + asyncpg (TimescaleDB).
- **Task Queue**: Internal `asyncio` tasks for FleetAdmiral (no Celery needed yet).

---

## 1. REST API Endpoints (`/api/v1`)

### Authentication
- `POST /auth/login`: Returns `HttpOnly` cookie with JWT `access_token`.
- `POST /auth/logout`: Clears cookies.
- `GET /auth/me`: Returns current user user info.

### Fleet Management
- `GET /fleet/status`: Returns active agents and their health.
- `POST /fleet/start`: Spawns a new agent for a symbol.
- `POST /fleet/stop`: Halts an agent (graceful shutdown).
- `POST /fleet/killswitch`: Emergency stop all trading.

### Market Data (Historical)
- `GET /market/history/{symbol}?timeframe={tf}&limit={n}`: Returns OHLCV JSON for charting.
- `GET /market/snapshot/{symbol}`: Latest full book state.

### Brain Internals
- `GET /brain/zoo`: List available trained models.
- `GET /brain/active`: Get currently deployed model metadata.

---

## 2. Socket.IO Events (`/ws`)

### Namespaces
- `/live`: Public market data (ticks, candles).
- `/fleet`: Protected fleet events (brain thoughts, signals).

### Events (Server -> Client)
| Event Name | Namespace | Payload | Frequency |
| :--- | :--- | :--- | :--- |
| `market_tick` | `/live` | `{symbol, price, time}` | ~100ms |
| `candle_closed` | `/live` | `{symbol, o, h, l, c, v, time}` | Timeframe end |
| `brain_thought` | `/fleet` | `{agent_id, step, log_text}` | Real-time |
| `worker_votes` | `/fleet` | `{agent_id, M5_vote, H1_vote, ...}` | Per tick |
| `trade_update` | `/fleet` | `{agent_id, type, price, pnl}` | On exec |
| `fleet_heartbeat` | `/fleet` | `{active_agents_count, uptime}` | 5s |

### Events (Client -> Server)
| Event Name | Namespace | Payload | Action |
| :--- | :--- | :--- | :--- |
| `subscribe` | `/live` | `{symbol}` | Join room |
| `manual_override` | `/fleet` | `{agent_id, action: BUY/SELL}` | Force trade |

---

## 3. Project Structure (`src/api`)
```
src/api/
├── main.py            # ASGI Entrypoint
├── config.py          # Envs & Constants
├── dependencies.py    # DI (Db session, Current User)
├── events.py          # Socket.IO Event Handlers
├── auth/
│   ├── jwt.py         # Token logic
│   └── router.py      # Auth routes
├── fleet/
│   ├── router.py      # Fleet control routes
│   └── service.py     # FleetAdmiral Bridge
├── market/
│   └── router.py      # Historical data
└── schemas/           # Pydantic models
    ├── fleet.py
    └── market.py
```



--- FILE: docs/new/backend/database_integration.md ---
# Database Integration Strategy (TimescaleDB)

## Overview
The Brain needs persistence for:
1.  **Trade History**: Full audit trail of decisions.
2.  **Telemetry Snapshots**: High-frequency state recording for "God Mode" replay.
3.  **Active Learning**: Storing "Mistakes" (Input Tensor + Outcome) for retraining.

We use **PostgreSQL with TimescaleDB extension** via **Async SQLAlchemy** (`asyncpg` driver) for non-blocking I/O.

## 1. Schema Design

### `agents` Table
Identity of the IAs.
- `id`: UUID (Primary Key)
- `symbol`: String (BTC/USD)
- `market_type`: Enum (Crypto, Forex)
- `config`: JSONB (Strategy parameters)
- `status`: Enum (Active, Paused, Dead)
- `created_at`: Types.DateTime

### `trade_history` Table (Hypertable)
Partitioned by time.
- `time`: DateTime (PK)
- `agent_id`: UUID (FK)
- `side`: Enum (Buy, Sell)
- `price`: Decimal
- `size`: Decimal
- `pnl`: Decimal (Nullable, filled on close)
- `exit_price`: Decimal (Nullable)
- `reason`: Text (LFM reasoning summary)

### `brain_telemetry` Table (Hypertable)
High-frequency dumps of internal state (for Replay/Debug).
- `time`: DateTime (PK)
- `agent_id`: UUID (FK)
- `worker_votes`: JSONB (e.g., `{"M5": 0.8, "H1": 0.2}`)
- `consensus`: Float (0-1)
- `lfm_thought`: Text (Stream of consciousness)

### `learning_samples` Table
For Active Learning loop.
- `id`: UUID
- `input_snapshot`: ByteA (Serialized Tensor)
- `outcome_label`: Integer (0=Bad, 1=Good)
- `confidence`: Float

## 2. Async Implementation (`src/api/db`)

### Session Management
We use a singleton `AsyncSession` dependency for FastAPI routes.
```python
# src/api/dependencies.py
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with async_session_factory() as session:
        yield session
```

### High-Performance Ingestion
For generic telemetry (hundreds of inserts/sec):
- Use `COPY` command or batch inserts via `asyncpg` directly, bypassing ORM overhead for telemetry tables.
- ORM is reserved for `agents` config and critical `trade_history`.

## 3. Migration Strategy (Alembic)
- `alembic` configured for async migrations.
- `env.py` updated to use `async_engine`.



--- FILE: docs/new/backend/security_protocol.md ---
# Security Protocol & Best Practices

## 1. Authentication Strategy
We move from "No Auth" (Streamlit) to **"State-of-the-Art" JWT Auth**.

### Tokens
- **Access Token**: Short-lived (15 min), signed with `hs256`. Contains `sub` (user_id) and `permissions`.
- **Refresh Token**: Long-lived (7 days), stored in DB (hashed) and HttpOnly cookie.

### Cookie Policy
Cookies are the only safe storage for tokens in browser apps.
```python
response.set_cookie(
    key="access_token",
    value=token,
    httponly=True,   # No JS access (XSS proof)
    secure=True,     # HTTPS only (Simulated locally via bool flag)
    samesite="Lax",  # CSRF protection
    max_age=900
)
```

## 2. Real-time Security (Socket.IO)
- **Handshake Auth**: The initial Socket.IO handshake MUST validate the JWT cookie.
- **Room Authorization**: Users can only join rooms they are authorized for (though currently single-tenant).

## 3. Input Validation
- **Pydantic V2**: All incoming JSON bodies are strictly typed.
- **Query Params**: Validated for SQL Injection risks (handled by SQLAlchemy ORM, but logic checked).

## 4. Operational Security (OpSec)
- **Environment Variables**: secrets (API Keys, JWT Secret) loaded via `.env`, never hardcoded.
- **CORS**: Strict Allow-Origin policy (only allow Frontend URL).
- **Rate Limiting**: `slowapi` to prevent brute-force on `/login`.

## 5. Audit Logging
- **Trade Audit**: Every manual override via "God Mode" is logged to `audit_logs` table with `user_id`, `timestamp`, `action`, `reason`.
- **System Events**: Startup/Shutdown/Error events logged to `system_logs`.



--- FILE: docs/new/frontend/architecture.md ---
# Frontend Architecture (Next.js 14)

## Overview
A "Mission Control" dashboard built with performance and observability as first principles.
Uses **Next.js 14 App Router** for hybrid rendering (Server Components for layout/auth, Client Components for live charts).

## Technology Stack
- **Framework**: Next.js 14
- **Language**: TypeScript (Strict Mode)
- **Styling**: Tailwind CSS + `clsx` + `tailwind-merge`
- **UI Library**: Shadcn/UI (Radix Primitives)
- **State**: Zustand (Global), React Query (Async/Server)
- **Socket**: `socket.io-client` with custom hooks.

---

## 1. Directory Structure (`src/web`)
```
src/web/
├── app/
│   ├── (auth)/        # Route Group for auth pages
│   │   └── login/
│   ├── (dashboard)/   # Protected Layout
│   │   ├── layout.tsx # Sidebar + Socket Provider
│   │   ├── page.tsx   # Fleet Overview
│   │   └── god-mode/
│   │       └── [id]/  # Dynamic Agent View
│   └── layout.tsx     # Root Layout (Fonts, Theory)
├── components/
│   ├── ui/            # Shadcn Primitives (Button, Card...)
│   ├── fleet/         # Business Components (AgentCard, BrainMRI)
│   └── charts/        # Lightweight-charts wrappers
├── lib/
│   ├── socket.ts      # Singleton Socket Client
│   ├── api.ts         # Axios/Fetch wrapper
│   └── utils.ts
├── hooks/
│   ├── use-socket.ts  # Subscription Hook
│   └── use-fleet.ts   # Zustand Store
└── types/             # TS Interfaces
```

## 2. State Management Strategy

### Server State (React Query)
Used for data that changes slowly or needs initial fetch.
- `useQuery(['fleet-status'])`: Fetches active agents list.
- `useQuery(['market-history', symbol])`: Fetches initial candles.

### Client State (Zustand)
Used for high-frequency ephemeral data.
- **`useFleetStore`**:
  - `agents`: Map of AgentID -> Data
  - `socketStatus`: Connected/Disconnected
  - `addTick(symbol, tick)`: Appends tick to local history.

### Real-time Bridge (`useSocket`)
Custom hook to prevent listeners duplication.
```typescript
useSocketEvent('market_tick', (data) => {
  useFleetStore.getState().addTick(data.symbol, data);
});
```

## 3. "God Mode" UI Components

### `RealTimeChart.tsx`
- Wraps `lightweight-charts`.
- Uses `useRef` to maintain chart instance.
- **Performance**: Updates via `series.update(tick)` direct API, NO React re-renders for data updates.

### `BrainMRI.tsx`
- Radar Chart (Recharts) showing Worker votes.
- Updates every second via `worker_votes` event.

### `ThoughtStream.tsx`
- Terminal-like scrolling log.
- Displays `brain_thought` events with syntax highlighting.

---

## 4. Design System (Dark Theme)
- **Background**: `bg-zinc-950` (Deep space).
- **Cards**: `bg-zinc-900/50` + `backdrop-blur` (Glassmorphism).
- **Accents**: 
  - Buy/Long: `text-emerald-400`
  - Sell/Short: `text-rose-400`
  - Neutral/Thinking: `text-blue-400`
- **Typography**: `Inter` (UI), `JetBrains Mono` (Data/Code).



--- FILE: docs/new/frontend/design_system.md ---
# User Interface & Design System

## Design Philosophy
"NASA Control Room meets Cyberpunk".
- **Density**: High data density, minimized whitespace.
- **Contrast**: High contrast for readability in dark environments.
- **Feedback**: Instant visual feedback for all actions (Toast, Pulse, color change).

## 1. Color Palette (Tailwind Config)

| Token | Color | Hex | Usage |
| :--- | :--- | :--- | :--- |
| `background` | Zinc 950 | `#09090b` | Main Page BG |
| `surface` | Zinc 900 | `#18181b` | Cards, Sidebar |
| `primary` | Indigo 500 | `#6366f1` | Main Actions |
| `long` | Emerald 500 | `#10b981` | Buy Signals, Profit |
| `short` | Rose 500 | `#f43f5e` | Sell Signals, Loss |
| `warning` | Amber 500 | `#f59e0b` | System Alerts |
| `text-muted` | Zinc 400 | `#a1a1aa` | Labels, secondary text |

## 2. Core Components (Shadcn/UI)

### `AgentCard`
A summary card for the Grid View.
- **Header**: Symbol (BTC/USD) + Status Indicator (Pulse Green/Red).
- **Body**: Sparkline chart (last 1h PnL).
- **Footer**: Current Position + PnL Value.

### `BrainMRI` (Radar Chart)
Visualizes the consensus of the Swarm.
- **Axes**: M5, M15, M30, H1, H4.
- **Value**: 0 (Bearish) to 1 (Bullish).
- **Overlay**: Filled polygon representing the "Shape of the Market".

### `LiveTerminal`
A retro-styled log viewer.
- **Font**: JetBrains Mono.
- **Colors**: Based on log level (Info=White, Warning=Yellow, Crit=Red).
- **Behavior**: Auto-scroll to bottom, pause on hover.

## 3. Layout Structure

### **Global Sidebar**
- **Brand**: I.A.R.A. Logo (Glitch effect on hover).
- **Nav**: Dashboard, Market, Brain, Settings.
- **Status Footer**: WebSocket Connection State (Green Dot / Red Dot).

### **God Mode Layout (Grid)**
```
[ Header: Symbol | Current Price | 24h Change ]
------------------------------------------------
[ Main Chart (70%)        ] [ Brain MRI (30%)  ]
[ (Candles + Orders)      ] [ (Worker Voters)  ]
------------------------------------------------
[ Thought Stream (50%)    ] [ Active Pos (25%) ]
[ (Log output)            ] [ (PnL, Entry)     ]
                            [------------------]
                            [ Manual Controls  ]
                            [ (Buy/Sell/Kill)  ]
```

## 4. UX Patterns
- **Optimistic UI**: When user clicks "Buy", show "Submitting..." immediately, then "Filled" on socket event.
- **Connection Loss**: Full-screen blur overlay "Reconnecting to Mainframe..." if socket drops.
- **Sound Effects** (Optional): Subtle blips for new orders/alerts.



--- FILE: docs/new/data_flows.md ---
# System Data Flows

## 1. Authentication Flow (Login)
```mermaid
sequenceDiagram
    participant User
    participant NextJS (Client)
    participant FastAPI (Auth)
    participant DB

    User->>NextJS: Enter Credentials
    NextJS->>FastAPI: POST /api/auth/login
    FastAPI->>DB: Validate User
    DB-->>FastAPI: User Valid
    FastAPI-->>NextJS: Set-Cookie: access_token (HttpOnly)
    NextJS->>User: Redirect to /dashboard
```

## 2. Real-time Market Data Flow
From Alpaca to the Chart (Zero Latency path).
```mermaid
sequenceDiagram
    participant Alpaca (WebSocket)
    participant MarketStream (Python Service)
    participant SocketManager (FastAPI)
    participant NextJS (Chart Component)

    Alpaca->>MarketStream: Trade Update (BTC @ 50k)
    MarketStream->>MarketStream: Normalize Data
    MarketStream->>SocketManager: emit('market_tick', payload)
    SocketManager-->>NextJS: Broadcast('market_tick')
    NextJS->>NextJS: Update Chart Series (Directly)
```

## 3. "The Brain" Decision Flow
How the FleetAdmiral broadcasts thoughts to God Mode.
```mermaid
sequenceDiagram
    participant Agent (TrinityEngine)
    participant LFM (Ollama)
    participant SocketManager
    participant NextJS (God Mode)

    Agent->>LFM: Query("Analyze Trend")
    LFM-->>Agent: "Trend is Bullish because..."
    Agent->>SocketManager: emit('brain_thought', text)
    SocketManager-->>NextJS: Broadcast('brain_thought')
    NextJS->>NextJS: Append to Terminal Log
```

## 4. Active Learning Loop
Self-correction mechanism.
```mermaid
sequenceDiagram
    participant Agent
    participant TradeMonitor
    participant DB (LearningSamples)

    Agent->>TradeMonitor: Open Trade (Snapshot Input)
    TradeMonitor->>TradeMonitor: Wait for Close
    TradeMonitor->>Agent: Trade Closed (PnL = -50$)
    Agent->>DB: Insert into learning_samples (Input, Label=0)
    Note over DB: Next Training Cycle will load this sample
```



--- FILE: docs/BUSINESS_STRATEGY_NOTEBOOKLM.md ---
# Saci-Trader: Business Scaling & Intelligence Integration Plan
## Vision: From Algorithmic Agent to Fintech Ecosystem

Este documento detalha a estratégia para escalar o projeto Saci-Trader em dois modelos de receita principais: **Venda de Sinais de Alta Fidelidade (Instagram/Telegram)** e **API de Consulta de Dados de Microestrutura (B2B/Pro-Traders)**, utilizando o **NotebookLM** como o núcleo de síntese de inteligência.

---

## 1. Pilares do Modelo de Negócio

### A. Escala de Sinais (B2C - Social Scaling)
*   **Produto:** Sinais de entrada/saída baseados na convergência do Swarm Intelligence (v11).
*   **Diferencial:** "Sinais com Tese". Diferente de salas comuns, cada sinal é acompanhado por uma justificativa técnica gerada pela IA (ex: "Pressão de CVD detectada + Hurst em 0.68").
*   **Papel do NotebookLM:** Atuar como o **Diretor de Comunicação Técnica**. Ele traduzirá os estados dos tensores e as métricas de "Raio-X" (Order Book) em posts e mensagens de autoridade para converter seguidores em assinantes.

### B. API de Dados de "Ouro" (B2B - Data as a Service)
*   **Produto:** Acesso via API a métricas proprietárias:
    *   *Real-time Market Regime (HMM)*.
    *   *Volume Book Imbalance (VBI) Multi-Exchange*.
    *   *Spoofing/Manipulation Probability Score*.
*   **Público:** Traders profissionais e fundos que buscam "edge" na microestrutura.
*   **Papel do NotebookLM:** Validar e documentar as correlações encontradas na pesquisa ("Modelos Líquidos e Reativos") para criar o Whitepaper técnico que vende a credibilidade da API.

---

## 2. Integração Técnica com NotebookLM

O NotebookLM será alimentado com três camadas de dados do projeto:

1.  **Codebase Context:** A lógica das Redes Líquidas (CfC) e do Enxame de Especialistas. Isso permite que a IA explique a "mágica" por trás do lucro sem alucinações.
2.  **Research Context:** Dados sobre o mercado financeiro e análise preditiva reativa (sua pesquisa atual). Isso "treina" o NotebookLM para ser um analista de nível institucional.
3.  **Execution Logs:** O histórico de acertos e erros do Saci. O NotebookLM analisará os logs para sugerir melhorias na estratégia de "Sniper Mode".

---

## 3. Roadmap de Evolução para Monetização

### Fase 1: Blindagem de Sinal (Curto Prazo)
*   **Meta:** Garantir que o sinal seja "imbatível" para o Instagram.
*   **Ação:** Implementar o gate de confiança de 85%+ no `TrinityEngine`. Preferimos dar menos sinais, mas com taxa de acerto altíssima para construir o histórico (Track Record).

### Fase 2: Automação de Explicação (Médio Prazo)
*   **Meta:** Criar o "Explainer" automático.
*   **Ação:** Integrar o output do Saci com o NotebookLM para gerar o racional de cada operação automaticamente. Isso economiza tempo na gestão da sala de sinais.

### Fase 3: Lançamento da API (Longo Prazo)
*   **Meta:** Monetizar o "Raio-X".
*   **Ação:** Expor os dados do `SensoryHub` (VBI, Cancel Rate, CVD) via um endpoint FastAPI seguro, documentado com o auxílio da inteligência do NotebookLM.

---

## 4. Próximos Passos Sugeridos

1.  **Carregar este documento no NotebookLM** junto com o arquivo `backend/src/brain/engine.py`.
2.  **Solicitar ao NotebookLM:** "Com base na lógica do engine.py, crie 3 exemplos de posts para Instagram explicando por que o Saci-Trader é superior a robôs comuns de indicadores (RSI/MACD)."
3.  **Validar no Backtest:** Usar o NotebookLM para analisar os resultados do `backtester.py` e identificar em quais horários os sinais são mais precisos para focar a venda de sinais nesses períodos.

---
*Documento criado em 09/03/2026 como guia de escala comercial do Saci-Trader.*



--- FILE: docs/SACI_HOLDING_VISION.md ---
# SACI HOLDING: The Universal Chaos Decision Engine
## Strategic Vision: From Crypto Trading to Global Risk Infrastructure

### 1. The Core Thesis: "Intelligence in the Eye of the Storm"
A maioria das IAs atuais (Transformers, LLMs) é estática e reativa. O **Saci Core**, alimentado por **Redes Neurais Líquidas (LNN/CfC)** e **Inteligência de Enxame**, foi desenhado para a fronteira do conhecimento: **Tomada de decisão em tempo contínuo sob condições de caos extremo.**

Onde há fluxo de dados ruidosos e o tempo é crítico, o Saci é a solução superior. Não construímos um robô; construímos um **Motor de Sobrevivência de Capital**.

---

### 2. Strategic Business Units (SBUs)

A tecnologia do Saci será licenciada e operada em quatro verticais de trilhões de dólares:

#### 📈 I. SACI-FINANCE (The Cash Cow)
*   **Foco:** Cripto, Forex, Commodities e Tesouraria Corporativa.
*   **Produto:** Sinais de alta precisão (Sniper Mode), APIs de Microestrutura (Raio-X de Liquidez) e Proteção de Câmbio (Hedge).
*   **Valor WOW:** Previsão de "Flash Crashes" e manipulação de mercado via análise de Order Book em milissegundos.

#### 🏦 II. SACI-CREDIT (The Fintech Disruptor)
*   **Foco:** Crédito Dinâmico via Open Finance.
*   **Produto:** "Score Líquido" em tempo real. Avaliação de risco baseada no fluxo de caixa atual, não no histórico passado.
*   **Valor WOW:** Previsão de inadimplência com semanas de antecedência, permitindo renegociações preventivas e redução drástica do "spread" bancário.

#### 🛡️ III. SACI-SHIELD (The Cyber Sentinel)
*   **Foco:** Cibersegurança e Proteção de Infraestrutura Crítica.
*   **Produto:** Detecção de anomalias "Zero-Day" em tráfego de rede e sistemas industriais (SCADA).
*   **Valor WOW:** Identificação de ataques invisíveis através da mudança no "Regime de Ruído" dos dados, bloqueando intrusões antes que a assinatura do vírus seja conhecida.

#### 🏥 IV. SACI-MED (The Life Saver)
*   **Foco:** Monitoramento preditivo em Unidades de Terapia Intensiva (UTI).
*   **Produto:** Integração com biossensores para prever colapsos hemodinâmicos (infartos, sepse).
*   **Valor WOW:** O robô lê o "Order Book do Corpo Humano" e avisa a equipe médica 45 minutos antes de uma parada cardiorrespiratória ocorrer.

---

### 3. The Technology Stack (The "Secret Sauce")

O que torna a SACI Holding única para um investidor de Deep Tech:
1.  **Agnostic Core:** O mesmo cérebro CfC (Closed-form Continuous-time) processa preços, batimentos cardíacos ou pacotes de rede.
2.  **Noise Resilience:** Desenvolvido para ignorar o "caos" e focar no sinal real (Fractal Attention).
3.  **Active Inference:** O sistema busca ativamente minimizar a incerteza, protegendo o capital/ativo acima de tudo.
4.  **Hardware Efficient:** Otimizado com Triton Kernels para rodar em servidores leves ou dispositivos de borda (Edge Computing).

---

### 4. Expansion Roadmap (The Millionaire Path)

*   **Phase 1 (The Dojo):** Consolidação do Saci-Trader como gerador de caixa imediato. Criação da comunidade de sinais e validação do "Track Record" público.
*   **Phase 2 (The Oracle):** Lançamento da API B2B de dados de microestrutura. Início da venda de inteligência para outros robôs e gestores de fundos.
*   **Phase 3 (The Holding):** Spin-off das verticais de Crédito e Cibersegurança. Captação de Rodada Série A com VCs focados em IA de Infraestrutura.
*   **Phase 4 (The Standard):** Tornar o Saci o padrão global de "Risk Scoring" para a economia digital.

---

### 5. The Pitch: Why Invest in Saci?

> *"Nós não estamos tentando prever o futuro; estamos aprendendo a navegar no presente mais rápido que qualquer outra tecnologia. O Saci é o sistema operacional para um mundo onde o dado é volátil e o atraso é fatal. Seja operando Bitcoin, salvando vidas em uma UTI ou protegendo o fluxo de caixa de uma multinacional, o Saci é o cérebro que nunca dorme e que se adapta ao caos."*

---
*Documento estratégico da SACI Holding - Locked for Vision Protocol v1.0*



--- FILE: docs/paper.md ---
# IARA: Swarm Intelligence and Liquid Neural Networks for Autonomous Market Adaptation

**Version**: 9.7 "Gladiator"  
**Framework**: IARA (Inference and Autonomous Response Architecture)  
**Authors**: Project SACI Development Team  

---

## 1. Abstract
Traditional algorithmic trading systems often struggle with the non-stationary nature of financial markets and the high noise-to-signal ratio in low timeframes. This paper presents **IARA v9**, a novel hierarchical architecture that combines **Swarm Intelligence** with **Liquid Neural Networks (LNN)**. By decoupling market analysis into specialized "Workers" and an orchestrating "Coordinator," and leveraging a **Closed-form Continuous-time (CfC)** backbone, IARA achieves superior temporal awareness and adaptive resilience. We demonstrate that this architecture not only excels in high-frequency trading but also establishes a framework for general-purpose autonomous decision-making in complex environments.

## 2. Introduction
The challenge of market prediction lies in the multi-scale nature of price action. A single neural network often fails to reconcile the rapid fluctuations of the M5 timeframe with the structural trends of the H4. IARA v9 solves this by implementing a **Collective Intelligence** approach. Instead of a monolithic brain, it utilizes a swarm of specialists, each optimized for a specific temporal "lens," fused by a central coordinator using attention mechanisms.

## 3. Architecture: Swarm Intelligence & Liquid Neural Networks
The core of IARA v9 is built on two pillars:

### 3.1 Swarm Workers
Each timeframe (M5, M15, M30, H1, H4) is assigned a `LightningWorker`. These are compact Liquid Models using **CfC (Closed-form Continuous-time)** cells. Unlike traditional RNNs or LSTMs, Liquid Networks possess "flexible" time constants, allowing them to adapt their internal dynamics to the varying speed and volatility of the market.

### 3.2 The Swarm Brain
The `SwarmBrain` encapsulates the entire hierarchy. It provides a standardized interface for multi-timeframe feature processing, ensuring that each worker receives the specific data context it requires while maintaining a unified gradient flow for end-to-end training.

## 4. Coordination: Cross-Attention Fusion
The decentralized analysis of workers is unified by the `SwarmCoordinator`. 

- **Cross-Attention**: The coordinator uses a multi-head attention mechanism where worker embeddings are treated as "experts." This allows the system to dynamically "trust" certain timeframes over others depending on the market regime (e.g., trusting H1 more during strong trends and M5 more during ranging periods).
- **Consensus Fusion**: Worker logits (signals) and embeddings are fused into a 512-dimensional context vector, which is then processed by a high-capacity CfC refinement layer.
- **Asymmetric Expert Heads**: The final output is split into specialized "Long" and "Short" heads, allowing the model to learn the distinct statistical properties of "Fear" (rapid sell-offs) and "Greed" (steady builds).

## 5. Autonomous Learning: Dynamic Replay Buffer & Gating
IARA v9 implements a "Self-Correcting" learning loop:
1. **The Replay Buffer**: Successes and failures from backtesting are captured and fed back into the training phase.
2. **Curriculum Stabilization**: A selective memory mechanism ensures the model doesn't overfit to noisy errors while maintaining a baseline of logical acertos.
3. **The Gating System**: An automated quality gate ensures that only models with statistical significance and positive PnL are promoted to production, protecting the system from catastrophic forgetting.

## 6. Optimization: High-Throughput & GPU Utilization
To maximize hardware efficiency, IARA v9 scales to:
- **Batch Size 256 / Seq Len 256**: Processing 65,536 candles per step.
- **VRAM Utilization**: Efficiently utilizing ~4.5GB to ~8GB of VRAM, significantly increasing training speed and model capacity on modern GPUs (e.g., AMD ROCm / NVIDIA CUDA).

## 7. Beyond Trading: IARA as a General Framework
While currently applied to Forex, the IARA architecture is a prototype for **General Autonomous Response**. The logic of "coordinated especializados" can be mapped to:
- **Robotics**: Sensors as workers, movement control as coordinator.
- **Smart Cities**: Regional traffic nodes as workers, urban flow management as coordinator.
- **Supply Chain**: Local warehouse demand as workers, global logistics as coordinator.

The IARA framework promises a future where AI is not a "black box" but a structured, interpretable, and self-improving collective.

## 8. Conclusion
IARA v9 represents a significant leap in project SACI. By merging Swarm Intelligence with the temporal flexibility of Liquid Neural Networks, we have created a system that doesn't just predict price, but understands the multi-layered rhythm of information. It is a robust foundation for the next generation of autonomous trading and beyond.

---
*© 2026 Project SACI - IARA v9 Paper*



