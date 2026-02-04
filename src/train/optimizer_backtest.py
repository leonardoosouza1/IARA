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
