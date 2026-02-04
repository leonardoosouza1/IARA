"""
Inference script for I.A.R.A. Liquid Neural Network
====================================================
Loads trained model and evaluates on test data.
"""

import argparse
import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import polars as pl
import torch

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
    
    model_config = checkpoint.get("model_config")
    if model_config is None:
        # Fallback to default config
        model_config = LiquidConfig()
    
    model = IARA(model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    logger.info(f"Loaded model from {checkpoint_path}")
    logger.info(f"Epoch: {checkpoint.get('epoch', 'N/A')}, Best Loss: {checkpoint.get('best_loss', 'N/A'):.4f}")
    
    return model, checkpoint


def load_test_data(
    ticker: str = "EURUSD",
    normalizer: Optional[Normalizer] = None,
    limit: int = 500
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load test data for a specific ticker."""
    if normalizer is None:
        normalizer = Normalizer.load(SCALER_PATH)
    
    file_path = DATA_PROCESSED_DIR / f"{ticker}.parquet"
    if not file_path.exists():
        raise FileNotFoundError(f"Ticker {ticker} not found at {file_path}")
    
    df = pl.read_parquet(file_path)
    
    # Get normalized feature columns
    feature_cols = [f"{c}_norm" for c in normalizer.state.feature_columns]
    available_cols = [c for c in feature_cols if c in df.columns]
    
    # Take last N samples for testing
    df = df.tail(limit)
    
    features = df.select(available_cols).to_numpy().astype(np.float32)
    target_reg = df["close_log_ret"].to_numpy().astype(np.float32)
    target_cls = df["direction"].to_numpy().astype(np.float32)
    
    return features, target_reg, target_cls


@torch.no_grad()
def evaluate(
    model: IARA,
    features: np.ndarray,
    target_reg: np.ndarray,
    target_cls: np.ndarray,
    device: torch.device,
    seq_len: int = 64
) -> dict:
    """Evaluate model on test data."""
    model.eval()
    
    n_samples = len(features) - seq_len
    if n_samples <= 0:
        raise ValueError(f"Not enough samples. Need at least {seq_len + 1}")
    
    all_pred_reg = []
    all_pred_cls = []
    all_true_reg = []
    all_true_cls = []
    
    for i in range(n_samples):
        # Create sequence
        x = torch.tensor(features[i:i + seq_len], dtype=torch.float32).unsqueeze(0).to(device)
        
        # Forward pass
        pred_reg, pred_cls, _ = model(x)
        
        # Get last prediction (next step)
        # Undescale target (Trainer scaled by 1000.0)
        pred_reg_val = pred_reg[0, -1].cpu().item() / 1000.0
        
        pred_cls_val = torch.sigmoid(pred_cls[0, -1]).cpu().item()
        
        # True values (shifted by 1)
        true_reg_val = target_reg[i + seq_len]
        true_cls_val = target_cls[i + seq_len]
        
        all_pred_reg.append(pred_reg_val)
        all_pred_cls.append(pred_cls_val)
        all_true_reg.append(true_reg_val)
        all_true_cls.append(true_cls_val)
    
    all_pred_reg = np.array(all_pred_reg)
    all_pred_cls = np.array(all_pred_cls)
    all_true_reg = np.array(all_true_reg)
    all_true_cls = np.array(all_true_cls)
    
    # Compute metrics
    mse = np.mean((all_pred_reg - all_true_reg) ** 2)
    mae = np.mean(np.abs(all_pred_reg - all_true_reg))
    
    # Directional accuracy
    pred_direction = (all_pred_cls > 0.5).astype(float)
    dir_accuracy = np.mean(pred_direction == all_true_cls) * 100
    
    # Sign accuracy (does prediction sign match true sign)
    pred_sign = np.sign(all_pred_reg)
    true_sign = np.sign(all_true_reg)
    sign_accuracy = np.mean(pred_sign == true_sign) * 100
    
    return {
        "mse": mse,
        "mae": mae,
        "dir_accuracy": dir_accuracy,
        "sign_accuracy": sign_accuracy,
        "n_samples": n_samples,
        "pred_reg": all_pred_reg,
        "true_reg": all_true_reg,
        "pred_cls": all_pred_cls,
        "true_cls": all_true_cls
    }


def print_sample_predictions(results: dict, n_samples: int = 10):
    """Print sample predictions."""
    print("\n" + "=" * 60)
    print("Sample Predictions (Last 10)")
    print("=" * 60)
    print(f"{'True LogRet':>12} | {'Pred LogRet':>12} | {'True Dir':>8} | {'Pred Dir':>8}")
    print("-" * 60)
    
    for i in range(-n_samples, 0):
        true_reg = results["true_reg"][i]
        pred_reg = results["pred_reg"][i]
        true_cls = "↑ UP" if results["true_cls"][i] > 0.5 else "↓ DOWN"
        pred_cls = "↑ UP" if results["pred_cls"][i] > 0.5 else "↓ DOWN"
        
        match = "✓" if (results["true_cls"][i] > 0.5) == (results["pred_cls"][i] > 0.5) else "✗"
        
        print(f"{true_reg:>12.6f} | {pred_reg:>12.6f} | {true_cls:>8} | {pred_cls:>8} {match}")


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
    features, target_reg, target_cls = load_test_data(args.ticker, normalizer, args.limit)
    logger.info(f"Loaded {len(features)} samples, {features.shape[1]} features")
    
    # Evaluate
    logger.info("Evaluating...")
    results = evaluate(model, features, target_reg, target_cls, device, args.seq_len)
    
    # Print results
    print("\n" + "=" * 60)
    print(f"I.A.R.A. Evaluation Results - {args.ticker}")
    print("=" * 60)
    print(f"Samples Evaluated: {results['n_samples']}")
    print(f"MSE (Log-Return):  {results['mse']:.6f}")
    print(f"MAE (Log-Return):  {results['mae']:.6f}")
    print(f"Direction Accuracy: {results['dir_accuracy']:.2f}%")
    print(f"Sign Accuracy:      {results['sign_accuracy']:.2f}%")
    
    print_sample_predictions(results)
    
    # Trading interpretation
    print("\n" + "=" * 60)
    print("Trading Interpretation")
    print("=" * 60)
    if results["dir_accuracy"] > 55:
        print("✅ Model shows predictive edge! (Dir Acc > 55%)")
    elif results["dir_accuracy"] > 50:
        print("⚠️  Marginal edge. May need more training or tuning.")
    else:
        print("❌ No evident edge. Consider more epochs or architecture changes.")


if __name__ == "__main__":
    main()
