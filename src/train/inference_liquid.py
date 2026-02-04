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
    seq_len: int = 64
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load test data for a specific ticker."""
    file_path = DATA_PROCESSED_DIR / f"{ticker}.parquet"
    if not file_path.exists():
        raise FileNotFoundError(f"Ticker {ticker} not found at {file_path}")
    
    df = pl.read_parquet(file_path)
    
    # Dynamically detect all *_norm columns (same as trainer)
    feature_cols = sorted([c for c in df.columns if c.endswith("_norm")])
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




@torch.no_grad()
def evaluate(
    model: IARA,
    features: np.ndarray,
    target_cls: np.ndarray,
    device: torch.device,
    seq_len: int = 64
) -> dict:
    """Evaluate model on test data."""
    model.eval()
    
    n_samples = len(features) - seq_len
    if n_samples <= 0:
        raise ValueError(f"Not enough samples. Need at least {seq_len + 1}")
    
    predictions = []
    
    for i in range(n_samples):
        # Create sequence
        # We start at i, go to i+seq_len
        # x shape: [1, seq_len, features]
        x = torch.tensor(features[i:i + seq_len], dtype=torch.float32).unsqueeze(0).to(device)
        
        # Forward pass
        logits, _ = model(x)
        logits = logits.squeeze(-1)
        
        # Probabilities
        probs = torch.sigmoid(logits)
        pred_prob = probs.cpu().item() # Last step prob
        
        # Predictions
        pred_cls = 1.0 if pred_prob > 0.5 else 0.0
        confidence = abs(pred_prob - 0.5) * 2  # 0 to 1
        
        # True value
        # The target for the sequence ending at `i+seq_len` is at index `i+seq_len-1`
        # Because in dataset we setup: x[0..64] -> predict target[63] (which is trend T+1..T+5)
        # Wait, trainer uses: raw_target = targets[end_row - 1]
        # end_row is start + seq_len. So target index is start + seq_len - 1.
        # Here `start` corresponds to `i`.
        true_idx = i + seq_len - 1
        true_cls_val = target_cls[true_idx]
        
        predictions.append({
            "true_cls": true_cls_val,
            "pred_prob": pred_prob,
            "pred_cls": pred_cls,
            "confidence": confidence
        })
    
    # Calculate Metrics
    true_classes = [p["true_cls"] for p in predictions]
    pred_classes = [p["pred_cls"] for p in predictions]
    pred_probs = [p["pred_prob"] for p in predictions]
    
    tp = sum(1 for t, p in zip(true_classes, pred_classes) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(true_classes, pred_classes) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(true_classes, pred_classes) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(true_classes, pred_classes) if t == 1 and p == 0)
    
    accuracy = (tp + tn) / len(predictions) * 100 if predictions else 0
    precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    try:
        auc = roc_auc_score(true_classes, pred_probs)
    except:
        auc = 0.5
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
        "confusion_matrix": {"TP": tp, "TN": tn, "FP": fp, "FN": fn},
        "n_samples": n_samples,
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
    features, target_reg, target_cls = load_test_data(args.ticker, normalizer, args.limit, args.seq_len)
    logger.info(f"Loaded {len(features)} samples, {features.shape[1]} features")
    
    # Evaluate
    logger.info("Evaluating...")
    results = evaluate(model, features, target_cls, device, args.seq_len)
    
    # Print results
    print("\n" + "=" * 60)
    print(f"I.A.R.A. Evaluation Results - {args.ticker}")
    print("=" * 60)
    print(f"Samples Evaluated: {results['n_samples']}")
    print(f"Accuracy:  {results['accuracy']:.2f}%")
    print(f"AUC:       {results['auc']:.4f}")
    print(f"Precision: {results['precision']:.2f}%")
    print(f"Recall:    {results['recall']:.2f}%")
    print(f"F1 Score:  {results['f1']:.4f}")
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
