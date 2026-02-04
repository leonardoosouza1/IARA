"""
Signal Generator (Live Engine)
==============================
Bridges the model to real-time data.
- Ingests Raw Candles (M1/M5)
- Consistent Feature Engineering (DataProcessor)
- Fast Inference (AMP + TensorFloat32)
- Rule Engine (Thresholds from Calibration)
"""

import argparse
import logging
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import numpy as np
import polars as pl
import torch
import torch.nn as nn

from src.model.iara_liquid import IARA, LiquidConfig
from src.data.processor import process_single_asset, Normalizer, NormalizerState
from src.analysis.active_learning import ActiveLearner

# Pickle Patch: dataset was saved with processor.py as __main__, so it expects NormalizerState in __main__
import sys
sys.modules['__main__'].NormalizerState = NormalizerState

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Paths
CHECKPOINT_DIR = Path(__file__).parent.parent.parent / "checkpoints"
DATA_DIR = Path(__file__).parent.parent.parent / "data"

@dataclass
class TradeSignal:
    timestamp: str
    ticker: str
    direction: str  # BUY, SELL, WAIT
    confidence: float
    adx: float
    rsi: float
    threshold_check: bool
    adaptive_threshold: float  # Include validation info based on regime

class SignalEngine:
    def __init__(self, checkpoint_path: str, thresholds: Dict[str, float] = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Adaptive Threshold Logic
        self.learner = ActiveLearner()
        self.adaptive_threshold = self.learner.get_adaptive_threshold()
        
        # Fallback to manual if provided, else use adaptive
        manual_thresh = thresholds.get("threshold", 0.60) if thresholds else 0.60
        
        # We prioritize Adaptive unless manual is strictly enforced (could differ in prod logic)
        # For now, let's use Adaptive
        self.output_threshold = self.adaptive_threshold
        logger.info(f"Signal Engine initialized. Adaptive Threshold: {self.output_threshold:.2f}")
        
        self._load_model(checkpoint_path)
        self._load_normalizer()
        
        # Enable optimizations
        torch.set_float32_matmul_precision('high')
        
    def _load_model(self, path: str):
        logger.info(f"Loading Signal Model from {path}...")
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.config = checkpoint['config']
        
        liquid_config = LiquidConfig(
            input_dim=self.config.input_dim,
            hidden_dim=self.config.hidden_dim,
            num_heads=self.config.num_heads,
            dropout=self.config.dropout, # Dropout inactive in eval, but needed for init
            max_seq_len=self.config.max_seq_len
        )
        
        self.model = IARA(liquid_config).to(self.device)
        
        state_dict = checkpoint['model']
        if list(state_dict.keys())[0].startswith('_orig_mod.'):
            state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
            
        self.model.load_state_dict(state_dict)
        self.model.eval()
        logger.info("Model loaded and optimized.")

    def _load_normalizer(self):
        # We need the Normalizer to scale live data exactly like training data
        try:
            self.normalizer = Normalizer.load(DATA_DIR / "scaler.pkl")
            logger.info("Normalizer loaded.")
        except Exception as e:
            logger.error(f"Normalizer usage failed: {e}")
            raise

    def process_live_data(self, parquet_path: Path) -> Tuple[torch.Tensor, float, float]:
        """
        Mimics training preprocessing for a single live file.
        Returns: (Tensor [1, Seq, Feat], Current ADX, Current RSI)
        """
        # 1. Feature Engineering (RSI, ADX, etc.)
        # This function re-calculates indicators from raw OHLC
        # In a real API setting, we would append the new candle to a buffer and update.
        # Here we re-process the file.
        df = process_single_asset(parquet_path)
        
        if df is None:
            raise ValueError("Data processing failed")
            
        # 2. Extract Features
        # Select Normalized Columns - using the EXACT names/order from the fitted normalizer
        feature_cols = self.normalizer.feature_names
        
        # Validate columns exist
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing features in live data: {missing}")
            
        # Extract numpy array for scaler
        features_np = df.select(feature_cols).to_numpy()
        
        # Transform
        features_norm_np = self.normalizer.transform(features_np)
        
        # Create normalized dataframe columns
        norm_exprs = [
            pl.Series(f"{col}_norm", features_norm_np[:, i]) 
            for i, col in enumerate(feature_cols)
        ]
        
        df_norm = df.with_columns(norm_exprs)
        
        # Select only the normalized columns for model input
        norm_cols = [f"{c}_norm" for c in feature_cols]
        # Ensure alphabetical sort if that's what the model expects (MultiAssetDataset logic)
        norm_cols = sorted(norm_cols)
        
        # Verify dimension
        if len(norm_cols) != self.config.input_dim:
            logger.warning(f"Dimension mismatch! Model expects {self.config.input_dim}, got {len(norm_cols)}")
        
        # Get last sequence
        seq_len = self.config.max_seq_len
        if len(df_norm) < seq_len:
            raise ValueError(f"Not enough data: {len(df_norm)} < {seq_len}")
            
        recent_data = df_norm.select(norm_cols).tail(seq_len).to_numpy().astype(np.float32)
        
        # Extract metadata (raw values for reporting) used for filters
        last_row = df.tail(1)
        current_adx = last_row["adx"].item()
        current_rsi = last_row["rsi"].item()
        
        tensor = torch.tensor(recent_data).unsqueeze(0).to(self.device) # [1, Seq, Feat]
        return tensor, current_adx, current_rsi

    def predict(self, ticker: str, data_path: Path) -> TradeSignal:
        try:
            input_tensor, adx, rsi = self.process_live_data(data_path)
            
            with torch.no_grad():
                with torch.amp.autocast('cuda'):
                    logits, _ = self.model(input_tensor)
                    prob = torch.sigmoid(logits).item()
            
            # Logic
            direction = "WAIT"
            check = False
            
            # Calibration Threshold
            if prob > self.output_threshold:
                direction = "BUY"
                check = True
            elif prob < (1 - self.output_threshold):
                direction = "SELL"
                check = True
                
            # Filters (Example: Don't trade if ADX < 20)
            if adx < 20: 
                check = False
                direction = f"{direction} (WK TREND)"
                
            return TradeSignal(
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                ticker=ticker,
                direction=direction,
                confidence=prob,
                adx=adx,
                rsi=rsi,
                threshold_check=check,
                adaptive_threshold=self.adaptive_threshold
            )
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", type=str, required=True, help="Asset Ticker (e.g. EURUSD)")
    parser.add_argument("--path", type=str, required=True, help="Path to raw parquet file")
    args = parser.parse_args()
    
    # Load thresholds (optional)
    thresholds = {"threshold": 0.60} # Default
    try:
        with open("artifacts/calibration.json", "r") as f:
            # calibration.json depends on dynamic_calibrator output structure
            # Let's assume generic load
            pass
    except:
        pass
        
    engine = SignalEngine(
        checkpoint_path=str(CHECKPOINT_DIR / "best_model.pth"),
        thresholds=thresholds
    )
    
    signal = engine.predict(args.ticker, Path(args.path))
    
    if signal:
        # CLI Output
        icon = "🟢" if "BUY" in signal.direction else "🔴" if "SELL" in signal.direction else "⚪"
        print(f"\n{icon} I.A.R.A. v6.1 Signal [{signal.ticker}]")
        print(f"   Time: {signal.timestamp}")
        print(f"   Action: {signal.direction}")
        print(f"   Conf: {signal.confidence:.4f}")
        print(f"   ADX: {signal.adx:.1f} | RSI: {signal.rsi:.1f}")
        print(f"   Execute: {signal.threshold_check}\n")
        
        # Machine Output
        print(json.dumps(signal.__dict__))

if __name__ == "__main__":
    main()
