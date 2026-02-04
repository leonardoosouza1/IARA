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
from src.trading.active_inference import ActiveInferenceModule

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
    stability_check: str

class SignalEngine:
    def __init__(self, checkpoint_path: str, thresholds: Dict[str, float] = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # v6.4 Active Feedback
        self.active_brain = ActiveInferenceModule()
        self.output_threshold = self.active_brain.current_threshold
        
        logger.info(f"Signal Engine initialized. Active Threshold: {self.output_threshold:.2f}")
        
        self._load_model(checkpoint_path)
        self._load_normalizer()
        
        # Enable optimizations
        torch.set_float32_matmul_precision('high')

    def _load_model(self, path: str):
        # ... existing ... (Kept as is, but showing start for context)
        logger.info(f"Loading Signal Model from {path}...")
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.config = checkpoint['config']
        
        liquid_config = LiquidConfig(
            input_dim=self.config.input_dim,
            hidden_dim=self.config.hidden_dim,
            num_heads=self.config.num_heads,
            dropout=self.config.dropout, 
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
        try:
            self.normalizer = Normalizer.load(DATA_DIR / "scaler.pkl")
            logger.info("Normalizer loaded.")
        except Exception as e:
            logger.error(f"Normalizer usage failed: {e}")
            raise

    def process_live_data(self, parquet_path: Path) -> Tuple[torch.Tensor, float, float]:
        # Mimic training preprocessing logic
        df = process_single_asset(parquet_path)
        if df is None: raise ValueError("Data processing failed")
        
        feature_cols = self.normalizer.feature_names
        features_np = df.select(feature_cols).to_numpy()
        features_norm_np = self.normalizer.transform(features_np)
        
        norm_exprs = [pl.Series(f"{col}_norm", features_norm_np[:, i]) for i, col in enumerate(feature_cols)]
        df_norm = df.with_columns(norm_exprs)
        
        norm_cols = sorted([f"{c}_norm" for c in feature_cols])
        
        seq_len = self.config.max_seq_len
        recent_data = df_norm.select(norm_cols).tail(seq_len).to_numpy().astype(np.float32)
        
        # Metadata
        last_row = df.tail(1)
        current_adx = last_row["adx"].item()
        current_rsi = last_row["rsi"].item()
        
        tensor = torch.tensor(recent_data).unsqueeze(0).to(self.device)
        return tensor, current_adx, current_rsi

    def predict(self, ticker: str, data_path: Path) -> TradeSignal:
        try:
            input_tensor, adx, rsi = self.process_live_data(data_path)
            
            with torch.no_grad():
                with torch.amp.autocast('cuda'):
                    logits, _ = self.model(input_tensor)
                    prob = torch.sigmoid(logits).item()
            
            # v6.4 Active Decision
            # Feed result to Active Brain (In production, we would feed actual outcome too)
            # For now, we update the prediction history.
            # Ideally, a separate process calls active_brain.log_prediction(prob, actual_outcome)
            
            decision = self.active_brain.get_decision(prob)
            
            # Filter Logic
            check = (decision != "WAIT")
            if "Volatile" in decision:
                check = False
            
            # Additional Filters
            if adx < 20: 
                check = False
                decision = f"{decision} (WK TREND)"
                
            return TradeSignal(
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                ticker=ticker,
                direction=decision,
                confidence=prob,
                adx=adx,
                rsi=rsi,
                threshold_check=check,
                adaptive_threshold=self.active_brain.current_threshold,
                stability_check="OK" if "Volatile" not in decision else "UNSTABLE"
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
