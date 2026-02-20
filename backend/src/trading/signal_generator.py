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

from src.brain.models.liquid_v9 import SwarmBrain, CoordinatorConfig
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
        input_dim = self._detect_input_dim(checkpoint)
        logger.info(f"  Detected input_dim={input_dim}")

        worker_cfg = WorkerConfig(
            input_dim=input_dim,
            hidden_dim=worker_hidden,
            dropout=0.1,
        )
        coord_cfg = CoordinatorConfig(
            num_workers=5,
            worker_dim=worker_hidden,
            hidden_dim=coord_hidden,
            dropout=0.3,
        )

        self.model = SwarmBrain(worker_cfg, coord_cfg).to(self.device)
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

            if len(df_tf) < seq_len:
                logger.warning(
                    f"{tf_key}: only {len(df_tf)} candles (need {seq_len}), padding"
                )
                # Zero-pad on the left
                pad_len = seq_len - len(df_tf)
                features = compute_features_from_df(df_tf, time_col="time")
                padded = np.zeros((seq_len, features.shape[1]), dtype=np.float32)
                padded[pad_len:] = features[-seq_len:]
                tensor = torch.tensor(padded, dtype=torch.float32)
            else:
                features = compute_features_from_df(df_tf, time_col="time")
                tensor = torch.tensor(
                    features[-seq_len:], dtype=torch.float32
                )

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
                    logits, worker_logits, long_score, short_score = self.model(
                        x_dict
                    )

            # Coordinator probability
            coord_prob = torch.sigmoid(logits).item()

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
