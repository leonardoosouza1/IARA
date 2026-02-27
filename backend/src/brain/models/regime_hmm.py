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
