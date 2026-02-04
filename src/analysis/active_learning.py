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
