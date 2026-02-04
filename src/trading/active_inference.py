from dataclasses import dataclass, field
from typing import List, Deque, Dict
import numpy as np
from collections import deque
import logging

logger = logging.getLogger(__name__)

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
        if acc < 0.52:
            self.current_threshold = 0.75 # DEFENSIVE
            mode = "DEFENSIVE 🛡️"
        elif acc > 0.58:
            self.current_threshold = 0.55 # AGGRESSIVE ⚔️
            mode = "AGGRESSIVE ⚔️"
        else:
            self.current_threshold = 0.60 # NEUTRAL 😐
            mode = "NEUTRAL 😐"
            
        logger.info(f"Active Calibration: Acc {acc:.1%} -> Threshold {self.current_threshold} ({mode})")
        
    def get_decision(self, prob: float) -> str:
        """
        Returns 'BUY', 'SELL', or 'WAIT' based on Dynamic Threshold and Stability.
        """
        # Stability Check (Volatility of confidence)
        if len(self.confidence_history) >= 5:
            volatility = np.std(list(self.confidence_history))
            if volatility > 0.2:
                return "WAIT (Volatile)"
        
        # Threshold Check
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
