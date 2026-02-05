
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

from src.data.feature_factory import process_hierarchical
from src.train.trainer_liquid import TrainerConfig

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
        seq_len = self.model.config.seq_len if hasattr(self.model, 'config') else 256
        
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
        
        self.current_step += 1
        return brain_state

