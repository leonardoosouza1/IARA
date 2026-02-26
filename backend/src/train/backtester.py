import os
import torch
import torch.nn.functional as F
from datetime import datetime
import json
import logging
from typing import Dict, Any

from src.brain.models.swarm import SwarmBrain, WorkerConfig, CoordinatorConfig
from src.database.core import get_db
from src.database.models import Trade

logger = logging.getLogger("api.backtester")

class Backtester:
    """
    Evaluates the historical performance of a SwarmBrain model
    applying varying degrees of Sniper Mode precision.
    """
    def __init__(self, model_path: str = "logs/models/active_model_v10.pth", device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        
        # Load Model Dimensions dynamically
        self.input_dim = 26
        self.worker_hidden = 64
        self.coord_hidden = 256
        self.seq_len = 128
        
        norm_path = "data/cache/BTCUSD/norm_params.pt" # Defaulting for now
        if os.path.exists(norm_path):
            try:
                meta = torch.load(norm_path, map_location='cpu', weights_only=True)
                if 'feature_cols' in meta:
                    self.input_dim = len(meta['feature_cols'])
            except Exception:
                pass
                
        self.model = self._load_model()
        
    def _load_model(self) -> SwarmBrain:
        worker_cfg = WorkerConfig(input_dim=self.input_dim, hidden_dim=self.worker_hidden)
        coord_cfg = CoordinatorConfig(num_workers=5, worker_dim=self.worker_hidden, hidden_dim=self.coord_hidden)
        model = SwarmBrain(worker_cfg, coord_cfg).to(self.device)
        
        if os.path.exists(self.model_path):
            state = torch.load(self.model_path, map_location=self.device)
            state_dict = state.get('model_state_dict', state)
            model.load_state_dict(state_dict, strict=False)
            logger.info(f"Loaded model from {self.model_path}")
        else:
            logger.warning(f"Model path {self.model_path} not found! Backtest will use random weights.")
            
        model.eval()
        return model

    def _simulate_timeframes(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Same logic as continuous_learner.py to build the timeframe dictionary."""
        seq_len = x.shape[1]
        tfs = {
            "M5": x,
            "M15": x[:, max(0, seq_len-32):, :] if seq_len > 32 else x,
            "M30": x[:, max(0, seq_len-16):, :] if seq_len > 16 else x,
            "H1": x[:, max(0, seq_len-8):, :] if seq_len > 8 else x,
            "H4": x[:, max(0, seq_len-2):, :] if seq_len > 2 else x,
        }
        return tfs

    def evaluate_shards(self, dataset_name: str, sniper_threshold: float = 0.50, callback=None) -> Dict[str, Any]:
        """
        Runs the model over the validation shards, enforcing the sniper_threshold.
        If the network's confidence is lower than sniper_threshold, the trade is skipped (classified as WAIT).
        """
        val_shard_dir = f"data/cache/{dataset_name}/val_shards"
        
        if not os.path.exists(val_shard_dir):
            return {"error": f"Validation shards not found for {dataset_name}"}
            
        shards = sorted([f for f in os.listdir(val_shard_dir) if f.endswith(".pt")])
        if not shards:
            return {"error": f"No shards found in {val_shard_dir}"}
            
        metrics = {
            "total_samples": 0,
            "trades_executed": 0,
            "wins": 0,
            "losses": 0,
            "sniper_threshold": sniper_threshold,
            "false_positives_avoided": 0 # Trades the network wanted to take but Sniper Mode blocked
        }
        
        batch_size = 512
        
        if callback: callback(f"🚀 Starting Backtest on {len(shards)} shards with Sniper mode: {sniper_threshold:.2f}...")

        with torch.no_grad():
            for s_file in shards:
                if callback: callback(f"Evaluating {s_file}...")
                s_data = torch.load(os.path.join(val_shard_dir, s_file), map_location='cpu')
                X_cpu, y_cpu = s_data['X'], s_data['y']
                metrics["total_samples"] += len(X_cpu)
                
                for j in range(0, len(X_cpu), batch_size):
                    batch_X = X_cpu[j : j + batch_size].to(self.device)
                    batch_y = y_cpu[j : j + batch_size].long().to(self.device)
                    
                    # Robust Tensor Padding
                    shard_dim = batch_X.shape[-1]
                    if shard_dim > self.input_dim:
                        batch_X = batch_X[:, :, :self.input_dim]
                    elif shard_dim < self.input_dim:
                        padding = torch.zeros(batch_X.shape[0], batch_X.shape[1], self.input_dim - shard_dim, dtype=batch_X.dtype, device=batch_X.device)
                        batch_X = torch.cat([batch_X, padding], dim=-1)
                        
                    batch_X = torch.nan_to_num(batch_X)
                    batch_X = torch.clamp(batch_X, min=-20.0, max=20.0)
                    
                    x_dict = self._simulate_timeframes(batch_X)
                    
                    # Forward pass with telemetry
                    logits, _, _, _, _, confidence = self.model(x_dict, compute_telemetry=True)
                    
                    # Logits are [Batch, 3] corresponding to Long(0), Short(1), Wait(2)
                    probs = F.softmax(logits, dim=-1)
                    
                    # Confidence is already the max probability from the forward pass
                    max_probs = confidence.squeeze(-1) if confidence is not None else probs.max(dim=-1)[0]
                    base_predictions = probs.argmax(dim=-1)
                    
                    # 🎯 SNIPER MODE FILTRATION
                    # If prediction is Long or Short (0 or 1), but confidence is below threshold, override to Wait (2)
                    sniper_predictions = base_predictions.clone()
                    is_action = (base_predictions == 0) | (base_predictions == 1)
                    low_confidence = max_probs < sniper_threshold
                    
                    # Block trades that lack confidence
                    blocked_mask = is_action & low_confidence
                    sniper_predictions[blocked_mask] = 2 
                    metrics["false_positives_avoided"] += blocked_mask.sum().item()
                    
                    # Evaluate Results
                    for i in range(len(sniper_predictions)):
                        pred = sniper_predictions[i].item()
                        true = batch_y[i].item()
                        
                        if pred != 2: # Model decided to act
                            metrics["trades_executed"] += 1
                            if pred == true:
                                metrics["wins"] += 1
                            else:
                                metrics["losses"] += 1

        if metrics["trades_executed"] > 0:
            metrics["win_rate"] = round((metrics["wins"] / metrics["trades_executed"]) * 100, 2)
        else:
            metrics["win_rate"] = 0.0

        if callback: 
            callback(f"✅ Backtest Complete!")
            callback(f"🎯 Win Rate: {metrics['win_rate']}% | Trades: {metrics['trades_executed']} | Blocked by Sniper: {metrics['false_positives_avoided']}")
            
        return metrics

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="BTCUSD")
    parser.add_argument("--threshold", type=float, default=0.75)
    args = parser.parse_args()
    
    bt = Backtester()
    def console_cb(msg): print(msg)
    res = bt.evaluate_shards(args.dataset, args.threshold, callback=console_cb)
    print(json.dumps(res, indent=2))
