import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Tuple

import torch
import torch.nn as nn
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from src.database.models import Trade, Signal, ModelVersion
from src.brain.models.liquid_v9 import SwarmBrain, CoordinatorConfig
from src.brain.models.workers import WorkerConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("ContinuousLearner")

class ContinuousLearner:
    def __init__(self, checkpoint_path: str = "logs/models/best_model_v9.pth", device: str = "cpu"):
        self.checkpoint_path = checkpoint_path
        self.device = torch.device(device)
        
        # Setup synchronous DB Engine for the learning loop
        db_uri = os.getenv("DB_URI", "sqlite:///./saci_core.db")
        if db_uri.startswith("postgresql+asyncpg://"):
             db_uri = db_uri.replace("postgresql+asyncpg://", "postgresql://")
        elif db_uri.startswith("postgresql://") and "+asyncpg" not in db_uri:
             pass # Use psycopg2 natively
             
        sync_engine = create_engine(db_uri)
        SyncSessionLocal = sessionmaker(bind=sync_engine)
        self.db: Session = SyncSessionLocal()
        
        # We dummy-initialize architecture sizes matching v9.7 Gladiator defaults
        # In a robust system, these should be loaded from the checkpoint metadata or args
        self.worker_hidden = 128
        self.coord_hidden = 512
        self.seq_len = 128
        
        self.model = self._load_model()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-5)
        self.criterion = nn.CrossEntropyLoss()

    def _load_model(self) -> SwarmBrain:
        worker_cfg = WorkerConfig(input_dim=12, hidden_dim=self.worker_hidden)
        coord_cfg = CoordinatorConfig(num_workers=5, worker_dim=self.worker_hidden, hidden_dim=self.coord_hidden)
        
        model = SwarmBrain(worker_cfg, coord_cfg).to(self.device)
        
        # Load weights if available
        if os.path.exists(self.checkpoint_path):
            state = torch.load(self.checkpoint_path, map_location=self.device)
            # handle 'model_state_dict' or direct
            if 'model_state_dict' in state:
                model.load_state_dict(state['model_state_dict'])
            else:
                model.load_state_dict(state)
            logger.info(f"Loaded existing model from {self.checkpoint_path}")
        else:
            logger.warning("Checkpoint not found. Using initialized weights.")
            
        return model

    def build_dataset(self) -> Tuple[torch.Tensor, torch.Tensor, set]:
        """
        Queries DB for Trades that have Unrealized PnL or Realized PnL evaluated.
        Finds corresponding Signal snapshot. Generates features (X) and target (y).
        Returns X, y, and the set of symbols processed.
        """
        # Look back 24 hours
        cutoff = datetime.utcnow() - timedelta(days=1)
        trades = self.db.query(Trade).filter(Trade.timestamp >= cutoff).all()
        
        if not trades:
            return None, None, set()

        X = []
        Y = []
        symbols_seen = set()
        
        for trade in trades:
            # Pnl check. If pnl is None, assume we evaluate by current market_price if provided or skip.
            # Setup ideal targets logic
            pnl = trade.pnl if trade.pnl is not None else 0.0
            
            # Find closest signal just before trade execution
            signal = self.db.query(Signal).filter(
                Signal.symbol == trade.symbol,
                Signal.timestamp <= trade.timestamp
            ).order_by(Signal.timestamp.desc()).first()
            
            if not signal or not signal.features_snapshot:
                continue
                
            features = signal.features_snapshot
            symbols_seen.add(trade.symbol)
            
            # RL Logic: If trade made money, reinforce it. If lost money, penalize it by selecting hold or opposite.
            original_action = signal.signal_value # 1 (Buy), -1 (Sell), 0 (Wait)
            ideal_action = original_action
            
            if pnl < 0:
                # If we lost money, the ideal action was probably to hold (0)
                ideal_action = 0
            
            # Wait, the model output logic wants classes 0 (Wait), 1 (Long), 2 (Short)
            # Let's map -1 -> 2
            target_scalar = 0
            if ideal_action == 1:
                target_scalar = 1
            elif ideal_action == -1:
                target_scalar = 2
                
            # Dummy reconstructing sequence Tensor from the features_snapshot for MVP
            # Realistacally features_snapshot should store the raw Sequence or we have to query the historical ticks
            # Here we simulate the [batch, num_workers, seq_len, features] shape just to prove the loop
            dummy_sequence = torch.zeros(1, 3, self.seq_len, 27) 
            # In a real impl, you'd use self.db to query market_candles to rebuild the sequence precisely.
            
            X.append(dummy_sequence)
            Y.append(target_scalar)
            
        if not X:
            return None, None, set()
            
        x_tensor = torch.cat(X, dim=0).to(self.device)
        y_tensor = torch.tensor(Y, dtype=torch.long).to(self.device)
        
        return x_tensor, y_tensor, symbols_seen

    def train_and_evaluate(self):
        logger.info("Building dataset from Trade Journal...")
        x_tensor, y_tensor, symbols = self.build_dataset()
        
        if x_tensor is None or len(x_tensor) == 0:
            logger.info("Not enough trade data for learning cycle yet.")
            return

        logger.info(f"Training on {len(x_tensor)} historical decisions...")
        self.model.train()
        
        self.optimizer.zero_grad()
        # Mock Context (LFM)
        ctx = torch.zeros(len(x_tensor), self.coord_hidden).to(self.device)
        
        logits, _, _ = self.model(x_tensor, ctx)
        loss = self.criterion(logits, y_tensor)
        
        loss.backward()
        self.optimizer.step()
        
        final_loss = loss.item()
        logger.info(f"Learning Cycle Complete. Loss: {final_loss:.4f}")
        
        # Save Model
        torch.save({'model_state_dict': self.model.state_dict()}, self.checkpoint_path)
        logger.info(f"Model saved to {self.checkpoint_path}")
        
        self.update_registry(final_loss, symbols)
        
    def update_registry(self, loss: float, symbols: set):
        """Updates the Model Registry in the database"""
        version_tag = "v9.7-Live"
        registry_entry = self.db.query(ModelVersion).filter(ModelVersion.tag == version_tag).first()
        
        if not registry_entry:
            registry_entry = ModelVersion(
                tag=version_tag,
                path=self.checkpoint_path,
                is_active=True,
                metrics={}
            )
            self.db.add(registry_entry)
            self.db.commit()
            self.db.refresh(registry_entry)
            
        # Update Metrics
        current_metrics = registry_entry.metrics or {}
        
        for symbol in symbols:
            current_metrics[symbol] = {
                "loss": round(loss, 4),
                "last_trained": datetime.utcnow().isoformat()
            }
            
        registry_entry.metrics = current_metrics
        
        # Categorize
        # If the model has seen more than 1 symbol in its lifetime, it is GENERALIST
        if len(current_metrics.keys()) > 1:
            registry_entry.category_type = "GENERALIST"
        else:
            registry_entry.category_type = "SPECIALIST"
            
        self.db.commit()
        logger.info(f"Model Registry Updated: {registry_entry.category_type} | {list(current_metrics.keys())}")

if __name__ == "__main__":
    learner = ContinuousLearner()
    learner.train_and_evaluate()
