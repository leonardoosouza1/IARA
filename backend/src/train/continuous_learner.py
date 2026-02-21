import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Tuple, Optional

import torch
import torch.nn as nn
from sqlalchemy import create_engine, desc
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
    def __init__(self, checkpoint_path: str = "logs/models/active_model.pth", device: Optional[str] = None):
        self.checkpoint_path = checkpoint_path
        
        # 🚀 Phase 9: Automatic Hardware Detection (CUDA/ROCm)
        if device:
            self.device = torch.device(device)
        else:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                # Add ROCm check if possible, but PyTorch uses 'cuda' label for both
                logger.info(f"🚀 GPU Acceleration Enabled: Using {torch.cuda.get_device_name(0)}")
            else:
                self.device = torch.device("cpu")
                logger.warning("⚠️ GPU Not Found. Falling back to CPU training.")
        
        self.mode = "finetune"
        
        # Setup synchronous DB Engine for the learning loop
        db_uri = os.getenv("DB_URI", "sqlite:///./saci_core.db")
        if db_uri.startswith("postgresql+asyncpg://"):
             db_uri = db_uri.replace("postgresql+asyncpg://", "postgresql://")
        elif db_uri.startswith("postgresql://") and "+asyncpg" not in db_uri:
             pass 
             
        sync_engine = create_engine(db_uri)
        SyncSessionLocal = sessionmaker(bind=sync_engine)
        self.db: Session = SyncSessionLocal()
        
        # Defaults for architecture
        self.worker_hidden = 64
        self.coord_hidden = 256
        self.seq_len = 128
        
        # Hyperparameters (Gladiator v9.7 Presets)
        self.epochs = 5
        self.batch_size = 256
        self.learning_rate = 1e-4
        
        self.model = self._load_model()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.CrossEntropyLoss().to(self.device)

    def _load_model(self) -> SwarmBrain:
        worker_cfg = WorkerConfig(input_dim=12, hidden_dim=self.worker_hidden)
        coord_cfg = CoordinatorConfig(num_workers=5, worker_dim=self.worker_hidden, hidden_dim=self.coord_hidden)
        
        model = SwarmBrain(worker_cfg, coord_cfg).to(self.device)
        
        # Load weights if available
        if self.checkpoint_path and os.path.exists(self.checkpoint_path):
            try:
                state = torch.load(self.checkpoint_path, map_location=self.device)
                if 'model_state_dict' in state:
                    model.load_state_dict(state['model_state_dict'], strict=False)
                else:
                    model.load_state_dict(state, strict=False)
                logger.info(f"Loaded existing model from {self.checkpoint_path}")
            except Exception as e:
                logger.error(f"Failed to load checkpoint: {e}")
        else:
            msg = "Initializing fresh model with random weights." if not self.checkpoint_path else f"Checkpoint {self.checkpoint_path} not found. Using fresh weights."
            logger.warning(msg)
            
        return model

    def build_dataset(self) -> Tuple[torch.Tensor, torch.Tensor, set]:
        cutoff = datetime.utcnow() - timedelta(days=1)
        trades = self.db.query(Trade).filter(Trade.timestamp >= cutoff).all()
        
        if not trades:
            return None, None, set()

        X = []
        Y = []
        symbols_seen = set()
        
        for trade in trades:
            pnl = trade.pnl if trade.pnl is not None else 0.0
            signal = self.db.query(Signal).filter(
                Signal.symbol == trade.symbol,
                Signal.timestamp <= trade.timestamp
            ).order_by(Signal.timestamp.desc()).first()
            
            if not signal or not signal.features_snapshot:
                continue
                
            symbols_seen.add(trade.symbol)
            original_action = signal.signal_value 
            ideal_action = original_action
            
            if pnl < 0:
                ideal_action = 0 
            
            target_scalar = 0
            if ideal_action == 1:
                target_scalar = 1
            elif ideal_action == -1:
                target_scalar = 2
                
            dummy_sequence = torch.zeros(1, self.seq_len, 12).to(self.device)
            X.append(dummy_sequence)
            Y.append(target_scalar)
            
        if not X:
            return None, None, set()
            
        x_tensor = torch.cat(X, dim=0).to(self.device)
        y_tensor = torch.tensor(Y, dtype=torch.long).to(self.device)
        
        return x_tensor, y_tensor, symbols_seen

    def train_and_evaluate(self, callback=None, stop_check=None):
        if callback: callback(f"🖥️ Device: {self.device} | Building dataset from Trade Journal...")
        x_tensor, y_tensor, symbols = self.build_dataset()
        
        if x_tensor is None or len(x_tensor) == 0:
            if callback: callback("Not enough trade data for learning cycle yet.")
            return

        # Update optimizer
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.learning_rate

        self.model.train()
        best_loss = float('inf')
        best_state = None
        total_samples = len(x_tensor)
        num_batches = max(1, (total_samples + self.batch_size - 1) // self.batch_size)
        total_steps = self.epochs * num_batches
        current_step = 0
        stopped = False
        
        for epoch in range(self.epochs):
            if stopped:
                break
            epoch_loss = 0.0
            batch_count = 0
            
            for i in range(0, total_samples, self.batch_size):
                # Check for stop request
                if stop_check and stop_check():
                    if callback: callback(f"⛔ Stop requested at Epoch {epoch+1}, Batch {batch_count+1}. Saving best model...")
                    stopped = True
                    break
                
                batch_X = x_tensor[i : i + self.batch_size].to(self.device)
                batch_y = y_tensor[i : i + self.batch_size].to(self.device)
                
                self.optimizer.zero_grad()
                ctx = torch.zeros(len(batch_X), self.coord_hidden).to(self.device)
                x_dict = {"M5": batch_X}
                logits, _, _, _ = self.model(x_dict, ctx)
                loss = self.criterion(logits, batch_y)
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                batch_count += 1
                current_step += 1
            
            if batch_count > 0:
                avg_loss = epoch_loss / batch_count
                progress = int((current_step / total_steps) * 100)
                if callback: callback(f"Epoch {epoch+1}/{self.epochs} | Loss: {avg_loss:.4f} | Progress: {progress}%")
                
                # 🏆 Best Model Checkpoint
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                    if callback: callback(f"🏆 New best model! Loss: {best_loss:.4f} (Epoch {epoch+1})")

        # Restore and save the best model weights
        if best_state:
            self.model.load_state_dict(best_state)
        
        save_path = self.checkpoint_path or "logs/models/active_model.pth"
        if callback: callback(f"💾 Saving best checkpoint (Loss: {best_loss:.4f}) to {os.path.basename(save_path)}...")
        torch.save({'model_state_dict': self.model.state_dict()}, save_path)
        self.update_registry(best_loss, symbols)

    def train_on_dataset(self, dataset_name: str, callback=None, stop_check=None):
        shard_dir = f"data/cache/{dataset_name}/train_shards"
        if not os.path.exists(shard_dir):
            if callback: callback(f"Error: Dataset {dataset_name} shards not found.")
            return

        shards = sorted([f for f in os.listdir(shard_dir) if f.endswith(".pt")])
        if not shards:
            if callback: callback(f"Error: No shards found in {shard_dir}")
            return

        if callback: callback(f"🖥️ Device: {self.device} | 🚀 Found {len(shards)} shards. Running Supervised Protocol...")
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.learning_rate

        self.model.train()
        total_batches = 0
        best_loss = float('inf')
        best_state = None
        stopped = False
        
        for epoch in range(self.epochs):
            if stopped:
                break
            epoch_loss = 0.0
            epoch_batches = 0
            shard_count = 0
            for shard_file in shards:
                if stopped:
                    break
                shard_path = os.path.join(shard_dir, shard_file)
                # 🔥 CRITICAL FIX: Keep large dataset on CPU to prevent OOM
                data = torch.load(shard_path, map_location='cpu')
                X = data['X']
                y = data['y']
                
                if X.shape[-1] > 12:
                    X = X[:, :, :12]
                
                num_batches_in_shard = max(1, (len(X) + self.batch_size - 1) // self.batch_size)
                
                for i in range(0, len(X), self.batch_size):
                    # Check for stop request
                    if stop_check and stop_check():
                        if callback: callback(f"⛔ Stop requested at Epoch {epoch+1}, Batch {epoch_batches+1}. Saving best model...")
                        stopped = True
                        break
                    
                    batch_X = X[i : i + self.batch_size].to(self.device)
                    batch_y = y[i : i + self.batch_size].long().to(self.device)
                    
                    self.optimizer.zero_grad()
                    ctx = torch.zeros(len(batch_X), self.coord_hidden).to(self.device)
                    x_dict = {"M5": batch_X}
                    logits, _, _, _ = self.model(x_dict, ctx)
                    
                    mapped_y = torch.ones_like(batch_y).to(self.device)
                    mapped_y[batch_y == 1] = 1
                    mapped_y[batch_y == 0] = 2
                    
                    loss = self.criterion(logits, mapped_y)
                    loss.backward()
                    self.optimizer.step()
                    epoch_loss += loss.item()
                    epoch_batches += 1
                    total_batches += 1
                    
                    # Report every 25 batches for UI telemetry
                    if callback and epoch_batches % 25 == 0:
                        progress = int((epoch_batches / num_batches_in_shard) * 100)
                        callback(f"Epoch {epoch+1}/{self.epochs} | Batch {epoch_batches}/{num_batches_in_shard} | Loss: {loss.item():.4f} | {progress}%")
                
                shard_count += 1

            if epoch_batches > 0:
                avg_epoch_loss = epoch_loss / epoch_batches
                if callback: callback(f"✅ Epoch {epoch+1} Complete | Avg Loss: {avg_epoch_loss:.4f}")
                
                # 🏆 Best Model Checkpoint
                if avg_epoch_loss < best_loss:
                    best_loss = avg_epoch_loss
                    best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                    if callback: callback(f"🏆 New best model! Loss: {best_loss:.4f} (Epoch {epoch+1})")

        # Restore and save the best model weights
        if best_state:
            self.model.load_state_dict(best_state)
        
        save_path = self.checkpoint_path or "logs/models/active_model.pth"
        if callback: callback(f"💾 Saving best checkpoint (Loss: {best_loss:.4f}) to {os.path.basename(save_path)}...")
        torch.save({'model_state_dict': self.model.state_dict()}, save_path)
        self.update_registry(best_loss, {dataset_name})
        
    def update_registry(self, loss: float, symbols: set):
        version_tag = "v9.7-Live"
        registry_entry = self.db.query(ModelVersion).filter(ModelVersion.tag == version_tag).first()
        
        if not registry_entry:
            registry_entry = ModelVersion(
                tag=version_tag,
                path="logs/models/active_model.pth",
                is_active=True,
                metrics={}
            )
            self.db.add(registry_entry)
            self.db.commit()
            self.db.refresh(registry_entry)
            
        current_metrics = registry_entry.metrics or {}
        for symbol in symbols:
            current_metrics[symbol] = {
                "loss": round(loss, 4),
                "last_trained": datetime.utcnow().isoformat()
            }
        registry_entry.metrics = current_metrics
        registry_entry.category_type = "GENERALIST" if len(current_metrics.keys()) > 1 else "SPECIALIST"
        self.db.commit()

if __name__ == "__main__":
    learner = ContinuousLearner()
    learner.train_and_evaluate()
