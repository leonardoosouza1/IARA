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
from src.brain.losses import SwarmLoss # 🚀 Swarm v9.7 Loss
from src.brain.models.generator import ChallengerGenerator, OverfittingGuard # 🚀 Phase 13

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
        
        # Defaults for architecture (Golden v10.0 / Sniper Mode)
        self.worker_hidden = 64 # Matches Golden active_model.pth
        self.coord_hidden = 256 # Matches Golden active_model.pth
        self.seq_len = 128
        self.input_dim = 14 # Grown from 12 to 14 (Added ATR and Volume VMA)
        
        # Hyperparameters (Gladiator v9.7 Presets)
        self.epochs = 5
        self.batch_size = 256
        self.learning_rate = 1e-4
        
        self.model = self._load_model()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        
        # 🚀 Phase 11: SwarmLoss with Dynamic Weighting
        self.criterion = SwarmLoss(alpha=0.75, gamma=2.0, consensus_weight=0.1).to(self.device)
        self.scaler = torch.amp.GradScaler('cuda', enabled=self._use_grad_scaler())

        # 🚀 Phase 13: Adversarial Components
        self.challenger = ChallengerGenerator(input_dim=self.input_dim, hidden_dim=64).to(self.device)
        self.challenger_opt = torch.optim.AdamW(self.challenger.parameters(), lr=1e-4)
        self.guard = OverfittingGuard(ema_alpha=0.1).to(self.device)

    def _use_grad_scaler(self) -> bool:
        """Detect if GradScaler should be enabled. Disabled on AMD ROCm."""
        if self.device.type != "cuda":
            return False
        # ROCm/HIP reports torch.version.hip
        if hasattr(torch.version, 'hip') and torch.version.hip is not None:
            logger.info("🔧 AMD ROCm detected — GradScaler DISABLED (autocast still active)")
            return False
        return True

    def _load_model(self) -> SwarmBrain:
        worker_cfg = WorkerConfig(input_dim=self.input_dim, hidden_dim=self.worker_hidden)
        coord_cfg = CoordinatorConfig(num_workers=5, worker_dim=self.worker_hidden, hidden_dim=self.coord_hidden)
        
        model = SwarmBrain(worker_cfg, coord_cfg).to(self.device)
        
        # Load weights if available
        if self.checkpoint_path and os.path.exists(self.checkpoint_path):
            try:
                state = torch.load(self.checkpoint_path, map_location=self.device)
                state_dict = state.get('model_state_dict', state)
                
                # 🚀 Phase 3: Dynamic Padding for Input Growth
                # If loading older v9.7 weights (12 dims) into v10.0 model (14 dims)
                for name, param in model.named_parameters():
                    if name in state_dict and any(key in name for key in ["input_proj.weight", "cfc.x_proj.weight", "res_proj.weight"]):
                        old_w = state_dict[name]
                        if len(old_w.shape) == 2 and old_w.shape[1] < param.shape[1]:
                            logger.info(f"🦾 Zero-padding {name} from {old_w.shape[1]} to {param.shape[1]} dims")
                            new_w = torch.zeros_like(param)
                            new_w[:, :old_w.shape[1]] = old_w
                            state_dict[name] = new_w

                model.load_state_dict(state_dict, strict=False)
                logger.info(f"Loaded existing model from {self.checkpoint_path} (strict=False)")
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
            
            # 🚀 Phase 11: Standardized Mapping (0: Long, 1: Short, 2: Wait)
            target_scalar = 2 # Default to Wait
            if ideal_action == 1:
                target_scalar = 0 # Long
            elif ideal_action == -1:
                target_scalar = 1 # Short
                
            # 🚀 Phase 11: Correct Data Parsing (No more Dummies)
            # Parse the features_snapshot from the Signal
            try:
                features = torch.tensor(signal.features_snapshot, dtype=torch.float32)
                if features.dim() == 2: # [seq_len, dim]
                    features = features.unsqueeze(0) # [1, seq_len, dim]
                
                # Truncate/Pad to match seq_len and feature_dim
                if features.shape[1] > self.seq_len:
                    features = features[:, -self.seq_len:, :]
                
                
                # 🚀 Dynamic Tensor Padding for Backwards Compatibility
                # Old snapshots typically had 22 or 26 features, or were already truncated to 12.
                is_old_snapshot = (features.shape[-1] in [12, 22, 26])
                
                if is_old_snapshot:
                    core_feats = features[:, :, :12]
                    if self.input_dim > 12:
                        padding = torch.zeros(features.shape[0], features.shape[1], self.input_dim - 12, dtype=features.dtype, device=features.device)
                        features = torch.cat([core_feats, padding], dim=-1)
                    else:
                        features = core_feats
                else:
                    if features.shape[-1] > self.input_dim:
                        features = features[:, :, :self.input_dim]
                    elif features.shape[-1] < self.input_dim:
                        padding = torch.zeros(features.shape[0], features.shape[1], self.input_dim - features.shape[-1], dtype=features.dtype, device=features.device)
                        features = torch.cat([features, padding], dim=-1)
                    
                X.append(features.to(self.device))
                Y.append(target_scalar)
            except Exception as e:
                logger.error(f"Failed to parse snapshot for signal {signal.id}: {e}")
                continue
            
        if not X:
            return None, None, set()
            
        x_tensor = torch.cat(X, dim=0).to(self.device)
        y_tensor = torch.tensor(Y, dtype=torch.long).to(self.device)
        
        return x_tensor, y_tensor, symbols_seen

    def _simulate_timeframes(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Simulates the Multi-Timeframe Brain views from a single M1 sequence.
        Downsamples the [B, L, C] tensor to provide specialists with localized views.
        """
        seq_len = x.shape[1]
        x_dict = {}
        
        # M5: Standard view
        x_dict["M5"] = x
        
        # M15: Every 3rd candle
        x_dict["M15"] = x[:, ::3, :]
        
        # M30: Every 6th candle
        x_dict["M30"] = x[:, ::6, :]
        
        # H1: Every 12th candle
        x_dict["H1"] = x[:, ::12, :]
        
        # H4: Every 48th candle
        x_dict["H4"] = x[:, ::16, :] # Truncated to avoid too small sequences in test
        
        return x_dict

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
                
                # 🛡️ Sync Dimensions: Slice to model's expected input_dim and seq_len
                if batch_X.shape[-1] > self.input_dim:
                    batch_X = batch_X[:, :, :self.input_dim]
                if batch_X.shape[1] > self.seq_len:
                    batch_X = batch_X[:, -self.seq_len:, :]
                
                self.optimizer.zero_grad()
                
                # 🚀 Swarm Implementation: Multi-TF Input
                x_dict = self._simulate_timeframes(batch_X)
                
                with torch.amp.autocast('cuda', enabled=(self.device.type == "cuda")):
                    logits, worker_logits, _, _ = self.model(x_dict)
                    
                    # 🚀 Phase 11: Swarm v9.7 Targets {0: Long, 1: Short, 2: Wait}
                    # 🎯 Phase 3: High Precision (Sniper Mode) - Asymmetry handled directly by SwarmLoss
                    loss_dict = self.criterion(logits, worker_logits, batch_y)
                    loss = loss_dict["total"]

                if torch.isnan(loss):
                    if callback: callback(f"⚠️ Warning: NaN detected in Batch {batch_count+1}. Skipping update.")
                    continue

                self.scaler.scale(loss).backward()
                
                # 🚀 Phase 11: Gradient Clipping for Stability
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
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
                
                
                # 🚀 Dynamic Tensor Padding for Older Datasets
                # Old shards (v9) have 26 cols, or were truncated to 12. New shards (v10) have 28 cols.
                is_old_shard = (X.shape[-1] in [12, 22, 26])
                
                if is_old_shard:
                    core_X = X[:, :, :12]
                    if self.input_dim > 12:
                        padding = torch.zeros(X.shape[0], X.shape[1], self.input_dim - 12, dtype=X.dtype, device=X.device)
                        X = torch.cat([core_X, padding], dim=-1)
                    else:
                        X = core_X
                else:
                    if X.shape[-1] > self.input_dim:
                        X = X[:, :, :self.input_dim]
                    elif X.shape[-1] < self.input_dim:
                        padding = torch.zeros(X.shape[0], X.shape[1], self.input_dim - X.shape[-1], dtype=X.dtype, device=X.device)
                        X = torch.cat([X, padding], dim=-1)
                
                num_batches_in_shard = max(1, (len(X) + self.batch_size - 1) // self.batch_size)
                
                for i in range(0, len(X), self.batch_size):
                    # Check for stop request
                    if stop_check and stop_check():
                        if callback: callback(f"⛔ Stop requested at Epoch {epoch+1}, Batch {epoch_batches+1}. Saving best model...")
                        stopped = True
                        break
                    
                    batch_X = X[i : i + self.batch_size].to(self.device)
                    batch_y = y[i : i + self.batch_size].long().to(self.device)
                    
                    # 🛡️ Sync Dimensions: Slice to model's expected input_dim and seq_len
                    if batch_X.shape[-1] > self.input_dim:
                        batch_X = batch_X[:, :, :self.input_dim]
                    if batch_X.shape[1] > self.seq_len:
                        batch_X = batch_X[:, -self.seq_len:, :]
                    
                    # 🚀 Phase 11: Feature Sanitization
                    batch_X = torch.nan_to_num(batch_X)
                    
                    # 🚀 Phase 13: Adversarial "Prank" Generation
                    # 30% chance to face a 'Prank' from the Challenger
                    is_adversarial = torch.rand(1).item() < 0.3
                    current_pressure = self.guard.get_pressure()
                    
                    if is_adversarial:
                        # Challenger attempts to 'prank' the features to confuse the Trader
                        self.challenger.eval() # Generator doesn't update its own weights during this pass
                        batch_X_orig = batch_X.clone()
                        
                        # LOG: Feed back the "Sadistic Professor" behavior to the UI
                        if callback and epoch_batches % 50 == 0:
                            callback(f"😈 Professor Sádico: Aplicando modo aleatório (Pressão: {current_pressure:.2f})")
                            
                        with torch.no_grad():
                            batch_X = self.challenger(batch_X, pressure=current_pressure, callback=callback if epoch_batches % 50 == 0 else None)
                    
                    self.optimizer.zero_grad()
                    if is_adversarial:
                        self.challenger_opt.zero_grad()

                    x_dict = self._simulate_timeframes(batch_X)
                    
                    with torch.amp.autocast('cuda', enabled=(self.device.type == "cuda")):
                        logits, worker_logits, _, _ = self.model(x_dict)
                        loss_dict = self.criterion(logits, worker_logits, batch_y)
                        loss = loss_dict["total"]

                    if torch.isnan(loss):
                        continue

                    # 1. Trader Backward
                    self.scaler.scale(loss).backward()
                    
                    # 2. Challenger Backward (if needed)
                    if is_adversarial:
                        self.challenger.train()
                        for p in self.model.parameters(): p.requires_grad = False
                        
                        batch_X_adv = self.challenger(batch_X_orig, pressure=current_pressure)
                        x_dict_adv = self._simulate_timeframes(batch_X_adv)
                        
                        with torch.amp.autocast('cuda', enabled=(self.device.type == "cuda")):
                            logits_adv, worker_logits_adv, _, _ = self.model(x_dict_adv)
                            loss_adv_dict = self.criterion(logits_adv, worker_logits_adv, batch_y)
                            challenger_loss = -loss_adv_dict["total"] 
                        
                        self.scaler.scale(challenger_loss).backward()
                        for p in self.model.parameters(): p.requires_grad = True

                    # 3. Step Optimizers
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    
                    self.scaler.step(self.optimizer)
                    if is_adversarial:
                        self.scaler.unscale_(self.challenger_opt)
                        self.scaler.step(self.challenger_opt)
                        
                    # 4. Final Scaler Update
                    self.scaler.update()
                    
                    # 🚀 Phase 11: Multi-Class Metrics (Accuracy, Precision)
                    from sklearn.metrics import accuracy_score, precision_score
                    with torch.no_grad():
                        preds = logits.argmax(dim=-1).cpu().numpy()
                        true_y = batch_y.cpu().numpy()
                        acc = accuracy_score(true_y, preds)
                        # We use macro precision to avoid bias if classes are imbalanced
                        prec = precision_score(true_y, preds, average='macro', zero_division=0)
                    
                    epoch_loss += loss.item()
                    epoch_batches += 1
                    total_batches += 1
                    
                    # Report every 25 batches for UI telemetry
                    if callback and epoch_batches % 25 == 0:
                        progress = int((epoch_batches / num_batches_in_shard) * 100)
                        sw_weights = [f"{w:.2f}" for w in loss_dict["weights"]]
                        
                        # Expert "Thoughts" Insight
                        # M5, M15, M30, H1, H4
                        exp_votes = worker_logits.argmax(dim=-1).cpu().numpy()[0] # Take first batch sample
                        v_map = {0: "L", 1: "S", 2: "W"}
                        thought_str = "-".join([v_map[v] for v in exp_votes])
                        
                        callback(f"Epoch {epoch+1} | B {epoch_batches} | Loss: {loss.item():.4f} | Acc: {acc:.2f} | Prec: {prec:.2f} | Swarm: {thought_str} | W: {sw_weights} | {progress}%")
                
                shard_count += 1
                
                # Cleanup VRAM between shards to prevent fragmentation
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()

            # 🚀 Phase 13: Validation Pass & OverfittingGuard Update
            val_shard_dir = f"data/cache/{dataset_name}/val_shards"
            if os.path.exists(val_shard_dir):
                self.model.eval()
                val_loss = 0.0
                val_batches = 0
                val_shards = sorted([f for f in os.listdir(val_shard_dir) if f.endswith(".pt")])
                
                with torch.no_grad():
                    for v_shard in val_shards[:2]: # Sample 2 shards for speed
                        v_data = torch.load(os.path.join(val_shard_dir, v_shard), map_location='cpu')
                        v_X, v_y = v_data['X'].to(self.device), v_data['y'].to(self.device).long()
                        
                        # 🚀 Dynamic Tensor Padding for Validation Datasets
                        is_old_val_shard = (v_X.shape[-1] in [12, 22, 26])
                        
                        if is_old_val_shard:
                            core_v_X = v_X[:, :, :12]
                            if self.input_dim > 12:
                                padding = torch.zeros(v_X.shape[0], v_X.shape[1], self.input_dim - 12, dtype=v_X.dtype, device=v_X.device)
                                v_X = torch.cat([core_v_X, padding], dim=-1)
                            else:
                                v_X = core_v_X
                        else:
                            if v_X.shape[-1] > self.input_dim:
                                v_X = v_X[:, :, :self.input_dim]
                            elif v_X.shape[-1] < self.input_dim:
                                padding = torch.zeros(v_X.shape[0], v_X.shape[1], self.input_dim - v_X.shape[-1], dtype=v_X.dtype, device=v_X.device)
                                v_X = torch.cat([v_X, padding], dim=-1)
                        
                        v_x_dict = self._simulate_timeframes(v_X)
                        v_logits, v_worker_logits, _, _ = self.model(v_x_dict)
                        v_loss_dict = self.criterion(v_logits, v_worker_logits, v_y)
                        val_loss += v_loss_dict["total"].item()
                        val_batches += 1
                
                if val_batches > 0:
                    avg_val_loss = val_loss / val_batches
                    avg_train_loss = epoch_loss / epoch_batches
                    self.guard.update(avg_train_loss, avg_val_loss)
                    pressure = self.guard.get_pressure()
                    if callback: callback(f"🛡️ Guard: Precision-Recall Check | Pressure: {pressure:.2f} | Val Loss: {avg_val_loss:.4f}")

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
