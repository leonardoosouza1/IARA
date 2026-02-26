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
import numpy as np
import warnings
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
    def __init__(self, checkpoint_path: str = "logs/models/active_model_v10.pth", device: Optional[str] = None):
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
        
        # 🚀 Phase 22: Dynamic Input Dimension from Advanced Data Science Pipeline
        self.input_dim = 26 # Fallback
        norm_path = "data/cache/EURUSD/norm_params.pt"
        if os.path.exists(norm_path):
            try:
                meta = torch.load(norm_path, map_location='cpu', weights_only=True)
                if 'feature_cols' in meta:
                    self.input_dim = len(meta['feature_cols'])
                    logger.info(f"🧬 Loaded dynamic input_dim={self.input_dim} from Boruta ExtraTrees pruning.")
            except Exception as e:
                logger.warning(f"⚠️ Could not load norm_params.pt: {e}. Using fallback input_dim=26")
        
        
        # Hyperparameters (Gladiator v9.7 Presets)
        self.epochs = 5
        self.batch_size = 256
        self.learning_rate = 1e-4
        
        # 🚀 Phase 11: SwarmLoss with Dynamic Weighting
        self.criterion = SwarmLoss(alpha=0.75, gamma=2.0, consensus_weight=0.1).to(self.device)

        import itertools
        self.model = self._load_model()
        
        # 🚀 Phase 11 & 19: Separate learning rates for Model and Meta-Weights
        # Swarm Weights (logits) need a much higher LR to adapt quickly (1e-2 vs 1e-4)
        optimizer_grouped_parameters = [
            {"params": self.model.parameters(), "lr": self.learning_rate},
            {"params": self.criterion.parameters(), "lr": 1e-2}
        ]
        self.optimizer = torch.optim.AdamW(optimizer_grouped_parameters)
        
        # 🚀 Phase 14 v2: Persistent Prioritized Experience Replay
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=2)
        self.trauma_buffer_X = []
        self.trauma_buffer_Y = []
        self.trauma_buffer_losses = []  # Track individual sample losses for priority
        self.trauma_threshold = 1.15  # Minimum loss to enter trauma memory (adjusted for stabilized loss)
        self.trauma_mastery_threshold = 0.95  # Loss below this = mastered, can be evicted
        self.trauma_mix_ratio = 0.20  # 20% of each batch will be trauma samples
        self.trauma_mix_interval = 3  # Mix trauma every N batches
        self.MAX_TRAUMA = 15000  # Larger persistent memory
        
        self.scaler = torch.amp.GradScaler('cuda', enabled=self._use_grad_scaler())

        # 🚀 Phase 13: Adversarial Components
        self.challenger = ChallengerGenerator(input_dim=self.input_dim, hidden_dim=64).to(self.device)
        self.challenger_opt = torch.optim.AdamW(self.challenger.parameters(), lr=1e-4)
        self.guard = OverfittingGuard(ema_alpha=0.1).to(self.device)

        # 🚀 Plateau Breaker Variables (Phase 19)
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        self.patience_threshold = 7
        
        # 🚀 Phase 20: Anti-DDoS Challenger Cooldown
        self.challenger_cooldown = 0

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
                    
                # 🚀 Phase 11: Feature Sanitization
                features = torch.nan_to_num(features)
                features = torch.clamp(features, min=-20.0, max=20.0)
                    
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

    def _add_to_trauma_buffer(self, x: torch.Tensor, y: torch.Tensor, loss_val: float = None):
        """Phase 14 v2: Persistent Memory. Stores high-loss samples with their loss scores."""
        for i in range(len(x)):
            self.trauma_buffer_X.append(x[i].cpu())
            self.trauma_buffer_Y.append(y[i].cpu())
            self.trauma_buffer_losses.append(loss_val if loss_val else self.trauma_threshold)
        
        if len(self.trauma_buffer_X) > self.MAX_TRAUMA:
            # Evict the EASIEST traumas (lowest loss), keeping the hardest ones
            indexed = sorted(enumerate(self.trauma_buffer_losses), key=lambda x: x[1], reverse=True)
            keep_indices = [idx for idx, _ in indexed[:self.MAX_TRAUMA]]
            keep_indices.sort()  # Preserve temporal order
            self.trauma_buffer_X = [self.trauma_buffer_X[i] for i in keep_indices]
            self.trauma_buffer_Y = [self.trauma_buffer_Y[i] for i in keep_indices]
            self.trauma_buffer_losses = [self.trauma_buffer_losses[i] for i in keep_indices]

    def _get_trauma_mix_batch(self, current_batch_size: int):
        """Phase 14 v2: Sample trauma memories to mix into current training batch.
        Returns a subset of trauma samples proportional to trauma_mix_ratio.
        Higher-loss traumas have higher probability of being selected (Prioritized Replay)."""
        if not self.trauma_buffer_X:
            return None, None
        
        n_trauma = max(1, int(current_batch_size * self.trauma_mix_ratio))
        n_trauma = min(n_trauma, len(self.trauma_buffer_X))
        
        # Prioritized sampling: probability proportional to loss
        losses = np.array(self.trauma_buffer_losses)
        probs = losses / losses.sum()  # Normalize to probability distribution
        
        indices = np.random.choice(len(self.trauma_buffer_X), size=n_trauma, replace=False, p=probs)
        
        X = torch.stack([self.trauma_buffer_X[i] for i in indices]).to(self.device)
        Y = torch.stack([self.trauma_buffer_Y[i] for i in indices]).to(self.device)
        return X, Y

    def _train_trauma_replay(self, callback=None, epochs: int = 3):
        """Phase 14 v2: Intensive Persistent Replay with Graduated Eviction.
        Replays trauma buffer multiple times to force error correction.
        Instead, samples that the model has 'mastered' (loss < mastery_threshold) are evicted."""
        if not self.trauma_buffer_X:
            return
            
        if callback: callback(f"🧠 Initiating Intensive Trauma Replay ({epochs} epochs)... Retraining on {len(self.trauma_buffer_X)} persistent memories.")
        self.model.train()
        
        X = torch.stack(self.trauma_buffer_X)
        Y = torch.stack(self.trauma_buffer_Y)
        
        mastered_indices = set()  # Track which samples have been mastered across all replays
        
        for replay_epoch in range(epochs):
            total_loss = 0.0
            batches = 0
            sample_offset = 0
            
            # Shuffle data for this replay epoch
            indices = torch.randperm(len(X))
            X_shuffled = X[indices]
            Y_shuffled = Y[indices]
            
            for i in range(0, len(X), self.batch_size):
                batch_X = X_shuffled[i:i+self.batch_size].to(self.device)
                batch_y = Y_shuffled[i:i+self.batch_size].to(self.device)
                batch_len = len(batch_X)
                
                self.optimizer.zero_grad()
                x_dict = self._simulate_timeframes(batch_X)
            
            with torch.amp.autocast('cuda', enabled=(self.device.type == "cuda")):
                logits, worker_logits, _, _, attn_w, confidence = self.model(x_dict)
                loss_dict = self.criterion(logits, worker_logits, batch_y)
                loss = loss_dict["total"]
                
                # Titanium Shield for Trauma Replay
                if not torch.isfinite(loss) or loss.item() > 15.0:
                    sample_offset += batch_len
                    continue
                
                # Check per-sample losses to identify mastered items
                with torch.no_grad():
                    per_sample_loss = torch.nn.functional.cross_entropy(logits, batch_y, reduction='none')
                    for j in range(batch_len):
                        orig_idx = indices[sample_offset + j].item()
                        sample_loss = per_sample_loss[j].item()
                        # Update the stored loss for this sample
                        if orig_idx < len(self.trauma_buffer_losses):
                            self.trauma_buffer_losses[orig_idx] = sample_loss
                        # Mark as mastered if loss is low enough
                        if sample_loss < self.trauma_mastery_threshold:
                            mastered_indices.add(orig_idx)
                    
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                total_loss += loss.item()
                batches += 1
                sample_offset += batch_len
                
            if callback and batches > 0:
                callback(f"  ↪ Pass {replay_epoch+1}/{epochs} | Avg Loss: {total_loss/batches:.4f} | Cumulative Mastered: {len(mastered_indices)}/{len(X)}")
        
        # Graduated Eviction: Only remove samples the model has proven mastery over
        if mastered_indices:
            keep_mask = set(range(len(self.trauma_buffer_X))) - mastered_indices
            keep_list = sorted(keep_mask)
            self.trauma_buffer_X = [self.trauma_buffer_X[i] for i in keep_list]
            self.trauma_buffer_Y = [self.trauma_buffer_Y[i] for i in keep_list]
            self.trauma_buffer_losses = [self.trauma_buffer_losses[i] for i in keep_list]

    def _emit_epoch_comparison(self, epoch_history: list, callback=None):
        """Phase 14 v2: Emit a bird's-eye comparison table of all completed epochs."""
        if not callback or not epoch_history:
            return
        
        callback("━" * 90)
        callback("📊 EPOCH COMPARISON DASHBOARD")
        callback("━" * 90)
        callback(f"{'Epoch':<7} {'Loss':>8} {'Acc':>6} {'Prec':>6} {'AUC':>6} {'F1':>6} {'Traumas':>9} {'Mastered':>10} {'LR':>10}")
        callback("─" * 90)
        
        for entry in epoch_history:
            lr_str = f"{entry['lr']:.6f}"
            callback(
                f"  E{entry['epoch']:<4} "
                f"{entry['loss']:>8.4f} "
                f"{entry['acc']:>6.2f} "
                f"{entry['prec']:>6.2f} "
                f"{entry['auc']:>6.2f} "
                f"{entry['f1']:>6.2f} "
                f"{entry['trauma_size']:>9} "
                f"{entry['mastered']:>10} "
                f"{lr_str:>10}"
            )
        
        # Show trend arrows comparing last two epochs
        if len(epoch_history) >= 2:
            prev = epoch_history[-2]
            curr = epoch_history[-1]
            arrows = []
            for metric in ['loss', 'acc', 'prec', 'auc', 'f1']:
                diff = curr[metric] - prev[metric]
                if metric == 'loss':
                    # Lower loss is better
                    arrow = "🟢↓" if diff < -0.01 else ("🔴↑" if diff > 0.01 else "⚪→")
                else:
                    # Higher is better for other metrics
                    arrow = "🟢↑" if diff > 0.01 else ("🔴↓" if diff < -0.01 else "⚪→")
                arrows.append(f"{metric.upper()}: {arrow}")
            callback("─" * 90)
            callback(f"📈 Tendência: {' | '.join(arrows)}")
        
        callback("━" * 90)

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
        epoch_history = []  # 📊 Bird's-eye epoch comparison tracker
        
        for epoch in range(self.epochs):
            if stopped:
                break
            epoch_loss = 0.0
            batch_count = 0
            
            # 🧠 Phase 20: Epoch-level accumulators for robust AUC
            epoch_all_preds = []
            epoch_all_labels = []
            epoch_all_probs = []
            
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
                
                # 🚀 Dojo Warmup: Curriculum Learning Schedule 
                adv_prob = 0.0 if epoch == 0 else (0.1 if epoch == 1 else 0.3)
                
                # 🚀 Phase 13: Adversarial "Prank" Generation
                if self.challenger_cooldown > 0:
                    is_adversarial = False
                    self.challenger_cooldown -= 1
                else:
                    is_adversarial = torch.rand(1).item() < adv_prob
                
                current_pressure = self.guard.get_pressure()
                
                if is_adversarial:
                    self.challenger_cooldown = 3  # 🔥 Cooldown of 3 batches to prevent DDoS
                    self.challenger.eval()
                    batch_X_orig = batch_X.clone()
                    if callback and batch_count % 50 == 0:
                        callback(f"😈 Professor Sádico: Aplicando modo aleatório (Pressão: {current_pressure:.2f})")
                    with torch.no_grad():
                        batch_X, _ = self.challenger(batch_X, pressure=current_pressure, callback=callback if batch_count % 50 == 0 else None)
                
                self.optimizer.zero_grad()
                if is_adversarial:
                    self.challenger_opt.zero_grad()

                # 🚀 Swarm Implementation: Multi-TF Input
                x_dict = self._simulate_timeframes(batch_X)
                
                with torch.amp.autocast('cuda', enabled=(self.device.type == "cuda")):
                    logits, worker_logits, _, _, _, _ = self.model(x_dict)
                    
                    # 🚀 Phase 11: Swarm v9.7 Targets {0: Long, 1: Short, 2: Wait}
                    # 🎯 Phase 3: High Precision (Sniper Mode) - Asymmetry handled directly by SwarmLoss
                    loss_dict = self.criterion(logits, worker_logits, batch_y)
                    loss = loss_dict["total"]

                # 🛡️ Titanium Shield: Pre-Backward Protection against NaNs and Exceeding Bounds
                if torch.isnan(loss) or torch.isinf(loss) or loss.item() > 15.0:
                    val_str = f"NaN/Inf" if not torch.isfinite(loss) else f"{loss.item():.2f}"
                    if callback: callback(f"⚠️ Red Alert: Extreme Loss ({val_str}) detected in Batch {batch_count+1}. Titanium Shield activated. Skipping update to prevent collapse.")
                    continue

                # Phase 14 v2: Populate Trauma Buffer with loss score
                current_loss = loss.item()
                if current_loss > self.trauma_threshold:
                    self._add_to_trauma_buffer(batch_X.detach(), batch_y.detach(), loss_val=current_loss)
                
                # Phase 14 v2: Inline Trauma Mix - inject old mistakes into current learning
                if batch_count % self.trauma_mix_interval == 0 and len(self.trauma_buffer_X) > 0:
                    trauma_X, trauma_Y = self._get_trauma_mix_batch(len(batch_X))
                    if trauma_X is not None:
                        trauma_dict = self._simulate_timeframes(trauma_X)
                        with torch.amp.autocast('cuda', enabled=(self.device.type == "cuda")):
                            t_logits, t_worker_logits, _, _, _, _ = self.model(trauma_dict)
                            t_loss_dict = self.criterion(t_logits, t_worker_logits, trauma_Y)
                            trauma_loss = t_loss_dict["total"] * 0.3  # Weight trauma replay at 30%
                        if not torch.isnan(trauma_loss):
                            self.scaler.scale(trauma_loss).backward()

                # 1. Trader Backward
                self.scaler.scale(loss).backward()
                
                # 2. Challenger Backward
                if is_adversarial:
                    self.challenger.train()
                    for p in self.model.parameters(): p.requires_grad = False
                    batch_X_adv, _ = self.challenger(batch_X_orig, pressure=current_pressure)
                    x_dict_adv = self._simulate_timeframes(batch_X_adv)
                    with torch.amp.autocast('cuda', enabled=(self.device.type == "cuda")):
                        logits_adv, _, _, _, _, _ = self.model(x_dict_adv)
                        loss_adv_dict = self.criterion(logits_adv, worker_logits, batch_y) # worker_logits reuse
                        challenger_loss = -loss_adv_dict["total"]
                    self.scaler.scale(challenger_loss).backward()
                    for p in self.model.parameters(): p.requires_grad = True

                # 3. Step Optimizers
                self.scaler.unscale_(self.optimizer)
                # 🛡️ Titanium Shield: Strict Maximum Gradient Normalization
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.scaler.step(self.optimizer)
                if is_adversarial:
                    self.scaler.unscale_(self.challenger_opt)
                    self.scaler.step(self.challenger_opt)
                
                self.scaler.update()
                
                # 🚀 Phase 11 & 14: Multi-Class Metrics (Accuracy, Precision, AUC, F1)
                from sklearn.metrics import accuracy_score, precision_score, f1_score, roc_auc_score
                with torch.no_grad():
                    logits_np = logits.detach().cpu().numpy()
                    preds = logits_np.argmax(axis=-1)
                    true_y = batch_y.cpu().numpy()
                    
                    acc = accuracy_score(true_y, preds)
                    prec = precision_score(true_y, preds, average='macro', zero_division=0)
                    f1 = f1_score(true_y, preds, average='macro', zero_division=0)
                    
                    # 🧠 Phase 20: Accumulate for epoch-level AUC
                    probs = torch.softmax(logits.detach(), dim=-1).cpu().numpy()
                    epoch_all_preds.append(preds)
                    epoch_all_labels.append(true_y)
                    epoch_all_probs.append(probs)
                    
                    # Compute running AUC from accumulated data (use only present classes)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        try:
                            all_labels_so_far = np.concatenate(epoch_all_labels)
                            all_probs_so_far = np.concatenate(epoch_all_probs)
                            present_classes = np.unique(all_labels_so_far).astype(int).tolist()
                            if len(present_classes) == 3:
                                # Force strict normalization to exactly 1.0 to avoid scikit-learn float32 precision crash
                                norm_probs = all_probs_so_far / all_probs_so_far.sum(axis=1, keepdims=True)
                                auc = roc_auc_score(all_labels_so_far, norm_probs, multi_class='ovr', average='macro', labels=[0, 1, 2])
                            elif len(present_classes) == 2:
                                # Binary classification fallback
                                pos_class = present_classes[1]
                                binary_probs = all_probs_so_far[:, present_classes]
                                binary_probs = binary_probs / binary_probs.sum(axis=1, keepdims=True)
                                auc = roc_auc_score(all_labels_so_far, binary_probs[:, 1], labels=present_classes)
                            else:
                                auc = 0.5
                                
                            if np.isnan(auc):
                                auc = 0.5
                        except Exception as e:
                            logger.error(f"AUC Error A: {e}") 
                            auc = 0.5
                
                epoch_loss += loss.item()
                batch_count += 1
                current_step += 1
                
                # Report every 10 batches for high-fidelity UI tracking
                if batch_count % 10 == 0:
                    progress = int((current_step / total_steps) * 100)
                    
                    # Log to console less frequently (every 100 batches) to prevent spam
                    if callback and batch_count % 100 == 0:
                        sw_weights = [f"{w:.2f}" for w in loss_dict["weights"]]
                        exp_votes = worker_logits.argmax(dim=-1).cpu().numpy()[0]
                        v_map = {0: "L", 1: "S", 2: "W"}
                        thought_str = "-".join([v_map[v] for v in exp_votes])
                        callback(f"Epoch {epoch+1} | B {batch_count} | Loss: {loss.item():.4f} | Acc: {acc:.2f} | Prec: {prec:.2f} | AUC: {auc:.2f} | F1: {f1:.2f} | Swarm: {thought_str} | W: {sw_weights} | Pressure: {is_adversarial * current_pressure:.2f} | {progress}%")
                    
                    # 🧠 Phase 20: Micro-State Telemetry for "Opened Brain" Dashboard
                    if hasattr(self, '_micro_state_callback') and self._micro_state_callback:
                        try:
                            with torch.no_grad():
                                _, _, _, _, attn_w, confidence = self.model(x_dict, compute_telemetry=True)
                                w_probs = torch.softmax(worker_logits[0], dim=-1).cpu().tolist()
                                c_policy = torch.softmax(logits[0], dim=-1).cpu().tolist()
                                a_weights = attn_w[0].cpu().tolist() if attn_w is not None else None
                                conf_scores = confidence[0].squeeze(-1).cpu().tolist() if confidence is not None else None
                            self._micro_state_callback({
                                "worker_thoughts": w_probs,
                                "attention_weights": a_weights,
                                "consensus_weights": [float(w) for w in loss_dict["weights"]],
                                "coordinator_policy": c_policy,
                                "confidence_scores": conf_scores,
                                "pressure": float(is_adversarial * current_pressure),
                                "trauma_buffer_size": len(self.trauma_buffer_X),
                                "learning_rate": float(self.optimizer.param_groups[0]['lr']),
                            })
                        except Exception:
                            pass
            
            if batch_count > 0:
                avg_loss = epoch_loss / batch_count
                progress = int((current_step / total_steps) * 100)
                if callback: callback(f"Epoch {epoch+1}/{self.epochs} | Loss: {avg_loss:.4f} | Progress: {progress}%")
                
                # 🏆 Best Model Checkpoint
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                    if callback: callback(f"🏆 Epoch {epoch+1} | New best model (Journal)! Loss: {best_loss:.4f}")
                    # Phase 14: Immediate Auto-Save
                    save_path = self.checkpoint_path or "logs/models/active_model_v10.pth"
                    torch.save({'model_state_dict': self.model.state_dict()}, save_path)
                    if callback: callback(f"💾 Epoch {epoch+1} | Auto-saved best model to {os.path.basename(save_path)}")
                
            # Phase 14: Execute Trauma Replay
            trauma_mastered = 0
            trauma_size_before = len(self.trauma_buffer_X)
            if len(self.trauma_buffer_X) > 0:
                if callback: callback(f"🧠 Epoch {epoch+1} | Initiating Trauma Replay... Retraining on {len(self.trauma_buffer_X)} persistent memories.")
                trauma_size_before = len(self.trauma_buffer_X)
                self._train_trauma_replay(lambda m: callback(f"Epoch {epoch+1} | {m}"))
                trauma_mastered = trauma_size_before - len(self.trauma_buffer_X)
                if callback: callback(f"✅ Epoch {epoch+1} | Trauma Replay completed.")
            
            # 📊 Epoch Comparison: Compute aggregate metrics and record snapshot
            if batch_count > 0:
                epoch_avg_acc = np.mean([accuracy_score(l, p) for l, p in zip(epoch_all_labels, epoch_all_preds)]) if epoch_all_preds else 0.0
                epoch_avg_prec = np.mean([precision_score(l, p, average='macro', zero_division=0) for l, p in zip(epoch_all_labels, epoch_all_preds)]) if epoch_all_preds else 0.0
                epoch_avg_f1 = np.mean([f1_score(l, p, average='macro', zero_division=0) for l, p in zip(epoch_all_labels, epoch_all_preds)]) if epoch_all_preds else 0.0
                # Epoch-level AUC from all accumulated probs
                epoch_auc = 0.5
                if epoch_all_probs and epoch_all_labels:
                    try:
                        all_l = np.concatenate(epoch_all_labels)
                        all_p = np.concatenate(epoch_all_probs)
                        present = np.unique(all_l).astype(int).tolist()
                        if len(present) >= 2:
                            norm_p = all_p / all_p.sum(axis=1, keepdims=True)
                            if len(present) == 3:
                                epoch_auc = roc_auc_score(all_l, norm_p, multi_class='ovr', average='macro', labels=[0,1,2])
                            else:
                                bp = norm_p[:, present]
                                bp = bp / bp.sum(axis=1, keepdims=True)
                                epoch_auc = roc_auc_score(all_l, bp[:, 1], labels=present)
                    except Exception:
                        epoch_auc = 0.5
                
                epoch_history.append({
                    'epoch': epoch + 1,
                    'loss': epoch_loss / batch_count,
                    'acc': epoch_avg_acc,
                    'prec': epoch_avg_prec,
                    'auc': epoch_auc,
                    'f1': epoch_avg_f1,
                    'trauma_size': len(self.trauma_buffer_X),
                    'mastered': trauma_mastered,
                    'lr': self.optimizer.param_groups[0]['lr'],
                })
                self._emit_epoch_comparison(epoch_history, callback)

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
        
        # --- GLOBAL PROGRESS PRE-CALCULATION ---
        # Sum total batches once to avoid redundant I/O per epoch
        total_epoch_batches = 0
        for s_file in shards:
            try:
                # Only load enough to get the size
                s_meta = torch.load(os.path.join(shard_dir, s_file), map_location='cpu', weights_only=True)
                s_X = s_meta.get('X')
                if s_X is not None:
                    total_epoch_batches += (len(s_X) + self.batch_size - 1) // self.batch_size
            except Exception:
                continue
        
        if total_epoch_batches == 0: total_epoch_batches = 1
        epoch_history = []  # 📊 Bird's-eye epoch comparison tracker
        
        for epoch in range(self.epochs):
            if stopped:
                break
            epoch_loss = 0.0
            epoch_batches = 0
            
            # 🧠 Phase 20: Consolidate ALL shards into one dataset for maximum GPU utilization
            all_X_list = []
            all_y_list = []
            for shard_file in shards:
                shard_path = os.path.join(shard_dir, shard_file)
                data = torch.load(shard_path, map_location='cpu')
                X_s = data['X']
                y_s = data['y']
                
                # Robust Dimension Migration
                shard_dim = X_s.shape[-1]
                if shard_dim == 12 and self.input_dim > 12:
                    padding = torch.zeros(X_s.shape[0], X_s.shape[1], self.input_dim - 12, dtype=X_s.dtype)
                    X_s = torch.cat([X_s, padding], dim=-1)
                elif shard_dim > self.input_dim:
                    X_s = X_s[:, :, :self.input_dim]
                elif shard_dim < self.input_dim:
                    padding = torch.zeros(X_s.shape[0], X_s.shape[1], self.input_dim - shard_dim, dtype=X_s.dtype)
                    X_s = torch.cat([X_s, padding], dim=-1)
                
                all_X_list.append(X_s)
                all_y_list.append(y_s)
            
            X = torch.cat(all_X_list, dim=0)
            y = torch.cat(all_y_list, dim=0)
            del all_X_list, all_y_list
            
            if callback and epoch == 0:
                callback(f"📂 Consolidated {len(shards)} shards → {len(X)} samples in memory")
            
            total_epoch_batches = max(1, (len(X) + self.batch_size - 1) // self.batch_size)
            
            # 🧠 Phase 20: Epoch-level accumulators for robust AUC
            epoch_all_preds = []
            epoch_all_labels = []
            epoch_all_probs = []
                
            for i in range(0, len(X), self.batch_size):
                # Check for stop request
                if stop_check and stop_check():
                    if callback: callback(f"⛔ Stop requested at Epoch {epoch+1}, Batch {epoch_batches+1}. Saving best model...")
                    stopped = True
                    break
                
                batch_X = X[i : i + self.batch_size].to(self.device)
                batch_y = y[i : i + self.batch_size].long().to(self.device)
                
                # 🛡️ Sync Dimensions
                if batch_X.shape[-1] > self.input_dim:
                    batch_X = batch_X[:, :, :self.input_dim]
                if batch_X.shape[1] > self.seq_len:
                    batch_X = batch_X[:, -self.seq_len:, :]
                
                # 🚀 Phase 11: Feature Sanitization
                batch_X = torch.nan_to_num(batch_X)
                batch_X = torch.clamp(batch_X, min=-20.0, max=20.0)
                
                # 🚀 Phase 13 & 16: Extreme Warmup Schedule
                # Ep 1-2: 0%, Ep 3-4: 5%, Ep 5-6: 10%, Ep 7-8: 15%, Ep 9+: 30%
                adv_schedule = [0.0, 0.0, 0.05, 0.05, 0.10, 0.10, 0.15, 0.15]
                adv_prob = adv_schedule[epoch] if epoch < len(adv_schedule) else 0.30
                
                # 🚀 Phase 13: Adversarial "Prank" Generation (With Cooldown)
                if self.challenger_cooldown > 0:
                    is_adversarial = False
                    self.challenger_cooldown -= 1
                else:
                    is_adversarial = torch.rand(1).item() < adv_prob
                    
                current_pressure = self.guard.get_pressure()
                
                if is_adversarial:
                    self.challenger_cooldown = 3 # 🔥 Cooldown
                    # Challenger attempts to 'prank' the features to confuse the Trader
                    self.challenger.eval() # Generator doesn't update its own weights during this pass
                    batch_X_orig = batch_X.clone()
                    
                    # LOG: Feed back the "Sadistic Professor" behavior to the UI
                    if callback and epoch_batches % 50 == 0:
                        callback(f"😈 Professor Sádico: Aplicando modo aleatório (Pressão: {current_pressure:.2f})")
                        
                    # 🌋 Phase 19: Professor Breathe (Harmonic Oscillation)
                    import math
                    breathe_factor = 1.0 + 0.2 * math.sin(epoch_batches * 0.1)
                    dynamic_pressure = current_pressure * breathe_factor
                    
                    with torch.no_grad():
                        batch_X, _ = self.challenger(batch_X, pressure=dynamic_pressure, callback=callback if epoch_batches % 50 == 0 else None)
                 
                self.optimizer.zero_grad()
                if is_adversarial:
                    self.challenger_opt.zero_grad()

                x_dict = self._simulate_timeframes(batch_X)
                
                with torch.amp.autocast('cuda', enabled=(self.device.type == "cuda")):
                    logits, worker_logits, _, _, _, _ = self.model(x_dict)
                    loss_dict = self.criterion(logits, worker_logits, batch_y)
                    loss = loss_dict["total"]

                # 🛡️ Titanium Shield: Pre-Backward Protection against NaNs and Exceeding Bounds
                if torch.isnan(loss) or torch.isinf(loss) or loss.item() > 15.0:
                    val_str = f"NaN/Inf" if not torch.isfinite(loss) else f"{loss.item():.2f}"
                    if callback: callback(f"⚠️ Red Alert: Extreme Loss ({val_str}) detected in Shard. Titanium Shield activated. Skipping update.")
                    continue

                # Phase 14 v2: Populate Trauma Buffer with loss score
                current_loss = loss.item()
                if current_loss > self.trauma_threshold:
                    self._add_to_trauma_buffer(batch_X.detach(), batch_y.detach(), loss_val=current_loss)
                
                # Phase 14 v2: Inline Trauma Mix - inject old mistakes into current learning
                if epoch_batches % self.trauma_mix_interval == 0 and len(self.trauma_buffer_X) > 0:
                    trauma_X, trauma_Y = self._get_trauma_mix_batch(len(batch_X))
                    if trauma_X is not None:
                        trauma_dict = self._simulate_timeframes(trauma_X)
                        with torch.amp.autocast('cuda', enabled=(self.device.type == "cuda")):
                            t_logits, t_worker_logits, _, _, _, _ = self.model(trauma_dict)
                            t_loss_dict = self.criterion(t_logits, t_worker_logits, trauma_Y)
                            trauma_loss = t_loss_dict["total"] * 0.3  # Weight trauma replay at 30%
                        if not torch.isnan(trauma_loss):
                            self.scaler.scale(trauma_loss).backward()

                # 1. Trader Backward
                self.scaler.scale(loss).backward()
                
                # 2. Challenger Backward (if needed)
                if is_adversarial:
                    self.challenger.train()
                    for p in self.model.parameters(): p.requires_grad = False
                    
                    batch_X_adv, _ = self.challenger(batch_X_orig, pressure=current_pressure)
                    x_dict_adv = self._simulate_timeframes(batch_X_adv)
                    
                    with torch.amp.autocast('cuda', enabled=(self.device.type == "cuda")):
                        logits_adv, worker_logits_adv, _, _, _, _ = self.model(x_dict_adv)
                        loss_adv_dict = self.criterion(logits_adv, worker_logits_adv, batch_y)
                        challenger_loss = -loss_adv_dict["total"] 
                    
                    self.scaler.scale(challenger_loss).backward()
                    for p in self.model.parameters(): p.requires_grad = True

                # 3. Step Optimizers
                self.scaler.unscale_(self.optimizer)
                # 🛡️ Titanium Shield: Strict Maximum Gradient Normalization
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.scaler.step(self.optimizer)
                if is_adversarial:
                    self.scaler.unscale_(self.challenger_opt)
                    self.scaler.step(self.challenger_opt)
                    
                # 4. Final Scaler Update
                self.scaler.update()
                
                # 🚀 Phase 11 & 14: Multi-Class Metrics (Accuracy, Precision, AUC, F1)
                from sklearn.metrics import accuracy_score, precision_score, f1_score, roc_auc_score
                with torch.no_grad():
                    logits_np = logits.detach().cpu().numpy()
                    preds = logits_np.argmax(axis=-1)
                    true_y = batch_y.cpu().numpy()
                    
                    acc = accuracy_score(true_y, preds)
                    prec = precision_score(true_y, preds, average='macro', zero_division=0)
                    f1 = f1_score(true_y, preds, average='macro', zero_division=0)
                    
                    # 🧠 Phase 20: Accumulate for epoch-level AUC
                    probs = torch.softmax(logits.detach(), dim=-1).cpu().numpy()
                    epoch_all_preds.append(preds)
                    epoch_all_labels.append(true_y)
                    epoch_all_probs.append(probs)
                    
                    # Compute running AUC from accumulated data (use only present classes)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        try:
                            all_labels_so_far = np.concatenate(epoch_all_labels)
                            all_probs_so_far = np.concatenate(epoch_all_probs)
                            present_classes = np.unique(all_labels_so_far).astype(int).tolist()
                            if len(present_classes) == 3:
                                # Force strict normalization to exactly 1.0 to avoid scikit-learn float32 precision crash
                                norm_probs = all_probs_so_far / all_probs_so_far.sum(axis=1, keepdims=True)
                                auc = roc_auc_score(all_labels_so_far, norm_probs, multi_class='ovr', average='macro', labels=[0, 1, 2])
                            elif len(present_classes) == 2:
                                # Binary classification fallback
                                # Scikit-Learn expects a 1D array of probabilities for the positive class (the second class in present_classes)
                                pos_class = present_classes[1]
                                # Re-normalize probabilities just for the two present classes
                                binary_probs = all_probs_so_far[:, present_classes]
                                binary_probs = binary_probs / binary_probs.sum(axis=1, keepdims=True)
                                auc = roc_auc_score(all_labels_so_far, binary_probs[:, 1], labels=present_classes)
                            else:
                                auc = 0.5
                                
                            if np.isnan(auc):
                                auc = 0.5
                        except Exception as e:
                            logger.error(f"AUC Error B: {e}") 
                            auc = 0.5
                    
                    # Phase 13: "Panic Protection"
                    if is_adversarial and acc < 0.40:
                        self.guard.pressure = torch.tensor(max(0.1, self.guard.get_pressure() * 0.9), device=self.device)
                        if callback: callback(f"🛡️ Epoch {epoch+1} | Panic Protection: Reducing Professor pressure to {self.guard.get_pressure():.2f}")
                
                epoch_loss += loss.item()
                epoch_batches += 1
                total_batches += 1
                
                # Report every 5 batches for high-fidelity "Chart Dance"
                if epoch_batches % 5 == 0:
                    progress = int((epoch_batches / total_epoch_batches) * 100)
                    
                    # Log to console less frequently (every 100 batches)
                    if callback and epoch_batches % 100 == 0:
                        sw_weights = [f"{w:.2f}" for w in loss_dict["weights"]]
                        exp_votes = worker_logits.argmax(dim=-1).cpu().numpy()[0] 
                        v_map = {0: "L", 1: "S", 2: "W"}
                        thought_str = "-".join([v_map[v] for v in exp_votes])
                        callback(f"Epoch {epoch+1} | B {epoch_batches} | Loss: {loss.item():.4f} | Acc: {acc:.2f} | Prec: {prec:.2f} | AUC: {auc:.2f} | F1: {f1:.2f} | Swarm: {thought_str} | W: {sw_weights} | Pressure: {is_adversarial * current_pressure:.2f} | {progress}%")
                    
                    # 🧠 Phase 20: Micro-State Telemetry for "Opened Brain" Dashboard
                    if hasattr(self, '_micro_state_callback') and self._micro_state_callback:
                        try:
                            with torch.no_grad():
                                _, _, _, _, attn_w, confidence = self.model(x_dict, compute_telemetry=True)
                                w_probs = torch.softmax(worker_logits[0], dim=-1).cpu().tolist()
                                c_policy = torch.softmax(logits[0], dim=-1).cpu().tolist()
                                a_weights = attn_w[0].cpu().tolist() if attn_w is not None else None
                                conf_scores = confidence[0].squeeze(-1).cpu().tolist() if confidence is not None else None
                            self._micro_state_callback({
                                "worker_thoughts": w_probs,
                                "attention_weights": a_weights,
                                "consensus_weights": [float(w) for w in loss_dict["weights"]],
                                "coordinator_policy": c_policy,
                                "confidence_scores": conf_scores,
                                "pressure": float(is_adversarial * current_pressure),
                                "trauma_buffer_size": len(self.trauma_buffer_X),
                                "learning_rate": float(self.optimizer.param_groups[0]['lr']),
                            })
                        except Exception:
                            pass


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
                        v_X_cpu, v_y_cpu = v_data['X'], v_data['y']
                        
                        for j in range(0, len(v_X_cpu), self.batch_size):
                            v_X = v_X_cpu[j : j + self.batch_size].to(self.device)
                            v_y = v_y_cpu[j : j + self.batch_size].to(self.device).long()
                            
                            # 🚀 Robust Dimension Migration for Validation
                            shard_dim = v_X.shape[-1]
                            if shard_dim > self.input_dim:
                                v_X = v_X[:, :, :self.input_dim]
                            elif shard_dim < self.input_dim:
                                padding = torch.zeros(v_X.shape[0], v_X.shape[1], self.input_dim - shard_dim, dtype=v_X.dtype, device=v_X.device)
                                v_X = torch.cat([v_X, padding], dim=-1)
                                
                            # Ensure no NaNs or extreme values in validation
                            v_X = torch.nan_to_num(v_X)
                            v_X = torch.clamp(v_X, min=-20.0, max=20.0)
                            
                            v_x_dict = self._simulate_timeframes(v_X)
                            v_logits, v_worker_logits, _, _, _, _ = self.model(v_x_dict)
                            v_loss_dict = self.criterion(v_logits, v_worker_logits, v_y)
                            val_loss += v_loss_dict["total"].item()
                            val_batches += 1
                            
                if val_batches > 0:
                    avg_val_loss = val_loss / val_batches
                    if epoch_batches > 0:
                        avg_train_loss = epoch_loss / epoch_batches 
                        # Phase 13: Delay OverfittingGuard until masters basic data (Epoch 5+)
                        if epoch >= 4:
                            self.guard.update(avg_train_loss, avg_val_loss)
                        else:
                            if callback: callback(f"🧸 Epoch {epoch+1} | Professor Gentil: Pressão travada para masters inicial...")
                    
                    val_label = "Limpo (Base)" if epoch < 2 else "vs Professor Sádico"
                    if callback: callback(f"📉 Epoch {epoch+1} | Validation Loss ({val_label}): {avg_val_loss:.4f}")
                        
                    # Phase 14: ReduceLROnPlateau
                    self.scheduler.step(avg_val_loss)
                        
                    current_lr = self.optimizer.param_groups[0]['lr']
                    if callback and epoch > 0:
                        callback(f"🔧 Learning Rate: {current_lr:.6f} | Challenger Pressure: {self.guard.get_pressure():.2f}")

                    # 🌋 Phase 19: Plateau Breaker (Warm Restart & Professor Retreat)
                    if avg_val_loss < self.best_val_loss:
                        self.best_val_loss = avg_val_loss
                        self.epochs_without_improvement = 0
                    else:
                        self.epochs_without_improvement += 1
                        
                    if self.epochs_without_improvement >= self.patience_threshold:
                        if callback: callback(f"🌋 Plateau Detected! Breaking stagnancy with Warm Restart & Professor Retreat.")
                        # 1. Warm Restart: Reset LR to 75% of initial
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = self.learning_rate * 0.75
                        
                        # 2. Professor Retreat: Temporary Pressure Nerf (Reduce by 50%)
                        self.guard.pressure = torch.tensor(max(0.1, self.guard.get_pressure() * 0.5), device=self.device)
                        
                        self.epochs_without_improvement = 0 # Reset tracker

            # Phase 14: Execute Trauma Replay at the end of every epoch if accumulated
            trauma_mastered = 0
            trauma_size_before = len(self.trauma_buffer_X)
            if len(self.trauma_buffer_X) > 0:
                if callback: callback(f"🧠 Epoch {epoch+1} | Initiating Trauma Replay... Retraining on {len(self.trauma_buffer_X)} persistent memories.")
                trauma_size_before = len(self.trauma_buffer_X)
                self._train_trauma_replay(lambda m: callback(f"Epoch {epoch+1} | {m}"))
                trauma_mastered = trauma_size_before - len(self.trauma_buffer_X)
                if callback: callback(f"✅ Epoch {epoch+1} | Trauma Replay completed.")
            
            if epoch_batches > 0:
                avg_epoch_loss = epoch_loss / epoch_batches
                if callback: callback(f"✅ Epoch {epoch+1} Complete | Avg Loss: {avg_epoch_loss:.4f}")
                
                # 🏆 Best Model Checkpoint
                if avg_epoch_loss < best_loss:
                    best_loss = avg_epoch_loss
                    best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                    if callback: callback(f"🏆 Epoch {epoch+1} | New best model (Dataset)! Loss: {best_loss:.4f}")
                    # Phase 14: Immediate Auto-Save
                    save_path = self.checkpoint_path or "logs/models/active_model_v10.pth"
                    torch.save({'model_state_dict': self.model.state_dict()}, save_path)
                    if callback: callback(f"💾 Epoch {epoch+1} | Auto-saved best model to {os.path.basename(save_path)}")
                
                # 📊 Epoch Comparison: Compute aggregate metrics and record snapshot
                epoch_avg_acc = np.mean([accuracy_score(l, p) for l, p in zip(epoch_all_labels, epoch_all_preds)]) if epoch_all_preds else 0.0
                epoch_avg_prec = np.mean([precision_score(l, p, average='macro', zero_division=0) for l, p in zip(epoch_all_labels, epoch_all_preds)]) if epoch_all_preds else 0.0
                epoch_avg_f1 = np.mean([f1_score(l, p, average='macro', zero_division=0) for l, p in zip(epoch_all_labels, epoch_all_preds)]) if epoch_all_preds else 0.0
                epoch_auc = 0.5
                if epoch_all_probs and epoch_all_labels:
                    try:
                        all_l = np.concatenate(epoch_all_labels)
                        all_p = np.concatenate(epoch_all_probs)
                        present = np.unique(all_l).astype(int).tolist()
                        if len(present) >= 2:
                            norm_p = all_p / all_p.sum(axis=1, keepdims=True)
                            if len(present) == 3:
                                epoch_auc = roc_auc_score(all_l, norm_p, multi_class='ovr', average='macro', labels=[0,1,2])
                            else:
                                bp = norm_p[:, present]
                                bp = bp / bp.sum(axis=1, keepdims=True)
                                epoch_auc = roc_auc_score(all_l, bp[:, 1], labels=present)
                    except Exception:
                        epoch_auc = 0.5
                
                epoch_history.append({
                    'epoch': epoch + 1,
                    'loss': avg_epoch_loss,
                    'acc': epoch_avg_acc,
                    'prec': epoch_avg_prec,
                    'auc': epoch_auc,
                    'f1': epoch_avg_f1,
                    'trauma_size': len(self.trauma_buffer_X),
                    'mastered': trauma_mastered,
                    'lr': self.optimizer.param_groups[0]['lr'],
                })
                self._emit_epoch_comparison(epoch_history, callback)

        # Restore and save the best model weights
        if best_state:
            self.model.load_state_dict(best_state)
        
        save_path = self.checkpoint_path or "logs/models/active_model_v10.pth"
        if callback: callback(f"💾 Saving best checkpoint (Loss: {best_loss:.4f}) to {os.path.basename(save_path)}...")
        torch.save({'model_state_dict': self.model.state_dict()}, save_path)
        self.update_registry(best_loss, {dataset_name})
        
    def update_registry(self, loss: float, symbols: set):
        version_tag = "v10.0-Live"
        registry_entry = self.db.query(ModelVersion).filter(ModelVersion.tag == version_tag).first()
        
        if not registry_entry:
            registry_entry = ModelVersion(
                tag=version_tag,
                path="logs/models/active_model_v10.pth",
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
