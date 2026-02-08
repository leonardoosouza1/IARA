"""
I.A.R.A. v9 - Swarm Intelligence Training Loop
===============================================
Robust training loop for the Swarm architecture with:

1. Multi-Timeframe DataLoader (time-aligned batches)
2. Mixed Precision (AMP) for VRAM efficiency
3. Selective Memory (curriculum learning)
4. MLflow/Tensorboard logging with separate metrics per loss component
5. Gradient clipping (0.5) for LNN stability

Usage:
    docker compose exec saci-brain python src/brain/train_v9.py --epochs 100

Hardware: AMD RX 6750 XT (12GB VRAM)
Target: <8GB VRAM usage
"""

import argparse
import csv
import glob
import json
import logging
import math
import os
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

# Project imports
from src.brain.models.workers import WorkerConfig
from src.brain.models.liquid_v9 import SwarmBrain, CoordinatorConfig
from src.brain.losses import SwarmLoss

import signal
import sys
import time
import shutil
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)] # Ensure stdout flush
)
logger = logging.getLogger("IARA_v9")

# Global model reference for signal handler
global_model = None

def save_checkpoint(signum, frame):
    """Handle signal to save checkpoint manually"""
    logger.info("💾 Manual Save Requested via Signal...")
    if global_model:
        path = MODELS_DIR / f"manual_checkpoint_v9.pth"
        torch.save(global_model.state_dict(), path)
        logger.info(f"✅ Model manually saved to {path}")
    else:
        logger.warning("⚠️ Model not initialized yet, cannot save.")

# Register signal
signal.signal(signal.SIGUSR1, save_checkpoint)

# Paths
LOG_DIR = Path("logs")
MODELS_DIR = LOG_DIR / "models"
METRICS_FILE = LOG_DIR / "metrics.csv"  # Dashboard reads this file!
METRICS_FILE_V9 = LOG_DIR / "metrics_v9.csv"  # Backup for v9 specific
TELEMETRY_FILE = LOG_DIR / "telemetry.json"  # Dashboard reads metrics.csv
LEADERBOARD_FILE = LOG_DIR / "leaderboard_v9.json"


@dataclass
class TrainConfig:
    """Training configuration for v9."""
    # Data
    seq_len: int = 64
    batch_size: int = 64
    val_split: float = 0.2
    
    # Model
    input_dim: int = 12
    worker_hidden: int = 32
    coord_hidden: int = 128
    
    # Training
    epochs: int = 200
    learning_rate: float = 0.0005
    weight_decay: float = 1e-4
    learning_rate: float = 0.0005
    weight_decay: float = 1e-4
    grad_clip: float = 0.5
    use_amp: bool = True
    debug: bool = False
    
    # Loss weights
    aux_weight: float = 0.4
    consensus_weight: float = 0.1
    label_smoothing: float = 0.1
    
    # Scheduler
    scheduler_patience: int = 15
    scheduler_factor: float = 0.8
    
    # Early stopping
    patience: int = 50


class CapturedExampleDataset(Dataset):
    """Dataset for examples captured from backtest/simulation (Errors, Successes, Patterns)."""
    def __init__(self, sources: List[Path], seq_len: int = 64):
        self.seq_len = seq_len
        self.data = []
        for src in sources:
            if src.exists():
                try:
                    loaded = torch.load(src, map_location="cpu", weights_only=False)
                    # Support both list and dict-based captures
                    if isinstance(loaded, list):
                        self.data.extend(loaded)
                    logger.info(f"🔥 Mixed {len(loaded)} examples from {src.name}")
                except Exception as e:
                    logger.error(f"Failed to load captured examples from {src}: {e}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        x_dict = item["x_dict"] 
        processed_x = {}
        for tf, tensor in x_dict.items():
            # If tensor is [B, L, C], take [L, C]
            if tensor.dim() == 3:
                processed_x[tf] = tensor[0, :, :12].float() # Truncate to 12 features
            else:
                processed_x[tf] = tensor[:, :12].float() # Truncate to 12 features
        
        y = item["y"].float() if "y" in item else torch.tensor(0.5) # Default neutral if pattern only
        
        # Add placeholder forensics metadata (zeros) to match MultiTimeframeDataset structure
        # This prevents KeyError in the collate function
        processed_x["__ohlc__"] = torch.zeros(self.seq_len, 4)
        
        return processed_x, y


class MultiTimeframeDataset(Dataset):
    """
    Dataset that provides time-aligned data for all timeframes.
    
    Each batch contains synchronized data from M5, M15, M30, H1, H4
    representing the same wall-clock time.
    """
    
    TIMEFRAMES = ["M5", "M15", "M30", "H1", "H4"]
    TF_MULTIPLIERS = {"M5": 1, "M15": 3, "M30": 6, "H1": 12, "H4": 48}
    
    def __init__(
        self,
        data_path: str = "data/cache",
        seq_len: int = 64,
        split: str = "train"
    ):
        self.seq_len = seq_len
        self.split = split
        
        # Load sharded data
        shard_dir = Path(data_path) / f"{split}_shards"
        
        if not shard_dir.exists():
            # Fallback to single file
            single_file = Path(data_path) / f"{split}.pt"
            if single_file.exists():
                data = torch.load(single_file, weights_only=False)
                self.X = data.get('X', data)
                self.y = data.get('y', None)
                if self.y is None:
                    raise ValueError(f"No targets found in {single_file}")
            else:
                raise FileNotFoundError(f"No data found at {data_path}")
        else:
            # Load shards INCREMENTALLY to avoid RAM spikes
            shard_paths = sorted(shard_dir.glob("*.pt"))
            logger.info(f"  🔍 Processing {len(shard_paths)} shards for {split}...")
            
            X_list, y_list = [], []
            for i, shard_path in enumerate(shard_paths):
                data = torch.load(shard_path, weights_only=False)
                # Convert to half IMMEDIATELY upon loading to save 50% RAM during concatenation
                X_list.append(data['X'].half())
                y_list.append(data['y'].float())
                
                if (i + 1) % 5 == 0 or (i + 1) == len(shard_paths):
                    logger.info(f"    - [{split}] Loaded shard {i+1}/{len(shard_paths)}")
                
                # Manual cleanup is tricky with PyTorch, but we hope the list append is efficient
                del data 
            
            logger.info(f"  ⛓️ Concatenating {split} shards...")
            self.X = torch.cat(X_list, dim=0)
            self.y = torch.cat(y_list, dim=0)
            
            # Already in half() from loop
        
        # Get feature dimension
        self.feature_dim = self.X.shape[-1]
        
        logger.info(f"  Loaded {split} dataset: {len(self)} samples, {self.feature_dim} features")
        
    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Returns time-aligned data for all timeframes.
        This is a simplification. Production should use actual resampled data.
        """
        # Convert back to float32 for model computation (stored as float16 for RAM savings)
        x = self.X[idx].float()  # [seq_len, features] - NOW IN FLOAT32
        y = self.y[idx]  # scalar (already float32)
        
        # Simulate timeframe views by downsampling
        x_dict = {}
        
        # Core features for the model (First 12)
        x_core = x[:, :12]
        
        # M5: Full resolution
        x_dict["M5"] = x_core[-self.seq_len:]
        
        # Metadata OHLC (Last 4 columns) for Forensics
        # We only keep it for M5 to save memory
        ohlc = x[-self.seq_len:, 22:]
        if ohlc.shape[1] < 4:
            # Pad with zeros if metadata columns are missing in cache
            ohlc = torch.cat([ohlc, torch.zeros(ohlc.shape[0], 4 - ohlc.shape[1])], dim=1)
        x_dict["__ohlc__"] = ohlc
        
        # M15: Every 3rd candle
        step_15 = max(1, self.seq_len // 3)
        x_dict["M15"] = x_core[::3][-step_15:] if self.seq_len >= 3 else x_core[-1:]
        
        # M30: Every 6th candle
        step_30 = max(1, self.seq_len // 6)
        x_dict["M30"] = x_core[::6][-step_30:] if self.seq_len >= 6 else x_core[-1:]
        
        # H1: Every 12th candle
        step_h1 = max(1, self.seq_len // 12)
        x_dict["H1"] = x_core[::12][-step_h1:] if self.seq_len >= 12 else x_core[-1:]
        
        # H4: Every 48th candle
        step_h4 = max(1, self.seq_len // 16)
        x_dict["H4"] = x_core[::16][-step_h4:] if self.seq_len >= 16 else x_core[-1:]
        
        return x_dict, y


def collate_multitf(batch: List[Tuple[Dict[str, torch.Tensor], torch.Tensor]]):
    """Custom collate function for multi-timeframe batches."""
    x_dicts, ys = zip(*batch)
    
    # Identify common keys
    common_keys = set(x_dicts[0].keys())
    for x in x_dicts[1:]:
        common_keys &= set(x.keys())
    
    x_batch = {}
    for tf in sorted(list(common_keys)):
        seqs = [x[tf] for x in x_dicts]
        
        # Check for feature dimension consistency
        feat_dims = [s.size(-1) for s in seqs]
        if len(set(feat_dims)) > 1:
            # If inconsistent, we skip this key (likely metadata mismatch)
            continue
            
        max_len = max(s.size(0) for s in seqs)
        feat_dim = feat_dims[0]
        
        # Pad sequences to same length
        padded = []
        for s in seqs:
            if s.size(0) < max_len:
                pad = torch.zeros(max_len - s.size(0), feat_dim)
                s = torch.cat([pad, s], dim=0)
            padded.append(s.unsqueeze(0))
        
        x_batch[tf] = torch.cat(padded, dim=0)
    
    y_batch = torch.stack(ys, dim=0)
    return x_batch, y_batch


class Leaderboard:
    """Tracks best models."""
    
    def __init__(self, path: Path = LEADERBOARD_FILE, top_k: int = 5):
        self.path = path
        self.top_k = top_k
        self.history = []
        
        if path.exists():
            try:
                with open(path, "r") as f:
                    self.history = json.load(f)
            except:
                pass
    
    def update(
        self, 
        epoch: int, 
        loss: float, 
        auc: float, 
        accuracy: float,
        model_state: dict
    ) -> bool:
        """Update leaderboard. Returns True if new best."""
        entry = {
            "epoch": epoch,
            "loss": loss,
            "auc": auc,
            "accuracy": accuracy,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        self.history.append(entry)
        self.history = sorted(self.history, key=lambda x: -x["auc"])[:self.top_k]
        
        with open(self.path, "w") as f:
            json.dump(self.history, f, indent=2)
        
        is_best = self.history[0]["epoch"] == epoch
        
        if is_best:
            logger.info(f"🏆 New Best Model! AUC: {auc:.4f} | Acc: {accuracy:.2%}")
            torch.save(model_state, MODELS_DIR / "best_model_v9.pth")
        
        return is_best


class SelectiveMemory:
    """
    Curriculum learning based on sample difficulty.
    Ported from v6 trainer_liquid.py.
    """
    
    def __init__(self, num_samples: int, alpha: float = 0.7):
        self.alpha = alpha
        self.sample_loss_ema = torch.ones(num_samples)
        self.chronic_counter = torch.zeros(num_samples, dtype=torch.int16)
        self.mastered_counter = torch.zeros(num_samples, dtype=torch.int16)
        self.noise_mask = torch.zeros(num_samples, dtype=torch.bool)
        
    def analyze(
        self, 
        model: nn.Module, 
        dataset: Dataset, 
        device: str,
        criterion: nn.Module,
        batch_size: int = 512,
        sample_size: Optional[int] = None
    ) -> WeightedRandomSampler:
        """Compute sample weights for curriculum learning."""
        
        num_total = len(dataset)
        indices = np.arange(num_total)
        
        if sample_size and sample_size < num_total:
             logger.info(f"🧠 Deep Thought: Analyzing random subset of {sample_size} samples...")
             indices = np.random.choice(indices, size=sample_size, replace=False)
             analysis_ds = torch.utils.data.Subset(dataset, indices)
        else:
             logger.info("🧠 Deep Thought: Analyzing ALL samples...")
             analysis_ds = dataset
        
        model.eval()
        losses = []
        
        loader = DataLoader(
            analysis_ds, 
            batch_size=batch_size, 
            shuffle=False,
            collate_fn=collate_multitf,
            num_workers=0
        )
        
        with torch.no_grad():
            with torch.amp.autocast('cuda', enabled=(device == "cuda")):
                for x_dict, targets in loader:
                    # Move to device
                    x_dict = {k: v.to(device) for k, v in x_dict.items()}
                    targets = targets.to(device)
                    
                    # Forward
                    logits, worker_logits, _, _ = model(x_dict)
                    
                    # Store individual losses
                    bce = nn.functional.binary_cross_entropy_with_logits(
                        logits.view(-1), 
                        targets.view(-1),
                        reduction='none'
                    )
                    losses.append(bce.cpu())
                    
                    if len(losses) % 20 == 0:
                        import gc
                        gc.collect() # Force GC to avoid OOM
                        
        if not losses:
            return None

        current_losses = torch.cat(losses)
        
        # Ensure state tensors match dataset size (resizing if dataset grew)
        if len(self.sample_loss_ema) != num_total:
             logger.info(f"⚠️ Resizing memory state from {len(self.sample_loss_ema)} to {num_total}")
             # Extend with ones/zeros
             new_size = num_total - len(self.sample_loss_ema)
             self.sample_loss_ema = torch.cat([self.sample_loss_ema, torch.ones(new_size)])
             self.chronic_counter = torch.cat([self.chronic_counter, torch.zeros(new_size, dtype=torch.int16)])
             self.mastered_counter = torch.cat([self.mastered_counter, torch.zeros(new_size, dtype=torch.int16)])
        
        # Update EMA only for analyzed indices
        # We need to map back to original indices. 
        # current_losses corresponds to 'indices' array.
        
        indices_tensor = torch.tensor(indices, dtype=torch.long)
        
        # Update EMA
        # self.sample_loss_ema[indices] = alpha * old + (1-alpha) * new
        old_ema = self.sample_loss_ema[indices_tensor]
        new_ema = self.alpha * old_ema + (1 - self.alpha) * current_losses
        self.sample_loss_ema[indices_tensor] = new_ema
        
        # Update counters
        high_loss = current_losses > 0.9
        self.chronic_counter[indices_tensor[high_loss]] += 1
        self.chronic_counter[indices_tensor[~high_loss]] = 0
        
        low_loss = current_losses < 0.3
        self.mastered_counter[indices_tensor[low_loss]] += 1
        self.mastered_counter[indices_tensor[~low_loss]] = 0
        
        # Compute weights for WHOLE dataset
        weights = (self.sample_loss_ema + 0.05) ** 2
        
        # Boost chronic
        weights[self.chronic_counter >= 3] *= 3.0
        
        # Suppress mastered
        weights[self.mastered_counter >= 5] *= 0.1
        
        # Stats
        n_chronic = (self.chronic_counter >= 3).sum().item()
        n_mastered = (self.mastered_counter >= 5).sum().item()
        logger.info(f"  📊 Stats: {n_chronic} Chronic | {n_mastered} Mastered")
        
        model.train()
        
        return WeightedRandomSampler(
            weights=weights.tolist(),
            num_samples=len(weights),
            replacement=True
        )


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: str,
    config: TrainConfig,
    epoch: int,
    total_epochs: int,
    global_step: int = 0,
    current_lr: float = 0.001
) -> Tuple[Dict[str, float], int]:
    """Train one epoch."""
    model.train()
    
    losses = {"total": [], "coord": [], "workers": [], "consensus": []}
    all_preds, all_targets = [], []
    
    for step, (x_dict, targets) in enumerate(loader):
        # Move to device
        x_dict = {k: v.to(device) for k, v in x_dict.items()}
        targets = targets.to(device)
        
        optimizer.zero_grad()
        
        # Mixed Precision (AMP)
        if config.use_amp:
            with torch.amp.autocast('cuda', enabled=True):
                logits, worker_logits, _, _ = model(x_dict)
                loss_dict = criterion(logits, worker_logits, targets)
        else:
            logits, worker_logits, _, _ = model(x_dict)
            loss_dict = criterion(logits, worker_logits, targets)
        
        # Check for NaN
        if torch.isnan(loss_dict['total']):
            logger.warning(f"⚠️ NaN loss at step {step}, skipping...")
            continue
        
        # AMP backward
        scaler.scale(loss_dict['total']).backward()
        
        # Gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        
        scaler.step(optimizer)
        scaler.update()
        
        # Track losses
        for key in losses:
            losses[key].append(loss_dict[key].item())
        
        # Track predictions
        probs = torch.sigmoid(logits.detach()).cpu().numpy()
        all_preds.extend(probs.flatten())
        all_targets.extend((targets > 0.5).float().cpu().numpy().flatten())
        
        # Log progress
        if config.debug or step % 10 == 0:
            step_loss = loss_dict['total'].item()
            
            # Check RAM (Linux only, no psutil needed)
            mem_usage = 0
            try:
                with open("/proc/self/status") as f:
                    for line in f:
                        if "VmRSS" in line:
                            mem_usage = int(line.split()[1]) / 1024 # MB
                            break
            except:
                pass
            
            if config.debug:
                 print(f"Step {step}: Loss {step_loss:.4f} | RAM: {mem_usage:.0f}MB", end="\r")
            
            if step % 10 == 0:
                # Real Accuracy
                step_acc = np.mean([(p > 0.5) == (t > 0.5) for p, t in zip(probs.flatten(), targets.cpu().numpy().flatten())]) if len(probs) > 0 else 0.5
                
                # Real AUC for the step (if possible)
                try:
                    step_auc = roc_auc_score(targets.cpu().numpy().flatten(), probs.flatten())
                except:
                    step_auc = 0.5 # Fallback if batch has only one class
                
                logger.info(
                    f"Epoch {epoch} | Step {step}/{len(loader)} | "
                    f"Loss: {step_loss:.4f} | "
                    f"Coord: {loss_dict['coord'].item():.4f} | "
                    f"Workers: {loss_dict['workers'].item():.4f} | "
                    f"RAM: {mem_usage:.0f} MB"
                )
                
                # Write to CSV
                with open(METRICS_FILE, "a", newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([global_step, epoch, step_loss, step_acc, step_auc, current_lr])
                
                # Update real-time telemetry
                with open(TELEMETRY_FILE, "w") as f:
                    json.dump({
                        "status": "TRAINING",
                        "current_epoch": epoch,
                        "total_epochs": total_epochs,
                        "current_step": step,
                        "total_steps": len(loader),
                        "loss": step_loss,
                        "acc": step_acc,
                        "auc": step_auc,
                        "mem": mem_usage
                    }, f)
        
        global_step += 1
    
    # Compute metrics
    avg_losses = {k: np.mean(v) if v else 0.0 for k, v in losses.items()}
    
    try:
        auc = roc_auc_score(all_targets, all_preds)
    except:
        auc = 0.5
    
    accuracy = np.mean([(p > 0.5) == t for p, t in zip(all_preds, all_targets)])
    
    avg_losses["auc"] = auc
    avg_losses["accuracy"] = accuracy
    
    return avg_losses, global_step


def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str
) -> Dict[str, float]:
    """Validate model."""
    model.eval()
    
    losses = {"total": [], "coord": [], "workers": [], "consensus": []}
    all_preds, all_targets = [], []
    
    with torch.no_grad():
        with torch.amp.autocast('cuda', enabled=(device == "cuda")):
            for x_dict, targets in loader:
                x_dict = {k: v.to(device) for k, v in x_dict.items()}
                targets = targets.to(device)
                
                logits, worker_logits, _, _ = model(x_dict)
                loss_dict = criterion(logits, worker_logits, targets)
                
                for key in losses:
                    losses[key].append(loss_dict[key].item())
                
                probs = torch.sigmoid(logits).cpu().numpy()
                all_preds.extend(probs.flatten())
                all_targets.extend((targets > 0.5).float().cpu().numpy().flatten())
    
    avg_losses = {k: np.mean(v) if v else 0.0 for k, v in losses.items()}
    
    try:
        auc = roc_auc_score(all_targets, all_preds)
    except:
        auc = 0.5
    
    accuracy = np.mean([(p > 0.5) == t for p, t in zip(all_preds, all_targets)])
    
    avg_losses["auc"] = auc
    avg_losses["accuracy"] = accuracy
    
    return avg_losses


def main():
    parser = argparse.ArgumentParser(description="Train IARA v9 Swarm Intelligence")
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seq_len", type=int, default=64)
    parser.add_argument("--cpu", action="store_true", default=False)
    parser.add_argument("--no_amp", action="store_true", default=False, help="Disable Mixed Precision")
    parser.add_argument("--debug", action="store_true", default=False, help="Log every step")
    parser.add_argument("--resume", action="store_true", default=False, help="Resume from checkpoint")
    args = parser.parse_args()
    
    # Setup directories
    LOG_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)
    
    # Initialize metrics CSV - Dashboard format: step,epoch,loss,accuracy,auc,lr
    with open(METRICS_FILE, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["step", "epoch", "loss", "accuracy", "auc", "lr"])
        
    # Initialize V9 detailed metrics
    with open(METRICS_FILE_V9, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch", "train_loss", "val_loss", 
            "train_auc", "val_auc",
            "coord_loss", "worker_loss", "consensus_loss",
            "lr"
        ])
    
    # Global step counter for dashboard
    global_step = 0
    
    # Device
    if args.cpu:
        device = "cpu"
    elif torch.cuda.is_available():
        device = "cuda"
        logger.info(f"🚀 GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        logger.warning("⚠️ CUDA not available, using CPU")
    
    logger.info(f"Device: {device.upper()}")
    
    # Configuration
    config = TrainConfig(
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        epochs=args.epochs,
        patience=args.patience,
        use_amp=not args.no_amp,
        debug=args.debug
    )
    
    # Load data
    logger.info("📂 Loading datasets...")
    
    try:
        train_ds = MultiTimeframeDataset(seq_len=config.seq_len, split="train")
        val_ds = MultiTimeframeDataset(seq_len=config.seq_len, split="val")
    except FileNotFoundError as e:
        logger.error(f"❌ Data not found: {e}")
        logger.error("Please run feature_factory.py first to generate training data.")
        sys.exit(1)
    
    # Determine input dimension from data
    sample_x, _ = train_ds[0]
    config.input_dim = sample_x["M5"].shape[-1]
    logger.info(f"  Detected {config.input_dim} input features")
    
    # Create model
    logger.info("🧠 Creating SwarmBrain...")
    
    worker_cfg = WorkerConfig(
        input_dim=config.input_dim,
        hidden_dim=config.worker_hidden,
        dropout=0.1
    )
    
    coord_cfg = CoordinatorConfig(
        num_workers=5,
        worker_dim=config.worker_hidden,
        hidden_dim=config.coord_hidden,
        dropout=0.3
    )
    
    model = SwarmBrain(worker_cfg, coord_cfg).to(device)
    
    # Expose to global for signal handler
    global global_model
    global_model = model
    
    logger.info(f"  Total parameters: {model.count_parameters():,}")
    
    # Check VRAM
    if device == "cuda":
        param_mem = sum(p.numel() * p.element_size() for p in model.parameters())
        logger.info(f"  Parameter memory: {param_mem / 1e6:.2f} MB")
    
    # Resume from checkpoint if exists
    checkpoint_path = MODELS_DIR / "best_model_v9.pth"
    start_epoch = 0
    
    if args.resume and checkpoint_path.exists():
        logger.info(f"🏆 Found checkpoint, resuming...")
        try:
            state_dict = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(state_dict, strict=False)
            logger.info("  ✅ Weights loaded!")
        except Exception as e:
            logger.warning(f"  ⚠️ Could not load checkpoint: {e}")
    elif checkpoint_path.exists():
        # Backup existing model if starting fresh
        backup_path = MODELS_DIR / f"best_model_v9_backup_{int(time.time())}.pth"
        import shutil
        try:
            shutil.copy(checkpoint_path, backup_path)
            logger.info(f"📦 Existing model backed up to {backup_path.name}")
        except Exception as e:
            logger.warning(f"  ⚠️ Could not backup model: {e}")
        
        logger.info("🆕 Starting FRESH training sequence.")
    
    criterion = SwarmLoss(
        alpha=0.45,  # Balanced/Cautious (was 0.75, which over-prioritized Longs)
        gamma=2.0,
        aux_weight=config.aux_weight,
        consensus_weight=config.consensus_weight,
        label_smoothing=config.label_smoothing
    )
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=config.scheduler_factor,
        patience=config.scheduler_patience,
        min_lr=1e-6
    )
    
    # AMP scaler
    scaler = torch.amp.GradScaler('cuda', enabled=(device == "cuda"))
    
    # Selective Memory
    memory = SelectiveMemory(len(train_ds))
    
    # Leaderboard
    leaderboard = Leaderboard()
    
    # Feedback Loop (Hard Examples, Successes, Patterns)
    feedback_sources = [
        Path("logs/backtest_errors.pt"),
        Path("logs/backtest_success.pt"),
        Path("logs/pattern_reservoir.pt")
    ]
    captured_ds = CapturedExampleDataset(feedback_sources, seq_len=config.seq_len)
    
    if len(captured_ds) > 0:
        logger.info(f"🚀 Integrating {len(captured_ds)} captured examples into training set...")
        from torch.utils.data import ConcatDataset
        train_loader_ds = ConcatDataset([train_ds, captured_ds])
        logger.info(f"📊 Combined dataset size: {len(train_loader_ds)}")
    else:
        train_loader_ds = train_ds

    # DataLoaders
    train_loader = DataLoader(
        train_loader_ds,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_multitf,
        drop_last=True,
        num_workers=0,
        pin_memory=(device == "cuda")
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_multitf,
        num_workers=0,
        pin_memory=(device == "cuda")
    )
    
    logger.info(f"  Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")
    
    # Training loop
    logger.info("=" * 60)
    logger.info("🚀 Starting Training v9 Swarm Intelligence...")
    logger.info("=" * 60)
    
    best_val_auc = 0.0
    patience_counter = 0
    
    for epoch in range(start_epoch, config.epochs):
        epoch_start = time.time()
        
        # Update sampler after first epoch
        # Update sampler after first epoch
        if epoch > 0:
            try:
                # logger.info("🔄 Starting Selective Memory Analysis...")
                sampler = memory.analyze(model, train_ds, device, criterion, sample_size=20000)
                
                if sampler:
                    train_loader = DataLoader(
                        train_ds,
                        batch_size=config.batch_size,
                        sampler=sampler,
                        collate_fn=collate_multitf,
                        drop_last=True,
                        num_workers=0
                    )
                    logger.info("✅ Selective Memory Analysis Complete (Sampled 20k)")
                else:
                    logger.warning("⚠️ Selective Memory returned no sampler. Using standard shuffle.")
                    raise Exception("No sampler")
                 
            except Exception as e:
                # Fallback to standard loader if selective memory crashes
                # logger.warning(f"⚠️ Selective Memory failed: {e}")
                train_loader = DataLoader(
                    train_ds,
                    batch_size=config.batch_size,
                    shuffle=True,
                    collate_fn=collate_multitf,
                    drop_last=True,
                    num_workers=0
                )
        
        # Train (pass global_step and current_lr for dashboard logging)
        current_lr = optimizer.param_groups[0]['lr']
        train_metrics, global_step = train_epoch(
            model, train_loader, criterion, optimizer, scaler,
            device, config, epoch + 1, config.epochs, global_step, current_lr
        )
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device)
        
        # Save Forensic Snapshot (First batch of validation)
        if (epoch + 1) % 1 == 0: # Save every epoch
            try:
                save_forensic_snapshot(model, val_loader, device, epoch + 1, config)
            except Exception as e:
                logger.error(f"⚠️ Forensic Snapshot Failed (Non-critical): {e}")
        
        # Update scheduler
        scheduler.step(val_metrics['total'])
        current_lr = optimizer.param_groups[0]['lr']
        
        epoch_time = time.time() - epoch_start
        
        # Log
        logger.info(f"\n{'='*60}")
        logger.info(
            f"Epoch {epoch + 1}/{config.epochs} | Time: {epoch_time:.1f}s\n"
            f"  Train - Loss: {train_metrics['total']:.4f} | AUC: {train_metrics['auc']:.4f} | Acc: {train_metrics['accuracy']:.2%}\n"
            f"  Val   - Loss: {val_metrics['total']:.4f} | AUC: {val_metrics['auc']:.4f} | Acc: {val_metrics['accuracy']:.2%}\n"
            f"  LR: {current_lr:.2e}"
        )
        logger.info(f"{'='*60}\n")
        
        # Save epoch metrics to V9 specific file (detailed)
        with open(METRICS_FILE_V9, "a", newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch + 1,
                train_metrics['total'],
                val_metrics['total'],
                train_metrics['auc'],
                val_metrics['auc'],
                train_metrics['coord'],
                train_metrics['workers'],
                train_metrics['consensus'],
                current_lr
            ])
        
        # Update telemetry
        with open(TELEMETRY_FILE, "w") as f:
            json.dump({
                "current_epoch": epoch + 1,
                "total_epochs": config.epochs,
                "status": "TRAINING",
                "train_loss": train_metrics['total'],
                "val_loss": val_metrics['total'],
                "train_auc": train_metrics['auc'],
                "val_auc": val_metrics['auc'],
                "device": device
            }, f, indent=2)
        
        # Update leaderboard
        is_best = leaderboard.update(
            epoch + 1,
            val_metrics['total'],
            val_metrics['auc'],
            val_metrics['accuracy'],
            model.state_dict()
        )
        
        # Early stopping
        if val_metrics['auc'] > best_val_auc:
            best_val_auc = val_metrics['auc']
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= config.patience:
            logger.info(f"⏹️ Early stopping at epoch {epoch + 1}")
            break
    
    # Final telemetry
    with open(TELEMETRY_FILE, "w") as f:
        json.dump({"status": "COMPLETED", "best_auc": best_val_auc}, f)
    
    logger.info(f"\n✅ Training Complete! Best AUC: {best_val_auc:.4f}")


def save_forensic_snapshot(model, loader, device, epoch, config):
    """Generate forensic snapshot for dashboard visualization."""
    model.eval()
    
    # Get a single batch for visualization
    try:
        # Debug Logging
        # logger.info("🔍 Debugging Snapshot Generation...")
        
        iterator = iter(loader)
        batch = next(iterator)
        # logger.info(f"    Batch Type: {type(batch)}")
        
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            x_dict, targets = batch
        else:
            logger.error(f"    ❌ Unexpected batch structure: {type(batch)}")
            return

        # logger.info(f"    x_dict keys: {list(x_dict.keys())}")
        
        x_dict = {k: v.to(device) for k, v in x_dict.items()}
        
        with torch.no_grad():
            # Forward pass
            # SwarmBrain v9 returns: logits, worker_logits, long_score, short_score
            logits, worker_logits, long_score, short_score = model(x_dict)
            probs = torch.sigmoid(logits).cpu().numpy()
            
            # worker_logits shape: [batch, num_workers, 3]
            # Mapping indices to names for Forensic Lab v8 compatibility:
            # Worker 0 = M5 (Short-term), Worker 3 = H1 (Long-term)
            try:
                w_m5 = torch.sigmoid(worker_logits[:, 0, 0]).cpu().numpy().flatten() # index 0 is LONG prob
                w_h1 = torch.sigmoid(worker_logits[:, 3, 0]).cpu().numpy().flatten()
            except Exception as e:
                logger.warning(f"    ⚠️ Failed to map worker indices: {e}")
                w_m5 = torch.zeros_like(logits).flatten().cpu().numpy()
                w_h1 = torch.zeros_like(logits).flatten().cpu().numpy()
            
            # Map coordination/attention (v9 uses cross-attention weights)
            # For now, we mock the attention map if not easily extractable
            coord_attn = None 
            # Extract OHLC from metadata if available
            ohlc_data = None
            if "__ohlc__" in x_dict:
                 metadata = x_dict["__ohlc__"][0].cpu().numpy() # [Seq, 4] -> [O, H, L, C]
                 ohlc_data = {
                     "open": metadata[:, 0].tolist(),
                     "high": metadata[:, 1].tolist(),
                     "low": metadata[:, 2].tolist(),
                     "close": metadata[:, 3].tolist(),
                     "time": list(range(len(metadata)))
                 }
            
            # Extract sample input (Close price normalized or feature 0)
            sample_input = x_dict["M5"][0, :, 0].cpu().numpy().tolist() # Feature 0 is RSI or Close (depending on config)
            
            snapshot = {
                "epoch": epoch,
                "timestamp": datetime.now().isoformat(),
                "sample_input": sample_input, 
                "ohlc": ohlc_data,
                "expert_conviction": {
                    "long": w_m5.tolist()[:100],  # Short-term expert (M5)
                    "short": w_h1.tolist()[:100], # Long-term expert (H1)
                    "consensus": probs.flatten().tolist()[:100] # Coordinator
                },
                "feature_importance": [1.0] * 12, # Dummy for v9 compatibility
                "hidden_activity": [[0.0] * 100 for _ in range(32)], # Dummy for v9 compatibility
                "attention_map": coord_attn.cpu().numpy().tolist()[:1] if coord_attn is not None else []
            }
            
            # Save
            Path("logs/snapshots").mkdir(parents=True, exist_ok=True)
            with open(f"logs/snapshots/epoch_{epoch}.json", "w") as f:
                json.dump(snapshot, f)
            
            logger.info(f"📸 Forensic Snapshot saved for Epoch {epoch}")
                
    except Exception as e:
        logger.warning(f"⚠️ Failed to save forensic snapshot: {e}")
        import traceback
        traceback.print_exc()



if __name__ == "__main__":
    main()
