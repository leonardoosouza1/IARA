"""
Trainer for I.A.R.A. v6.1 - Fractal Attention-CfC (High Volume + AMP + Optimization)
================================================================================
Features:
- Multi-asset training (Majors)
- Sequence length 256
- Mixed Precision (AMP) for 2x speedup
- Stride Optimization (Stride 8 = 40 min shift) for 8x speedup
- Gradient Accumulation
- ReduceLROnPlateau Scheduler
"""

import argparse
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import polars as pl
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

from src.model.iara_liquid import IARA, LiquidConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Paths
DATA_PROCESSED_DIR = Path(__file__).parent.parent.parent / "data" / "processed"
CHECKPOINT_DIR = Path(__file__).parent.parent.parent / "checkpoints"


@dataclass
class TrainerConfig:
    """Training configuration v6.1 (High Volume + AMP + Opt)."""
    
    # Data
    seq_len: int = 256
    stride: int = 8  # Optimized: 8x faster, 97% overlap preserved
    batch_size: int = 768  # Tuned for AMP
    accumulation_steps: int = 2  # Effective batch = 1536
    num_workers: int = 4
    val_split: float = 0.2
    
    # Model
    input_dim: int = 15
    hidden_dim: int = 512
    num_heads: int = 4
    dropout: float = 0.5
    max_seq_len: int = 256
    
    # Training
    epochs: int = 200
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    grad_clip: float = 0.5
    resume: bool = False  # Resume flag
    
    # Label Smoothing
    label_smooth: float = 0.1
    
    # Augmentation
    noise_std: float = 0.02
    scale_min: float = 0.98
    scale_max: float = 1.02
    
    # Early Stopping
    patience: int = 40
    
    # Checkpointing
    checkpoint_dir: Path = CHECKPOINT_DIR


class MultiAssetDataset(Dataset):
    """Multi-asset dataset with support for all processed tickers."""
    
    def __init__(
        self,
        data_dir: Path = DATA_PROCESSED_DIR,
        seq_len: int = 256,
        stride: int = 1,
        tickers: Optional[List[str]] = None,
        target_col: str = "close_log_ret",
        augment: bool = False,
        noise_std: float = 0.02,
        scale_range: Tuple[float, float] = (0.98, 1.02),
        label_smooth: float = 0.0
    ):
        self.data_dir = Path(data_dir)
        self.seq_len = seq_len
        self.stride = stride
        self.target_col = target_col
        self.augment = augment
        self.noise_std = noise_std
        self.scale_range = scale_range
        self.label_smooth = label_smooth
        self.feature_cols = None
        self.samples = self._load_data(tickers)
        self.indices = self._compute_indices()
        
        aug_str = " (augmented)" if augment else ""
        smooth_str = f", smooth={label_smooth}" if label_smooth > 0 else ""
        n_tickers = len(self.samples)
        logger.info(f"Dataset{aug_str}{smooth_str}: {len(self.indices)} samples from {n_tickers} tickers, {len(self.feature_cols) if self.feature_cols else 0} features")
    
    def _load_data(self, tickers: Optional[List[str]]) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, str]]:
        parquet_files = list(self.data_dir.glob("*.parquet"))
        if tickers:
            parquet_files = [f for f in parquet_files if f.stem in tickers]
        
        data_list = []
        self.feature_cols = None
        
        for file_path in parquet_files:
            try:
                df = pl.read_parquet(file_path)
                ticker = file_path.stem
                
                if self.target_col not in df.columns:
                    continue
                
                if self.feature_cols is None:
                    self.feature_cols = sorted([c for c in df.columns if c.endswith("_norm")])
                    logger.info(f"Features: {self.feature_cols}")
                
                available = [c for c in self.feature_cols if c in df.columns]
                
                features = df.select(available).to_numpy().astype(np.float32)
                targets = df[self.target_col].to_numpy().astype(np.float32)
                
                if "valid_sample" in df.columns:
                    valid_mask = df["valid_sample"].to_numpy().astype(np.float32)
                else:
                    valid_mask = np.ones(len(features), dtype=np.float32)
                
                features = np.nan_to_num(features, nan=0.0)
                targets = np.nan_to_num(targets, nan=0.0)
                
                if len(available) == len(self.feature_cols) and len(features) > self.seq_len:
                    data_list.append((features, targets, valid_mask, ticker))
                
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
        
        return data_list
    
    def _compute_indices(self) -> List[Tuple[int, int]]:
        indices = []
        for i, (feats, targets, valid_mask, ticker) in enumerate(self.samples):
            max_start = len(feats) - self.seq_len
            if max_start < 0:
                continue
            for start in range(0, max_start, self.stride):
                end_row = start + self.seq_len
                if valid_mask[end_row - 1] > 0.5:
                    indices.append((i, start))
        return indices
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample_idx, start_row = self.indices[idx]
        features, targets, _, _ = self.samples[sample_idx]
        
        end_row = start_row + self.seq_len
        x = torch.tensor(features[start_row:end_row], dtype=torch.float32)
        
        if self.augment:
            x = x + torch.randn_like(x) * self.noise_std
            scale = random.uniform(self.scale_range[0], self.scale_range[1])
            x = x * scale
        
        raw_target = targets[end_row - 1]
        binary_target = 1.0 if raw_target > 0 else 0.0
        
        if self.label_smooth > 0:
            binary_target = binary_target * (1 - 2 * self.label_smooth) + self.label_smooth
        
        return x, torch.tensor(binary_target, dtype=torch.float32)
    
    @property
    def input_dim(self) -> int:
        return len(self.feature_cols) if self.feature_cols else 0
    
    def get_chronological_split(self, val_ratio: float = 0.2) -> Tuple[List[int], List[int]]:
        train_indices = []
        val_indices = []
        
        ticker_indices: Dict[int, List[int]] = {}
        for i, (sample_idx, start_row) in enumerate(self.indices):
            if sample_idx not in ticker_indices:
                ticker_indices[sample_idx] = []
            ticker_indices[sample_idx].append(i)
        
        for sample_idx, idx_list in ticker_indices.items():
            split_point = int(len(idx_list) * (1 - val_ratio))
            train_indices.extend(idx_list[:split_point])
            val_indices.extend(idx_list[split_point:])
        
        return train_indices, val_indices
    
    def compute_class_weights(self, indices: List[int]) -> float:
        pos_count = 0
        neg_count = 0
        
        for idx in indices:
            sample_idx, start_row = self.indices[idx]
            _, targets, _, _ = self.samples[sample_idx]
            end_row = start_row + self.seq_len
            
            if targets[end_row - 1] > 0:
                pos_count += 1
            else:
                neg_count += 1
        
        if pos_count == 0:
            return 1.0
        
        weight = neg_count / pos_count
        logger.info(f"Class distribution: UP={pos_count}, DOWN={neg_count}, pos_weight={weight:.4f}")
        return weight


class Trainer:
    def __init__(self, config: TrainerConfig, device: torch.device):
        self.config = config
        self.device = device
        
        # Load datasets (using config.stride)
        full_dataset = MultiAssetDataset(seq_len=config.seq_len, stride=config.stride, augment=False)
        
        if full_dataset.input_dim > 0:
            config.input_dim = full_dataset.input_dim
        
        train_idxs, val_idxs = full_dataset.get_chronological_split(config.val_split)
        pos_weight = full_dataset.compute_class_weights(train_idxs)
        
        # Training dataset
        train_dataset = MultiAssetDataset(
            seq_len=config.seq_len,
            stride=config.stride,
            augment=True,
            noise_std=config.noise_std,
            scale_range=(config.scale_min, config.scale_max),
            label_smooth=config.label_smooth
        )
        # Validation
        val_dataset = MultiAssetDataset(seq_len=config.seq_len, stride=config.stride, augment=False, label_smooth=0.0)
        
        train_subset = Subset(train_dataset, train_idxs)
        val_subset = Subset(val_dataset, val_idxs)
        
        logger.info(f"Train: {len(train_subset)}, Val: {len(val_subset)}")
        
        self.train_loader = DataLoader(
            train_subset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=True,
            drop_last=True
        )
        
        self.val_loader = DataLoader(
            val_subset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True
        )
        
        # Model v6.1 (Fractal CfC)
        liquid_config = LiquidConfig(
            input_dim=config.input_dim,
            hidden_dim=config.hidden_dim,
            num_heads=config.num_heads,
            dropout=config.dropout,
            max_seq_len=config.max_seq_len
        )
        self.model = IARA(liquid_config).to(device)
        
        # Enable TensorFloat32 for faster matmul
        torch.set_float32_matmul_precision('high')
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )
        
        # Loss
        pos_weight_tensor = torch.tensor([pos_weight], device=device)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
        
        # AMP Scaler
        self.scaler = torch.amp.GradScaler('cuda')
        
        # State
        self.start_epoch = 0
        self.epoch = 0
        self.best_val_auc = 0.0
        self.patience_counter = 0
        
        # Resume Logic
        if config.resume:
            self._resume_checkpoint()
        
        logger.info(f"Model Parameters: {self.model.count_parameters():,}")
        logger.info(f"Config: seq_len={config.seq_len}, stride={config.stride}, batch={config.batch_size}, AMP=True, Resume={config.resume}")

    def _resume_checkpoint(self):
        latest_path = self.config.checkpoint_dir / "latest.pth"
        if not latest_path.exists():
            logger.warning(f"Resume requested but {latest_path} not found. Starting fresh.")
            return

        logger.info(f"Resuming from {latest_path}...")
        checkpoint = torch.load(latest_path, map_location=self.device, weights_only=False)
        
        # Load Model
        state_dict = checkpoint['model']
        if list(state_dict.keys())[0].startswith('_orig_mod.'):
            state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
        self.model.load_state_dict(state_dict)
        
        # Load Optimizer & Scheduler
        if 'optimizer' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        if 'scheduler' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler'])
            
        # Load State
        self.start_epoch = checkpoint['epoch'] + 1
        self.epoch = self.start_epoch
        self.best_val_auc = checkpoint.get('best_val_auc', 0.0)
        
        logger.info(f"Resumed at Epoch {self.start_epoch}, Best AUC: {self.best_val_auc:.4f}")

    def _run_epoch(self, loader: DataLoader, training: bool, desc: str = "") -> Dict[str, float]:
        if training:
            self.model.train()
        else:
            self.model.eval()
        
        losses, all_preds, all_targets = [], [], []
        correct, total = 0, 0
        
        pbar = tqdm(loader, desc=desc, leave=False) if training else loader
        
        if training:
            self.optimizer.zero_grad()
        
        context = torch.enable_grad() if training else torch.no_grad()
        
        with context:
            for i, (x, y) in enumerate(pbar):
                x, y = x.to(self.device), y.to(self.device)
                
                # Mixed Precision Forward
                with torch.amp.autocast('cuda'):
                    logits, _ = self.model(x)
                    loss = self.criterion(logits, y)
                
                if training:
                    scaled_loss = loss / self.config.accumulation_steps
                    self.scaler.scale(scaled_loss).backward()
                    
                    if (i + 1) % self.config.accumulation_steps == 0:
                        self.scaler.unscale_(self.optimizer)
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.optimizer.zero_grad()
                        
                        if isinstance(pbar, tqdm):
                            lr = self.optimizer.param_groups[0]['lr']
                            # Multiply loss by steps to show real loss for display
                            pbar.set_postfix(loss=f"{loss.item() * self.config.accumulation_steps:.4f}", lr=f"{lr:.2e}")
                
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()
                
                hard_y = (y > 0.5).float()
                correct += (preds == hard_y).sum().item()
                total += y.size(0)
                losses.append(loss.item() * self.config.accumulation_steps if training else loss.item())
                
                all_preds.extend(probs.detach().cpu().numpy())
                all_targets.extend(hard_y.detach().cpu().numpy())
        
        epoch_loss = np.mean(losses)
        epoch_acc = correct / total if total > 0 else 0
        
        try:
            auc = roc_auc_score(all_targets, all_preds)
        except:
            auc = 0.5
            
        return {"loss": epoch_loss, "accuracy": epoch_acc, "auc": auc}

    def train_epoch(self) -> Tuple[Dict[str, float], Dict[str, float]]:
        train_m = self._run_epoch(self.train_loader, training=True, desc=f"Epoch {self.epoch}")
        val_m = self._run_epoch(self.val_loader, training=False)
        return train_m, val_m

    def save_checkpoint(self, filename="latest.pth"):
        self.config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        # Handle compiled model (if applicable)
        model_state = self.model._orig_mod.state_dict() if hasattr(self.model, '_orig_mod') else self.model.state_dict()
        
        torch.save({
            'epoch': self.epoch,
            'model': model_state,
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'config': self.config,
            'best_val_auc': self.best_val_auc
        }, self.config.checkpoint_dir / filename)

    def train(self):
        logger.info("Starting Training v6.1 (AMP + Opt)...")
        
        for epoch in range(self.start_epoch, self.config.epochs):
            self.epoch = epoch
            train_m, val_m = self.train_epoch()
            
            self.scheduler.step(val_m['loss'])
            
            logger.info(
                f"Epoch {epoch}: "
                f"Train[Loss={train_m['loss']:.4f}, Acc={train_m['accuracy']:.2%}, AUC={train_m['auc']:.4f}] | "
                f"Val[Loss={val_m['loss']:.4f}, Acc={val_m['accuracy']:.2%}, AUC={val_m['auc']:.4f}]"
            )
            
            if val_m['auc'] > self.best_val_auc:
                self.best_val_auc = val_m['auc']
                self.patience_counter = 0
                self.save_checkpoint("best_model.pth")
                logger.info(f"  ✓ New best Val AUC: {self.best_val_auc:.4f}")
            else:
                self.patience_counter += 1
                logger.info(f"  No improvement ({self.patience_counter}/{self.config.patience})")
            
            # Always save latest for resume
            self.save_checkpoint("latest.pth")
            
            if self.patience_counter >= self.config.patience:
                logger.info(f"Early stopping after {epoch + 1} epochs")
                break
        
        logger.info(f"Training finished. Best Val AUC: {self.best_val_auc:.4f}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=768)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    config = TrainerConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        seq_len=args.seq_len,
        resume=args.resume
    )
    
    trainer = Trainer(config, device)
    trainer.train()
