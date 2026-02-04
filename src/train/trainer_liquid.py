"""
Trainer for I.A.R.A. v4.1 - Liquid Neural Network
=================================================
Features:
- AUC-based checkpointing (not accuracy)
- Data augmentation (Gaussian noise + scaling)
- Chronological validation split
- 20-epoch early stopping patience
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
    """Training configuration v4.1."""
    
    # Data
    seq_len: int = 64
    batch_size: int = 128
    num_workers: int = 4
    val_split: float = 0.2
    
    # Model
    input_dim: int = 11
    hidden_dim: int = 256
    dropout: float = 0.4
    
    # Training
    epochs: int = 100
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    
    # Augmentation  
    noise_std: float = 0.02
    scale_min: float = 0.98
    scale_max: float = 1.02
    
    # Early Stopping
    patience: int = 20
    
    # Checkpointing
    checkpoint_dir: Path = CHECKPOINT_DIR


class LiquidDataset(Dataset):
    """Dataset with optional augmentation and smart label filtering."""
    
    def __init__(
        self,
        data_dir: Path = DATA_PROCESSED_DIR,
        seq_len: int = 64,
        stride: int = 16,
        tickers: Optional[List[str]] = None,
        target_col: str = "close_log_ret",
        augment: bool = False,
        noise_std: float = 0.02,
        scale_range: Tuple[float, float] = (0.98, 1.02)
    ):
        self.data_dir = Path(data_dir)
        self.seq_len = seq_len
        self.stride = stride
        self.target_col = target_col
        self.augment = augment
        self.noise_std = noise_std
        self.scale_range = scale_range
        self.feature_cols = None
        self.samples = self._load_data(tickers)
        self.indices = self._compute_indices()
        
        aug_str = " (augmented)" if augment else ""
        logger.info(f"Dataset{aug_str}: {len(self.indices)} samples, {len(self.feature_cols) if self.feature_cols else 0} features")
    
    def _load_data(self, tickers: Optional[List[str]]) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        parquet_files = list(self.data_dir.glob("*.parquet"))
        if tickers:
            parquet_files = [f for f in parquet_files if f.stem in tickers]
        
        data_list = []
        self.feature_cols = None
        
        for file_path in parquet_files:
            try:
                df = pl.read_parquet(file_path)
                
                if self.target_col not in df.columns:
                    continue
                
                if self.feature_cols is None:
                    self.feature_cols = sorted([c for c in df.columns if c.endswith("_norm")])
                    logger.info(f"Features: {self.feature_cols}")
                
                available = [c for c in self.feature_cols if c in df.columns]
                
                features = df.select(available).to_numpy().astype(np.float32)
                targets = df[self.target_col].to_numpy().astype(np.float32)
                
                # Smart label mask
                if "valid_sample" in df.columns:
                    valid_mask = df["valid_sample"].to_numpy().astype(np.float32)
                else:
                    valid_mask = np.ones(len(features), dtype=np.float32)
                
                if len(available) == len(self.feature_cols) and len(features) > self.seq_len:
                    data_list.append((features, targets, valid_mask))
                
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
        
        return data_list
    
    def _compute_indices(self) -> List[Tuple[int, int]]:
        """Only include indices where target is valid."""
        indices = []
        for i, (feats, targets, valid_mask) in enumerate(self.samples):
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
        features, targets, _ = self.samples[sample_idx]
        
        end_row = start_row + self.seq_len
        x = torch.tensor(features[start_row:end_row], dtype=torch.float32)
        
        # Augmentation
        if self.augment:
            x = x + torch.randn_like(x) * self.noise_std
            scale = random.uniform(self.scale_range[0], self.scale_range[1])
            x = x * scale
        
        raw_target = targets[end_row - 1]
        binary_target = 1.0 if raw_target > 0 else 0.0
        
        return x, torch.tensor(binary_target, dtype=torch.float32)
    
    @property
    def input_dim(self) -> int:
        return len(self.feature_cols) if self.feature_cols else 0
    
    def get_chronological_split(self, val_ratio: float = 0.2) -> Tuple[List[int], List[int]]:
        n = len(self.indices)
        split_idx = int(n * (1 - val_ratio))
        return list(range(split_idx)), list(range(split_idx, n))


class Trainer:
    def __init__(self, config: TrainerConfig, device: torch.device):
        self.config = config
        self.device = device
        
        # Load datasets
        full_dataset = LiquidDataset(seq_len=config.seq_len, augment=False)
        
        if full_dataset.input_dim > 0:
            config.input_dim = full_dataset.input_dim
        
        train_idxs, val_idxs = full_dataset.get_chronological_split(config.val_split)
        
        train_dataset = LiquidDataset(
            seq_len=config.seq_len,
            augment=True,
            noise_std=config.noise_std,
            scale_range=(config.scale_min, config.scale_max)
        )
        val_dataset = LiquidDataset(seq_len=config.seq_len, augment=False)
        
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
        
        # Model
        liquid_config = LiquidConfig(
            input_dim=config.input_dim,
            hidden_dim=config.hidden_dim,
            dropout=config.dropout
        )
        self.model = IARA(liquid_config).to(device)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=0.005,
            steps_per_epoch=len(self.train_loader),
            epochs=config.epochs,
            pct_start=0.3,
            div_factor=25,
            final_div_factor=1000
        )
        
        # Loss
        self.criterion = nn.BCEWithLogitsLoss()
        
        # State
        self.epoch = 0
        self.best_val_auc = 0.0
        self.patience_counter = 0
        
        logger.info(f"Model Parameters: {self.model.count_parameters():,}")

    def _run_epoch(self, loader: DataLoader, training: bool, desc: str = "") -> Dict[str, float]:
        if training:
            self.model.train()
        else:
            self.model.eval()
        
        losses, all_preds, all_targets = [], [], []
        correct, total = 0, 0
        
        context = torch.no_grad() if not training else torch.enable_grad()
        pbar = tqdm(loader, desc=desc, leave=False) if training else loader
        
        with context:
            for x, y in pbar:
                x, y = x.to(self.device), y.to(self.device)
                
                logits, _ = self.model(x)
                loss = self.criterion(logits, y)
                
                if training:
                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                    self.optimizer.step()
                    self.scheduler.step()
                    
                    if isinstance(pbar, tqdm):
                        pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{self.scheduler.get_last_lr()[0]:.2e}")
                
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()
                
                correct += (preds == y).sum().item()
                total += y.size(0)
                losses.append(loss.item())
                
                all_preds.extend(probs.detach().cpu().numpy())
                all_targets.extend(y.detach().cpu().numpy())
        
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
        torch.save({
            'model': self.model.state_dict(),
            'config': self.config,
            'epoch': self.epoch,
            'best_val_auc': self.best_val_auc
        }, self.config.checkpoint_dir / filename)

    def train(self):
        logger.info("Starting Training v4.1 (AUC-based checkpointing)...")
        
        for epoch in range(self.config.epochs):
            self.epoch = epoch
            train_m, val_m = self.train_epoch()
            
            # Log
            logger.info(
                f"Epoch {epoch}: "
                f"Train[Acc={train_m['accuracy']:.2%}, AUC={train_m['auc']:.4f}] | "
                f"Val[Acc={val_m['accuracy']:.2%}, AUC={val_m['auc']:.4f}]"
            )
            
            # Checkpoint based on VALIDATION AUC
            if val_m['auc'] > self.best_val_auc:
                self.best_val_auc = val_m['auc']
                self.patience_counter = 0
                self.save_checkpoint("best_model.pth")
                logger.info(f"  ✓ New best Val AUC: {self.best_val_auc:.4f}")
            else:
                self.patience_counter += 1
                logger.info(f"  No improvement ({self.patience_counter}/{self.config.patience})")
            
            self.save_checkpoint()
            
            # Early Stopping
            if self.patience_counter >= self.config.patience:
                logger.info(f"Early stopping after {epoch + 1} epochs")
                break
        
        logger.info(f"Training finished. Best Val AUC: {self.best_val_auc:.4f}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--noise", type=float, default=0.02)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    config = TrainerConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        noise_std=args.noise
    )
    
    trainer = Trainer(config, device)
    trainer.train()
