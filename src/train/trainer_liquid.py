"""
Trainer for I.A.R.A. Liquid Neural Network
==========================================
Training loop for continuous-time market prediction.
Dual-objective: MSE (price) + BCE (direction).
"""

import argparse
import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from src.model.iara_liquid import IARA, LiquidConfig
from src.data.processor import Normalizer, NormalizerState  # NormalizerState needed for pickle

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Paths
DATA_PROCESSED_DIR = Path(__file__).parent.parent.parent / "data" / "processed"
SCALER_PATH = Path(__file__).parent.parent.parent / "data" / "scaler.pkl"
CHECKPOINT_DIR = Path(__file__).parent.parent.parent / "checkpoints"


@dataclass
class TrainerConfig:
    """Training configuration."""
    
    # Data
    seq_len: int = 64
    batch_size: int = 32
    num_workers: int = 4
    
    # Model
    input_dim: int = 64
    hidden_dim: int = 512        # Wide Single Layer
    n_neurons: int = 512         # Wide Single Layer
    n_layers: int = 1            # Single Layer
    sparsity: float = 0.6
    
    # Training
    epochs: int = 50
    learning_rate: float = 0.002
    weight_decay: float = 1e-2
    grad_clip: float = 0.5       # Tighter clipping per instructions
    
    # Loss weights (Aggressive Direction Focus)
    mse_weight: float = 0.5      # Balanced with scaling
    bce_weight: float = 1.0      # Standard BCE
    
    # Target Scaling (Amplify Gradients)
    target_scale: float = 1000.0
    
    # Checkpointing
    save_every: int = 5
    checkpoint_dir: Path = CHECKPOINT_DIR


class LiquidDataset(Dataset):
    """
    Dataset for Liquid Neural Network training.
    
    Returns sequences of normalized features and targets.
    """
    
    def __init__(
        self,
        data_dir: Path = DATA_PROCESSED_DIR,
        scaler_path: Path = SCALER_PATH,
        seq_len: int = 64,
        stride: int = 16,
        tickers: Optional[List[str]] = None,
        target_col: str = "close_log_ret"
    ):
        self.data_dir = Path(data_dir)
        self.seq_len = seq_len
        self.stride = stride
        self.target_col = target_col
        
        # Load normalizer to get feature columns
        self.normalizer = Normalizer.load(scaler_path)
        
        # Feature columns are the normalized ones
        self.feature_cols = [f"{c}_norm" for c in self.normalizer.state.feature_columns]
        
        # Load data
        self.samples = self._load_data(tickers)
        self.indices = self._compute_indices()
        
        logger.info(
            f"LiquidDataset initialized: {len(self.indices)} windows, "
            f"{len(self.feature_cols)} features"
        )
    
    def _load_data(self, tickers: Optional[List[str]]) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Load data into memory.
        
        Returns list of (features, target_reg, target_cls) per ticker.
        """
        parquet_files = list(self.data_dir.glob("*.parquet"))
        if tickers:
            parquet_files = [f for f in parquet_files if f.stem in tickers]
        
        data_list = []
        
        for file_path in parquet_files:
            try:
                df = pl.read_parquet(file_path)
                
                # Check required columns
                if self.target_col not in df.columns:
                    logger.warning(f"Skipping {file_path.name}: Target {self.target_col} not found")
                    continue
                
                if "direction" not in df.columns:
                    logger.warning(f"Skipping {file_path.name}: direction column not found")
                    continue
                
                # Get available feature columns
                available_feat_cols = [c for c in self.feature_cols if c in df.columns]
                
                if len(available_feat_cols) < 5:
                    logger.warning(f"Skipping {file_path.name}: Only {len(available_feat_cols)} features found")
                    continue
                
                # Extract arrays
                features = df.select(available_feat_cols).to_numpy().astype(np.float32)
                target_reg = df[self.target_col].to_numpy().astype(np.float32)
                target_cls = df["direction"].to_numpy().astype(np.float32)
                
                # Update feature cols to actual available
                if len(available_feat_cols) != len(self.feature_cols):
                    self.feature_cols = available_feat_cols
                
                if len(features) > self.seq_len:
                    data_list.append((features, target_reg, target_cls))
                    
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
        
        return data_list
    
    def _compute_indices(self) -> List[Tuple[int, int]]:
        """Compute (sample_idx, start_row) for sliding windows."""
        indices = []
        
        for i, (feats, _, _) in enumerate(self.samples):
            n_rows = len(feats)
            max_start = n_rows - self.seq_len - 1  # -1 for target shift
            
            if max_start < 0:
                continue
            
            for start in range(0, max_start, self.stride):
                indices.append((i, start))
        
        return indices
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sample_idx, start_row = self.indices[idx]
        features, target_reg, target_cls = self.samples[sample_idx]
        
        # Slice window
        end_row = start_row + self.seq_len
        
        # Features for input
        x = torch.tensor(features[start_row:end_row], dtype=torch.float32)
        
        # Targets are shifted by 1 (predict next step)
        y_reg = torch.tensor(target_reg[start_row + 1:end_row + 1], dtype=torch.float32)
        y_cls = torch.tensor(target_cls[start_row + 1:end_row + 1], dtype=torch.float32)
        
        return x, y_reg, y_cls
    
    @property
    def input_dim(self) -> int:
        """Return the number of input features."""
        if self.samples:
            return self.samples[0][0].shape[1]
        return 0


class LiquidTrainer:
    """
    Trainer for I.A.R.A. Liquid Neural Network.
    """
    
    def __init__(self, config: TrainerConfig, device: torch.device):
        self.config = config
        self.device = device
        
        # Create dataset
        self.dataset = LiquidDataset(
            data_dir=DATA_PROCESSED_DIR,
            scaler_path=SCALER_PATH,
            seq_len=config.seq_len,
        )
        
        # Update input_dim from dataset
        actual_input_dim = self.dataset.input_dim
        if actual_input_dim > 0:
            config.input_dim = actual_input_dim
        
        # Create dataloader
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=True,
            drop_last=True
        )
        
        # Create model
        liquid_config = LiquidConfig(
            input_dim=config.input_dim,
            hidden_dim=config.hidden_dim,
            n_neurons=config.n_neurons,
            n_layers=config.n_layers,
            sparsity=config.sparsity
        )
        self.model = IARA(liquid_config).to(device)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=100,  # Slower decay (was epochs=50)
            eta_min=1e-6
        )
        
        # Losses
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        
        # Tracking
        self.epoch = 0
        self.global_step = 0
        self.best_loss = float('inf')
        
        logger.info(f"Model parameters: {self.model.count_parameters():,}")
        logger.info(f"Active connections: {self.model.count_active_connections():,}")
    
    def train_epoch(self) -> Dict[str, float]:
        """Train one epoch with numerical stability checks."""
        self.model.train()
        
        total_loss = 0.0
        total_mse = 0.0
        total_bce = 0.0
        total_dir_acc = 0.0
        n_batches = 0
        nan_count = 0
        
        pbar = tqdm(self.dataloader, desc=f"Epoch {self.epoch}")
        
        for batch_idx, (x, y_reg, y_cls) in enumerate(pbar):
            x = x.to(self.device)
            y_reg = y_reg.to(self.device)
            y_cls = y_cls.to(self.device)
            
            # Input Noise Injection (Prevent Mode Collapse)
            if self.model.training:
                noise = torch.randn_like(x) * 0.01  # 1% Noise
                x = x + noise
            
            # Forward
            pred_reg, pred_cls, _ = self.model(x)
            
            # Target Scaling: Amplify targets to [ -10, 10 ] range instead of [ -0.01, 0.01 ]
            scaled_y_reg = y_reg * self.config.target_scale
            
            # Compute losses
            loss_mse = self.mse_loss(pred_reg, scaled_y_reg)
            loss_bce = self.bce_loss(pred_cls, y_cls)
            
            loss = self.config.mse_weight * loss_mse + self.config.bce_weight * loss_bce
            
            # NaN Check - skip batch if loss is NaN
            if torch.isnan(loss) or torch.isinf(loss):
                nan_count += 1
                self.optimizer.zero_grad()
                if nan_count > 10:
                    logger.warning(f"Too many NaN losses ({nan_count}). Reducing LR.")
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] *= 0.5
                    nan_count = 0
                continue
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping (CRITICAL for RNNs/ODEs)
            grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            
            # Skip update if gradients are also NaN
            if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                nan_count += 1
                continue
            
            self.optimizer.step()
            
            # Directional accuracy
            pred_dir = (torch.sigmoid(pred_cls) > 0.5).float()
            dir_acc = (pred_dir == y_cls).float().mean().item()
            
            # Track metrics
            total_loss += loss.item()
            total_mse += loss_mse.item()
            total_bce += loss_bce.item()
            total_dir_acc += dir_acc
            n_batches += 1
            self.global_step += 1
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'mse': f'{loss_mse.item():.4f}',
                'dir_acc': f'{dir_acc*100:.1f}%',
                'lr': f'{self.scheduler.get_last_lr()[0]:.2e}'
            })
        
        self.scheduler.step()
        
        return {
            'loss': total_loss / n_batches,
            'mse': total_mse / n_batches,
            'bce': total_bce / n_batches,
            'dir_acc': total_dir_acc / n_batches * 100
        }
    
    def save_checkpoint(self, filename: str = "latest.pth"):
        """Save model checkpoint."""
        self.config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'model_config': self.model.config,
            'trainer_config': self.config,
            'best_loss': self.best_loss
        }
        
        path = self.config.checkpoint_dir / filename
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved: {path}")
    
    def load_checkpoint(self, filename: str = "latest.pth") -> bool:
        """Load model checkpoint if exists."""
        path = self.config.checkpoint_dir / filename
        
        if not path.exists():
            return False
        
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_loss = checkpoint.get('best_loss', float('inf'))
        
        logger.info(f"Loaded checkpoint from epoch {self.epoch}")
        return True
    
    def train(self, resume: bool = True):
        """Full training loop."""
        if resume:
            self.load_checkpoint()
        
        logger.info(f"Starting training from epoch {self.epoch}")
        
        for epoch in range(self.epoch, self.config.epochs):
            self.epoch = epoch
            
            metrics = self.train_epoch()
            
            logger.info(
                f"Epoch {epoch}: loss={metrics['loss']:.4f}, "
                f"mse={metrics['mse']:.4f}, bce={metrics['bce']:.4f}, "
                f"dir_acc={metrics['dir_acc']:.2f}%"
            )
            
            # Save checkpoint
            self.save_checkpoint("latest.pth")
            
            if (epoch + 1) % self.config.save_every == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch}.pth")
            
            # Track best
            if metrics['loss'] < self.best_loss:
                self.best_loss = metrics['loss']
                self.save_checkpoint("best_model.pth")
        
        self.save_checkpoint("final_model.pth")
        logger.info("Training complete!")


def parse_args():
    parser = argparse.ArgumentParser(description="Train I.A.R.A. Liquid Neural Network")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--seq-len", type=int, default=64, help="Sequence length")

    parser.add_argument("--hidden-dim", type=int, default=512, help="Hidden dimension")
    parser.add_argument("--n-neurons", type=int, default=512, help="Number of liquid neurons")
    parser.add_argument("--n-layers", type=int, default=1, help="Number of stacked layers")
    parser.add_argument("--sparsity", type=float, default=0.6, help="Connection sparsity")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of DataLoader workers")
    parser.add_argument("--no-resume", action="store_true", help="Start from scratch")
    return parser.parse_args()


def main():
    args = parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    config = TrainerConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        seq_len=args.seq_len,
        hidden_dim=args.hidden_dim,
        n_neurons=args.n_neurons,
        n_layers=args.n_layers,
        sparsity=args.sparsity,
        num_workers=args.num_workers
    )
    
    trainer = LiquidTrainer(config, device)
    trainer.train(resume=not args.no_resume)


if __name__ == "__main__":
    main()
