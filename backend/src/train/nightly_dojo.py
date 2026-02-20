"""
Saci Dojo (Smart Nightly Trainer)
=================================
"Treine enquanto eles dormem."

Concepts:
1. Finetuning: Não começa do zero. Pega o `best_model.pth`.
2. Regime Adaptation: Foca nos últimos N meses (default: 6).
3. Importance Sampling: Dá peso maior para candles com alta volatilidade (Onde o dinheiro está).
"""

import sys
import argparse
import logging
from pathlib import Path
from typing import Optional, List, Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
import numpy as np

# Add src to path
sys.path.append("/app")

from src.train.trainer_liquid import TrainerConfig, MultiAssetDataset, Trainer
from src.model.iara_liquid import IARA, LiquidConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class DojoTrainer(Trainer):
    def __init__(self, config: TrainerConfig, device: torch.device, recent_months: int = 6):
        super().__init__(config, device)
        self.recent_months = recent_months
        
    def prepare_smart_data(self):
        """
        Overrides standard data loading to implement:
        1. Time Decay (Recent Bias)
        2. Volatility Importance Sampling
        """
        full_dataset = MultiAssetDataset(seq_len=self.config.seq_len, stride=self.config.stride, augment=False)
        
        # 1. Recent Bias: Select last N% of data roughly fitting 'recent_months'
        # Assuming M5 candles: 1 month ~ 6000 candles (if 24/5)
        # 6 months ~ 36,000 candles.
        
        total_len = len(full_dataset)
        # Let's verify we have enough data
        if total_len < 1000:
            logger.warning("Not enough data for Smart Sampling. Using full dataset.")
            idxs = list(range(total_len))
        else:
            # Take last 50% or fixed window? Let's take last 8000 samples (optimize for speed)
            window_size = 12000 # ~2 months of M5 data Stride 8? 
            # Stride 8: 12000 * 8 = 96,000 raw candles / 12/h = 8000 hours / 24 = 333 days. ~1 year.
            
            start_idx = max(0, total_len - window_size)
            idxs = list(range(start_idx, total_len))
            logger.info(f"Dojo Window: Focusing on last {len(idxs)} samples (Market Regime).")

        # Split Train/Val on this window (Last 20% is Val)
        split = int(len(idxs) * (1 - self.config.val_split))
        train_idxs = idxs[:split]
        val_idxs = idxs[split:]
        
        # 2. Importance Sampling (Volatility)
        # We want to train harder on candles where 'close_log_ret' is high.
        weights = []
        for i in train_idxs:
            sample_idx, start_row = full_dataset.indices[i]
            _, targets, _, _ = full_dataset.samples[sample_idx]
            end_row = start_row + self.config.seq_len
            
            # Target is log_ret
            ret = abs(targets[end_row-1])
            w = 1.0 + (ret * 1000) # Boost high volatility
            weights.append(w)
            
        weights = torch.tensor(weights, dtype=torch.double)
        sampler = WeightedRandomSampler(weights, len(weights))
        
        # Create Loaders
        train_subset = Subset(full_dataset, train_idxs) # Dataset is generic, sampler picks indices relative to subset? 
        # WeightedRandomSampler picks INDICES from 0..len(weights)-1.
        # But DataLoader(subset) maps 0..N to subset indices.
        # So we pass sampler to DataLoader.
        
        self.train_loader = DataLoader(
            train_subset,
            batch_size=self.config.batch_size,
            sampler=sampler, # SMART SAMPLING
            num_workers=self.config.num_workers,
            pin_memory=True
        )
        
        val_subset = Subset(full_dataset, val_idxs)
        self.val_loader = DataLoader(
            val_subset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers
        )
        
        logger.info(f"Smart Data Prepared: Train={len(train_subset)} (Importance Sampled), Val={len(val_subset)}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20) 
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--batch-size", type=int, default=512) # Reduced for stability
    parser.add_argument("--resume", action="store_true", default=True)
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Config for Finetuning
    config = TrainerConfig(
        epochs=args.epochs,
        learning_rate=args.lr,
        resume=args.resume,
        batch_size=args.batch_size,
        stride=4, 
        patience=10
    )
    
    trainer = DojoTrainer(config, device)
    
    # 1. Override Data Prepare
    trainer.prepare_smart_data()
    
    # 2. Train (Logic handled by parent, but uses new loaders)
    trainer.train()

if __name__ == "__main__":
    main()
