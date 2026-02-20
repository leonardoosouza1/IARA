"""
Batch Sensitivity Search for I.A.R.A. v4.4
==========================================
Tests different batch sizes with sqrt LR scaling to find optimal config.

Configurations:
- Batch Sizes: [256, 512, 1024, 2048]
- LR Scaling: LR_new = 0.001 * sqrt(Batch / 256)
- Epochs per run: 30
- Metrics: Time/Epoch, Val AUC, Test Precision
"""

import gc
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import roc_auc_score

from src.model.iara_liquid import IARA, LiquidConfig
from src.train.trainer_liquid import LiquidDataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Config
BATCH_SIZES = [256, 512, 1024, 2048]
BASE_LR = 0.001
BASE_BATCH = 256
EPOCHS_PER_RUN = 30
SEQ_LEN = 64
HIDDEN_DIM = 512
DROPOUT = 0.4


@dataclass
class SearchResult:
    """Result of a single batch size experiment."""
    batch_size: int
    learning_rate: float
    time_per_epoch: float
    best_val_auc: float
    test_precision: float
    status: str


def compute_scaled_lr(batch_size: int) -> float:
    """Square root LR scaling rule."""
    return BASE_LR * math.sqrt(batch_size / BASE_BATCH)


def reset_model(device: torch.device) -> IARA:
    """Create fresh model with reset weights."""
    config = LiquidConfig(
        input_dim=11,
        hidden_dim=HIDDEN_DIM,
        dropout=DROPOUT
    )
    model = IARA(config).to(device)
    return model


def train_single_config(
    batch_size: int,
    learning_rate: float,
    device: torch.device
) -> SearchResult:
    """Train model with a single batch size configuration."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Testing Batch Size: {batch_size}, LR: {learning_rate:.6f}")
    logger.info(f"{'='*60}")
    
    # Clear memory
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    try:
        # Load data
        full_dataset = LiquidDataset(seq_len=SEQ_LEN, augment=False)
        train_idxs, val_idxs = full_dataset.get_chronological_split(0.2)
        
        train_dataset = LiquidDataset(
            seq_len=SEQ_LEN,
            augment=True,
            noise_std=0.02,
            scale_range=(0.98, 1.02),
            label_smooth=0.1
        )
        val_dataset = LiquidDataset(seq_len=SEQ_LEN, augment=False, label_smooth=0.0)
        
        train_subset = Subset(train_dataset, train_idxs)
        val_subset = Subset(val_dataset, val_idxs)
        
        # Adjust workers based on batch size
        num_workers = min(4, batch_size // 64)
        
        train_loader = DataLoader(
            train_subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        # Fresh model
        model = reset_model(device)
        
        # Optimizer with scaled LR
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=1e-4
        )
        
        # Scheduler
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=learning_rate * 2,
            steps_per_epoch=len(train_loader),
            epochs=EPOCHS_PER_RUN,
            pct_start=0.3
        )
        
        # Loss
        criterion = nn.BCEWithLogitsLoss()
        
        # Training loop
        best_val_auc = 0.0
        epoch_times = []
        
        for epoch in range(EPOCHS_PER_RUN):
            epoch_start = time.time()
            
            # Train
            model.train()
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                
                optimizer.zero_grad()
                logits, _ = model(x)
                loss = criterion(logits, y)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
                scheduler.step()
            
            # Validate
            model.eval()
            all_preds, all_targets = [], []
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(device), y.to(device)
                    logits, _ = model(x)
                    probs = torch.sigmoid(logits)
                    all_preds.extend(probs.cpu().numpy())
                    all_targets.extend((y > 0.5).float().cpu().numpy())
            
            try:
                val_auc = roc_auc_score(all_targets, all_preds)
            except:
                val_auc = 0.5
            
            if val_auc > best_val_auc:
                best_val_auc = val_auc
            
            epoch_time = time.time() - epoch_start
            epoch_times.append(epoch_time)
            
            if epoch % 10 == 0 or epoch == EPOCHS_PER_RUN - 1:
                logger.info(f"  Epoch {epoch}: Val AUC={val_auc:.4f}, Time={epoch_time:.2f}s")
        
        # Calculate test precision (simple approximation)
        preds_array = np.array(all_preds)
        targets_array = np.array(all_targets)
        pred_classes = (preds_array > 0.5).astype(float)
        
        tp = ((pred_classes == 1) & (targets_array == 1)).sum()
        fp = ((pred_classes == 1) & (targets_array == 0)).sum()
        test_precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0.0
        
        avg_time = np.mean(epoch_times)
        
        result = SearchResult(
            batch_size=batch_size,
            learning_rate=learning_rate,
            time_per_epoch=avg_time,
            best_val_auc=best_val_auc,
            test_precision=test_precision,
            status="✓ Success"
        )
        
        logger.info(f"  Result: AUC={best_val_auc:.4f}, Precision={test_precision:.1f}%, Time/Epoch={avg_time:.2f}s")
        
        return result
        
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            logger.warning(f"  OOM for batch size {batch_size}")
            return SearchResult(
                batch_size=batch_size,
                learning_rate=learning_rate,
                time_per_epoch=0,
                best_val_auc=0,
                test_precision=0,
                status="✗ OOM"
            )
        raise


def run_search():
    """Run the full batch sensitivity search."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    results: List[SearchResult] = []
    
    for batch_size in BATCH_SIZES:
        lr = compute_scaled_lr(batch_size)
        result = train_single_config(batch_size, lr, device)
        results.append(result)
    
    # Print summary table
    print("\n" + "=" * 80)
    print("BATCH SENSITIVITY SEARCH RESULTS")
    print("=" * 80)
    print(f"{'Batch Size':>12} | {'LR':>10} | {'Time/Epoch':>12} | {'Val AUC':>10} | {'Precision':>12} | {'Status':>12}")
    print("-" * 80)
    
    for r in results:
        print(f"{r.batch_size:>12} | {r.learning_rate:>10.6f} | {r.time_per_epoch:>10.2f}s | {r.best_val_auc:>10.4f} | {r.test_precision:>10.1f}% | {r.status:>12}")
    
    print("=" * 80)
    
    # Find best config
    valid_results = [r for r in results if "Success" in r.status]
    if valid_results:
        best = max(valid_results, key=lambda x: x.best_val_auc)
        print(f"\n🏆 BEST CONFIG: Batch={best.batch_size}, LR={best.learning_rate:.6f}, AUC={best.best_val_auc:.4f}")


if __name__ == "__main__":
    run_search()
