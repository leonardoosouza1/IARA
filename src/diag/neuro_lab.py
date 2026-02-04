#!/usr/bin/env python3
"""
I.A.R.A. Neuro-Lab - Real-Time Training Monitor
================================================
A "Mission Control" dashboard for visualizing the AI's evolution.

Features:
- Telemetry parsing from JSONL logs
- Neural attention heatmaps
- Dark-mode dashboard rendering

Usage:
    docker compose exec saci-brain python src/diag/neuro_lab.py --watch
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import seaborn as sns
import torch

# Add project root to path
sys.path.append(".")

from src.model.iara_liquid import IARA, LiquidConfig
from src.train.trainer_liquid import MultiAssetDataset, TrainerConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger("NeuroLab")

# Paths
LOGS_DIR = Path("logs")
CHECKPOINTS_DIR = Path("checkpoints")
OUTPUT_DIR = Path("reports/lab")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dark mode aesthetics
plt.style.use('dark_background')
plt.rcParams['font.family'] = 'monospace'
plt.rcParams['axes.facecolor'] = '#0a0a0a'
plt.rcParams['figure.facecolor'] = '#0a0a0a'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.labelcolor'] = '#cccccc'
plt.rcParams['xtick.color'] = '#888888'
plt.rcParams['ytick.color'] = '#888888'
plt.rcParams['grid.color'] = '#222222'


class TelemetryParser:
    """Parse JSONL training logs."""
    
    def __init__(self, logs_dir: Path = LOGS_DIR):
        self.logs_dir = logs_dir
        self.data = []
        
    def find_latest_log(self) -> Optional[Path]:
        """Find the most recent JSONL file."""
        jsonl_files = list(self.logs_dir.glob("training_history_*.jsonl"))
        if not jsonl_files:
            return None
        return max(jsonl_files, key=lambda p: p.stat().st_mtime)
    
    def parse(self) -> List[Dict]:
        """Parse the latest log file robustly."""
        log_file = self.find_latest_log()
        if not log_file:
            logger.warning("No JSONL logs found.")
            return []
        
        self.data = []
        try:
            with open(log_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        self.data.append(entry)
                    except json.JSONDecodeError:
                        # Incomplete line (being written), skip
                        continue
        except Exception as e:
            logger.error(f"Error parsing log: {e}")
        
        return self.data
    
    def get_metrics(self) -> Dict[str, List]:
        """Extract key metrics for plotting."""
        if not self.data:
            self.parse()
        
        metrics = {
            'epochs': [],
            'train_auc': [],
            'val_auc': [],
            'train_loss': [],
            'val_loss': [],
            'chronic': [],
            'mastered': [],
            'lr': [],
            'vram': []
        }
        
        for entry in self.data:
            if 'status' in entry:  # Skip INTERRUPTED entries
                continue
            
            epoch = entry.get('epoch', 0)
            train = entry.get('train', {})
            val = entry.get('val', {})
            memory = entry.get('memory', {})
            hardware = entry.get('hardware', {})
            meta = entry.get('metrics', {})
            
            metrics['epochs'].append(epoch)
            metrics['train_auc'].append(train.get('auc', 0.5))
            metrics['val_auc'].append(val.get('auc', 0.5))
            metrics['train_loss'].append(train.get('loss', 0))
            metrics['val_loss'].append(val.get('loss', 0))
            metrics['chronic'].append(memory.get('chronic', 0))
            metrics['mastered'].append(memory.get('mastered', 0))
            metrics['lr'].append(meta.get('learning_rate', 0.001))
            metrics['vram'].append(hardware.get('vram_percent', 0))
        
        return metrics


class NeuralImager:
    """Extract and visualize neural attention patterns."""
    
    def __init__(self, checkpoint_path: Path = CHECKPOINTS_DIR / "latest.pth"):
        self.checkpoint_path = checkpoint_path
        self.model = None
        self.config = None
        self.val_batch = None
        
    def load_model(self) -> bool:
        """Load the latest checkpoint."""
        if not self.checkpoint_path.exists():
            logger.warning(f"Checkpoint not found: {self.checkpoint_path}")
            return False
        
        try:
            checkpoint = torch.load(self.checkpoint_path, map_location=DEVICE, weights_only=False)
            self.config = checkpoint.get('config')
            
            # Handle both object and dict configs
            if isinstance(self.config, TrainerConfig):
                t_conf = self.config
            else:
                t_conf = TrainerConfig()
            
            liquid_config = LiquidConfig(
                input_dim=t_conf.input_dim,
                hidden_dim=t_conf.hidden_dim,
                num_heads=t_conf.num_heads,
                dropout=t_conf.dropout,
                max_seq_len=t_conf.max_seq_len,
                num_cfc_layers=t_conf.num_cfc_layers
            )
            
            self.model = IARA(liquid_config).to(DEVICE)
            
            state_dict = checkpoint['model']
            # Handle compiled model weights
            if list(state_dict.keys())[0].startswith('_orig_mod.'):
                state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
            
            self.model.load_state_dict(state_dict, strict=False)
            self.model.eval()
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def get_validation_batch(self, batch_size: int = 32) -> Optional[torch.Tensor]:
        """Get a fixed validation batch for consistent attention visualization."""
        if self.val_batch is not None:
            return self.val_batch
        
        try:
            dataset = MultiAssetDataset(
                seq_len=256,
                stride=100,
                augment=False
            )
            _, val_idxs = dataset.get_chronological_split(0.2)
            
            # Get first N samples
            batch_tensors = []
            for i in range(min(batch_size, len(val_idxs))):
                idx = val_idxs[i]
                x, _, _ = dataset[idx]
                batch_tensors.append(x)
            
            self.val_batch = torch.stack(batch_tensors).to(DEVICE)
            return self.val_batch
            
        except Exception as e:
            logger.error(f"Failed to load validation data: {e}")
            return None
    
    def extract_attention(self) -> Optional[np.ndarray]:
        """Extract attention weights from the model."""
        if self.model is None:
            if not self.load_model():
                return None
        
        batch = self.get_validation_batch()
        if batch is None:
            return None
        
        attention_maps = {}
        
        # Hook to capture attention from FeatureAttention gate
        def hook_fn(module, input, output):
            attention_maps['gate'] = output.detach()
        
        # Register hook on the sigmoid of FeatureAttention.gate
        try:
            hook = self.model.feature_attention.gate[-1].register_forward_hook(hook_fn)
        except AttributeError:
            logger.warning("FeatureAttention not found in model.")
            return None
        
        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                _, _ = self.model(batch)
        
        hook.remove()
        
        if 'gate' not in attention_maps:
            return None
        
        # Back-project to input features
        gate_values = attention_maps['gate'].float()  # [B, T, H] - Ensure float32
        proj_weight = self.model.input_proj.weight.float().abs()  # [H, D]
        
        # [B, T, H] x [H, D] -> [B, T, D]
        feature_importance = torch.matmul(gate_values, proj_weight)
        
        # Average over batch, transpose for heatmap [D, T]
        importance = feature_importance.mean(dim=0).detach().cpu().numpy().T
        
        return importance


class DashboardRenderer:
    """Render the Mission Control dashboard."""
    
    def __init__(self, output_path: Path = OUTPUT_DIR / "mission_control_live.png"):
        self.output_path = output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Feature names for heatmap labels
        self.feature_names = [
            'rsi', 'macd', 'macd_sig', 'macd_hist',
            'bb_pct', 'bb_width', 'atr',
            'stoch_k', 'stoch_d', 'velocity',
            'adx', 'sentiment',
            'h_sin', 'h_cos', 'd_sin', 'd_cos'
        ]
    
    def render(
        self,
        metrics: Dict[str, List],
        attention_map: Optional[np.ndarray] = None,
        current_epoch: int = 0,
        current_lr: float = 0.001,
        vram_usage: float = 0.0
    ):
        """Generate the dashboard image."""
        
        fig = plt.figure(figsize=(16, 10), dpi=100)
        gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1.2], hspace=0.3, wspace=0.2)
        
        # === TOP LEFT: AUC Growth Chart ===
        ax1 = fig.add_subplot(gs[0, 0])
        if metrics['epochs']:
            ax1.plot(metrics['epochs'], metrics['train_auc'], 
                     color='#00bfff', linewidth=2, label='Train AUC', alpha=0.8)
            ax1.plot(metrics['epochs'], metrics['val_auc'], 
                     color='#00ff88', linewidth=2.5, label='Val AUC')
            ax1.axhline(y=0.5, color='#ff4444', linestyle='--', alpha=0.5, label='Random')
            ax1.fill_between(metrics['epochs'], 0.5, metrics['val_auc'], 
                             color='#00ff88', alpha=0.1)
        
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('AUC Score')
        ax1.set_title('🧠 Intelligence Growth', fontsize=12, fontweight='bold', color='#00ff88')
        ax1.legend(loc='lower right', framealpha=0.3)
        ax1.set_ylim(0.45, 0.65)
        ax1.grid(True, alpha=0.3)
        
        # === TOP RIGHT: Memory Therapy Chart ===
        ax2 = fig.add_subplot(gs[0, 1])
        if metrics['epochs']:
            width = 0.8
            ax2.bar(metrics['epochs'], metrics['chronic'], 
                    width, label='Chronic Errors', color='#ff6b6b', alpha=0.8)
            ax2.bar(metrics['epochs'], metrics['mastered'], 
                    width, bottom=metrics['chronic'], label='Mastered', color='#4ecdc4', alpha=0.8)
        
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Sample Count')
        ax2.set_title('🧬 Memory Therapy Progress', fontsize=12, fontweight='bold', color='#ff6b6b')
        ax2.legend(loc='upper right', framealpha=0.3)
        ax2.grid(True, alpha=0.3)
        
        # === BOTTOM: Brain Scan Heatmap ===
        ax3 = fig.add_subplot(gs[1, :])
        
        if attention_map is not None:
            # Limit time steps for readability
            display_map = attention_map[:, -100:]  # Last 100 timesteps
            
            # Use shortened feature names
            n_features = min(len(self.feature_names), display_map.shape[0])
            yticklabels = self.feature_names[:n_features]
            
            sns.heatmap(
                display_map[:n_features],
                ax=ax3,
                cmap='magma',
                robust=True,
                yticklabels=yticklabels,
                xticklabels=False,
                cbar_kws={'label': 'Attention', 'shrink': 0.5}
            )
            ax3.set_xlabel('Time Steps (Past → Now)')
            ax3.set_ylabel('Features')
        else:
            ax3.text(0.5, 0.5, 'Awaiting Neural Data...', 
                     ha='center', va='center', fontsize=14, color='#666666')
            ax3.set_xlim(0, 1)
            ax3.set_ylim(0, 1)
        
        ax3.set_title('🔬 Real-Time Brain Scan (Attention Heatmap)', 
                      fontsize=12, fontweight='bold', color='#ff9f1c')
        
        # === FOOTER: Status Bar ===
        status_text = (
            f"Epoch: {current_epoch}  |  "
            f"LR: {current_lr:.2e}  |  "
            f"VRAM: {vram_usage:.1f}%  |  "
            f"Samples: {len(metrics['epochs'])}  |  "
            f"Updated: {time.strftime('%H:%M:%S')}"
        )
        fig.text(0.5, 0.02, status_text, ha='center', fontsize=10, 
                 color='#888888', family='monospace')
        
        # Title
        fig.suptitle('I.A.R.A. NEURO-LAB v6.7', fontsize=16, fontweight='bold', 
                     color='#ffffff', y=0.98)
        
        # Save
        plt.savefig(self.output_path, bbox_inches='tight', facecolor='#0a0a0a', edgecolor='none')
        plt.close(fig)
        
        logger.info(f"Dashboard saved: {self.output_path}")


def run_neuro_lab(watch: bool = False, interval: int = 30):
    """Main loop for the Neuro-Lab."""
    
    logger.info("=" * 50)
    logger.info("  I.A.R.A. NEURO-LAB v6.7")
    logger.info("  Real-Time Training Monitor")
    logger.info("=" * 50)
    
    telemetry = TelemetryParser()
    imager = NeuralImager()
    renderer = DashboardRenderer()
    
    while True:
        try:
            # 1. Parse Telemetry
            logger.info("Parsing telemetry...")
            metrics = telemetry.get_metrics()
            
            # 2. Neural Imaging (Reload model each time to get latest weights)
            logger.info("Extracting neural patterns...")
            imager.load_model()
            attention_map = imager.extract_attention()
            
            # 3. Extract current state
            current_epoch = metrics['epochs'][-1] if metrics['epochs'] else 0
            current_lr = metrics['lr'][-1] if metrics['lr'] else 0.001
            vram_usage = metrics['vram'][-1] if metrics['vram'] else 0.0
            
            # 4. Render Dashboard
            logger.info("Rendering dashboard...")
            renderer.render(
                metrics=metrics,
                attention_map=attention_map,
                current_epoch=current_epoch,
                current_lr=current_lr,
                vram_usage=vram_usage
            )
            
            if not watch:
                logger.info("Single render complete. Use --watch for continuous mode.")
                break
            
            logger.info(f"Sleeping {interval}s before next update...")
            time.sleep(interval)
            
        except KeyboardInterrupt:
            logger.info("\nNeuro-Lab shutting down. Goodbye!")
            break
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
            if not watch:
                raise
            time.sleep(interval)


def main():
    parser = argparse.ArgumentParser(description="I.A.R.A. Neuro-Lab Dashboard")
    parser.add_argument("--watch", action="store_true", help="Continuous monitoring mode")
    parser.add_argument("--interval", type=int, default=30, help="Update interval in seconds")
    args = parser.parse_args()
    
    run_neuro_lab(watch=args.watch, interval=args.interval)


if __name__ == "__main__":
    main()
