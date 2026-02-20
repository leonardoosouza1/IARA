"""
Trainer - Training Loop for Saci-Nano (Causal Language Modeling)
=================================================================
Training loop with gradient accumulation, warmup, checkpointing,
and continuous learning support for AMD ROCm.

Supports configurable training modes via config/training_modes.yaml
Usage: python src/train/trainer.py --mode <mode_name>
"""

import argparse
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.model.saci_nano import SaciNano, SaciNanoConfig
from src.train.dataset import create_dataloader
from src.data.tokenizer import FinancialTokenizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Paths
CHECKPOINT_DIR = Path(__file__).parent.parent.parent / "checkpoints"
DATA_TOKENIZED_DIR = Path(__file__).parent.parent.parent / "data" / "tokenized"
TOKENIZER_PATH = Path(__file__).parent.parent.parent / "data" / "tokenizer.pkl"
CONFIG_DIR = Path(__file__).parent.parent.parent / "config"
MODES_CONFIG_PATH = CONFIG_DIR / "training_modes.yaml"


@dataclass
class TrainerConfig:
    """Training configuration."""
    
    # Model
    dim: int = 512
    n_layers: int = 8
    n_heads: int = 8
    n_kv_heads: int = 4
    dropout: float = 0.1
    
    # Data
    seq_len: int = 384
    batch_size: int = 16
    
    # Training
    epochs: int = 50
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    
    # Scheduler
    warmup_steps: int = 500
    min_lr_ratio: float = 0.1
    
    # Checkpointing
    save_every_n_epochs: int = 5
    checkpoint_dir: Path = field(default_factory=lambda: CHECKPOINT_DIR)
    resume_from: Optional[Path] = None
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Metadata
    mode_name: str = "custom"
    mode_description: str = ""
    
    @classmethod
    def from_yaml_mode(cls, mode_name: str, config_path: Path = MODES_CONFIG_PATH) -> "TrainerConfig":
        """
        Load configuration from a YAML mode preset.
        
        Args:
            mode_name: Name of the mode (e.g., 'fast', 'moderate', 'full')
            config_path: Path to the training_modes.yaml file
            
        Returns:
            TrainerConfig instance
        """
        if not config_path.exists():
            logger.warning(f"Config file not found: {config_path}. Using defaults.")
            return cls()
        
        with open(config_path, 'r') as f:
            all_modes = yaml.safe_load(f)
        
        # Get default mode if none specified
        if mode_name == "default":
            mode_name = all_modes.get("_active_default", "moderate")
        
        if mode_name not in all_modes:
            available = [k for k in all_modes.keys() if not k.startswith("_")]
            raise ValueError(f"Mode '{mode_name}' not found. Available: {available}")
        
        mode = all_modes[mode_name]
        
        # Extract nested config
        model_cfg = mode.get("model", {})
        data_cfg = mode.get("data", {})
        train_cfg = mode.get("training", {})
        ckpt_cfg = mode.get("checkpointing", {})
        
        return cls(
            # Model
            dim=model_cfg.get("dim", 512),
            n_layers=model_cfg.get("n_layers", 8),
            n_heads=model_cfg.get("n_heads", 8),
            n_kv_heads=model_cfg.get("n_kv_heads", 4),
            dropout=model_cfg.get("dropout", 0.1),
            # Data
            seq_len=data_cfg.get("seq_len", 384),
            batch_size=data_cfg.get("batch_size", 16),
            # Training
            epochs=train_cfg.get("epochs", 50),
            learning_rate=train_cfg.get("learning_rate", 3e-4),
            weight_decay=train_cfg.get("weight_decay", 0.1),
            gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 4),
            max_grad_norm=train_cfg.get("max_grad_norm", 1.0),
            warmup_steps=train_cfg.get("warmup_steps", 500),
            # Checkpointing
            save_every_n_epochs=ckpt_cfg.get("save_every_n_epochs", 5),
            # Metadata
            mode_name=mode_name,
            mode_description=mode.get("description", "")
        )
    
    @staticmethod
    def list_available_modes(config_path: Path = MODES_CONFIG_PATH) -> list:
        """List all available training modes."""
        if not config_path.exists():
            return []
        
        with open(config_path, 'r') as f:
            all_modes = yaml.safe_load(f)
        
        modes = []
        default_mode = all_modes.get("_active_default", "moderate")
        
        for name, config in all_modes.items():
            if name.startswith("_"):
                continue
            desc = config.get("description", "No description")
            is_default = " (default)" if name == default_mode else ""
            modes.append(f"  {name}{is_default}: {desc}")
        
        return modes


def get_lr_scheduler(
    optimizer: AdamW,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.1
) -> LambdaLR:
    """Create learning rate scheduler with warmup and cosine decay."""
    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        else:
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(min_lr_ratio, 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159)).item()))
    
    return LambdaLR(optimizer, lr_lambda)


class Trainer:
    """
    Trainer for Saci-Nano Causal Language Model.
    
    Features:
    - Cross-Entropy Loss for Next Token Prediction
    - AdamW with warmup + cosine decay
    - Gradient accumulation
    - Continuous learning (resume from checkpoint)
    """
    
    def __init__(self, config: TrainerConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        logger.info(f"Using device: {self.device}")
        if self.device.type == "cuda":
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        
        if config.mode_name != "custom":
            logger.info(f"Training mode: {config.mode_name}")
            if config.mode_description:
                logger.info(f"Description: {config.mode_description}")
        
        # Create checkpoint directory
        self.config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Placeholders
        self.model: Optional[SaciNano] = None
        self.optimizer: Optional[AdamW] = None
        self.scheduler: Optional[LambdaLR] = None
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_loss = float('inf')
    
    def build_model(self, vocab_size: int) -> SaciNano:
        """Build the model."""
        model_config = SaciNanoConfig(
            vocab_size=vocab_size,
            dim=self.config.dim,
            n_layers=self.config.n_layers,
            n_heads=self.config.n_heads,
            n_kv_heads=self.config.n_kv_heads,
            max_seq_len=self.config.seq_len,
            dropout=self.config.dropout
        )
        
        model = SaciNano(model_config)
        model = model.to(self.device)
        
        logger.info(f"Model parameters: {model.count_parameters():,}")
        logger.info(f"Estimated VRAM: {model.estimate_vram_mb(self.config.batch_size):.2f} MB")
        
        return model
    
    def build_optimizer(self) -> AdamW:
        """Build optimizer with weight decay."""
        decay_params = []
        no_decay_params = []
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if "norm" in name or "bias" in name:
                    no_decay_params.append(param)
                else:
                    decay_params.append(param)
        
        optim_groups = [
            {"params": decay_params, "weight_decay": self.config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0}
        ]
        
        return AdamW(optim_groups, lr=self.config.learning_rate)
    
    def train_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        self.optimizer.zero_grad()
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, (inputs, targets) in enumerate(progress_bar):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            _, loss = self.model(inputs, targets)
            
            # Scale for gradient accumulation
            loss = loss / self.config.gradient_accumulation_steps
            
            # Backward pass
            loss.backward()
            
            # Update after accumulation
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm
                )
                
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                
                self.global_step += 1
            
            # Track loss
            batch_loss = loss.item() * self.config.gradient_accumulation_steps
            total_loss += batch_loss
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{batch_loss:.4f}',
                'ppl': f'{torch.exp(torch.tensor(batch_loss)):.2f}',
                'lr': f'{self.scheduler.get_last_lr()[0]:.2e}'
            })
        
        return total_loss / num_batches
    
    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> tuple[float, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        
        for inputs, targets in dataloader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            _, loss = self.model(inputs, targets)
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        return avg_loss, perplexity
    
    def save_checkpoint(self, path: Path, is_best: bool = False):
        """Save checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_loss': self.best_loss,
            'config': self.config,
            'model_config': self.model.config
        }
        
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved: {path}")
        
        if is_best:
            best_path = self.config.checkpoint_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            logger.info(f"Best model saved: {best_path}")
    
    def load_checkpoint(self, path: Path, vocab_size: int):
        """Load checkpoint for continuous learning."""
        logger.info(f"Loading checkpoint from {path}")
        checkpoint = torch.load(path, map_location=self.device)
        
        model_config = checkpoint.get('model_config', SaciNanoConfig(vocab_size=vocab_size))
        self.model = SaciNano(model_config).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_loss = checkpoint['best_loss']
        
        logger.info(f"Resumed from epoch {self.current_epoch}, step {self.global_step}")
    
    def train(
        self,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        vocab_size: int = 16384
    ):
        """Main training loop."""
        # Check for existing checkpoint (continuous learning)
        latest_checkpoint = self.config.checkpoint_dir / "latest.pth"
        
        if self.config.resume_from and self.config.resume_from.exists():
            self.load_checkpoint(self.config.resume_from, vocab_size)
        elif latest_checkpoint.exists():
            logger.info("Found existing checkpoint, resuming training...")
            self.load_checkpoint(latest_checkpoint, vocab_size)
        else:
            self.model = self.build_model(vocab_size)
        
        # Build optimizer and scheduler
        total_steps = len(train_dataloader) * self.config.epochs // self.config.gradient_accumulation_steps
        self.optimizer = self.build_optimizer()
        self.scheduler = get_lr_scheduler(
            self.optimizer,
            self.config.warmup_steps,
            total_steps,
            self.config.min_lr_ratio
        )
        
        # Resume optimizer state if checkpoint loaded
        if self.current_epoch > 0:
            if latest_checkpoint.exists():
                ckpt = torch.load(latest_checkpoint, map_location=self.device)
                self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        
        logger.info(f"Starting training from epoch {self.current_epoch + 1}")
        logger.info(f"Total steps: {total_steps:,}")
        
        start_time = time.time()
        
        for epoch in range(self.current_epoch, self.config.epochs):
            self.current_epoch = epoch + 1
            
            # Train
            train_loss = self.train_epoch(train_dataloader)
            train_ppl = torch.exp(torch.tensor(train_loss)).item()
            
            # Validate
            if val_dataloader is not None:
                val_loss, val_ppl = self.validate(val_dataloader)
                logger.info(
                    f"Epoch {self.current_epoch}: "
                    f"train_loss={train_loss:.4f}, train_ppl={train_ppl:.2f}, "
                    f"val_loss={val_loss:.4f}, val_ppl={val_ppl:.2f}"
                )
                
                if val_loss < self.best_loss:
                    self.best_loss = val_loss
                    self.save_checkpoint(
                        self.config.checkpoint_dir / "best_model.pth",
                        is_best=True
                    )
            else:
                logger.info(
                    f"Epoch {self.current_epoch}: "
                    f"train_loss={train_loss:.4f}, train_ppl={train_ppl:.2f}"
                )
            
            # Save latest checkpoint
            self.save_checkpoint(self.config.checkpoint_dir / "latest.pth")
            
            # Periodic checkpointing
            if self.current_epoch % self.config.save_every_n_epochs == 0:
                self.save_checkpoint(
                    self.config.checkpoint_dir / f"checkpoint_epoch_{self.current_epoch}.pth"
                )
        
        # Final checkpoint
        self.save_checkpoint(self.config.checkpoint_dir / "final_model.pth")
        
        elapsed = time.time() - start_time
        logger.info(f"Training complete in {elapsed / 60:.2f} minutes")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Saci-Nano Training",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--mode", "-m",
        type=str,
        default="default",
        help="Training mode preset (see --list-modes)"
    )
    
    parser.add_argument(
        "--list-modes", "-l",
        action="store_true",
        help="List available training modes and exit"
    )
    
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Start fresh, ignore existing checkpoints"
    )
    
    return parser.parse_args()


def main():
    """Main training entry point."""
    args = parse_args()
    
    # List modes and exit
    if args.list_modes:
        print("\nAvailable training modes:")
        print("-" * 50)
        modes = TrainerConfig.list_available_modes()
        for mode in modes:
            print(mode)
        print("-" * 50)
        print(f"\nConfig file: {MODES_CONFIG_PATH}")
        print("Edit this file to add/modify modes.\n")
        return
    
    # Load config from mode
    try:
        config = TrainerConfig.from_yaml_mode(args.mode)
    except ValueError as e:
        logger.error(str(e))
        return
    
    # Handle --no-resume
    if args.no_resume:
        latest_ckpt = CHECKPOINT_DIR / "latest.pth"
        if latest_ckpt.exists():
            latest_ckpt.unlink()
            logger.info("Removed existing checkpoint (--no-resume)")
    
    # Load tokenizer
    try:
        tokenizer = FinancialTokenizer.load(TOKENIZER_PATH)
        vocab_size = tokenizer.vocab_size
        logger.info(f"Tokenizer loaded. Vocab size: {vocab_size}")
    except FileNotFoundError:
        logger.warning("Tokenizer not found. Using default vocab size 16384")
        vocab_size = 16384
    
    try:
        # Create data loader
        train_loader = create_dataloader(
            data_dir=DATA_TOKENIZED_DIR,
            seq_len=config.seq_len,
            batch_size=config.batch_size
        )
        
        # Create trainer and train
        trainer = Trainer(config)
        trainer.train(train_loader, vocab_size=vocab_size)
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()
