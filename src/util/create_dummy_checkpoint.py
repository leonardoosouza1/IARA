
import sys
import torch
from pathlib import Path

# Add src to path
sys.path.append("/app")

from src.model.iara_liquid import IARA, LiquidConfig
from src.train.trainer_liquid import TrainerConfig

def create_checkpoint():
    print("Creating Dummy Checkpoint...")
    
    config = TrainerConfig()
    
    liquid_config = LiquidConfig(
        input_dim=config.input_dim,
        hidden_dim=config.hidden_dim,
        num_heads=config.num_heads,
        dropout=config.dropout,
        max_seq_len=config.max_seq_len
    )
    
    model = IARA(liquid_config)
    
    # Checkpoint Dir
    checkpoint_dir = Path("/app/checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    path = checkpoint_dir / "best_model.pth"
    
    torch.save({
        'model': model.state_dict(),
        'config': config,
        'epoch': 0,
        'best_val_auc': 0.5
    }, path)
    
    print(f"Saved: {path}")

if __name__ == "__main__":
    create_checkpoint()
