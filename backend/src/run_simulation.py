
import torch
import os
import time
from src.trading.simulation import MarketSimulator
from src.model.iara_liquid import IARA, LiquidConfig
from dataclasses import dataclass
from pathlib import Path

@dataclass
class TrainerConfig:
    # Minimal config to satisfy unpickler
    seq_len: int = 256
    stride: int = 2
    batch_size: int = 1024
    accumulation_steps: int = 1
    num_workers: int = 8
    val_split: float = 0.2
    input_dim: int = 16
    hidden_dim: int = 128
    num_heads: int = 4
    dropout: float = 0.5
    max_seq_len: int = 256
    num_cfc_layers: int = 2
    epochs: int = 200
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    grad_clip: float = 0.5
    resume: bool = False
    label_smooth: float = 0.1
    noise_std: float = 0.02
    scale_min: float = 0.98
    scale_max: float = 1.02
    patience: int = 150
    target_lookahead: int = 5
    checkpoint_dir: Path = Path("checkpoints")
    asymmetry_penalty: float = 2.5
    error_memory_size: int = 500

# Config
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_PATH = "checkpoints/best_model.pth"

def main():
    print(f"🚀 Starting SACI Protocol Simulation (Automated)")
    print(f"Device: {DEVICE}")

    # 1. Load Model
    if not os.path.exists(CHECKPOINT_PATH):
        print("❌ Checkpoint not found. Please train first.")
        return

    print("Loading Model...")
    config = LiquidConfig(input_dim=18, hidden_dim=128)
    model = IARA(config).to(DEVICE)
    try:
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            model.load_state_dict(checkpoint["model"])
        else:
            model.load_state_dict(checkpoint)
        model.eval()
        print("✅ Model Loaded.")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    # 2. Init Simulator
    sim = MarketSimulator(model, ticker="EURUSD", device=DEVICE)
    
    # 3. Run Loop
    print("running simulation steps...")
    steps = 50
    for i in range(steps):
        state = sim.step()
        if state:
            status = "WAIT"
            if "BUY" in str(state.logic_log): status = "BUY"
            elif "SELL" in str(state.logic_log): status = "SELL"
            
            print(f"Step {i+1}/{steps} | Prob: {state.prob:.1%} | Action: {status}")
            time.sleep(0.05) # simulate real-time
        else:
            print("End of Data.")
            break
            
    print(f"✅ Simulation Complete. {steps} BrainStates persisted to TimescaleDB.")

if __name__ == "__main__":
    main()
