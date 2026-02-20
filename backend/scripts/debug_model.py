import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os

# Ensure src is in path
sys.path.append(os.getcwd())

from src.brain.models.trinity_liquid import TrinityLiquidBrain
from src.brain.models.liquid_v8 import ModulatedIARA, LiquidConfig

def debug_overfit():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Debugging on {device}", flush=True)
    
    # 1. create dummy data (Batch=1, Seq=64, Feat=18)
    torch.manual_seed(42)
    inputs = torch.randn(1, 64, 18).to(device)
    # Simple pattern: if last value > 0 -> 1
    targets = (inputs[:, -1, 0] > 0).float()
    
    print(f"Targets Mean: {targets.mean().item()}", flush=True)
    
    # 2. Init Model
    INPUT_DIM = 18 
    liq_cfg = LiquidConfig(input_dim=INPUT_DIM, hidden_dim=128, max_seq_len=64)
    lnn = ModulatedIARA(liq_cfg)
    model = TrinityLiquidBrain(lnn, lfm_hidden_dim=4096).to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    print("\n🔄 Starting Overfit Test (100 steps)...", flush=True)
    for step in range(100):
        print(f"--- Step {step} Start ---", flush=True)
        llm_ctx = torch.zeros(1, 4096).to(device)
        news_vec = torch.zeros(1, 2).to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs, llm_ctx, news_vec).squeeze(-1)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        preds = (torch.sigmoid(outputs) > 0.5).float()
        acc = (preds == targets).float().mean().item()
        print(f"Step {step}: Loss {loss.item():.4f} | Acc {acc:.2%}", flush=True)
            
    print("\n✅ Final Check:", flush=True)
    with torch.no_grad():
        outputs = model(inputs, llm_ctx, news_vec).squeeze(-1)
        probs = torch.sigmoid(outputs)
        print(f"Probs: {probs[:5].cpu().numpy()}", flush=True)
        print(f"Targets: {targets[:5].cpu().numpy()}", flush=True)

if __name__ == "__main__":
    try:
        debug_overfit()
    except Exception as e:
        import traceback
        traceback.print_exc()
