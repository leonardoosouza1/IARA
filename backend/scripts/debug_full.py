import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim

print("🚀 Starting Full Debug", flush=True)
sys.path.append(os.getcwd())

try:
    print("1. Importing...", flush=True)
    from src.brain.models.liquid_v8 import LiquidConfig, ModulatedIARA
    from src.brain.models.trinity_liquid import TrinityLiquidBrain
    print("✅ Imported", flush=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"2. Device: {device}", flush=True)
    
    print("3. Creating Data...", flush=True)
    inputs = torch.randn(16, 64, 18).to(device)
    targets = (inputs[:, -1, 0] > 0).float()
    print("✅ Data Created", flush=True)
    
    print("4. Instantiating Model...", flush=True)
    cfg = LiquidConfig(input_dim=18, hidden_dim=32, max_seq_len=64)
    lnn = ModulatedIARA(cfg)
    model = TrinityLiquidBrain(lnn, lfm_hidden_dim=4096).to(device)
    print("✅ Model Instantiated", flush=True)
    
    print("5. Instantiating Criterion...", flush=True)
    criterion = nn.BCEWithLogitsLoss()
    print("✅ Criterion Instantiated", flush=True)
    
    print("6. Instantiating Optimizer...", flush=True)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    print("✅ Optimizer Instantiated", flush=True)
    
    print("7. Starting Loop...", flush=True)
    for step in range(100):
        llm_ctx = torch.zeros(16, 4096).to(device)
        news_vec = torch.zeros(16, 2).to(device)
        
        optimizer.zero_grad()
        
        # Forward
        outputs = model(inputs, llm_ctx, news_vec)
        if outputs.dim() > 1:
            outputs = outputs.squeeze(-1)
            
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        if step % 10 == 0:
             preds = (torch.sigmoid(outputs) > 0.5).float()
             acc = (preds == targets).float().mean().item()
             print(f"Step {step}: Loss {loss.item():.4f} | Acc {acc:.2%}", flush=True)
    
    print("🎉 FULL SUCCESS", flush=True)

except Exception as e:
    print(f"❌ Error: {e}", flush=True)
    import traceback
    traceback.print_exc()
