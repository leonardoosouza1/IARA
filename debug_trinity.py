import sys
import os
import torch

print("🚀 Starting Trinity Test", flush=True)
sys.path.append(os.getcwd())

try:
    print("1. Importing Liquid/Trinity...", flush=True)
    from src.brain.models.liquid_v8 import LiquidConfig, ModulatedIARA
    from src.brain.models.trinity_liquid import TrinityLiquidBrain
    print("✅ Modules Imported", flush=True)
    
    print("2. Instantiating Trinity...", flush=True)
    cfg = LiquidConfig(input_dim=18, hidden_dim=32, max_seq_len=64)
    lnn = ModulatedIARA(cfg)
    model = TrinityLiquidBrain(lnn, lfm_hidden_dim=4096)
    print("✅ Trinity Instantiated", flush=True)

    print("3. Moving to CUDA...", flush=True)
    if torch.cuda.is_available():
        model = model.cuda()
        print("✅ Moved to CUDA", flush=True)
    
    print("4. Running Dummy Forward...", flush=True)
    inputs = torch.randn(2, 64, 18).cuda()
    ctx = torch.zeros(2, 4096).cuda()
    news = torch.zeros(2, 2).cuda()
    
    out = model(inputs, ctx, news)
    print(f"✅ Forward Pass Complete. Output Shape: {out.shape}", flush=True)

except Exception as e:
    print(f"❌ Error: {e}", flush=True)
    import traceback
    traceback.print_exc()
