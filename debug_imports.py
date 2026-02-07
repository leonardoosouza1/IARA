import sys
import os
import torch

print("🚀 Starting Import Test", flush=True)
sys.path.append(os.getcwd())

try:
    print("1. Importing LiquidConfig...", flush=True)
    from src.brain.models.liquid_v8 import LiquidConfig
    print("✅ LiquidConfig Imported", flush=True)
    
    print("2. Instantiating Config...", flush=True)
    cfg = LiquidConfig(input_dim=18, hidden_dim=32, max_seq_len=64)
    print("✅ Config Instantiated", flush=True)
    
    print("3. Importing ModulatedIARA...", flush=True)
    from src.brain.models.liquid_v8 import ModulatedIARA
    print("✅ ModulatedIARA Imported", flush=True)

    print("4. Instantiating ModulatedIARA...", flush=True)
    lnn = ModulatedIARA(cfg)
    print("✅ ModulatedIARA Instantiated", flush=True)
    
    print("5. Moving to CUDA...", flush=True)
    if torch.cuda.is_available():
        lnn = lnn.cuda()
        print("✅ Moved to CUDA", flush=True)
    else:
        print("⚠️ CUDA not available", flush=True)

except Exception as e:
    print(f"❌ Error: {e}", flush=True)
    import traceback
    traceback.print_exc()
