import torch
from src.brain.models.liquid_v9 import SwarmBrain, WorkerConfig, CoordinatorConfig

def test_architecture():
    print("🚀 Testing IARA v10 Hybrid Architecture Upgrades...")
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  🔹 Device: {device}")
    
    # 1. Initialize Model
    worker_cfg = WorkerConfig(hidden_dim=64)
    model = SwarmBrain(worker_config=worker_cfg).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  🔹 Model initialized. Total params: {total_params:,}")
    
    # 2. Preparation of Dummy Data
    batch_size = 4
    seq_len = 256
    features = 12
    timeframes = ["M5", "M15", "M30", "H1", "H4"]
    
    x_dict = {
        tf: torch.randn(batch_size, seq_len, features).to(device)
        for tf in timeframes
    }
    
    # Add optional dt if we want to test that too
    for tf in timeframes:
        x_dict[f"{tf}_dt"] = torch.rand(batch_size, seq_len, 1).to(device)
    
    print(f"  🔹 Dummy data prepared (Batch={batch_size}, Seq={seq_len})")
    
    # 3. Forward Pass
    print("  🔹 Running forward pass...")
    logits, worker_logits, long_score, short_score = model(x_dict)
    
    # 4. Verify Shapes
    print(f"  🔹 Output shapes:")
    print(f"    - Final Logits: {logits.shape}")
    print(f"    - Worker Logits: {worker_logits.shape}")
    
    assert logits.shape == (batch_size, 1)
    assert worker_logits.shape == (batch_size, 5, 3)
    
    print("\n✅ Architecture Verification SUCCESSFUL!")
    print("Final Logits (sample):", logits[0].item())

if __name__ == "__main__":
    test_architecture()
