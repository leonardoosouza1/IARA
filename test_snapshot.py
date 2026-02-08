
import torch
import sys
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sys.path.append(os.getcwd())

from src.brain.train_v9 import save_forensic_snapshot, SwarmBrain, TrainConfig

def test_snapshot():
    print("Testing Forensic Snapshot Generation...")
    
    # Mock Config
    config = TrainConfig()
    device = "cpu"
    
    # Mock Model
    model = SwarmBrain(config).to(device)
    model.eval()
    
    # Mock Data
    batch_size = 4
    seq_len = 60
    input_dim = 16 # Approx
    
    x_dict = {
        "M5": torch.randn(batch_size, seq_len, input_dim).to(device),
        "M15": torch.randn(batch_size, seq_len, input_dim).to(device),
        "H1": torch.randn(batch_size, seq_len, input_dim).to(device),
        "H4": torch.randn(batch_size, seq_len, input_dim).to(device),
        "D1": torch.randn(batch_size, seq_len, input_dim).to(device),
    }
    targets = torch.randint(0, 2, (batch_size, 1)).float().to(device)
    
    # Mock Loader (yields one batch)
    class MockLoader:
        def __iter__(self):
            yield x_dict, targets
            
    loader = MockLoader()
    
    # Try to save
    try:
        save_forensic_snapshot(model, loader, device, 999, config)
        print("✅ Snapshot saved successfully!")
    except Exception as e:
        print(f"❌ Failed to save snapshot: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_snapshot()
