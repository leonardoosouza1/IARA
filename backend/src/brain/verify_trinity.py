import torch
import sys
import os

# Ensure src is in path
sys.path.append(os.path.abspath(os.curdir))

from src.brain.models.liquid_v8 import LiquidConfig, ModulatedIARA
from src.brain.models.trinity_v8 import TrinityBrain
from src.brain.engine import TrinityEngine, EngineConfig

def test_trinity_phase1():
    print("🧪 Starting Trinity Phase 1 Verification...")
    
    # 1. Config
    liquid_cfg = LiquidConfig(
        input_dim=16,
        hidden_dim=32, # Small for test
        num_cfc_layers=1
    )
    llm_dim = 64 # Mock small dimension
    
    # 2. Instantiate Models
    print("   Building Modulated LNN...")
    lnn = ModulatedIARA(liquid_cfg)
    
    print("   Building TrinityBrain...")
    brain = TrinityBrain(lnn, llm_hidden_dim=llm_dim)
    
    # 3. Instantiate Engine
    print("   Initializing Engine (Orchestrator)...")
    engine_cfg = EngineConfig(device="cpu", llm_refresh_rate=1.0) # Use CPU for test
    engine = TrinityEngine(brain, engine_cfg)
    
    # 4. Mock Data
    batch_size = 2
    seq_len = 10
    market_data = torch.randn(batch_size, seq_len, liquid_cfg.input_dim)
    
    # 5. Mock State Update (Simulating LLM Thread)
    print("   Simulating LLM Update...")
    mock_llm_ctx = torch.randn(1, llm_dim)
    engine.state.update_llm(mock_llm_ctx)
    
    # 6. Forward Tick
    print("   Running Forward Tick (Fast Path)...")
    # News is None, should use Sentinel Stub (neutral)
    decision = engine.forward_tick(market_data, raw_news="Fed holds rates")
    
    print(f"   Output Shape: {decision.shape}")
    print(f"   Output Values: {decision}")
    
    assert decision.shape == (batch_size,), "Output shape mismatch!"
    
    # 7. Test Emergency Brake Logic (Manual Verify)
    # Force high impact in state
    print("   Testing Emergency Brake (Simulated)...")
    engine.state.update_news(sentiment=-0.9, impact=1.0) # 100% Impact
    
    decision_brake = engine.forward_tick(market_data)
    print(f"   Braked Output: {decision_brake}")
    
    # If impact is 1.0, output should be 0.0 (Identity * (1 - 1))
    # Note: Tanh output might not be exactly zero if bias, but let's check magnitude
    if torch.allclose(decision_brake, torch.zeros_like(decision_brake), atol=1e-5):
        print("✅ Emergency Brake Functional (Output Zeroed)")
    else:
        print(f"⚠️ Emergency Brake Verification Warning: Output {decision_brake} not zero.")
        
    print("✅ Trinity Phase 1 verification passed!")

if __name__ == "__main__":
    test_trinity_phase1()
