"""
PURE LIQUID TRINITY v8.5 - LIVE DEMO
====================================
This script demonstrates the "Think-Query-Decide" loop of the Strategic Agent (LFM).
It simulates a market feed and connects to Ollama to show the AI 'thinking' and using tools.

Usage:
    python src/demo_pure_liquid.py
"""

import time
import torch
import logging
import sys
import os
import pandas as pd
import numpy as np

# Add src to path
sys.path.append(os.getcwd())

from src.brain.models.liquid_v8 import LiquidConfig, ModulatedIARA
from src.brain.models.trinity_liquid import TrinityLiquidBrain
from src.brain.engine import TrinityEngine, EngineConfig

# Configure nice logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("Demo")

def generate_mock_data(length=1000):
    """Generates a random walk for simulation."""
    price = 1.0850
    data = []
    prices = []
    
    for _ in range(length):
        change = np.random.normal(0, 0.0001)
        price += change
        prices.append(price)
        
        # Mock features: [NormPrice, RSI, MACD, BB_Up, LogRet, Vol] (size 18)
        feats = np.random.randn(18)
        feats[0] = (price - 1.08) / 0.01 # Mock norm price
        data.append(feats)
        
    return prices, np.array(data)

def main():
    print("\n💧 INITIALIZING PURE LIQUID TRINITY (v8.5) 💧")
    print("===============================================")
    
    # 1. Config
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔧 Hardware: {device.upper()}")
    
    # Check Ollama
    ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
    print(f"🧠 Strategic Agent: Connecting to {ollama_url}...")
    
    # 2. Initialize Neural Core (Tactical)
    print("⚡ Initializing Tactical LNN...")
    lnn_cfg = LiquidConfig(input_dim=18, hidden_dim=128)
    lnn_core = ModulatedIARA(lnn_cfg)
    
    # 3. Initialize Trinity Brain (Fusion)
    # LFM Output Dim: 4096 (standard Llama) -> Projected to 18 (LNN Input)
    brain = TrinityLiquidBrain(lnn_core, lfm_hidden_dim=4096).to(device)
    
    # 4. Start Engine (Strategic)
    engine_cfg = EngineConfig(
        lfm_model_url=ollama_url,
        lfm_model_name="lfm2.5-thinking:1.2b", # Ensure this matches what user pulled
        device=device,
        lfm_refresh_rate=10.0 # Fast thinking for demo (every 10s)
    )
    
    engine = TrinityEngine(brain, engine_cfg)
    engine.start()
    
    # 5. Simulation Loop
    prices, features = generate_mock_data()
    print("\n▶️  STARTING MARKET SIMULATION")
    print("   (Watch the logs for '🧠 LFM' activity...)\n")
    
    try:
        for i in range(len(prices)):
            price = prices[i]
            
            # Feed input to engine (Mock 1-step sequence)
            # Shape: [Batch=1, Seq=1, Feat=18]
            feat_tensor = torch.tensor(features[i]).float().unsqueeze(0).unsqueeze(0).to(device)
            
            # Tactical Execution (Fast)
            # The engine automatically injects the latest Strategic Context
            decision = engine.forward_tick(feat_tensor)
            
            # Update Engine State (so LFM sees price)
            engine.state.update_price(price)
            
            # Visual feedback every tick
            if i % 10 == 0:
                print(f"\rTick {i}: EURUSD {price:.5f} | Tactical Signal: {decision.item():.4f}", end="")
            
            time.sleep(0.1) # Simulate real-time (10 ticks/sec)
            
    except KeyboardInterrupt:
        print("\n\n🛑 Stopping Demo...")
    finally:
        engine.stop()
        print("✅ Engine Shutdown.")

if __name__ == "__main__":
    main()
