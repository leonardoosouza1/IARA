
import sys
import os
import asyncio
import torch
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "backend"))

from src.api.sensory_hub import AlpacaSensoryHub
from src.trading.signal_generator import SignalEngine
from src.api.market_data import MarketDataService

async def test_sensory_hub():
    print("--- Testing AlpacaSensoryHub ---")
    hub = AlpacaSensoryHub("fake_key", "fake_secret")
    
    # Mock Orderbook data
    mock_data = MagicMock()
    mock_data.symbol = "BTC/USD"
    mock_data.bids = [MagicMock(price=96000 + i*10, size=1.0 + i) for i in range(10)]
    mock_data.asks = [MagicMock(price=96100 + i*10, size=1.0 + i) for i in range(10)]
    mock_data.timestamp = pd.Timestamp.now()
    
    # Process it
    await hub._on_orderbook(mock_data)
    
    # Get vector
    vector = hub.get_sensory_vector("BTC/USD")
    print(f"Sensory Vector Shape: {vector.shape}")
    print(f"VBI (Index 40): {vector[40]:.4f}")
    print(f"Cancel Rate (Index 41): {vector[41]:.4f}")
    
    assert vector.shape == (42,)
    print("✅ SensoryHub Vector Generation OK")
    return hub

async def test_signal_engine_integration(hub):
    print("\n--- Testing SignalEngine Integration ---")
    
    # Create a dummy M5 OHLCV dataset
    data = {
        "time": pd.date_range(start="2024-01-01", periods=200, freq="5min"),
        "open": np.random.randn(200) + 96000,
        "high": np.random.randn(200) + 96100,
        "low": np.random.randn(200) + 95900,
        "close": np.random.randn(200) + 96000,
        "volume": np.random.randint(100, 1000, 200)
    }
    df = pd.DataFrame(data)
    
    # Enrich the last candle with sensory_l2 (simulating MarketDataService logic)
    df["sensory_l2"] = [None] * 200
    vector = hub.get_sensory_vector("BTC/USD")
    df.at[199, "sensory_l2"] = vector.tolist()
    
    # Mock a v10 checkpoint path to trigger v10 logic
    checkpoint_path = "dummy_v10_model.pth"
    
    # Mock torch.load to return a dummy state dict for v10
    dummy_state = {
        "swarm.workers.M5.input_proj.weight": torch.randn(64, 54) # 12 base + 42 sensory
    }
    
    with patch("torch.load", return_value=dummy_state):
        try:
            # We don't need to actually load weights, just test build_feature_dict
            engine = SignalEngine(checkpoint_path, worker_hidden=64, coord_hidden=256, seq_len=128)
            
            x_dict = engine.build_feature_dict(df)
            
            m5_features = x_dict["M5"]
            print(f"M5 feature tensor shape: {m5_features.shape}")
            
            # Expected shape: [1, seq_len, 54]
            assert m5_features.shape == (1, 128, 54)
            print("✅ SignalEngine Feature Merging OK")
            
        except Exception as e:
            print(f"❌ SignalEngine Test FAILED: {e}")
            import traceback
            traceback.print_exc()

async def main():
    hub = await test_sensory_hub()
    await test_signal_engine_integration(hub)

if __name__ == "__main__":
    asyncio.run(main())
