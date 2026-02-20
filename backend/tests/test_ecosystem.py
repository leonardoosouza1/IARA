import sys
from pathlib import Path
from datetime import datetime
import torch

# Add src to path
# sys.path.append("/app") - Removed, rely on PYTHONPATH

from src.analysis.active_learning import ActiveLearner
from src.trading.signal_generator import SignalEngine

def test_adaptation():
    print("\n--- Testing Active Learning Adaptation ---")
    learner = ActiveLearner()
    initial_thresh = learner.get_adaptive_threshold()
    print(f"Initial Threshold: {initial_thresh}")
    
    # Simulate a LOSING STREAK (20 Losses) -> Should Tighten Threshold
    print("Simulating 20 Losses...")
    for _ in range(20):
        learner.log_trade("TEST", "BUY", 0.60, datetime.now().isoformat(), outcome=0.0)
        
    tight_thresh = learner.get_adaptive_threshold()
    print(f"Threshold after Losing Streak: {tight_thresh} (Should be > {initial_thresh})")
    
    # Simulate a WINNING STREAK (30 Wins) -> Should Loosen Threshold
    print("Simulating 30 Wins...")
    for _ in range(30):
        learner.log_trade("TEST", "BUY", 0.60, datetime.now().isoformat(), outcome=1.0)
        
    loose_thresh = learner.get_adaptive_threshold()
    print(f"Threshold after Winning Streak: {loose_thresh} (Should be < {tight_thresh})")

def test_signal_generation():
    print("\n--- Testing Signal Generator (Live Inference) ---")
    
    # Find a parquet file
    data_dir = Path("/app/data/raw")
    files = list(data_dir.glob("*.parquet"))
    if not files:
        print("No data file found to test!")
        return
        
    test_file = files[0]
    ticker = test_file.stem.split("_")[0]
    print(f"Testing on file: {test_file}")
    
    # Checkpoint (use best or latest, or mock if still training)
    checkpoint = Path("/app/checkpoints/best_model.pth")
    if not checkpoint.exists():
        checkpoint = Path("/app/checkpoints/latest.pth")
    
    if not checkpoint.exists():
        print("No checkpoint found yet (Training might clearly started). Skipping inference test.")
        return

    try:
        engine = SignalEngine(checkpoint_path=str(checkpoint))
        signal = engine.predict(ticker, test_file)
        
        if signal:
            print(f"Signal Generated Successfully: {signal}")
        else:
            print("Signal generation returned None (possibly error inside predict)")
            
    except Exception as e:
        print(f"Signal Generation Failed: {e}")

if __name__ == "__main__":
    test_adaptation()
    test_signal_generation()
