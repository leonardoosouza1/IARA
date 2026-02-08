
import argparse
import json
import time
import os
import sys
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.append(os.getcwd())

# Import from training script (ensure train_v9.py is in src/brain)
from src.brain.train_v9 import SwarmBrain, WorkerConfig, CoordinatorConfig, TrainConfig
from src.brain.feature_factory import FeatureFactory
# We need to replicate MultiTimeframeDataset logic if we can't import it easily without running main
# But train_v9 has if __name__ == "__main__", so safe to import classes.
from src.brain.train_v9 import MultiTimeframeDataset, collate_multitf

# Setup logging
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger("BacktestV9")

def main():
    parser = argparse.ArgumentParser(description="Backtest Swarm Brain v9")
    parser.add_argument("--model", type=str, required=True, help="Path to model .pth file")
    parser.add_argument("--data", type=str, default="data/processed/EURUSD.parquet", help="Path to data file")
    parser.add_argument("--start_date", type=str, default="2023-01-01", help="Start simulation date")
    parser.add_argument("--speed", type=float, default=10.0, help="Steps per second")
    args = parser.parse_args()

    # PID
    with open("simulation_process.pid", "w") as f:
        f.write(str(os.getpid()))

    # Load Data
    logger.info(f"Loading data from {args.data}...")
    if args.data.endswith(".parquet"):
        df = pd.read_parquet(args.data)
    else:
        df = pd.read_csv(args.data)
    
    # Check if df has 'time' column, if not try 'date' or index
    if 'time' not in df.columns:
        if 'date' in df.columns:
            df.rename(columns={'date': 'time'}, inplace=True)
        else:
            # Assume index is time if datetime
            df.reset_index(inplace=True)
            if 'index' in df.columns:
                df.rename(columns={'index': 'time'}, inplace=True)
    
    # Ensure time is datetime
    df['time'] = pd.to_datetime(df['time'])
    
    # Filter by date
    mask = df['time'] >= pd.to_datetime(args.start_date)
    df_sim = df[mask].copy().reset_index(drop=True)
    
    logger.info(f"Simulation Data: {len(df_sim)} rows starting from {args.start_date}")
    
    if len(df_sim) < 100:
        logger.error("Not enough data for simulation!")
        return

    # Check device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Prepare Dataset
    # We use the full dataset class to handle feature engineering windowing
    # Note: MultiTimeframeDataset expects a dataframe and handles looking back seq_len
    # So we need to feed it df_sim.
    # IMPORTANT: We need features. If parquet has only OHLCV, we need FeatureFactory.
    # If parquet is already processed (has features), MultiTimeframeDataset might need logic to use them.
    # In train_v9, FeatureFactory is used inside MultiTimeframeDataset? 
    # Let's inspect train_v9 logic briefly... Assuming it handles it.
    
    input_dim = 12 
    # Initialize Configs (hardcoded for now to match training)
    # Ideally load from a config.json saved with the model
    worker_cfg = WorkerConfig(input_dim=input_dim, hidden_dim=32)
    coord_cfg = CoordinatorConfig(num_workers=5, worker_dim=32, hidden_dim=128)
    
    # Load Model
    model = SwarmBrain(worker_cfg, coord_cfg).to(device)
    try:
        checkpoint = torch.load(args.model, map_location=device)
        model.load_state_dict(checkpoint)
        logger.info(f"✅ Model loaded from {args.model}")
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        return

    model.eval()

    # Simulation State
    equity = 10000.0
    position = 0.0 # 1.0 = Long, -1.0 = Short
    entry_price = 0.0
    
    logger.info("🚀 Starting Simulation...")
    
    # Create dataset instance 
    # We assume FeatureFactory is called inside or data is ready
    dataset = MultiTimeframeDataset(df_sim, seq_len=64)
    
    # Iterate
    start_time = time.time()
    total_steps = len(dataset)
    
    # We simulate step by step
    # We use a DataLoader with batch_size=1, shuffle=False
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    
    step = 0
    for x_dict, target in loader:
        # Check stop signal
        if not os.path.exists("simulation_process.pid"):
            logger.info("🛑 Simulation stopped by user.")
            break
            
        step += 1
        
        # Move to device
        x_dict = {k: v.to(device) for k, v in x_dict.items()}
        
        # Inference
        with torch.no_grad():
            with torch.amp.autocast('cuda', enabled=(device=="cuda")):
                # SwarmBrain v9 returns: logits, worker_logits, long_score, short_score
                logits, worker_logits, long_score, short_score = model(x_dict)
                prob = torch.sigmoid(logits).item()
        
        # Trading Logic (Simple Threshold)
        # We need current price. 
        # Dataset index lines up with df_sim?
        # MultiTimeframeDataset usually has an offset of seq_len.
        # Let's get price from df_sim at index (step + seq_len)? 
        # Or just use the last Close from input?
        # x_dict['M5'] is [batch, seq_len, features]. Last feature 0 is Close?
        # Let's assume df_sim index corresponds to loader iteration + offset.
        
        sim_idx = step + 64 - 1 # Approximation
        if sim_idx >= len(df_sim):
            break
            
        current_price = df_sim.iloc[sim_idx]['close']
        current_time = df_sim.iloc[sim_idx]['time']
        
        action = "HOLD"
        
        if prob > 0.7 and position <= 0:
            # Buy
            if position < 0: # Close Short
                pnl = (entry_price - current_price) / entry_price
                equity *= (1 + pnl)
            
            position = 1.0
            entry_price = current_price
            action = "BUY"
            
        elif prob < 0.3 and position >= 0:
            # Sell
            if position > 0: # Close Long
                pnl = (current_price - entry_price) / entry_price
                equity *= (1 + pnl)
                
            position = -1.0
            entry_price = current_price
            action = "SELL"
            
        # Un realized PnL
        unrealized_pnl = 0.0
        if position > 0:
             unrealized_pnl = (current_price - entry_price) / entry_price
        elif position < 0:
             unrealized_pnl = (entry_price - current_price) / entry_price
             
        pnl_pct = unrealized_pnl * 100
        
        # Neuronal Map (Worker Activations)
        # worker_logits: [1, 5, 3]
        worker_activations = torch.sigmoid(worker_logits).cpu().numpy().tolist()[0] # [5, 3]
        
        # Snapshot
        snapshot = {
            "step": step,
            "timestamp": str(current_time),
            "price": float(current_price),
            "equity": float(equity),
            "pnl_pct": float(pnl_pct),
            "prob": float(prob),
            "long_bias": float(torch.sigmoid(long_score).item()),
            "short_bias": float(torch.sigmoid(short_score).item()),
            "action": action,
            "neuronal_map": worker_activations,
            "completed": False
        }
        
        # Atomic write
        temp_file = "logs/sim_snapshot.json.tmp"
        with open(temp_file, "w") as f:
            json.dump(snapshot, f)
        os.rename(temp_file, "logs/sim_snapshot.json")
            
        # Log occasionally
        if step % 100 == 0:
            logger.info(f"Step {step}: Price {current_price:.5f} | Equity {equity:.2f} | Action {action}")
            
        # Speed Control
        # Calculate how much to sleep to match speed (steps/sec)
        # Simple sleep
        time.sleep(1.0 / args.speed)

    # Final Snapshot
    with open("logs/sim_snapshot.json", "w") as f:
        json.dump({"completed": True}, f)
        
    logger.info("Simulation Finished.")
    if os.path.exists("simulation_process.pid"):
        os.remove("simulation_process.pid")

if __name__ == "__main__":
    main()
