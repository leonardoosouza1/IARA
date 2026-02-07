import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import argparse
import time
import json
import os
import sys
from datetime import datetime

# Import Architecture
# Import Architecture
from src.brain.models.liquid_v8 import LiquidConfig, ModulatedIARA
from src.brain.models.trinity_v8 import TrinityBrain
try:
    from src.trading.active_inference import ActiveInferenceModule
except ImportError:
    ActiveInferenceModule = None

# Constants
LOG_DIR = "logs"
SIM_TELEMETRY_FILE = os.path.join(LOG_DIR, "sim_telemetry.json")
SIM_SNAPSHOT_FILE = os.path.join(LOG_DIR, "sim_snapshot.json")
SIM_PID_FILE = "simulation_process.pid"

# Legacy Compatibility
class TrainerConfig:
    pass

def load_data(csv_path, start_date=None, end_date=None):
    print(f"📂 Loading Data: {csv_path}")
    df = pd.read_csv(csv_path)
    df.columns = [c.lower().strip() for c in df.columns]
    
    # Date parsing
    if "date" in df.columns:
        df["timestamp"] = pd.to_datetime(df["date"])
    elif "time" in df.columns:
        df["timestamp"] = pd.to_datetime(df["time"])
    else:
        df["timestamp"] = pd.date_range(end=datetime.now(), periods=len(df), freq="15min")
        
    df = df.sort_values("timestamp").reset_index(drop=True)
    
    # Filtering
    if start_date:
        df = df[df["timestamp"] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df["timestamp"] <= pd.to_datetime(end_date)]
        
    if len(df) == 0:
        raise ValueError("Date range filtered out all data!")
        
    print(f"✅ Data Loaded: {len(df)} candles.")
    return df

def feature_engineering(df):
    # Quick FE matching training logic
    # Ideally should be a shared library, but for now copying logic for robustness
    delta = df["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-6)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50) / 100.0
    
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    macd = (macd - macd.mean()) / (macd.std() + 1e-6)
    
    sma20 = df["close"].rolling(window=20).mean()
    std20 = df["close"].rolling(window=20).std()
    bb_upper = (df["close"] - sma20) / (2 * std20 + 1e-6)
    bb_upper = bb_upper.fillna(0)
    
    log_ret = np.log(df["close"] / df["close"].shift(1)).fillna(0)
    volatility = log_ret.rolling(window=20).std().fillna(0)
    
    # Normalize Close
    norm_close = (df["close"] - df["close"].mean()) / (df["close"].std() + 1e-6)
    
    # Stack 6 base features
    base_feats = np.column_stack([norm_close, rsi, macd, bb_upper, log_ret, volatility])
    
    # Pad to 18 (Input Dim)
    padding = np.zeros((len(df), 12))
    full_feats = np.column_stack([base_feats, padding])
    
    return full_feats

def run_simulation(model_path, data_path, start_date, speed, seq_len=256):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Starting Simulation on {device}...")
    
    # 1. Load Checkpoint First (to get config)
    try:
        # Weights_only=False for legacy support
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    except Exception as e:
        print(f"❌ Failed to load checkpoint: {e}")
        return

    # 2. Determine Config
    input_dim = 18
    hidden_dim = 128
    
    if "config" in checkpoint:
        cfg = checkpoint["config"]
        # If it's a TrainerConfig object, extract attr
        if hasattr(cfg, 'input_dim'):
            input_dim = cfg.input_dim
            hidden_dim = cfg.hidden_dim
            print(f"ℹ️ Loaded Config from Checkpoint: Input={input_dim}, Hidden={hidden_dim}")
            
    # 3. Init Model
    liq_cfg = LiquidConfig(input_dim=input_dim, hidden_dim=hidden_dim, max_seq_len=seq_len)
    lnn = ModulatedIARA(liq_cfg)
    model = TrinityBrain(lnn, llm_hidden_dim=32).to(device)
    
    # 4. Load State Dict
    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint
    
    try:
        model.load_state_dict(state_dict, strict=False)
        print(f"✅ Model Weights Loaded: {os.path.basename(model_path)}")
    except Exception as e:
         print(f"⚠️ Weight loading warning (mismatch?): {e}")

    model.eval()
    
    # hooks for neurons
    hidden_states = []
    def hook_cfc(module, input, output):
        hidden_states.append(output.detach().cpu().numpy())
    
    # Hook registration (safe check)
    if len(model.lnn.cfc_layers) > 0:
        handle = model.lnn.cfc_layers[-1].register_forward_hook(hook_cfc)
    
    # 5. Load Data
    df = load_data(data_path, start_date)
    features = feature_engineering(df)
    
    # 6. Active Inference Init
    active_brain = ActiveInferenceModule() if ActiveInferenceModule else None
    if active_brain:
        print("🧠 Active Inference Module: ENABLED")
    else:
        print("⚠️ Active Inference Module: NOT FOUND (Using static thresholds)")
    
    # 7. Loop
    total_steps = len(df) - seq_len - 1 # -1 for ground truth check
    print(f"▶️ Playing {total_steps} steps...")
    
    capital = 10000.0
    position = 0.0
    entry_price = 0.0
    
    for i in range(total_steps):
        time.sleep(1.0 / speed)
        
        step_idx = i + seq_len
        
        # Prepare Input
        input_slice = features[i : i+seq_len]
        input_tensor = torch.FloatTensor(input_slice).unsqueeze(0).to(device) # [1, 256, 18]
        
        llm_ctx = torch.zeros(1, 32).to(device)
        news_vec = torch.zeros(1, 2).to(device)
        
        hidden_states = [] # Reset hook
        
        with torch.no_grad():
             logits = model(input_tensor, llm_ctx, news_vec)
             prob = torch.sigmoid(logits).item()
             
        # Stack Hidden States
        if hidden_states:
             h_full = np.stack(hidden_states, axis=1) # [Batch, Seq, Hidden]
             h_matrix = h_full[0].T.tolist() 
             h_last_step = h_full[0][-1] 
             h_last = torch.tensor(h_last_step).float().to(device).unsqueeze(0)
        else:
            h_matrix = []
            h_last = torch.zeros(1, hidden_dim).to(device)

        # Decision Logic (Active Inference)
        price = df.iloc[step_idx]["close"]
        time_str = df.iloc[step_idx]["timestamp"].strftime("%Y-%m-%d %H:%M")
        
        if active_brain:
            # Look ahead for ground truth (Simulating real-time feedback after the fact)
            next_price = df.iloc[step_idx + 1]["close"]
            actual_move = 1.0 if next_price > price else 0.0
            
            # Log PREVIOUS prediction's outcome (In real life we'd log at step T+1)
            # For simplicity in simulation loop, we log 'prob' vs 'actual_move' instantaneously 
            # (treating it as if the 'epoch' closed immediately)
            # A more correct way involves a delay, but this is fine for calibration testing.
            active_brain.log_prediction(prob, actual_move)
            
            decision = active_brain.get_decision(prob) # Can pass features too if we had ADX
            
            # Translate Decision
            if decision == "BUY" and position == 0:
                action = "BUY"
            elif decision == "SELL" and position > 0:
                action = "SELL"
            else:
                action = "HOLD"
        else:
            # Fallback Static
            if prob > 0.6 and position == 0:
                action = "BUY"
            elif prob < 0.4 and position > 0:
                 action = "SELL"
            else:
                action = "HOLD"
        
        # Execution
        if action == "BUY":
            position = capital / price
            entry_price = price
            capital = 0
        elif action == "SELL":
            capital = position * price
            position = 0
            
        # PnL
        current_val = capital + (position * price)
        pnl_pct = (current_val - 10000.0) / 10000.0 * 100
        
        # Expert Curves
        long_bias = model.lnn.long_expert(h_last).item()
        short_bias = model.lnn.short_expert(h_last).item()
        
        # Emit State
        state = {
            "step": i,
            "timestamp": time_str,
            "price": float(price),
            "prob": prob,
            "action": action,
            "equity": current_val,
            "pnl_pct": pnl_pct,
            "long_bias": long_bias,
            "short_bias": short_bias,
            "neuronal_map": h_matrix,
            "completed": False,
            "threshold": active_brain.current_threshold if active_brain else 0.6
        }
        
        temp_file = SIM_SNAPSHOT_FILE + ".tmp"
        with open(temp_file, "w") as f:
            json.dump(state, f)
        os.replace(temp_file, SIM_SNAPSHOT_FILE)
        
        with open(SIM_TELEMETRY_FILE, "w") as f:
            json.dump({"step": i, "equity": current_val, "pnl": pnl_pct}, f)
            
    with open(SIM_SNAPSHOT_FILE, "w") as f:
        json.dump({"completed": True}, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--start_date", default=None)
    parser.add_argument("--speed", type=float, default=10.0)
    args = parser.parse_args()
    
    # Write PID
    with open(SIM_PID_FILE, "w") as f:
        f.write(str(os.getpid()))
        
    try:
        run_simulation(args.model, args.data, args.start_date, args.speed)
    finally:
        if os.path.exists(SIM_PID_FILE):
             os.remove(SIM_PID_FILE)
