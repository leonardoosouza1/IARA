
import argparse
import json
import time
import os
import sys
import pandas as pd
import numpy as np
import torch
import talib
import polars as pl
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.append(os.getcwd())

# Import Model Architecture (must import classes, not run script)
from src.brain.train_v9 import SwarmBrain, WorkerConfig, CoordinatorConfig

# Setup logging
import logging
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("BacktestV9")

def compute_v9_features(df: pd.DataFrame) -> np.ndarray:
    """Replicates the 22 features from feature_factory.py."""
    close = df['close'].values.astype(np.float64)
    high = df['high'].values.astype(np.float64)
    low = df['low'].values.astype(np.float64)
    opens = df['open'].values.astype(np.float64)
    
    # --- 1. CORE 12 ---
    # RSI
    rsi = talib.RSI(close, timeperiod=14)
    rsi_norm = (rsi - 50) / 25
    
    # MACD
    macd, macdsignal, macdhist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
    macd_norm = macd * 100
    macd_sig_norm = macdsignal * 100
    macd_hist_norm = macdhist * 100
    
    # BBands
    upper, middle, lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
    bb_width = (upper - lower) / (middle + 1e-10)
    bb_pct_b = (close - lower) / (upper - lower + 1e-10)
    bb_width_norm = bb_width * 10
    bb_pct_b_norm = (bb_pct_b - 0.5) * 2
    
    # ATR
    atr = talib.ATR(high, low, close, timeperiod=14)
    atr_norm = (atr / (close + 1e-10)) * 100
    
    # Stochastic
    stoch_k, stoch_d = talib.STOCH(high, low, close, fastk_period=14, slowk_period=3, slowd_period=3)
    stoch_k_norm = (stoch_k - 50) / 25
    stoch_d_norm = (stoch_d - 50) / 25
    
    # Velocity (ROC)
    velocity = talib.ROC(close, timeperiod=5)
    velocity_norm = velocity * 10
    
    # ADX
    adx = talib.ADX(high, low, close, timeperiod=14)
    adx_norm = (adx - 25) / 25
    
    # Sentiment (EMA 50)
    ema_50 = talib.EMA(close, timeperiod=50)
    ema_50 = np.nan_to_num(ema_50, nan=close[0])
    sentiment = (close - ema_50) / (ema_50 + 1e-10) * 100
    sentiment_norm = sentiment * 10
    
    # --- 2. MINDSET 6 ---
    # Trend Alignment
    sma_200 = talib.SMA(close, timeperiod=200)
    sma_200 = np.nan_to_num(sma_200, nan=close[0])
    sma_20 = talib.SMA(close, timeperiod=20)
    sma_20 = np.nan_to_num(sma_20, nan=close[0])
    
    dist_sma_200 = (close - sma_200) / (sma_200 + 1e-10) * 100
    dist_sma_20 = (close - sma_20) / (sma_20 + 1e-10) * 100
    
    # Sup/Res
    roll_max_50 = talib.MAX(high, timeperiod=50)
    roll_min_50 = talib.MIN(low, timeperiod=50)
    roll_max_50 = np.nan_to_num(roll_max_50, nan=close[0])
    roll_min_50 = np.nan_to_num(roll_min_50, nan=close[0])
    
    dist_res_50 = (roll_max_50 - close) / (close + 1e-10) * 100
    dist_sup_50 = (close - roll_min_50) / (close + 1e-10) * 100
    
    # Candle Physics
    candle_range = high - low
    upper_wick = high - np.maximum(close, opens)
    upper_wick_ratio = upper_wick / (candle_range + 1e-10)
    lower_wick = np.minimum(close, opens) - low
    lower_wick_ratio = lower_wick / (candle_range + 1e-10)
    
    # --- 3. TIME 4 ---
    dt_index = pd.to_datetime(df['time'])
    hours = dt_index.dt.hour.values
    days = dt_index.dt.dayofweek.values
    
    hour_sin = np.sin(2 * np.pi * hours / 24).astype(np.float32)
    hour_cos = np.cos(2 * np.pi * hours / 24).astype(np.float32)
    day_sin = np.sin(2 * np.pi * days / 7).astype(np.float32)
    day_cos = np.cos(2 * np.pi * days / 7).astype(np.float32)

    features = np.column_stack([
        rsi_norm, 
        macd_norm, macd_sig_norm, macd_hist_norm,
        bb_width_norm, bb_pct_b_norm,
        atr_norm,
        stoch_k_norm, stoch_d_norm,
        velocity_norm,
        adx_norm,
        sentiment_norm,
        # Mindset
        dist_sma_200, dist_sma_20,
        dist_res_50, dist_sup_50,
        upper_wick_ratio, lower_wick_ratio,
        # Time
        hour_sin, hour_cos, day_sin, day_cos
    ])
    
    return np.nan_to_num(features).astype(np.float32)

def resample_and_extract(df_pl: pl.DataFrame, tf: str) -> pd.DataFrame:
    """Resamples data and returns pandas DF with features."""
    if tf == "5m":
        resampled = df_pl
    else:
        # Polars resampling
        resampled = df_pl.upsample(time_column="time", every="1m").interpolate().group_by_dynamic(
            "time", every=tf, label="left"
        ).agg([
            pl.col("open").first(),
            pl.col("high").max(),
            pl.col("low").min(),
            pl.col("close").last(),
            pl.col("volume").sum()
        ])
    
    pdf = resampled.to_pandas()
    if 'time' in pdf.columns:
        pdf['time'] = pd.to_datetime(pdf['time']).dt.tz_localize(None)
    # Compute features on the resampled dataframe
    features = compute_v9_features(pdf)
    
    # Create a feature dataframe
    feat_cols = [f"f_{i}" for i in range(features.shape[1])]
    feat_df = pd.DataFrame(features, columns=feat_cols)
    feat_df['time'] = pdf['time']
    
    return feat_df

def main():
    parser = argparse.ArgumentParser(description="Backtest Swarm Brain v9 (Production Edition)")
    parser.add_argument("--model", type=str, required=True, help="Path to model .pth file")
    parser.add_argument("--data", type=str, default="data/processed/EURUSD.parquet", help="Path to data file")
    parser.add_argument("--start_date", type=str, default="2023-01-01", help="Start simulation date")
    parser.add_argument("--speed", type=float, default=10.0, help="Steps per second")
    args = parser.parse_args()

    # PID
    PID_FILE = "simulation_process.pid"
    with open(PID_FILE, "w") as f:
        f.write(str(os.getpid()))

    try:
        # Load Data
        logger.info(f"Loading data from {args.data}...")
        if args.data.endswith(".parquet"):
            df = pd.read_parquet(args.data)
        else:
            df = pd.read_csv(args.data)
        
        # Ensure column names are correct
        df.columns = [c.lower() for c in df.columns]
        if 'date' in df.columns and 'time' not in df.columns:
            df.rename(columns={'date': 'time'}, inplace=True)
        
        df['time'] = pd.to_datetime(df['time']).dt.tz_localize(None)
        df = df.sort_values('time').reset_index(drop=True)
        
        # Convert to Polars for fast resampling
        df_pl = pl.from_pandas(df)
        
        # Prepare all timeframes
        logger.info("🛠️  Preparing Multi-Timeframe Features (22 cols)...")
        tf_dfs = {}
        tf_map = {"5m": "M5", "15m": "M15", "30m": "M30", "1h": "H1", "4h": "H4"}
        for tf, tf_key in tf_map.items():
            tf_dfs[tf_key] = resample_and_extract(df_pl, tf)
            logger.info(f"   - {tf_key} ready: {len(tf_dfs[tf_key])} candles")

        # Load Model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")
        
        # Config (Matched to V9 training - 22 features)
        worker_cfg = WorkerConfig(input_dim=22, hidden_dim=32)
        coord_cfg = CoordinatorConfig(num_workers=5, worker_dim=32, hidden_dim=128)
        model = SwarmBrain(worker_cfg, coord_cfg).to(device)
        
        checkpoint = torch.load(args.model, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint)
        model.eval()
        logger.info(f"✅ Model loaded from {args.model}")

        # Replay Loop
        m5_df = tf_dfs["M5"]
        mask = m5_df['time'] >= pd.to_datetime(args.start_date)
        sim_m5 = m5_df[mask].reset_index(drop=True)
        
        if len(sim_m5) < 100:
            logger.error("❌ Not enough data for simulation!")
            return

        from collections import deque
        history = deque(maxlen=100)
        
        # Initial State
        equity = 10000.0
        position = 0.0
        entry_price = 0.0
        seq_len = 64
        is_paused = False
        
        COMMAND_FILE = "logs/sim_command.json"
        if os.path.exists(COMMAND_FILE): os.remove(COMMAND_FILE)
        
        # Capture Storage
        LAST_ENTRY_STATE = None
        ERRORS_FILE = "logs/backtest_errors.pt"
        SUCCESS_FILE = "logs/backtest_success.pt"
        RESERVOIR_FILE = "logs/pattern_reservoir.pt"
        
        processed_errors = torch.load(ERRORS_FILE, map_location="cpu", weights_only=False) if os.path.exists(ERRORS_FILE) else []
        processed_success = torch.load(SUCCESS_FILE, map_location="cpu", weights_only=False) if os.path.exists(SUCCESS_FILE) else []
        pattern_reservoir = torch.load(RESERVOIR_FILE, map_location="cpu", weights_only=False) if os.path.exists(RESERVOIR_FILE) else []
        
        logger.info(f"🚀 Simulation starting on {len(sim_m5)} M5 ticks...")
        
        is_paused = False
        i = 0
        while i < len(sim_m5):
            if not os.path.exists(PID_FILE): break

            # --- Control Logic ---
            if os.path.exists(COMMAND_FILE):
                try:
                    with open(COMMAND_FILE, "r") as f:
                        cmd_data = json.load(f)
                    cmd = cmd_data.get("command")
                    
                    if cmd == "PAUSE":
                        is_paused = True
                        if os.path.exists(COMMAND_FILE): os.remove(COMMAND_FILE)
                    elif cmd == "RESUME":
                        is_paused = False
                        if os.path.exists(COMMAND_FILE): os.remove(COMMAND_FILE)
                    elif cmd == "STEP":
                        is_paused = False # Run one step
                    elif cmd == "FLAG_ERROR":
                        if LAST_ENTRY_STATE is not None:
                             target_label = 1 if LAST_ENTRY_STATE["action"] == "BUY" else 0
                             error_item = {
                                 "x_dict": LAST_ENTRY_STATE["x"],
                                 "y": torch.tensor(target_label, dtype=torch.long),
                                 "timestamp": LAST_ENTRY_STATE["time"].isoformat(),
                                 "pnl": 0.0, "manual": True
                             }
                             processed_errors.append(error_item)
                             try: torch.save(processed_errors, ERRORS_FILE)
                             except: pass
                        if os.path.exists(COMMAND_FILE): os.remove(COMMAND_FILE)
                    elif cmd == "FLAG_PATTERN":
                        x_cpu = {k: v.cpu().clone() for k, v in x_dict.items()}
                        pattern_reservoir.append({
                            "x_dict": x_cpu, "timestamp": current_time.isoformat(), "label": "user_flagged"
                        })
                        try: torch.save(pattern_reservoir, RESERVOIR_FILE)
                        except: pass
                        if os.path.exists(COMMAND_FILE): os.remove(COMMAND_FILE)
                except Exception as e:
                    logger.error(f"Error handling command: {e}")

            # --- Handle Pause State ---
            if is_paused:
                time.sleep(0.1)
                continue
            
            # --- Check if we just did a STEP ---
            # If the command was STEP, we pause after this tick
            if os.path.exists(COMMAND_FILE):
                with open(COMMAND_FILE, "r") as f:
                    if json.load(f).get("command") == "STEP":
                        is_paused = True
                        os.remove(COMMAND_FILE)
            current_time = sim_m5.iloc[i]['time']
            row_ohlc = df[df['time'] == current_time]
            if row_ohlc.empty: 
                i += 1
                continue
            
            o, h, l, c = row_ohlc[['open', 'high', 'low', 'close']].values[0]
            current_price = c
            
            # Build v9 Input Map
            x_dict = {}
            can_predict = True
            
            for tf_key in ["M5", "M15", "M30", "H1", "H4"]:
                tf_data = tf_dfs[tf_key]
                tf_slice = tf_data[tf_data['time'] <= current_time].tail(seq_len)
                
                if len(tf_slice) < seq_len:
                    can_predict = False; break
                
                feat_vals = tf_slice.drop(columns=['time']).values
                x_dict[tf_key] = torch.tensor(feat_vals, dtype=torch.float32).unsqueeze(0).to(device)
            
            if not can_predict: 
                i += 1
                continue
            
            # Inference
            with torch.no_grad():
                with torch.amp.autocast('cuda', enabled=(device=="cuda")):
                    logits, worker_logits, long_imp, short_imp = model(x_dict)
                    prob = torch.sigmoid(logits).item()
            
            # Trading Logic
            action = "HOLD"
            units = 10000 
            
            # Conviction Thresholds
            BUY_THRESH = 0.65
            SELL_THRESH = 0.35

            # --- 1. EXIT LOGIC (TP/TS) ---
            trade_closed = False
            pnl_final = 0.0
            unrealized_pnl = 0.0
            
            if position != 0:
                if position > 0: unrealized_pnl = (current_price - entry_price) * units
                else: unrealized_pnl = (entry_price - current_price) * units
                
                # Update Max PnL for trailing
                if "max_pnl" not in LAST_ENTRY_STATE: LAST_ENTRY_STATE["max_pnl"] = unrealized_pnl
                LAST_ENTRY_STATE["max_pnl"] = max(LAST_ENTRY_STATE["max_pnl"], unrealized_pnl)
                max_pnl = LAST_ENTRY_STATE["max_pnl"]

                # SCALPING EXIT RULES (Requested by user: $0.50, $1.00)
                # Rule A: Take Profit @ $1.00 (Aggressive Scalp Target)
                if unrealized_pnl >= 1.0:
                    action, trade_closed = "TP_SCALP", True
                
                # Rule B: Trailing Stop (Protect Profit > $0.60)
                elif max_pnl > 0.6 and (max_pnl - unrealized_pnl) > 0.25:
                    action, trade_closed = "TS_ACTIVE", True

                # Rule C: Conviction Fade (Close if profit is positive and trend weakens)
                elif position > 0 and prob < 0.55 and unrealized_pnl > 0.3:
                    action, trade_closed = "EXIT_NEUTRAL", True
                elif position < 0 and prob > 0.45 and unrealized_pnl > 0.3:
                    action, trade_closed = "EXIT_NEUTRAL", True

                # Rule D: Signal Reversal (Standard)
                elif prob > BUY_THRESH and position < 0:
                    action, trade_closed = "BUY", True
                elif prob < SELL_THRESH and position > 0:
                    action, trade_closed = "SELL", True

            if trade_closed:
                pnl_final = unrealized_pnl
                equity += pnl_final
                
                if LAST_ENTRY_STATE is not None:
                    # Capture Metadata for training
                    target_label = 1 if LAST_ENTRY_STATE["action"] == "BUY" else 0
                    item = {
                        "x_dict": LAST_ENTRY_STATE["x"],
                        "y": torch.tensor(target_label, dtype=torch.long),
                        "timestamp": LAST_ENTRY_STATE["time"].isoformat(),
                        "pnl": float(pnl_final)
                    }

                    if pnl_final < 0:
                        processed_errors.append(item)
                        try: torch.save(processed_errors, ERRORS_FILE)
                        except: pass
                    else:
                        processed_success.append(item)
                        try: torch.save(processed_success, SUCCESS_FILE)
                        except: pass
                
                # Reset state
                if action not in ["BUY", "SELL"]: # If it wasn't a reversal
                    position, entry_price = 0.0, 0.0
                    LAST_ENTRY_STATE = None

            # Cooldown check
            is_cooldown = False
            last_exit_time = LAST_ENTRY_STATE.get("exit_time") if LAST_ENTRY_STATE and "exit_time" in LAST_ENTRY_STATE else None
            if last_exit_time:
                # Cooldown of 3 M5 candles
                candles_passed = (current_time - last_exit_time).total_seconds() / 300
                if candles_passed < 3:
                    is_cooldown = True

            # --- 2. ENTRY LOGIC ---
            # IMPORTANT: Don't re-enter immediately if we just closed a trade for profit (TP/TS)
            # OR if we are in a cooldown period.
            can_enter = (not trade_closed or action in ["BUY", "SELL"]) and not is_cooldown
            
            if can_enter:
                if prob > BUY_THRESH and position <= 0:
                    position, entry_price, action = 1.0, current_price, "BUY"
                    x_cpu = {k: v.cpu().clone() for k, v in x_dict.items()}
                    LAST_ENTRY_STATE = {"x": x_cpu, "action": "BUY", "time": current_time, "max_pnl": 0.0}
                elif prob < SELL_THRESH and position >= 0:
                    position, entry_price, action = -1.0, current_price, "SELL"
                    x_cpu = {k: v.cpu().clone() for k, v in x_dict.items()}
                    LAST_ENTRY_STATE = {"x": x_cpu, "action": "SELL", "time": current_time, "max_pnl": 0.0}
            
            # Record exit time for cooldown if we just closed
            if trade_closed and action not in ["BUY", "SELL"]:
                if LAST_ENTRY_STATE:
                    LAST_ENTRY_STATE["exit_time"] = current_time

            # Current Unrealized PnL
            unrealized_pnl = 0.0
            if position > 0: unrealized_pnl = (current_price - entry_price) * units
            elif position < 0: unrealized_pnl = (entry_price - current_price) * units
            
            total_equity = equity + unrealized_pnl
            pnl_money = total_equity - 10000.0
            
            # Build History (Include ACTION for reliable visual signals)
            tick_data = {
                "t": current_time.strftime("%Y-%m-%d %H:%M"),
                "o": float(o), "h": float(h), "l": float(l), "c": float(c),
                "prob": float(prob), "pos": float(position), "act": action,
                "price": float(current_price)
            }
            history.append(tick_data)
            
            # Snapshot
            votes = model.get_worker_votes(worker_logits)
            snapshot = {
                "step": i, "timestamp": current_time.strftime("%Y-%m-%d %H:%M"),
                "price": float(current_price), 
                "entry_price": float(entry_price),
                "units": int(units if position != 0 else 0),
                "equity": float(total_equity), 
                "pnl_usd": float(pnl_money),
                "unrealized_pnl": float(unrealized_pnl),
                "realized_pnl": float(equity - 10000.0),
                "prob": float(prob), "action": action,
                "pos": float(position),
                "is_cooldown": is_cooldown,
                "long_bias": float(long_imp.item()), # Raw score
                "short_bias": float(short_imp.item()), # Raw score
                "agreement": float(votes["agreement"].mean().item()),
                "n_long": int(votes["n_long"].mean().item()),
                "n_short": int(votes["n_short"].mean().item()),
                "neuronal_map": torch.sigmoid(worker_logits[0]).cpu().tolist(),
                "completed": False,
                "history": list(history)
            }
            
            with open("logs/sim_snapshot.json", "w") as f: json.dump(snapshot, f)
            
            if i % 50 == 0:
                print(f"[{current_time}] Equity: ${total_equity:.2f} | PnL: ${pnl_money:.2f} | Prob: {prob:.2f}")
            
            # Post-tick Pause if Step
            if os.path.exists(COMMAND_FILE):
                with open(COMMAND_FILE, "r") as f:
                    final_cmd = json.load(f).get("command")
                if final_cmd == "PAUSED_AFTER_STEP":
                    is_paused = True
                    # Keep the file but mark as PAUSE
                    with open(COMMAND_FILE, "w") as f: json.dump({"command": "PAUSE"}, f)

            i += 1
            time.sleep(1.0 / args.speed)

        with open("logs/sim_snapshot.json", "w") as f: json.dump({"completed": True}, f)
        logger.info("Simulation Finished.")

        
    except Exception as e:
        logger.error(f"💥 Simulation Crashed: {e}", exc_info=True)
    finally:
        if os.path.exists(PID_FILE): os.remove(PID_FILE)

if __name__ == "__main__":
    main()
