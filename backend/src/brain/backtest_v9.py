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

# Pickle compatibility — older checkpoints (e.g. interrupted_model.pth from
# Feb 2026) reference a TrainerConfig class that lived in this module then.
# Expose it here so torch.load resolves the original symbol path.
try:
    from src.train.trainer import TrainerConfig as _TrainerConfig
    TrainerConfig = _TrainerConfig
except Exception:
    try:
        from src.brain.simulate_v8 import TrainerConfig as _TrainerConfig
        TrainerConfig = _TrainerConfig
    except Exception:
        class TrainerConfig:        # permissive fallback for unpickling only
            def __init__(self, **kw):
                for k, v in kw.items():
                    setattr(self, k, v)

# Setup logging
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("BacktestV9")


def compute_v9_features(df: pd.DataFrame) -> np.ndarray:
    """Compute the canonical 22 features using centralized indicators module."""
    from src.data.indicators import compute_features_from_df
    return compute_features_from_df(df, time_col="time", include_time=True)


def resample_and_extract(df_pl: pl.DataFrame, tf: str) -> pd.DataFrame:
    """Resamples data and returns pandas DF with features."""
    if tf == "5m":
        resampled = df_pl
    else:
        # Polars resampling
        resampled = (
            df_pl.upsample(time_column="time", every="1m")
            .interpolate()
            .group_by_dynamic("time", every=tf, label="left")
            .agg(
                [
                    pl.col("open").first(),
                    pl.col("high").max(),
                    pl.col("low").min(),
                    pl.col("close").last(),
                    pl.col("volume").sum(),
                ]
            )
        )

    pdf = resampled.to_pandas()
    if "time" in pdf.columns:
        pdf["time"] = pd.to_datetime(pdf["time"]).dt.tz_localize(None)
    # Compute features on the resampled dataframe
    features = compute_v9_features(pdf)

    # Create a feature dataframe
    feat_cols = [f"f_{i}" for i in range(features.shape[1])]
    feat_df = pd.DataFrame(features, columns=feat_cols)
    feat_df["time"] = pdf["time"]

    return feat_df


def main():
    parser = argparse.ArgumentParser(
        description="Backtest Swarm Brain v9 (Production Edition)"
    )
    parser.add_argument(
        "--model", type=str, required=True, help="Path to model .pth file"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/processed/EURUSD.parquet",
        help="Path to data file",
    )
    parser.add_argument(
        "--start_date", type=str, default="2023-01-01", help="Start simulation date"
    )
    parser.add_argument(
        "--end_date", type=str, default=None, help="End simulation date (inclusive)"
    )
    parser.add_argument("--speed", type=float, default=10.0, help="Steps per second")
    parser.add_argument(
        "--max_steps", type=int, default=None, help="Maximum number of simulation steps"
    )
    parser.add_argument(
        "--report_path",
        type=str,
        default="logs/backtest_report.json",
        help="Where to write summary report JSON",
    )
    parser.add_argument(
        "--no_sleep",
        action="store_true",
        default=False,
        help="Disable real-time sleeping (useful for automated loops)",
    )
    parser.add_argument("--worker_hidden", type=int, default=64)
    parser.add_argument("--coord_hidden", type=int, default=256)
    parser.add_argument("--seq_len", type=int, default=128)
    args = parser.parse_args()

    def decide_by_consensus(
        worker_logits: torch.Tensor, coord_logit: torch.Tensor
    ) -> str:
        """Consensus-based decision (Option 2).

        - If >=3 workers vote NEUTRAL -> WAIT
        - If coordinator agrees with strong majority -> LONG/SHORT
        - If coordinator is very confident -> override
        - Else -> WAIT
        """
        # worker_logits: [1, num_workers, 3] or [num_workers, 3]
        wl = worker_logits
        if wl.dim() == 3:
            wl = wl[0]

        votes = wl.argmax(dim=-1)  # 0=LONG, 1=SHORT, 2=NEUTRAL
        n_long = int((votes == 0).sum().item())
        n_short = int((votes == 1).sum().item())
        n_wait = int((votes == 2).sum().item())

        coord_prob = torch.sigmoid(coord_logit.view(-1))[0].item()

        if n_wait >= 4:
            return "WAIT"

        if coord_prob > 0.55 and n_long >= 2:
            return "LONG"
        if coord_prob < 0.45 and n_short >= 2:
            return "SHORT"

        if coord_prob > 0.65:
            return "LONG"
        if coord_prob < 0.35:
            return "SHORT"

        return "WAIT"

    def summarize_raw_votes(worker_logits: torch.Tensor) -> dict:
        """Summarize raw argmax votes from worker logits for debugging/telemetry."""
        wl = worker_logits
        if wl.dim() == 3:
            wl = wl[0]
        votes = wl.argmax(dim=-1)  # 0=LONG, 1=SHORT, 2=NEUTRAL
        n_long = int((votes == 0).sum().item())
        n_short = int((votes == 1).sum().item())
        n_wait = int((votes == 2).sum().item())
        agreement = float(max(n_long, n_short, n_wait) / max(1, votes.numel()))
        return {
            "n_long": n_long,
            "n_short": n_short,
            "n_wait": n_wait,
            "agreement": agreement,
        }

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
        if "date" in df.columns and "time" not in df.columns:
            df.rename(columns={"date": "time"}, inplace=True)

        df["time"] = pd.to_datetime(df["time"]).dt.tz_localize(None)
        df = df.sort_values("time").reset_index(drop=True)

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

        checkpoint = torch.load(args.model, map_location="cpu", weights_only=False)
        ckpt_input_dim = None
        try:
            w = checkpoint.get("swarm.workers.M5.input_proj.weight")
            if w is not None and hasattr(w, "shape") and len(w.shape) == 2:
                ckpt_input_dim = int(w.shape[1])
        except Exception:
            ckpt_input_dim = None

        if ckpt_input_dim is None:
            ckpt_input_dim = 12
            logger.warning("⚠️ Could not infer checkpoint input_dim; defaulting to 12")

        logger.info(f"🧩 Checkpoint expects input_dim={ckpt_input_dim}")

        # Config (Match checkpoint input dimension)
        worker_cfg = WorkerConfig(input_dim=ckpt_input_dim, hidden_dim=args.worker_hidden)
        coord_cfg = CoordinatorConfig(
            num_workers=5, worker_dim=args.worker_hidden, hidden_dim=args.coord_hidden
        )
        model = SwarmBrain(worker_cfg, coord_cfg).to(device)

        model.load_state_dict(checkpoint, strict=False)
        model.eval()
        logger.info(f"✅ Model loaded from {args.model}")

        # Replay Loop
        m5_df = tf_dfs["M5"]
        start_dt = pd.to_datetime(args.start_date)
        mask = m5_df["time"] >= start_dt
        end_dt = None
        if args.end_date:
            end_dt = pd.to_datetime(args.end_date)
            mask = mask & (m5_df["time"] <= end_dt)
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
        units = 10000
        is_paused = False
        last_exit_time = None
        trade_count = 0
        last_save_time = 0
        seq_len = args.seq_len

        COMMAND_FILE = "logs/sim_command.json"
        if os.path.exists(COMMAND_FILE):
            os.remove(COMMAND_FILE)

        # Capture Storage
        LAST_ENTRY_STATE = None
        ERRORS_FILE = "logs/backtest_errors.pt"
        SUCCESS_FILE = "logs/backtest_success.pt"
        RESERVOIR_FILE = "logs/pattern_reservoir.pt"

        processed_errors = (
            torch.load(ERRORS_FILE, map_location="cpu", weights_only=False)
            if os.path.exists(ERRORS_FILE)
            else []
        )
        processed_success = (
            torch.load(SUCCESS_FILE, map_location="cpu", weights_only=False)
            if os.path.exists(SUCCESS_FILE)
            else []
        )
        pattern_reservoir = (
            torch.load(RESERVOIR_FILE, map_location="cpu", weights_only=False)
            if os.path.exists(RESERVOIR_FILE)
            else []
        )

        logger.info(f"🚀 Simulation starting on {len(sim_m5)} M5 ticks...")

        is_paused = False
        i = 0
        max_steps = int(args.max_steps) if args.max_steps is not None else None
        n_total = len(sim_m5) if max_steps is None else min(len(sim_m5), max_steps)
        while i < n_total:
            if not os.path.exists(PID_FILE):
                break

            # --- Control Logic ---
            if os.path.exists(COMMAND_FILE):
                try:
                    with open(COMMAND_FILE, "r") as f:
                        cmd_data = json.load(f)
                    cmd = cmd_data.get("command")

                    if cmd == "PAUSE":
                        is_paused = True
                        if os.path.exists(COMMAND_FILE):
                            os.remove(COMMAND_FILE)
                    elif cmd == "RESUME":
                        is_paused = False
                        if os.path.exists(COMMAND_FILE):
                            os.remove(COMMAND_FILE)
                    elif cmd == "STEP":
                        is_paused = False  # Run one step
                    elif cmd == "FLAG_ERROR":
                        if LAST_ENTRY_STATE is not None:
                            # INVERT: If the error was a BUY, the target should have been 0 (SELL/NEUTRAL)
                            target_label = (
                                0 if LAST_ENTRY_STATE["action"] == "BUY" else 1
                            )
                            error_item = {
                                "x_dict": LAST_ENTRY_STATE["x"],
                                "y": torch.tensor(target_label, dtype=torch.long),
                                "timestamp": LAST_ENTRY_STATE["time"].isoformat(),
                                "pnl": 0.0,
                                "manual": True,
                            }
                            processed_errors.append(error_item)
                            try:
                                torch.save(processed_errors, ERRORS_FILE)
                            except:
                                pass
                        if os.path.exists(COMMAND_FILE):
                            os.remove(COMMAND_FILE)
                    elif cmd == "FLAG_PATTERN":
                        x_cpu = {k: v.cpu().clone() for k, v in x_dict.items()}
                        pattern_reservoir.append(
                            {
                                "x_dict": x_cpu,
                                "timestamp": current_time.isoformat(),
                                "label": "user_flagged",
                            }
                        )
                        try:
                            torch.save(pattern_reservoir, RESERVOIR_FILE)
                        except:
                            pass
                        if os.path.exists(COMMAND_FILE):
                            os.remove(COMMAND_FILE)
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
                    final_cmd = json.load(f).get("command")
                if final_cmd == "STEP":
                    is_paused = True
                    # Keep the file but mark as PAUSE
                    with open(COMMAND_FILE, "w") as f:
                        json.dump({"command": "PAUSE"}, f)

            current_time = sim_m5.iloc[i]["time"]
            row_ohlc = df[df["time"] == current_time]
            if row_ohlc.empty:
                i += 1
                continue

            o, h, l, c = row_ohlc[["open", "high", "low", "close"]].values[0]
            current_price = c

            # Build v9 Input Map
            x_dict = {}
            can_predict = True

            for tf_key in ["M5", "M15", "M30", "H1", "H4"]:
                tf_data = tf_dfs[tf_key]
                tf_slice = tf_data[tf_data["time"] <= current_time].tail(seq_len)

                if len(tf_slice) < seq_len:
                    can_predict = False
                    break

                feat_vals = tf_slice.drop(columns=["time"]).values
                if feat_vals.shape[1] > ckpt_input_dim:
                    feat_vals = feat_vals[:, :ckpt_input_dim]
                x_dict[tf_key] = (
                    torch.tensor(feat_vals, dtype=torch.float32).unsqueeze(0).to(device)
                )

            if not can_predict:
                i += 1
                continue

            # Inference
            with torch.no_grad():
                with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                    _out = model(x_dict)
                    # SwarmBrain.forward now returns 6 values:
                    # (policy_logits, worker_logits, long_score, short_score,
                    #  attn_weights, confidence). Legacy backtest expected
                    # a single scalar `logits` (diff = long − short), so we
                    # reconstruct it here for backward compat with
                    # decide_by_consensus/sigmoid downstream.
                    _policy = _out[0]
                    worker_logits = _out[1]
                    long_imp = _out[2]
                    short_imp = _out[3]
                    logits = (long_imp - short_imp).view(-1)[:1]
                    prob = torch.sigmoid(logits).item()

            # Consensus Decision (Swarm)
            desired_action = decide_by_consensus(
                worker_logits, logits
            )  # LONG/SHORT/WAIT
            raw_votes = summarize_raw_votes(worker_logits)
            coord_prob = float(torch.sigmoid(logits.view(-1))[0].item())

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
                if position > 0:
                    unrealized_pnl = (current_price - entry_price) * units
                else:
                    unrealized_pnl = (entry_price - current_price) * units

                # Update Max PnL for trailing
                if "max_pnl" not in LAST_ENTRY_STATE:
                    LAST_ENTRY_STATE["max_pnl"] = unrealized_pnl
                LAST_ENTRY_STATE["max_pnl"] = max(
                    LAST_ENTRY_STATE["max_pnl"], unrealized_pnl
                )
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
                elif (desired_action == "LONG") and position < 0:
                    action, trade_closed = "BUY", True
                elif (desired_action == "SHORT") and position > 0:
                    action, trade_closed = "SELL", True

            if trade_closed:
                pnl_final = unrealized_pnl
                equity += pnl_final

                if LAST_ENTRY_STATE is not None:
                    # Capture Metadata for training
                    taken_label = 1 if LAST_ENTRY_STATE["action"] == "BUY" else 0

                    if pnl_final < -0.5: # Only capture significant mistakes (at least 0.5 pips)
                        # ERROR: Target should have been the OPPOSITE of what we did
                        target_label = 1 - taken_label
                        item = {
                            "x_dict": LAST_ENTRY_STATE["x"],
                            "y": torch.tensor(target_label, dtype=torch.long),
                            "timestamp": LAST_ENTRY_STATE["time"].isoformat(),
                            "pnl": float(pnl_final),
                        }
                        processed_errors.append(item)
                        try:
                            torch.save(processed_errors, ERRORS_FILE)
                        except:
                            pass
                    else:
                        # SUCCESS: Target was indeed what we did
                        target_label = taken_label
                        item = {
                            "x_dict": LAST_ENTRY_STATE["x"],
                            "y": torch.tensor(target_label, dtype=torch.long),
                            "timestamp": LAST_ENTRY_STATE["time"].isoformat(),
                            "pnl": float(pnl_final),
                        }
                        processed_success.append(item)
                        try:
                            torch.save(processed_success, SUCCESS_FILE)
                        except:
                            pass

                # Reset state
                if action not in ["BUY", "SELL"]:  # If it wasn't a reversal
                    position, entry_price = 0.0, 0.0
                    LAST_ENTRY_STATE = None

            # Cooldown check
            is_cooldown = False
            if last_exit_time:
                # Cooldown of 3 M5 candles (15 mins)
                candles_passed = (current_time - last_exit_time).total_seconds() / 300
                if candles_passed < 3:
                    is_cooldown = True

            # --- 2. ENTRY LOGIC ---
            # IMPORTANT: Don't re-enter immediately if we just closed a trade for profit (TP/TS)
            # OR if we are in a cooldown period.
            can_enter = (
                not trade_closed or action in ["BUY", "SELL"]
            ) and not is_cooldown

            if can_enter:
                if (desired_action == "LONG") and position <= 0:
                    position, entry_price, action = 1.0, current_price, "BUY"
                    x_cpu = {k: v.cpu().clone() for k, v in x_dict.items()}
                    LAST_ENTRY_STATE = {
                        "x": x_cpu,
                        "action": "BUY",
                        "time": current_time,
                        "max_pnl": 0.0,
                    }
                elif (desired_action == "SHORT") and position >= 0:
                    position, entry_price, action = -1.0, current_price, "SELL"
                    x_cpu = {k: v.cpu().clone() for k, v in x_dict.items()}
                    LAST_ENTRY_STATE = {
                        "x": x_cpu,
                        "action": "SELL",
                        "time": current_time,
                        "max_pnl": 0.0,
                    }

            # Record exit time for cooldown if we just closed
            if trade_closed and action not in ["BUY", "SELL"]:
                last_exit_time = current_time
                trade_count += 1

            # Current Unrealized PnL
            unrealized_pnl = 0.0
            if position > 0:
                unrealized_pnl = (current_price - entry_price) * units
            elif position < 0:
                unrealized_pnl = (entry_price - current_price) * units

            total_equity = equity + unrealized_pnl
            pnl_money = total_equity - 10000.0

            # Build History (Include ACTION for reliable visual signals)
            tick_data = {
                "t": current_time.strftime("%Y-%m-%d %H:%M"),
                "o": float(o),
                "h": float(h),
                "l": float(l),
                "c": float(c),
                "prob": float(prob),
                "pos": float(position),
                "act": action,
                "price": float(current_price),
            }
            history.append(tick_data)

            # Save Snapshot (Freq Capped at 5/sec)
            now = time.time()
            if now - last_save_time > 0.2:
                # Newer SwarmBrain dropped .get_worker_votes(); compute votes
                # inline from worker_logits — same semantics as the old helper.
                if hasattr(model, "get_worker_votes"):
                    votes = model.get_worker_votes(worker_logits)
                else:
                    # Same semantics as the legacy helper — produce a dict
                    # carrying per-batch agreement + long/short counts as
                    # zero-d tensors (so consumers can .mean().item()).
                    _wl = worker_logits[0:1] if worker_logits.dim() == 3 else worker_logits.unsqueeze(0)
                    _idx = _wl.argmax(dim=-1)        # [B, num_workers]
                    n_long  = (_idx == 0).sum(dim=-1).float()
                    n_short = (_idx == 1).sum(dim=-1).float()
                    n_wait  = (_idx == 2).sum(dim=-1).float()
                    total   = _idx.shape[-1]
                    # agreement = max-vote share, in [1/k, 1]
                    agreement = torch.stack([n_long, n_short, n_wait], dim=-1).max(dim=-1).values / total
                    votes = {
                        "agreement": agreement,
                        "n_long":    n_long,
                        "n_short":   n_short,
                        "n_wait":    n_wait,
                    }
                snapshot = {
                    "step": i,
                    "timestamp": current_time.strftime("%Y-%m-%d %H:%M"),
                    "price": float(current_price),
                    "entry_price": float(entry_price),
                    "units": int(units if position != 0 else 0),
                    "equity": float(total_equity),
                    "pnl_usd": float(pnl_money),
                    "unrealized_pnl": float(unrealized_pnl),
                    "realized_pnl": float(equity - 10000.0),
                    "prob": float(prob),
                    "action": action,
                    "desired_action": desired_action,
                    "coord_prob": coord_prob,
                    "pos": float(position),
                    "is_cooldown": is_cooldown,
                    "trade_count": trade_count,
                    "long_bias": float(long_imp.item()),  # Raw score
                    "short_bias": float(short_imp.item()),  # Raw score
                    "agreement": float(votes["agreement"].mean().item()),
                    "n_long": int(votes["n_long"].mean().item()),
                    "n_short": int(votes["n_short"].mean().item()),
                    "agreement_raw": float(raw_votes["agreement"]),
                    "n_long_raw": int(raw_votes["n_long"]),
                    "n_short_raw": int(raw_votes["n_short"]),
                    "n_wait_raw": int(raw_votes["n_wait"]),
                    "neuronal_map": torch.sigmoid(worker_logits[0]).cpu().tolist(),
                    "completed": False,
                    "history": list(history),
                }
                with open("logs/sim_snapshot.json", "w") as f:
                    json.dump(snapshot, f)
                last_save_time = now

            if i % 50 == 0:
                print(
                    f"[{current_time}] Equity: ${total_equity:.2f} | PnL: ${pnl_money:.2f} | Prob: {prob:.2f}"
                )

            # Post-tick Pause if Step
            if os.path.exists(COMMAND_FILE):
                with open(COMMAND_FILE, "r") as f:
                    final_cmd = json.load(f).get("command")
                if final_cmd == "PAUSED_AFTER_STEP":
                    is_paused = True
                    # Keep the file but mark as PAUSE
                    with open(COMMAND_FILE, "w") as f:
                        json.dump({"command": "PAUSE"}, f)

            i += 1
            if not args.no_sleep:
                time.sleep(1.0 / args.speed)

        with open("logs/sim_snapshot.json", "w") as f:
            json.dump({"completed": True}, f)

        report = {
            "model": args.model,
            "data": args.data,
            "start_date": args.start_date,
            "end_date": args.end_date,
            "steps": int(n_total),
            "equity_final": float(equity),
            "pnl_realized": float(equity - 10000.0),
            "trade_count": int(trade_count),
            "errors_captured": int(len(processed_errors)),
            "success_captured": int(len(processed_success)),
            "patterns_captured": int(len(pattern_reservoir)),
        }
        try:
            Path(os.path.dirname(args.report_path) or ".").mkdir(
                parents=True, exist_ok=True
            )
            with open(args.report_path, "w") as f:
                json.dump(report, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to write report: {e}")

        logger.info("Simulation Finished.")

    except Exception as e:
        logger.error(f"💥 Simulation Crashed: {e}", exc_info=True)
    finally:
        if os.path.exists(PID_FILE):
            os.remove(PID_FILE)


if __name__ == "__main__":
    main()
