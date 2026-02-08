import streamlit as st
import sys
import subprocess
import os
import signal
import time
import pandas as pd
import json
import plotly.graph_objects as go
import plotly.express as px
import torch

# Constants
LOG_DIR = "logs"
ZOO_DIR = os.path.join(LOG_DIR, "model_zoo")
SIM_DATA = "data/processed/EURUSD.parquet"
SIM_SCRIPT = "src/brain/backtest_v9.py"
SIM_SNAPSHOT_FILE = os.path.join(LOG_DIR, "sim_snapshot.json")
SIM_PID_FILE = "simulation_process.pid"

def is_sim_running():
    if os.path.exists(SIM_PID_FILE):
        try:
            with open(SIM_PID_FILE, "r") as f:
                pid = int(f.read().strip())
            os.kill(pid, 0)
            return True, pid
        except:
            return False, None
    return False, None

def stop_sim():
    running, pid = is_sim_running()
    if running and pid:
        try:
            os.kill(pid, signal.SIGTERM)
        except: pass
    if os.path.exists(SIM_PID_FILE):
        os.remove(SIM_PID_FILE)

def render():
    st.title("🧪 Simulation Lab (Production Replay)")
    st.markdown("Replay historical data through the AI Brain to verify behavior before live deployment.")

    running, pid = is_sim_running()

    # --- SIDEBAR CONFIG ---
    with st.sidebar:
        st.header("🎮 Game Controls")
        
        # Model Selector
        models = [f for f in os.listdir(ZOO_DIR) if f.endswith(".pth")] if os.path.exists(ZOO_DIR) else []
        selected_model = st.selectbox("Select Brain", models, disabled=running)
        
        # Data Config
        start_date = st.text_input("Start Date (YYYY-MM-DD)", "2023-01-01", disabled=running)
        
        # Speed
        speed = st.slider("Playback Speed (Steps/Sec)", 1, 100, 10, disabled=running)
        
        if not running:
            if st.button("🚀 START SIMULATION", type="primary"):
                if not selected_model:
                     st.error("No model selected!")
                     st.stop()
                if not os.path.exists(SIM_DATA):
                     st.error(f"Data file not found: {SIM_DATA}")
                     st.stop()
                if not os.path.exists(SIM_SCRIPT):
                     st.error(f"Script not found: {SIM_SCRIPT}")
                     st.stop()

                if os.path.exists(SIM_SNAPSHOT_FILE):
                    try: os.remove(SIM_SNAPSHOT_FILE)
                    except: pass
                
                model_path = os.path.join(ZOO_DIR, selected_model)
                cmd = [
                    sys.executable, "-u", SIM_SCRIPT,
                    "--model", model_path,
                    "--data", SIM_DATA,
                    "--start_date", start_date,
                    "--speed", str(speed)
                ]
                
                with open("sim_std.log", "w") as log:
                    subprocess.Popen(cmd, stdout=log, stderr=log, preexec_fn=os.setpgrp)
                
                st.toast("Simulation Started!", icon="🚀")
                time.sleep(1)
                st.rerun()
        else:
            st.info(f"🟢 Sim Running (PID {pid})")
            col1, col2 = st.columns(2)
            COMMAND_FILE = "logs/sim_command.json"
            
            if col1.button("⏸️ PAUSE", use_container_width=True):
                with open(COMMAND_FILE, "w") as f: json.dump({"command": "PAUSE"}, f)
                st.toast("Pausing...")
                
            if col2.button("▶️ RESUME", use_container_width=True):
                with open(COMMAND_FILE, "w") as f: json.dump({"command": "RESUME"}, f)
                st.toast("Resuming...")

            if st.button("⏭️ NEXT TICK (STEP)", use_container_width=True):
                with open(COMMAND_FILE, "w") as f: json.dump({"command": "STEP"}, f)
                st.toast("Advancing one tick...")

            if st.button("🚨 ENSINAR ERRO (FLAG)", use_container_width=True):
                with open(COMMAND_FILE, "w") as f: json.dump({"command": "FLAG_ERROR"}, f)
                st.warning("IA avisada!")

            if st.button("🛑 STOP SIMULATION", type="primary", use_container_width=True):
                stop_sim()
                st.rerun()
                    
            st.divider()
            def get_pt_count(path):
                if not os.path.exists(path): return 0
                try: return len(torch.load(path, map_location="cpu", weights_only=False))
                except: return 0
                
            c1, c2, c3 = st.columns(3)
            c1.metric("Errors", get_pt_count("logs/backtest_errors.pt"))
            c2.metric("Success", get_pt_count("logs/backtest_success.pt"))
            c3.metric("Patterns", get_pt_count("logs/pattern_reservoir.pt"))

    # --- MAIN STAGE ---
    if not running:
        st.info("👈 Configure and Start the Simulation from the Sidebar.")
        return

    # --- LIVE REPLAY VIEW ---
    if not os.path.exists(SIM_SNAPSHOT_FILE):
        st.info("🕒 Waiting for simulation snapshot...")
        time.sleep(1)
        st.rerun()
        
    try:
        with open(SIM_SNAPSHOT_FILE, "r") as f:
            state = json.load(f)
    except:
        time.sleep(0.5); st.rerun()
        
    if state.get("completed"):
        st.success("Simulation Completed!")
        stop_sim()
        st.rerun()
        return

    # TOP BAR: METRICS
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Date", state.get("timestamp", "N/A"))
    c2.metric("Price", f"{state.get('price', 0):.5f}")
    
    realized = state.get("realized_pnl", 0.0)
    floating = state.get("unrealized_pnl", 0.0)
    
    c3.metric("Realized PnL", f"${realized:+.2f}")
    c4.metric("Floating PnL", f"${floating:+.2f}", delta=f"{floating:+.2f}")
    
    # --- POSITION & COOLDOWN ---
    pos = state.get("pos", 0.0)
    units = state.get("units", 0)
    entry_p = state.get("entry_price", 0.0)
    is_cd = state.get("is_cooldown", False)
    
    pos_label = "⚪ NEUTRAL"
    if is_cd: pos_label = "⏳ COOLDOWN"
    elif pos > 0: pos_label = f"🟢 LONG (${units:,})"
    elif pos < 0: pos_label = f"🔴 SHORT (${units:,})"
    
    pos_help = f"Entry: {entry_p:.5f}" if pos != 0 else "Market Neutral"
    c5.metric("Position", pos_label, help=pos_help)
    
    col_chart, col_brain = st.columns([3, 1])
    
    with col_brain:
        st.subheader("🧠 Intelligence")
        st.progress(state.get("prob", 0.5), text=f"Buy Conviction: {state.get('prob',0)*100:.1f}%")
        
        b1, b2 = st.columns(2)
        b1.metric("L-Expert", f"{state.get('long_bias', 0):.2f}")
        b2.metric("S-Expert", f"{state.get('short_bias', 0):.2f}")
        
        agreement = state.get("agreement", 0.0)
        n_l, n_s, n_n = state.get("n_long", 0), state.get("n_short", 0), state.get("n_neutral", 0)
        st.markdown(f"**Agreement:** {agreement*100:.0f}% | 🟢{n_l} 🔴{n_s} ⚪{n_n}")
        
        show_brain = st.checkbox("Show Neuronal Map", value=True)
        if show_brain:
            neurons = state.get("neuronal_map", [])
            if neurons:
                 fig_n = px.imshow(
                     pd.DataFrame(neurons), aspect="auto", color_continuous_scale="Viridis",
                     labels=dict(x="Memory", y="Worker IP")
                 )
                 fig_n.update_layout(height=250, margin=dict(l=0,r=0,t=0,b=0), coloraxis_showscale=False)
                 st.plotly_chart(fig_n, use_container_width=True, key=f"n_{state['step']}")

    with col_chart:
        st.subheader("👁️ Visual Cortex (Live Replay)")
        history = state.get("history", [])
        if history:
            df_full = pd.DataFrame(history)
            df_hist = df_full.tail(100).copy()
            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x=df_hist["t"], open=df_hist["o"], high=df_hist["h"],
                low=df_hist["l"], close=df_hist["c"], name="Market"
            ))
            
            # Highlight Position
            if "pos" in df_hist.columns:
                long_mask = df_hist["pos"] > 0
                if long_mask.any():
                    fig.add_trace(go.Scatter(x=df_hist["t"], y=df_hist["h"] * 1.0005, mode='lines', line=dict(color='rgba(0, 204, 150, 0.5)', width=3), name="LONG Active", hoverinfo='none'))
                short_mask = df_hist["pos"] < 0
                if short_mask.any():
                    fig.add_trace(go.Scatter(x=df_hist["t"], y=df_hist["l"] * 0.9995, mode='lines', line=dict(color='rgba(255, 75, 75, 0.5)', width=3), name="SHORT Active", hoverinfo='none'))

            # Action Signals (INCLUDING EXITS)
            # Map all exit actions to SELL/BUY for visualization
            def map_action(a):
                if a in ["BUY", "SELL"]: return a
                if a in ["TP_SCALP", "TS_ACTIVE", "EXIT_NEUTRAL", "REVERSAL_EXIT"]: return "EXIT"
                return None

            df_hist["act_val"] = df_hist["act"].apply(map_action)
            
            buy_ents = df_hist[df_hist["act"] == "BUY"]
            sell_ents = df_hist[df_hist["act"] == "SELL"]
            exit_ents = df_hist[df_hist["act_val"] == "EXIT"]

            if not buy_ents.empty:
                fig.add_trace(go.Scatter(x=buy_ents["t"], y=buy_ents["l"]*0.9985, mode="markers", marker=dict(symbol="triangle-up", size=16, color="#00CC96", line=dict(width=2, color="white")), name="BUY ENTRY"))
            if not sell_ents.empty:
                fig.add_trace(go.Scatter(x=sell_ents["t"], y=sell_ents["h"]*1.0015, mode="markers", marker=dict(symbol="triangle-down", size=16, color="#FF4B4B", line=dict(width=2, color="white")), name="SELL ENTRY"))
            if not exit_ents.empty:
                fig.add_trace(go.Scatter(x=exit_ents["t"], y=exit_ents["c"], mode="markers", marker=dict(symbol="x", size=12, color="yellow", line=dict(width=1, color="black")), name="EXIT (TP/TS)"))

            fig.update_layout(template="plotly_dark", xaxis_rangeslider_visible=False, height=500, margin=dict(l=0,r=0,t=0,b=0))
            st.plotly_chart(fig, use_container_width=True, key=f"chart_{state['step']}")
        else:
            st.warning("Waiting for history buffer...")

    # Cycle speed control & rerun
    time.sleep(1.0 / speed)
    st.rerun()

if __name__ == "__main__":
    render()
