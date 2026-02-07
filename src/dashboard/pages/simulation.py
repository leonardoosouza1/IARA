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

# Constants
LOG_DIR = "logs"
ZOO_DIR = os.path.join(LOG_DIR, "model_zoo")
SIM_DATA = "data/raw/EURUSD_synthetic.csv"
SIM_SCRIPT = "src/brain/simulate_v8.py"
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
        # In real app, date picker. For now, text.
        start_date = st.text_input("Start Date (YYYY-MM-DD)", "2023-01-01", disabled=running)
        
        # Speed
        speed = st.slider("Playback Speed (Steps/Sec)", 1, 100, 10, disabled=running)
        
        if not running:
            if st.button("🚀 START SIMULATION", type="primary"):
                # Pre-flight Checks
                if not selected_model:
                     st.error("No model selected!")
                     st.stop()
                
                if not os.path.exists(SIM_DATA):
                     st.error(f"Data file not found: {SIM_DATA}")
                     st.stop()
                     
                if not os.path.exists(SIM_SCRIPT):
                     st.error(f"Script not found: {SIM_SCRIPT}")
                     st.stop()

                # Cleanup old artifacts
                if os.path.exists(SIM_SNAPSHOT_FILE):
                    try: os.remove(SIM_SNAPSHOT_FILE)
                    except: pass
                
                # Launch
                model_path = os.path.join(ZOO_DIR, selected_model)
                cmd = [
                    sys.executable, "-u", SIM_SCRIPT, # Use same python as streamlit, unbuffered
                    "--model", model_path,
                    "--data", SIM_DATA,
                    "--start_date", start_date,
                    "--speed", str(speed)
                ]
                
                st.info(f"Executing: {' '.join(cmd)}")
                
                # Log file for debug
                with open("sim_std.log", "w") as log:
                    subprocess.Popen(cmd, stdout=log, stderr=log, preexec_fn=os.setpgrp)
                
                st.toast("Simulation Started!", icon="🚀")
                time.sleep(1)
                st.rerun()
        else:
            st.info(f"🟢 Sim Running (PID {pid})")
            if st.button("🛑 STOP SIMULATION", type="primary"):
                stop_sim()
                st.rerun()
                
            # Log Viewer while running
            with st.expander("View Live Logs", expanded=False):
                if os.path.exists("sim_std.log"):
                    with open("sim_std.log", "r") as f:
                        lines = f.readlines()
                        st.code("".join(lines[-20:]))

    # --- MAIN STAGE ---
    if not running:
        st.info("👈 Configure and Start the Simulation from the Sidebar.")
        
        # Show last result/error
        if os.path.exists("sim_std.log"):
            st.markdown("### Last Session Log")
            try:
                with open("sim_std.log", "r") as f:
                    st.code(f.read())
            except: pass
        return

    # --- LIVE REPLAY VIEW ---
    placeholder = st.empty()
    
    # Auto-refresh loop
    last_step = -1
    
    while True:
        if not os.path.exists(SIM_SNAPSHOT_FILE):
            time.sleep(0.5)
            continue
            
        try:
            with open(SIM_SNAPSHOT_FILE, "r") as f:
                state = json.load(f)
        except:
            time.sleep(0.1)
            continue
            
        if state.get("completed"):
            st.success("Simulation Completed!")
            stop_sim()
            break
        
        # Optimize: Only render if step changed
        current_step = state.get("step", -1)
        if current_step == last_step:
            time.sleep(0.1)
            continue
        last_step = current_step
            
        with placeholder.container():
            # TOP BAR: METRICS
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Date", state["timestamp"])
            c2.metric("Price", f"{state['price']:.5f}")
            c3.metric("Equity", f"${state['equity']:.2f}")
            c4.metric("PnL", f"{state['pnl_pct']:.2f}%", delta=state['pnl_pct'])
            
            action_color = "gray"
            if state["action"] == "BUY": action_color = "green"
            elif state["action"] == "SELL": action_color = "red"
            c5.markdown(f"**Action**: :{action_color}[{state['action']}]")
            
            # VISUALIZATION
            col_chart, col_brain = st.columns([2, 1])
            
            with col_brain:
                st.subheader("🧠 Neural Activity")
                # Neural Grid
                neurons = state.get("neuronal_map", [])
                if neurons:
                     # Heatmap of [128, 256]. Display last 50 ticks for speed
                     arr = pd.DataFrame(neurons) # [128, 256]
                     
                     fig_n = px.imshow(
                         arr, 
                         aspect="auto",
                         color_continuous_scale="Magma",
                         labels=dict(x="Time Window (Memory)", y="Neuron ID")
                     )
                     fig_n.update_layout(height=400, margin=dict(l=0,r=0,t=0,b=0))
                     st.plotly_chart(fig_n, use_container_width=True, key=f"neuro_{state['step']}")
                
                # Biases
                st.progress(state.get("prob", 0.5), text=f"Buy Probability: {state.get('prob',0)*100:.1f}%")
                
                b1, b2 = st.columns(2)
                b1.metric("Long Bias", f"{state.get('long_bias', 0):.2f}")
                b2.metric("Short Bias", f"{state.get('short_bias', 0):.2f}")
                
            with col_chart:
                st.subheader("📈 Market View")
                # Ideally we build a history buffer here, but Streamlit rerun clears state unless session_state.
                # For simplicity, we just show current price point or rely on user knowing context.
                # Building a full chart in replay requires accumulating data efficiently.
                
                st.info(f"Tick: {state['step']} | Processing {state['timestamp']}")
                
                # Mock Chart: Current Price Indicator
                fig_p = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = state["price"],
                    title = {'text': "EURUSD"},
                    gauge = {'axis': {'range': [state["price"]*0.99, state["price"]*1.01]}}
                ))
                st.plotly_chart(fig_p, use_container_width=True, key=f"gauge_{state['step']}")

        time.sleep(1.0 / speed) 
        # Note: In Streamlit, this while loop with sleep blocks interaction. 
        # A better pattern uses st.empty() and st.rerun() but that flickers.
        # This while loop is a "Blocking Live View". To stop, user needs to kill the tab or we check a stop signal.
        # We checked `is_sim_running` at top, but inside loop we rely on file checks.
        
        # Check if user clicked stop (Streamlit interaction is hard inside loop).
        # We rely on the Sidebar 'Stop' button which is OUTSIDE this container. 
        # Actually sidebar is rendered once.
        # To make it responsive, we break often?
        pass

