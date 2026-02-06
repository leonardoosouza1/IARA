import streamlit as st
import subprocess
import os
import signal # For killing the process
import time
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import numpy as np

# Constants
TRAIN_SCRIPT = "src/brain/train_v8.py"
LOG_DIR = "logs"
METRICS_FILE = os.path.join(LOG_DIR, "metrics.csv")
SNAPSHOT_DIR = os.path.join(LOG_DIR, "snapshots")
PID_FILE = "training_process.pid"

def is_training_running():
    if os.path.exists(PID_FILE):
        try:
            with open(PID_FILE, "r") as f:
                pid = int(f.read().strip())
            os.kill(pid, 0) # Check if process is alive
            return True
        except OSError:
            os.remove(PID_FILE)
    return False

def stop_training():
    if os.path.exists(PID_FILE):
        with open(PID_FILE, "r") as f:
            pid = int(f.read().strip())
        try:
            os.kill(pid, signal.SIGTERM)
            st.toast(f"🛑 Process {pid} terminated.", icon="🛑")
        except OSError:
            pass
        os.remove(PID_FILE)

def render():
    # Note: st.set_page_config must NOT be called here as it is set in app.py

    # --- UI TABS ---
    st.title("🧪 Trinity Training Lab Pro")
    tab_config, tab_monitor, tab_forensics = st.tabs(["⚙️ Config & Launch", "📈 Live Monitor", "🧠 Forensic Lab"])

    # === TAB 1: CONFIG & LAUNCH ===
    with tab_config:
        col1, col2 = st.columns([1, 2])
        with col1:
            st.subheader("Mission Paramters")
            epochs = st.slider("Epochs", 1, 100, 10)
            lr = st.number_input("Learning Rate", value=0.001, format="%.5f")
            batch_size = st.selectbox("Batch Size", [16, 32, 64, 128, 256, 512, 1024], index=1)
            device_mode = st.radio("Compute Unit", ["GPU (ROCm)", "CPU"], index=0)
            
            launch_btn = st.button("🚀 DEPLOY TRAINING MISSION", type="primary", disabled=is_training_running())
            stop_btn = st.button("🛑 ABORT MISSION", type="secondary", disabled=not is_training_running())

        with col2:
            st.subheader("System Status")
            if is_training_running():
                st.success("🟢 TRAINING IN PROGRESS")
                st.markdown("Go to **Live Monitor** tab to see telemetry.")
            else:
                st.warning("⚪ SYSTEM IDLE")

        if launch_btn:
            cmd = ["python3", TRAIN_SCRIPT, "--epochs", str(epochs), "--lr", str(lr), "--batch_size", str(batch_size)]
            if device_mode == "CPU":
                cmd.append("--cpu")
                
            # Spawn detached process
            log_file = open("training_std.log", "w")
            proc = subprocess.Popen(cmd, stdout=log_file, stderr=log_file, preexec_fn=os.setpgrp)
            
            with open(PID_FILE, "w") as f:
                f.write(str(proc.pid))
                
            st.toast(f"🚀 Mission Deployed! PID: {proc.pid}", icon="🚀")
            time.sleep(1)
            st.rerun()
            
        if stop_btn:
            stop_training()
            st.rerun()

    # === TAB 2: LIVE MONITOR ===
    with tab_monitor:
        st.markdown("### 📡 Real-time Telemetry")
        placeholder = st.empty()
        
        if st.toggle("Auto-Refresh Monitor", value=True):
            time.sleep(1)
            st.rerun()
            
        if os.path.exists(METRICS_FILE):
            try:
                df = pd.read_csv(METRICS_FILE)
                if not df.empty:
                    col_m1, col_m2 = st.columns(2)
                    
                    # Loss Chart
                    fig_loss = px.line(df, x="step", y="loss", title="Loss Convergence", template="plotly_dark")
                    fig_loss.update_traces(line_color='#FF4B4B')
                    col_m1.plotly_chart(fig_loss, key="loss_chart", use_container_width=True)
                    
                    # Accuracy Chart
                    fig_acc = px.line(df, x="step", y="accuracy", title="Validation Accuracy", template="plotly_dark")
                    fig_acc.update_traces(line_color='#00CC96')
                    col_m2.plotly_chart(fig_acc, key="acc_chart", use_container_width=True)
                    
                    with st.expander("Show Raw Data"):
                        st.dataframe(df.tail(10))
                else:
                    st.info("Waiting for telemetry data...")
            except Exception as e:
                st.error(f"Telemetry Error: {e}")
        else:
            st.info("No telemetry file found. Start training first.")

    # === TAB 3: FORENSIC LAB ===
    with tab_forensics:
        st.markdown("### 🕵️ Forensic Replay (Time Travel)")
        st.info("Analyze the internal state of the Trinity Brain during past epochs.")
        
        # 1. Snapshot Loader
        if not os.path.exists(SNAPSHOT_DIR):
            st.warning(f"No snapshots found in {SNAPSHOT_DIR}")
        else:
            snapshots = sorted([f for f in os.listdir(SNAPSHOT_DIR) if f.endswith(".json")])
            if not snapshots:
                st.warning("No snapshots available yet.")
            else:
                selected_snap = st.selectbox("Select Epoch Snapshot", snapshots, index=len(snapshots)-1)
                
                # Load Data
                with open(os.path.join(SNAPSHOT_DIR, selected_snap), "r") as f:
                    data = json.load(f)
                    
                # Data Containers
                # expert_conviction: {long: [], short: []}
                long_curve = data["expert_conviction"]["long"]
                short_curve = data["expert_conviction"]["short"]
                inputs = data["sample_input"] # [Seq]
                seq_len = len(inputs)
                
                # Defensive Fix for Legacy/Scalar Snapshots
                if isinstance(long_curve, (float, int)):
                    long_curve = [long_curve] * seq_len
                if isinstance(short_curve, (float, int)):
                    short_curve = [short_curve] * seq_len
                
                # 2. Time Travel Slider
                selected_tick = st.slider("⏳ Time Travel (Tick)", 0, seq_len-1, seq_len-1)
                
                # 3. Visualizations
                col_f1, col_f2 = st.columns([2, 1])
                
                with col_f1:
                    st.subheader("🔎 Decision Architecture")
                    
                    # Chart: Price vs Expert Activation
                    # We want to see price history up to Tick, and Expert decision AT Tick.
                    
                    # Expert Bar Chart
                    expert_fig = go.Figure()
                    expert_fig.add_trace(go.Bar(
                        x=["Long Expert", "Short Expert"],
                        y=[long_curve[selected_tick], short_curve[selected_tick]],
                        marker_color=['#00CC96', '#FF4B4B'],
                        text=[f"{long_curve[selected_tick]:.2f}", f"{short_curve[selected_tick]:.2f}"],
                        textposition='auto'
                    ))
                    expert_fig.update_layout(title=f"Expert Conviction at Tick {selected_tick}", template="plotly_dark", yaxis_range=[-5, 5])
                    st.plotly_chart(expert_fig, key="expert_chart", use_container_width=True)
                    
                    # Price Context Chart
                    price_fig = px.line(y=inputs[:selected_tick+1], title="Market Context (What the Brain Saw)", template="plotly_dark")
                    price_fig.add_vline(x=selected_tick, line_dash="dash", line_color="yellow")
                    st.plotly_chart(price_fig, key="price_chart", use_container_width=True)

                with col_f2:
                    st.subheader("🧠 Neural Activity")
                    
                    # Feature Importance (Global for this sample)
                    feats = data["feature_importance"]
                    feat_fig = px.bar(x=list(range(len(feats))), y=feats, title="Feature Gating (Global)", labels={'x': 'Feature ID', 'y': "Gate Open %"}, template="plotly_dark")
                    st.plotly_chart(feat_fig, key="feat_chart", use_container_width=True)
                    
                    # Hidden State Heatmap (Current Tick)
                    # hidden_activity is [Hidden, Seq]. We want the column at selected_tick
                    hidden_state = np.array(data["hidden_activity"]) # [Hidden, Seq]
                    current_hidden = hidden_state[:, selected_tick].reshape(16, 8) # Reshape 128 to 16x8 grid
                    
                    heatmap_fig = px.imshow(current_hidden, title=f"Neuron Activation (Tick {selected_tick})", color_continuous_scale="Viridis", template="plotly_dark")
                    st.plotly_chart(heatmap_fig, key="heatmap_chart", use_container_width=True)
