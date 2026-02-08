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
TRAIN_SCRIPT = "src/brain/train_v9.py"  # Updated to v9 Swarm Intelligence
LOG_DIR = "logs"
METRICS_FILE = os.path.join(LOG_DIR, "metrics.csv")
SNAPSHOT_DIR = os.path.join(LOG_DIR, "snapshots")
PID_FILE = "training_process.pid"
TELEMETRY_FILE = os.path.join(LOG_DIR, "telemetry.json")
MODELS_DIR = os.path.join(LOG_DIR, "models")
MODEL_ZOO_DIR = os.path.join(LOG_DIR, "model_zoo")

def save_checkpoint():
    """Send signal to training process to save checkpoint"""
    if os.path.exists(PID_FILE):
        try:
            with open(PID_FILE, "r") as f:
                pid = int(f.read().strip())
            os.kill(pid, signal.SIGUSR1)
            st.toast("💾 Save signal sent!", icon="✅")
        except Exception as e:
            st.error(f"Failed to send save signal: {e}")

def clear_logs():
    """Clear metric logs"""
    try:
        if os.path.exists(METRICS_FILE): os.remove(METRICS_FILE)
        # Also clear v9 file if exists
        v9_metrics = os.path.join(LOG_DIR, "metrics_v9.csv")
        if os.path.exists(v9_metrics): os.remove(v9_metrics)
        st.toast("🧹 Logs cleared!", icon="✅")
        time.sleep(0.5)
        st.rerun()
    except Exception as e:
        st.error(f"Failed to clear logs: {e}")

def is_training_running():
    if os.path.exists(PID_FILE):
        try:
            with open(PID_FILE, "r") as f:
                pid = int(f.read().strip())
            
            # Check if process exists via signal
            os.kill(pid, 0)
            
            # Check if Zombie (Linux only)
            try:
                with open(f"/proc/{pid}/stat", "r") as stat_file:
                    state = stat_file.read().split()[2]
                    if state == 'Z':
                        # It's a zombie, cleanup
                        os.waitpid(pid, os.WNOHANG) # Reap it
                        os.remove(PID_FILE)
                        return False
            except:
                pass # Not linux or permission error
                
            return True
        except OSError:
            if os.path.exists(PID_FILE): os.remove(PID_FILE)
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
    tab_config, tab_monitor, tab_forensics, tab_zoo = st.tabs(["⚙️ Config & Launch", "📈 Live Monitor", "🧠 Forensic Lab", "🦁 Model Zoo"])

    # === TAB 1: CONFIG & LAUNCH ===
    with tab_config:
        col1, col2 = st.columns([1, 2])
        with col1:
            st.subheader("Mission Paramters")
            epochs = st.slider("Epochs", 1, 100, 10)
            lr = st.number_input("Learning Rate", value=0.001, format="%.5f")
            batch_size = st.selectbox("Batch Size", [16, 32, 64, 128, 256, 512, 1024], index=1)
            device_mode = st.radio("Compute Unit", ["GPU (ROCm)", "CPU"], index=0)
            
            # Resume/Reset Logic
            start_mode = st.radio("Starting Point", ["Resume Training (Load Best Checkpoint)", "Start Fresh (Reset & Backup)"], index=0, help="Resume: Continues from best_model_v9.pth. Fresh: Backs up old model and starts new.")
            
            # Advanced Options
            with st.expander("Advanced Configuration"):
                disable_amp = st.checkbox("Disable Mixed Precision (AMP)", value=True, help="Check this if training hangs on ROCm")
                debug_mode = st.checkbox("Debug Mode (Log Steps)", value=True, help="Detailed logging for diagnosis")
            
            launch_btn = st.button("🚀 DEPLOY TRAINING MISSION", type="primary", disabled=is_training_running())
            stop_btn = st.button("🛑 ABORT MISSION", type="secondary", disabled=not is_training_running())
            
            # Manual Controls
            if is_training_running():
                save_btn = st.button("💾 SAVE CHECKPOINT", help="Force save current model state")
                if save_btn:
                    save_checkpoint()
                    
            if not is_training_running():
                clear_logs_btn = st.button("🧹 CLEAR LOGS", help="Delete old metric logs to reset charts")
                if clear_logs_btn:
                    clear_logs()

        with col2:
            st.subheader("System Status")
            if is_training_running():
                st.success("🟢 TRAINING IN PROGRESS")
                st.markdown("Go to **Live Monitor** tab to see telemetry.")
            else:
                st.warning("⚪ SYSTEM IDLE")
            
            # Leaderboard Display
            leaderboard_path = os.path.join(LOG_DIR, "leaderboard.json")
            if os.path.exists(leaderboard_path):
                try:
                    with open(leaderboard_path, "r") as f:
                        lb_data = json.load(f)
                    
                    st.markdown("#### 🏆 Top Models")
                    if lb_data:
                        best = lb_data[0]
                        b_col1, b_col2 = st.columns(2)
                        b_col1.metric("Best Loss", f"{best['loss']:.4f}", f"Epoch {best['epoch']}")
                        b_col2.metric("Best AUC", f"{best.get('auc', 0):.4f}", delta_color="normal")
                        
                        lb_df = pd.DataFrame(lb_data)[["epoch", "loss", "accuracy", "auc", "timestamp"]]
                        st.dataframe(lb_df, height=150, hide_index=True)
                    else:
                        st.info("No models ranked yet.")
                except:
                    pass

        if launch_btn:
            cmd = ["python3", TRAIN_SCRIPT, "--epochs", str(epochs), "--lr", str(lr), "--batch_size", str(batch_size)]
            if device_mode == "CPU":
                cmd.append("--cpu")
            
            if "Resume" in start_mode:
                cmd.append("--resume")
            
            if disable_amp:
                cmd.append("--no_amp")
                
            if debug_mode:
                cmd.append("--debug")
                
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
        
        auto_refresh = st.toggle("Auto-Refresh Monitor", value=True)
        
        # 📡 Live Progress section
        if os.path.exists(TELEMETRY_FILE):
            try:
                with open(TELEMETRY_FILE, "r") as f:
                    telemetry = json.load(f)
                
                if telemetry.get("status") == "TRAINING":
                    curr_step = telemetry.get("current_step", 0)
                    total_steps = telemetry.get("total_steps", 1)
                    curr_epoch = telemetry.get("current_epoch", 0)
                    total_epochs = telemetry.get("total_epochs", 100)
                    
                    st.markdown(f"#### 🚀 Mission Progress")
                    progress = min(100, int((curr_step / total_steps) * 100))
                    
                    p_col1, p_col2 = st.columns([3, 1])
                    p_col1.progress(progress / 100, text=f"Step {curr_step} / {total_steps} ({progress}%)")
                    p_col2.metric("Epoch", f"{curr_epoch + 1}")
                    
                    st.divider()
            except:
                pass

            
        if os.path.exists(METRICS_FILE):
            try:
                df = pd.read_csv(METRICS_FILE)
                if not df.empty:
                    col_m1, col_m2 = st.columns(2)
                    
                    # Loss Chart
                    fig_loss = px.line(df, x="step", y="loss", title="Loss Convergence", template="plotly_dark")
                    fig_loss.update_traces(line_color='#FF4B4B')
                    col_m1.plotly_chart(fig_loss, key="loss_chart", width="stretch")
                    
                    # Accuracy Chart
                    fig_acc = px.line(df, x="step", y="accuracy", title="Validation Accuracy", template="plotly_dark")
                    fig_acc.update_traces(line_color='#00CC96')
                    col_m2.plotly_chart(fig_acc, key="acc_chart", width="stretch")
                    
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
        st.info("Analise o estado interno do Swarm Brain durante épocas passadas. Workers M5 e H1 são mapeados como \"Long\" e \"Short\" experts.")
        
        # 1. Snapshot Loader
        if not os.path.exists(SNAPSHOT_DIR):
            st.warning(f"No snapshots found in {SNAPSHOT_DIR}")
        else:
            # Sort properly by epoch number (integer)
            try:
                snapshots = sorted(
                    [f for f in os.listdir(SNAPSHOT_DIR) if f.endswith(".json")],
                    key=lambda x: int(x.split("_")[1].split(".")[0])
                )
            except:
                snapshots = sorted([f for f in os.listdir(SNAPSHOT_DIR) if f.endswith(".json")]) # Fallback

            if not snapshots:
                st.warning("No snapshots available yet.")
            else:
                # --- SNAPSHOT MANAGEMENT ---
                with st.expander("🗑️ Manage Snapshots (Clean History)"):
                    to_delete = st.multiselect("Select snapshots to remove:", snapshots)
                    if st.button(f"Delete {len(to_delete)} Snapshots", type="primary", disabled=len(to_delete)==0):
                        for snap in to_delete:
                            try:
                                os.remove(os.path.join(SNAPSHOT_DIR, snap))
                            except Exception as e:
                                st.error(f"Failed to delete {snap}: {e}")
                        st.toast(f"Deleted {len(to_delete)} snapshots.", icon="🗑️")
                        time.sleep(1)
                        st.rerun()

                # Re-read snapshots after potential deletion
                try:
                    snapshots = sorted(
                        [f for f in os.listdir(SNAPSHOT_DIR) if f.endswith(".json")],
                        key=lambda x: int(x.split("_")[1].split(".")[0])
                    )
                except:
                     snapshots = sorted([f for f in os.listdir(SNAPSHOT_DIR) if f.endswith(".json")])

                if not snapshots:
                    st.warning("No snapshots available after cleanup.")
                    st.stop()
                
                selected_snap = st.selectbox("Select Epoch Snapshot", snapshots, index=len(snapshots)-1)
                
                # Load Data
                try:
                    with open(os.path.join(SNAPSHOT_DIR, selected_snap), "r") as f:
                        data = json.load(f)
                except json.JSONDecodeError:
                    st.error(f"Snapshot {selected_snap} is corrupted.")
                    return
                    
                # Data Containers
                # expert_conviction: {long: [], short: []}
                long_curve = data["expert_conviction"]["long"]
                short_curve = data["expert_conviction"]["short"]
                
                # Handle inputs for seq_len calculation
                if "ohlc" in data:
                    inputs = data["ohlc"]["close"] # Use Close price as reference
                elif "sample_input" in data:
                     inputs = data["sample_input"]
                else:
                     inputs = [0] * len(long_curve) # Fallback empty
                
                seq_len = len(inputs)
                
                # Defensive Fix for Legacy/Scalar Snapshots
                if isinstance(long_curve, (float, int)):
                    long_curve = [long_curve] * seq_len
                if isinstance(short_curve, (float, int)):
                    short_curve = [short_curve] * seq_len
                    
                # Robustness: Ensure list length matches seq_len
                if len(long_curve) < seq_len:
                     # Pad with last value
                     long_curve += [long_curve[-1]] * (seq_len - len(long_curve))
                if len(short_curve) < seq_len:
                     short_curve += [short_curve[-1]] * (seq_len - len(short_curve))
                
                # Truncate if too long
                long_curve = long_curve[:seq_len]
                short_curve = short_curve[:seq_len]
                
                # 2. Time Travel Slider
                selected_tick = st.slider("⏳ Time Travel (Tick)", 0, seq_len-1, seq_len-1)
                
                # 3. Visualizations (The Visual Cortex)
                
                # Layout:
                # [ Candlestick Chart with Trade Markers ]
                # [ Neural Heatmap (Spectrogram)         ]
                
                # --- A. Context & Market Data ---
                # Retrieve OHLC from snapshot if available, else mocking it for legacy snapshots would be hard.
                # Assuming new snapshots have 'ohlc' key.
                
                has_ohlc = "ohlc" in data
                
                col_cortex_1, col_cortex_2 = st.columns([3, 1])
                
                with col_cortex_1:
                    st.subheader("👁️ Visual Cortex (Market View)")
                    
                    if has_ohlc:
                        ohlc = data["ohlc"]
                        times = ohlc["time"]
                        opens = ohlc["open"]
                        highs = ohlc["high"]
                        lows = ohlc["low"]
                        closes = ohlc["close"]
                        
                        # Trim to seq_len just in case
                        times = times[:seq_len]
                        opens = opens[:seq_len]
                        highs = highs[:seq_len]
                        lows = lows[:seq_len]
                        closes = closes[:seq_len]
                        
                        fig = go.Figure()
                        
                        # 1. Candlestick Main
                        fig.add_trace(go.Candlestick(
                            x=list(range(seq_len)),
                            open=opens, high=highs, low=lows, close=closes,
                            name="Market",
                            increasing_line_color='#00CC96', decreasing_line_color='#FF4B4B'
                        ))
                        
                        # 2. Brain Overlays (Trade Markers)
                        # Identify points of high conviction
                        buy_signals = [i for i, v in enumerate(long_curve) if v > 1.5] # Threshold 1.5
                        sell_signals = [i for i, v in enumerate(short_curve) if v > 1.5]
                        
                        if buy_signals:
                            fig.add_trace(go.Scatter(
                                x=buy_signals, 
                                y=[lows[i] * 0.9995 for i in buy_signals],
                                mode='markers',
                                marker=dict(symbol='triangle-up', size=12, color='#00CC96'),
                                name="AI Buy Conviction"
                            ))
                            
                        if sell_signals:
                             fig.add_trace(go.Scatter(
                                x=sell_signals, 
                                y=[highs[i] * 1.0005 for i in sell_signals],
                                mode='markers',
                                marker=dict(symbol='triangle-down', size=12, color='#FF4B4B'),
                                name="AI Sell Conviction"
                            ))
                            
                        # 3. Time Travel Marker
                        fig.add_vline(x=selected_tick, line_width=2, line_dash="dash", line_color="yellow")
                        
                        fig.update_layout(
                            title="Market Context + AI Intent",
                            template="plotly_dark",
                            xaxis_rangeslider_visible=False,
                            height=400,
                            margin=dict(l=0, r=0, t=30, b=0)
                        )
                        st.plotly_chart(fig, key="cortex_chart", width="stretch")
                        
                    else:
                        st.warning("⚠️ Legacy Snapshot: No OHLC Data found. Run new training to see Visual Cortex.")
                        # Fallback to simple line
                        fig = px.line(y=inputs, title="Normalized Input (Legacy)")
                        fig.add_vline(x=selected_tick, line_color="yellow")
                        st.plotly_chart(fig, width="stretch")

                with col_cortex_2:
                    st.subheader("🧠 Synaptic Map")
                    
                    # 1. Expert Bar Chart (Current Tick)
                    expert_fig = go.Figure()
                    expert_fig.add_trace(go.Bar(
                        x=["Long", "Short"],
                        y=[long_curve[selected_tick], short_curve[selected_tick]],
                        marker_color=['#00CC96', '#FF4B4B'],
                        text=[f"{long_curve[selected_tick]:.2f}", f"{short_curve[selected_tick]:.2f}"],
                        textposition='auto'
                    ))
                    expert_fig.update_layout(
                        title=f"Conviction (Tick {selected_tick})", 
                        template="plotly_dark", 
                        yaxis_range=[-2, 5],
                        height=200,
                        margin=dict(l=0, r=0, t=30, b=0)
                    )
                    st.plotly_chart(expert_fig, key="expert_conviction_bar", width="stretch")
                    
                    
                    # 2. Feature Gating (Global)
                    feats = data["feature_importance"]
                    
                    # Map Feature Indices to Names (Based on train_v8.py generator)
                    # 0=NormClose, 1=NormMom, 2=NormVol, 3..15=Noise
                    feat_names = []
                    for i in range(len(feats)):
                        if i == 0: feat_names.append("F0: Price")
                        elif i == 1: feat_names.append("F1: Momentum")
                        elif i == 2: feat_names.append("F2: Volatility")
                        else: feat_names.append(f"F{i}: Noise")
                        
                    # Top 5 Features
                    indices = np.argsort(feats)[::-1][:5]
                    top_feats = [feats[i] for i in indices]
                    top_names = [feat_names[i] for i in indices]
                    
                    feat_fig = px.bar(x=top_names, y=top_feats, title="Active Senses (Input Gating)", template="plotly_dark", height=200)
                    feat_fig.update_layout(margin=dict(l=0, r=0, t=30, b=0))
                    st.plotly_chart(feat_fig, key="feat_imp_chart", width="stretch")

                # --- B. System 2 Narrative (The "Thought") ---
                # Mocking logic to generate "Language" from the internal state
                sys2_msg = ""
                top_idx = indices[0]
                expert_diff = long_curve[selected_tick] - short_curve[selected_tick]
                
                if top_idx == 0: focus = "Price Action"
                elif top_idx == 1: focus = "Momentum Acceleration"
                elif top_idx == 2: focus = "Volatility/Risk"
                else: focus = "Market Noise"
                
                if expert_diff > 1.0:
                    intent = "Strong BUY"
                    reason = "Upward pressure confirmed."
                elif expert_diff < -1.0:
                    intent = "Strong SELL"
                    reason = "Downward pressure confirmed."
                else:
                    intent = "HOLD"
                    reason = "Uncertainty in signal."
                    
                sys2_msg = f"**System 2 Analysis**: Focusing heavily on **{focus}**. Detect {intent} signal. {reason} (Confidence: {abs(expert_diff):.2f})"
                
                st.info(sys2_msg, icon="🧠")

                # --- C. Neural Spectrogram (Bottom) ---
                st.markdown("#### ⚡ Neural Activation Spectrogram")
                # Ensure we have the full sequence of hidden states
                hidden_state = np.array(data["hidden_activity"]) # [Hidden, Seq]
                
                # Trim to seq_len
                hidden_state = hidden_state[:, :seq_len]
                
                # Colorscale: Magma or Viridis. Magma looks more "electric".
                spectro_fig = px.imshow(
                    hidden_state, 
                    labels=dict(x="Time (Tick)", y="Neuron ID", color="Activation"),
                    x=list(range(seq_len)),
                    color_continuous_scale="Magma",
                    template="plotly_dark",
                    aspect="auto",
                    height=250
                )
                spectro_fig.update_layout(
                    margin=dict(l=0, r=0, t=0, b=0),
                    xaxis_title="Time Sequence"
                )
                # Mark current tick
                spectro_fig.add_vline(x=selected_tick, line_color="yellow", line_width=2)
                
                st.plotly_chart(spectro_fig, key="spectrogram", width="stretch")

    # === TAB 4: MODEL ZOO ===
    with tab_zoo:
        st.markdown("### 🦁 Model Zoo - Gerenciador de Modelos")
        
        col_backups, col_zoo = st.columns(2)
        
        # --- LEFT: Backups (logs/models) ---
        with col_backups:
            st.markdown("#### 📦 Backups Temporários")
            st.caption(f"Diretório: `{MODELS_DIR}`")
            
            if os.path.exists(MODELS_DIR):
                backup_files = sorted([f for f in os.listdir(MODELS_DIR) if f.endswith(".pth")], reverse=True)
                
                if backup_files:
                    st.info(f"**{len(backup_files)}** modelos encontrados ({sum(os.path.getsize(os.path.join(MODELS_DIR, f)) for f in backup_files) / 1024 / 1024:.1f} MB)")
                    
                    # Seleção para deletar
                    to_delete = st.multiselect("Selecione para deletar:", backup_files, key="delete_backups")
                    
                    col_del, col_promote = st.columns(2)
                    
                    if col_del.button("🗑️ Deletar Selecionados", disabled=len(to_delete) == 0, type="secondary"):
                        for f in to_delete:
                            try:
                                os.remove(os.path.join(MODELS_DIR, f))
                            except Exception as e:
                                st.error(f"Erro ao deletar {f}: {e}")
                        st.toast(f"Deletados {len(to_delete)} modelos!", icon="🗑️")
                        time.sleep(0.5)
                        st.rerun()
                    
                    # Promoção para Zoo
                    to_promote = st.selectbox("Promover para Zoo:", ["-- Selecione --"] + backup_files, key="promote_model")
                    promote_name = st.text_input("Nome no Zoo (ex: v9_auc78):", key="promote_name")
                    
                    if col_promote.button("🏆 Promover para Zoo", disabled=(to_promote == "-- Selecione --" or not promote_name)):
                        src = os.path.join(MODELS_DIR, to_promote)
                        dst = os.path.join(MODEL_ZOO_DIR, f"{promote_name}.pth")
                        try:
                            os.makedirs(MODEL_ZOO_DIR, exist_ok=True)
                            import shutil
                            shutil.copy2(src, dst)
                            st.toast(f"Modelo promovido para Zoo como {promote_name}.pth!", icon="🏆")
                            time.sleep(0.5)
                            st.rerun()
                        except Exception as e:
                            st.error(f"Erro: {e}")
                    
                    # Lista com detalhes
                    with st.expander("Ver detalhes dos backups"):
                        for f in backup_files[:10]:
                            fpath = os.path.join(MODELS_DIR, f)
                            size = os.path.getsize(fpath) / 1024
                            mtime = time.strftime("%d/%m %H:%M", time.localtime(os.path.getmtime(fpath)))
                            st.text(f"{f} ({size:.0f} KB) - {mtime}")
                else:
                    st.success("Nenhum backup pendente! 🎉")
            else:
                st.warning("Diretório de modelos não encontrado.")
        
        # --- RIGHT: Zoo (logs/model_zoo) ---
        with col_zoo:
            st.markdown("#### 🏆 Model Zoo (Produção)")
            st.caption(f"Diretório: `{MODEL_ZOO_DIR}`")
            
            if os.path.exists(MODEL_ZOO_DIR):
                zoo_files = sorted([f for f in os.listdir(MODEL_ZOO_DIR) if f.endswith(".pth")], reverse=True)
                
                if zoo_files:
                    for f in zoo_files:
                        fpath = os.path.join(MODEL_ZOO_DIR, f)
                        size = os.path.getsize(fpath) / 1024 / 1024
                        mtime = time.strftime("%d/%m/%Y %H:%M", time.localtime(os.path.getmtime(fpath)))
                        
                        with st.container():
                            st.markdown(f"**🦁 {f}**")
                            st.caption(f"{size:.2f} MB | {mtime}")
                            if st.button(f"🗑️ Remover", key=f"del_zoo_{f}"):
                                os.remove(fpath)
                                st.toast(f"Removido {f} do Zoo!", icon="🗑️")
                                time.sleep(0.5)
                                st.rerun()
                            st.divider()
                else:
                    st.info("Zoo vazio. Promova modelos da esquerda!")
            else:
                st.warning("Diretório do Zoo não encontrado.")

    # Auto-Refresh Logic (Moved to end to prevent blocking)
    if auto_refresh:
        time.sleep(1)
        st.rerun()

