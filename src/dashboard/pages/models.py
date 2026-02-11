import streamlit as st
import os
import json
import shutil
import time
from datetime import datetime
from pathlib import Path
import pandas as pd
import torch

# Constants
LOG_DIR = "logs"
MODELS_DIR = os.path.join(LOG_DIR, "models")
ZOO_DIR = os.path.join(LOG_DIR, "model_zoo")
LEADERBOARD_FILE = os.path.join(LOG_DIR, "leaderboard.json")
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_model.pth")
ACTIVE_MODEL_PATH = os.path.join(MODELS_DIR, "active_model.pth")
ACTIVE_MODEL_META = os.path.join(MODELS_DIR, "active_model.json")

# Ensure directories exist
os.makedirs(ZOO_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


def _detect_model_symbol() -> list:
    """Scan for symbol-specific best models in the models directory."""
    models = []
    for f in Path(MODELS_DIR).glob("best_model_v9_*.pth"):
        # Extract symbol from filename: best_model_v9_BTCUSD.pth -> BTCUSD
        parts = f.stem.replace("best_model_v9_", "")
        if parts and not parts.startswith("backup"):
            models.append({"filename": f.name, "symbol": parts, "path": str(f)})
    return models


def render():
    st.title("🗄️ Model Manager")
    st.info("Manage your trained Brains. Save the champions, discard the failures. Deploy models to live trading.")

    tab_current, tab_zoo, tab_deploy, tab_upload = st.tabs(
        ["🔥 Current Training", "🦁 Model Zoo", "🚀 Deploy", "📤 Upload"]
    )

    # --- TAB 1: CURRENT TRAINING ---
    with tab_current:
        st.subheader("Active Session Artifacts")
        
        # Find all symbol-specific models
        available_models = _detect_model_symbol()
        
        if available_models:
            st.success(f"Found {len(available_models)} trained model(s)")
            
            for model_info in available_models:
                with st.expander(f"🧠 {model_info['symbol']}", expanded=True):
                    # Try to find leaderboard for this symbol
                    lb_path = os.path.join(LOG_DIR, f"leaderboard_v9.json")
                    if os.path.exists(lb_path):
                        try:
                            with open(lb_path, "r") as f:
                                lb_data = json.load(f)
                            
                            if lb_data:
                                best = lb_data[0]
                                col1, col2, col3, col4 = st.columns(4)
                                col1.metric("AUC", f"{best.get('auc', 0):.4f}")
                                col2.metric("Loss", f"{best['loss']:.4f}")
                                col3.metric("Accuracy", f"{best['accuracy']:.2%}")
                                col4.metric("Epoch", best['epoch'])
                        except Exception:
                            pass
                    
                    # Save to Zoo button
                    auc_str = "0000"
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
                    default_name = f"swarm_v9_{model_info['symbol']}_{timestamp}.pth"
                    
                    save_name = st.text_input("Filename", value=default_name, key=f"save_{model_info['symbol']}")
                    
                    if st.button(f"💾 Save {model_info['symbol']} to Zoo", key=f"btn_save_{model_info['symbol']}"):
                        source_path = model_info['path']
                        if os.path.exists(source_path):
                            dest = os.path.join(ZOO_DIR, save_name)
                            shutil.copy(source_path, dest)
                            
                            # Save metadata
                            meta = {
                                "symbol": model_info['symbol'],
                                "market_type": "crypto" if model_info['symbol'] in ["BTCUSD", "ETHUSD", "SOLUSD", "DOGEUSD", "LTCUSD"] else "forex",
                                "architecture": "SwarmBrain_v9",
                                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "auc": 0.0,
                                "loss": 0.0,
                                "accuracy": 0.0,
                                "epoch": 0,
                            }
                            
                            # Try to get actual metrics from leaderboard
                            lb_path = os.path.join(LOG_DIR, "leaderboard_v9.json")
                            if os.path.exists(lb_path):
                                try:
                                    with open(lb_path, "r") as f:
                                        lb = json.load(f)
                                    if lb:
                                        meta.update({
                                            "auc": lb[0].get("auc", 0),
                                            "loss": lb[0].get("loss", 0),
                                            "accuracy": lb[0].get("accuracy", 0),
                                            "epoch": lb[0].get("epoch", 0),
                                        })
                                except Exception:
                                    pass
                            
                            meta_path = dest + ".json"
                            with open(meta_path, "w") as f:
                                json.dump(meta, f, indent=2)
                            
                            st.toast(f"Saved to {save_name}", icon="🦁")
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error("Model file not found!")
        else:
            # Fallback to legacy leaderboard check
            if os.path.exists(LEADERBOARD_FILE):
                try:
                    with open(LEADERBOARD_FILE, "r") as f:
                        lb_data = json.load(f)
                    if lb_data:
                        best = lb_data[0]
                        st.success(f"🏆 Current Champion found from Epoch {best['epoch']}")
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("AUC", f"{best.get('auc', 0):.4f}")
                        col2.metric("Loss", f"{best['loss']:.4f}")
                        col3.metric("Accuracy", f"{best['accuracy']:.2%}")
                        col4.metric("Date", best['timestamp'])
                    else:
                        st.warning("Leaderboard is empty.")
                except Exception as e:
                    st.error(f"Error reading leaderboard: {e}")
            else:
                st.info("No active training data found. Train a model first!")

    # --- TAB 2: MODEL ZOO ---
    with tab_zoo:
        st.subheader("🦁 Hall of Fame (Saved Models)")
        
        models = [f for f in os.listdir(ZOO_DIR) if f.endswith(".pth")]
        
        if not models:
            st.warning("Zoo is empty. Save a model from training or upload one.")
        else:
            # Parse Metadata
            zoo_data = []
            for m in models:
                full_path = os.path.join(ZOO_DIR, m)
                stat = os.stat(full_path)
                size_mb = stat.st_size / (1024 * 1024)
                created = datetime.fromtimestamp(stat.st_ctime).strftime("%Y-%m-%d %H:%M")
                
                # Check for sidecar JSON
                meta_path = full_path + ".json"
                auc = 0.0
                loss = 0.0
                acc = 0.0
                epoch = 0
                symbol = "Unknown"
                market = "Unknown"
                if os.path.exists(meta_path):
                    try:
                        with open(meta_path, "r") as f:
                            meta = json.load(f)
                            auc = meta.get("auc", 0)
                            loss = meta.get("loss", 0)
                            acc = meta.get("accuracy", 0)
                            epoch = meta.get("epoch", 0)
                            symbol = meta.get("symbol", "Unknown")
                            market = meta.get("market_type", "Unknown")
                    except Exception:
                        pass
                
                zoo_data.append({
                    "Filename": m,
                    "Symbol": symbol,
                    "Market": market.capitalize(),
                    "AUC": auc,
                    "Loss": loss,
                    "Accuracy": acc,
                    "Epoch": epoch,
                    "Size (MB)": f"{size_mb:.2f}",
                    "Created": created,
                })
            
            df = pd.DataFrame(zoo_data)
            
            # Filter by market
            market_filter = st.radio("Filter by Market", ["All", "Forex", "Crypto"], horizontal=True, key="zoo_filter")
            if market_filter != "All":
                df = df[df["Market"] == market_filter]
            
            # Sort by AUC descending by default
            df = df.sort_values("AUC", ascending=False)
            
            # Interactive Table
            st.dataframe(
                df,
                column_config={
                    "AUC": st.column_config.ProgressColumn(
                        "AUC",
                        help="Area Under Curve",
                        format="%.4f",
                        min_value=0,
                        max_value=1,
                    ),
                    "Accuracy": st.column_config.NumberColumn(
                        "Accuracy",
                        format="%.2f",
                    ),
                    "Epoch": st.column_config.NumberColumn(
                        "Epoch",
                        format="%d",
                    ),
                },
                width="stretch",
                hide_index=True
            )
            
            # Selection for Actions
            st.divider()
            col_sel, col_act1, col_act2 = st.columns([3, 1, 1])
            selected_model = col_sel.selectbox("Select Model", df["Filename"])
            
            if col_act1.button("🗑️ Delete", type="secondary"):
                if selected_model:
                    try:
                        target = os.path.join(ZOO_DIR, selected_model)
                        os.remove(target)
                        if os.path.exists(target + ".json"):
                            os.remove(target + ".json")
                        st.toast(f"Deleted {selected_model}")
                        time.sleep(1)
                        st.rerun()
                    except Exception as e:
                        st.error(str(e))
            
            if col_act2.button("🚀 Deploy", type="primary"):
                if selected_model:
                    try:
                        source = os.path.join(ZOO_DIR, selected_model)
                        shutil.copy(source, ACTIVE_MODEL_PATH)
                        
                        # Copy or create metadata
                        meta_source = source + ".json"
                        if os.path.exists(meta_source):
                            shutil.copy(meta_source, ACTIVE_MODEL_META)
                        else:
                            # Create basic metadata
                            meta = {
                                "filename": selected_model,
                                "deployed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            }
                            with open(ACTIVE_MODEL_META, "w") as f:
                                json.dump(meta, f, indent=2)
                        
                        st.toast(f"🚀 {selected_model} deployed to Live Trading!", icon="🚀")
                        time.sleep(1)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Deploy failed: {e}")

    # --- TAB 3: DEPLOY STATUS ---
    with tab_deploy:
        st.subheader("🚀 Active Deployment")
        
        if os.path.exists(ACTIVE_MODEL_META):
            try:
                with open(ACTIVE_MODEL_META, "r") as f:
                    active_meta = json.load(f)
                
                st.success("🟢 Model is deployed and ready for Live Trading")
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Symbol", active_meta.get("symbol", "N/A"))
                col2.metric("Market", active_meta.get("market_type", "N/A").capitalize())
                col3.metric("AUC", f"{active_meta.get('auc', 0):.4f}")
                
                st.json(active_meta)
                
                if st.button("🔴 Undeploy Model", type="secondary"):
                    try:
                        os.remove(ACTIVE_MODEL_PATH)
                        os.remove(ACTIVE_MODEL_META)
                        st.toast("Model undeployed", icon="🔴")
                        time.sleep(1)
                        st.rerun()
                    except Exception as e:
                        st.error(str(e))
            except Exception as e:
                st.error(f"Error reading deployment info: {e}")
        else:
            st.warning("⚪ No model deployed. Go to Model Zoo tab to deploy one.")
            st.info("💡 Tip: Train a model → Save to Zoo → Deploy to Live Trading")

    # --- TAB 4: UPLOAD ---
    with tab_upload:
        st.subheader("📤 Import External Brain")
        st.markdown("Upload a `.pth` file trained elsewhere.")
        
        upload_symbol = st.text_input("Symbol for this model", value="UNKNOWN")
        upload_market = st.selectbox("Market Type", ["forex", "crypto"])
        
        uploaded_file = st.file_uploader("Choose a .pth file", type="pth")
        
        if uploaded_file is not None:
            save_path = os.path.join(ZOO_DIR, uploaded_file.name)
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Save metadata
            meta = {
                "symbol": upload_symbol,
                "market_type": upload_market,
                "architecture": "uploaded",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "auc": 0.0,
                "loss": 0.0,
                "accuracy": 0.0,
                "epoch": 0,
            }
            with open(save_path + ".json", "w") as f:
                json.dump(meta, f, indent=2)
            
            st.success(f"Saved {uploaded_file.name} to Zoo.")
            time.sleep(1)
            st.rerun()
