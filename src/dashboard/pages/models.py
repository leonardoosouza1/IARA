import streamlit as st
import os
import json
import shutil
import time
from datetime import datetime
import pandas as pd

# Constants
LOG_DIR = "logs"
MODELS_DIR = os.path.join(LOG_DIR, "models")
ZOO_DIR = os.path.join(LOG_DIR, "model_zoo")
LEADERBOARD_FILE = os.path.join(LOG_DIR, "leaderboard.json")
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_model.pth")

# Ensure Zoo Exists
os.makedirs(ZOO_DIR, exist_ok=True)

def render():
    st.title("🗄️ Model Manager")
    st.info("Manage your trained Brains. Save the champions, discard the failures.")

    tab_current, tab_zoo, tab_upload = st.tabs(["🔥 Current Training", "🦁 Model Zoo", "📤 Upload"])

    # --- TAB 1: CURRENT TRAINING ---
    with tab_current:
        st.subheader("Active Session Artifacts")
        
        # Check Leaderboard
        if os.path.exists(LEADERBOARD_FILE):
             try:
                with open(LEADERBOARD_FILE, "r") as f:
                    lb_data = json.load(f)
                
                if lb_data:
                    best = lb_data[0]
                    st.success(f"🏆 Current Champion found from Epoch {best['epoch']}")
                    
                    # Metrics Display
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("AUC", f"{best.get('auc', 0):.4f}")
                    col2.metric("Loss", f"{best['loss']:.4f}")
                    col3.metric("Accuracy", f"{best['accuracy']:.2%}")
                    col4.metric("Date", best['timestamp'])
                    
                    # Save Actions
                    st.markdown("### 💾 Save to Zoo")
                    
                    # Smart Naming
                    auc_str = f"{best.get('auc', 0):.4f}".replace(".", "")
                    loss_str = f"{best['loss']:.4f}".replace(".", "")
                    epoch_str = best['epoch']
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
                    default_name = f"trinity_v8_ep{epoch_str}_auc{auc_str}_loss{loss_str}_{timestamp}.pth"
                    
                    save_name = st.text_input("Filename", value=default_name)
                    
                    if st.button("💾 Save Champion to Zoo", type="primary"):
                        if os.path.exists(BEST_MODEL_PATH):
                            dest = os.path.join(ZOO_DIR, save_name)
                            shutil.copy(BEST_MODEL_PATH, dest)
                            
                            # Also save metadata
                            meta_path = dest + ".json"
                            with open(meta_path, "w") as f:
                                json.dump(best, f, indent=2)
                                
                            st.toast(f"Saved to {save_name}", icon="🦁")
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error("best_model.pth not found! Did training finish saving?")
                    
                else:
                    st.warning("Leaderboard is empty.")
             except Exception as e:
                 st.error(f"Error reading leaderboard: {e}")
        else:
            st.info("No active training data found.")

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
                if os.path.exists(meta_path):
                    try:
                        with open(meta_path, "r") as f:
                            meta = json.load(f)
                            auc = meta.get("auc", 0)
                            loss = meta.get("loss", 0)
                            acc = meta.get("accuracy", 0)
                            epoch = meta.get("epoch", 0)
                    except: pass
                
                zoo_data.append({
                    "Filename": m,
                    "AUC": auc,
                    "Loss": loss,
                    "Accuracy": acc,
                    "Epoch": epoch,
                    "Size (MB)": f"{size_mb:.2f}",
                    "Created": created,
                })
            
            df = pd.DataFrame(zoo_data)
            
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
                use_container_width=True,
                hide_index=True
            )
            
            # Selection for Actions
            st.divider()
            col_sel, col_act = st.columns([3, 1])
            selected_model = col_sel.selectbox("Select Model to Manage", df["Filename"])
            
            if col_act.button("🗑️ Delete Selected", type="secondary"):
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

    # --- TAB 3: UPLOAD ---
    with tab_upload:
        st.subheader("📤 Import External Brain")
        st.markdown("Upload a `.pth` file trained elsewhere.")
        uploaded_file = st.file_uploader("Choose a .pth file", type="pth")
        
        if uploaded_file is not None:
            save_path = os.path.join(ZOO_DIR, uploaded_file.name)
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success(f"Saved {uploaded_file.name} to Zoo.")
            time.sleep(1)
            st.rerun()
