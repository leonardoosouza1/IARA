import streamlit as st
import os
import sys
import importlib

# Ensure src is in path for imports
sys.path.append(os.path.abspath(os.curdir))

st.set_page_config(
    page_title="I.A.R.A. Mission Control",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar Navigation
st.sidebar.title("🧬 I.A.R.A. v8.5")
st.sidebar.markdown("---")

# Navigation
page = st.sidebar.radio("Navigation", ["Mission Control", "Training Lab"])

st.sidebar.markdown("---")
st.sidebar.info(
    """
    **System Status:**
    - **Engine:** Trinity v8.5
    - **Mode:** Dual Core (Phase 1)
    - **Device:** ROCm (RX 6750 XT)
    """
)

# Router
if page == "Mission Control":
    from src.dashboard.pages import mission_control
    importlib.reload(mission_control)
    mission_control.render()
elif page == "Training Lab":
    from src.dashboard.pages import training
    importlib.reload(training)
    training.render()
