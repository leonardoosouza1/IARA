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

# ── Professional Dark Theme CSS ──────────────────────────────
st.markdown("""
<style>
    /* Dark pro background */
    .stApp {
        background-color: #0e1117;
    }
    /* Metric cards */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #1a1f2e 0%, #141824 100%);
        border: 1px solid rgba(128, 128, 128, 0.15);
        border-radius: 12px;
        padding: 12px 16px;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.1rem;
        font-weight: 700;
    }
    [data-testid="stMetricDelta"] > div {
        font-size: 0.85rem;
    }
    /* Dataframe styling */
    .stDataFrame {
        border-radius: 8px;
        overflow: hidden;
    }
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0e1117 0%, #161b22 100%);
        border-right: 1px solid rgba(128, 128, 128, 0.1);
    }
    /* Plotly chart containers */
    .stPlotlyChart {
        border-radius: 8px;
        border: 1px solid rgba(128, 128, 128, 0.1);
    }
    /* Tabs */
    .stTabs [data-baseweb="tab"] {
        font-weight: 600;
    }
    /* Dividers */
    hr {
        border-color: rgba(128, 128, 128, 0.1) !important;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar Navigation
st.sidebar.title("🧬 I.A.R.A. v8.5")
st.sidebar.markdown("---")

# Navigation
page = st.sidebar.radio("Navigation", ["🏢 Trading Desk", "Mission Control", "Training Lab", "Model Manager", "Simulation Lab"])

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
if page == "🏢 Trading Desk":
    from src.dashboard.pages import live_trading
    importlib.reload(live_trading)
    live_trading.render()
elif page == "Mission Control":
    from src.dashboard.pages import mission_control
    importlib.reload(mission_control)
    mission_control.render()
elif page == "Training Lab":
    from src.dashboard.pages import training
    importlib.reload(training)
    training.render()
elif page == "Model Manager":
    from src.dashboard.pages import models
    importlib.reload(models)
    models.render()
elif page == "Simulation Lab":
    from src.dashboard.pages import simulation
    try:
        importlib.reload(simulation)
        simulation.render()
    except ImportError:
        st.warning("Simulation Lab module not found yet. Please wait for installation.")
    except Exception as e:
         st.error(f"Sim Lab Error: {e}")
