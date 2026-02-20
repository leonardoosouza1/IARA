import streamlit as st
import time
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
from src.brain.tools.alpaca_executor import AlpacaExecutor

# Import Engine Components (Mock/Stub for UI if Engine not running in same process)
# In a real deployment, the Dashboard might communicate via DB or API.
# For Phase 1, we simulate the Shared State visualization.

def render():
    st.title("🚀 Trinity Mission Control")
    st.markdown("### Strategic Oversight & Tactical Intervention")
    
    # Top Row: KPI Metrics
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Market Regime", "Volatile", "+2.4%")
    kpi2.metric("LLM Context", "Bearish Divergence", "-0.85")
    kpi3.metric("LNN Confidence", "Neutral", "0.12")
    kpi4.metric("Active Agents", "2/3", "Sentinel Passive")
    
    st.markdown("---")
    
    # Main Control Panel
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        st.subheader("📡 Sentinel Watchdog (News Feed)")
        
        # Simulated News Feed
        news_data = [
            {"time": "10:32:05", "headline": "Fed Chair Powell to speak at 2 PM", "impact": 0.3, "sentiment": -0.2},
            {"time": "10:15:00", "headline": "Tech sector rallies on earnings", "impact": 0.1, "sentiment": 0.6},
            {"time": "09:45:30", "headline": "Oil prices stable amid supply concerns", "impact": 0.05, "sentiment": 0.1},
        ]
        
        # Check for Manual Trigger via Session State
        if 'simulated_trigger' in st.session_state and st.session_state.simulated_trigger:
             news_data.insert(0, {
                 "time": datetime.now().strftime("%H:%M:%S"), 
                 "headline": "🚨 SIMULATED: CENTRAL BANK UNEXPECTED RATE HIKE", 
                 "impact": 0.95, 
                 "sentiment": -0.9
             })
        
        df_news = pd.DataFrame(news_data)
        
        # Display as a styled dataframe or list
        st.dataframe(
            df_news, 
            column_config={
                "impact": st.column_config.ProgressColumn("Impact Score", min_value=0, max_value=1, format="%.2f"),
                "sentiment": st.column_config.NumberColumn("Sentiment", format="%.2f")
            },
            width="stretch",
            hide_index=True
        )
        
    with col_right:
        st.subheader("🛡️ Defense Systems")
        
        # DEFCON Gauge (Sentinel Impact Level)
        current_impact = 0.0
        if 'simulated_trigger' in st.session_state and st.session_state.simulated_trigger:
            current_impact = 0.95
            
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = current_impact * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "THREAT LEVEL (Impact %)"},
            gauge = {
                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "darkred" if current_impact > 0.8 else "green"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 50], 'color': 'rgba(0, 255, 0, 0.3)'},
                    {'range': [50, 80], 'color': 'rgba(255, 255, 0, 0.3)'},
                    {'range': [80, 100], 'color': 'rgba(255, 0, 0, 0.3)'}],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90}}))
        
        st.plotly_chart(fig, width="stretch")
        
        st.markdown("### Manual Override")
        
        # Emergency Buttons
        if st.button("🚨 SIMULATE SENTINEL TRIGGER (TEST)", type="secondary"):
            st.session_state.simulated_trigger = True
            st.rerun()
            
        if st.button("✅ CLEAR ALARM", type="primary"):
            st.session_state.simulated_trigger = False
            st.rerun()
            
        if st.button("🔴 EMERGENCY FREEZE (HARD STOP)", type="primary"):
            st.error("DANGER: SYSTEM HALTED MANUALLY")
            # Logic to write to DB/File to stop engine would go here
            executor = AlpacaExecutor()
            executor.close_all_positions()

    # --- ALPACA LIVE MONITOR ---
    st.markdown("---")
    st.subheader("🦙 Alpaca Paper Trading Monitor")
    
    executor = AlpacaExecutor()
    if not executor.api_key or not executor.secret_key:
        st.warning("⚠️ Alpaca Credentials missing in .env. Integration disabled.")
    else:
        acc_col, trade_col = st.columns([1, 2])
        
        with acc_col:
            st.markdown("#### Account Summary")
            summary = executor.get_account_summary()
            if summary:
                st.metric("Total Equity", f"${float(summary.get('equity', 0)):,.2f}")
                st.metric("Buying Power", f"${float(summary.get('buying_power', 0)):,.2f}")
                st.metric("Unrealized P/L", f"${float(summary.get('unrealized_pl', 0)):,.2f}")
                st.write(f"**Currency:** {summary.get('currency')}")
                st.write(f"**Environment:** {'Paper' if executor.paper else 'Live'}")
            else:
                st.error("Failed to fetch Alpaca summary.")

        with trade_col:
            st.markdown("#### Open Positions")
            positions = executor.get_open_positions()
            if positions:
                st.dataframe(pd.DataFrame(positions), use_container_width=True, hide_index=True)
            else:
                st.info("No active positions on Alpaca.")
