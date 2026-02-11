"""
🏢 TRADING DESK - Professional Live Trading Monitor
====================================================
Real-time candlestick chart with order markers, equity curve,
order history, and AI brain panel. Powered by Alpaca Markets.
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import time
import os

# Service
from src.dashboard.services.alpaca_data_service import AlpacaDataService


def render():
    # ── Init Service ──────────────────────────────────────────
    service = AlpacaDataService()

    if not service.is_configured:
        st.error("⚠️ Alpaca não configurado. Preencha `ALPACA_API_KEY` e `ALPACA_SECRET_KEY` no `.env`.")
        st.stop()

    # ── Sidebar Controls ──────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Trading Desk Config")
        
        # Asset Selection
        asset_type = st.radio("Asset Class", ["Crypto", "Stocks"], horizontal=True)
        available_assets = service.get_available_assets("crypto" if asset_type == "Crypto" else "us_equity")
        
        # Default index logic
        default_idx = 0
        target = service.symbol if service.symbol in available_assets else ("BTC/USD" if asset_type=="Crypto" else "AAPL")
        if target in available_assets:
            default_idx = available_assets.index(target)
            
        symbol = st.selectbox("Symbol", available_assets, index=default_idx)
        timeframe = st.selectbox("Timeframe", ["1Min", "5Min", "15Min", "1Hour", "1Day"], index=1)
        chart_limit = st.slider("Candles", 50, 500, 200)
        days_back = st.slider("Days Back", 1, 30, 3) # Default increased to 3
        auto_refresh = st.toggle("Auto-Refresh (3s)", value=True)
        order_filter = st.selectbox("Orders Filter", ["all", "closed", "open"])

    # ── Fetch All Data ────────────────────────────────────────
    # FORCE PRINT DEBUGGING TO DOCKER LOGS
    print(f"\n--- [LIVE TRADING REFRESH] {datetime.now()} ---")
    print(f"Requesting: {symbol} {timeframe} Limit={chart_limit}")

    account = service.get_account()
    positions = service.get_positions()
    
    try:
        df_bars = service.get_bars(symbol=symbol, timeframe=timeframe, limit=chart_limit, days_back=days_back)
        print(f"Received DF Shape: {df_bars.shape}")
        if not df_bars.empty:
            print("Head:\n", df_bars.head(2))
            print("Tail:\n", df_bars.tail(2))
        else:
            print("!!! DF IS EMPTY !!!")
            
    except Exception as e:
        print(f"!!! EXCEPTION in live_trading.get_bars: {e}")
        df_bars = pd.DataFrame()

    df_orders = service.get_orders(status=order_filter, limit=100)
    df_equity = service.get_portfolio_history(period="1D", timeframe="5Min")
    
    # ── Debug Expander for User ──────────────────────────────
    with st.expander("🔍 System Internals (Debug)", expanded=True):
        st.write(f"**Symbol**: `{symbol}`")
        if df_bars.empty:
            st.error("❌ Dataframe is EMPTY. Check Docker Logs.")
        else:
            st.success(f"✅ Loaded {len(df_bars)} bars.")
            st.dataframe(df_bars.tail(5))

    # ══════════════════════════════════════════════════════════
    # ██  HEADER: TITLE + STATUS BADGE
    # ══════════════════════════════════════════════════════════
    header_left, header_right = st.columns([3, 1])
    with header_left:
        st.markdown(f"## 🏢 Trading Desk — `{symbol}`")
    with header_right:
        env_label = "🟢 PAPER" if service.paper else "🔴 LIVE"
        status = account.get("status", "UNKNOWN")
        st.markdown(f"### {env_label} &nbsp; | &nbsp; `{status}`")

    # ══════════════════════════════════════════════════════════
    # ██  KPI BAR
    # ══════════════════════════════════════════════════════════
    k1, k2, k3, k4, k5, k6 = st.columns(6)

    equity = account.get("equity", 0)
    cash = account.get("cash", 0)
    buying_power = account.get("buying_power", 0)
    unrealized_pl = account.get("unrealized_pl", 0)
    unrealized_plpc = account.get("unrealized_plpc", 0)

    # Win rate from closed orders
    if not df_orders.empty and "status" in df_orders.columns:
        closed = df_orders[df_orders["status"] == "filled"]
        total_trades = len(closed)
    else:
        total_trades = 0

    active_positions = len(positions)

    k1.metric("💰 Equity", f"${equity:,.2f}")
    k2.metric("💵 Cash", f"${cash:,.2f}")
    k3.metric("🔋 Buying Power", f"${buying_power:,.2f}")
    k4.metric("📈 Unrealized P/L", f"${unrealized_pl:+,.2f}",
              delta=f"{unrealized_plpc * 100:+.2f}%" if unrealized_plpc else None)
    k5.metric("📊 Trades (Filled)", total_trades)
    k6.metric("📍 Open Positions", active_positions)

    st.markdown("---")

    # ══════════════════════════════════════════════════════════
    # ██  MAIN LAYOUT: CHART + AI BRAIN
    # ══════════════════════════════════════════════════════════
    col_chart, col_brain = st.columns([3, 1])

    # ── CANDLESTICK CHART ─────────────────────────────────────
    with col_chart:
        # Calculate Real-Time Stats
        current_price = 0.0
        variation = 0.0
        if not df_bars.empty:
            latest = df_bars.iloc[-1]
            current_price = latest["close"]
            if len(df_bars) > 1:
                prev = df_bars.iloc[-2]["close"]
                if prev > 0:
                    variation = ((current_price - prev) / prev) * 100
        
        # Display Price Header (if data exists)
        if current_price > 0:
            color = "#00CC96" if variation >= 0 else "#FF4B4B"
            st.markdown(f"### {symbol} <span style='color:{color}'>${current_price:,.2f} ({variation:+.2f}%)</span>", unsafe_allow_html=True)

        if df_bars.empty:
            st.warning(f"📊 No bar data available for {symbol} ({timeframe}). Try increasing 'Days Back'.")
            if service.paper:
                st.info("💡 Note: Paper Trading data for Stocks may be delayed by 15min. Crypto is usually real-time.")
        else:
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.03,
                row_heights=[0.8, 0.2],
                subplot_titles=("", "Volume"),
            )

            # Candlestick
            fig.add_trace(
                go.Candlestick(
                    x=df_bars["time"],
                    open=df_bars["open"],
                    high=df_bars["high"],
                    low=df_bars["low"],
                    close=df_bars["close"],
                    name="Market",
                    increasing_line_color="#00CC96",
                    decreasing_line_color="#FF4B4B",
                ),
                row=1, col=1,
            )

            # Volume bars
            if "volume" in df_bars.columns:
                colors = ["#00CC96" if c >= o else "#FF4B4B"
                          for c, o in zip(df_bars["close"], df_bars["open"])]
                fig.add_trace(
                    go.Bar(
                        x=df_bars["time"],
                        y=df_bars["volume"],
                        marker_color=colors,
                        opacity=0.5,
                        name="Volume",
                        showlegend=False,
                    ),
                    row=2, col=1,
                )

            # ── Order Markers on Chart ────────────────────────
            if not df_orders.empty and "filled_price" in df_orders.columns:
                filled = df_orders[
                    (df_orders["status"] == "filled") &
                    (df_orders["symbol"].str.replace("/", "") == symbol.replace("/", ""))
                ].copy()

                if not filled.empty and "filled_at" in filled.columns:
                    buys = filled[filled["side"] == "buy"]
                    sells = filled[filled["side"] == "sell"]

                    if not buys.empty:
                        fig.add_trace(
                            go.Scatter(
                                x=pd.to_datetime(buys["filled_at"]),
                                y=buys["filled_price"],
                                mode="markers",
                                marker=dict(
                                    symbol="triangle-up",
                                    size=14,
                                    color="#00CC96",
                                    line=dict(width=2, color="white"),
                                ),
                                name="BUY",
                                hovertemplate="BUY @ %{y:.2f}<extra></extra>",
                            ),
                            row=1, col=1,
                        )

                    if not sells.empty:
                        fig.add_trace(
                            go.Scatter(
                                x=pd.to_datetime(sells["filled_at"]),
                                y=sells["filled_price"],
                                mode="markers",
                                marker=dict(
                                    symbol="triangle-down",
                                    size=14,
                                    color="#FF4B4B",
                                    line=dict(width=2, color="white"),
                                ),
                                name="SELL",
                                hovertemplate="SELL @ %{y:.2f}<extra></extra>",
                            ),
                            row=1, col=1,
                        )

            # Chart Layout
            fig.update_layout(
                template="plotly_dark",
                height=550,
                margin=dict(l=0, r=0, t=10, b=0),
                xaxis_rangeslider_visible=False,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1,
                    font=dict(size=10),
                ),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            fig.update_xaxes(showgrid=True, gridcolor="rgba(128,128,128,0.1)")
            fig.update_yaxes(showgrid=True, gridcolor="rgba(128,128,128,0.1)")

            st.plotly_chart(fig, use_container_width=True, key=f"main_chart_{time.time()}")

    # ── AI BRAIN PANEL ────────────────────────────────────────
    with col_brain:
        st.markdown("### 🧠 AI Brain")

        # Signal Log (from recent orders as proxy)
        st.markdown("#### 📡 Signal Log")
        if not df_orders.empty:
            recent = df_orders.head(8)
            for _, row in recent.iterrows():
                side_icon = "🟢" if row.get("side") == "buy" else "🔴"
                status_icon = "✅" if row.get("status") == "filled" else "⏳"
                price_str = f"@ ${row['filled_price']:.2f}" if row.get("filled_price") else ""
                st.markdown(
                    f"`{row.get('time', '')}` {side_icon} **{row.get('side', '').upper()}** "
                    f"{row.get('symbol', '')} {price_str} {status_icon}"
                )
        else:
            st.info("No signals dispatched yet.")

        st.markdown("---")

        # Active Positions
        st.markdown("#### 📍 Active Positions")
        if positions:
            for p in positions:
                pl = p.get("unrealized_pl", 0)
                pl_icon = "🟢" if pl >= 0 else "🔴"
                st.markdown(
                    f"{pl_icon} **{p['symbol']}** — {p['qty']} units\n"
                    f"Entry: `${p['avg_entry']:.2f}` → Now: `${p['current_price']:.2f}`\n"
                    f"P/L: **${pl:+.2f}**"
                )
                st.markdown("---")
        else:
            st.caption("No open positions.")

        st.markdown("---")

        # Sentinel Status
        st.markdown("#### 🛡️ Sentinel")
        sentinel_file = "logs/sentinel_status.json"
        if os.path.exists(sentinel_file):
            import json
            with open(sentinel_file) as f:
                sentinel = json.load(f)
            threat = sentinel.get("threat_level", 0)
            st.progress(min(threat, 1.0), text=f"Threat: {threat * 100:.0f}%")
        else:
            st.progress(0.0, text="Threat: 0% (Sentinel Idle)")

    # ══════════════════════════════════════════════════════════
    # ██  BOTTOM SECTION: ORDER HISTORY + EQUITY CURVE
    # ══════════════════════════════════════════════════════════
    st.markdown("---")
    bottom_left, bottom_right = st.columns([3, 2])

    # ── Order History Table ───────────────────────────────────
    with bottom_left:
        st.markdown("### 📋 Order History")
        if df_orders.empty:
            st.info("No orders found.")
        else:
            styled_orders = df_orders.copy()

            # Color-code status
            st.dataframe(
                styled_orders,
                column_config={
                    "id": st.column_config.TextColumn("ID", width="small"),
                    "time": st.column_config.TextColumn("Submitted", width="medium"),
                    "symbol": st.column_config.TextColumn("Symbol", width="small"),
                    "side": st.column_config.TextColumn("Side", width="small"),
                    "qty": st.column_config.NumberColumn("Qty", format="%.4f"),
                    "type": st.column_config.TextColumn("Type", width="small"),
                    "status": st.column_config.TextColumn("Status", width="small"),
                    "filled_price": st.column_config.NumberColumn("Fill Price", format="$%.2f"),
                    "filled_at": st.column_config.TextColumn("Filled At", width="medium"),
                },
                use_container_width=True,
                hide_index=True,
                height=300,
            )

    # ── Equity Curve ──────────────────────────────────────────
    with bottom_right:
        st.markdown("### 💰 Equity Curve")
        if df_equity.empty:
            st.info("Portfolio history not available yet.")
        else:
            fig_eq = go.Figure()
            fig_eq.add_trace(
                go.Scatter(
                    x=df_equity["time"],
                    y=df_equity["equity"],
                    mode="lines",
                    name="Equity",
                    line=dict(color="#00CC96", width=2),
                    fill="tozeroy",
                    fillcolor="rgba(0, 204, 150, 0.1)",
                )
            )

            # P/L coloring
            if "profit_loss" in df_equity.columns:
                fig_eq.add_trace(
                    go.Bar(
                        x=df_equity["time"],
                        y=df_equity["profit_loss"],
                        name="P/L",
                        marker_color=[
                            "#00CC96" if v >= 0 else "#FF4B4B"
                            for v in df_equity["profit_loss"]
                        ],
                        opacity=0.3,
                    )
                )

            fig_eq.update_layout(
                template="plotly_dark",
                height=300,
                margin=dict(l=0, r=0, t=10, b=0),
                showlegend=False,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            fig_eq.update_xaxes(showgrid=False)
            fig_eq.update_yaxes(showgrid=True, gridcolor="rgba(128,128,128,0.1)")
            st.plotly_chart(fig_eq, use_container_width=True, key=f"equity_{time.time()}")

    # ── Auto-Refresh Loop ─────────────────────────────────────
    if auto_refresh:
        time.sleep(3)
        st.rerun()


if __name__ == "__main__":
    render()
