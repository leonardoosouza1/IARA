import os
import psycopg2
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Union

from sqlalchemy import create_engine, text

class MarketScanner:
    """
    Tool Set for the Strategic Agent (LFM).
    Allows the Slow System to query historical context from TimescaleDB/Postgres.
    """
    def __init__(self, db_uri: Optional[str] = None):
        self.dsn = db_uri or os.getenv("DB_URI", "postgresql://saci:saci_secret@localhost:5432/saci_core")
        self.engine = create_engine(self.dsn)
        
    def _get_conn(self):
        return self.engine.connect()

    def check_support_resistance(self, symbol: str, current_price: float, lookback_days: int = 30) -> str:
        """
        Scans for price levels that have been tested multiple times (High Volume Nodes or Rejections).
        """
        try:
            with self._get_conn() as conn:
                # 1. Fetch simplified PA (Highs and Lows)
                query = """
                    SELECT high, low 
                    FROM market_candles 
                    WHERE symbol = %s 
                      AND time > NOW() - INTERVAL '%s days'
                """
                df = pd.read_sql(query, conn, params=(symbol, lookback_days))
                
            if df.empty:
                return f"No historical data for {symbol} in last {lookback_days} days."

            # 2. Logic: Find clusters of lows (Support) and highs (Resistance)
            # Tolerance: 0.1% of price
            tolerance = current_price * 0.001
            
            # Supports
            lows = df['low'].values
            nearby_supports = []
            # Naive cluster: Check if 'low' is within tolerance of other lows
            # Optimization: Histogram
            hist_l, bins_l = np.histogram(lows, bins=50)
            # Find bins with high count
            peak_idxs_l = np.where(hist_l > len(lows)*0.05)[0] # Peaks with >5% of data
            
            support_msg = "None"
            for p in peak_idxs_l:
                level = bins_l[p]
                if level < current_price: # Only supports below
                     count = hist_l[p]
                     if abs(current_price - level) / current_price < 0.05: # Relevant (near 5%)
                        nearby_supports.append(f"{level:.5f} ({count} tests)")
            
            if nearby_supports:
                support_msg = ", ".join(nearby_supports)

            # Resistances
            highs = df['high'].values
            nearby_resistances = []
            hist_h, bins_h = np.histogram(highs, bins=50)
            peak_idxs_h = np.where(hist_h > len(highs)*0.05)[0]
            
            res_msg = "None"
            for p in peak_idxs_h:
                level = bins_h[p]
                if level > current_price: # Only resistances above
                    count = hist_h[p]
                    if abs(current_price - level) / current_price < 0.05:
                        nearby_resistances.append(f"{level:.5f} ({count} tests)")
                        
            if nearby_resistances:
                res_msg = ", ".join(nearby_resistances)

            return f"Market Scan ({symbol}):\n- Supports nearby: {support_msg}\n- Resistances nearby: {res_msg}"

        except Exception as e:
            return f"Error scanning S/R for {symbol}: {e}"

    def get_trend_alignment(self, symbol: str, timeframe: str = '1h', lookback: int = 50) -> str:
        """
        Verifies trend across the specified timeframe.
        Returns: "BULLISH", "BEARISH", or "RANGING" with slope details.
        """
        try:
            with self._get_conn() as conn:
                query = """
                    SELECT close 
                    FROM market_candles 
                    WHERE symbol = %s AND interval = %s 
                    ORDER BY time DESC 
                    LIMIT %s
                """
                df = pd.read_sql(query, conn, params=(symbol, timeframe, lookback))
            
            if len(df) < 10:
                return f"Insufficient data for {timeframe} trend analysis."
                
            closes = df['close'].values[::-1] # Reverse to chronological
            
            # Simple Linear Regression for slope
            x = np.arange(len(closes))
            slope, intercept = np.polyfit(x, closes, 1)
            
            # R-squared integration for quality?
            # For text output, slope is enough.
            
            trend = "RANGING"
            if slope > 0.0001: trend = "BULLISH"
            elif slope < -0.0001: trend = "BEARISH"
            
            return f"Trend ({timeframe}): {trend} (Slope: {slope:.6f}, Sample: {len(closes)} candles)"

        except Exception as e:
            return f"Error trend analysis: {e}"
