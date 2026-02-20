"""
Alpaca Data Service - Centralized data layer for the Trading Desk.
Fetches bars, orders, positions, and account data from Alpaca APIs.
"""
import os
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import pandas as pd
from dotenv import load_dotenv

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOrdersRequest
from alpaca.trading.enums import QueryOrderStatus
from alpaca.data.historical import CryptoHistoricalDataClient, StockHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest, StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.common.exceptions import APIError


load_dotenv()
logger = logging.getLogger("AlpacaDataService")


class AlpacaDataService:
    """
    Unified data service for Alpaca Markets.
    Provides bars, orders, positions, and account info for the Trading Desk.
    """

    def __init__(self):
        self.api_key = os.getenv("ALPACA_API_KEY")
        self.secret_key = os.getenv("ALPACA_SECRET_KEY")
        self.paper = os.getenv("ALPACA_PAPER", "true").lower() == "true"
        self.symbol = os.getenv("ALPACA_SYMBOL", "BTC/USD")

        self._trading_client = None
        self._crypto_data_client = None
        self._stock_data_client = None

        if self.api_key and self.secret_key:
            self._trading_client = TradingClient(self.api_key, self.secret_key, paper=self.paper)
            self._crypto_data_client = CryptoHistoricalDataClient()
            self._stock_data_client = StockHistoricalDataClient(self.api_key, self.secret_key)

    @property
    def is_configured(self) -> bool:
        return self._trading_client is not None

    def is_crypto(self, symbol: str = None) -> bool:
        s = symbol or self.symbol
        # Enhanced check: slashed symbols or known crypto assets
        return "/" in s or s in ["BTC", "ETH", "SOL", "LTC", "DOGE"]

    def get_available_assets(self, asset_class: str = "crypto") -> List[str]:
        """Fetches list of active tradable assets for the dropdown."""
        if not self.is_configured:
            return ["BTC/USD", "ETH/USD", "AAPL", "TSLA"]
        try:
            from alpaca.trading.requests import GetAssetsRequest
            from alpaca.trading.enums import AssetStatus, AssetClass

            req = GetAssetsRequest(
                status=AssetStatus.ACTIVE,
                asset_class=AssetClass.CRYPTO if asset_class == "crypto" else AssetClass.US_EQUITY
            )
            assets = self._trading_client.get_all_assets(req)
            return [a.symbol for a in assets if a.tradable]
        except Exception as e:
            logger.error(f"Assets fetch error: {e}")
            return ["BTC/USD", "ETH/USD"]

    # ─── Account ───────────────────────────────────────────────
    def get_account(self) -> Dict[str, Any]:
        if not self.is_configured:
            return {}
        try:
            acc = self._trading_client.get_account()
            # Safely access attributes available in recent SDK versions
            return {
                "equity": float(acc.equity) if hasattr(acc, "equity") else 0.0,
                "cash": float(acc.cash) if hasattr(acc, "cash") else 0.0,
                "buying_power": float(acc.buying_power) if hasattr(acc, "buying_power") else 0.0,
                "portfolio_value": float(acc.portfolio_value) if hasattr(acc, "portfolio_value") else 0.0,
                # Start with 0 if not available; calculation can be improved later
                "unrealized_pl": 0.0, 
                "unrealized_plpc": 0.0,
                "currency": getattr(acc, "currency", "USD"),
                "status": getattr(acc, "status", "UNKNOWN"),
            }
        except Exception as e:
            logger.error(f"Account fetch error: {e}")
            return {}

    # ─── Positions ─────────────────────────────────────────────
    def get_positions(self) -> List[Dict[str, Any]]:
        if not self.is_configured:
            return []
        try:
            positions = self._trading_client.get_all_positions()
            return [
                {
                    "symbol": p.symbol,
                    "side": p.side.value if hasattr(p.side, "value") else str(p.side),
                    "qty": float(p.qty),
                    "avg_entry": float(p.avg_entry_price),
                    "current_price": float(p.current_price) if p.current_price else 0,
                    "market_value": float(p.market_value) if p.market_value else 0,
                    "unrealized_pl": float(p.unrealized_pl) if p.unrealized_pl else 0,
                    "unrealized_plpc": float(p.unrealized_plpc) if p.unrealized_plpc else 0,
                    "change_today": float(p.change_today) if p.change_today else 0,
                }
                for p in positions
            ]
        except Exception as e:
            logger.error(f"Positions fetch error: {e}")
            return []

    # ─── Orders ────────────────────────────────────────────────
    def get_orders(self, status: str = "all", limit: int = 50) -> pd.DataFrame:
        if not self.is_configured:
            return pd.DataFrame()
        try:
            status_map = {
                "all": QueryOrderStatus.ALL,
                "open": QueryOrderStatus.OPEN,
                "closed": QueryOrderStatus.CLOSED,
            }
            request = GetOrdersRequest(
                status=status_map.get(status, QueryOrderStatus.ALL),
                limit=limit,
                nested=False  # Simple list
            )
            orders = self._trading_client.get_orders(filter=request)
            if not orders:
                return pd.DataFrame()

            rows = []
            for o in orders:
                filled_price = float(o.filled_avg_price) if o.filled_avg_price else None
                # Handle enums safely
                side = o.side.value if hasattr(o.side, "value") else str(o.side)
                status_val = o.status.value if hasattr(o.status, "value") else str(o.status)
                type_val = o.type.value if hasattr(o.type, "value") else str(o.type)
                
                rows.append({
                    "id": str(o.id)[:8],
                    "time": o.submitted_at.strftime("%Y-%m-%d %H:%M") if o.submitted_at else "",
                    "symbol": o.symbol,
                    "side": side,
                    "qty": float(o.qty) if o.qty else float(o.notional) if o.notional else 0,
                    "type": type_val,
                    "status": status_val,
                    "filled_price": filled_price,
                    "filled_at": o.filled_at.strftime("%Y-%m-%d %H:%M") if o.filled_at else "",
                })
            return pd.DataFrame(rows)
        except Exception as e:
            logger.error(f"Orders fetch error: {e}")
            return pd.DataFrame()

    # ─── Bars (Candlestick Data) ───────────────────────────────
    def get_bars(self, symbol: str = None, timeframe: str = "1Min",
                 limit: int = 200, days_back: int = 1) -> pd.DataFrame:
        symbol = symbol or self.symbol
        if not self.is_configured:
            return pd.DataFrame()

        tf_map = {
            "1Min": TimeFrame.Minute,
            "5Min": TimeFrame(5, TimeFrameUnit.Minute),
            "15Min": TimeFrame(15, TimeFrameUnit.Minute),
            "1Hour": TimeFrame.Hour,
            "1Day": TimeFrame.Day,
        }
        tf = tf_map.get(timeframe, TimeFrame.Minute)
        
        # Ensure we look back enough to find data
        lookback_days = max(days_back, 7) 
        start = datetime.now() - timedelta(days=lookback_days)
        
        # Fetch a reasonable multiple of requested bars to ensure coverage
        fetch_limit = min(limit * 3, 2000)

        try:
            if self.is_crypto(symbol):
                request = CryptoBarsRequest(
                    symbol_or_symbols=[symbol],
                    timeframe=tf,
                    start=start,
                    limit=fetch_limit,
                )
                bars = self._crypto_data_client.get_crypto_bars(request)
            else:
                request = StockBarsRequest(
                    symbol_or_symbols=[symbol],
                    timeframe=tf,
                    start=start,
                    limit=fetch_limit,
                )
                bars = self._stock_data_client.get_stock_bars(request)

            df = bars.df
            if df.empty:
                return pd.DataFrame()

            # Flatten multi-index if present
            if isinstance(df.index, pd.MultiIndex):
                df = df.reset_index()
            else:
                df = df.reset_index()

            # Normalize column names
            df.columns = [c.lower() for c in df.columns]
            
            # Ensure timestamp is 'time'
            if "timestamp" in df.columns:
                df = df.rename(columns={"timestamp": "time"})
                
            # Filter for specific symbol
            if "symbol" in df.columns:
                df = df[df["symbol"] == symbol]

            # Return the requested number of LATEST bars
            return df.tail(limit)

        except Exception as e:
            logger.error(f"Bars fetch error: {e}")
            return pd.DataFrame()

    # ─── Portfolio History (Equity Curve) ──────────────────────
    def get_portfolio_history(self, period: str = "1D", timeframe: str = "5Min") -> pd.DataFrame:
        if not self.is_configured:
            return pd.DataFrame()
        try:
            # SDK's get_portfolio_history might not accept args in this version
            # We try without args first as seen in debug
            history = self._trading_client.get_portfolio_history()
            
            if hasattr(history, "timestamp") and hasattr(history, "equity"):
                # Timestamps might be unix ints or datetime objects
                ts_data = history.timestamp
                if ts_data and isinstance(ts_data[0], int):
                    ts_data = [datetime.fromtimestamp(t) for t in ts_data]
                
                df = pd.DataFrame({
                    "time": ts_data,
                    "equity": history.equity,
                    "profit_loss": history.profit_loss,
                    "profit_loss_pct": history.profit_loss_pct,
                })
                # Basic filtering if we want to simulate 'period'
                # (Optional: Filter df by date here if needed)
                return df
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Portfolio history error: {e}")
            return pd.DataFrame()
