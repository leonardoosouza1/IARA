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
from alpaca.data.timeframe import TimeFrame
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
        return "/" in s

    # ─── Account ───────────────────────────────────────────────
    def get_account(self) -> Dict[str, Any]:
        if not self.is_configured:
            return {}
        try:
            acc = self._trading_client.get_account()
            return {
                "equity": float(acc.equity),
                "cash": float(acc.cash),
                "buying_power": float(acc.buying_power),
                "portfolio_value": float(acc.portfolio_value) if acc.portfolio_value else 0.0,
                "unrealized_pl": float(acc.unrealized_intraday_pl) if acc.unrealized_intraday_pl else 0.0,
                "unrealized_plpc": float(acc.unrealized_intraday_plpc) if acc.unrealized_intraday_plpc else 0.0,
                "currency": acc.currency,
                "status": acc.status.value if acc.status else "UNKNOWN",
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
                    "side": p.side.value if p.side else "long",
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
            )
            orders = self._trading_client.get_orders(filter=request)
            if not orders:
                return pd.DataFrame()

            rows = []
            for o in orders:
                filled_price = float(o.filled_avg_price) if o.filled_avg_price else None
                rows.append({
                    "id": str(o.id)[:8],
                    "time": o.submitted_at.strftime("%Y-%m-%d %H:%M") if o.submitted_at else "",
                    "symbol": o.symbol,
                    "side": o.side.value if o.side else "",
                    "qty": float(o.qty) if o.qty else float(o.notional) if o.notional else 0,
                    "type": o.type.value if o.type else "",
                    "status": o.status.value if o.status else "",
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
        """
        Fetches recent bar data for charting.
        timeframe: '1Min', '5Min', '15Min', '1Hour', '1Day'
        """
        symbol = symbol or self.symbol
        if not self.is_configured:
            return pd.DataFrame()

        tf_map = {
            "1Min": TimeFrame.Minute,
            "5Min": TimeFrame(5, "Min"),
            "15Min": TimeFrame(15, "Min"),
            "1Hour": TimeFrame.Hour,
            "1Day": TimeFrame.Day,
        }
        tf = tf_map.get(timeframe, TimeFrame.Minute)
        start = datetime.now() - timedelta(days=days_back)

        try:
            if self.is_crypto(symbol):
                request = CryptoBarsRequest(
                    symbol_or_symbols=symbol,
                    timeframe=tf,
                    start=start,
                    limit=limit,
                )
                bars = self._crypto_data_client.get_crypto_bars(request)
            else:
                request = StockBarsRequest(
                    symbol_or_symbols=symbol,
                    timeframe=tf,
                    start=start,
                    limit=limit,
                )
                bars = self._stock_data_client.get_stock_bars(request)

            df = bars.df
            if df.empty:
                return pd.DataFrame()

            # Flatten multi-index (symbol, timestamp) to single index
            if isinstance(df.index, pd.MultiIndex):
                df = df.reset_index(level=0, drop=True)

            df = df.reset_index()
            df.columns = [c.lower() for c in df.columns]
            df = df.rename(columns={"timestamp": "time"})
            return df.tail(limit)

        except Exception as e:
            logger.error(f"Bars fetch error: {e}")
            return pd.DataFrame()

    # ─── Portfolio History (Equity Curve) ──────────────────────
    def get_portfolio_history(self, period: str = "1D", timeframe: str = "5Min") -> pd.DataFrame:
        """
        Uses Alpaca REST endpoint for portfolio/equity history.
        period: '1D', '1W', '1M', '3M', '1A', 'all'
        timeframe: '1Min', '5Min', '15Min', '1H', '1D'
        """
        if not self.is_configured:
            return pd.DataFrame()
        try:
            history = self._trading_client.get_portfolio_history(
                period=period,
                timeframe=timeframe,
            )
            if history and history.timestamp:
                df = pd.DataFrame({
                    "time": [datetime.fromtimestamp(t) for t in history.timestamp],
                    "equity": history.equity,
                    "profit_loss": history.profit_loss,
                    "profit_loss_pct": history.profit_loss_pct,
                })
                return df
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Portfolio history error: {e}")
            return pd.DataFrame()
