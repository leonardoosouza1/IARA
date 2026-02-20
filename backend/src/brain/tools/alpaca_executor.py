import os
import logging
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

# Alpaca-py imports
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest, GetOrdersRequest
from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus
from alpaca.common.exceptions import APIError

load_dotenv()

class AlpacaExecutor:
    """
    Execution Layer for Alpaca Markets.
    Uses alpaca-py SDK to interact with Paper or Live accounts.
    """
    def __init__(self):
        self.api_key = os.getenv("ALPACA_API_KEY")
        self.secret_key = os.getenv("ALPACA_SECRET_KEY")
        self.paper = os.getenv("ALPACA_PAPER", "true").lower() == "true"
        
        if not self.api_key or not self.secret_key:
            self.client = None
            logging.error("Alpaca credentials missing in .env")
        else:
            self.client = TradingClient(self.api_key, self.secret_key, paper=self.paper)
            
        self.logger = logging.getLogger("AlpacaExecutor")

    def get_account_summary(self) -> Dict[str, Any]:
        """Fetches equity, balance, and buying power from Alpaca."""
        if not self.client:
            return {}
        try:
            acc = self.client.get_account()
            return {
                "equity": acc.equity,
                "balance": acc.cash,
                "buying_power": acc.buying_power,
                "unrealized_pl": acc.unrealized_intraday_pl,
                "currency": acc.currency
            }
        except Exception as e:
            self.logger.error(f"Failed to fetch Alpaca account info: {e}")
            return {}

    def place_market_order(self, symbol: str, volume: float, side: str = "BUY") -> Optional[Any]:
        """
        Places a market order via Alpaca.
        side: 'BUY' or 'SELL'
        volume: float (shares/units)
        """
        if not self.client:
            return None
            
        order_side = OrderSide.BUY if side.upper() == "BUY" else OrderSide.SELL
        
        # Preparing order
        # Note: For crypto, symbol should be like "BTC/USD"
        # For stocks, symbol should be like "AAPL"
        try:
            order_request = MarketOrderRequest(
                symbol=symbol,
                qty=abs(volume),
                side=order_side,
                time_in_force=TimeInForce.GTC
            )
            
            order = self.client.submit_order(order_data=order_request)
            self.logger.info(f"🚀 Alpaca Order Submitted: {symbol} {side} {volume}")
            return order
        except APIError as e:
            self.logger.error(f"Alpaca API Error: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Failed to place Alpaca order: {e}")
            return None

    def get_open_positions(self) -> List[Dict[str, Any]]:
        """Returns active positions via Alpaca."""
        if not self.client:
            return []
        try:
            positions = self.client.get_all_positions()
            return [
                {
                    "symbol": p.symbol,
                    "qty": p.qty,
                    "entry": p.avg_entry_price,
                    "market_value": p.market_value,
                    "unrealized_pl": p.unrealized_pl
                }
                for p in positions
            ]
        except Exception as e:
            self.logger.error(f"Failed to fetch Alpaca positions: {e}")
            return []

    def close_all_positions(self):
        """Emergency stop: Close everything."""
        if not self.client:
            return
        try:
            self.client.close_all_positions(cancel_orders=True)
            self.logger.warning("🔴 EMERGENCY: All Alpaca positions closed and orders cancelled.")
        except Exception as e:
            self.logger.error(f"Failed to close all positions: {e}")
