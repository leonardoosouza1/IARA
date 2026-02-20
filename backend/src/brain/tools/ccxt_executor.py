import os
import logging
from typing import Dict, Any, List, Optional, Tuple
from dotenv import load_dotenv

# Try importing ccxt
try:
    import ccxt
except ImportError:
    ccxt = None

load_dotenv()

logger = logging.getLogger("CCXTExecutor")

class CCXTExecutor:
    """
    Execution Layer for Crypto Exchanges via CCXT.
    Supports Binance (Spot/Futures) and Kraken.
    """
    def __init__(self, exchange_id: str = "binance", default_type: str = "spot"):
        self.exchange_id = exchange_id
        self.exchange = None
        self.active = False
        
        if not ccxt:
            logger.error("CCXT library not installed.")
            return

        # Load Keys based on exchange_id
        api_key = os.getenv(f"{exchange_id.upper()}_API_KEY")
        secret = os.getenv(f"{exchange_id.upper()}_SECRET_KEY")
        
        if not api_key or not secret:
            logger.warning(f"No API Keys found for {exchange_id.upper()} in .env. Execution will fail.")
            # We initialize anyway to allow dry-run or verification steps, 
            # but private methods will fail.
        
        try:
            exchange_class = getattr(ccxt, exchange_id)
            self.exchange = exchange_class({
                'apiKey': api_key,
                'secret': secret,
                'enableRateLimit': True,
                'options': {'defaultType': default_type}
            })
            
            # Load markets to verify connection
            # self.exchange.load_markets() # This might block, better to do lazy or async if possible
            # But Executor is usually sync in this architecture or threaded? 
            # SentinelStub and Engine seem to use blocking calls or threads. 
            # We'll stick to blocking for simplicity in this Executor class.
            self.active = True
            logger.info(f"✅ CCXT Executor initialized for {exchange_id} ({default_type})")
            
        except Exception as e:
            logger.error(f"Failed to initialize CCXT Executor: {e}")

        # Symbol Mapping (Internal -> Exchange)
        self._symbol_map = {
            "BTC/USD": "BTC/USDT", # Binance standard
            "ETH/USD": "ETH/USDT",
            # Add more as needed or make dynamic
        }

    def _resolve_symbol(self, symbol: str) -> str:
        """Maps internal symbol (BTC/USD) to exchange symbol (BTC/USDT)."""
        return self._symbol_map.get(symbol, symbol)

    def get_account_summary(self) -> Dict[str, Any]:
        """Fetches free balance and total equity."""
        if not self.exchange or not self.exchange.apiKey:
            return {}
            
        try:
            balance = self.exchange.fetch_balance()
            # Parse standard CCXT balance response
            total = balance.get('total', {})
            free = balance.get('free', {})
            
            # Estimate equity in USDT (simplified)
            equity = conversion = 0.0
            if 'USDT' in total:
                equity = total['USDT']
            
            return {
                "equity": equity, # Approximate
                "balance": free.get('USDT', 0.0),
                "buying_power": free.get('USDT', 0.0),
                "raw": balance['info']
            }
        except Exception as e:
            logger.error(f"Failed to fetch balance: {e}")
            return {}

    def place_market_order(self, symbol: str, volume: float, side: str = "BUY") -> Optional[Any]:
        """
        Places a market order.
        """
        if not self.exchange or not self.exchange.apiKey:
            logger.error("Cannot place order: No API Keys or Exchange not init.")
            return None
            
        remote_symbol = self._resolve_symbol(symbol)
        side = side.lower() # ccxt expects 'buy' or 'sell'
        
        try:
            logger.info(f"🚀 Sending Order: {remote_symbol} {side} {volume}")
            order = self.exchange.create_market_order(remote_symbol, side, volume)
            logger.info(f"✅ Order Filled: {order['id']} @ {order.get('average', 'Market')}")
            return order
        except Exception as e:
            logger.error(f"CCXT Order Failed: {e}")
            return None

    def get_open_positions(self) -> List[Dict[str, Any]]:
        """
        Fetches open positions. 
        Note: For Spot, this is just positive balances. 
        For Futures, it's actual positions.
        """
        if not self.exchange:
            return []
            
        # Logic differs for Spot vs Futures
        # Assuming Spot for now as per defaultType='spot'
        try:
            balance = self.exchange.fetch_balance()
            total = balance.get('total', {})
            
            positions = []
            for coin, amount in total.items():
                if amount > 0 and coin != 'USDT':
                     positions.append({
                         "symbol": f"{coin}/USDT",
                         "qty": amount,
                         "entry": 0.0, # Spot doesn't easily track avg entry without history
                         "market_value": 0.0, # Would need current price
                         "unrealized_pl": 0.0
                     })
            return positions
        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            return []

    def close_all_positions(self):
        """Emergency panic button."""
        # Complex to implement for Spot (Sell all assets).
        # Implemented as log warning for now.
        logger.warning("🔴 Close All Positions not fully implemented for CCXT Spot yet.")
