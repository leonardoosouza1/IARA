import os
import time
import logging
from typing import Dict, Any, Optional
from binance.client import Client
from binance.exceptions import BinanceAPIException, BinanceOrderException

logger = logging.getLogger("api.binance_executor")

class BinanceExecutor:
    """
    Execution Layer for Binance Markets (Phase 5).
    Handles LIVE execution and TESTNET simulated logging with Slippage.
    Includes a Latency Kill-Switch to prevent execution during API lag spikes.
    """
    def __init__(self, symbol: str = "BTCUSDT", mode: str = "TESTNET"):
        self.symbol = symbol.upper().replace("/", "").replace("-", "")
        self.mode = os.getenv("TRADE_MODE", mode).upper() # TESTNET or LIVE
        
        # Kill-Switch Tolerance
        self.max_latency_ms = 500
        
        # Keys
        self.api_key = os.getenv("BINANCE_API_KEY", "")
        self.secret_key = os.getenv("BINANCE_SECRET_KEY", "")
        
        self.client = Client(self.api_key, self.secret_key)
        
        # If true TESTNET on Binance API
        if self.mode == "TESTNET" and "testnet" in self.api_key.lower():
             self.client.API_URL = 'https://testnet.binance.vision/api'
             
        logger.info(f"🛡️ Binance Executor online for {self.symbol} | Mode: {self.mode}")

    def _check_latency_killswitch(self) -> bool:
        """Pings Binance API. Returns False if latency is > 500ms or connection fails."""
        try:
            start_time = time.time()
            self.client.ping()
            latency_ms = (time.time() - start_time) * 1000
            
            if latency_ms > self.max_latency_ms:
                logger.error(f"🚨 TICKER LAG DETECTED: {latency_ms:.1f}ms (Limit: {self.max_latency_ms}ms). Kill-Switch Activated. Order Aborted.")
                return False
                
            logger.debug(f"API Latency: {latency_ms:.1f}ms - OK")
            return True
        except Exception as e:
            logger.error(f"🚨 API PING FAILED: {e}. Kill-Switch Activated.")
            return False

    def execute_signal(self, action: str, confidence: float, current_price: float, qty: float = 0.001) -> Optional[Dict[str, Any]]:
        """
        Receives signal from IARA (e.g., LONG, SHORT).
        Action should be "BUY" or "SELL".
        """
        if action.upper() not in ["BUY", "SELL"]:
            logger.warning(f"Invalid signal received: {action}")
            return None
            
        logger.info(f"🧠 IARA Signal Triggered -> {action} {qty} {self.symbol} | Confidence: {confidence:.1%}")

        # 1. Latency Kill-Switch
        if not self._check_latency_killswitch():
            return {"status": "aborted", "reason": "latency_killswitch"}

        # 2. Execution Logic
        if self.mode == "TESTNET":
            # Simulate Execution
            slippage_pct = 0.001 # 0.1% Average Slippage
            taker_fee = 0.0004   # 0.04% Typical taker fee
            
            # Calculate slippage impact based on direction
            if action.upper() == "BUY":
                executed_price = current_price * (1 + slippage_pct)
            else:
                executed_price = current_price * (1 - slippage_pct)
                
            fee_cost = (executed_price * qty) * taker_fee
            
            order_data = {
                "symbol": self.symbol,
                "orderId": f"SIM_{int(time.time()*1000)}",
                "clientOrderId": "sim_testnet_saci",
                "transactTime": int(time.time() * 1000),
                "price": 0.0, # Market order
                "origQty": qty,
                "executedQty": qty,
                "status": "FILLED",
                "type": "MARKET",
                "side": action.upper(),
                "fills": [{"price": str(executed_price), "qty": str(qty), "commission": str(fee_cost), "commissionAsset": "USDT"}]
            }
            logger.info(f"🟢 [TESTNET] Simulated {action} executed at ~${executed_price:.2f} (Slippage applied)")
            return order_data

        elif self.mode == "LIVE":
            # LIVE Market Execution
            try:
                order = self.client.create_order(
                    symbol=self.symbol,
                    side=action.upper(),
                    type=Client.ORDER_TYPE_MARKET,
                    quantity=qty
                )
                logger.info(f"🔥 [LIVE TRADING] Order executed successfully: {order['orderId']}")
                return order
            except BinanceAPIException as e:
                logger.error(f"❌ [LIVE TRADING] Binance API Error: {e}")
                return {"status": "failed", "reason": str(e)}
            except BinanceOrderException as e:
                logger.error(f"❌ [LIVE TRADING] Binance Order Exception: {e}")
                return {"status": "failed", "reason": str(e)}
