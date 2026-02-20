"""
MarketDataService
==================
Market Data Orchestrator that:
1. Fetches real-time market data via CCXT (Binance/Kraken)
2. Falls back to simulation if data fetch fails
3. Feeds data to FleetManager engines
4. Broadcasts events to Frontend via Socket.IO
"""

import asyncio
import logging
import random
import time
import os
from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta

from src.api.events import SocketEmitter
from src.api.fleet.manager import fleet_manager

# Alpaca Data API
from alpaca.data.historical import CryptoHistoricalDataClient, StockHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest, CryptoLatestTradeRequest, StockBarsRequest, StockLatestTradeRequest
from alpaca.data.timeframe import TimeFrame

logger = logging.getLogger("MarketDataService")

# Minimum interval (seconds) between engine state emissions
_EMIT_DEBOUNCE_S = 1.0

class MarketDataService:
    def __init__(self):
        self.active = False
        self.symbols = ["BTC/USD", "ETH/USD"] # Alpaca Crypto Symbols
        
        # Default starting prices (fallback for first tick)
        self.prices = {"BTC/USD": 97500.0, "ETH/USD": 3800.0}
        
        self.candle_start: Dict[str, float] = {}
        self.candle_ohlc: Dict[str, dict] = {}
        self._last_emit: Dict[str, float] = {}
        self.history: Dict[str, list] = {}  # In-memory history buffer
        
        # Clients
        self.crypto_client: Optional[CryptoHistoricalDataClient] = None
        
        self.api_key = os.getenv("ALPACA_API_KEY")
        self.secret_key = os.getenv("ALPACA_SECRET_KEY")

    async def _init_exchanges(self):
        """Initialize Alpaca Data Clients."""
        if not self.api_key:
             logger.warning("Alpaca keys missing. Running in simulation mode.")
             return

        try:
            self.crypto_client = CryptoHistoricalDataClient(self.api_key, self.secret_key)
            logger.info("✅ Connected to Alpaca Data API (Crypto)")
        except Exception as e:
            logger.error(f"Failed to initialize Alpaca Data: {e}")
            self.crypto_client = None

    async def _close_exchanges(self):
        """Close clients (No-op for REST clients, but kept for interface compatibility)."""
        pass

    def _init_fleet(self):
        """Ensure default agents are running."""
        for symbol in self.symbols:
            try:
                fleet_manager.start_agent(
                    symbol=symbol,
                    market_type="crypto",
                    execution_venue="ALPACA"
                )
            except Exception as e:
                logger.error(f"Failed to auto-start agent for {symbol}: {e}")

    async def _fetch_price(self, symbol: str) -> Optional[float]:
        """Fetch latest price from Alpaca."""
        if not self.crypto_client:
            return None

        # Determine asset class (simplistic check)
        try:
             # Alpaca Crypto symbols: BTC/USD
             req = CryptoLatestTradeRequest(symbol_or_symbols=[symbol])
             res = self.crypto_client.get_crypto_latest_trade(req)
             
             if symbol in res:
                 return float(res[symbol].price)
             return None

        except Exception as e:
            # logger.warning(f"Fetch failed for {symbol}: {e}") # Reduce noise
            return None

    async def _fetch_and_emit_history(self, symbol: str):
        """Fetch historical candles from Alpaca and emit to frontend."""
        candles = []
        
        if self.crypto_client:
            try:
                # Fetch last 2 hours of 1-minute bars
                req = CryptoBarsRequest(
                    symbol_or_symbols=[symbol],
                    timeframe=TimeFrame.Minute,
                    start=datetime.now() - timedelta(hours=2),
                    limit=100
                )
                bars = self.crypto_client.get_crypto_bars(req)
                
                if symbol in bars.df.index.get_level_values(0):
                    df = bars.df.loc[symbol]
                    # Convert to our format
                    for index, row in df.iterrows():
                        # index is timestamp
                         candles.append({
                            "time": int(index.timestamp()),
                            "open": float(row["open"]),
                            "high": float(row["high"]),
                            "low": float(row["low"]),
                            "close": float(row["close"]),
                            "volume": float(row["volume"])
                        })
                    logger.info(f"Fetched {len(candles)} historical candles for {symbol} via Alpaca")
            except Exception as e:
                logger.warning(f"History fetch failed for {symbol}: {e}")

        # Fallback to simulation if empty
        if not candles:
            logger.info(f"Generating simulated history for {symbol}")
            candles = self._generate_simulated_history(symbol, 100)
            
        # Emit
        if candles:
             # Initialize internal state with last candle
            last = candles[-1]
            self.prices[symbol] = last["close"]
            self.candle_ohlc[symbol] = {
                "open": last["close"], "high": last["close"], 
                "low": last["close"], "close": last["close"]
            }
            # Store in buffer
            self.history[symbol] = candles
            
            await SocketEmitter.emit_historical_candles(symbol, candles)

    def get_history(self, symbol: str) -> list:
        """Get cached history for a symbol."""
        return self.history.get(symbol, [])

    def _generate_simulated_history(self, symbol: str, count: int) -> list:
        """Generate random walk history if API fails."""
        candles = []
        price = self.prices.get(symbol, 1000.0)
        volatility = {"BTC/USD": 15, "ETH/USD": 1.5}
        vol = volatility.get(symbol, 1.0)
        
        now = int(time.time())
        # Generate backwards
        history_prices = [price]
        for _ in range(count):
            change = (random.random() - 0.5) * vol * 5 # slightly higher vol for candles
            history_prices.append(history_prices[-1] - change)
        
        history_prices.reverse()
        
        for i, p in enumerate(history_prices):
            open_p = p
            close_p = p + (random.random() - 0.5) * vol
            high = max(open_p, close_p) + random.random() * vol * 0.5
            low = min(open_p, close_p) - random.random() * vol * 0.5
            
            candles.append({
                "time": now - (count - i) * 60, # 1m intervals
                "open": round(open_p, 5),
                "high": round(high, 5),
                "low": round(low, 5),
                "close": round(close_p, 5),
                "volume": random.randint(100, 1000)
            })
            
        return candles

    def _simulate_price_movement(self, symbol: str) -> float:
        """Fallback simulation (Random Walk)."""
        volatility = {"BTC/USD": 15, "ETH/USD": 1.5}
        change = (random.random() - 0.5) * volatility.get(symbol, 1.0)
        # Ensure we have a price
        if symbol not in self.prices:
             self.prices[symbol] = 1000.0
        self.prices[symbol] += change
        return self.prices[symbol]

    async def run(self):
        """Main loop."""
        self.active = True
        self._init_fleet()
        await self._init_exchanges()
        
        # Initial Data Load & Emit
        logger.info("Initializing market data history...")
        for symbol in self.symbols:
            await self._fetch_and_emit_history(symbol)
        
        logger.info("🚀 Market Data Service Started (Alpaca Source)")
        
        now = time.time()
        for s in self.symbols:
            self.candle_start[s] = now
            self._last_emit[s] = 0.0

        while self.active:
            try:
                tasks = []
                now_ts = time.time()
                
                for symbol in self.symbols:
                    # 1. Get Price (Real or Simulated)
                    price = await self._fetch_price(symbol)
                    
                    if price is None:
                        # Fallback
                        price = self._simulate_price_movement(symbol)
                    else:
                        self.prices[symbol] = price
                    
                    # 2. Emit Tick
                    tasks.append(SocketEmitter.emit_market_tick(
                        symbol=symbol,
                        price=round(price, 5),
                        timestamp=str(int(now_ts)),
                    ))
                    
                    # 3. Update Candle Logic
                    self._update_ohlc(symbol, price)
                    
                    # 4. Check Candle Close (5s demo interval for UI responsiveness)
                    # Real application would use 1m, 5m etc. keeping 5s for 'live feel'
                    if now_ts - self.candle_start.get(symbol, now_ts) >= 5:
                        candle = {
                            "time": int(now_ts),
                            "open": self.candle_ohlc[symbol]["open"],
                            "high": self.candle_ohlc[symbol]["high"],
                            "low": self.candle_ohlc[symbol]["low"],
                            "close": self.candle_ohlc[symbol]["close"],
                            "volume": random.randint(100, 5000), # Volume simulated for now
                        }
                        
                        # Emit & Feed
                        tasks.append(SocketEmitter.emit_candle_closed(symbol, candle))
                        tasks.append(self._feed_engine(symbol, candle))
                        
                        # Update History Buffer
                        if symbol not in self.history:
                            self.history[symbol] = []
                        self.history[symbol].append(candle)
                        # Keep buffer size manageable (e.g. 500)
                        if len(self.history[symbol]) > 500:
                            self.history[symbol] = self.history[symbol][-500:]
                        
                        # Reset
                        self.candle_start[symbol] = now_ts
                        self.candle_ohlc[symbol] = {
                           "open": price, "high": price, "low": price, "close": price
                        }
                        
                        # 5. Emit Engine State
                        if now_ts - self._last_emit.get(symbol, 0) >= _EMIT_DEBOUNCE_S:
                            await self._emit_engine_state(symbol)
                            self._last_emit[symbol] = now_ts

                if tasks:
                    await asyncio.gather(*tasks)
                
                # Sleep to respect rate limits (1s is safe for 3 symbols on public API)
                await asyncio.sleep(1.0)

            except Exception as e:
                logger.error(f"Market loop error: {e}")
                await asyncio.sleep(5) # Backoff on error

    def _update_ohlc(self, symbol: str, price: float):
        if symbol not in self.candle_ohlc:
            self.candle_ohlc[symbol] = {
                "open": price, "high": price, "low": price, "close": price
            }
        else:
            self.candle_ohlc[symbol]["high"] = max(self.candle_ohlc[symbol]["high"], price)
            self.candle_ohlc[symbol]["low"] = min(self.candle_ohlc[symbol]["low"], price)
            self.candle_ohlc[symbol]["close"] = price

    async def _feed_engine(self, symbol: str, candle: dict):
        agent_id = f"agent-{symbol.lower().replace('/', '-')}"
        engine = fleet_manager.engines.get(agent_id)
        if engine and engine.active:
            try:
                await engine.forward_tick(candle)
            except Exception as e:
                logger.error(f"Engine push failed for {agent_id}: {e}")

    async def _emit_engine_state(self, symbol: str):
        """Check engine state and emit thoughts/updates to the frontend."""
        agent_id = f"agent-{symbol.lower().replace('/', '-')}"
        engine = fleet_manager.engines.get(agent_id)
        if not engine:
            return
            
        state = engine.state.get_state()
        signal = state.get("signal")
        lfm_summary = state.get("lfm_summary")
        current_position = engine.state.get_position_summary()
        
        if lfm_summary:
            await SocketEmitter.emit_brain_thought(
                agent_id=agent_id,
                step="strategy",
                text=f"LFM: {lfm_summary}",
                level="info"
            )
            
        if signal:
            await SocketEmitter.emit_brain_thought(
                agent_id=agent_id,
                step="signal",
                text=f"Swarm Consensus: {signal.confidence:.2f} ({signal.direction}) | Agreement: {signal.worker_agreement:.1%}",
                level="success" if signal.confidence > 0.6 else "info"
            )
            
            # Use real position data if available, or signal-derived
            pos_data = {
                "side": current_position["side"] if current_position else (signal.direction.lower() if signal.direction != "WAIT" else "none"),
                "size": current_position["size"] if current_position else 0.01,
            }

            await SocketEmitter.emit_agent_update({
                "id": agent_id,
                "symbol": symbol,
                "status": "active" if pos_data["side"] == "none" else "in_trade",
                "current_position": pos_data
            })

    async def stop(self):
        """Stop service and cleanup."""
        self.active = False
        await self._close_exchanges()
        logger.info("Market Data Service Stopped")

market_service = MarketDataService()

