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
from src.database.core import AsyncSessionLocal
from src.database.models import AgentProfile
from sqlalchemy.future import select

# Binance API
from binance.client import Client
from src.api.sensory_hub import BinanceSensoryHub

logger = logging.getLogger("MarketDataService")

# Minimum interval (seconds) between engine state emissions
_EMIT_DEBOUNCE_S = 1.0

class MarketDataService:
    def __init__(self):
        self.active = False
        self.symbols = ["BTC/USD", "EUR/USD"] # Default Crypto & Forex symbol
        
        # Default starting prices (fallback for first tick)
        self.prices = {"BTC/USD": 97500.0, "EUR/USD": 1.0500}
        
        self.candle_start: Dict[str, float] = {}
        self.candle_ohlc: Dict[str, dict] = {}
        self._last_emit: Dict[str, float] = {}
        self.history: Dict[str, list] = {}  # In-memory history buffer
        
        # Clients
        self.crypto_client: Optional[Client] = None
        
        self.api_key = os.getenv("BINANCE_API_KEY")
        self.secret_key = os.getenv("BINANCE_SECRET_KEY")
        
        # Sensory Hub (RAIO-X)
        self.sensory_hub: Optional[BinanceSensoryHub] = None

    async def _init_exchanges(self):
        """Initialize Binance Data Clients."""
        if not self.api_key:
             logger.warning("Binance keys missing. Running in simulation mode.")
             return

        try:
            self.crypto_client = Client(self.api_key, self.secret_key)
            self.sensory_hub = BinanceSensoryHub(self.api_key, self.secret_key)
            logger.info("✅ Connected to Binance Data API (Crypto & Sensory Hub)")
        except Exception as e:
            logger.error(f"Failed to initialize Binance Data: {e}")
            self.crypto_client = None
            self.sensory_hub = None

    async def _close_exchanges(self):
        """Close clients (No-op for REST clients, but kept for interface compatibility)."""
        pass

    async def _init_fleet(self):
        """Ensure default agents are running based on active profiles in database."""
        try:
            async with AsyncSessionLocal() as session:
                result = await session.execute(select(AgentProfile).where(AgentProfile.is_active == True))
                agents = result.scalars().all()
                self.symbols = []
                for a in agents:
                    self.symbols.append(a.symbol)
                    try:
                        agent_id = fleet_manager.start_agent(
                            symbol=a.symbol,
                            market_type=a.market_type,
                            model_name=a.model_version,
                            execution_venue="BINANCE",
                        )
                        engine = fleet_manager.engines.get(agent_id)
                        if engine:
                            await self._pre_warm_engine(a.symbol, engine)
                    except Exception as e:
                        logger.error(f"Failed to auto-start agent for {a.symbol}: {e}")
                
                if not self.symbols:
                    logger.info("No active agents found in DB. Waiting for UI configuration.")
        except Exception as e:
            logger.error(f"Error querying active agents: {e}")

    async def _pre_warm_engine(self, symbol: str, engine):
        """Fetch 48h of 5m bars and instantly fill the TrinityEngine buffer."""
        if not self.crypto_client:
            logger.warning(f"No Binance client. Skipping pre-warm for {symbol}.")
            return

        try:
            logger.info(f"🔥 Pre-warming engine for {symbol} with 48h of 5m historical data...")
            binance_symbol = symbol.replace("/", "").replace("-", "")
            klines = self.crypto_client.get_historical_klines(binance_symbol, Client.KLINE_INTERVAL_5MINUTE, "48 hours ago UTC")
            
            candles = []
            for k in klines:
                candles.append({
                    "time": int(k[0] / 1000),
                    "open": float(k[1]),
                    "high": float(k[2]),
                    "low": float(k[3]),
                    "close": float(k[4]),
                    "volume": float(k[5])
                })
                
            if candles:
                engine.pre_warm(candles)
            else:
                logger.warning(f"No historical data found for {symbol} pre-warming.")
        except Exception as e:
            logger.error(f"Pre-warming failed for {symbol}: {e}")

    async def _fetch_price(self, symbol: str) -> Optional[float]:
        """Fetch latest price from Binance."""
        if not self.crypto_client:
            return None

        try:
             binance_symbol = symbol.replace("/", "").replace("-", "")
             res = self.crypto_client.get_symbol_ticker(symbol=binance_symbol)
             return float(res['price'])
        except Exception as e:
            # logger.warning(f"Fetch failed for {symbol}: {e}") # Reduce noise
            return None

    async def _fetch_and_emit_history(self, symbol: str):
        """Fetch historical candles from Binance and emit to frontend."""
        candles = []
        
        if self.crypto_client:
            try:
                # Fetch last 2 hours of 1-minute bars
                binance_symbol = symbol.replace("/", "").replace("-", "")
                klines = self.crypto_client.get_historical_klines(binance_symbol, Client.KLINE_INTERVAL_1MINUTE, "2 hours ago UTC")
                
                for k in klines:
                    candles.append({
                        "time": int(k[0] / 1000),
                        "open": float(k[1]),
                        "high": float(k[2]),
                        "low": float(k[3]),
                        "close": float(k[4]),
                        "volume": float(k[5])
                    })
                logger.info(f"Fetched {len(candles)} historical candles for {symbol} via Binance")
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
        volatility = {"BTC/USD": 15, "EUR/USD": 0.00015}
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
        volatility = {"BTC/USD": 15, "EUR/USD": 0.00015}
        change = (random.random() - 0.5) * volatility.get(symbol, 1.0)
        # Ensure we have a price
        if symbol not in self.prices:
             self.prices[symbol] = 1000.0
        self.prices[symbol] += change
        return self.prices[symbol]

    async def run(self):
        """Main loop."""
        self.active = True
        await self._init_fleet()
        await self._init_exchanges()
        
        # Initial Data Load & Emit
        logger.info("Initializing market data history & sensory hub...")
        for symbol in self.symbols:
            await self._fetch_and_emit_history(symbol)
            
        if self.sensory_hub:
            await self.sensory_hub.start(self.symbols)
        
        logger.info("🚀 Market Data Service Started (Binance Source)")
        
        now = time.time()
        for s in self.symbols:
            self.candle_start[s] = now
            self._last_emit[s] = 0.0

        while self.active:
            try:
                tasks = []
                now_ts = time.time()
                
                # Check for dynamically added agents
                current_fleet_symbols = [cfg.symbol for cfg in fleet_manager.configs.values()]
                for sym in current_fleet_symbols:
                    if sym not in self.symbols:
                        self.symbols.append(sym)
                        self.candle_start[sym] = now_ts
                        self._last_emit[sym] = 0.0
                        logger.info(f"MarketDataService picked up new symbol: {sym}")
                        agent_id = f"agent-{sym.lower().replace('/', '-')}"
                        engine = fleet_manager.engines.get(agent_id)
                        if engine:
                            await self._pre_warm_engine(sym, engine)
                
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
                # Enrich candle with Sensory Data (L2)
                if self.sensory_hub:
                    sensory_vector = self.sensory_hub.get_sensory_vector(symbol)
                    if sensory_vector is not None:
                        candle["sensory_l2"] = sensory_vector.tolist()
                
                await engine.forward_tick(candle)
            except Exception as e:
                logger.error(f"Engine push failed for {agent_id}: {e}")

    async def _emit_engine_state(self, symbol: str):
        """Check engine state and emit thoughts/updates to the frontend."""
        if self.sensory_hub:
            l2_data = self.sensory_hub.l2_cache.get(symbol)
            if l2_data:
                vbi = l2_data["vbi"]
                cancel = l2_data["cancel_rate"]
                await SocketEmitter.emit_brain_thought(
                    agent_id=f"agent-{symbol.lower().replace('/', '-')}",
                    step="raio-x",
                    text=f"Vision: Imbalance {vbi:+.2f} | Cancel {cancel:.1%}",
                    level="warning" if abs(vbi) > 0.7 or cancel > 0.3 else "info"
                )

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

