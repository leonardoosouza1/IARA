"""
Market Data Orchestrator
========================
Generates or fetches market data, feeds it to the FleetManager engines,
and broadcasts events to the frontend via Socket.IO.
"""

import asyncio
import logging
import random
import time
from typing import Dict, List

from src.api.events import SocketEmitter
from src.api.fleet.manager import fleet_manager

logger = logging.getLogger("MarketDataService")

# Minimum interval (seconds) between engine state emissions per symbol
_EMIT_DEBOUNCE_S = 3.0


class MarketDataService:
    def __init__(self):
        self.active = False
        self.symbols = ["BTC/USD", "ETH/USD", "EUR/USD"]
        
        # Simulation state
        self.prices = {"BTC/USD": 97500.0, "ETH/USD": 3800.0, "EUR/USD": 1.0850}
        self.candle_start: Dict[str, float] = {}
        self.candle_ohlc: Dict[str, dict] = {}
        self._last_emit: Dict[str, float] = {}

    def _init_fleet(self):
        """Ensure default agents are running. Called once from run()."""
        market_types = {"BTC/USD": "crypto", "ETH/USD": "crypto", "EUR/USD": "forex"}
        for symbol in self.symbols:
            try:
                fleet_manager.start_agent(
                    symbol=symbol,
                    market_type=market_types.get(symbol, "crypto"),
                )
            except Exception as e:
                logger.error(f"Failed to auto-start agent for {symbol}: {e}")

    async def run(self):
        """Main loop: generate data, feed engines, emit events."""
        self._init_fleet()
        self.active = True
        
        now = time.time()
        for s in self.symbols:
            self.candle_start[s] = now
            self._last_emit[s] = 0.0
        
        logger.info("🚀 Market Data Service Started")
        
        while self.active:
            try:
                tasks = []
                now_ts = time.time()
                
                for symbol in self.symbols:
                    # 1. Simulate Price Movement
                    volatility = {"BTC/USD": 30, "ETH/USD": 3, "EUR/USD": 0.0001}
                    change = (random.random() - 0.48) * volatility[symbol]
                    self.prices[symbol] += change
                    price = self.prices[symbol]
                    
                    # 2. Emit Tick
                    tasks.append(SocketEmitter.emit_market_tick(
                        symbol=symbol,
                        price=round(price, 5),
                        timestamp=str(int(now_ts)),
                    ))
                    
                    # 3. Update Candle
                    self._update_ohlc(symbol, price)
                    
                    # 4. Check Candle Close (5s demo interval)
                    if now_ts - self.candle_start.get(symbol, now_ts) >= 5:
                        candle = {
                            "time": int(now_ts),
                            "open": self.candle_ohlc[symbol]["open"],
                            "high": self.candle_ohlc[symbol]["high"],
                            "low": self.candle_ohlc[symbol]["low"],
                            "close": self.candle_ohlc[symbol]["close"],
                            "volume": random.randint(100, 5000),
                        }
                        
                        # Emit Candle
                        tasks.append(SocketEmitter.emit_candle_closed(symbol, candle))
                        
                        # Feed Engine
                        self._feed_engine(symbol, candle)
                        
                        # Reset Candle
                        self.candle_start[symbol] = now_ts
                        self.candle_ohlc[symbol] = {
                           "open": price, "high": price, "low": price, "close": price
                        }
                        
                        # 5. Emit Engine State (with debounce)
                        if now_ts - self._last_emit.get(symbol, 0) >= _EMIT_DEBOUNCE_S:
                            await self._emit_engine_state(symbol)
                            self._last_emit[symbol] = now_ts

                await asyncio.gather(*tasks)
                await asyncio.sleep(0.1)

            except Exception as e:
                logger.error(f"Market loop error: {e}")
                await asyncio.sleep(1)

    def _update_ohlc(self, symbol: str, price: float):
        if symbol not in self.candle_ohlc:
            self.candle_ohlc[symbol] = {
                "open": price, "high": price, "low": price, "close": price
            }
        else:
            self.candle_ohlc[symbol]["high"] = max(self.candle_ohlc[symbol]["high"], price)
            self.candle_ohlc[symbol]["low"] = min(self.candle_ohlc[symbol]["low"], price)
            self.candle_ohlc[symbol]["close"] = price

    def _feed_engine(self, symbol: str, candle: dict):
        """Push candle to the corresponding engine."""
        agent_id = f"agent-{symbol.lower().replace('/', '-')}"
        engine = fleet_manager.engines.get(agent_id)
        if engine and engine.active:
            try:
                engine.forward_tick(candle)
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
            
            await SocketEmitter.emit_agent_update({
                "id": agent_id,
                "symbol": symbol,
                "status": "active",
                "current_position": {
                    "side": signal.direction.lower() if signal.direction != "WAIT" else "none",
                    "size": 0.01,
                }
            })

    def stop(self):
        self.active = False
