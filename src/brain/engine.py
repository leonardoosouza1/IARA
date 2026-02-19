"""
TRINITY ENGINE v9.7 "SWARM LIQUID"
====================================
The Orchestrator. Manages the live trading loop using SwarmBrain v9.7.

Architecture:
  Fast Brain (SwarmBrain): Multi-timeframe workers + Coordinator for tactical decisions
  Slow Brain (LFM):       Optional strategic modulation via Ollama/LFM
  Sentinel:               Safety override (news, volatility, drawdown)
  
Data Flow:
  Market Tick -> Feature Engineering -> SwarmBrain Inference -> Consensus Decision -> Execution

Usage:
  engine = TrinityEngine.from_checkpoint("logs/models/best_model_v9_BTCUSD.pth")
  engine.start()
"""

import threading
import time
import os
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List

import torch
import pandas as pd
import requests

# Models
from src.brain.models.liquid_v9 import SwarmBrain, CoordinatorConfig
from src.brain.models.workers import WorkerConfig

# Trading
from src.trading.signal_generator import SignalEngine, TradeSignal

# Tools
from src.brain.tools.market_scanner import MarketScanner
from src.brain.tools.alpaca_executor import AlpacaExecutor
from src.brain.agents.sentinel_stub import SentinelStub

logger = logging.getLogger("TrinityEngine")


@dataclass
class EngineConfig:
    """Configuration for the TrinityEngine."""
    # LFM (Strategic Agent)
    lfm_model_url: str = os.getenv("OLLAMA_URL", "http://ollama:11434/api/generate")
    lfm_model_name: str = "lfm2.5-thinking:1.2b"
    lfm_refresh_rate: float = 60.0   # Strategic thinking every 60s
    
    # Trading
    symbol: str = "BTC/USD"
    alpaca_symbol: str = "BTC/USD"
    market_type: str = "crypto" # "crypto" or "stock"
    execution_venue: str = "ALPACA" # "ALPACA" or "BINANCE"
    trade_qty: float = 0.01
    
    # Model
    checkpoint_path: str = "logs/models/best_model_v9_BTCUSD.pth"
    worker_hidden: int = 128         # v9.7 "Gladiator" dims

# ... (SharedState/DataBuffer classes omitted for brevity) ...

class TrinityEngine:
    """
    Live trading orchestrator using SwarmBrain v9.7.
    """

    def __init__(self, signal_engine: SignalEngine, config: EngineConfig):
        from src.brain.tools.ccxt_executor import CCXTExecutor # Lazy import to avoid circular deps if any
        
        self.signal_engine = signal_engine
        self.config = config
        self.device = torch.device(config.device)

        # State
        self.state = SharedState(device=config.device)
        self.start_time: float = 0.0  # Set when start() is called
        # H4 resampling needs ~6144 M5 candles (128 seq × 4h × 12 per hour)
        self.buffer = DataBuffer(max_len=10000)

        # Tools
        self.sentinel = SentinelStub()
        self.scanner = MarketScanner()
        
        # Select Executor
        if config.execution_venue == "BINANCE" and config.market_type == "crypto":
             self.executor = CCXTExecutor(exchange_id="binance")
             logger.info(f"🔌 Using CCXT Executor (Binance) for {config.symbol}")
        else:
             # Default to Alpaca (Stocks or Crypto Paper)
             self.executor = AlpacaExecutor()
             logger.info(f"🦙 Using Alpaca Executor for {config.symbol} (Venue: {config.execution_venue})")

        # Thread control
        self.lfm_thread: Optional[threading.Thread] = None
        self.active = False

    @classmethod
    def from_checkpoint(cls, config: Optional[EngineConfig] = None) -> "TrinityEngine":
        """Factory: build engine from checkpoint path in config."""
        if config is None:
            config = EngineConfig()

        signal_engine = SignalEngine(
            checkpoint_path=config.checkpoint_path,
            worker_hidden=config.worker_hidden,
            coord_hidden=config.coord_hidden,
            seq_len=config.seq_len,
            device=config.device,
        )
        return cls(signal_engine, config)

    # ──────────────────────────────────────────
    # Fast Path: Tactical Inference
    # ──────────────────────────────────────────
    async def forward_tick(self, candle: dict) -> Optional[TradeSignal]:
        """
        Process a new M5 candle from the stream.

        Args:
            candle: Dict with {time, open, high, low, close, volume} (time in seconds)

        Returns:
            TradeSignal if inference ran, else None
        """
        # 1. Update Buffer
        self.buffer.add_candle(candle)
        
        # 2. Check if enough data
        # We need enough history for the longest timeframe (H4)
        if len(self.buffer.data) < 200: # Fast fail
            return None
            
        # 3. Prepare DataFrame
        df_m5 = self.buffer.get_dataframe()
        
        # 4. Get signal from SwarmBrain
        # Note: Inference is sync blockage here. Ideally offload to thread.
        signal = self.signal_engine.predict(df_m5)
        if signal is None:
            return None

        # Update shared state
        current_price = candle["close"]
        self.state.update_signal(signal, current_price)

        # 5. Persist Decision (Async)
        await self._persist_signal(signal, candle)

        # Check Sentinel (Watchdog)
        if not self.sentinel.is_safe():
            logger.warning("⚠️ Sentinel BLOCKED execution: safety override active")
            return signal

        # Execute trade
        if signal.direction == "LONG" and signal.confidence > 0.55:
            logger.info(
                f"🚀 LONG ({signal.confidence:.2f}) | "
                f"Workers: L={signal.n_long} S={signal.n_short} W={signal.n_wait}"
            )
            order = self.executor.place_market_order(
                self.config.symbol, self.config.trade_qty, "BUY"
            )
            if order:
                await self._persist_trade(order, signal, "BUY", current_price)

        elif signal.direction == "SHORT" and signal.confidence < 0.45:
            logger.info(
                f"🔻 SHORT ({signal.confidence:.2f}) | "
                f"Workers: L={signal.n_long} S={signal.n_short} W={signal.n_wait}"
            )
            order = self.executor.place_market_order(
                self.config.symbol, self.config.trade_qty, "SELL"
            )
            if order:
                await self._persist_trade(order, signal, "SELL", current_price)

        return signal

    async def _persist_signal(self, signal: TradeSignal, candle: dict):
        """Save decision to DB (Fire and Forget technically, but we await to ensure data integrity)."""
        from src.database.core import AsyncSessionLocal
        from src.database.models import Signal
        
        try:
             async with AsyncSessionLocal() as session:
                db_signal = Signal(
                    symbol=self.config.symbol,
                    timeframe="M5",
                    signal_value=1 if signal.direction == "LONG" else (-1 if signal.direction == "SHORT" else 0),
                    confidence=float(signal.confidence),
                    model_version="v9.7-Gladiator",
                    features_snapshot={
                        "price": candle["close"],
                        "workers_agreement": signal.worker_agreement
                    },
                    worker_votes={
                        "long": signal.n_long,
                        "short": signal.n_short,
                        "wait": signal.n_wait
                    }
                )
                session.add(db_signal)
                await session.commit()
        except Exception as e:
            logger.error(f"Failed to persist signal: {e}")

    async def _persist_trade(self, order: Any, signal: TradeSignal, side: str, price: float):
        """Save trade execution to DB."""
        from src.database.core import AsyncSessionLocal
        from src.database.models import Trade
        
        # Parse order object (CCXT vs Alpaca)
        # CCXT returns dict, Alpaca returns object or dict depending on usage
        order_id = "unknown"
        exec_price = price
        
        if isinstance(order, dict): # CCXT
            order_id = str(order.get('id', 'unknown'))
            # CCXT market orders might not have price immediately if async, but we have 'average' or current price
            exec_price = float(order.get('average') or price)
        else: # Alpaca Object
            order_id = str(getattr(order, 'id', 'unknown'))
        
        try:
             async with AsyncSessionLocal() as session:
                db_trade = Trade(
                    symbol=self.config.symbol,
                    side=side,
                    qty=self.config.trade_qty,
                    price=exec_price,
                    order_id=order_id,
                    strategy="Trinity-V9.7"
                )
                session.add(db_trade)
                await session.commit()
                logger.info(f"💾 Trade saved to DB: {side} @ {exec_price}")
        except Exception as e:
             logger.error(f"Failed to persist trade: {e}")

    # ──────────────────────────────────────────
    # Slow Path: Strategic LFM Worker
    # ──────────────────────────────────────────
    def _lfm_worker(self):
        """Background thread for strategic market analysis via LFM."""
        logger.info(f"🧠 Strategic Agent (LFM: {self.config.lfm_model_name}) Online")

        while self.active:
            start_time = time.time()
            try:
                price = self.state.last_price
                if price == 0:
                    time.sleep(1)
                    continue

                # Phase 1: Tool usage
                prompt_scan = (
                    f"You are a Strategic Trading AI. Current {self.config.symbol} "
                    f"Price: {price:.2f}. Do you need to check Support/Resistance or "
                    f"Trend? Reply with 'CHECK_SR' or 'CHECK_TREND' or 'NO_ACTION'."
                )
                decision = self._query_lfm(prompt_scan).strip()

                tool_output = ""
                if "CHECK_SR" in decision:
                    logger.info("🧠 LFM: CHECK_SR")
                    tool_output = self.scanner.check_support_resistance(
                        self.config.symbol, price
                    )
                elif "CHECK_TREND" in decision:
                    logger.info("🧠 LFM: CHECK_TREND")
                    tool_output = self.scanner.get_trend_alignment(
                        self.config.symbol, "4h"
                    )

                # Phase 2: Synthesize bias
                bias = 0.0
                if "Support" in tool_output:
                    bias += 0.3
                if "Resistance" in tool_output:
                    bias -= 0.3
                if "Bullish" in tool_output:
                    bias += 0.2
                if "Bearish" in tool_output:
                    bias -= 0.2

                self.state.update_lfm(bias, tool_output[:200])
                logger.info(f"🧠 Strategic Bias Updated: {bias:+.2f}")

            except Exception as e:
                logger.error(f"LFM Worker Error: {e}")

            elapsed = time.time() - start_time
            sleep_time = max(1.0, self.config.lfm_refresh_rate - elapsed)
            time.sleep(sleep_time)

    def _query_lfm(self, prompt: str) -> str:
        """Query the Ollama/LFM endpoint."""
        try:
            payload = {
                "model": self.config.lfm_model_name,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.2},
            }
            res = requests.post(
                self.config.lfm_model_url, json=payload, timeout=30
            )
            if res.status_code == 200:
                return res.json().get("response", "")
            else:
                logger.error(f"LFM Error: {res.status_code}")
                return ""
        except Exception as e:
            logger.error(f"LFM Connection Failed: {e}")
            return ""

    # ──────────────────────────────────────────
    # Lifecycle
    # ──────────────────────────────────────────
    def start(self):
        """Start the engine (launches LFM background thread)."""
        self.active = True
        self.start_time = time.time()
        self.lfm_thread = threading.Thread(target=self._lfm_worker, daemon=True)
        self.lfm_thread.start()
        logger.info("🚀 Trinity Engine v9.7 Started")

    def stop(self):
        """Gracefully stop the engine."""
        self.active = False
        if self.lfm_thread:
            self.lfm_thread.join(timeout=5)
        logger.info("🛑 Trinity Engine Stopped")
