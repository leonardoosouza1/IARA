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
from src.brain.tools.binance_executor import BinanceExecutor
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
    binance_symbol: str = "BTCUSDT"
    market_type: str = "crypto" # "crypto" or "stock"
    execution_venue: str = "BINANCE"
    trade_qty: float = 0.01
    
    # Model
    checkpoint_path: str = "logs/models/best_model_v9_BTCUSD.pth"
    worker_hidden: int = 0   # 0 = auto-detect from checkpoint
    coord_hidden: int = 0    # 0 = auto-detect from checkpoint
    seq_len: int = 128
    device: str = "cpu"


class DataBuffer:
    """Manages streaming market data and converts to SwarmBrain-ready DataFrames."""
    def __init__(self, max_len: int = 10000):
        self.max_len = max_len
        self.data: List[dict] = []

    def add_candle(self, candle: dict):
        self.data.append(candle)
        if len(self.data) > self.max_len:
            self.data.pop(0)

    def get_dataframe(self) -> pd.DataFrame:
        df = pd.DataFrame(self.data)
        return df

class SharedState:
    """Thread-safe state shared between LFM and fast SwarmBrain."""
    def __init__(self, device: str = "cpu"):
        self.lock = threading.Lock()
        self.lfm_bias = 0.0
        self.lfm_rationale = ""
        self.last_price: float = 0.0
        self.last_signal: Optional[TradeSignal] = None
        self.cumulative_pnl: float = 0.0
        self.recent_trades: list = []

    def update_lfm(self, bias: float, rationale: str):
        with self.lock:
            self.lfm_bias = bias
            self.lfm_rationale = rationale

    def update_signal(self, signal: TradeSignal, current_price: float):
        with self.lock:
            self.last_signal = signal
            self.last_price = current_price

    def add_trade(self, trade: dict):
        with self.lock:
            self.recent_trades.insert(0, trade)
            if len(self.recent_trades) > 50:
                self.recent_trades.pop()
            # Simple mock PnL for demo purposes if not updating fully from closures
            self.cumulative_pnl += trade.get("pnl", 0.0)

    def get_state(self) -> dict:
        with self.lock:
            return {
                "lfm_bias": self.lfm_bias,
                "lfm_summary": self.lfm_rationale,
                "last_price": self.last_price,
                "signal": self.last_signal,
                "pnl": self.cumulative_pnl,
                "trades": list(self.recent_trades)
            }
            
    def get_position_summary(self) -> Optional[dict]:
        # Placeholder for actual position tracking in engine
        return None

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
        
        # Executor
        self.executor = BinanceExecutor(symbol=config.symbol)
        logger.info(f"🔌 Using Binance Executor for {config.symbol} | Mode: {self.executor.mode}")

        # Thread control
        self.lfm_thread: Optional[threading.Thread] = None
        self.active = False
        
    def pre_warm(self, candles: List[dict]):
        """Inject historical candles into the engine's buffer without triggering ML inference or executions."""
        for c in candles:
            self.buffer.add_candle(c)
        logger.info(f"⚙️ TrinityEngine pre-warmed with {len(candles)} historical candles.")

    @classmethod
    def from_checkpoint(cls, config: Optional[EngineConfig] = None) -> "TrinityEngine":
        """Factory: build engine from checkpoint path in config."""
        if config is None:
            config = EngineConfig()

        # Auto-detect dimensions from checkpoint if not explicitly set
        worker_hidden = config.worker_hidden
        coord_hidden = config.coord_hidden
        
        if (worker_hidden == 0 or coord_hidden == 0) and os.path.exists(config.checkpoint_path):
            try:
                state = torch.load(config.checkpoint_path, map_location="cpu", weights_only=False)
                # Detect worker_hidden from input_proj weight shape
                w = state.get("swarm.workers.M5.input_proj.weight")
                if w is not None and hasattr(w, 'shape') and len(w.shape) == 2:
                    worker_hidden = int(w.shape[0])  # hidden_dim
                    # We NO LONGER auto-detect input_dim to allow it to grow. 
                    # If input_dim grows, the inner SwarmBrain handles the padding.
                    logger.info(f"  Auto-detected worker_hidden={worker_hidden} from checkpoint")
                    
                # Detect coord_hidden from policy_head weight shape
                ph = state.get("coordinator.policy_head.weight")
                if ph is not None and hasattr(ph, 'shape') and len(ph.shape) == 2:
                    coord_hidden = int(ph.shape[1])  # hidden_dim
                    logger.info(f"  Auto-detected coord_hidden={coord_hidden} from checkpoint")
            except Exception as e:
                logger.warning(f"  Could not auto-detect dims: {e}. Using defaults.")
        
        # Fallback to safe defaults if still 0
        if worker_hidden == 0: worker_hidden = 64
        if coord_hidden == 0: coord_hidden = 256

        signal_engine = SignalEngine(
            checkpoint_path=config.checkpoint_path,
            worker_hidden=worker_hidden,
            coord_hidden=coord_hidden,
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

        # 🎯 High Precision Sniper Mode (Threshold > 0.82)
        if signal.direction == "LONG" and signal.confidence > 0.82:
            logger.info(
                f"🚀 [SNIPER] LONG ({signal.confidence:.2f}) | "
                f"Workers: L={signal.n_long} S={signal.n_short} W={signal.n_wait}"
            )
            order = self.executor.execute_signal(
                action="BUY", confidence=signal.confidence, current_price=current_price, qty=self.config.trade_qty
            )
            if order and order.get("status") != "aborted":
                await self._persist_trade(order, signal, "BUY", current_price)

        elif signal.direction == "SHORT" and signal.confidence > 0.82:
            logger.info(
                f"🔻 [SNIPER] SHORT ({signal.confidence:.2f}) | "
                f"Workers: L={signal.n_long} S={signal.n_short} W={signal.n_wait}"
            )
            order = self.executor.execute_signal(
                action="SELL", confidence=signal.confidence, current_price=current_price, qty=self.config.trade_qty
            )
            if order and order.get("status") != "aborted":
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
        
        # Parse order object (Binance native)
        order_id = "unknown"
        exec_price = price
        
        if isinstance(order, dict):
            order_id = str(order.get('orderId', 'unknown'))
            # Calculate average execution price if available
            fills = order.get("fills", [])
            if fills:
                try:
                    total_cost = sum(float(f["price"]) * float(f["qty"]) for f in fills)
                    total_qty = sum(float(f["qty"]) for f in fills)
                    if total_qty > 0:
                        exec_price = total_cost / total_qty
                except Exception:
                    pass
        
        # Update live telemetry state
        import time
        # Random mock PnL logic for immediate visual feedback on the Live Terminal since we don't have position closing modeled in this sprint
        mock_pnl = (exec_price * self.config.trade_qty * 0.005) if side == "BUY" else -(exec_price * self.config.trade_qty * 0.002)
        
        trade_event = {
            "id": order_id,
            "symbol": self.config.symbol,
            "side": side,
            "price": exec_price,
            "pnl": mock_pnl,
            "status": "OPEN",
            "confidence": float(signal.confidence),
            "time": time.time() * 1000
        }
        self.state.add_trade(trade_event)
        
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
