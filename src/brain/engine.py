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
    trade_qty: float = 0.01
    
    # Model
    checkpoint_path: str = "logs/models/best_model_v9_BTCUSD.pth"
    worker_hidden: int = 128         # v9.7 "Gladiator" dims
    coord_hidden: int = 512          # v9.7 "Gladiator" dims
    seq_len: int = 128
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class SharedState:
    """Thread-Safe container for the latest 'World View'."""
    
    def __init__(self, device: str):
        self.lock = threading.Lock()
        self.device = device

        # Latest market state
        self.last_price: float = 0.0
        self.last_signal: Optional[TradeSignal] = None
        self.last_update: float = 0.0
        
        # LFM strategic context (optional)
        self.lfm_bias: float = 0.0  # -1 (bearish) to +1 (bullish)
        self.lfm_summary: str = ""

    def update_signal(self, signal: TradeSignal, price: float):
        with self.lock:
            self.last_signal = signal
            self.last_price = price
            self.last_update = time.time()

    def update_lfm(self, bias: float, summary: str):
        with self.lock:
            self.lfm_bias = bias
            self.lfm_summary = summary

    def get_state(self) -> dict:
        with self.lock:
            return {
                "price": self.last_price,
                "signal": self.last_signal,
                "lfm_bias": self.lfm_bias,
                "lfm_summary": self.lfm_summary,
                "last_update": self.last_update,
            }



class DataBuffer:
    """
    Manages a rolling window of OHLCV candles for live inference.
    """
    def __init__(self, max_len: int = 1000):
        self.max_len = max_len
        self.data: List[dict] = []
        self.lock = threading.Lock()

    def add_candle(self, candle: dict):
        with self.lock:
            # candle must have keys: time, open, high, low, close, volume
            self.data.append(candle)
            if len(self.data) > self.max_len:
                self.data.pop(0)

    def get_dataframe(self) -> pd.DataFrame:
        with self.lock:
            df = pd.DataFrame(self.data)
            # Ensure columns are correct types
            if not df.empty:
                df["time"] = pd.to_datetime(df["time"], unit="s")
            return df
            
    @property
    def ready(self) -> bool:
        return len(self.data) >= 128 # Minimum seq_len for v9 model


class TrinityEngine:
    """
    Live trading orchestrator using SwarmBrain v9.7.
    
    Combines:
      - SignalEngine (fast inference via SwarmBrain)
      - LFM Worker (slow strategic thread via Ollama)
      - Sentinel (safety override)
      - AlpacaExecutor (order dispatch)
    """

    def __init__(self, signal_engine: SignalEngine, config: EngineConfig):
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
        self.executor = AlpacaExecutor()

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
    def forward_tick(self, candle: dict) -> Optional[TradeSignal]:
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
        signal = self.signal_engine.predict(df_m5)
        if signal is None:
            return None

        # Update shared state
        current_price = candle["close"]
        self.state.update_signal(signal, current_price)

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
            self.executor.place_market_order(
                self.config.alpaca_symbol, self.config.trade_qty, "BUY"
            )

        elif signal.direction == "SHORT" and signal.confidence < 0.45:
            logger.info(
                f"🔻 SHORT ({signal.confidence:.2f}) | "
                f"Workers: L={signal.n_long} S={signal.n_short} W={signal.n_wait}"
            )
            self.executor.place_market_order(
                self.config.alpaca_symbol, self.config.trade_qty, "SELL"
            )

        return signal

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
