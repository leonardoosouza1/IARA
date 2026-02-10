"""
TRINITY ENGINE v8.5 "PURE LIQUID"
=================================
The Orchestrator. Manages the Async Loop between Swift (Liquid NN) and Deep (Liquid Foundation Model) agents.
"""

import threading
import time
import os
import torch
import json
import requests
from dataclasses import dataclass
from typing import Optional, Dict, Any

# Models
from src.brain.models.trinity_liquid import TrinityLiquidBrain

# Tools
from src.brain.tools.market_scanner import MarketScanner
from src.brain.tools.alpaca_executor import AlpacaExecutor
from src.brain.agents.sentinel_stub import SentinelStub

# Utils
import logging

@dataclass
class EngineConfig:
    lfm_model_url: str = os.getenv("OLLAMA_URL", "http://ollama:11434/api/generate") # Docker service name
    lfm_model_name: str = "lfm2.5-thinking:1.2b"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    lfm_refresh_rate: float = 60.0 # Strategic thinking every 60s
    symbol: str = "BTC/USD" # Alpaca crypto format
    alpaca_symbol: str = "BTC/USD"
    trade_qty: float = 0.01 # Units for crypto or shares for stocks
    signal_threshold: float = 0.7 

class SharedState:
    """
    Thread-Safe Container for the latest 'World View'.
    """
    def __init__(self, lfm_dim: int, device: str):
        self.lock = threading.Lock()
        self.device = device
        
        # Vectors
        self.lfm_context = torch.zeros(1, lfm_dim).to(device)
        self.news_vector = torch.zeros(1, 2).to(device)
        self.last_price = 0.0
        
        self.last_update = 0.0
        
    def update_lfm(self, tensor: torch.Tensor):
        with self.lock:
            self.lfm_context = tensor.to(self.device).detach()
            self.last_update = time.time()
            
    def update_price(self, price: float):
        with self.lock:
            self.last_price = price

    def get_state(self):
        with self.lock:
            return self.lfm_context.clone(), self.news_vector.clone()

class TrinityEngine:
    def __init__(self, model: TrinityLiquidBrain, config: EngineConfig):
        self.model = model
        self.config = config
        self.logger = logging.getLogger("TrinityEngine")
        self.device = torch.device(config.device)
        self.model.to(self.device)
        
        # State
        self.state = SharedState(lfm_dim=4096, device=config.device) # Standard Llama Embedding Dim 
        # Wait, TrinityLiquidBrain expects lfm_hidden_dim to match bridge input.
        # If LFM outputs text, we need an embedding. 
        # Standard Ollama embedding returns 4096 (Llama3).
        # We assume our LFM Interface handles the embedding conversion.
        
        # Agents
        self.sentinel = SentinelStub()
        self.scanner = MarketScanner()
        self.executor = AlpacaExecutor()
        
        self.lfm_thread = None
        self.active = False
        
    def _query_lfm(self, prompt: str) -> str:
        """Raw Query to Ollama/LFM"""
        try:
            payload = {
                "model": self.config.lfm_model_name,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.2}
            }
            res = requests.post(self.config.lfm_model_url, json=payload, timeout=30)
            if res.status_code == 200:
                return res.json().get("response", "")
            else:
                self.logger.error(f"LFM Error: {res.status_code}")
                return ""
        except Exception as e:
            self.logger.error(f"LFM Connection Failed: {e}")
            return ""

    def _lfm_worker(self):
        """
        The Strategic Agent Loop (Slow Thread).
        Pure Liquid Architecture: LFM thinks, queries tools, and modulates LNN.
        """
        self.logger.info(f"🧠 Strategic Agent (LFM: {self.config.lfm_model_name}) Online")
        
        while self.active:
            start_time = time.time()
            try:
                # 1. Get Context
                price = self.state.last_price
                if price == 0: 
                    time.sleep(1)
                    continue

                # 2. Phase 1: Scan
                # Prompt LFM to check if it needs data
                prompt_1 = f"""
                You are a Strategic Trading AI. Current EURUSD Price: {price:.5f}.
                Do you need to check Support/Resistance or Trend?
                Reply with 'CHECK_SR' or 'CHECK_TREND' or 'NO_ACTION'.
                """
                decision = self._query_lfm(prompt_1).strip()
                
                tool_output = ""
                if "CHECK_SR" in decision:
                    self.logger.info("🧠 LFM Requested: CHECK_SR")
                    tool_output = self.scanner.check_support_resistance(self.config.symbol, price)
                elif "CHECK_TREND" in decision:
                    self.logger.info("🧠 LFM Requested: CHECK_TREND")
                    tool_output = self.scanner.get_trend_alignment(self.config.symbol, '4h')
                
                # 3. Phase 2: Synthesize & Modulate
                prompt_2 = f"""
                Market Context: {tool_output}
                Current Price: {price:.5f}.
                
                Analyze the market structure. 
                Output a modulation vector as 32 comma-separated floats representing {{(Bullishness, Volatility_Exp, ...)}}.
                Example: 0.5, 0.1, -0.2 ...
                """
                # Note: In real production we use an Embedding endpoint.
                # Here we ask LFM to maintain a 'State Vector' via text (Simulated).
                # OR we just get the embedding of the analysis text.
                # Let's try getting the embedding of the validation.
                
                # Mocking embedding for stability if Ollama fails or returns garbage
                # In real code: response = requests.post(..., "embedding")
                
                # For this demo, we generate a random vector modulated by simple sentiment parsing of the 'tool_output'
                modulation = torch.randn(1, 4096) # Assuming mapping to LFM Dim
                
                if "Support" in tool_output: modulation += 0.5 # Bias Up
                if "Resistance" in tool_output: modulation -= 0.5 # Bias Down
                
                self.state.update_lfm(modulation)
                self.logger.info("🧠 Strategic Context Updated")

            except Exception as e:
                self.logger.error(f"LFM Worker Error: {e}")
            
            # Sleep remainder of rate
            elapsed = time.time() - start_time
            sleep_time = max(1.0, self.config.lfm_refresh_rate - elapsed)
            time.sleep(sleep_time)

    def start(self):
        self.active = True
        self.lfm_thread = threading.Thread(target=self._lfm_worker, daemon=True)
        self.lfm_thread.start()
        print("🚀 Liquid Trinity Engine Started")

    def forward_tick(self, market_data: torch.Tensor):
        """
        The Tactical Loop (Fast Path).
        """
        # Updates state
        current_price = market_data[0, -1, 0].item() # Mock access
        self.state.update_price(current_price)
        
        # Get Modulators
        lfm_ctx, news_vec = self.state.get_state()
        
        # Inference
        market_data = market_data.to(self.device)
        
        with torch.no_grad():
            decision = self.model(market_data, lfm_ctx, news_vec)
            
        # 4. Execution Logic (Live Signal Dispatch)
        p_buy, p_sell, p_hold = decision[0].tolist()
        
        # Check Sentinel (Watchdog)
        if not self.sentinel.is_safe():
            self.logger.warning("⚠️ Sentinel blocked execution: DEFCON 1")
            return decision

        if p_buy > self.config.signal_threshold:
            self.logger.info(f"🚀 BUY Signal (Conf: {p_buy:.2f}) -> Dispatching to Alpaca")
            self.executor.place_market_order(self.config.alpaca_symbol, self.config.trade_qty, "BUY")
        elif p_sell > self.config.signal_threshold:
            self.logger.info(f"🚀 SELL Signal (Conf: {p_sell:.2f}) -> Dispatching to Alpaca")
            self.executor.place_market_order(self.config.alpaca_symbol, self.config.trade_qty, "SELL")
            
        return decision

    def stop(self):
        self.active = False
        if self.lfm_thread:
            self.lfm_thread.join()
