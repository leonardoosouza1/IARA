
import asyncio
import logging
import os
from typing import Dict, List, Optional
import numpy as np
from alpaca.data.live import CryptoDataStream
from alpaca.data.models import Orderbook

logger = logging.getLogger("SensoryHub")

class AlpacaSensoryHub:
    """
    I.A.R.A. v10.0 - Sensory Hub (RAIO-X)
    =====================================
    Handles real-time L2 Order Book streaming via Alpaca WebSockets.
    Processes depth data into model-ready sensory vectors.
    """
    
    def __init__(self, api_key: str, secret_key: str):
        self.api_key = api_key
        self.secret_key = secret_key
        self.stream: Optional[CryptoDataStream] = None
        self.active_symbols: List[str] = []
        
        # Latest L2 Snapshots: { "BTC/USD": { "bids": [...], "asks": [...], "vbi": 0.0 } }
        self.l2_cache: Dict[str, dict] = {}
        
        # History for Spoofing detection
        self.prev_l2: Dict[str, dict] = {}
        self.cancel_rate: Dict[str, float] = {} # { "BTC/USD": 0.0 }
        
    async def start(self, symbols: List[str]):
        """Initialize and run the WebSocket stream."""
        if not self.api_key or not self.secret_key:
            logger.error("Alpaca keys missing! Cannot start Sensory Hub.")
            return

        self.active_symbols = symbols
        self.stream = CryptoDataStream(self.api_key, self.secret_key)
        
        # Subscribe to Order Books
        for symbol in symbols:
            # Note: Alpaca-py expects symbols like BTC/USD or BTC/USDT depending on the provider
            # We will use the provided symbols directly
            self.stream.subscribe_crypto_orderbooks(self._on_orderbook, symbol)
            logger.info(f"📡 Sensory Hub: Subscribed to Order Book for {symbol}")

        # Run in background
        asyncio.create_task(self.stream._run_forever())

    async def _on_orderbook(self, data: Orderbook):
        """Callback for incoming orderbook updates."""
        symbol = data.symbol
        
        # Process top 10 levels
        bids = data.bids[:10]
        asks = data.asks[:10]
        
        # Calculate VBI (Volume Book Imbalance)
        # VBI = (BidVol - AskVol) / (BidVol + AskVol)
        bid_vol = sum(b.size for b in bids)
        ask_vol = sum(a.size for a in asks)
        vbi = (bid_vol - ask_vol) / (bid_vol + ask_vol + 1e-10)
        
        # Update Cancel Rate (Spoofing Proxy)
        self._update_cancel_rate(symbol, bids, asks)
        
        # Store in cache
        self.l2_cache[symbol] = {
            "bids": [[b.price, b.size] for b in bids],
            "asks": [[a.price, a.size] for a in asks],
            "vbi": vbi,
            "cancel_rate": self.cancel_rate.get(symbol, 0.0),
            "timestamp": data.timestamp
        }
        
        # Backup for next diff
        self.prev_l2[symbol] = {
            "bids": {b.price: b.size for b in bids},
            "asks": {a.price: a.size for a in asks}
        }
        
        # Optional: Log anomaly if VBI is extreme (Spoofing/Manipulation detection)
        if abs(vbi) > 0.8:
            logger.warning(f"⚠️ Extreme Order Book Imbalance on {symbol}: {vbi:.2f}")

    def _update_cancel_rate(self, symbol: str, bids: List, asks: List):
        """Estimate cancellation rates by comparing level volume shifts."""
        if symbol not in self.prev_l2:
            return
            
        prev = self.prev_l2[symbol]
        curr_bids = {b.price: b.size for b in bids}
        curr_asks = {a.price: a.size for a in asks}
        
        cancels = 0.0
        total_delta = 0.0
        
        # Check Bids
        for price, prev_size in prev["bids"].items():
            if price in curr_bids:
                delta = prev_size - curr_bids[price]
                if delta > 0: # Volume decreased
                    # In a real engine we'd subtract actual trades at this price
                    # Here we treat any unexplained decrease as a potential cancel
                    cancels += delta
                    total_delta += delta
            else:
                # Level disappeared completely
                cancels += prev_size
                total_delta += prev_size
                
        # Normalize and smooth
        alpha = 0.1 # Smoothing factor
        new_rate = cancels / (sum(prev["bids"].values()) + 1e-10)
        self.cancel_rate[symbol] = (1 - alpha) * self.cancel_rate.get(symbol, 0.0) + alpha * new_rate

    def get_sensory_vector(self, symbol: str) -> Optional[np.ndarray]:
        """
        Returns a normalized sensory vector for the model.
        Dimensions: 10 Bids (P, S) + 10 Asks (P, S) + VBI + CancelRate = 42 features.
        """
        data = self.l2_cache.get(symbol)
        if not data:
            return None
            
        vector = []
        
        # Normalize relative to the mid-price
        best_bid = data["bids"][0][0] if data["bids"] else 0
        best_ask = data["asks"][0][0] if data["asks"] else 0
        mid_price = (best_bid + best_ask) / 2 if (best_bid and best_ask) else (best_bid or best_ask)
        
        if mid_price == 0: return None

        # Process Bids
        for price, size in data["bids"]:
            vector.append((price - mid_price) / mid_price * 1000) # Price as bps
            vector.append(size) 
            
        # Pad if less than 10
        while len(vector) < 20: vector.extend([0, 0])
        
        # Process Asks
        for price, size in data["asks"]:
            vector.append((price - mid_price) / mid_price * 1000)
            vector.append(size)
            
        # Pad if less than 10
        while len(vector) < 40: vector.extend([0, 0])
        
        # VBI & CancelRate
        vector.append(data["vbi"])
        vector.append(data["cancel_rate"])
        
        return np.array(vector, dtype=np.float32)

    async def stop(self):
        if self.stream:
            await self.stream.stop()
            logger.info("📡 Sensory Hub Stopped")

# Singleton (or initialized in MarketDataService)
sensory_hub = None # Initialized after keys are available
