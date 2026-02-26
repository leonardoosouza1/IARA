import asyncio
import logging
import os
from typing import Dict, List, Optional
import numpy as np
from binance import AsyncClient, BinanceSocketManager

logger = logging.getLogger("SensoryHub")

class BinanceSensoryHub:
    """
    I.A.R.A. v10.0 - Sensory Hub (RAIO-X)
    =====================================
    Handles real-time L2 Order Book streaming via Binance WebSockets.
    Processes depth data into model-ready sensory vectors.
    """
    
    def __init__(self, api_key: str, secret_key: str):
        self.api_key = api_key
        self.secret_key = secret_key
        self.client: Optional[AsyncClient] = None
        self.bsm: Optional[BinanceSocketManager] = None
        self.active_symbols: List[str] = []
        self.active = False
        
        # Latest L2 Snapshots: { "BTCUSDT": { "bids": [...], "asks": [...], "vbi": 0.0 } }
        self.l2_cache: Dict[str, dict] = {}
        
        # History for Spoofing detection
        self.prev_l2: Dict[str, dict] = {}
        self.cancel_rate: Dict[str, float] = {} # { "BTCUSDT": 0.0 }
        
    async def start(self, symbols: List[str]):
        """Initialize and run the WebSocket stream."""
        if not self.api_key or not self.secret_key:
            logger.error("Binance keys missing! Cannot start Sensory Hub.")
            return

        self.active_symbols = [s.replace("/", "").replace("-", "") for s in symbols]
        self.client = await AsyncClient.create(self.api_key, self.secret_key)
        self.bsm = BinanceSocketManager(self.client)
        self.active = True
        
        # Run in background
        asyncio.create_task(self._stream_manager())

    async def _stream_manager(self):
        """Manages the Multiplex depth stream."""
        # Binance format for depth: <symbol>@depth20@100ms
        streams = [f"{sym.lower()}@depth20@100ms" for sym in self.active_symbols]
        for sym in self.active_symbols:
            logger.info(f"📡 Sensory Hub: Subscribed to Order Book for {sym}")
            
        while self.active:
            try:
                async with self.bsm.multiplex_socket(streams) as stream:
                    while self.active:
                        res = await stream.recv()
                        if 'data' in res:
                            await self._on_orderbook(res['data'])
            except Exception as e:
                logger.error(f"WebSocket Error (Sensory Hub): {e}. Reconnecting...")
                await asyncio.sleep(5)

    async def _on_orderbook(self, data: dict):
        """Callback for incoming orderbook updates."""
        # Binance depth payload structure
        # 'bids': [ [ "Price", "Qty" ], ... ]
        symbol = data.get('s', '').upper()
        if not symbol: return
        
        # Process top 10 levels
        bids = data.get('bids', [])[:10]
        asks = data.get('asks', [])[:10]
        
        # Calculate VBI (Volume Book Imbalance)
        # VBI = (BidVol - AskVol) / (BidVol + AskVol)
        try:
             bid_vol = sum(float(b[1]) for b in bids)
             ask_vol = sum(float(a[1]) for a in asks)
             vbi = (bid_vol - ask_vol) / (bid_vol + ask_vol + 1e-10)
        except Exception:
             vbi = 0.0
             
        # Update Cancel Rate (Spoofing Proxy)
        # Parse bids into {price: qty} floats
        float_bids = [(float(b[0]), float(b[1])) for b in bids]
        float_asks = [(float(a[0]), float(a[1])) for a in asks]
        
        self._update_cancel_rate(symbol, float_bids, float_asks)
        
        # Store in cache
        self.l2_cache[symbol] = {
            "bids": float_bids,
            "asks": float_asks,
            "vbi": vbi,
            "cancel_rate": self.cancel_rate.get(symbol, 0.0),
            "timestamp": data.get('E', 0)
        }
        
        # Backup for next diff
        self.prev_l2[symbol] = {
            "bids": {b[0]: b[1] for b in float_bids},
            "asks": {a[0]: a[1] for a in float_asks}
        }
        
        # Optional: Log anomaly if VBI is extreme (Spoofing/Manipulation detection)
        if abs(vbi) > 0.8:
            pass # logger.warning(f"⚠️ Extreme Order Book Imbalance on {symbol}: {vbi:.2f}")

    def _update_cancel_rate(self, symbol: str, bids: List[tuple], asks: List[tuple]):
        """Estimate cancellation rates by comparing level volume shifts."""
        if symbol not in self.prev_l2:
            return
            
        prev = self.prev_l2[symbol]
        curr_bids = {b[0]: b[1] for b in bids}
        curr_asks = {a[0]: a[1] for a in asks}
        
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
        # Engine might ask for BTC/USD format, map it internally
        binance_symbol = symbol.replace("/", "").replace("-", "").upper()
            
        data = self.l2_cache.get(binance_symbol)
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
        self.active = False
        if self.client:
           await self.client.close_connection()
        logger.info("📡 Sensory Hub Stopped")

# Singleton (or initialized in MarketDataService)
sensory_hub = None # Initialized after keys are available
