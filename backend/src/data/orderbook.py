"""
I.A.R.A. v11 — Order Book Microstructure Ingestor
===================================================
Real-time Level 2 Order Book data from Binance Depth WebSocket.

Computes:
  - OBI (Order Book Imbalance): Bid vs Ask volume pressure
  - VPIN (Volume-Synchronized Probability of Informed Trading): Toxicity metric
  - Spread %: Relative bid-ask spread
  - Depth Ratio: Deep bid vs ask liquidity
  - Bid/Ask Walls: Large resting orders detection

The "center of gravity" of the market — reveals intent before price moves.
"""

import asyncio
import json
import logging
import threading
import time
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("OrderBookIngestor")


class OrderBookSnapshot:
    """Single snapshot of the order book state."""

    __slots__ = ["timestamp", "bids", "asks"]

    def __init__(self, timestamp: float, bids: List[List[float]], asks: List[List[float]]):
        self.timestamp = timestamp
        self.bids = bids  # [[price, qty], ...] sorted best (highest) first
        self.asks = asks  # [[price, qty], ...] sorted best (lowest) first


class OrderBookIngestor:
    """
    Binance Depth WebSocket Ingestor.

    Connects to the partial book depth stream and maintains a rolling buffer
    of snapshots. Computes LOB microstructure features on demand.

    Usage:
        ingestor = OrderBookIngestor(symbol="btcusdt")
        ingestor.start()  # Launches background WebSocket thread

        features = ingestor.get_features()  # [obi, vpin, spread_pct, depth_ratio, bid_wall, ask_wall]

        ingestor.stop()
    """

    def __init__(
        self,
        symbol: str = "btcusdt",
        depth_levels: int = 20,
        buffer_size: int = 500,
        vpin_buckets: int = 50,
        vpin_bucket_volume: float = 1.0,
    ):
        self.symbol = symbol.lower()
        self.depth_levels = depth_levels
        self.buffer_size = buffer_size
        self.vpin_buckets = vpin_buckets
        self.vpin_bucket_volume = vpin_bucket_volume

        # Rolling buffer of snapshots
        self._snapshots: deque = deque(maxlen=buffer_size)

        # VPIN state
        self._vpin_buy_volume: deque = deque(maxlen=vpin_buckets)
        self._vpin_sell_volume: deque = deque(maxlen=vpin_buckets)
        self._current_bucket_buy = 0.0
        self._current_bucket_sell = 0.0
        self._current_bucket_total = 0.0
        self._last_mid_price = 0.0

        # Thread management
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # Cached features (updated on each snapshot)
        self._cached_features = np.zeros(6, dtype=np.float32)

    def start(self):
        """Start the background WebSocket listener thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run_ws_loop, daemon=True)
        self._thread.start()
        logger.info(f"📦 LOB Ingestor started for {self.symbol.upper()}")

    def stop(self):
        """Stop the WebSocket listener."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("📦 LOB Ingestor stopped")

    def _run_ws_loop(self):
        """Run the asyncio WebSocket event loop in a background thread."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._ws_listener())
        except Exception as e:
            logger.error(f"WebSocket loop error: {e}")
        finally:
            loop.close()

    async def _ws_listener(self):
        """Connect to Binance and process depth updates."""
        try:
            import websockets
        except ImportError:
            logger.error("websockets package not installed. LOB Ingestor disabled.")
            self._running = False
            return

        url = f"wss://stream.binance.com:9443/ws/{self.symbol}@depth{self.depth_levels}@100ms"
        logger.info(f"Connecting to {url}")

        while self._running:
            try:
                async with websockets.connect(url, ping_interval=20) as ws:
                    logger.info("✅ LOB WebSocket connected")
                    while self._running:
                        try:
                            msg = await asyncio.wait_for(ws.recv(), timeout=5.0)
                            data = json.loads(msg)
                            self._process_depth_update(data)
                        except asyncio.TimeoutError:
                            continue
                        except Exception as e:
                            logger.warning(f"WS message error: {e}")
                            break
            except Exception as e:
                logger.warning(f"WS connection error: {e}. Reconnecting in 5s...")
                await asyncio.sleep(5)

    def _process_depth_update(self, data: dict):
        """Process a depth update message and compute features."""
        try:
            bids = [[float(p), float(q)] for p, q in data.get("bids", [])]
            asks = [[float(p), float(q)] for p, q in data.get("asks", [])]

            if not bids or not asks:
                return

            snapshot = OrderBookSnapshot(
                timestamp=time.time(),
                bids=bids,
                asks=asks,
            )

            with self._lock:
                self._snapshots.append(snapshot)
                self._update_vpin(bids, asks)
                self._cached_features = self._compute_features_internal(snapshot)

        except Exception as e:
            logger.debug(f"Depth processing error: {e}")

    def _update_vpin(self, bids: list, asks: list):
        """Update VPIN buckets based on mid-price changes."""
        if not bids or not asks:
            return

        mid_price = (bids[0][0] + asks[0][0]) / 2.0

        if self._last_mid_price > 0:
            price_change = mid_price - self._last_mid_price
            # Approximate volume classification using price direction
            total_volume = sum(q for _, q in bids[:5]) + sum(q for _, q in asks[:5])
            if total_volume > 0:
                if price_change > 0:
                    self._current_bucket_buy += total_volume * 0.1
                else:
                    self._current_bucket_sell += total_volume * 0.1
                self._current_bucket_total += total_volume * 0.1

                # Check if bucket is full
                if self._current_bucket_total >= self.vpin_bucket_volume:
                    self._vpin_buy_volume.append(self._current_bucket_buy)
                    self._vpin_sell_volume.append(self._current_bucket_sell)
                    self._current_bucket_buy = 0.0
                    self._current_bucket_sell = 0.0
                    self._current_bucket_total = 0.0

        self._last_mid_price = mid_price

    def _compute_features_internal(self, snapshot: OrderBookSnapshot) -> np.ndarray:
        """Compute 6 LOB microstructure features from current snapshot."""
        bids = snapshot.bids
        asks = snapshot.asks

        if not bids or not asks:
            return np.zeros(6, dtype=np.float32)

        # 1. OBI — Order Book Imbalance (top N levels)
        bid_vol = sum(q for _, q in bids[:10])
        ask_vol = sum(q for _, q in asks[:10])
        total_vol = bid_vol + ask_vol
        obi = (bid_vol - ask_vol) / (total_vol + 1e-10)

        # 2. VPIN — Volume-Synchronized Probability of Informed Trading
        vpin = 0.0
        if len(self._vpin_buy_volume) >= 5:
            buy_arr = np.array(list(self._vpin_buy_volume))
            sell_arr = np.array(list(self._vpin_sell_volume))
            total_arr = buy_arr + sell_arr + 1e-10
            vpin = float(np.mean(np.abs(buy_arr - sell_arr) / total_arr))

        # 3. Spread % — Relative bid-ask spread
        best_bid = bids[0][0]
        best_ask = asks[0][0]
        mid_price = (best_bid + best_ask) / 2.0
        spread_pct = (best_ask - best_bid) / (mid_price + 1e-10) * 100

        # 4. Depth Ratio (deep levels 5-20)
        deep_bid = sum(q for _, q in bids[5:]) if len(bids) > 5 else 0
        deep_ask = sum(q for _, q in asks[5:]) if len(asks) > 5 else 0
        deep_total = deep_bid + deep_ask
        depth_ratio = (deep_bid - deep_ask) / (deep_total + 1e-10)

        # 5. Bid Wall Detection (single large order > 3x mean)
        bid_qtys = np.array([q for _, q in bids]) if bids else np.zeros(1)
        bid_mean = np.mean(bid_qtys)
        bid_wall = float(np.max(bid_qtys) / (bid_mean + 1e-10) > 3.0)

        # 6. Ask Wall Detection (single large order > 3x mean)
        ask_qtys = np.array([q for _, q in asks]) if asks else np.zeros(1)
        ask_mean = np.mean(ask_qtys)
        ask_wall = float(np.max(ask_qtys) / (ask_mean + 1e-10) > 3.0)

        # Normalize all to [-1, 1] or [0, 1]
        features = np.array([
            np.clip(obi, -1.0, 1.0),                   # OBI: already in [-1, 1]
            np.clip(vpin * 2 - 1, -1.0, 1.0),          # VPIN: [0,1] → [-1,1]
            np.clip(spread_pct * 10 - 1, -1.0, 1.0),   # Spread: normalized
            np.clip(depth_ratio, -1.0, 1.0),            # Depth: already in [-1, 1]
            bid_wall,                                    # Binary: 0 or 1
            ask_wall,                                    # Binary: 0 or 1
        ], dtype=np.float32)

        return features

    def get_features(self) -> np.ndarray:
        """
        Get the latest LOB microstructure features.

        Returns:
            np.ndarray of shape [6]:
              [obi, vpin, spread_pct, depth_ratio, bid_wall, ask_wall]
        """
        with self._lock:
            return self._cached_features.copy()

    def get_features_tensor(self):
        """Get features as a PyTorch tensor [1, 6]."""
        import torch
        features = self.get_features()
        return torch.tensor(features, dtype=torch.float32).unsqueeze(0)

    @property
    def is_connected(self) -> bool:
        return self._running and len(self._snapshots) > 0

    @property
    def snapshot_count(self) -> int:
        return len(self._snapshots)


class MockOrderBookIngestor(OrderBookIngestor):
    """
    Mock ingestor for offline/testing mode.
    Generates realistic synthetic LOB features.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._mock_step = 0

    def start(self):
        """No-op for mock mode."""
        self._running = True
        logger.info("📦 LOB Mock Ingestor started (synthetic data)")

    def stop(self):
        self._running = False

    def get_features(self) -> np.ndarray:
        """Generate synthetic LOB features with realistic distributions."""
        self._mock_step += 1
        t = self._mock_step * 0.01

        obi = np.sin(t * 0.7) * 0.3 + np.random.normal(0, 0.1)
        vpin = 0.3 + np.sin(t * 0.2) * 0.15 + np.random.normal(0, 0.05)
        spread = 0.01 + abs(np.random.normal(0, 0.005))
        depth = np.sin(t * 0.3) * 0.4 + np.random.normal(0, 0.15)
        bid_wall = float(np.random.random() > 0.85)
        ask_wall = float(np.random.random() > 0.85)

        return np.array([
            np.clip(obi, -1.0, 1.0),
            np.clip(vpin * 2 - 1, -1.0, 1.0),
            np.clip(spread * 10 - 1, -1.0, 1.0),
            np.clip(depth, -1.0, 1.0),
            bid_wall,
            ask_wall,
        ], dtype=np.float32)
