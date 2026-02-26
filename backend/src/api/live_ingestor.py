import asyncio
import logging
from typing import Dict, Any, Callable
from binance import AsyncClient, BinanceSocketManager
import numpy as np
import collections

logger = logging.getLogger("api.live_ingestor")

class LiveIngestor:
    """
    HFT Live Ingestor (Phase 5)
    Connects to Binance WebSockets to stream M5 and H1 candles,
    along with raw order flow to compute local Cumulative Volume Delta (CVD).
    """
    def __init__(self, symbol: str = "BTCUSDT"):
        self.symbol = symbol.upper()
        self.client: AsyncClient = None
        self.bsm: BinanceSocketManager = None
        self.active = False
        
        # Real-time state
        self.current_m5: Dict[str, Any] = None
        self.current_h1: Dict[str, Any] = None
        
        # CVD (Cumulative Volume Delta) Tracking
        self.cvd_period = 300 # Reset CVD every 5 minutes (300s)
        self.current_cvd = 0.0
        self.last_cvd_reset = 0
        
        # Callbacks for Engine feeding
        self.on_candle_close: Callable = None

    async def connect(self):
        """Initializes the Async Binance Client and Socket Manager."""
        self.client = await AsyncClient.create()
        self.bsm = BinanceSocketManager(self.client)
        self.active = True
        logger.info(f"🔗 Live Ingestor connected to Binance for {self.symbol}")

    async def disconnect(self):
        """Cleanly closes the WebSocket connections."""
        self.active = False
        if self.client:
            await self.client.close_connection()
        logger.info("🛑 Live Ingestor disconnected.")

    async def _stream_klines(self, interval: str):
        """Streams OHLCV candles for a specific timeframe."""
        while self.active:
            try:
                # Use multiplex socket for resilience or direct kline socket
                async with self.bsm.kline_socket(symbol=self.symbol, interval=interval) as stream:
                    while self.active:
                        res = await stream.recv()
                        if 'k' in res:
                            kline = res['k']
                            candle_data = {
                                "time": int(kline['t'] / 1000), # Start time
                                "open": float(kline['o']),
                                "high": float(kline['h']),
                                "low": float(kline['l']),
                                "close": float(kline['c']),
                                "volume": float(kline['v']),
                                "is_closed": kline['x']
                            }
                            
                            if interval == AsyncClient.KLINE_INTERVAL_5MINUTE:
                                self.current_m5 = candle_data
                                if candle_data["is_closed"] and self.on_candle_close:
                                    # Inject CVD into the closed candle
                                    candle_data["cvd"] = self.current_cvd
                                    await self.on_candle_close(interval, candle_data)
                                    # Reset local CVD after 5m close
                                    self.current_cvd = 0.0 
                            
                            elif interval == AsyncClient.KLINE_INTERVAL_1HOUR:
                                self.current_h1 = candle_data
                                if candle_data["is_closed"] and self.on_candle_close:
                                    await self.on_candle_close(interval, candle_data)

            except Exception as e:
                logger.error(f"WebSocket Error (Kline {interval}): {e}. Reconnecting in 5s...")
                await asyncio.sleep(5)

    async def _stream_trades(self):
        """Streams raw trades to calculate CVD (Buy Volume - Sell Volume)."""
        while self.active:
            try:
                async with self.bsm.trade_socket(symbol=self.symbol) as stream:
                    while self.active:
                        res = await stream.recv()
                        # 'm' is True if the buyer is the market maker (i.e., someone sold into the bid) -> Bearish
                        is_buyer_maker = res['m']
                        qty = float(res['q'])
                        
                        if is_buyer_maker:
                            self.current_cvd -= qty # Sell pressure
                        else:
                            self.current_cvd += qty # Buy pressure
                            
            except Exception as e:
                logger.error(f"WebSocket Error (Trades): {e}. Reconnecting in 5s...")
                await asyncio.sleep(5)

    async def start_streaming(self, on_candle_close: Callable):
        """Launches the async tasks for M5, H1, and Raw Trade streams."""
        self.on_candle_close = on_candle_close
        
        await self.connect()
        
        # Fire and forget concurrent streams
        asyncio.create_task(self._stream_klines(AsyncClient.KLINE_INTERVAL_5MINUTE))
        asyncio.create_task(self._stream_klines(AsyncClient.KLINE_INTERVAL_1HOUR))
        asyncio.create_task(self._stream_trades())
        
        logger.info(f"🌊 Live Streaming active for {self.symbol} (M5, H1, CVD)")

if __name__ == "__main__":
    # Test script
    logging.basicConfig(level=logging.INFO)
    async def process_tick(interval, candle):
        cvd_msg = f" | CVD: {candle.get('cvd', 0):.2f}" if interval == '5m' else ""
        print(f"[{interval}] Candle Closed at {candle['close']} (Vol: {candle['volume']}){cvd_msg}")
        
    async def main():
        ingestor = LiveIngestor("BTCUSDT")
        await ingestor.start_streaming(process_tick)
        await asyncio.sleep(300) # Run for 5 minutes
        await ingestor.disconnect()
        
    asyncio.run(main())
