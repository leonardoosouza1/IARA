import asyncio
import logging
from src.brain.tools.binance_executor import BinanceExecutor
from src.api.live_ingestor import LiveIngestor

logging.basicConfig(level=logging.INFO)

async def print_candle(interval, candle):
    print(f"[{interval}] Candle Data: {candle}")

async def test_binance():
    print("--- 1. Testing Binance Executor (TESTNET) ---")
    executor = BinanceExecutor("BTCUSDT", mode="TESTNET")
    
    # Force killswitch test
    ping_ok = executor._check_latency_killswitch()
    print(f"Ping OK: {ping_ok}")
    
    # Test execution locally
    order = executor.execute_signal("BUY", 0.95, 95000.0, 0.01)
    print(f"Simulated Order: {order}")

    print("\n--- 2. Testing Live Ingestor (WebSocket) ---")
    ingestor = LiveIngestor("BTCUSDT")
    asyncio.create_task(ingestor.start_streaming(print_candle))
    
    print("Streaming for 10 seconds...")
    await asyncio.sleep(10)
    await ingestor.disconnect()
    print("Live Ingestor test complete.")

if __name__ == "__main__":
    asyncio.run(test_binance())
