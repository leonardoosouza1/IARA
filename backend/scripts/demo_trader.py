import time
import logging
import asyncio
from datetime import datetime
import pandas as pd

from src.brain.engine import TrinityEngine, EngineConfig
from src.dashboard.services.alpaca_data_service import AlpacaDataService
from src.database.core import Base, init_db

# Setup Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("DemoTrader")

async def run_demo_trader():
    """
    Continuous Live Trading Loop with Alpaca (Paper Money).
    """
    logger.info("Initializing Alpaca Demo Trader...")
    
    # Ensure DB tables exist
    await init_db()
    
    # Configure Engine
    # Using BTC/USD as the default pair from our Implementation Plan
    from pathlib import Path
    config = EngineConfig(
        symbol="BTC/USD",
        execution_venue="ALPACA", # This ensures engine uses AlpacaExecutor
        trade_qty=0.01,
        checkpoint_path=Path("logs/models/best_model_v9.pth")
    )
    
    logger.info(f"Loading Trinity Engine Brain from: {config.checkpoint_path}")
    try:
        engine = TrinityEngine.from_checkpoint(config)
    except Exception as e:
        logger.error(f"Failed to load checkpoint. Make sure the model exists: {e}")
        return

    engine.start() # Start LFM background thread and initialization
    
    # Data Service
    data_svc = AlpacaDataService()
    if not data_svc.is_configured:
        logger.error("Alpaca Data Service not configured. Check .env")
        engine.stop()
        return

    # 1. Warm-up Buffer
    logger.info("🔥 Warming up brain buffer with historical M5 candles...")
    df_history = data_svc.get_bars(symbol=config.symbol, timeframe="5Min", limit=300)
    
    if df_history.empty:
        logger.error("Failed to fetch historical data for warmup.")
        engine.stop()
        return

    # Feed history sequentially so buffer is full
    for _, row in df_history.iterrows():
        candle = {
            "time": row["time"].timestamp(), # Convert to unix float
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": float(row["volume"])
        }
        engine.buffer.add_candle(candle)
        
    logger.info(f"✅ Buffer warmed up with {len(engine.buffer.data)} candles.")

    # 2. Live Loop
    logger.info(f"🔄 Starting Live Polling Loop for {config.symbol} (5 Min intervals)")
    last_processed_time = df_history.iloc[-1]["time"].timestamp()
    
    try:
        while True:
            # Sleep 1 minute before polling (to avoid rate limits and check continuously)
            await asyncio.sleep(60)
            
            # Fetch latest handful of bars to ensure we catch any closures
            df_live = data_svc.get_bars(symbol=config.symbol, timeframe="5Min", limit=5)
            if df_live.empty:
                continue
                
            # Filter only new candles
            new_rows = []
            for _, row in df_live.iterrows():
                row_time = row["time"].timestamp()
                if row_time > last_processed_time:
                    new_rows.append((row_time, row))
            
            # Process new candles
            for row_time, row in new_rows:
                candle = {
                    "time": row_time,
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                    "volume": float(row["volume"])
                }
                
                logger.info(f"⚡ Processing New M5 Bar: {datetime.fromtimestamp(row_time)}")
                signal = await engine.forward_tick(candle)
                
                if signal:
                    logger.info(f"🧭 Signal: {signal.direction} (Conf: {signal.confidence:.2f})")
                
                last_processed_time = row_time
                
    except asyncio.CancelledError:
        logger.info("Demo Trader stopping...")
    except KeyboardInterrupt:
        logger.info("Demo Trader manually stopped.")
    finally:
        engine.stop()

if __name__ == "__main__":
    asyncio.run(run_demo_trader())
