import os
import json
import logging
import asyncio
import psycopg2
from datetime import datetime
from dotenv import load_dotenv

# Alpaca-py imports
from alpaca.data.live import CryptoDataStream, StockDataStream
from alpaca.data.enums import DataFeed

load_dotenv()

# Config
API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
DB_URI = os.getenv("DB_URI")
SYMBOL = "BTC/USD"

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AlpacaIngest")

# Database connection
conn = psycopg2.connect(DB_URI)
cur = conn.cursor()

async def crypto_handler(data):
    """
    Handles incoming crypto trade/bar data.
    """
    try:
        symbol = data.symbol
        price = data.price if hasattr(data, 'price') else data.close
        timestamp = data.timestamp
        
        logger.info(f"📊 {symbol} Tick: {price} at {timestamp}")
        
        # Ingest into TimescaleDB
        cur.execute("""
            INSERT INTO market_candles (time, symbol, interval, open, high, low, close, volume)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT DO NOTHING;
        """, (timestamp, symbol.replace("/", ""), 'tick', price, price, price, price, 1))
        conn.commit()
    except Exception as e:
        logger.error(f"DB Error: {e}")

async def main():
    logger.info(f"🚀 Starting Alpaca Live Ingest for {SYMBOL}...")
    
    # Using CryptoDataStream for BTC/USD
    # For stocks, use StockDataStream
    stream = CryptoDataStream(API_KEY, SECRET_KEY)
    
    # Subscribe to trades or bars
    # stream.subscribe_bars(crypto_handler, SYMBOL) # M1 bars if preferred
    stream.subscribe_trades(crypto_handler, SYMBOL)
    
    try:
        await stream._run_forever()
    except Exception as e:
        logger.error(f"Stream Error: {e}")
        await asyncio.sleep(5)
        await main()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Stopping Ingest...")
        cur.close()
        conn.close()
