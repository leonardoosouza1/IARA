"""
I.A.R.A. Database Ingestion
===========================
Reads Forex CSV data (M1) and populates the TimescaleDB 'market_candles' table.
Crucial for the Strategic Agent's 'MarketScanner' tool.

Usage:
    python src/data/ingest_db.py --file data/raw/EURUSD.csv --symbol EURUSD
"""

import os
import argparse
import pandas as pd
from sqlalchemy import create_engine, text
from datetime import datetime
import logging
from dotenv import load_dotenv

# Load env vars
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# DB Connection
DB_URI = os.getenv("DB_URI")

def ingest_csv(file_path: str, symbol: str):
    if not DB_URI:
        logger.error("DB_URI environment variable not set.")
        return

    logger.info(f"📂 Reading {file_path}...")
    try:
        # Load CSV
        df = pd.read_csv(file_path)
        
        # Normalize columns
        df.columns = [c.lower().strip() for c in df.columns]
        
        # Ensure timestamp
        if "date" in df.columns:
            df["time"] = pd.to_datetime(df["date"])
        elif "timestamp" in df.columns:
            df["time"] = pd.to_datetime(df["timestamp"])
            
        # Ensure required cols
        # req_cols = ["time", "open", "high", "low", "close", "volume"] # validation logic can be added here
        
        df["symbol"] = symbol
        df["interval"] = "1m" # Assuming M1 data
        df["source"] = "histdata"
        
        # Filter for DB Schema
        db_cols = ["time", "symbol", "interval", "open", "high", "low", "close", "volume", "source"]
        
        # Rename if necessary
        if "tick_volume" in df.columns:
            df = df.rename(columns={"tick_volume": "volume"})
            
        # Select only existing columns that match schema
        available_cols = [c for c in db_cols if c in df.columns]
        df = df[available_cols]
        
        logger.info(f"🚀 Pushing {len(df)} rows to TimescaleDB...")
        engine = create_engine(DB_URI)
        
        # Batch insert using pandas
        df.to_sql("market_candles", engine, if_exists="append", index=False, chunksize=10000, method='multi')
        
        logger.info(f"✅ Successfully ingested {symbol} data!")
        
    except Exception as e:
        logger.error(f"❌ Ingestion Failed: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True, help="Path to CSV file")
    parser.add_argument("--symbol", required=True, help="Symbol (e.g. EURUSD)")
    args = parser.parse_args()
    
    ingest_csv(args.file, args.symbol)
