#!/usr/bin/env python3
"""
I.A.R.A. Historical Data Downloader
====================================
Downloads forex M1 (1-minute) data from Histdata.com

This script downloads years of historical forex data for training.
Histdata.com provides free M1 data since 2001.

Usage:
    docker compose exec saci-brain python src/data/download_histdata.py --pairs EURUSD GBPUSD --years 2020 2021 2022 2023

Output:
    data/raw/{PAIR}_{YEAR}.csv
"""

import argparse
import logging
import os
import zipfile
from io import BytesIO
from pathlib import Path
from typing import List
import time

import requests
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("DataDownloader")

# Output directory
RAW_DATA_DIR = Path("data/raw")

# Histdata.com download URL pattern
# They use a specific format for their downloads
HISTDATA_BASE_URL = "https://www.histdata.com/download-free-forex-historical-data/?/ascii/1-minute-bar-quotes"

# Alternative: Direct download URLs (these are more reliable)
# Histdata provides data in monthly chunks
HISTDATA_DOWNLOAD_URL = "https://www.histdata.com/download-free-forex-data/?/ascii/1-minute-bar-quotes/{pair}/{year}"


def download_from_dukascopy(pair: str, year: int, month: int) -> pd.DataFrame:
    """
    Download data from Dukascopy (alternative source).
    Note: Dukascopy uses tick data, more complex to process.
    """
    # This is a placeholder - Dukascopy requires more complex handling
    raise NotImplementedError("Dukascopy downloader not implemented yet")


def download_from_forexsb(pair: str, years: List[int]) -> pd.DataFrame:
    """
    Download from Forex Strategy Builder's data server.
    They provide clean CSV files.
    """
    # FSB data URL pattern
    base_url = f"https://forexsb.com/historical-data/{pair.lower()}"
    
    all_data = []
    for year in years:
        url = f"{base_url}/{year}"
        logger.info(f"Trying ForexSB: {url}")
        try:
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                # Parse CSV
                from io import StringIO
                df = pd.read_csv(StringIO(response.text))
                all_data.append(df)
                logger.info(f"  Downloaded {len(df)} rows")
        except Exception as e:
            logger.warning(f"  Failed: {e}")
    
    if all_data:
        return pd.concat(all_data, ignore_index=True)
    return None


def generate_sample_data(pair: str, years: List[int], output_dir: Path) -> Path:
    """
    Generate synthetic forex data for testing when downloads fail.
    This creates realistic-looking price action for development.
    """
    import numpy as np
    
    logger.info(f"Generating synthetic data for {pair} ({years[0]}-{years[-1]})")
    
    # Starting prices for different pairs
    start_prices = {
        'EURUSD': 1.1000,
        'GBPUSD': 1.2500,
        'USDJPY': 110.00,
        'AUDUSD': 0.7500,
    }
    
    base_price = start_prices.get(pair, 1.0)
    
    all_rows = []
    current_price = base_price
    
    for year in years:
        # Generate one year of M15 data (less data, faster processing)
        # ~35k candles per year for M15 (252 trading days * 24 hours * ~6 candles)
        n_candles = 35000
        
        for i in range(n_candles):
            # Random walk with mean reversion
            change_pct = np.random.normal(0, 0.0005)  # 0.05% std dev
            
            # Mean reversion toward base price
            reversion = (base_price - current_price) * 0.001
            change_pct += reversion
            
            # Add some trends
            if np.random.random() < 0.1:  # 10% chance of trend
                trend_direction = np.random.choice([-1, 1])
                change_pct += trend_direction * 0.001
            
            open_price = current_price
            close_price = current_price * (1 + change_pct)
            high_price = max(open_price, close_price) * (1 + abs(np.random.normal(0, 0.0002)))
            low_price = min(open_price, close_price) * (1 - abs(np.random.normal(0, 0.0002)))
            volume = int(np.random.exponential(1000))
            
            # Timestamp
            day_of_year = i // 96  # ~96 candles per day for M15
            hour = (i % 96) // 4
            minute = (i % 4) * 15
            
            timestamp = pd.Timestamp(year=year, month=1, day=1) + pd.Timedelta(days=day_of_year, hours=hour, minutes=minute)
            
            all_rows.append({
                'date': timestamp,
                'open': round(open_price, 5),
                'high': round(high_price, 5),
                'low': round(low_price, 5),
                'close': round(close_price, 5),
                'volume': volume
            })
            
            current_price = close_price
    
    df = pd.DataFrame(all_rows)
    
    # Save to CSV
    output_path = output_dir / f"{pair}_synthetic.csv"
    df.to_csv(output_path, index=False)
    logger.info(f"Generated {len(df)} synthetic candles -> {output_path}")
    
    return output_path


def download_from_yfinance(pair: str, years: List[int]) -> pd.DataFrame:
    """
    Download from Yahoo Finance using yfinance library.
    Note: Yahoo has limited forex data, mainly major pairs.
    """
    try:
        import yfinance as yf
    except ImportError:
        logger.error("yfinance not installed. Run: pip install yfinance")
        return None
    
    # Yahoo Finance forex symbol format
    symbol = f"{pair}=X"
    
    start_date = f"{min(years)}-01-01"
    end_date = f"{max(years) + 1}-01-01"
    
    logger.info(f"Downloading {symbol} from Yahoo Finance ({start_date} to {end_date})")
    
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_date, end=end_date, interval="1h")  # 1h is the minimum for forex
        
        if df.empty:
            logger.warning(f"No data returned for {symbol}")
            return None
        
        # Rename columns to match our format
        df = df.reset_index()
        df = df.rename(columns={
            'Datetime': 'date',
            'Date': 'date',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        })
        
        df = df[['date', 'open', 'high', 'low', 'close', 'volume']]
        logger.info(f"Downloaded {len(df)} candles from Yahoo Finance")
        
        return df
        
    except Exception as e:
        logger.error(f"Yahoo Finance download failed: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Download forex historical data")
    parser.add_argument("--pairs", nargs="+", default=["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"],
                        help="Currency pairs to download")
    parser.add_argument("--years", nargs="+", type=int, default=[2020, 2021, 2022, 2023, 2024],
                        help="Years to download")
    parser.add_argument("--source", choices=["yahoo", "synthetic"], default="yahoo",
                        help="Data source")
    parser.add_argument("--output-dir", type=Path, default=RAW_DATA_DIR,
                        help="Output directory")
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("  I.A.R.A. Historical Data Downloader")
    logger.info("=" * 60)
    logger.info(f"Pairs: {args.pairs}")
    logger.info(f"Years: {args.years}")
    logger.info(f"Source: {args.source}")
    logger.info(f"Output: {args.output_dir}")
    
    downloaded_files = []
    
    for pair in args.pairs:
        logger.info(f"\n--- Processing {pair} ---")
        
        df = None
        
        if args.source == "yahoo":
            df = download_from_yfinance(pair, args.years)
        
        if df is None or args.source == "synthetic":
            # Fallback to synthetic data
            logger.warning(f"Using synthetic data for {pair}")
            output_path = generate_sample_data(pair, args.years, args.output_dir)
            downloaded_files.append(output_path)
            continue
        
        # Save to CSV
        output_path = args.output_dir / f"{pair}.csv"
        df.to_csv(output_path, index=False)
        logger.info(f"Saved {len(df)} candles -> {output_path}")
        downloaded_files.append(output_path)
        
        # Rate limiting
        time.sleep(1)
    
    logger.info("\n" + "=" * 60)
    logger.info("Download complete!")
    logger.info(f"Files: {[str(f) for f in downloaded_files]}")
    logger.info("\nNext steps:")
    logger.info("1. Process data: python src/data/processor.py")
    logger.info("2. Generate cache: python src/data/feature_factory.py")
    logger.info("3. Train: python src/train/trainer_liquid.py --use-cache")


if __name__ == "__main__":
    main()
