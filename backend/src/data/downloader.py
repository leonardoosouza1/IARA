"""
Bulk Downloader - Forex Data Ingestion Module
==============================================
Async bulk downloader for Forex pairs using yfinance.
Saves raw OHLCV data as Parquet files partitioned by ticker.
"""

import asyncio
import logging
from pathlib import Path
from typing import List, Optional

import polars as pl
import yfinance as yf
from tqdm import tqdm

# Fix yfinance cache issue
try:
    yf.set_tz_cache_location("/tmp/py-yfinance")
except Exception:
    pass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Forex pairs (Target Majors for v6.1)
FOREX_PAIRS: List[str] = [
    "EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X"
]

# Configuration
DATA_RAW_DIR = Path(__file__).parent.parent.parent / "data" / "raw"
MAX_RETRIES = 3
RETRY_DELAY_BASE = 2  # seconds (exponential backoff)


async def download_ticker(
    ticker: str,
    period: str = "max",
    interval: str = "1d",
    retries: int = MAX_RETRIES
) -> Optional[pl.DataFrame]:
    """
    Download historical data for a single ticker with retry logic.
    
    Args:
        ticker: Yahoo Finance ticker symbol (e.g., "EURUSD=X")
        period: Data period (e.g., "max", "5y", "1y")
        interval: Data interval (e.g., "1d", "1h")
        retries: Number of retry attempts
        
    Returns:
        Polars DataFrame with OHLCV data or None if failed
    """
    for attempt in range(retries):
        try:
            # yfinance is not truly async, run in executor
            loop = asyncio.get_event_loop()
            df_pandas = await loop.run_in_executor(
                None,
                lambda: yf.Ticker(ticker).history(period=period, interval=interval)
            )
            
            if df_pandas.empty:
                logger.warning(f"{ticker}: Empty DataFrame received")
                return None
            
            # Reset index to get Date as column
            df_pandas = df_pandas.reset_index()
            
            # Convert to Polars
            df = pl.from_pandas(df_pandas)
            
            # Standardize column names
            rename_map = {
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume"
            }
            
            # Handle Date vs Datetime
            if "Date" in df.columns:
                rename_map["Date"] = "date"
            elif "Datetime" in df.columns:
                rename_map["Datetime"] = "date"
                
            # Filter map to only existing columns to avoid Polars errors
            rename_map = {k: v for k, v in rename_map.items() if k in df.columns}
            
            df = df.rename(rename_map)
            
            # Select only needed columns
            available_cols = df.columns
            cols_to_keep = [c for c in ["date", "open", "high", "low", "close", "volume"] 
                          if c in available_cols]
            df = df.select(cols_to_keep)
            
            # Add ticker column
            df = df.with_columns(pl.lit(ticker.replace("=X", "")).alias("ticker"))
            
            # Validate data
            if df.height == 0:
                logger.warning(f"{ticker}: No valid rows after processing")
                return None
                
            logger.info(f"{ticker}: Downloaded {df.height} rows")
            return df
            
        except Exception as e:
            delay = RETRY_DELAY_BASE ** (attempt + 1)
            logger.warning(f"{ticker}: Attempt {attempt + 1} failed - {e}. Retrying in {delay}s...")
            await asyncio.sleep(delay)
    
    logger.error(f"{ticker}: All {retries} attempts failed")
    return None


async def download_all(
    tickers: List[str] = FOREX_PAIRS,
    period: str = "max",
    interval: str = "1d",
    max_concurrent: int = 5
) -> dict[str, pl.DataFrame]:
    """
    Download all tickers concurrently with rate limiting.
    
    Args:
        tickers: List of ticker symbols
        period: Data period
        interval: Data interval
        max_concurrent: Maximum concurrent downloads
        
    Returns:
        Dictionary mapping ticker to DataFrame
    """
    semaphore = asyncio.Semaphore(max_concurrent)
    results = {}
    
    async def download_with_semaphore(ticker: str):
        async with semaphore:
            df = await download_ticker(ticker, period, interval)
            if df is not None:
                results[ticker] = df
    
    tasks = [download_with_semaphore(t) for t in tickers]
    
    # Use tqdm for progress
    for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Downloading"):
        await coro
    
    return results


def save_to_parquet(dataframes: dict[str, pl.DataFrame], output_dir: Path = DATA_RAW_DIR):
    """
    Save DataFrames as Parquet files partitioned by ticker.
    
    Args:
        dataframes: Dictionary mapping ticker to DataFrame
        output_dir: Output directory path
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for ticker, df in dataframes.items():
        ticker_clean = ticker.replace("=X", "")
        output_path = output_dir / f"{ticker_clean}.parquet"
        
        df.write_parquet(output_path)
        logger.info(f"Saved {ticker_clean} to {output_path}")


async def run_ingestion(
    tickers: List[str] = FOREX_PAIRS,
    period: str = "60d",
    interval: str = "5m"
):
    """
    Main ingestion pipeline.
    
    Args:
        tickers: List of ticker symbols
        period: Data period
        interval: Data interval
    """
    logger.info(f"Starting ingestion for {len(tickers)} tickers")
    
    # Download all data
    dataframes = await download_all(tickers, period, interval)
    
    logger.info(f"Successfully downloaded {len(dataframes)}/{len(tickers)} tickers")
    
    # Save to parquet
    save_to_parquet(dataframes)
    
    logger.info("Ingestion complete!")
    
    return dataframes


def main():
    """CLI entry point."""
    print("Starting massive data download (60d 5m)... This may take a minute.")
    asyncio.run(run_ingestion())


if __name__ == "__main__":
    main()
