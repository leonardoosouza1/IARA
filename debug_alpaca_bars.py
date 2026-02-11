import os
from dotenv import load_dotenv
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame
from datetime import datetime, timedelta

load_dotenv()

print("--- Debugging Crypto Bars ---")
try:
    client = CryptoHistoricalDataClient() # No keys needed for crypto typically, but can use them
    
    symbol = "BTC/USD"
    start_time = datetime.now() - timedelta(days=1)
    
    print(f"Fetching bars for {symbol} since {start_time}")
    
    req = CryptoBarsRequest(
        symbol_or_symbols=[symbol],
        timeframe=TimeFrame.Minute,
        start=start_time,
        limit=10
    )
    
    bars = client.get_crypto_bars(req)
    
    print(f"Response Type: {type(bars)}")
    print(f"Dir(bars): {dir(bars)}")
    
    try:
        df = bars.df
        print("\nDataFrame Shape:", df.shape)
        print("DataFrame Head:")
        print(df.head())
        print("DataFrame Columns:", df.columns)
        print("DataFrame Index:", df.index)
    except Exception as e:
        print(f"Error accessing .df: {e}")
        
    # Try accessing as dict/list if .df fails or is empty
    try:
        print("\nRaw Data Access:")
        print(bars[symbol][0] if symbol in bars else "Symbol not found in bars")
    except Exception as e:
        print(f"Error accessing raw data: {e}")

except Exception as e:
    print(f"CRITICAL ERROR: {e}")
