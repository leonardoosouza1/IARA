import os
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame
from datetime import datetime, timedelta

load_dotenv()

api_key = os.getenv("ALPACA_API_KEY")
secret_key = os.getenv("ALPACA_SECRET_KEY")
paper = True

client = TradingClient(api_key, secret_key, paper=paper)

print("--- Account Object ---")
try:
    acc = client.get_account()
    # Print all attributes of the account object
    print(dir(acc))
    print(acc)
except Exception as e:
    print(f"Error fetching account: {e}")

print("\n--- Portfolio History ---")
try:
    # Check method signature or available arguments
    # help(client.get_portfolio_history) 
    # Try fetching without arguments first to see default
    hist = client.get_portfolio_history()
    print(dir(hist))
    print(hist)
except Exception as e:
    print(f"Error fetching portfolio history: {e}")

print("\n--- Bars ---")
try:
    data_client = CryptoHistoricalDataClient()
    req = CryptoBarsRequest(
        symbol_or_symbols="BTC/USD",
        timeframe=TimeFrame.Minute,
        start=datetime.now() - timedelta(days=1),
        limit=5
    )
    bars = data_client.get_crypto_bars(req)
    print("Bars keys:", bars.keys())
    if "BTC/USD" in bars:
        print("First bar:", bars["BTC/USD"][0])
        print("First bar dir:", dir(bars["BTC/USD"][0]))
except Exception as e:
    print(f"Error fetching bars: {e}")
