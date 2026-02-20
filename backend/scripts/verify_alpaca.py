import os
from pprint import pprint
import sys
from dotenv import load_dotenv

# Path alignment since this runs in /app inside Docker
sys.path.append("/app")

# Ensure .env is explicitly loaded if not already in env vars
load_dotenv("/app/.env")

try:
    from alpaca.trading.client import TradingClient
    from alpaca.data.historical import CryptoHistoricalDataClient
    from alpaca.data.requests import CryptoLatestQuoteRequest
except ImportError:
    print("❌ Alpaca Python SDK not installed. Please run: pip install alpaca-py")
    sys.exit(1)

# Initialize clients
api_key = os.getenv("ALPACA_API_KEY")
secret_key = os.getenv("ALPACA_SECRET_KEY")
paper = os.getenv("ALPACA_PAPER", "true").lower() == "true"

print(f"🔑 Using Keys:")
print(f"API KEY: {api_key[:5]}...{api_key[-4:] if api_key else 'None'}")
print(f"MODE: {'🧻 PAPER' if paper else '💸 LIVE'}\n")

if not api_key or not secret_key:
    print("❌ API Keys missing in environment.")
    sys.exit(1)

trading_client = TradingClient(api_key, secret_key, paper=paper)
data_client = CryptoHistoricalDataClient(api_key, secret_key)

print("--- Testing Account Details ---")
try:
    account = trading_client.get_account()
    print("✅ Account Fetched Successfully.")
    print(f"💵 Status: {account.status}")
    print(f"💵 Equity: ${float(account.equity):.2f}")
    print(f"💵 Buying Power: ${float(account.buying_power):.2f}")
    if account.status != "ACTIVE":
        print("⚠️ Warning: Account is not active!")
except Exception as e:
    print(f"❌ Failed to fetch account: {e}")

print("\n--- Testing Data Fetching (BTC/USD) ---")
try:
    req = CryptoLatestQuoteRequest(symbol_or_symbols=["BTC/USD"])
    quote = data_client.get_crypto_latest_quote(req)
    btc_quote = quote.get("BTC/USD")
    if btc_quote:
        print(f"✅ Data Fetched Successfully.")
        print(f"📈 BTC/USD Ask Price: ${btc_quote.ask_price:.2f}")
    else:
        print("⚠️ No data returned for BTC/USD")
except Exception as e:
    print(f"❌ Failed to fetch market data: {e}")
