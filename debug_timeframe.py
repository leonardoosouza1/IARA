from datetime import datetime
from alpaca.data.timeframe import TimeFrame
from alpaca.data.requests import CryptoBarsRequest
from src.dashboard.services.alpaca_data_service import AlpacaDataService

print("--- DEBUG TIMEFRAME ---")
svc = AlpacaDataService()
symbol = "BTC/USD"

# Test 1: 1Min (Default - known to work?)
print("\n--- Test 1: 1Min ---")
try:
    df = svc.get_bars(symbol, timeframe="1Min", limit=5)
    print(f"Success! Rows: {len(df)}")
except Exception as e:
    print(f"Failed: {e}")

# Test 2: 5Min (Suspected failure)
print("\n--- Test 2: 5Min ---")
try:
    df = svc.get_bars(symbol, timeframe="5Min", limit=5)
    print(f"Success! Rows: {len(df)}")
except Exception as e:
    print(f"Failed: {e}")

# Manual Test with TimeFrame constructor
print("\n--- Manual TimeFrame Construction ---")
try:
    from alpaca.data.timeframe import TimeFrameUnit
    tf_enum = TimeFrame(5, TimeFrameUnit.Minute)
    print("Created TimeFrame with Enum:", tf_enum)
    
    tf_str = TimeFrame(5, "Min")
    print("Created TimeFrame with Str:", tf_str)
except Exception as e:
    print(f"Construction Error: {e}")
