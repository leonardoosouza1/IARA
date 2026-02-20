import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

def check_setup():
    print("🔍 Checking Alpaca Markets Integration Setup...")
    
    # Check .env
    key = os.getenv("ALPACA_API_KEY")
    secret = os.getenv("ALPACA_SECRET_KEY")
    
    if not key or key == "YOUR_ALPACA_KEY":
        print("❌ ALPACA_API_KEY not set in .env")
    else:
        print("✅ ALPACA_API_KEY found")
        
    if not secret or secret == "YOUR_ALPACA_SECRET":
        print("❌ ALPACA_SECRET_KEY not set in .env")
    else:
        print("✅ ALPACA_SECRET_KEY found")

    # Check dependencies
    try:
        import alpaca
        print(f"✅ alpaca-py library installed (v{alpaca.__version__ if hasattr(alpaca, '__version__') else 'unknown'})")
    except ImportError:
        print("❌ alpaca-py library MISSING. Run 'pip install alpaca-py'")

    # Check files
    files = [
        "src/brain/tools/alpaca_executor.py",
        "scripts/live_ingest_alpaca.py",
        "src/brain/engine.py",
        "src/dashboard/pages/mission_control.py"
    ]
    
    for f in files:
        if Path(f).exists():
            print(f"✅ File exists: {f}")
        else:
            print(f"❌ File missing: {f}")

    print("\n🚀 Alpaca setup check complete. Insert your keys in .env to start trading.")

if __name__ == "__main__":
    check_setup()
