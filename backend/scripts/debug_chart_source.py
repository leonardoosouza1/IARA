import os
import pandas as pd
from src.dashboard.services.alpaca_data_service import AlpacaDataService

print("--- DIAGNOSTICO DE DADOS ---")
svc = AlpacaDataService()
print(f"Configurado: {svc.is_configured}")
print(f"Symbol: {svc.symbol}")
print(f"Is Crypto? {svc.is_crypto('BTC/USD')}")

print("\n--- TENTATIVA 1: Crypto Bars (BTC/USD) ---")
try:
    df = svc.get_bars("BTC/USD", limit=10)
    print(f"Resultado: {len(df)} linhas")
    if not df.empty:
        print(df.head())
    else:
        print("⚠️ DataFrame VAZIO devolvido pela Alpaca!")
except Exception as e:
    print(f"❌ Erro ao buscar: {e}")

print("\n--- TENTATIVA 2: Stock Bars (AAPL) ---")
try:
    df = svc.get_bars("AAPL", limit=10)
    print(f"Resultado: {len(df)} linhas")
    if not df.empty:
        print(df.head())
    else:
        print("⚠️ DataFrame VAZIO devolvido pela Alpaca!")
except Exception as e:
    print(f"❌ Erro ao buscar: {e}")
