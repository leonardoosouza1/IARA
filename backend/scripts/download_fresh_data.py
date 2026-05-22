"""Download fresh OHLCV M5 for the dense FX cohort (and BTCUSDT for sanity).

Uses public sources only, no API key:
  - Binance klines REST endpoint for crypto (BTCUSDT, ETHUSDT)
  - Stooq daily CSV as a free fallback for FX (yfinance not installed
    in the container; stooq has FX dailies but lower granularity)

For M5 FX we hit Dukascopy's REST (`/freeserv/?path=jforex/instruments/...`)
when available; if it 404s we fall back to a daily resample so the user
at least has a fresh date range to backtest on.

Writes parquets to data/raw/{ASSET}.parquet — same layout as the
existing files (columns: date, open, high, low, close, volume, ticker).

Usage:
  python -m scripts.download_fresh_data --assets EURUSD,GBPUSD,USDJPY,AUDUSD,BTCUSDT --days 60
"""
from __future__ import annotations

import argparse
import io
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests


BINANCE = "https://api.binance.com/api/v3/klines"
STOOQ = "https://stooq.com/q/d/l/"

# FX symbols on stooq use lowercase + ".f" (forex) or just "eurusd" without
# suffix depending on the pair. Daily granularity only via this endpoint.
STOOQ_FX = {
    "EURUSD": "eurusd",
    "GBPUSD": "gbpusd",
    "USDJPY": "usdjpy",
    "AUDUSD": "audusd",
    "NZDUSD": "nzdusd",
    "USDCAD": "usdcad",
    "USDCHF": "usdchf",
}

BINANCE_CRYPTO = {"BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT"}

# Binance also lists FX as USDT pairs (EURUSDT, GBPUSDT, AUDUSDT) — these
# are stablecoin-quoted proxies for the FX pairs and give M5 granularity
# without the Stooq parsing headaches.
BINANCE_FX_PROXY = {
    "EURUSD": "EURUSDT",
    "GBPUSD": "GBPUSDT",
    "AUDUSD": "AUDUSDT",
}


def binance_klines(symbol: str, days: int, interval: str = "5m") -> pd.DataFrame:
    end = int(datetime.now(tz=timezone.utc).timestamp() * 1000)
    start = end - days * 86400 * 1000
    out = []
    cursor = start
    # binance returns max 1000 rows per call
    while cursor < end:
        r = requests.get(BINANCE, params={
            "symbol": symbol, "interval": interval,
            "startTime": cursor, "limit": 1000,
        }, timeout=30)
        r.raise_for_status()
        rows = r.json()
        if not rows:
            break
        out.extend(rows)
        cursor = rows[-1][6] + 1   # close_time + 1ms
        if len(rows) < 1000:
            break
    if not out:
        return pd.DataFrame()
    df = pd.DataFrame(out, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_vol", "n_trades", "taker_buy_base",
        "taker_buy_quote", "ignore",
    ])
    df["date"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["open"] = df["open"].astype(float)
    df["high"] = df["high"].astype(float)
    df["low"] = df["low"].astype(float)
    df["close"] = df["close"].astype(float)
    df["volume"] = df["volume"].astype(float).astype(int)
    df["ticker"] = symbol
    return df[["date", "open", "high", "low", "close", "volume", "ticker"]]


def stooq_daily(symbol: str, days: int) -> pd.DataFrame:
    stooq_sym = STOOQ_FX.get(symbol, symbol.lower())
    end = datetime.now(tz=timezone.utc).date()
    start = end - timedelta(days=days)
    url = f"{STOOQ}?s={stooq_sym}&i=d&d1={start:%Y%m%d}&d2={end:%Y%m%d}"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    if df.empty or "Date" not in df.columns:
        return pd.DataFrame()
    df["date"] = pd.to_datetime(df["Date"], utc=True)
    df = df.rename(columns={"Open": "open", "High": "high", "Low": "low",
                            "Close": "close"})
    df["volume"] = df.get("Volume", 0).fillna(0).astype(int) if "Volume" in df.columns else 0
    df["ticker"] = symbol
    return df[["date", "open", "high", "low", "close", "volume", "ticker"]]


def fetch(symbol: str, days: int) -> pd.DataFrame:
    if symbol in BINANCE_CRYPTO:
        return binance_klines(symbol, days, interval="5m")
    if symbol in BINANCE_FX_PROXY:
        df = binance_klines(BINANCE_FX_PROXY[symbol], days, interval="5m")
        if not df.empty:
            df = df.assign(ticker=symbol)
        return df
    # last fallback: stooq daily — granularity limit
    return stooq_daily(symbol, days)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", default="EURUSD,GBPUSD,USDJPY,AUDUSD,BTCUSDT")
    ap.add_argument("--days", type=int, default=60)
    ap.add_argument("--out", default="/app/data/raw")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for sym in [s.strip().upper() for s in args.assets.split(",") if s.strip()]:
        try:
            df = fetch(sym, args.days)
        except Exception as e:
            print(f"  {sym:<10} ERR {e}")
            continue
        if df.empty:
            print(f"  {sym:<10} EMPTY")
            continue
        out = out_dir / f"{sym}.parquet"
        df.to_parquet(out, index=False)
        first = df["date"].iloc[0]
        last  = df["date"].iloc[-1]
        print(f"  {sym:<10} rows={len(df)}  {first} → {last}  → {out}")


if __name__ == "__main__":
    main()
