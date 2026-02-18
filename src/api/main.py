"""
IARA API — Main Server Entrypoint
FastAPI + Socket.IO combined ASGI application.

Usage:
    python -m src.api.main
    # or
    uvicorn src.api.main:app --reload --port 8000
"""

import asyncio
import logging
import random
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import socketio

from src.api.config import settings
from src.api.events import sio, SocketEmitter
from src.api.auth.router import router as auth_router
from src.api.fleet.router import router as fleet_router

# ── Logging ───────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-18s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("iara.server")


# ── Demo Data Emitter (simulates live market + brain) ─────────

async def demo_market_emitter():
    """
    Background task that simulates real-time market data.
    Emits market_tick every ~500ms and candle_closed every ~30s.
    Also emits brain_thought periodically to populate the ThoughtStream.
    """
    symbols = ["BTC/USD", "ETH/USD", "EUR/USD"]
    prices = {"BTC/USD": 97500.0, "ETH/USD": 3800.0, "EUR/USD": 1.0850}
    candle_start = {s: time.time() for s in symbols}
    candle_ohlc = {}

    thoughts = [
        ("analysis", "Scanning order book for liquidity imbalances..."),
        ("analysis", "Checking 200-EMA alignment across M5, M15, H1..."),
        ("signal", "Worker M5 detected bullish divergence on RSI."),
        ("signal", "Worker H1 confirms uptrend continuation pattern."),
        ("decision", "Consensus above 0.72 threshold — preparing entry."),
        ("risk", "Position size calculated: 0.01 BTC at 2% risk."),
        ("execution", "Order submitted: LIMIT BUY @ $97,450."),
        ("monitor", "Monitoring stop-loss at $96,800 (0.67% below entry)."),
        ("analysis", "Volume profile shows absorption zone at $97,200."),
        ("signal", "LFM strategic context: 'Institutional accumulation probable.'"),
    ]

    logger.info("[DEMO] Market emitter started")
    tick_count = 0

    while True:
        try:
            for symbol in symbols:
                # Simulate price movement
                volatility = {"BTC/USD": 50, "ETH/USD": 5, "EUR/USD": 0.0003}
                prices[symbol] += (random.random() - 0.48) * volatility[symbol]
                now_ts = time.time()

                # Emit market tick
                await SocketEmitter.emit_market_tick(
                    symbol=symbol,
                    price=round(prices[symbol], 4),
                    timestamp=str(int(now_ts)),
                )

                # Track candle OHLC
                if symbol not in candle_ohlc:
                    candle_ohlc[symbol] = {
                        "open": prices[symbol],
                        "high": prices[symbol],
                        "low": prices[symbol],
                        "close": prices[symbol],
                    }
                else:
                    candle_ohlc[symbol]["high"] = max(candle_ohlc[symbol]["high"], prices[symbol])
                    candle_ohlc[symbol]["low"] = min(candle_ohlc[symbol]["low"], prices[symbol])
                    candle_ohlc[symbol]["close"] = prices[symbol]

                # Emit candle every ~30s
                if now_ts - candle_start[symbol] >= 30:
                    await SocketEmitter.emit_candle_closed(symbol, {
                        "time": int(now_ts),
                        "open": round(candle_ohlc[symbol]["open"], 4),
                        "high": round(candle_ohlc[symbol]["high"], 4),
                        "low": round(candle_ohlc[symbol]["low"], 4),
                        "close": round(candle_ohlc[symbol]["close"], 4),
                    })
                    candle_start[symbol] = now_ts
                    candle_ohlc[symbol] = {
                        "open": prices[symbol],
                        "high": prices[symbol],
                        "low": prices[symbol],
                        "close": prices[symbol],
                    }

            # Emit brain thoughts every ~3 seconds
            tick_count += 1
            if tick_count % 6 == 0:
                step, text = random.choice(thoughts)
                await SocketEmitter.emit_brain_thought(
                    agent_id="agent-btc-1",
                    step=step,
                    text=text,
                    level=random.choice(["info", "info", "info", "success", "warning"]),
                )

            # Emit agent updates (simulated PnL) every ~1s (tick_count % 2 == 0)
            if tick_count % 2 == 0:
                # Update BTC agent PnL
                btc_pnl_change = (random.random() - 0.45) * 5  # Slight positive bias
                await SocketEmitter.emit_agent_update({
                    "id": "agent-btc-1",
                    "pnl": round(324.50 + btc_pnl_change, 2), # Base + change (simplified for demo)
                    "status": "in_trade",
                    "symbol": "BTC/USD",
                    # In a real app we'd send the full object or delta, for demo we send partial
                    "current_position": {
                        "side": "long",
                        "size": 0.01,
                        "unrealizedPnl": round(7.0 + btc_pnl_change, 2),
                        "entryPrice": 97500,
                        "currentPrice": round(prices["BTC/USD"], 2),
                    }
                })

            await asyncio.sleep(0.5)

        except Exception as e:
            logger.error(f"[DEMO] Emitter error: {e}")
            await asyncio.sleep(1)


# ── App Lifecycle ─────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manages background tasks during app lifecycle."""
    logger.info("=" * 60)
    logger.info("  I.A.R.A. API Server starting...")
    logger.info(f"  Frontend URL: {settings.FRONTEND_URL}")
    logger.info(f"  Debug: {settings.DEBUG}")
    logger.info("=" * 60)

    # Start the demo market emitter
    emitter_task = asyncio.create_task(demo_market_emitter())

    yield

    # Shutdown
    emitter_task.cancel()
    logger.info("I.A.R.A. API Server stopped.")


# ── FastAPI Application ──────────────────────────────────────

api = FastAPI(
    title="I.A.R.A. API",
    description="Intelligent Autonomous Real-time Agent — Backend API",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow frontend origin
api.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.FRONTEND_URL],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API routes under /api/v1
api.include_router(auth_router, prefix="/api/v1")
api.include_router(fleet_router, prefix="/api/v1")


@api.get("/api/v1/health")
async def health_check():
    return {"status": "online", "service": "I.A.R.A.", "version": "1.0.0"}


# ── Combine FastAPI + Socket.IO into single ASGI app ──────────

app = socketio.ASGIApp(sio, other_asgi_app=api)


# ── CLI Entrypoint ────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info",
    )
