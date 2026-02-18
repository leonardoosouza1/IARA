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
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import socketio

from src.api.config import settings
from src.api.events import sio
from src.api.auth.router import router as auth_router
from src.api.fleet.router import router as fleet_router
from src.api.market_data import market_service

# ── Logging ───────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-18s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("iara.server")



# ── App Lifecycle ─────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manages background tasks during app lifecycle."""
    logger.info("=" * 60)
    logger.info("  I.A.R.A. API Server starting...")
    logger.info(f"  Frontend URL: {settings.FRONTEND_URL}")
    logger.info(f"  Debug: {settings.DEBUG}")
    logger.info("=" * 60)

    # Start the Market Data Service (feeds engines)
    service_task = asyncio.create_task(market_service.run())

    yield

    # Shutdown
    await market_service.stop()
    service_task.cancel()
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
