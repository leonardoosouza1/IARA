"""
IARA API — Socket.IO Event Manager
Handles real-time bidirectional communication.
"""

import logging
import socketio
from jose import jwt, JWTError

from src.api.config import settings

logger = logging.getLogger("iara.events")

# Create the async Socket.IO server
sio = socketio.AsyncServer(
    async_mode="asgi",
    cors_allowed_origins=settings.FRONTEND_URLS,
    logger=False,
    engineio_logger=False,
)


# ── Connection Lifecycle ──────────────────────────────────────

@sio.event
async def connect(sid: str, environ: dict, auth: dict | None = None):
    """Validate JWT cookie on WebSocket handshake."""
    # Extract cookies from the HTTP headers in the handshake
    cookie_header = environ.get("HTTP_COOKIE", "")
    cookies = {}
    for item in cookie_header.split(";"):
        item = item.strip()
        if "=" in item:
            key, val = item.split("=", 1)
            cookies[key.strip()] = val.strip()

    token = cookies.get("access_token")
    if not token:
        logger.warning(f"[WS] Connection rejected (no token): {sid}")
        raise socketio.exceptions.ConnectionRefusedError("Authentication required")

    try:
        payload = jwt.decode(token, settings.JWT_SECRET, algorithms=[settings.JWT_ALGORITHM])
        # Store user data in the session
        async with sio.session(sid) as session:
            session["user_id"] = payload["sub"]
            session["username"] = payload["username"]
        logger.info(f"[WS] Connected: {payload['username']} ({sid})")
    except JWTError:
        logger.warning(f"[WS] Connection rejected (invalid token): {sid}")
        raise socketio.exceptions.ConnectionRefusedError("Invalid token")


@sio.event
async def disconnect(sid: str):
    logger.info(f"[WS] Disconnected: {sid}")


# ── Client → Server Events ───────────────────────────────────

@sio.event
async def subscribe(sid: str, data: dict):
    """Subscribe to a market symbol's room for live data."""
    symbol = data.get("symbol")
    if symbol:
        await sio.enter_room(sid, f"market:{symbol}")
        logger.info(f"[WS] {sid} subscribed to market:{symbol}")


@sio.event
async def unsubscribe(sid: str, data: dict):
    """Unsubscribe from a symbol's room."""
    symbol = data.get("symbol")
    if symbol:
        await sio.leave_room(sid, f"market:{symbol}")
        logger.info(f"[WS] {sid} unsubscribed from market:{symbol}")


@sio.event
async def manual_override(sid: str, data: dict):
    """Handle manual Buy/Sell from God Mode."""
    async with sio.session(sid) as session:
        username = session.get("username", "unknown")

    agent_id = data.get("agent_id")
    action = data.get("action")
    logger.warning(f"[WS] MANUAL OVERRIDE by {username}: {action} on {agent_id}")

    # TODO: Forward to FleetAdmiral.manual_override(agent_id, action)
    # For now, emit confirmation back
    await sio.emit("trade_update", {
        "agent_id": agent_id,
        "type": f"manual_{action}",
        "price": 0,
        "pnl": 0,
        "initiated_by": username,
    }, room=sid)


# ── Server → Client Emission Helpers ─────────────────────────

class SocketEmitter:
    """
    Bridge between the trading engine and the frontend.
    FleetAdmiral calls these methods to push events to connected clients.
    """

    @staticmethod
    async def emit_market_tick(symbol: str, price: float, timestamp: str):
        await sio.emit("market_tick", {
            "symbol": symbol,
            "price": price,
            "time": timestamp,
        }, room=f"market:{symbol}")

    @staticmethod
    async def emit_candle_closed(symbol: str, candle: dict):
        await sio.emit("candle_closed", {
            "symbol": symbol,
            **candle,
        }, room=f"market:{symbol}")

    @staticmethod
    async def emit_brain_thought(agent_id: str, step: str, text: str, level: str = "info"):
        await sio.emit("brain_thought", {
            "agentId": agent_id,
            "timestamp": "",  # Will be set by caller
            "step": step,
            "text": text,
            "level": level,
        })

    @staticmethod
    async def emit_agent_update(agent: dict):
        await sio.emit("agent_update", agent)

    @staticmethod
    async def emit_fleet_sync(agents: list[dict]):
        await sio.emit("fleet_sync", agents)

    @staticmethod
    async def emit_worker_votes(agent_id: str, votes: dict):
        await sio.emit("worker_votes", {
            "agentId": agent_id,
            **votes,
        })
    @staticmethod
    async def emit_historical_candles(symbol: str, candles: list, sid: str = None):
        """Emit historical candles to frontend."""
        target = sid if sid else f"market:{symbol}"
        await sio.emit(
            "historical_candles",
            {
                "symbol": symbol,
                "candles": candles,
            },
            room=target,
        )

@sio.event
async def request_history(sid: str, data: dict):
    """Handle history request from frontend."""
    from src.api.market_data import market_service
    symbol = data.get("symbol")
    if symbol:
        history = market_service.get_history(symbol)
        if history:
            await SocketEmitter.emit_historical_candles(symbol, history, sid=sid)

@sio.event
async def manual_override(sid: str, data: dict):
    """Handle FORCE BUY / FORCE SELL events from frontend UI."""
    from src.api.fleet.manager import fleet_manager
    agent_id = data.get("agent_id")
    action = data.get("action")
    
    if not agent_id or not action:
        return
        
    try:
        # Pass manual override signal down to fleet manager
        fleet_manager.manual_override(agent_id, action.lower())
        logger.info(f"Manual Override Triggered: {action.upper()} on {agent_id}")
    except Exception as e:
        logger.error(f"Failed to execute manual override: {e}")

