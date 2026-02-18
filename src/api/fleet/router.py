"""
IARA API — Fleet Routes
Agent management: status, start, stop, killswitch.
"""

import time
from fastapi import APIRouter, Request, HTTPException, Depends
from src.api.auth.jwt import get_current_user
from src.api.schemas import (
    FleetStatusResponse,
    AgentStatus,
    StartAgentRequest,
    StopAgentRequest,
    ManualOverrideRequest,
)

router = APIRouter(prefix="/fleet", tags=["Fleet Management"])


# ── In-memory Fleet State (replaced by FleetAdmiral integration later) ──

_demo_agents: list[dict] = [
    {
        "id": "agent-btc-1",
        "symbol": "BTC/USD",
        "market_type": "crypto",
        "status": "in_trade",
        "model_name": "SwarmBrain v9",
        "pnl": 324.50,
        "win_rate": 0.67,
        "total_trades": 12,
        "uptime": time.time(),
        "current_position": {
            "side": "long",
            "entryPrice": 97500,
            "currentPrice": 98200,
            "size": 0.01,
            "unrealizedPnl": 7.0,
            "entryTime": "2026-02-17T10:00:00Z",
        },
        "last_signal": {
            "timestamp": "2026-02-17T19:00:06Z",
            "action": "buy",
            "confidence": 0.82,
            "longScore": 0.85,
            "shortScore": 0.12,
            "workerVotes": {"M5": 0.72, "M15": 0.55, "M30": 0.80, "H1": 0.38, "H4": 0.65, "consensus": 0.78},
            "lfmThought": "Bullish structure intact. Price holding above key support.",
        },
    },
    {
        "id": "agent-eth-1",
        "symbol": "ETH/USD",
        "market_type": "crypto",
        "status": "thinking",
        "model_name": "SwarmBrain v9",
        "pnl": -42.30,
        "win_rate": 0.45,
        "total_trades": 8,
        "uptime": time.time() - 1800,
        "current_position": None,
        "last_signal": None,
    },
    {
        "id": "agent-eur-1",
        "symbol": "EUR/USD",
        "market_type": "forex",
        "status": "stalking",
        "model_name": "SwarmBrain v9",
        "pnl": 89.20,
        "win_rate": 0.58,
        "total_trades": 19,
        "uptime": time.time() - 7200,
        "current_position": None,
        "last_signal": None,
    },
]


def _get_fleet_status() -> FleetStatusResponse:
    """Build fleet status from current agents."""
    agents = []
    for a in _demo_agents:
        uptime_s = time.time() - a["uptime"]
        agents.append(AgentStatus(
            id=a["id"],
            symbol=a["symbol"],
            market_type=a["market_type"],
            status=a["status"],
            model_name=a["model_name"],
            pnl=a["pnl"],
            win_rate=a["win_rate"],
            total_trades=a["total_trades"],
            uptime=uptime_s,
            current_position=a.get("current_position"),
            last_signal=a.get("last_signal"),
        ))

    total_pnl = sum(a.pnl for a in agents)
    active = sum(1 for a in agents if a.status not in ("idle", "paused"))

    return FleetStatusResponse(agents=agents, total_pnl=total_pnl, active_agents=active)


# ── Routes ──

@router.get("/status", response_model=FleetStatusResponse)
async def fleet_status(user: dict = Depends(get_current_user)):
    """Get the current status of all fleet agents."""
    return _get_fleet_status()


@router.post("/start")
async def start_agent(body: StartAgentRequest, user: dict = Depends(get_current_user)):
    """Start a new trading agent for the given symbol."""
    agent_id = f"agent-{body.symbol.lower().replace('/', '-')}-{len(_demo_agents) + 1}"
    new_agent = {
        "id": agent_id,
        "symbol": body.symbol,
        "market_type": body.market_type,
        "status": "idle",
        "model_name": body.model_name,
        "pnl": 0.0,
        "win_rate": 0.0,
        "total_trades": 0,
        "uptime": time.time(),
        "current_position": None,
        "last_signal": None,
    }
    _demo_agents.append(new_agent)
    return {"message": f"Agent {agent_id} started for {body.symbol}", "agent_id": agent_id}


@router.post("/stop")
async def stop_agent(body: StopAgentRequest, user: dict = Depends(get_current_user)):
    """Stop a trading agent gracefully."""
    for agent in _demo_agents:
        if agent["id"] == body.agent_id:
            agent["status"] = "paused"
            return {"message": f"Agent {body.agent_id} stopped"}

    raise HTTPException(status_code=404, detail=f"Agent {body.agent_id} not found")


@router.post("/killswitch")
async def killswitch(user: dict = Depends(get_current_user)):
    """Emergency stop all agents."""
    for agent in _demo_agents:
        agent["status"] = "paused"
    return {"message": "KILLSWITCH activated. All agents paused.", "agents_affected": len(_demo_agents)}
