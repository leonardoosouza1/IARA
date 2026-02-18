"""
IARA API — Fleet Routes
Agent management: status, start, stop, killswitch.
"""

import time
import logging
from fastapi import APIRouter, Request, HTTPException, Depends
from src.api.auth.jwt import get_current_user
from src.api.schemas import (
    FleetStatusResponse,
    AgentStatus,
    AgentConfig,
    StartAgentRequest,
    StopAgentRequest,
    ManualOverrideRequest,
)
from src.api.fleet.manager import fleet_manager

router = APIRouter(prefix="/fleet", tags=["Fleet Management"])
logger = logging.getLogger("api.fleet")


@router.get("/status", response_model=FleetStatusResponse)
async def fleet_status(user: dict = Depends(get_current_user)):
    """Get the current status of all fleet agents."""
    agents_data = fleet_manager.get_status()
    
    # Convert dicts to AgentStatus objects
    agents = []
    for a in agents_data:
        # Calculate uptime duration
        uptime_ts = a.get("uptime", 0)
        uptime_duration = time.time() - uptime_ts if uptime_ts > 0 else 0
        
        # Determine UI Config
        symbol = a["symbol"]
        status = a.get("status", "idle")
        
        # Color mapping by symbol
        color = "emerald"
        if "BTC" in symbol: color = "orange"
        elif "ETH" in symbol: color = "indigo"
        elif "EUR" in symbol: color = "blue"
            
        # Pulse animation by status
        pulse = "none"
        if status == "active" or status == "in_trade": pulse = color
        elif status == "thinking": pulse = "blue"
        
        agents.append(AgentStatus(
            id=a["id"],
            symbol=a["symbol"],
            market_type=a["market_type"],
            status=a.get("status", "idle"),
            model_name=a["model_name"],
            pnl=a.get("pnl", 0.0),
            win_rate=a.get("win_rate", 0.0),
            total_trades=a.get("total_trades", 0),
            uptime=uptime_duration,
            current_position=a.get("current_position"),
            last_signal=a.get("last_signal"),
            config=AgentConfig(
                pulse=pulse,
                label=symbol,
                color=color
            )
        ))

    total_pnl = sum(a.pnl for a in agents)
    active_count = sum(1 for a in agents if a.status == "active")

    return FleetStatusResponse(agents=agents, total_pnl=total_pnl, active_agents=active_count)


@router.post("/start")
async def start_agent(body: StartAgentRequest, user: dict = Depends(get_current_user)):
    """Start a new trading agent for the given symbol."""
    try:
        agent_id = fleet_manager.start_agent(
            symbol=body.symbol,
            market_type=body.market_type,
            model_name=body.model_name
        )
        return {"message": f"Agent {agent_id} started for {body.symbol}", "agent_id": agent_id}
    except Exception as e:
        logger.error(f"Failed to start agent: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stop")
async def stop_agent(body: StopAgentRequest, user: dict = Depends(get_current_user)):
    """Stop a trading agent gracefully."""
    try:
        fleet_manager.stop_agent(body.agent_id)
        return {"message": f"Agent {body.agent_id} stopped"}
    except Exception as e:
        logger.error(f"Failed to stop agent: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/killswitch")
async def killswitch(user: dict = Depends(get_current_user)):
    """Emergency stop all agents."""
    fleet_manager.stop_all()
    return {"message": "KILLSWITCH activated. All agents stopped.", "status": "shutting_down"}
