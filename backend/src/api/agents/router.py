"""
IARA API — Agent Profiles Routes
CRUD for dynamic trading agents.
"""

import logging
from typing import List
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from src.database.core import get_db
from src.database.models import AgentProfile
from src.api.auth.jwt import get_current_user
from pydantic import BaseModel
import os
import glob

router = APIRouter(prefix="/agents", tags=["Agents"])
logger = logging.getLogger("api.agents")

class AgentCreate(BaseModel):
    id: str
    symbol: str
    market_type: str
    model_version: str = "SwarmBrain v9.7"
    
class AgentResponse(BaseModel):
    id: str
    symbol: str
    market_type: str
    model_version: str
    is_active: bool

@router.get("", response_model=List[AgentResponse])
async def list_agents(db: AsyncSession = Depends(get_db)):
    """List all agent profiles."""
    result = await db.execute(select(AgentProfile))
    agents = result.scalars().all()
    # If empty, initialize defaults
    if not agents:
        defaults = [
            AgentProfile(id="agent-btc-usd", symbol="BTC/USD", market_type="crypto"),
            AgentProfile(id="agent-eur-usd", symbol="EUR/USD", market_type="forex"),
        ]
        db.add_all(defaults)
        await db.commit()
        for d in defaults:
            await db.refresh(d)
        agents = defaults
    return agents

@router.post("", response_model=AgentResponse)
async def create_agent(agent: AgentCreate, db: AsyncSession = Depends(get_db)):
    """Create a new agent profile."""
    # Check if exists
    result = await db.execute(select(AgentProfile).where(AgentProfile.id == agent.id))
    existing = result.scalars().first()
    if existing:
        raise HTTPException(status_code=400, detail="Agent ID already exists")

    new_agent = AgentProfile(
        id=agent.id,
        symbol=agent.symbol,
        market_type=agent.market_type,
        model_version=agent.model_version,
        is_active=True
    )
    db.add(new_agent)
    await db.commit()
    await db.refresh(new_agent)
    
    try:
        from src.api.fleet.manager import fleet_manager
        fleet_manager.start_agent(
            symbol=new_agent.symbol,
            market_type=new_agent.market_type,
            model_name=new_agent.model_version,
            execution_venue="ALPACA"
        )
    except Exception as e:
        logger.error(f"Failed to start newly created agent {new_agent.id}: {e}")

    return new_agent

@router.delete("/{agent_id}")
async def delete_agent(agent_id: str, db: AsyncSession = Depends(get_db)):
    """Delete an agent profile."""
    result = await db.execute(select(AgentProfile).where(AgentProfile.id == agent_id))
    agent = result.scalars().first()
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
        
    await db.delete(agent)
    await db.commit()
    return {"message": "Agent deleted successfully"}

@router.put("/{agent_id}/toggle")
async def toggle_agent(agent_id: str, db: AsyncSession = Depends(get_db)):
    """Toggle agent active status."""
    result = await db.execute(select(AgentProfile).where(AgentProfile.id == agent_id))
    agent = result.scalars().first()
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
        
    agent.is_active = not agent.is_active
    await db.commit()
    
    try:
        from src.api.fleet.manager import fleet_manager
        if agent.is_active:
            fleet_manager.start_agent(
                symbol=agent.symbol,
                market_type=agent.market_type,
                model_name=agent.model_version,
                execution_venue="ALPACA"
            )
        else:
            fleet_manager.stop_agent(agent.id)
    except Exception as e:
        logger.error(f"Failed to toggle agent {agent.id} in fleet: {e}")
        
    return {"message": f"Agent {'activated' if agent.is_active else 'deactivated'} successfully", "is_active": agent.is_active}

@router.get("/models")
async def list_models():
    """List available model checkpoints (.pth) in logs/models."""
    models_dir = os.path.join(os.getcwd(), "logs", "models")
    if not os.path.exists(models_dir):
        return []
        
    pths = glob.glob(os.path.join(models_dir, "*.pth"))
    models = []
    
    for pth in pths:
        filename = os.path.basename(pth)
        # Parse common v9 naming (e.g. best_model_v9_BTCUSD.pth)
        name = filename.replace(".pth", "")
        if "best_model_v9_" in name:
            pair = name.split("best_model_v9_")[-1]
            label = f"SwarmBrain v9.7 ({pair})"
        else:
            label = name
            
        models.append({
            "value": filename,
            "label": label
        })
        
    return models
