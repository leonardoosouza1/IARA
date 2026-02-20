"""
IARA API — Pydantic Schemas
Strict type definitions for request/response bodies.
"""

from pydantic import BaseModel, Field
from typing import Optional


# ── Auth ──────────────────────────────────────────────────────

class LoginRequest(BaseModel):
    username: str = Field(..., min_length=1, max_length=50)
    password: str = Field(..., min_length=1, max_length=128)


class UserResponse(BaseModel):
    id: str
    username: str
    role: str


class TokenResponse(BaseModel):
    user: UserResponse
    message: str


# ── Fleet ─────────────────────────────────────────────────────

class AgentConfig(BaseModel):
    pulse: str
    label: str
    color: str


class AgentStatus(BaseModel):
    id: str
    symbol: str
    market_type: str  # "crypto" | "forex"
    status: str  # "idle" | "thinking" | "stalking" | "in_trade" | "paused" | "error"
    model_name: str
    pnl: float
    win_rate: float
    total_trades: int
    uptime: float  # seconds
    current_position: Optional[dict] = None
    last_signal: Optional[dict] = None
    config: Optional[AgentConfig] = None


class FleetStatusResponse(BaseModel):
    agents: list[AgentStatus]
    total_pnl: float
    active_agents: int


class StartAgentRequest(BaseModel):
    symbol: str = Field(..., min_length=1, max_length=20)
    market_type: str = Field(..., pattern="^(crypto|forex)$")
    model_name: str = Field(default="SwarmBrain v9")


class StopAgentRequest(BaseModel):
    agent_id: str


class ManualOverrideRequest(BaseModel):
    agent_id: str
    action: str = Field(..., pattern="^(buy|sell)$")


# ── Market ────────────────────────────────────────────────────

class CandleData(BaseModel):
    time: int
    open: float
    high: float
    low: float
    close: float
    volume: Optional[float] = None
