from datetime import datetime
from typing import Optional, List, Dict, Any
from sqlalchemy import Column, Integer, String, Float, DateTime, JSON, ForeignKey, Boolean
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.sql import func

Base = declarative_base()

class Trade(Base):
    """
    Records executed trades.
    Replaces the ephemeral 'orders' list in SharedState.
    """
    __tablename__ = "trades"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, index=True, nullable=False)
    side = Column(String, nullable=False) # 'BUY' or 'SELL'
    qty = Column(Float, nullable=False)
    price = Column(Float, nullable=False)
    
    # Execution Details
    order_id = Column(String, unique=True, index=True) # ID from Exchange
    strategy = Column(String, default="Trinity-V9")
    
    # Profit Tracking
    pnl = Column(Float, nullable=True) # Realized PnL (filled on close)
    entry_price = Column(Float, nullable=True) # For closing trades
    
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    def __repr__(self):
        return f"<Trade(id={self.id}, {self.side} {self.symbol} @ {self.price})>"


class Signal(Base):
    """
    Records the 'Brain State' at the moment of decision.
    Essential for 'X-Ray' analysis and Retraining.
    """
    __tablename__ = "signals"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, index=True, nullable=False)
    timeframe = Column(String, default="M5")
    
    # Model Output
    signal_value = Column(Integer, nullable=False) # -1 (Sell), 0 (Hold), 1 (Buy)
    confidence = Column(Float, default=0.0)
    model_version = Column(String, nullable=False) # e.g. "v9.7-Gladiator"
    
    # Context Snapshot (The 'Why')
    # Use JSONB in Postgres for better query performance
    features_snapshot = Column(JSON, nullable=True) # RSI, Price, etc.
    worker_votes = Column(JSON, nullable=True) # { "worker_1": "BUY", ... }
    
    timestamp = Column(DateTime(timezone=True), server_default=func.now())

    def __repr__(self):
        return f"<Signal(id={self.id}, {self.symbol}, val={self.signal_value}, conf={self.confidence:.2f})>"


class ModelVersion(Base):
    """
    Registry of trained models.
    """
    __tablename__ = "model_versions"

    id = Column(Integer, primary_key=True, index=True)
    tag = Column(String, unique=True, index=True, nullable=False) # v9.7
    path = Column(String, nullable=False) # checkpoints/v9.7.pth
    
    metrics = Column(JSON, nullable=True) # { "BTC/USD": {"accuracy": 0.65}, "ETH/USD": {"accuracy": 0.61} }
    category_type = Column(String, default="SPECIALIST") # "SPECIALIST" (one pair) or "GENERALIST" (multi-pair)
    is_active = Column(Boolean, default=False)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
