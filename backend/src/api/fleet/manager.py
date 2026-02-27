"""
Fleet Manager
=============
Orchestrates multiple TrinityEngine instances for the API.
Singleton responsible for lifecycle management and state aggregation.
"""

import logging
import threading
import time
from typing import Dict, Optional, List

from src.brain.engine import TrinityEngine, EngineConfig
from src.trading.signal_generator import SignalEngine

logger = logging.getLogger("FleetManager")


class FleetManager:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(FleetManager, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
            
        self.engines: Dict[str, TrinityEngine] = {}
        self.configs: Dict[str, EngineConfig] = {}
        self._market_types: Dict[str, str] = {}
        self.logger = logger
        self._initialized = True
        self.logger.info("FleetManager initialized")

    def start_agent(self, symbol: str, market_type: str = "crypto", model_name: str = "best_model_v9_EURUSD.pth", execution_venue: str = "ALPACA") -> str:
        """Start a new TrinityEngine agent for a symbol."""
        agent_id = f"agent-{symbol.lower().replace('/', '-')}"
        
        if agent_id in self.engines:
            self.logger.warning(f"Agent {agent_id} already exists")
            if not self.engines[agent_id].active:
                self.logger.info(f"Restarting existing agent {agent_id}")
                self.engines[agent_id].start()
            return agent_id

        import os
        if os.path.exists(model_name):
            checkpoint_path = model_name
        else:
            checkpoint_path = f"logs/models/{model_name.replace('logs/models/', '')}"

        config = EngineConfig(
            symbol=symbol,
            market_type=market_type,
            execution_venue=execution_venue,
            checkpoint_path=checkpoint_path,
            lfm_model_name="lfm2.5-thinking:1.2b"
        )
        
        try:
            self.logger.info(f"Initializing engine for {symbol}...")
            engine = TrinityEngine.from_checkpoint(config)
            engine.start()
            
            self.engines[agent_id] = engine
            self.configs[agent_id] = config
            self._market_types[agent_id] = market_type
            self.logger.info(f"✅ Agent {agent_id} started successfully")
            return agent_id
            
        except Exception as e:
            self.logger.error(f"Failed to start agent {agent_id}: {e}")
            raise e

    def stop_agent(self, agent_id: str):
        """Stop a specific agent."""
        if agent_id in self.engines:
            self.logger.info(f"Stopping agent {agent_id}...")
            self.engines[agent_id].stop()
            # Remove from memory so it can be cleanly recreated/reloaded
            del self.engines[agent_id]
            if agent_id in self.configs:
                del self.configs[agent_id]
            if agent_id in self._market_types:
                del self._market_types[agent_id]
        else:
            self.logger.warning(f"Agent {agent_id} not found")

    def stop_all(self):
        """Emergency stop for all agents."""
        self.logger.warning("🚨 KILLSWITCH ACTIVATED - Stopping all agents")
        for agent_id, engine in self.engines.items():
            if engine.active:
                engine.stop()

    def manual_override(self, agent_id: str, action: str):
        """Force a manual buy/sell action on a specific agent."""
        if agent_id in self.engines:
            self.logger.info(f"Executing manual override {action.upper()} on {agent_id}")
            # The engine itself should handle the execution. For now, we just log it.
            # In a full implementation, you'd route this to the agent's executor
            self.engines[agent_id].executor.execute_signal(action.upper())
        else:
            self.logger.warning(f"Agent {agent_id} not found for manual override")

    def get_status(self) -> List[dict]:
        """Collect status from all agents."""
        status_list = []
        
        for agent_id, engine in self.engines.items():
            state = engine.state.get_state()
            config = self.configs[agent_id]
            
            last_signal = state.get("signal")
            signal_dict = None
            if last_signal:
                # Map direction to frontend action format
                action_map = {"long": "buy", "short": "sell", "wait": "hold"}
                action = action_map.get(last_signal.direction.lower(), "hold")
                
                # Build workerVotes in frontend format: {M5, M15, M30, H1, H4, consensus}
                worker_votes = last_signal.worker_scores.copy()
                worker_votes["consensus"] = last_signal.worker_agreement
                
                signal_dict = {
                    "timestamp": last_signal.timestamp,
                    "action": action,
                    "confidence": last_signal.confidence,
                    "longScore": last_signal.long_score,
                    "shortScore": last_signal.short_score,
                    "workerVotes": worker_votes,
                    "lfmThought": state.get("lfm_summary", "")
                }
            
            market_type = self._market_types.get(agent_id, "crypto")
            
            status_entry = {
                "id": agent_id,
                "symbol": config.symbol,
                "market_type": market_type,
                "status": "active" if engine.active else "paused",
                "model_name": config.checkpoint_path.split("/")[-1],
                "pnl": 0.0,
                "win_rate": 0.0,
                "total_trades": 0,
                "uptime": engine.start_time,
                "current_position": None,
                "last_signal": signal_dict
            }
            status_list.append(status_entry)
            
        return status_list

# Global instance
fleet_manager = FleetManager()
