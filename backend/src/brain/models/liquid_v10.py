"""
I.A.R.A. v10.0 - Swarm Coordinator (Armor Brain)
================================================
Advanced central brain for I.A.R.A. v10.0.
Supports augmented sensory input from Order Book (RAIO-X).

Architecture:
1. Cross-Attention: Multi-head worker consensus
2. Consensus Fusion: Confidence-weighted context
3. Liquid CfC: Time-continuous decision refinement
4. Sensory Bridge: Fusion of OHLCV + L2 Order Book
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.brain.models.workers import WorkerSwarm, WorkerConfig, CfCCell


@dataclass
class CoordinatorConfig:
    """Configuration for SwarmCoordinator v10.0."""
    num_workers: int = 5
    worker_dim: int = 128            # v10.0 Higher capacity
    hidden_dim: int = 512            # v10.0 Higher capacity
    num_heads: int = 8               # More heads for complex L2 patterns
    dropout: float = 0.2
    num_cfc_layers: int = 2          # Deeper liquid refinement

class CrossAttentionFusion(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attention(x, x, x, need_weights=False)
        x = self.norm(x + self.dropout(attn_out))
        return x

class ConsensusFusion(nn.Module):
    def __init__(self, num_workers: int, worker_dim: int, num_classes: int = 3, hidden_dim: int = 512):
        super().__init__()
        self.num_workers = num_workers
        self.cross_attn = CrossAttentionFusion(embed_dim=worker_dim, num_heads=8, dropout=0.1)
        self.confidence_proj = nn.Linear(num_classes, 1)
        fusion_input_dim = num_workers * worker_dim + num_workers
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.LayerNorm(hidden_dim)
        )
        
    def forward(self, embeddings: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        batch_size = embeddings.size(0)
        attended = self.cross_attn(embeddings)
        probs = F.softmax(logits, dim=-1)
        confidence = torch.sigmoid(self.confidence_proj(probs))
        weighted = attended * confidence
        combined = torch.cat([weighted.view(batch_size, -1), confidence.view(batch_size, -1)], dim=-1)
        return self.fusion(combined)

class SwarmCoordinator(nn.Module):
    def __init__(self, config: CoordinatorConfig = None):
        super().__init__()
        if config is None: config = CoordinatorConfig()
        self.config = config
        self.fusion = ConsensusFusion(
            num_workers=config.num_workers,
            worker_dim=config.worker_dim,
            num_classes=3,
            hidden_dim=config.hidden_dim
        )
        self.cfc = CfCCell(input_dim=config.hidden_dim, hidden_dim=config.hidden_dim, dropout=config.dropout)
        self.long_expert = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2), nn.SiLU(),
            nn.Dropout(config.dropout), nn.Linear(config.hidden_dim // 2, 1)
        )
        self.short_expert = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2), nn.SiLU(),
            nn.Dropout(config.dropout), nn.Linear(config.hidden_dim // 2, 1)
        )
        self.policy_head = nn.Linear(config.hidden_dim, 3)
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)
        
    def forward(self, embeddings: torch.Tensor, worker_logits: torch.Tensor, dt: Optional[torch.Tensor] = None, hx: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        context = self.fusion(embeddings, worker_logits)
        _, h_refined = self.cfc(context.unsqueeze(1), dt, hx)
        return self.policy_head(h_refined), self.long_expert(h_refined) - self.short_expert(h_refined), self.long_expert(h_refined), self.short_expert(h_refined)

class SwarmBrain(nn.Module):
    def __init__(self, worker_config: WorkerConfig = None, coordinator_config: CoordinatorConfig = None):
        super().__init__()
        if worker_config is None: worker_config = WorkerConfig(hidden_dim=128) # v10.0 Higher capacity
        if coordinator_config is None: coordinator_config = CoordinatorConfig(num_workers=5, worker_dim=worker_config.hidden_dim)
        self.worker_config = worker_config
        self.coordinator_config = coordinator_config
        self.swarm = WorkerSwarm(worker_config)
        self.coordinator = SwarmCoordinator(coordinator_config)
        
    def forward(self, x_dict: dict, hx: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        embeddings, worker_logits, _ = self.swarm(x_dict)
        return self.coordinator(embeddings, worker_logits, x_dict.get("coord_dt"), hx)
    
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
