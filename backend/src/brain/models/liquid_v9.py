"""
I.A.R.A. v9 - Swarm Coordinator (Liquid Brain)
===============================================
The central brain that fuses Worker embeddings via Cross-Attention
and makes the final trading decision.

Architecture:
1. Cross-Attention: Workers "discuss" with each other
2. Consensus Fusion: Weighted combination based on confidence
3. Final CfC: Refines the consensus into a decision
4. Expert Heads: Long/Short specialists for asymmetric learning

Reference: "Attention Is All You Need" + Liquid Neural Networks
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.brain.models.workers import WorkerSwarm, WorkerConfig, CfCCell


@dataclass
class CoordinatorConfig:
    """Configuration for SwarmCoordinator."""
    num_workers: int = 5
    worker_dim: int = 64
    hidden_dim: int = 256
    num_heads: int = 4
    dropout: float = 0.3
    num_cfc_layers: int = 1


class CrossAttentionFusion(nn.Module):
    """
    Cross-attention layer where workers "vote" on consensus.
    """
    
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
    """
    Fuses worker embeddings + logits into a single context vector.
    """
    
    def __init__(
        self, 
        num_workers: int, 
        worker_dim: int, 
        num_classes: int = 3,
        hidden_dim: int = 256
    ):
        super().__init__()
        
        self.num_workers = num_workers
        self.cross_attn = CrossAttentionFusion(
            embed_dim=worker_dim,
            num_heads=4,
            dropout=0.1
        )
        
        self.confidence_proj = nn.Linear(num_classes, 1)
        fusion_input_dim = num_workers * worker_dim + num_workers
        
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.LayerNorm(hidden_dim)
        )
        
    def forward(
        self, 
        embeddings: torch.Tensor, 
        logits: torch.Tensor
    ) -> torch.Tensor:
        batch_size = embeddings.size(0)
        attended = self.cross_attn(embeddings)
        
        probs = F.softmax(logits, dim=-1)
        confidence = self.confidence_proj(probs)
        confidence = torch.sigmoid(confidence)
        
        weighted = attended * confidence
        
        flat_emb = weighted.view(batch_size, -1)
        flat_conf = confidence.view(batch_size, -1)
        
        combined = torch.cat([flat_emb, flat_conf], dim=-1)
        context = self.fusion(combined)
        
        return context


class SwarmCoordinator(nn.Module):
    """
    The central brain of the Swarm Intelligence architecture.
    """
    
    def __init__(self, config: CoordinatorConfig = None):
        super().__init__()
        
        if config is None:
            config = CoordinatorConfig()
        
        self.config = config
        
        # Consensus fusion
        self.fusion = ConsensusFusion(
            num_workers=config.num_workers,
            worker_dim=config.worker_dim,
            num_classes=3,
            hidden_dim=config.hidden_dim
        )
        
        # Phase 5 (The Dojo): Macro Fusion Layer
        self.macro_fusion = nn.Linear(config.hidden_dim + (config.hidden_dim // 4), config.hidden_dim)
        
        # CfC refinement
        self.cfc = CfCCell(
            input_dim=config.hidden_dim,
            hidden_dim=config.hidden_dim,
            dropout=config.dropout
        )
        
        # Expert Heads
        self.long_expert = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, 1)
        )
        
        self.short_expert = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, 1)
        )
        
        # NEW: Multi-class Policy Head (0: Wait, 1: Long, 2: Short)
        self.policy_head = nn.Linear(config.hidden_dim, 3)
        
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
    def forward(
        self, 
        embeddings: torch.Tensor, 
        worker_logits: torch.Tensor,
        dt: Optional[torch.Tensor] = None,
        hx: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # 1. Fuse worker outputs
        context = self.fusion(embeddings, worker_logits)
        
        # 2. Refine through CfC
        context_seq = context.unsqueeze(1)
        _, h_refined = self.cfc(context_seq, dt, hx)
        
        # 3. Macro Fusion (Fuse micro execution with macro trend)
        if hasattr(self, 'macro_bias') and self.macro_bias is not None:
            # macro_bias is injected from SwarmBrain [B, Dim]
            # h_refined is the hidden state [B, hidden_dim]
            h_refined = self.macro_fusion(torch.cat([h_refined, self.macro_bias], dim=-1))
            h_refined = F.silu(h_refined)
            
        # 4. Expert heads
        long_score = self.long_expert(h_refined)
        short_score = self.short_expert(h_refined)
        
        # 4. Policy logits
        policy_logits = self.policy_head(h_refined)
        
        # 5. Differential logit for backward compatibility
        diff_logit = long_score - short_score
        
        return policy_logits, diff_logit, long_score, short_score


class SwarmBrain(nn.Module):
    """
    Complete Swarm Intelligence model.
    """
    
    def __init__(
        self, 
        worker_config: WorkerConfig = None,
        coordinator_config: CoordinatorConfig = None
    ):
        super().__init__()
        
        if worker_config is None:
            worker_config = WorkerConfig()
        
        if coordinator_config is None:
            coordinator_config = CoordinatorConfig(
                num_workers=5,
                worker_dim=worker_config.hidden_dim
            )
        
        self.worker_config = worker_config
        self.coordinator_config = coordinator_config
        
        # Phase 5 (The Dojo): Macro Context Encoder (Processa candles de H1)
        self.macro_encoder = nn.Sequential(
            nn.Linear(worker_config.input_dim, worker_config.hidden_dim),
            nn.SiLU(),
            nn.Linear(worker_config.hidden_dim, coordinator_config.hidden_dim // 4) 
        )
        
        self.swarm = WorkerSwarm(worker_config)
        self.coordinator = SwarmCoordinator(coordinator_config)
        
    def forward(
        self, 
        x_dict: dict,
        hx: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # 0. Process Macro Context (if provided, else dummy)
        # H1 is typically present in training (from _simulate_timeframes)
        # In live inference, we'll need to pass it explicitly in x_dict
        x_macro_h1 = x_dict.get("H1")
        if x_macro_h1 is not None and len(x_macro_h1.shape) == 3:
            macro_feat = self.macro_encoder(x_macro_h1) # [B, L, Dim]
            # Global macro state (average over the H1 sequence)
            self.coordinator.macro_bias = macro_feat.mean(dim=1) # [B, Dim]
        else:
            self.coordinator.macro_bias = None

        # 1. Process through worker swarm
        embeddings, worker_logits, _ = self.swarm(x_dict)
        
        # 2. Coordinate and decide
        coord_dt = x_dict.get("coord_dt")
        policy_logits, diff_logit, long_score, short_score = self.coordinator(
            embeddings, worker_logits, coord_dt, hx
        )
        
        # Return policy_logits as the primary decision logit
        return policy_logits, worker_logits, long_score, short_score
    
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
