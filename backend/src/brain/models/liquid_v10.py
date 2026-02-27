"""
I.A.R.A. v11.0 - Swarm Coordinator (Full Spectrum Brain)
=========================================================
Advanced central brain for I.A.R.A. v11.0.
Fuses multi-timeframe Workers with 3 new intelligence layers:

Architecture:
1. Cross-Attention: Multi-head worker consensus
2. Consensus Fusion: Confidence-weighted context
3. Sensory Bridge: LOB L2 Order Book encoder (OBI + VPIN)
4. Regime Gate: HMM regime embedding (Trend/Choppy/Crisis)
5. On-Chain GNN: Sparse graph whale flow embedding
6. Liquid CfC: Time-continuous decision refinement
7. Expert Heads: Long/Short/Policy outputs
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.brain.models.workers import WorkerSwarm, WorkerConfig, CfCCell
from src.brain.models.lob_encoder import LOBEncoder
from src.brain.models.gnn_onchain import OnChainGNN


@dataclass
class CoordinatorConfig:
    """Configuration for SwarmCoordinator v11.0."""
    num_workers: int = 5
    worker_dim: int = 128            # v10.0 Higher capacity
    hidden_dim: int = 512            # v10.0 Higher capacity
    num_heads: int = 8               # More heads for complex L2 patterns
    dropout: float = 0.2
    num_cfc_layers: int = 2          # Deeper liquid refinement
    # v11.0 New features
    lob_input_dim: int = 6           # LOB features: OBI, VPIN, spread, depth, walls
    lob_hidden_dim: int = 64         # LOB encoder hidden
    num_regimes: int = 4             # 3 states + 1 unknown
    regime_embed_dim: int = 32       # Regime embedding size
    gnn_output_dim: int = 32         # GNN graph embedding size
    gnn_node_features: int = 3       # On-chain node features


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
        if config is None:
            config = CoordinatorConfig()
        self.config = config

        # === Core Fusion (unchanged from v10) ===
        self.fusion = ConsensusFusion(
            num_workers=config.num_workers,
            worker_dim=config.worker_dim,
            num_classes=3,
            hidden_dim=config.hidden_dim
        )

        # === v11: LOB Sensory Bridge ===
        self.lob_encoder = LOBEncoder(
            input_dim=config.lob_input_dim,
            hidden_dim=config.lob_hidden_dim,
            output_dim=config.lob_hidden_dim,
            dropout=config.dropout,
        )

        # === v11: Regime Embedding (HMM state) ===
        self.regime_embedding = nn.Embedding(config.num_regimes, config.regime_embed_dim)

        # === v11: On-Chain GNN ===
        self.onchain_gnn = OnChainGNN(
            node_features=config.gnn_node_features,
            hidden_dim=config.gnn_output_dim,
            output_dim=config.gnn_output_dim,
            dropout=config.dropout,
        )

        # === Context Fusion: Combine all intelligence sources ===
        # hidden_dim (base) + lob_hidden (LOB) + regime_embed (HMM) + gnn_output (GNN)
        augmented_dim = config.hidden_dim + config.lob_hidden_dim + config.regime_embed_dim + config.gnn_output_dim
        self.context_bridge = nn.Sequential(
            nn.Linear(augmented_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.LayerNorm(config.hidden_dim),
        )

        # === CfC & Expert Heads (same structure) ===
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
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
    def forward(
        self,
        embeddings: torch.Tensor,
        worker_logits: torch.Tensor,
        dt: Optional[torch.Tensor] = None,
        hx: Optional[torch.Tensor] = None,
        lob_features: Optional[torch.Tensor] = None,
        regime_state: Optional[torch.Tensor] = None,
        gnn_node_features: Optional[torch.Tensor] = None,
        gnn_adj_sparse: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with all intelligence layers.

        Args:
            embeddings: [B, num_workers, worker_dim] Worker embeddings
            worker_logits: [B, num_workers, 3] Worker predictions
            dt: Optional time deltas for CfC
            hx: Optional hidden state for CfC
            lob_features: Optional [B, 6] LOB microstructure features
            regime_state: Optional [B] integer regime states (0-3)
            gnn_node_features: Optional [N, 3] on-chain node features
            gnn_adj_sparse: Optional [N, N] sparse adjacency matrix
        """
        batch_size = embeddings.size(0)
        device = embeddings.device

        # 1. Core consensus from workers
        context = self.fusion(embeddings, worker_logits)  # [B, hidden_dim]

        # 2. LOB encoding
        if lob_features is not None:
            lob_emb = self.lob_encoder(lob_features)  # [B, lob_hidden]
        else:
            lob_emb = torch.zeros(batch_size, self.config.lob_hidden_dim, device=device)

        # 3. Regime embedding
        if regime_state is not None:
            regime_state = regime_state.clamp(0, self.config.num_regimes - 1)
            regime_emb = self.regime_embedding(regime_state)  # [B, regime_embed_dim]
        else:
            # Unknown regime
            unknown = torch.full((batch_size,), self.config.num_regimes - 1, dtype=torch.long, device=device)
            regime_emb = self.regime_embedding(unknown)

        # 4. GNN on-chain embedding
        if gnn_node_features is not None and gnn_adj_sparse is not None:
            gnn_emb = self.onchain_gnn(gnn_node_features, gnn_adj_sparse)  # [1, gnn_dim]
            gnn_emb = gnn_emb.expand(batch_size, -1)  # Broadcast to batch
        else:
            gnn_emb = torch.zeros(batch_size, self.config.gnn_output_dim, device=device)

        # 5. Fuse all intelligence sources
        augmented = torch.cat([context, lob_emb, regime_emb, gnn_emb], dim=-1)
        fused_context = self.context_bridge(augmented)  # [B, hidden_dim]

        # 6. CfC temporal refinement
        _, h_refined = self.cfc(fused_context.unsqueeze(1), dt, hx)

        # 7. Expert heads
        long_out = self.long_expert(h_refined)
        short_out = self.short_expert(h_refined)

        return (
            self.policy_head(h_refined),       # [B, 3] policy logits
            long_out - short_out,               # [B, 1] directional logit
            long_out,                           # [B, 1] long expert
            short_out,                          # [B, 1] short expert
        )


class SwarmBrain(nn.Module):
    """
    Complete Swarm Intelligence model v11.0.
    Integrates Workers + Coordinator + LOB + HMM + GNN.
    """

    def __init__(self, worker_config: WorkerConfig = None, coordinator_config: CoordinatorConfig = None):
        super().__init__()
        if worker_config is None:
            worker_config = WorkerConfig(hidden_dim=128)
        if coordinator_config is None:
            coordinator_config = CoordinatorConfig(num_workers=5, worker_dim=worker_config.hidden_dim)
        self.worker_config = worker_config
        self.coordinator_config = coordinator_config
        self.swarm = WorkerSwarm(worker_config)
        self.coordinator = SwarmCoordinator(coordinator_config)
        
    def forward(
        self,
        x_dict: dict,
        hx: Optional[torch.Tensor] = None,
        compute_telemetry: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Full forward pass.

        x_dict keys:
            "M5", "M15", "M30", "H1", "H4":  [B, seq_len, input_dim] per-timeframe tensors
            "coord_dt": Optional time deltas
            "lob": Optional [B, 6] LOB features
            "regime": Optional [B] regime state integers
            "gnn_nodes": Optional [N, 3] node features
            "gnn_adj": Optional [N, N] sparse adjacency
        """
        # Workers
        embeddings, worker_logits, _ = self.swarm(x_dict)

        # Coordinator with augmented inputs
        policy, directional, long_out, short_out = self.coordinator(
            embeddings=embeddings,
            worker_logits=worker_logits,
            dt=x_dict.get("coord_dt"),
            hx=hx,
            lob_features=x_dict.get("lob"),
            regime_state=x_dict.get("regime"),
            gnn_node_features=x_dict.get("gnn_nodes"),
            gnn_adj_sparse=x_dict.get("gnn_adj"),
        )

        # Telemetry placeholders (for backward compat with training loop)
        attn_weights = None
        confidence = None

        return policy, worker_logits, long_out, short_out, attn_weights, confidence
    
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
