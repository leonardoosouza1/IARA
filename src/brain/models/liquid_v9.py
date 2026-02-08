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
    worker_dim: int = 32
    hidden_dim: int = 128
    num_heads: int = 4
    dropout: float = 0.3
    num_cfc_layers: int = 1


class CrossAttentionFusion(nn.Module):
    """
    Cross-attention layer where workers "vote" on consensus.
    
    Each worker embedding attends to all other workers
    to determine relative importance.
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
        """
        Args:
            x: Worker embeddings [batch, num_workers, embed_dim]
            
        Returns:
            Context-aware embeddings [batch, num_workers, embed_dim]
        """
        # Self-attention among workers
        attn_out, _ = self.attention(x, x, x, need_weights=False)
        
        # Residual + Norm
        x = self.norm(x + self.dropout(attn_out))
        
        return x


class ConsensusFusion(nn.Module):
    """
    Fuses worker embeddings + logits into a single context vector.
    
    Combines:
    1. Attended embeddings (what each worker "thinks")
    2. Worker confidence (how sure each worker is)
    """
    
    def __init__(
        self, 
        num_workers: int, 
        worker_dim: int, 
        num_classes: int = 3,
        hidden_dim: int = 128
    ):
        super().__init__()
        
        self.num_workers = num_workers
        
        # Cross-attention for worker consensus
        self.cross_attn = CrossAttentionFusion(
            embed_dim=worker_dim,
            num_heads=4,
            dropout=0.1
        )
        
        # Confidence weighting (learn to trust/distrust workers)
        self.confidence_proj = nn.Linear(num_classes, 1)
        
        # Fusion MLP
        # Input: (num_workers * worker_dim) + (num_workers * 1)
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
        """
        Args:
            embeddings: Worker embeddings [batch, num_workers, worker_dim]
            logits: Worker predictions [batch, num_workers, num_classes]
            
        Returns:
            context: Fused context vector [batch, hidden_dim]
        """
        batch_size = embeddings.size(0)
        
        # 1. Cross-attention: Workers discuss
        attended = self.cross_attn(embeddings)  # [batch, workers, dim]
        
        # 2. Confidence scores from logits (max probability = confidence)
        probs = F.softmax(logits, dim=-1)  # [batch, workers, 3]
        confidence = self.confidence_proj(probs)  # [batch, workers, 1]
        confidence = torch.sigmoid(confidence)  # Normalize to [0, 1]
        
        # 3. Weight embeddings by confidence
        weighted = attended * confidence  # [batch, workers, dim]
        
        # 4. Flatten and fuse
        flat_emb = weighted.view(batch_size, -1)  # [batch, workers * dim]
        flat_conf = confidence.view(batch_size, -1)  # [batch, workers]
        
        combined = torch.cat([flat_emb, flat_conf], dim=-1)
        
        context = self.fusion(combined)  # [batch, hidden_dim]
        
        return context


class SwarmCoordinator(nn.Module):
    """
    The central brain of the Swarm Intelligence architecture.
    
    Receives worker embeddings/logits and produces final trading decision.
    
    Features:
    - Cross-attention fusion for worker consensus
    - CfC refinement layer for temporal context
    - Dual expert heads (Long/Short) for asymmetric learning
    """
    
    def __init__(self, config: CoordinatorConfig = None):
        super().__init__()
        
        if config is None:
            config = CoordinatorConfig()
        
        self.config = config
        
        # Consensus fusion module
        self.fusion = ConsensusFusion(
            num_workers=config.num_workers,
            worker_dim=config.worker_dim,
            num_classes=3,
            hidden_dim=config.hidden_dim
        )
        
        # CfC refinement (single step, uses context as sequence of 1)
        self.cfc = CfCCell(
            input_dim=config.hidden_dim,
            hidden_dim=config.hidden_dim,
            dropout=config.dropout
        )
        
        # Expert Heads (Mixture of Experts lite)
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
        
        self._init_weights()
        
    def _init_weights(self):
        """Xavier init with zero bias for experts."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        # Initialize experts to output 0 (neutral)
        nn.init.zeros_(self.long_expert[-1].bias)
        nn.init.zeros_(self.short_expert[-1].bias)
        
    def forward(
        self, 
        embeddings: torch.Tensor, 
        worker_logits: torch.Tensor,
        hx: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            embeddings: Worker embeddings [batch, num_workers, worker_dim]
            worker_logits: Worker predictions [batch, num_workers, 3]
            hx: Optional hidden state for CfC
            
        Returns:
            logits: Final prediction logits [batch, 1]
            long_score: Long expert confidence [batch, 1]
            short_score: Short expert confidence [batch, 1]
        """
        # 1. Fuse worker outputs
        context = self.fusion(embeddings, worker_logits)  # [batch, hidden]
        
        # 2. Refine through CfC (treat context as single-step sequence)
        context_seq = context.unsqueeze(1)  # [batch, 1, hidden]
        _, h_refined = self.cfc(context_seq, hx)  # [batch, hidden]
        
        # 3. Expert heads
        long_score = self.long_expert(h_refined)
        short_score = self.short_expert(h_refined)
        
        # 4. Differential logit (Long - Short)
        logits = long_score - short_score
        
        return logits, long_score, short_score


class SwarmBrain(nn.Module):
    """
    Complete Swarm Intelligence model.
    
    Combines WorkerSwarm + SwarmCoordinator into a single
    end-to-end trainable architecture.
    
    Memory estimate (fp16):
    - 5 Workers: ~160KB * 5 = 800KB
    - Coordinator: ~2MB
    - Activations (batch=64): ~5MB
    - Total: ~8MB (well under 12GB limit)
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
        
        # The swarm of specialists
        self.swarm = WorkerSwarm(worker_config)
        
        # The coordinator brain
        self.coordinator = SwarmCoordinator(coordinator_config)
        
    def forward(
        self, 
        x_dict: dict,
        hx: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        End-to-end forward pass.
        
        Args:
            x_dict: Dictionary mapping timeframe -> tensor [batch, seq_len, features]
            hx: Optional hidden state for Coordinator CfC
            
        Returns:
            logits: Final prediction [batch, 1]
            worker_logits: All worker predictions [batch, num_workers, 3]
            long_score: Long expert score [batch, 1]
            short_score: Short expert score [batch, 1]
        """
        # 1. Process through worker swarm
        embeddings, worker_logits, _ = self.swarm(x_dict)
        
        # 2. Coordinate and decide
        logits, long_score, short_score = self.coordinator(
            embeddings, worker_logits, hx
        )
        
        return logits, worker_logits, long_score, short_score
    
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_worker_votes(self, worker_logits: torch.Tensor) -> dict:
        """
        Analyze worker voting pattern.
        
        Args:
            worker_logits: [batch, num_workers, 3]
            
        Returns:
            Dictionary with vote counts and agreement score
        """
        # Get predictions per worker
        votes = worker_logits.argmax(dim=-1)  # [batch, workers]
        
        # Count votes (0=LONG, 1=SHORT, 2=NEUTRAL)
        n_long = (votes == 0).float().sum(dim=-1)
        n_short = (votes == 1).float().sum(dim=-1)
        n_neutral = (votes == 2).float().sum(dim=-1)
        
        # Agreement score (how much workers agree)
        max_votes = torch.stack([n_long, n_short, n_neutral], dim=-1).max(dim=-1).values
        agreement = max_votes / self.swarm.num_workers
        
        return {
            "n_long": n_long,
            "n_short": n_short,
            "n_neutral": n_neutral,
            "agreement": agreement
        }


# Quick test
if __name__ == "__main__":
    print("Testing SwarmBrain...")
    
    # Configure
    worker_cfg = WorkerConfig(input_dim=12, hidden_dim=32)
    coord_cfg = CoordinatorConfig(num_workers=5, worker_dim=32, hidden_dim=128)
    
    # Create brain
    brain = SwarmBrain(worker_cfg, coord_cfg)
    
    # Simulate multi-timeframe input
    x_dict = {
        "M5": torch.randn(64, 64, 12),
        "M15": torch.randn(64, 32, 12),
        "M30": torch.randn(64, 16, 12),
        "H1": torch.randn(64, 8, 12),
        "H4": torch.randn(64, 4, 12),
    }
    
    # Forward pass
    logits, worker_logits, long_score, short_score = brain(x_dict)
    
    print(f"  Final logits shape: {logits.shape}")
    print(f"  Worker logits shape: {worker_logits.shape}")
    print(f"  Long score shape: {long_score.shape}")
    print(f"  Short score shape: {short_score.shape}")
    print(f"  Total parameters: {brain.count_parameters():,}")
    
    # Test vote analysis
    votes = brain.get_worker_votes(worker_logits)
    print(f"  Sample agreement: {votes['agreement'][0]:.2f}")
    
    # Memory estimate
    param_size = sum(p.numel() * p.element_size() for p in brain.parameters())
    print(f"  Parameter memory: {param_size / 1024:.2f} KB")
    print("  ✅ SwarmBrain OK!")
