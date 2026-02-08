"""
I.A.R.A. v9 - Lightning Workers (Swarm Intelligence)
=====================================================
Ultra-light Liquid Neural Network specialists for each timeframe.

Each worker learns patterns specific to its timeframe:
- M5: Micro-structure, scalping patterns, noise filtering
- M15: Short-term momentum, minor S/R levels
- M30: Intraday trends, session patterns
- H1: Swing trading signals, major S/R
- H4: Macro trends, position sizing context

Reference: Hasani et al. (2022) "Closed-form Continuous-time Neural Networks"
"""

from dataclasses import dataclass
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class WorkerConfig:
    """Configuration for LightningWorker."""
    input_dim: int = 12
    hidden_dim: int = 32
    num_classes: int = 3  # LONG, SHORT, NEUTRAL
    dropout: float = 0.1
    use_layer_norm: bool = True


class CfCCell(nn.Module):
    """
    Closed-form Continuous (CfC) Cell - Ultra-lightweight version.
    
    Symmetric gates with tanh activation for stable gradients.
    Optimized for memory efficiency on 12GB VRAM.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Fused Linear for efficiency (input + hidden -> candidate + gate)
        self.fused_ih = nn.Linear(input_dim, hidden_dim * 2)
        self.fused_hh = nn.Linear(hidden_dim, hidden_dim * 2, bias=False)
        
        # Regularization
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, hx: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process sequence through CfC dynamics.
        
        Args:
            x: Input tensor [batch, seq_len, input_dim]
            hx: Optional initial hidden state [batch, hidden_dim]
            
        Returns:
            output: All hidden states [batch, seq_len, hidden_dim]
            h_n: Final hidden state [batch, hidden_dim]
        """
        batch_size, seq_len, _ = x.shape
        device = x.device
        dtype = x.dtype
        
        if hx is None:
            hx = torch.zeros(batch_size, self.hidden_dim, device=device, dtype=dtype)
        
        outputs = []
        h_t = hx
        
        for t in range(seq_len):
            x_t = x[:, t, :]
            
            # Fused computation
            gates_i = self.fused_ih(x_t)
            gates_h = self.fused_hh(h_t)
            gates = gates_i + gates_h
            
            # Split into candidate and gate
            h_cand, gate_logits = gates.chunk(2, dim=-1)
            
            # Symmetric CfC dynamics
            h_cand = torch.tanh(h_cand)
            h_cand = self.dropout(h_cand)
            gate = torch.tanh(gate_logits)
            
            # Update rule: h_new = h + gate * (candidate - h)
            h_t = h_t + gate * (h_cand - h_t)
            
            # Stability: Normalize and clip
            h_t = self.layer_norm(h_t)
            h_t = torch.clamp(h_t, min=-1.0, max=1.0)
            
            outputs.append(h_t.unsqueeze(1))
        
        output = torch.cat(outputs, dim=1)
        return output, h_t


class LightningWorker(nn.Module):
    """
    Ultra-light specialist for a single timeframe.
    
    Features:
    - CfC backbone for efficient sequence processing
    - Dual output: embedding (for Coordinator) + logits (for Aux Loss)
    - Memory-efficient: ~100KB VRAM per worker
    
    Args:
        config: WorkerConfig with hyperparameters
        timeframe: String identifier ("M5", "M15", etc.)
    """
    
    def __init__(self, config: WorkerConfig = None, timeframe: str = "M5"):
        super().__init__()
        
        if config is None:
            config = WorkerConfig()
        
        self.config = config
        self.timeframe = timeframe
        
        # Feature projection (12 -> hidden)
        self.input_proj = nn.Linear(config.input_dim, config.hidden_dim)
        
        # CfC backbone
        self.backbone = CfCCell(
            input_dim=config.hidden_dim, 
            hidden_dim=config.hidden_dim,
            dropout=config.dropout
        )
        
        # Classification head for auxiliary loss
        self.head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, config.num_classes)
        )
        
        self._init_weights()
        
    def _init_weights(self):
        """Xavier initialization for stable training."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(
        self, 
        x: torch.Tensor, 
        hx: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process timeframe data.
        
        Args:
            x: Input features [batch, seq_len, input_dim]
            hx: Optional initial hidden state
            
        Returns:
            embedding: Final hidden state for Coordinator [batch, hidden_dim]
            logits: Class predictions for Auxiliary Loss [batch, num_classes]
        """
        # Project input
        x = self.input_proj(x)
        
        # Process through CfC
        output, h_n = self.backbone(x, hx)
        
        # Embedding is the final hidden state
        embedding = h_n
        
        # Logits for auxiliary loss (LONG=0, SHORT=1, NEUTRAL=2)
        logits = self.head(embedding)
        
        return embedding, logits
    
    @property
    def embedding_dim(self) -> int:
        return self.config.hidden_dim
    
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class WorkerSwarm(nn.Module):
    """
    Collection of LightningWorkers for all timeframes.
    
    Manages the ensemble of specialists and provides
    unified interface for the Coordinator.
    """
    
    TIMEFRAMES = ["M5", "M15", "M30", "H1", "H4"]
    
    def __init__(self, config: WorkerConfig = None):
        super().__init__()
        
        if config is None:
            config = WorkerConfig()
        
        self.config = config
        
        # Create worker for each timeframe
        self.workers = nn.ModuleDict({
            tf: LightningWorker(config, timeframe=tf)
            for tf in self.TIMEFRAMES
        })
        
    def forward(
        self, 
        x_dict: dict
    ) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        """
        Process all timeframes in parallel.
        
        Args:
            x_dict: Dictionary mapping timeframe -> tensor [batch, seq_len, features]
            
        Returns:
            embeddings: Stacked embeddings [batch, num_workers, hidden_dim]
            all_logits: Stacked logits [batch, num_workers, num_classes]
            timeframe_order: List of timeframe names in order
        """
        embeddings = []
        all_logits = []
        order = []
        
        for tf in self.TIMEFRAMES:
            if tf in x_dict:
                emb, logits = self.workers[tf](x_dict[tf])
                embeddings.append(emb.unsqueeze(1))
                all_logits.append(logits.unsqueeze(1))
                order.append(tf)
        
        if not embeddings:
            raise ValueError("No valid timeframe data provided")
        
        embeddings = torch.cat(embeddings, dim=1)  # [batch, num_workers, hidden]
        all_logits = torch.cat(all_logits, dim=1)  # [batch, num_workers, 3]
        
        return embeddings, all_logits, order
    
    @property
    def num_workers(self) -> int:
        return len(self.TIMEFRAMES)
    
    @property
    def embedding_dim(self) -> int:
        return self.config.hidden_dim
    
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Quick test
if __name__ == "__main__":
    print("Testing LightningWorker...")
    
    config = WorkerConfig(input_dim=12, hidden_dim=32)
    worker = LightningWorker(config, timeframe="M5")
    
    # Simulate batch of 64 sequences, 64 timesteps, 12 features
    x = torch.randn(64, 64, 12)
    
    embedding, logits = worker(x)
    
    print(f"  Input shape: {x.shape}")
    print(f"  Embedding shape: {embedding.shape}")
    print(f"  Logits shape: {logits.shape}")
    print(f"  Parameters: {worker.count_parameters():,}")
    print("  ✅ LightningWorker OK!")
    
    print("\nTesting WorkerSwarm...")
    swarm = WorkerSwarm(config)
    
    # Simulate multi-timeframe input
    x_dict = {
        "M5": torch.randn(64, 64, 12),
        "M15": torch.randn(64, 32, 12),
        "M30": torch.randn(64, 16, 12),
        "H1": torch.randn(64, 8, 12),
        "H4": torch.randn(64, 4, 12),
    }
    
    embeddings, all_logits, order = swarm(x_dict)
    
    print(f"  Embeddings shape: {embeddings.shape}")
    print(f"  All logits shape: {all_logits.shape}")
    print(f"  Timeframe order: {order}")
    print(f"  Total parameters: {swarm.count_parameters():,}")
    print("  ✅ WorkerSwarm OK!")
