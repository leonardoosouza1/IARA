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

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple, List


@dataclass
class WorkerConfig:
    """Configuration for a specialist worker."""
    input_dim: int = 12  # Standard for v9.7
    hidden_dim: int = 64
    dropout: float = 0.2
    backbone: str = "cfc"  # Closed-form Continuous-time


class CfCCell(nn.Module):
    """
    Closed-form Continuous-time (CfC) layer.
    A computationally efficient Liquid Neural Network.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # RNN-like projection
        self.x_proj = nn.Linear(input_dim, hidden_dim)
        self.h_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Liquid Gating Parameters
        self.gate_f = nn.Linear(hidden_dim, hidden_dim)
        self.gate_i = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self, 
        x: torch.Tensor, 
        dt: Optional[torch.Tensor] = None,
        hx: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = x.size()
        
        if hx is None:
            hx = torch.zeros(batch_size, self.hidden_dim).to(x.device)
            
        if dt is None:
            dt = torch.ones(batch_size, seq_len, 1).to(x.device)
            
        h_t = hx
        outputs = []
        
        for t in range(seq_len):
            xt = x[:, t, :]
            dt_t = dt[:, t, :]
            
            x_in = self.x_proj(xt)
            f_gate = torch.sigmoid(self.gate_f(h_t))
            i_gate = torch.tanh(self.gate_i(x_in + self.h_proj(h_t)))
            
            decay = torch.exp(-dt_t * f_gate)
            h_t = h_t * decay + i_gate * (1.0 - decay)
            h_t = self.dropout(h_t)
            
            outputs.append(h_t.unsqueeze(1))
            
        out = torch.cat(outputs, dim=1)
        return out, h_t


class LightningWorker(nn.Module):
    """
    A specialist worker optimized for a single timeframe.
    """
    
    def __init__(self, config: WorkerConfig, timeframe: str = "M5"):
        super().__init__()
        self.config = config
        self.timeframe = timeframe
        
        # 🌊 Phase 21: Temporal Self-Attention
        # A projection layer ensures robust Multi-head division regardless of how many features survive the ExtraTrees Pruning
        attn_dim = 64
        self.attn_proj = nn.Linear(config.input_dim, attn_dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=attn_dim,
            num_heads=4,
            dropout=config.dropout,
            batch_first=True
        )
        self.attn_norm = nn.LayerNorm(attn_dim)
        
        # Liquid Backbone
        self.cfc = CfCCell(
            input_dim=attn_dim,
            hidden_dim=config.hidden_dim,
            dropout=config.dropout
        )
        
        # Residual projection from the attended dims to hidden
        if attn_dim != config.hidden_dim:
            self.res_proj = nn.Linear(attn_dim, config.hidden_dim)
        else:
            self.res_proj = nn.Identity()
            
        # Final decision head (auxiliary)
        self.head = nn.Linear(config.hidden_dim, 3)  # Long, Short, Neutral
        
    def forward(
        self, 
        x: torch.Tensor, 
        dt: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        # 1. Base Projection (e.g. 18 pruned dimensions -> 64 embedding dimensions)
        x_proj = self.attn_proj(x)
        
        # 1.5 Pre-LayerNorm (CRITICAL for FP16 Autocast Stability!)
        x_norm = self.attn_norm(x_proj)
        
        # 2. Temporal Self-Attention (Locating crucial wicks/anomalies in the sequence)
        attn_out, _ = self.self_attn(x_norm, x_norm, x_norm, need_weights=False)
        x_attn = x_proj + F.dropout(attn_out, p=self.config.dropout, training=self.training)
        
        # 3. Process through Liquid Continuous-Time Backbone
        _, h_t = self.cfc(x_attn, dt)
        
        # Residual connection from LAST timestep of the attended sequence
        residual = self.res_proj(x_attn[:, -1, :])
        
        # Skip connection
        h_t = h_t + residual
        
        # Embedding is the final hidden state after skip connection
        embedding = h_t
        
        # Logits for auxiliary loss (LONG=0, SHORT=1, NEUTRAL=2)
        logits = self.head(embedding)
        
        return embedding, logits
    
    @property
    def embedding_dim(self) -> int:
        return self.config.hidden_dim


class WorkerSwarm(nn.Module):
    """
    Collection of LightningWorkers for all timeframes.
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
        embeddings = []
        all_logits = []
        order = []
        
        # Identify device and batch size from first available input
        first_tf = next(iter(x_dict.values()))
        batch_size = first_tf.size(0)
        device = first_tf.device
        
        for tf in self.TIMEFRAMES:
            if tf in x_dict:
                dt = x_dict.get(f"{tf}_dt")
                emb, logits = self.workers[tf](x_dict[tf], dt=dt)
                embeddings.append(emb.unsqueeze(1))
                all_logits.append(logits.unsqueeze(1))
            else:
                zero_emb = torch.zeros(batch_size, 1, self.embedding_dim).to(device)
                zero_logits = torch.zeros(batch_size, 1, 3).to(device)
                embeddings.append(zero_emb)
                all_logits.append(zero_logits)
            
            order.append(tf)
        
        embeddings = torch.cat(embeddings, dim=1)
        all_logits = torch.cat(all_logits, dim=1)
        
        return embeddings, all_logits, order
    
    @property
    def num_workers(self) -> int:
        return len(self.TIMEFRAMES)
    
    @property
    def embedding_dim(self) -> int:
        return self.config.hidden_dim


# Quick test
if __name__ == "__main__":
    config = WorkerConfig(input_dim=12, hidden_dim=64)
    swarm = WorkerSwarm(config)
    x_dict = {"M5": torch.randn(4, 64, 12)}
    embeddings, logits, _ = swarm(x_dict)
    print(f"Test Successful. Embeddings: {embeddings.shape}, Logits: {logits.shape}")
