"""
I.A.R.A. v8.5 - Modulated Liquid Neural Network
===============================================
Refactored for Trinity Engine (Protocol Trinity).
Adds support for 'external_modulation' (LLM Context Injection).

Based on v6.0 architecture by Hasani et al.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class LiquidConfig:
    """Configuration for IARA v8.5 Trinity."""
    input_dim: int = 16
    hidden_dim: int = 128
    num_heads: int = 4
    dropout: float = 0.5
    output_dim: int = 1
    max_seq_len: int = 256
    num_cfc_layers: int = 2

class LearnedPositionalEncoding(nn.Module):
    """Learned Positional Encoding for long sequences."""
    def __init__(self, d_model: int, max_len: int = 256, dropout: float = 0.1):
        super().__init__()
        self.position_embeddings = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)
        
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        self.position_embeddings.weight.data.copy_(pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        return self.dropout(x + self.position_embeddings(positions))

class FeatureAttention(nn.Module):
    """Feature-level attention gate."""
    def __init__(self, input_dim: int, dropout: float = 0.1):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim * 2, input_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gate(x)

class TemporalSelfAttention(nn.Module):
    """Multi-head self-attention."""
    def __init__(self, d_model: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attention(x, x, x, need_weights=False)
        return self.norm(x + self.dropout(attn_out))

class ModulatedCfCCell(nn.Module):
    """
    Symmetric CfC Cell v8.5 [MODULATED].
    
    Change:
    - forward() accepts `external_modulation` tensor.
    - Modulation is added to the system inputs before gating.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        self.linear_input = nn.Linear(input_dim, hidden_dim)
        self.linear_hidden = nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        self.gate_input = nn.Linear(input_dim, hidden_dim)
        self.gate_hidden = nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.Tanh()

    def forward(self, input_: torch.Tensor, hidden: torch.Tensor, external_modulation: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Standard CfC Input
        linear_part = self.linear_input(input_) + self.linear_hidden(hidden)
        gate_part = self.gate_input(input_) + self.gate_hidden(hidden)
        
        # [v8.5] Injection Point: External Modulation (LLM Context)
        # We inject it into the linear path (forcing the ODE to drift)
        if external_modulation is not None:
            # external_modulation: [batch, hidden_dim] (projected before call)
            # Ensure shape match. If modulation is [batch, hidden], we rely on broadcasting for input [batch, hidden]
            linear_part = linear_part + external_modulation
            # Optionally modulate gate too? For now, just linear dynamics.
        
        h_cand = self.activation(linear_part)
        h_cand = self.dropout(h_cand)
        
        gate = torch.tanh(gate_part)
        update = gate * (h_cand - hidden)
        h_new = hidden + update
        
        h_new = self.layer_norm(h_new)
        h_new = torch.clamp(h_new, min=-1.0, max=1.0)
        
        return h_new

class ModulatedIARA(nn.Module):
    """
    I.A.R.A. v8.5 - Modulated Backbone
    """
    
    def __init__(self, config: LiquidConfig):
        super().__init__()
        self.config = config
        
        self.input_proj = nn.Linear(config.input_dim, config.hidden_dim)
        self.pos_encoding = LearnedPositionalEncoding(config.hidden_dim, max_len=config.max_seq_len, dropout=config.dropout*0.5)
        self.self_attention = TemporalSelfAttention(config.hidden_dim, num_heads=config.num_heads, dropout=config.dropout)
        self.feature_attention = FeatureAttention(config.hidden_dim, dropout=config.dropout*0.5)
        
        self.attention_proj = nn.Linear(config.hidden_dim, config.input_dim)
        self.input_norm = nn.LayerNorm(config.input_dim)
        
        # [v8.5] Modulated CfC Layers
        self.cfc_layers = nn.ModuleList([
            ModulatedCfCCell(config.input_dim if i == 0 else config.hidden_dim, 
                             config.hidden_dim, 
                             dropout=config.dropout)
            for i in range(config.num_cfc_layers)
        ])
        
        # MOE Heads
        self.long_expert = nn.Sequential(nn.Linear(config.hidden_dim, config.hidden_dim // 2), nn.SiLU(), nn.Dropout(config.dropout), nn.Linear(config.hidden_dim // 2, 1))
        self.short_expert = nn.Sequential(nn.Linear(config.hidden_dim, config.hidden_dim // 2), nn.SiLU(), nn.Dropout(config.dropout), nn.Linear(config.hidden_dim // 2, 1))
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)
        nn.init.zeros_(self.long_expert[-1].bias)
        nn.init.zeros_(self.short_expert[-1].bias)
    
    @property
    def input_size(self):
        return self.config.hidden_dim # The modulation connects to the hidden state space

    def forward(self, x: torch.Tensor, external_modulation: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Pre-processing (Attention etc)
        x_p = self.input_proj(x)
        x_p = self.pos_encoding(x_p)
        x_a = self.self_attention(x_p)
        x_a = self.feature_attention(x_a)
        x_cfc = self.attention_proj(x_a)
        x_cfc = self.input_norm(x_cfc)
        
        batch_size, seq_len, _ = x.shape
        current_input = x_cfc
        final_hidden = None
        
        for i, cell in enumerate(self.cfc_layers):
            h_t = torch.zeros(batch_size, self.config.hidden_dim, device=x.device, dtype=x.dtype)
            outputs = []
            
            # Recurrent Loop
            for t in range(seq_len):
                # Pass modulation only if dimensions match or if we want to modulate every step
                # Assumption: external_modulation is [Batch, Hidden]. We broadcast across Time.
                h_t = cell(current_input[:, t, :], h_t, external_modulation=external_modulation)
                outputs.append(h_t.unsqueeze(1))
            
            current_input = torch.cat(outputs, dim=1)
            final_hidden = h_t
            
        long_score = self.long_expert(final_hidden)
        short_score = self.short_expert(final_hidden)
        return (long_score - short_score).squeeze(-1)
