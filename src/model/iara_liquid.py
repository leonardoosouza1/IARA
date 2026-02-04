"""
I.A.R.A. v6.0 - Fractal Attention-CfC Model
============================================
Deep temporal context for long-term dependencies.

Key Features:
- Sequence Length: 256 steps (~21 hours of 5m data)
- Learned Positional Encoding (vs sinusoidal)
- Multi-Head Self-Attention (4 heads)
- Feature Attention Gate
- Symmetric CfC with tanh gating
- Higher regularization (dropout 0.5)

Reference: Hasani et al. (2022) "Closed-form Continuous-time Neural Networks"
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LiquidConfig:
    """Configuration for IARA v6.0."""
    
    input_dim: int = 15  # Extended features
    hidden_dim: int = 512
    num_heads: int = 4
    dropout: float = 0.5  # Increased for anti-overconfidence
    output_dim: int = 1
    max_seq_len: int = 256


class LearnedPositionalEncoding(nn.Module):
    """
    Learned Positional Encoding for long sequences.
    Better than sinusoidal for distinguishing "10 min ago" vs "4 hours ago".
    """
    
    def __init__(self, d_model: int, max_len: int = 256, dropout: float = 0.1):
        super().__init__()
        
        self.position_embeddings = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Initialize with sinusoidal pattern for better starting point
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
        """Add learned positional encoding. x: [batch, seq_len, d_model]"""
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        positional_encoding = self.position_embeddings(positions)
        return self.dropout(x + positional_encoding)


class FeatureAttention(nn.Module):
    """Feature-level attention gate with gating mechanism."""
    
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
        attention = self.gate(x)
        return x * attention


class TemporalSelfAttention(nn.Module):
    """
    Multi-head self-attention (no feed-forward for speed).
    v6.1 Lite: Removed FF expansion for 3x speedup.
    """
    
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
        """
        Self-attention with residual.
        x: [batch, seq_len, d_model]
        """
        # Self-attention with residual
        attn_out, _ = self.attention(x, x, x, need_weights=False)
        x = self.norm(x + self.dropout(attn_out))
        return x


class CfCCell(nn.Module):
    """
    Symmetric Closed-form Continuous (CfC) Cell v6.0.
    
    Features:
    - Tanh gating for symmetric dynamics
    - Hidden state clipping [-1, 1]
    - Layer normalization for stability
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Candidate generation
        self.linear_input = nn.Linear(input_dim, hidden_dim)
        self.linear_hidden = nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        # Symmetric gate
        self.gate_input = nn.Linear(input_dim, hidden_dim)
        self.gate_hidden = nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        # Regularization
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
        self.activation = nn.Tanh()
        
    def forward(self, input_: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        # Candidate state
        h_cand = self.activation(
            self.linear_input(input_) + self.linear_hidden(hidden)
        )
        h_cand = self.dropout(h_cand)
        
        # Symmetric gate [-1, 1]
        gate_logits = self.gate_input(input_) + self.gate_hidden(hidden)
        gate = torch.tanh(gate_logits)
        
        # Symmetric update
        update = gate * (h_cand - hidden)
        h_new = hidden + update
        
        # Normalize and clip
        h_new = self.layer_norm(h_new)
        h_new = torch.clamp(h_new, min=-1.0, max=1.0)
        
        return h_new


class IARA(nn.Module):
    """
    I.A.R.A. v6.0 - Fractal Attention-CfC Architecture
    
    Flow:
    1. Input Projection (input_dim -> hidden_dim)
    2. Learned Positional Encoding (for 256-step sequences)
    3. Self-Attention with Feed-Forward
    4. Feature Attention Gate
    5. Project back to input_dim
    6. CfC Recurrent Loop
    7. Output Head with dropout
    """
    
    def __init__(self, config: LiquidConfig):
        super().__init__()
        self.config = config
        
        # Input projection to attention dimension
        self.input_proj = nn.Linear(config.input_dim, config.hidden_dim)
        
        # Learned positional encoding for long sequences
        self.pos_encoding = LearnedPositionalEncoding(
            config.hidden_dim, 
            max_len=config.max_seq_len,
            dropout=config.dropout * 0.5
        )
        
        # Self-attention layer with feed-forward
        self.self_attention = TemporalSelfAttention(
            d_model=config.hidden_dim,
            num_heads=config.num_heads,
            dropout=config.dropout
        )
        
        # Feature attention after self-attention
        self.feature_attention = FeatureAttention(
            config.hidden_dim, 
            dropout=config.dropout * 0.5
        )
        
        # Project back to original dimension for CfC
        self.attention_proj = nn.Linear(config.hidden_dim, config.input_dim)
        
        # Input normalization before CfC
        self.input_norm = nn.LayerNorm(config.input_dim)
        
        # CfC backbone with regularization
        self.cfc = CfCCell(config.input_dim, config.hidden_dim, dropout=config.dropout)
        
        # Output normalization
        self.output_norm = nn.LayerNorm(config.hidden_dim)
        
        # Dropout before head
        self.dropout = nn.Dropout(config.dropout)
        
        # Classification head
        self.head = nn.Linear(config.hidden_dim, config.output_dim)
        
        self._init_weights()
        
    def _init_weights(self):
        """Xavier init + negative output bias for neutral initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        # Negative bias to start neutral (sigmoid(-0.5) ≈ 0.38)
        nn.init.constant_(self.head.bias, -0.5)
    
    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch, seq_len, input_dim]
        Returns:
            logits: [batch]
            hidden: [batch, hidden_dim]
        """
        batch_size, seq_len, _ = x.shape
        
        # 1. Input projection
        x_proj = self.input_proj(x)  # [batch, seq, hidden_dim]
        
        # 2. Add learned positional encoding
        x_proj = self.pos_encoding(x_proj)
        
        # 3. Self-attention with feed-forward
        x_attn = self.self_attention(x_proj)
        
        # 4. Feature attention gate
        x_attn = self.feature_attention(x_attn)
        
        # 5. Project back for CfC
        x_cfc = self.attention_proj(x_attn)
        x_cfc = self.input_norm(x_cfc)
        
        # 6. Initialize hidden state
        if hidden is None:
            hidden = torch.zeros(
                batch_size, self.config.hidden_dim,
                device=x.device, dtype=x.dtype
            )
        
        # 7. CfC recurrent loop
        for t in range(seq_len):
            hidden = self.cfc(x_cfc[:, t, :], hidden)
        
        # 8. Output with regularization
        hidden = self.output_norm(hidden)
        out = self.dropout(hidden)
        logits = self.head(out).squeeze(-1)
        
        return logits, hidden
        
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
