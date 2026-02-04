"""
I.A.R.A. v4.4 - Feature Attention + Symmetric CfC
=================================================
New features:
- Feature Attention Gate: dynamically mutes noisy features
- Hidden size 512 for increased capacity
- Symmetric Tanh gating from v4.3
- Hidden clipping and negative output bias

Reference: Hasani et al. (2022) "Closed-form Continuous-time Neural Networks"
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class LiquidConfig:
    """Configuration for Liquid Neural Network v4.4."""
    
    input_dim: int = 11
    hidden_dim: int = 512  # Increased capacity
    dropout: float = 0.4
    output_dim: int = 1


class FeatureAttention(nn.Module):
    """
    Feature Attention Gate - learns which features matter per timestep.
    
    Produces attention weights [0, 1] for each input feature,
    allowing the model to mute noisy or irrelevant features
    based on current market context.
    """
    
    def __init__(self, input_dim: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, input_dim]
        Returns:
            x_attended: [batch, seq_len, input_dim]
        """
        attention = self.gate(x)  # [batch, seq_len, input_dim]
        return x * attention


class CfCCell(nn.Module):
    """
    Symmetric Closed-form Continuous (CfC) Cell v4.4.
    
    Features:
    - Tanh gating for symmetric UP/DOWN dynamics
    - No internal normalization
    - Hidden state clipping
    """
    
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Candidate generation
        self.linear_input = nn.Linear(input_dim, hidden_dim)
        self.linear_hidden = nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        # Symmetric gate
        self.gate_input = nn.Linear(input_dim, hidden_dim)
        self.gate_hidden = nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        self.activation = nn.Tanh()
        
    def forward(
        self,
        input_: torch.Tensor,
        hidden: torch.Tensor
    ) -> torch.Tensor:
        """Symmetric CfC dynamics with Tanh gating."""
        # Candidate state
        h_cand = self.activation(
            self.linear_input(input_) + self.linear_hidden(hidden)
        )
        
        # Symmetric gate [-1, 1]
        gate_logits = self.gate_input(input_) + self.gate_hidden(hidden)
        gate = torch.tanh(gate_logits)
        
        # Symmetric update
        update = gate * (h_cand - hidden)
        h_new = hidden + update
        
        # Clip to prevent explosion
        h_new = torch.clamp(h_new, min=-1.0, max=1.0)
        
        return h_new


class IARA(nn.Module):
    """
    I.A.R.A. v4.4 - Feature Attention + Symmetric CfC
    
    Architecture:
    1. Feature Attention Gate (learns feature importance)
    2. Input Projection + LayerNorm
    3. CfC recurrent loop (free dynamics)
    4. Output LayerNorm + Dropout + Head
    """
    
    def __init__(self, config: LiquidConfig):
        super().__init__()
        self.config = config
        
        # Feature Attention Gate
        self.attention = FeatureAttention(config.input_dim)
        
        # Input projection + normalization
        self.input_proj = nn.Linear(config.input_dim, config.input_dim)
        self.input_norm = nn.LayerNorm(config.input_dim)
        
        # CfC backbone
        self.cfc = CfCCell(config.input_dim, config.hidden_dim)
        
        # Output normalization
        self.output_norm = nn.LayerNorm(config.hidden_dim)
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
        
        # Classification head
        self.head = nn.Linear(config.hidden_dim, config.output_dim)
        
        self._init_weights()
        
    def _init_weights(self):
        """Xavier init + negative output bias."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        # Negative bias to start with DOWN preference
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
        # Feature Attention (learn which features matter)
        x = self.attention(x)
        
        # Input projection + normalization
        x = self.input_proj(x)
        x = self.input_norm(x)
        
        batch_size, seq_len, _ = x.shape
        
        if hidden is None:
            hidden = torch.zeros(
                batch_size, self.config.hidden_dim,
                device=x.device, dtype=x.dtype
            )
        
        # Process sequence
        for t in range(seq_len):
            hidden = self.cfc(x[:, t, :], hidden)
        
        # Output
        hidden = self.output_norm(hidden)
        out = self.dropout(hidden)
        logits = self.head(out).squeeze(-1)
        
        return logits, hidden
        
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
