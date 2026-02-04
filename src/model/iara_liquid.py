"""
I.A.R.A. v4.1 - Closed-form Continuous (CfC) Model
==================================================
Features:
- LayerNorm inside recurrent step for signal stability
- 256 hidden units, single layer
- Dropout 0.4 for aggressive regularization

Reference: Hasani et al. (2022) "Closed-form Continuous-time Neural Networks"
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class LiquidConfig:
    """Configuration for Liquid Neural Network v4.1."""
    
    input_dim: int = 11           # 7 indicators + 4 time features
    hidden_dim: int = 256         # Single wide layer
    dropout: float = 0.4          # Aggressive regularization
    output_dim: int = 1           # Binary classification


class CfCCell(nn.Module):
    """
    Closed-form Continuous (CfC) Cell v4.1 with LayerNorm.
    
    The LayerNorm inside the recurrent step prevents:
    - Vanishing gradients over long sequences
    - Hidden state explosion
    - Mode collapse to constant output
    """
    
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Input -> Hidden projection
        self.linear_input = nn.Linear(input_dim, hidden_dim)
        self.linear_hidden = nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        # Time constant (tau) network
        self.tau_input = nn.Linear(input_dim, hidden_dim)
        self.tau_hidden = nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        # LayerNorm for recurrent stability (KEY for v4.1)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        self.activation = nn.Tanh()
        
    def forward(
        self,
        input_: torch.Tensor,
        hidden: torch.Tensor,
        timespan: float = 1.0
    ) -> torch.Tensor:
        """
        Args:
            input_: [batch, input_dim]
            hidden: [batch, hidden_dim]
            timespan: dt (time step, default 1.0)
        Returns:
            new_hidden: [batch, hidden_dim]
        """
        # A) Candidate state
        h_cand = self.activation(
            self.linear_input(input_) + self.linear_hidden(hidden)
        )
        
        # B) Time constant (controls memory vs update rate)
        decay_logits = self.tau_input(input_) + self.tau_hidden(hidden)
        retention = torch.sigmoid(decay_logits)  # [0, 1] - how much to retain
        update_gate = 1.0 - retention            # [0, 1] - how much to update
        
        # C) CfC fusion
        new_hidden = h_cand * update_gate + hidden * retention
        
        # D) LayerNorm for stability (prevents signal decay/explosion)
        new_hidden = self.layer_norm(new_hidden)
        
        return new_hidden


class IARA(nn.Module):
    """
    I.A.R.A. v4.1 - Intelligent Adaptive Regressive Architecture
    
    Single CfC layer with LayerNorm for stable long-sequence processing.
    """
    
    def __init__(self, config: LiquidConfig):
        super().__init__()
        self.config = config
        
        # Input normalization
        self.input_norm = nn.LayerNorm(config.input_dim)
        
        # CfC backbone
        self.cfc = CfCCell(config.input_dim, config.hidden_dim)
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
        
        # Classification head
        self.head = nn.Linear(config.hidden_dim, config.output_dim)
        
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch, seq_len, input_dim]
        Returns:
            logits: [batch] (last step prediction)
            hidden: [batch, hidden_dim] (final hidden state)
        """
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
        out = self.dropout(hidden)
        logits = self.head(out).squeeze(-1)
        
        return logits, hidden
        
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
