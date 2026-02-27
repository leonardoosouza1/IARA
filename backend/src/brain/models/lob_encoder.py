"""
I.A.R.A. v11 — LOB Encoder (Sensory Bridge)
============================================
Maps raw LOB microstructure features into a learned embedding
that is fused with the SwarmCoordinator's consensus context.

Architecture:
  6 raw LOB features → Linear(64) → SiLU → Dropout → Linear(hidden) → LayerNorm
"""

import torch
import torch.nn as nn


class LOBEncoder(nn.Module):
    """
    Sensory Bridge for Level 2 Order Book data.

    Encodes 6 LOB microstructure features (OBI, VPIN, Spread, Depth, Walls)
    into a dense embedding compatible with the coordinator's CfC cell.
    """

    def __init__(self, input_dim: int = 6, hidden_dim: int = 64, output_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: LOB features [batch, 6]

        Returns:
            LOB embedding [batch, output_dim]
        """
        return self.encoder(x)
