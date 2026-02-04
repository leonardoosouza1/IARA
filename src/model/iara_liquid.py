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
    """Configuration for IARA v6.7 Expert Symmetry."""
    
    input_dim: int = 16  # v6.7: Added Global Sentiment
    hidden_dim: int = 128
    num_heads: int = 4
    dropout: float = 0.5
    output_dim: int = 1
    max_seq_len: int = 256
    num_cfc_layers: int = 2 # v6.7: Stacked CfC


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
    Symmetric Closed-form Continuous (CfC) Cell v6.7.
    
    Features:
    - Tanh gating for symmetric dynamics
    - Hidden state clipping [-1, 1]
    - Layer normalization for stability
    - Triton kernel acceleration (when available)
    """
    
    # Class-level flag for kernel availability
    _use_triton = None
    
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
        
        # Check Triton availability once
        if CfCCell._use_triton is None:
            try:
                from src.kernels import fused_cfc_update, TRITON_AVAILABLE
                CfCCell._use_triton = TRITON_AVAILABLE
                if TRITON_AVAILABLE:
                    import logging
                    logging.getLogger(__name__).info("CfCCell: Triton Flash Liquid kernels enabled!")
            except ImportError:
                CfCCell._use_triton = False
        
    def forward(self, input_: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        # Compute linear transforms
        input_linear = self.linear_input(input_) + self.linear_hidden(hidden)
        gate_logits = self.gate_input(input_) + self.gate_hidden(hidden)
        
        if CfCCell._use_triton and input_.is_cuda:
            # Use Triton Fused Kernel (Flash Liquid)
            from src.kernels import fused_cfc_update
            
            # The kernel expects: h_prev, input_linear, gate_linear
            # But our CfC uses tanh gate, not sigmoid. We need to adapt.
            # For now, use PyTorch fallback since the kernel assumes sigmoid gate.
            # TODO: Implement tanh-gate variant of the kernel.
            
            # Fallback to PyTorch for tanh gate
            h_cand = self.activation(input_linear)
            h_cand = self.dropout(h_cand)
            gate = torch.tanh(gate_logits)
            update = gate * (h_cand - hidden)
            h_new = hidden + update
        else:
            # Standard PyTorch Implementation
            h_cand = self.activation(input_linear)
            h_cand = self.dropout(h_cand)
            gate = torch.tanh(gate_logits)
            update = gate * (h_cand - hidden)
            h_new = hidden + update
        
        # Normalize and clip
        h_new = self.layer_norm(h_new)
        h_new = torch.clamp(h_new, min=-1.0, max=1.0)
        
        return h_new


class IARA(nn.Module):
    """
    I.A.R.A. v6.7 - Expert Symmetry (MoE Lite)
    
    Changes:
    - Stacked CfC (2 layers)
    - Gradient Checkpointing
    - Asymmetric Expert Heads (Long vs Short)
    """
    
    def __init__(self, config: LiquidConfig):
        super().__init__()
        self.config = config
        
        # Input projection to attention dimension
        self.input_proj = nn.Linear(config.input_dim, config.hidden_dim)
        
        # Learned positional encoding
        self.pos_encoding = LearnedPositionalEncoding(
            config.hidden_dim, 
            max_len=config.max_seq_len,
            dropout=config.dropout * 0.5
        )
        
        # Self-attention layer
        self.self_attention = TemporalSelfAttention(
            d_model=config.hidden_dim,
            num_heads=config.num_heads,
            dropout=config.dropout
        )
        
        # Feature attention
        self.feature_attention = FeatureAttention(
            config.hidden_dim, 
            dropout=config.dropout * 0.5
        )
        
        # Project back to input_dim (This was weird in v6.0, keeping structure but optimizing)
        # Actually for stacked CfC, better to keep in hidden_dim if possible.
        # But CfCCell takes input_dim. Let's adapt CfC to take hidden_dim or project back.
        # The design says: Attention -> Project -> CfC.
        # If we stack CfC, intermediate projection might be needed or we just chain hidden.
        # Let's keep logic: Attention features become input to CfC system.
        self.attention_proj = nn.Linear(config.hidden_dim, config.input_dim)
        self.input_norm = nn.LayerNorm(config.input_dim)
        
        # Stacked CfC Layers (v6.7)
        self.cfc_layers = nn.ModuleList([
            CfCCell(config.input_dim if i == 0 else config.hidden_dim, 
                   config.hidden_dim, 
                   dropout=config.dropout)
            for i in range(config.num_cfc_layers)
        ])
        
        # Expert Heads (MoE Lite)
        # Head A: Long Expert (Finds reasons to BUY) - Using SiLU per spec
        self.long_expert = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, 1)
        )
        
        # Head B: Short Expert (Finds reasons to SELL/AVOID) - Using SiLU per spec
        self.short_expert = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, 1)
        )
        
        self._init_weights()
        
    def _init_weights(self):
        """Xavier init + negative output bias for neutral initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        # Initialize heads to cancel out initially (0 - 0 = 0)
        nn.init.zeros_(self.long_expert[-1].bias)
        nn.init.zeros_(self.short_expert[-1].bias)
    
    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        # v6.8 "Blindfolded Prophet" - Targeted Feature Dropout
        # The last 4 features are temporal: hour_sin, hour_cos, day_sin, day_cos
        # By zeroing them out 30% of the time, we force the model to learn
        # price dynamics (RSI, MACD, velocity) instead of relying on "clock tricks".
        if self.training and torch.rand(1, device=x.device).item() < 0.3:
            x = x.clone()  # Avoid modifying original tensor
            x[:, :, -4:] = 0.0  # Blindfold the time perception
        
        # Gradient Checkpointing Helper (Enabled for v6.7 VRAM Optimization)
        def run_attention(x_in):
            x_p = self.input_proj(x_in)
            x_p = self.pos_encoding(x_p)
            x_a = self.self_attention(x_p)
            x_a = self.feature_attention(x_a)
            return self.attention_proj(x_a)

        if self.training:
            # Use checkpointing to trade compute for memory
            x_cfc = torch.utils.checkpoint.checkpoint(run_attention, x, use_reentrant=False)
        else:
            x_cfc = run_attention(x)
            
        x_cfc = self.input_norm(x_cfc)
        
        batch_size, seq_len, _ = x.shape
        
        # Stacked CfC Loop
        # We process layer by layer or time by time? 
        # Standard RNN stack: Layer 1 over all time, then Layer 2 over all time.
        # This is more efficient for parallelism if not fully coupled.
        # But CfC depends on prev hidden.
        
        current_input = x_cfc
        final_hidden = None
        
        for layer_idx, cell in enumerate(self.cfc_layers):
            h_t = torch.zeros(batch_size, self.config.hidden_dim, device=x.device, dtype=x.dtype)
            outputs = []
            
            # Recurrent Loop
            for t in range(seq_len):
                h_t = cell(current_input[:, t, :], h_t)
                outputs.append(h_t.unsqueeze(1))
            
            # Output of this layer is input to next
            current_input = torch.cat(outputs, dim=1) # [batch, seq, hidden]
            final_hidden = h_t
            
        # v6.7 Expert Heads
        # We use the final hidden state of the last layer
        
        long_score = self.long_expert(final_hidden)
        short_score = self.short_expert(final_hidden)
        
        # Differential Logit: "Buy Conviction" minus "Sell Conviction"
        logits = long_score - short_score
        
        return logits.squeeze(-1), final_hidden
        
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
