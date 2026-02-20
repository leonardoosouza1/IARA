"""
BitNet Core - 1.58-bit Quantized Linear Layers
===============================================
Implements BitLinear layer with ternary weight quantization {-1, 0, +1}
and 8-bit activation quantization for extreme memory efficiency.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Quantization Functions (Straight-Through Estimator)
# ============================================================================

def weight_quant(w: torch.Tensor) -> torch.Tensor:
    """
    Ternary weight quantization for BitNet b1.58.
    Quantizes weights to {-1, 0, +1} using absmean scaling.
    
    Args:
        w: Weight tensor
        
    Returns:
        Quantized weights with STE gradient
    """
    # Absmean scaling factor
    scale = w.abs().mean().clamp(min=1e-5)
    
    # Quantize: round(w / scale) clamped to {-1, 0, 1}
    w_scaled = w / scale
    w_quant = torch.clamp(torch.round(w_scaled), -1, 1)
    
    # Straight-Through Estimator
    return (w_quant * scale).detach() + w - w.detach()


def activation_quant(x: torch.Tensor, bits: int = 8) -> torch.Tensor:
    """
    Absmax quantization for activations.
    
    Args:
        x: Activation tensor
        bits: Quantization bits (default 8)
        
    Returns:
        Quantized activations with STE gradient
    """
    Qn = -(2 ** (bits - 1))
    Qp = 2 ** (bits - 1) - 1
    
    # Per-tensor absmax scale
    scale = x.abs().max().clamp(min=1e-5)
    
    # Quantize
    x_scaled = x / scale * Qp
    x_quant = torch.clamp(torch.round(x_scaled), Qn, Qp)
    
    # Dequantize with STE
    x_dequant = x_quant / Qp * scale
    return x_dequant.detach() + x - x.detach()


# ============================================================================
# BitLinear Layer
# ============================================================================

class BitLinear(nn.Module):
    """
    Linear layer with 1.58-bit quantized weights.
    
    Uses ternary quantization {-1, 0, +1} for weights
    and 8-bit quantization for activations.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        activation_bits: int = 8
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation_bits = activation_bits
        
        # Weight parameter (full precision during training)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        
        # Optional bias
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        
        # RMSNorm for input (BitNet uses pre-norm)
        self.norm = RMSNorm(in_features)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with quantization.
        
        Args:
            x: Input [..., in_features]
            
        Returns:
            Output [..., out_features]
        """
        # Normalize input
        x = self.norm(x)
        
        # Quantize activations
        x_quant = activation_quant(x, self.activation_bits)
        
        # Quantize weights
        w_quant = weight_quant(self.weight)
        
        return F.linear(x_quant, w_quant, self.bias)


# ============================================================================
# RMSNorm (Root Mean Square Layer Normalization)
# ============================================================================

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    More efficient than LayerNorm, used in Llama.
    """
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # RMS normalization
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x / rms * self.weight


# ============================================================================
# SwiGLU Feed-Forward Network (with BitLinear)
# ============================================================================

class SwiGLU(nn.Module):
    """
    SwiGLU activation for Feed-Forward network.
    Uses BitLinear layers for efficiency.
    
    SwiGLU(x) = (xW1) * SiLU(xV) @ W2
    """
    
    def __init__(
        self,
        dim: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.0
    ):
        super().__init__()
        hidden_dim = hidden_dim or int(dim * 8 / 3)
        # Round to nearest multiple of 256 for efficiency
        hidden_dim = ((hidden_dim + 255) // 256) * 256
        
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)  # Gate projection
        self.w2 = BitLinear(hidden_dim, dim, bias=False)  # Down projection  
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)  # Up projection
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: gate * up projection
        gate = F.silu(self.w1(x))
        up = self.w3(x)
        x = gate * up
        x = self.dropout(x)
        return self.w2(x)


# ============================================================================
# Rotary Positional Embeddings (RoPE)
# ============================================================================

def precompute_freqs_cis(dim: int, max_seq_len: int, theta: float = 10000.0) -> torch.Tensor:
    """
    Precompute rotary embedding frequencies.
    
    Args:
        dim: Head dimension (must be even)
        max_seq_len: Maximum sequence length
        theta: Base frequency
        
    Returns:
        Complex frequencies tensor [max_seq_len, dim//2]
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_seq_len)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)  # complex64


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to query and key tensors.
    
    Args:
        xq: Query tensor [batch, seq, n_heads, head_dim]
        xk: Key tensor [batch, seq, n_kv_heads, head_dim]
        freqs_cis: Precomputed frequencies [seq, head_dim//2]
        
    Returns:
        Rotated (xq, xk)
    """
    # Reshape to complex
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    
    # Apply rotation
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(2)  # [1, seq, 1, dim//2]
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(-2)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(-2)
    
    return xq_out.type_as(xq), xk_out.type_as(xk)


# ============================================================================
# Test
# ============================================================================

def main():
    """Test BitLinear layer."""
    batch_size = 4
    seq_len = 128
    dim = 256
    
    # Test BitLinear
    layer = BitLinear(dim, dim * 4)
    x = torch.randn(batch_size, seq_len, dim)
    y = layer(x)
    print(f"BitLinear: {x.shape} -> {y.shape}")
    
    # Test SwiGLU
    ffn = SwiGLU(dim)
    y = ffn(x)
    print(f"SwiGLU: {x.shape} -> {y.shape}")
    
    # Test RoPE
    freqs = precompute_freqs_cis(64, 512)
    print(f"RoPE freqs: {freqs.shape}")
    
    print("All tests passed!")


if __name__ == "__main__":
    main()
