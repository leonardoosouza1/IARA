"""
BitNet Transformer - 1.58-bit Quantized Transformer for Forex
==============================================================
Implements BitNet b1.58 architecture with ternary weight quantization.
Optimized for AMD ROCm (RX 6750 XT) with reduced VRAM usage.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ============================================================================
# Quantization Functions
# ============================================================================

def weight_quant(w: torch.Tensor) -> torch.Tensor:
    """
    Ternary weight quantization for BitNet b1.58.
    Quantizes weights to {-1, 0, +1} using sign and mean absolute value.
    
    Args:
        w: Weight tensor
        
    Returns:
        Quantized weights in {-1, 0, +1}
    """
    # Compute scaling factor (mean of absolute values)
    scale = w.abs().mean()
    
    # Quantize using round(w / scale) clamped to {-1, 0, 1}
    # For b1.58: use sign function with threshold
    w_quant = torch.sign(w) * (w.abs() > scale * 0.5).float()
    
    # Straight-Through Estimator (STE) for gradients
    return w_quant.detach() + w - w.detach()


def activation_quant(x: torch.Tensor, bits: int = 8) -> torch.Tensor:
    """
    Absmax quantization for activations.
    
    Args:
        x: Activation tensor
        bits: Number of bits for quantization
        
    Returns:
        Quantized activations (scaled to original range)
    """
    # Compute absmax scale
    Qn = -(2 ** (bits - 1))
    Qp = 2 ** (bits - 1) - 1
    
    # Per-tensor absmax
    scale = x.abs().max().clamp(min=1e-5)
    
    # Quantize
    x_scaled = x / scale
    x_quant = torch.clamp(torch.round(x_scaled * Qp), Qn, Qp)
    
    # Dequantize (STE)
    x_dequant = x_quant / Qp * scale
    
    return x_dequant.detach() + x - x.detach()


# ============================================================================
# BitLinear Layer
# ============================================================================

class BitLinear(nn.Module):
    """
    Linear layer with 1.58-bit quantized weights.
    Uses ternary quantization {-1, 0, +1} for weights and 8-bit for activations.
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
        
        # Initialize weights
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        
        # Optional bias (usually disabled for BitNet)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        
        # LayerNorm before quantization (as per BitNet paper)
        self.norm = nn.LayerNorm(in_features)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with quantized weights and activations.
        
        Args:
            x: Input tensor [..., in_features]
            
        Returns:
            Output tensor [..., out_features]
        """
        # Normalize input
        x = self.norm(x)
        
        # Quantize activations
        x_quant = activation_quant(x, self.activation_bits)
        
        # Quantize weights
        w_quant = weight_quant(self.weight)
        
        # Linear transformation
        output = F.linear(x_quant, w_quant, self.bias)
        
        return output


# ============================================================================
# BitNet Feed-Forward Network
# ============================================================================

class BitFeedForward(nn.Module):
    """
    Feed-Forward Network using BitLinear layers.
    Architecture: Linear -> GELU -> BitLinear -> GELU -> BitLinear
    """
    
    def __init__(
        self,
        d_model: int,
        d_ff: Optional[int] = None,
        dropout: float = 0.1
    ):
        super().__init__()
        d_ff = d_ff or d_model * 4
        
        # First projection (regular linear for input)
        self.w1 = nn.Linear(d_model, d_ff)
        
        # Second projection (BitLinear)
        self.w2 = BitLinear(d_ff, d_ff)
        
        # Output projection (BitLinear)
        self.w3 = BitLinear(d_ff, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through FFN.
        
        Args:
            x: Input tensor [batch, seq, d_model]
            
        Returns:
            Output tensor [batch, seq, d_model]
        """
        x = self.w1(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        x = self.w2(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        x = self.w3(x)
        return x


# ============================================================================
# Multi-Head Attention (Standard, no quantization on attention)
# ============================================================================

class MultiHeadAttention(nn.Module):
    """
    Standard Multi-Head Self-Attention.
    Attention weights are kept in full precision for stability.
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1
    ):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # Projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for self-attention.
        
        Args:
            x: Input tensor [batch, seq, d_model]
            mask: Optional attention mask
            
        Returns:
            Output tensor [batch, seq, d_model]
        """
        batch_size, seq_len, _ = x.shape
        
        # Project Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape to multi-head format
        q = rearrange(q, 'b s (h d) -> b h s d', h=self.n_heads)
        k = rearrange(k, 'b s (h d) -> b h s d', h=self.n_heads)
        v = rearrange(v, 'b s (h d) -> b h s d', h=self.n_heads)
        
        # Compute attention scores
        attn = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = torch.matmul(attn, v)
        
        # Reshape back
        out = rearrange(out, 'b h s d -> b s (h d)')
        
        return self.out_proj(out)


# ============================================================================
# Transformer Encoder Layer
# ============================================================================

class BitNetEncoderLayer(nn.Module):
    """
    Transformer Encoder Layer with BitNet FFN.
    Uses standard attention + BitNet feed-forward.
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: Optional[int] = None,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = BitFeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with pre-norm residual connections.
        
        Args:
            x: Input tensor [batch, seq, d_model]
            mask: Optional attention mask
            
        Returns:
            Output tensor [batch, seq, d_model]
        """
        # Self-attention with residual
        residual = x
        x = self.norm1(x)
        x = self.attention(x, mask)
        x = self.dropout1(x)
        x = residual + x
        
        # FFN with residual
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = self.dropout2(x)
        x = residual + x
        
        return x


# ============================================================================
# Positional Encoding
# ============================================================================

class SinusoidalPositionalEncoding(nn.Module):
    """
    Fixed sinusoidal positional encoding.
    """
    
    def __init__(self, d_model: int, max_len: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.
        
        Args:
            x: Input tensor [batch, seq, d_model]
            
        Returns:
            Output with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ============================================================================
# BitNet Encoder (Full Model)
# ============================================================================

class BitNetEncoder(nn.Module):
    """
    Full BitNet Transformer Encoder for forex prediction.
    
    Architecture:
    - Input projection (features -> d_model)
    - Positional encoding
    - N x BitNetEncoderLayer
    - Output projection (d_model -> 1)
    """
    
    def __init__(
        self,
        input_dim: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: Optional[int] = None,
        max_seq_len: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoding = SinusoidalPositionalEncoding(d_model, max_seq_len, dropout)
        
        # Encoder layers
        self.layers = nn.ModuleList([
            BitNetEncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # Final layer norm
        self.norm = nn.LayerNorm(d_model)
        
        # Output head (predicts next candle direction)
        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)  # Binary output (logit)
        )
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor [batch, seq, input_dim]
            mask: Optional attention mask
            
        Returns:
            Logits tensor [batch, 1]
        """
        # Project input to model dimension
        x = self.input_proj(x)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Pass through encoder layers
        for layer in self.layers:
            x = layer(x, mask)
        
        # Final normalization
        x = self.norm(x)
        
        # Use last token for prediction (like CLS token)
        x = x[:, -1, :]  # [batch, d_model]
        
        # Output head
        logits = self.output_head(x)  # [batch, 1]
        
        return logits
    
    def count_parameters(self) -> int:
        """Count total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def estimate_vram_mb(self, batch_size: int = 32, seq_len: int = 512) -> float:
        """
        Estimate VRAM usage in MB.
        
        Args:
            batch_size: Batch size
            seq_len: Sequence length
            
        Returns:
            Estimated VRAM in MB
        """
        # Parameters (assume float32 for gradients)
        param_bytes = self.count_parameters() * 4 * 2  # weights + gradients
        
        # Activations (rough estimate)
        activation_bytes = batch_size * seq_len * self.d_model * 4 * len(self.layers) * 2
        
        total_bytes = param_bytes + activation_bytes
        return total_bytes / (1024 * 1024)


# ============================================================================
# Main (Testing)
# ============================================================================

def main():
    """Test forward pass."""
    # Configuration
    batch_size = 4
    seq_len = 512
    input_dim = 128  # Number of features from feature engine
    
    # Create model
    model = BitNetEncoder(
        input_dim=input_dim,
        d_model=256,
        n_heads=8,
        n_layers=6,
        max_seq_len=seq_len,
        dropout=0.1
    )
    
    print(f"Model parameters: {model.count_parameters():,}")
    print(f"Estimated VRAM: {model.estimate_vram_mb(batch_size, seq_len):.2f} MB")
    
    # Test forward pass
    x = torch.randn(batch_size, seq_len, input_dim)
    logits = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {logits.shape}")
    print(f"Output: {logits.squeeze()}")
    
    # Test backward pass
    loss = logits.sum()
    loss.backward()
    print("Backward pass successful!")


if __name__ == "__main__":
    main()
