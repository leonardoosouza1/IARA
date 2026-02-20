"""
Saci-Nano - Causal Decoder-Only Transformer for Financial Token Prediction
===========================================================================
A GPT-style autoregressive model that treats market indicators as tokens.
Implements Next Token Prediction for market state forecasting.
"""

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from src.model.bitnet import (
    BitLinear,
    RMSNorm,
    SwiGLU,
    precompute_freqs_cis,
    apply_rotary_emb
)


@dataclass
class SaciNanoConfig:
    """Configuration for Saci-Nano model."""
    
    vocab_size: int = 16384  # Token vocabulary size
    dim: int = 512           # Model dimension
    n_layers: int = 8        # Number of transformer layers
    n_heads: int = 8         # Number of attention heads
    n_kv_heads: int = 4      # Number of key-value heads (GQA)
    max_seq_len: int = 512   # Maximum sequence length
    dropout: float = 0.1     # Dropout rate
    rope_theta: float = 10000.0  # RoPE base frequency


class CausalSelfAttention(nn.Module):
    """
    Multi-Head Causal Self-Attention with RoPE.
    Supports Grouped Query Attention (GQA) for efficiency.
    """
    
    def __init__(self, config: SaciNanoConfig):
        super().__init__()
        self.config = config
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.dim // config.n_heads
        self.n_rep = self.n_heads // self.n_kv_heads  # Repetitions for GQA
        
        # Projections
        self.wq = nn.Linear(config.dim, config.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(config.dim, config.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(config.dim, config.n_kv_heads * self.head_dim, bias=False)
        self.wo = BitLinear(config.n_heads * self.head_dim, config.dim, bias=False)
        
        self.dropout = nn.Dropout(config.dropout)
        
        # Causal mask (will be registered as buffer)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(config.max_seq_len, config.max_seq_len), diagonal=1).bool()
        )
    
    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass with causal masking.
        
        Args:
            x: Input [batch, seq, dim]
            freqs_cis: RoPE frequencies [seq, head_dim//2]
            
        Returns:
            Output [batch, seq, dim]
        """
        batch_size, seq_len, _ = x.shape
        
        # Project Q, K, V
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)
        
        # Reshape to heads
        q = rearrange(q, 'b s (h d) -> b s h d', h=self.n_heads)
        k = rearrange(k, 'b s (h d) -> b s h d', h=self.n_kv_heads)
        v = rearrange(v, 'b s (h d) -> b s h d', h=self.n_kv_heads)
        
        # Apply RoPE
        q, k = apply_rotary_emb(q, k, freqs_cis[:seq_len])
        
        # Repeat KV for GQA
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=2)
            v = v.repeat_interleave(self.n_rep, dim=2)
        
        # Rearrange for attention
        q = rearrange(q, 'b s h d -> b h s d')
        k = rearrange(k, 'b s h d -> b h s d')
        v = rearrange(v, 'b s h d -> b h s d')
        
        # Scaled dot-product attention
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        
        # Apply causal mask
        attn = attn.masked_fill(self.mask[:seq_len, :seq_len], float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply to values
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h s d -> b s (h d)')
        
        return self.wo(out)


class TransformerBlock(nn.Module):
    """
    Single transformer decoder block.
    Pre-norm architecture with RMSNorm.
    """
    
    def __init__(self, config: SaciNanoConfig):
        super().__init__()
        
        self.attention = CausalSelfAttention(config)
        self.ffn = SwiGLU(config.dim, dropout=config.dropout)
        
        self.norm1 = RMSNorm(config.dim)
        self.norm2 = RMSNorm(config.dim)
    
    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward with residual connections.
        
        Args:
            x: Input [batch, seq, dim]
            freqs_cis: RoPE frequencies
            
        Returns:
            Output [batch, seq, dim]
        """
        # Self-attention with residual
        x = x + self.attention(self.norm1(x), freqs_cis)
        
        # FFN with residual
        x = x + self.ffn(self.norm2(x))
        
        return x


class SaciNano(nn.Module):
    """
    Saci-Nano: Causal Decoder-Only Transformer for Financial Prediction.
    
    Treats market indicators as tokens and predicts the next token
    in an autoregressive manner (Next Token Prediction).
    """
    
    def __init__(self, config: SaciNanoConfig):
        super().__init__()
        self.config = config
        
        # Token embedding
        self.token_embedding = nn.Embedding(config.vocab_size, config.dim)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])
        
        # Final layer norm
        self.norm = RMSNorm(config.dim)
        
        # Output projection (tied with embedding)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)
        
        # Precompute RoPE frequencies
        head_dim = config.dim // config.n_heads
        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(head_dim, config.max_seq_len, config.rope_theta)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module: nn.Module):
        """Initialize weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(
        self,
        tokens: torch.Tensor,
        targets: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            tokens: Input token IDs [batch, seq]
            targets: Optional target token IDs for loss computation
            
        Returns:
            Tuple of (logits, loss)
        """
        batch_size, seq_len = tokens.shape
        
        # Token embeddings
        x = self.token_embedding(tokens)
        
        # Pass through transformer layers
        for layer in self.layers:
            x = layer(x, self.freqs_cis)
        
        # Final norm
        x = self.norm(x)
        
        # Output logits
        logits = self.output(x)
        
        # Compute loss if targets provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                targets.view(-1),
                ignore_index=-100  # Ignore padding
            )
        
        return logits, loss
    
    @torch.no_grad()
    def generate(
        self,
        tokens: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None
    ) -> torch.Tensor:
        """
        Generate new tokens autoregressively.
        
        Args:
            tokens: Initial tokens [batch, seq]
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling (None for greedy)
            
        Returns:
            Generated tokens [batch, seq + max_new_tokens]
        """
        for _ in range(max_new_tokens):
            # Truncate to max_seq_len
            tokens_cond = tokens[:, -self.config.max_seq_len:]
            
            # Forward pass
            logits, _ = self(tokens_cond)
            logits = logits[:, -1, :] / temperature
            
            # Top-k sampling
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            
            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append
            tokens = torch.cat([tokens, next_token], dim=1)
        
        return tokens
    
    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def estimate_vram_mb(self, batch_size: int = 32) -> float:
        """Estimate VRAM usage in MB."""
        params = self.count_parameters() * 4 * 2  # weights + gradients
        seq = self.config.max_seq_len
        hidden = self.config.dim
        layers = self.config.n_layers
        activations = batch_size * seq * hidden * 4 * layers * 2
        return (params + activations) / (1024 * 1024)


def main():
    """Test model forward pass."""
    config = SaciNanoConfig(
        vocab_size=8192,
        dim=256,
        n_layers=4,
        n_heads=4,
        n_kv_heads=2,
        max_seq_len=512
    )
    
    model = SaciNano(config)
    print(f"Parameters: {model.count_parameters():,}")
    print(f"Estimated VRAM: {model.estimate_vram_mb():.2f} MB")
    
    # Test forward
    tokens = torch.randint(0, config.vocab_size, (2, 128))
    targets = torch.randint(0, config.vocab_size, (2, 128))
    
    logits, loss = model(tokens, targets)
    print(f"Logits shape: {logits.shape}")
    print(f"Loss: {loss.item():.4f}")
    
    # Test generate
    generated = model.generate(tokens[:, :10], max_new_tokens=20, top_k=50)
    print(f"Generated shape: {generated.shape}")
    
    print("Forward pass successful!")


if __name__ == "__main__":
    main()
