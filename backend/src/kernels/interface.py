"""
Flash Liquid Interface - PyTorch Autograd Wrapper for Triton Kernels
=====================================================================
Provides a drop-in replacement for standard PyTorch CfC operations.
Automatically falls back to PyTorch if Triton is unavailable.
"""

import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)

# Check Triton Availability
TRITON_AVAILABLE = False
try:
    from src.kernels.fused_cfc import fused_cfc_forward, fused_cfc_backward
    TRITON_AVAILABLE = True
    logger.info("Triton kernels loaded successfully.")
except ImportError as e:
    logger.warning(f"Triton not available, falling back to PyTorch: {e}")


class FusedCfCFunction(torch.autograd.Function):
    """
    Autograd Function for Fused CfC Update.
    
    Forward:  h_new = h_prev * (1 - gate) + tanh(input) * gate
    Backward: Computes gradients for h_prev, input, and gate.
    """
    
    @staticmethod
    def forward(ctx, h_prev: torch.Tensor, input_linear: torch.Tensor, gate_linear: torch.Tensor):
        """
        Args:
            h_prev: Previous hidden state [B, H]
            input_linear: Linear transform of input (x @ W + b) [B, H]
            gate_linear: Gate logits (x @ Wg + h @ Ug + bg) [B, H]
        
        Returns:
            h_new: Updated hidden state [B, H]
        """
        # Ensure contiguous memory layout
        h_prev = h_prev.contiguous()
        input_linear = input_linear.contiguous()
        gate_linear = gate_linear.contiguous()
        
        if TRITON_AVAILABLE and h_prev.is_cuda:
            # Use Triton Kernel
            h_new = fused_cfc_forward(h_prev, input_linear, gate_linear)
        else:
            # PyTorch Fallback
            signal = torch.tanh(input_linear)
            gate = torch.sigmoid(gate_linear)
            h_new = h_prev * (1.0 - gate) + signal * gate
        
        # Save for backward
        ctx.save_for_backward(h_prev, input_linear, gate_linear)
        
        return h_new
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """
        Compute gradients.
        """
        h_prev, input_linear, gate_linear = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        
        if TRITON_AVAILABLE and h_prev.is_cuda:
            # Use Triton Kernel
            grad_h_prev, grad_input, grad_gate = fused_cfc_backward(
                grad_output, h_prev, input_linear, gate_linear
            )
        else:
            # PyTorch Fallback
            signal = torch.tanh(input_linear)
            gate = torch.sigmoid(gate_linear)
            
            # d(h_new)/d(h_prev) = (1 - gate)
            grad_h_prev = grad_output * (1.0 - gate)
            
            # d(h_new)/d(signal) = gate
            # d(signal)/d(input) = 1 - tanh^2
            grad_signal = grad_output * gate
            grad_input = grad_signal * (1.0 - signal * signal)
            
            # d(h_new)/d(gate) = signal - h_prev
            # d(gate)/d(gate_linear) = gate * (1 - gate)
            grad_gate_val = grad_output * (signal - h_prev)
            grad_gate = grad_gate_val * gate * (1.0 - gate)
        
        return grad_h_prev, grad_input, grad_gate


# Convenience wrapper
def fused_cfc_update(h_prev: torch.Tensor, input_linear: torch.Tensor, gate_linear: torch.Tensor) -> torch.Tensor:
    """
    Drop-in replacement for CfC update step.
    
    Uses Triton kernel if available, else falls back to PyTorch.
    """
    return FusedCfCFunction.apply(h_prev, input_linear, gate_linear)


class FusedCfCCell(nn.Module):
    """
    Fused CfC Cell using Triton Kernels.
    
    Drop-in replacement for standard CfCCell with ~2x speedup.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Linear transforms (computed in PyTorch, fused only the update)
        self.linear_input = nn.Linear(input_dim, hidden_dim)
        self.linear_hidden = nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        self.gate_input = nn.Linear(input_dim, hidden_dim)
        self.gate_hidden = nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, input_: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        """
        Forward pass using fused kernel for update step.
        
        Args:
            input_: Input tensor [B, D]
            hidden: Previous hidden state [B, H]
        
        Returns:
            h_new: Updated hidden state [B, H]
        """
        # Compute linear transforms (standard PyTorch)
        input_linear = self.linear_input(input_) + self.linear_hidden(hidden)
        gate_linear = self.gate_input(input_) + self.gate_hidden(hidden)
        
        # Fused CfC Update (Triton or PyTorch fallback)
        h_new = fused_cfc_update(hidden, input_linear, gate_linear)
        
        # Post-processing (standard PyTorch)
        h_new = self.layer_norm(h_new)
        h_new = self.dropout(h_new)
        h_new = torch.clamp(h_new, min=-1.0, max=1.0)
        
        return h_new
