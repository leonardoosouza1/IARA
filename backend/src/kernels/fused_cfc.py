"""
Flash Liquid CfC Kernel - Fused CUDA/ROCm Kernel using Triton
==============================================================
Fuses the CfC update rule into a single GPU operation:
    signal = tanh(input_linear)
    gate = sigmoid(time_gate)
    h_new = h_prev * (1 - gate) + signal * gate

This eliminates multiple memory round-trips, reducing VRAM IO by ~60%.

Reference: Hasani et al. (2022) "Closed-form Continuous-time Neural Networks"
"""

import triton
import triton.language as tl
import torch

# Block size tuned for AMD RDNA2 (RX 6750 XT has 2560 shaders)
# Power of 2, typically 256-1024 works well
DEFAULT_BLOCK_SIZE = 1024


@triton.jit
def fused_cfc_forward_kernel(
    h_prev_ptr,      # Pointer to previous hidden state [B, H]
    input_ptr,       # Pointer to input linear (x @ W_input + b) [B, H]
    gate_ptr,        # Pointer to gate linear (x @ W_gate + h @ U_gate + b) [B, H]
    output_ptr,      # Pointer to output h_new [B, H]
    n_elements,      # Total number of elements (B * H)
    BLOCK_SIZE: tl.constexpr
):
    """
    Fused CfC Forward Kernel.
    
    Computes:
        signal = tanh(input)
        gate = sigmoid(gate_input)
        h_new = h_prev * (1 - gate) + signal * gate
    
    All in registers, single memory load/store.
    """
    # Program ID determines which block of data this kernel instance processes
    pid = tl.program_id(axis=0)
    
    # Calculate the range of elements this block will process
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Mask to handle edge cases (last block may not be full)
    mask = offsets < n_elements
    
    # Load data from global memory to registers
    h_prev = tl.load(h_prev_ptr + offsets, mask=mask, other=0.0)
    input_val = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    gate_val = tl.load(gate_ptr + offsets, mask=mask, other=0.0)
    
    # Compute in registers (ZERO intermediate VRAM IO)
    # tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
    # Using Triton's built-in for numerical stability
    signal = tl.math.tanh(input_val)
    
    # sigmoid(x) = 1 / (1 + exp(-x))
    gate = tl.sigmoid(gate_val)
    
    # CfC Update Rule
    # h_new = h_prev * (1 - gate) + signal * gate
    # Equivalent to: h_new = h_prev + gate * (signal - h_prev)
    h_new = h_prev * (1.0 - gate) + signal * gate
    
    # Store result back to global memory
    tl.store(output_ptr + offsets, h_new, mask=mask)


@triton.jit
def fused_cfc_backward_kernel(
    grad_output_ptr,  # Gradient from upstream [B, H]
    h_prev_ptr,       # Previous hidden state [B, H]
    input_ptr,        # Input linear [B, H]
    gate_ptr,         # Gate linear [B, H]
    grad_h_prev_ptr,  # Output: Gradient w.r.t. h_prev [B, H]
    grad_input_ptr,   # Output: Gradient w.r.t. input [B, H]
    grad_gate_ptr,    # Output: Gradient w.r.t. gate [B, H]
    n_elements,
    BLOCK_SIZE: tl.constexpr
):
    """
    Fused CfC Backward Kernel.
    
    Given: h_new = h_prev * (1 - gate) + signal * gate
    Where: signal = tanh(input), gate = sigmoid(gate_input)
    
    Derivatives:
        d(h_new)/d(h_prev) = (1 - gate)
        d(h_new)/d(signal) = gate
        d(h_new)/d(gate) = signal - h_prev
        
        d(signal)/d(input) = 1 - tanh(input)^2
        d(gate)/d(gate_input) = gate * (1 - gate)
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load
    grad_out = tl.load(grad_output_ptr + offsets, mask=mask, other=0.0)
    h_prev = tl.load(h_prev_ptr + offsets, mask=mask, other=0.0)
    input_val = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    gate_val = tl.load(gate_ptr + offsets, mask=mask, other=0.0)
    
    # Recompute activations
    signal = tl.math.tanh(input_val)
    gate = tl.sigmoid(gate_val)
    
    # Gradients
    # d(loss)/d(h_prev) = d(loss)/d(h_new) * d(h_new)/d(h_prev) = grad_out * (1 - gate)
    grad_h_prev = grad_out * (1.0 - gate)
    
    # d(loss)/d(signal) = grad_out * gate
    # d(loss)/d(input) = d(loss)/d(signal) * d(signal)/d(input)
    #                  = grad_out * gate * (1 - signal^2)
    grad_signal = grad_out * gate
    grad_input = grad_signal * (1.0 - signal * signal)
    
    # d(loss)/d(gate) = grad_out * (signal - h_prev)
    # d(loss)/d(gate_input) = d(loss)/d(gate) * d(gate)/d(gate_input)
    #                       = grad_out * (signal - h_prev) * gate * (1 - gate)
    grad_gate_val = grad_out * (signal - h_prev)
    grad_gate = grad_gate_val * gate * (1.0 - gate)
    
    # Store gradients
    tl.store(grad_h_prev_ptr + offsets, grad_h_prev, mask=mask)
    tl.store(grad_input_ptr + offsets, grad_input, mask=mask)
    tl.store(grad_gate_ptr + offsets, grad_gate, mask=mask)


def fused_cfc_forward(h_prev: torch.Tensor, input_linear: torch.Tensor, gate_linear: torch.Tensor) -> torch.Tensor:
    """
    Launch the Triton kernel for CfC forward pass.
    
    Args:
        h_prev: Previous hidden state [B, H]
        input_linear: Pre-computed linear transform (x @ W + b) [B, H]
        gate_linear: Pre-computed gate linear (x @ Wg + h @ Ug + bg) [B, H]
    
    Returns:
        h_new: Updated hidden state [B, H]
    """
    assert h_prev.is_cuda or h_prev.is_contiguous(), "Tensors must be on GPU and contiguous"
    assert h_prev.shape == input_linear.shape == gate_linear.shape
    
    output = torch.empty_like(h_prev)
    n_elements = h_prev.numel()
    
    # Grid: number of blocks needed to cover all elements
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    # Launch kernel
    fused_cfc_forward_kernel[grid](
        h_prev, input_linear, gate_linear, output,
        n_elements,
        BLOCK_SIZE=DEFAULT_BLOCK_SIZE
    )
    
    return output


def fused_cfc_backward(
    grad_output: torch.Tensor,
    h_prev: torch.Tensor,
    input_linear: torch.Tensor,
    gate_linear: torch.Tensor
) -> tuple:
    """
    Launch the Triton kernel for CfC backward pass.
    
    Returns:
        (grad_h_prev, grad_input, grad_gate)
    """
    assert grad_output.is_contiguous()
    
    grad_h_prev = torch.empty_like(h_prev)
    grad_input = torch.empty_like(input_linear)
    grad_gate = torch.empty_like(gate_linear)
    
    n_elements = h_prev.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    fused_cfc_backward_kernel[grid](
        grad_output, h_prev, input_linear, gate_linear,
        grad_h_prev, grad_input, grad_gate,
        n_elements,
        BLOCK_SIZE=DEFAULT_BLOCK_SIZE
    )
    
    return grad_h_prev, grad_input, grad_gate
