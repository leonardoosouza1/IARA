"""
I.A.R.A. Liquid Neural Network - Continuous-Time Recurrent Model
================================================================
Implements Liquid Time-Constant (LTC) cells for adaptive market modeling.
Uses ODE-based dynamics with input-dependent time constants.

Reference: Hasani et al. (2020) "Liquid Time-constant Networks"
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LiquidConfig:
    """Configuration for Liquid Neural Network."""
    
    input_dim: int = 64           # Number of input features
    hidden_dim: int = 512         # Wide Single Layer
    n_neurons: int = 512          # Wide Single Layer
    n_layers: int = 1             # Single Layer to prevent vanishing gradients
    dt: float = 0.1               # Integration time step
    solver_steps: int = 5         # ODE solver substeps
    sparsity: float = 0.6         # Connection sparsity (60% zeros)
    tau_min: float = 0.1          # Minimum time constant
    tau_max: float = 10.0         # Maximum time constant
    dropout: float = 0.1          # Dropout rate


def create_sparse_mask(shape: Tuple[int, ...], sparsity: float) -> torch.Tensor:
    """
    Create a binary mask with given sparsity (fraction of zeros).
    
    Args:
        shape: Tensor shape
        sparsity: Fraction of connections to zero out
        
    Returns:
        Binary mask tensor
    """
    mask = torch.rand(shape) > sparsity
    return mask.float()


@torch.jit.script
def ltc_euler_step(
    x: torch.Tensor,
    input_: torch.Tensor,
    W_in: torch.Tensor,
    W_rec: torch.Tensor,
    A: torch.Tensor,
    bias: torch.Tensor,
    dt: float,
    tau_min: float,
    tau_max: float
) -> torch.Tensor:
    """
    Semi-Implicit Euler step for LTC dynamics with numerical stability.
    
    Equation:
        dx/dt = -(A + σ(W_in·I))·x + σ(W_rec·x + bias)
        
    Where the effective time constant τ = 1/(A + σ(W_in·I))
    adapts based on input intensity.
    
    Args:
        x: Current hidden state [batch, neurons]
        input_: Input features [batch, input_dim]
        W_in: Input weights [neurons, input_dim]
        W_rec: Recurrent weights [neurons, neurons]
        A: Base time constant parameters [neurons]
        bias: Bias term [neurons]
        dt: Integration time step
        tau_min, tau_max: Time constant bounds
        
    Returns:
        Updated hidden state [batch, neurons]
    """
    # Input-dependent time constant modulation
    # σ(W_in·I) gives input-dependent "gate"
    input_gate = torch.sigmoid(F.linear(input_, W_in))  # [batch, neurons]
    
    # Effective inverse time constant: 1/τ = A + input_gate
    # Add minimum bound (0.01) to prevent instability
    A_pos = F.softplus(A) + 0.01  # Ensure positive with minimum
    inv_tau = A_pos + input_gate
    
    # Clamp τ to reasonable range
    inv_tau = torch.clamp(inv_tau, 1.0 / tau_max, 1.0 / tau_min)
    
    # Clamp hidden state to prevent explosion
    x = torch.clamp(x, -20.0, 20.0)
    
    # Recurrent activation
    # Use LeakyReLU to prevent gradient saturation (Vanishing Gradient fix)
    rec_activation = F.leaky_relu(F.linear(x, W_rec) + bias, negative_slope=0.01)
    
    # Semi-implicit Euler: x_new = (x + dt * rec_activation) / (1 + dt * inv_tau)
    # This is more stable than explicit Euler for stiff ODEs
    denominator = 1.0 + dt * inv_tau
    x_new = (x + dt * rec_activation) / (denominator + 1e-8)  # eps for safety
    
    # Final clamp to ensure bounded output
    x_new = torch.clamp(x_new, -20.0, 20.0)
    
    return x_new


class LTCCell(nn.Module):
    """
    Liquid Time-Constant Cell.
    
    A single-step ODE-based recurrent cell where the time constant
    adapts based on input signals - allowing fast reactions to
    high-volatility inputs and slow integration for stable periods.
    """
    
    def __init__(self, config: LiquidConfig, input_dim: Optional[int] = None, n_neurons: Optional[int] = None):
        super().__init__()
        self.config = config
        
        # dynamic dimensions for stacking
        self.input_dim = input_dim if input_dim is not None else config.input_dim
        self.n_neurons = n_neurons if n_neurons is not None else config.n_neurons
        
        # Input projection
        self.W_in = nn.Parameter(torch.empty(self.n_neurons, self.input_dim))
        
        # Recurrent connections
        self.W_rec = nn.Parameter(torch.empty(self.n_neurons, self.n_neurons))
        
        # Base time constant (learnable)
        self.A = nn.Parameter(torch.empty(self.n_neurons))
        
        # Bias
        self.bias = nn.Parameter(torch.zeros(self.n_neurons))
        
        # Sparsity mask (frozen)
        self.register_buffer(
            "mask_in",
            create_sparse_mask((self.n_neurons, self.input_dim), config.sparsity)
        )
        self.register_buffer(
            "mask_rec",
            create_sparse_mask((self.n_neurons, self.n_neurons), config.sparsity)
        )
        
        # Initialize weights
        self._init_weights()
    
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Orthogonal initialization for stable dynamics."""
        # Orthogonal init for recurrent weights (critical for RNNs)
        nn.init.orthogonal_(self.W_rec)
        
        # Xavier uniform for input weights
        nn.init.xavier_uniform_(self.W_in)
        
        # Initialize bias to zero
        nn.init.zeros_(self.bias)
        
        # Initialize A (time-constant parameter)
        # We want sigmoid(A) to be around 0.5 initally for balanced flow
        # A=0 -> sigmoid(0)=0.5. Softplus(0)=0.69.
        # Let's initialize small positive to ensure flow
        nn.init.constant_(self.A, 0.0)
    
    def forward(
        self,
        input_: torch.Tensor,
        hidden: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for one time step.
        
        Args:
            input_: Input features [batch, input_dim]
            hidden: Previous hidden state [batch, n_neurons]
            
        Returns:
            New hidden state [batch, n_neurons]
        """
        batch_size = input_.size(0)
        
        # Replace NaN/Inf in input with 0
        input_ = torch.nan_to_num(input_, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Initialize hidden state if None
        if hidden is None:
            hidden = torch.zeros(
                batch_size, self.n_neurons,
                device=input_.device, dtype=input_.dtype
            )
        
        # Apply sparsity masks
        W_in_sparse = self.W_in * self.mask_in
        W_rec_sparse = self.W_rec * self.mask_rec
        
        # Multi-step integration for stability
        x = hidden
        dt_sub = self.config.dt / self.config.solver_steps
        
        for _ in range(self.config.solver_steps):
            x = ltc_euler_step(
                x, input_,
                W_in_sparse, W_rec_sparse,
                self.A, self.bias,
                dt_sub,
                self.config.tau_min,
                self.config.tau_max
            )
        
        return x


class LiquidRNN(nn.Module):
    """
    Liquid RNN wrapper for sequence processing.
    
    Processes a sequence of inputs through LTC cells
    and returns the sequence of hidden states.
    """
    
    def __init__(self, config: LiquidConfig):
        super().__init__()
        self.config = config
        
        # Input projection (if needed to match dimensions)
        self.input_proj = nn.Linear(config.input_dim, config.input_dim)
        
        # Single Wide LTC Cell
        # We enforce input_dim and n_neurons from config
        self.ltc_cell = LTCCell(config)
        
        # Output projection
        self.output_proj = nn.Linear(config.n_neurons, config.hidden_dim)
        
        self.dropout = nn.Dropout(config.dropout)
        self.norm = nn.LayerNorm(config.hidden_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
        return_all: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process a sequence through the Liquid RNN.
        
        Args:
            x: Input sequence [batch, seq_len, input_dim]
            hidden: Initial hidden state [batch, n_neurons] (Single Layer)
            return_all: If True, return all hidden states; else only final
            
        Returns:
            Tuple of (outputs, final_hidden)
            outputs: [batch, seq_len, hidden_dim] if return_all else [batch, hidden_dim]
        """
        batch_size, seq_len, _ = x.shape
        
        # Initialize hidden state if None
        if hidden is None:
            hidden = torch.zeros(
                batch_size, self.config.n_neurons,
                device=x.device, dtype=x.dtype
            )
        
        # Project input
        x = self.input_proj(x)
        x = self.dropout(x)
        
        # Process sequence
        outputs = []
        h = hidden
        
        # Temporal loop
        for t in range(seq_len):
            current_input = x[:, t, :]
            
            # Single Layer Step
            h = self.ltc_cell(current_input, h)
            
            # Project to output dim
            out = self.output_proj(h)
            out = self.norm(out)
            outputs.append(out)
        
        if return_all:
            outputs = torch.stack(outputs, dim=1)  # [batch, seq, hidden]
        else:
            outputs = outputs[-1]  # [batch, hidden]
        
        return outputs, h


class TraderHead(nn.Module):
    """
    Multi-task trading head.
    
    Outputs:
    1. Regression: Predicted log-return (continuous)
    2. Classification: Direction probability (up/down)
    """
    
    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        
        # Shared layer
        self.shared = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Regression head (predict log-return)
        self.reg_head = nn.Linear(hidden_dim, 1)
        
        # Classification head (predict direction)
        self.cls_head = nn.Linear(hidden_dim, 1)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Hidden states [batch, seq, hidden] or [batch, hidden]
            
        Returns:
            Tuple of (log_return, direction_logit)
        """
        # Shared features
        features = self.shared(x)
        
        # Predictions
        log_return = self.reg_head(features).squeeze(-1)  # [batch, seq] or [batch]
        direction = self.cls_head(features).squeeze(-1)   # [batch, seq] or [batch]
        
        return log_return, direction


class IARA(nn.Module):
    """
    I.A.R.A. - Intelligent Adaptive Regressive Architecture
    
    Full Liquid Neural Network for trading with:
    - Input normalization layer
    - LiquidRNN backbone
    - Multi-task TraderHead
    """
    
    def __init__(self, config: LiquidConfig):
        super().__init__()
        self.config = config
        
        # Input layer norm for stability
        self.input_norm = nn.LayerNorm(config.input_dim)
        
        # Liquid backbone
        self.liquid_rnn = LiquidRNN(config)
        
        # Trading head
        self.trader_head = TraderHead(config.hidden_dim, config.dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input features [batch, seq_len, input_dim]
            hidden: Initial hidden state
            
        Returns:
            Tuple of (log_return, direction_logit, final_hidden)
        """
        # Normalize input
        x = self.input_norm(x)
        
        # Process through Liquid RNN
        outputs, final_hidden = self.liquid_rnn(x, hidden, return_all=True)
        
        # Get trading predictions
        log_return, direction = self.trader_head(outputs)
        
        return log_return, direction, final_hidden
    
    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def count_active_connections(self) -> int:
        """Count non-zero connections (after sparsity mask)."""
        cell = self.liquid_rnn.ltc_cell
        active_in = (cell.mask_in > 0).sum().item()
        active_rec = (cell.mask_rec > 0).sum().item()
        return int(active_in + active_rec)


def main():
    """Test I.A.R.A. forward pass."""
    config = LiquidConfig(
        input_dim=64,
        hidden_dim=128,
        n_neurons=64,
        dt=0.1,
        solver_steps=5,
        sparsity=0.6
    )
    
    model = IARA(config)
    print(f"Parameters: {model.count_parameters():,}")
    print(f"Active connections: {model.count_active_connections():,}")
    
    # Test forward
    batch_size = 8
    seq_len = 32
    x = torch.randn(batch_size, seq_len, config.input_dim)
    
    log_ret, direction, hidden = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Log-return shape: {log_ret.shape}")
    print(f"Direction shape: {direction.shape}")
    print(f"Hidden shape: {hidden.shape}")
    
    # Test backward
    loss = log_ret.mean() + direction.mean()
    loss.backward()
    print("Backward pass successful!")


if __name__ == "__main__":
    main()
