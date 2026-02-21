"""
I.A.R.A. v10.0 - Challenger Brain (Adversarial Generator)
==========================================================
The "Evil Professor" model. Its sole purpose is to find weaknesses 
in the Swarm Intelligence model by generating "pranks" (adversarial noise)
that lead to incorrect trading decisions.

It uses a light MLP architecture to perturb input features while
staying within a certain "plausibility" budget.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class ChallengerGenerator(nn.Module):
    """
    Generates adversarial noise to challenge the Trader model.
    """
    
    def __init__(self, input_dim: int = 12, hidden_dim: int = 64):
        super().__init__()
        
        self.input_dim = input_dim
        
        # Encoder: Understand the current market context
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU()
        )
        
        # Noise Head: Predicts the perturbation delta
        self.noise_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Tanh() # Constrain perturbation to [-1, 1] relative scale
        )
        
        # Intensity controller (Dynamic Pressure)
        self.pressure_scaler = nn.Parameter(torch.tensor(0.01)) 
        
    def forward(
        self, 
        x: torch.Tensor, 
        pressure: float = 1.0
    ) -> torch.Tensor:
        """
        x: (batch_size, seq_len, input_dim)
        returns: x_pranked (batch_size, seq_len, input_dim)
        """
        batch_size, seq_len, _ = x.size()
        
        # Flatten time dimension for context analysis
        x_flat = x.view(-1, self.input_dim)
        
        # Generate noise
        features = self.encoder(x_flat)
        delta = self.noise_head(features)
        
        # Reshape back to (batch_size, seq_len, input_dim)
        delta = delta.view(batch_size, seq_len, self.input_dim)
        
        # Apply prank: x_new = x + pressure * epsilon * delta
        # We use a small base intensity and scale it by the current pressure
        x_pranked = x + (delta * self.pressure_scaler * pressure)
        
        return x_pranked

class OverfittingGuard(nn.Module):
    """
    Monitors Training vs Validation gap and adjusts Challenger Pressure.
    If the Trader starts 'memorizing' (Loss_Train << Loss_Val), 
    the Guard increases the difficulty of the pranks.
    """
    
    def __init__(self, ema_alpha: float = 0.1):
        super().__init__()
        self.ema_alpha = ema_alpha
        self.register_buffer("train_loss_ema", torch.tensor(1.0))
        self.register_buffer("val_loss_ema", torch.tensor(1.0))
        self.register_buffer("pressure", torch.tensor(1.0))
        
    def update(self, train_loss: float, val_loss: float):
        # Update EMA
        self.train_loss_ema = (1 - self.ema_alpha) * self.train_loss_ema + self.ema_alpha * train_loss
        self.val_loss_ema = (1 - self.ema_alpha) * self.val_loss_ema + self.ema_alpha * val_loss
        
        # Calculate Overfitting Ratio
        # If val_loss is significantly higher than train_loss, we have overfitting
        ratio = (self.val_loss_ema / (self.train_loss_ema + 1e-6)).item()
        
        # Adjust pressure: if ratio > 1.2, increase pressure
        if ratio > 1.2:
            self.pressure *= 1.05 # Increase by 5%
        elif ratio < 1.05:
            self.pressure *= 0.95 # Relax if generalizing well
            
        # Clamp pressure
        self.pressure = torch.clamp(self.pressure, 0.1, 5.0)
        
    def get_pressure(self) -> float:
        return self.pressure.item()
