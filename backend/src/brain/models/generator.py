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
    I.A.R.A. v10.0 - Challenger Brain (The Sadistic Professor)
    ==========================================================
    Generates realistic market manipulations to challenge the Trader.
    Modes:
      - Stop Hunt: Simulated wicks (Fake Breakouts)
      - Liquidation Cascade: Fast, high-volume price drops
      - Choppy Market: Lateral movement/noise injection
      - Neural Prank: Learned adversarial noise (v9.0)
    """
    
    def __init__(self, input_dim: int = 12, hidden_dim: int = 64, num_arms: int = 4):
        super().__init__()
        
        self.input_dim = input_dim
        self.num_arms = num_arms
        
        # RL / Multi-Armed Bandit State
        self.register_buffer("arm_counts", torch.ones(num_arms)) # Inicia com 1 para exploração
        self.register_buffer("arm_rewards", torch.zeros(num_arms)) # Recompensa: aumento da Loss do Trader
        self.exploration_c = 2.0 # UCB Exploration parameter
        
        # Neural Backbone (for learned pranks)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU()
        )
        
        self.noise_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Tanh()
        )
        
        self.pressure_scaler = nn.Parameter(torch.tensor(0.005)) 
        
    def _select_attack_mode(self) -> int:
        """UCB (Upper Confidence Bound) Arm Selection"""
        total_counts = self.arm_counts.sum()
        q_values = self.arm_rewards / self.arm_counts
        ucb_values = q_values + self.exploration_c * torch.sqrt(torch.log(total_counts) / self.arm_counts)
        return torch.argmax(ucb_values).item()

    def update_reward(self, arm: int, trader_loss_increase: float):
        """Called by ContinuousLearner after the Trader's forward pass"""
        self.arm_counts[arm] += 1
        self.arm_rewards[arm] += trader_loss_increase
        
    def _apply_stop_hunt(self, x: torch.Tensor, max_perturbation: torch.Tensor) -> torch.Tensor:
        """
        Simulates Stop Hunt (Violinada).
        Push price (Wick) past support/resistance and then snap back.
        Affected Features (indices):
          - 5: BB %B (Spike)
          - 6: ATR (Spike)
          - 9: Velocity (Sharp reversal)
        """
        batch_size, seq_len, dim = x.size()
        x_pranked = x.clone()
        
        # Pick 1-3 random time steps to inject wicks
        for b in range(batch_size):
            num_wicks = torch.randint(1, 4, (1,)).item()
            for _ in range(num_wicks):
                t = torch.randint(seq_len // 2, seq_len - 1, (1,)).item()
                direction = 1 if torch.rand(1).item() > 0.5 else -1
                
                # Intensidade baseada no cap (max_perturbation) do timestep atual
                intensidade = max_perturbation[b, t].item() * 2.0 
                
                # BB %B (idx 5) - Pura quebra de banda
                x_pranked[b, t, 5] += direction * intensidade
                
                # ATR (idx 6) - Volatilidade explode no hunt
                x_pranked[b, t, 6] *= (1.0 + 0.5 * max_perturbation[b, t].item())
                
                # Velocity (idx 9) - O movimento é rápido
                x_pranked[b, t, 9] += direction * intensidade * 3.0
                
        return x_pranked

    def _apply_liquidation_cascade(self, x: torch.Tensor, max_perturbation: torch.Tensor) -> torch.Tensor:
        """
        Simulates Efeito Dominó.
        High volume drops in a short period.
        """
        batch_size, seq_len, dim = x.size()
        x_pranked = x.clone()
        
        for b in range(batch_size):
            # Cascade starts in the last 20% of the sequence
            start_t = int(seq_len * 0.8)
            direction = -1 # Liquidation is usually a drop in this mode
            
            for t in range(start_t, seq_len):
                decay = (t - start_t + 1)
                drop_force = max_perturbation[b, t].item() * decay * 0.2
                
                # Sentiment (idx 11) - Panico
                x_pranked[b, t, 11] -= drop_force
                
                # Velocity (idx 9) - Acelerando
                x_pranked[b, t, 9] -= drop_force * 2.0
                
                # ATR (idx 6) - Panico gera volatilidade
                x_pranked[b, t, 6] *= (1.0 + 0.1 * max_perturbation[b, t].item() * decay)
                
        return x_pranked

    def _apply_choppy_market(self, x: torch.Tensor, max_perturbation: torch.Tensor) -> torch.Tensor:
        """
        Simulates Choppy Market (Ruído Lateral).
        Flatten indicators and add high-frequency noise.
        """
        # Diminue a variância dos indicadores (tendendo ao neutro/lateral)
        # 1.0 - pressure * 0.1 (clamped) using max_perturbation mean roughly
        pressure_proxy = max_perturbation.mean()
        factor = torch.clamp(torch.tensor(1.0 - pressure_proxy * 0.1), 0.1, 1.0).to(x.device)
        x_pranked = x * factor
        
        # Adiciona ruído jitter de alta frequência limitado pelo ATR gap
        noise = torch.randn_like(x) * max_perturbation.unsqueeze(-1)
        return x_pranked + noise

    def forward(
        self, 
        x: torch.Tensor, 
        pressure: float = 1.0,
        callback: Optional[callable] = None
    ) -> Tuple[torch.Tensor, int]:
        """
        Selection logic for the "Professor Sádico".
        Uses UCB Multi-Armed Bandit to exploit model weaknesses.
        """
        mode = self._select_attack_mode()
        
        # ATR Epsilon Cap (Realism Bound)
        # Assuming ATR is idx 6 (or proxy pressure if missing)
        batch_size, seq_len, dim = x.size()
        if dim > 6:
            atr_tensor = torch.abs(x[:, :, 6]) 
            max_perturbation = atr_tensor * 0.2 * pressure
        else:
            max_perturbation = torch.full((batch_size, seq_len), 0.1 * pressure, device=x.device)
            
        # Add tiny floor to ensure some perturbation happens even in flat markets
        max_perturbation = torch.clamp(max_perturbation, min=0.01)
        
        if mode == 0:
            if callback: callback(f"😈 Professor Sádico: Simulando Stop Hunt (Wick/Violinada) | Intensidade: {pressure:.2f}...")
            return self._apply_stop_hunt(x, max_perturbation), mode
        elif mode == 1:
            if callback: callback(f"😈 Professor Sádico: Simulando Liquidation Cascade (Efeito Dominó) | Intensidade: {pressure:.2f}...")
            return self._apply_liquidation_cascade(x, max_perturbation), mode
        elif mode == 2:
            if callback: callback(f"😈 Professor Sádico: Injetando Ruído Lateral (Choppy Market) | Intensidade: {pressure:.2f}...")
            return self._apply_choppy_market(x, max_perturbation), mode
        else:
            if callback: callback(f"😈 Professor Sádico: Aplicando Neural Prank (Adversarial Noise) | Intensidade: {pressure:.2f}...")
            # Traditional Neural Prank
            x_flat = x.view(-1, self.input_dim)
            features = self.encoder(x_flat)
            delta = self.noise_head(features)
            delta = delta.view(batch_size, seq_len, self.input_dim)
            
            # Use max_perturbation to cap neural noise
            delta_capped = torch.clamp(delta, min=-1.0, max=1.0) * max_perturbation.unsqueeze(-1)
            return x + (delta_capped * self.pressure_scaler), mode

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
