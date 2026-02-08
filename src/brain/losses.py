"""
I.A.R.A. v9 - Loss Functions (Swarm Intelligence)
==================================================
Multi-objective loss for training the Swarm:

1. Main Loss (Coordinator): AsymmetricFocalLoss
   - Focuses on hard examples
   - Higher weight for positive class (catching trends)
   
2. Auxiliary Loss (Workers): CrossEntropy
   - Forces each worker to be a competent predictor
   - Prevents "lazy workers" that rely on the Coordinator
   
3. Consensus Loss (Regularization): Disagreement Penalty
   - Penalizes when Coordinator contradicts strong worker consensus
   - Helps maintain ensemble coherence

Formula:
L_total = L_coord + λ * Σ L_workers + β * L_consensus

Reference: Focal Loss (Lin et al., 2017) + Ensemble Distillation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class AsymmetricFocalLoss(nn.Module):
    """
    Asymmetric Focal Loss for imbalanced classification.
    
    Features:
    - Focal term: Down-weights easy examples (gamma)
    - Alpha balancing: Higher weight for positive class
    - Works with logits (more numerically stable)
    
    Args:
        alpha: Weight for positive class (default 0.75)
        gamma: Focusing parameter (default 2.0)
        reduction: 'mean', 'sum', or 'none'
    """
    
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(
        self, 
        logits: torch.Tensor, 
        targets: torch.Tensor,
        weight: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            logits: Raw predictions [batch] or [batch, 1]
            targets: Binary targets [batch] (0 or 1)
            weight: Optional per-sample weights [batch]
            
        Returns:
            Loss value (scalar if reduction='mean' or 'sum')
        """
        logits = logits.view(-1)
        targets = targets.view(-1).float()
        
        # BCE with logits (numerically stable)
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        
        # Probability of correct class
        pt = torch.exp(-bce_loss)
        
        # Focal term: (1 - pt)^gamma
        focal_term = (1 - pt) ** self.gamma
        
        # Alpha balancing
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        # Combined loss
        loss = alpha_t * focal_term * bce_loss
        
        # Optional per-sample weighting
        if weight is not None:
            loss = loss * weight.view(-1)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class WorkerAuxiliaryLoss(nn.Module):
    """
    Auxiliary loss for worker predictions.
    
    Ensures each worker learns to predict independently,
    preventing "lazy workers" that rely on ensemble averaging.
    
    Uses label smoothing for regularization.
    """
    
    def __init__(self, num_classes: int = 3, label_smoothing: float = 0.1):
        super().__init__()
        self.num_classes = num_classes
        self.label_smoothing = label_smoothing
        
    def forward(
        self, 
        worker_logits: torch.Tensor, 
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            worker_logits: All worker predictions [batch, num_workers, num_classes]
            targets: Binary targets [batch] (0=SHORT, 1=LONG)
            
        Returns:
            Average auxiliary loss across all workers
        """
        batch_size, num_workers, num_classes = worker_logits.shape
        
        # Convert binary target to 3-class
        # 0 -> SHORT (class 1)
        # 1 -> LONG (class 0)
        # We treat NEUTRAL (class 2) as never being the target
        worker_targets = torch.zeros(batch_size, dtype=torch.long, device=targets.device)
        worker_targets[targets > 0.5] = 0  # LONG
        worker_targets[targets <= 0.5] = 1  # SHORT
        
        total_loss = torch.tensor(0.0, device=targets.device)
        
        for w in range(num_workers):
            w_logits = worker_logits[:, w, :]  # [batch, 3]
            loss = F.cross_entropy(
                w_logits, 
                worker_targets,
                label_smoothing=self.label_smoothing
            )
            total_loss = total_loss + loss
        
        # Average over workers
        return total_loss / num_workers


class ConsensusLoss(nn.Module):
    """
    Regularization loss to maintain ensemble coherence.
    
    Penalizes when the Coordinator strongly disagrees with
    a clear worker consensus.
    
    Logic:
    - If >=3 workers vote LONG with high confidence, 
      and Coordinator predicts SHORT, apply penalty.
    """
    
    def __init__(self, threshold: float = 0.6, consensus_min: int = 3):
        super().__init__()
        self.threshold = threshold  # Confidence threshold for "clear" vote
        self.consensus_min = consensus_min  # Minimum workers for consensus
        
    def forward(
        self, 
        coord_logits: torch.Tensor,
        worker_logits: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            coord_logits: Coordinator predictions [batch, 1]
            worker_logits: Worker predictions [batch, num_workers, 3]
            
        Returns:
            Consensus penalty loss
        """
        batch_size = coord_logits.size(0)
        
        # Coordinator probability
        coord_prob = torch.sigmoid(coord_logits.view(-1))  # [batch]
        
        # Worker probabilities
        worker_probs = F.softmax(worker_logits, dim=-1)  # [batch, workers, 3]
        
        # Worker votes (0=LONG, 1=SHORT, 2=NEUTRAL)
        worker_votes = worker_logits.argmax(dim=-1)  # [batch, workers]
        
        # Worker confidence (max probability)
        worker_conf, _ = worker_probs.max(dim=-1)  # [batch, workers]
        
        # Count confident votes
        confident_mask = worker_conf > self.threshold
        
        # Count confident LONG votes
        long_votes = (worker_votes == 0) & confident_mask
        n_long = long_votes.float().sum(dim=-1)  # [batch]
        
        # Count confident SHORT votes
        short_votes = (worker_votes == 1) & confident_mask
        n_short = short_votes.float().sum(dim=-1)  # [batch]
        
        # Consensus disagreement penalty
        # If workers say LONG but Coordinator says SHORT
        long_consensus = (n_long >= self.consensus_min).float()
        long_penalty = long_consensus * F.relu(0.5 - coord_prob)  # Penalize low prob
        
        # If workers say SHORT but Coordinator says LONG
        short_consensus = (n_short >= self.consensus_min).float()
        short_penalty = short_consensus * F.relu(coord_prob - 0.5)  # Penalize high prob
        
        total_penalty = long_penalty + short_penalty
        
        return total_penalty.mean()


class SwarmLoss(nn.Module):
    """
    Complete loss function for Swarm Intelligence training.
    
    Combines:
    1. AsymmetricFocalLoss for Coordinator
    2. CrossEntropyLoss for Workers (auxiliary)
    3. ConsensusLoss for regularization
    
    Formula:
    L_total = L_coord + λ * L_workers + β * L_consensus
    
    Args:
        alpha: Focal loss alpha (default 0.75)
        gamma: Focal loss gamma (default 2.0)
        aux_weight: Weight for auxiliary loss λ (default 0.4)
        consensus_weight: Weight for consensus loss β (default 0.1)
        label_smoothing: Smoothing for worker loss (default 0.1)
    """
    
    def __init__(
        self,
        alpha: float = 0.75,
        gamma: float = 2.0,
        aux_weight: float = 0.4,
        consensus_weight: float = 0.1,
        label_smoothing: float = 0.1
    ):
        super().__init__()
        
        self.aux_weight = aux_weight
        self.consensus_weight = consensus_weight
        
        # Sub-losses
        self.coord_loss = AsymmetricFocalLoss(alpha=alpha, gamma=gamma)
        self.worker_loss = WorkerAuxiliaryLoss(
            num_classes=3, 
            label_smoothing=label_smoothing
        )
        self.consensus_loss = ConsensusLoss()
        
    def forward(
        self,
        coord_logits: torch.Tensor,
        worker_logits: torch.Tensor,
        targets: torch.Tensor,
        weight: Optional[torch.Tensor] = None
    ) -> dict:
        """
        Compute all losses.
        
        Args:
            coord_logits: Coordinator predictions [batch, 1]
            worker_logits: Worker predictions [batch, num_workers, 3]
            targets: Binary targets [batch]
            weight: Optional per-sample weights [batch]
            
        Returns:
            Dictionary with:
            - total: Combined loss (for backward)
            - coord: Coordinator loss (for logging)
            - workers: Workers auxiliary loss (for logging)
            - consensus: Consensus regularization (for logging)
        """
        # 1. Main loss (Coordinator)
        l_coord = self.coord_loss(coord_logits, targets, weight)
        
        # 2. Auxiliary loss (Workers)
        l_workers = self.worker_loss(worker_logits, targets)
        
        # 3. Consensus regularization
        l_consensus = self.consensus_loss(coord_logits, worker_logits)
        
        # Total
        total = l_coord + self.aux_weight * l_workers + self.consensus_weight * l_consensus
        
        return {
            "total": total,
            "coord": l_coord,
            "workers": l_workers,
            "consensus": l_consensus
        }


# Quick test
if __name__ == "__main__":
    print("Testing SwarmLoss...")
    
    batch_size = 64
    num_workers = 5
    
    # Simulate outputs
    coord_logits = torch.randn(batch_size, 1)
    worker_logits = torch.randn(batch_size, num_workers, 3)
    targets = torch.randint(0, 2, (batch_size,)).float()
    
    # Create loss
    criterion = SwarmLoss(
        alpha=0.75,
        gamma=2.0,
        aux_weight=0.4,
        consensus_weight=0.1,
        label_smoothing=0.1
    )
    
    # Compute
    losses = criterion(coord_logits, worker_logits, targets)
    
    print(f"  Total loss: {losses['total']:.4f}")
    print(f"  Coord loss: {losses['coord']:.4f}")
    print(f"  Worker loss: {losses['workers']:.4f}")
    print(f"  Consensus loss: {losses['consensus']:.4f}")
    
    # Verify gradient flow
    losses['total'].backward()
    print("  ✅ Gradient flow OK!")
    print("  ✅ SwarmLoss OK!")
