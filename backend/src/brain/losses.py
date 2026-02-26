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
        pt = torch.clamp(pt, min=1e-7, max=1.0 - 1e-7)
        
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
    
    def __init__(self, num_classes: int = 2, label_smoothing: float = 0.1, weight_fp: float = 10.0, weight_fn: float = 1.0):
        super().__init__()
        self.num_classes = num_classes
        self.label_smoothing = label_smoothing
        # weights: [Long, Short, Wait] -> Target 2 is Wait (minority class)
        self.weights = torch.tensor([weight_fn, weight_fn, weight_fp], dtype=torch.float32)
        
    def forward(
        self, 
        worker_logits: torch.Tensor, 
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            worker_logits: All worker predictions [batch, num_workers, 3]
            targets: Action targets [batch] {0, 1, 2}
            
        Returns:
            Average auxiliary loss across all workers
        """
        if self.weights.device != worker_logits.device:
            self.weights = self.weights.to(worker_logits.device)
            
        batch_size, num_workers, num_classes = worker_logits.shape
        
        total_loss = torch.tensor(0.0, device=targets.device)
        
        for w in range(num_workers):
            w_logits = worker_logits[:, w, :]  # [Batch, 3]
            loss = F.cross_entropy(
                w_logits, 
                targets,
                weight=self.weights,
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
        
        # Coordinator probabilities (0=Long, 1=Short, 2=Wait)
        coord_probs = F.softmax(coord_logits, dim=-1)  # [batch, 3]
        coord_long_prob = coord_probs[:, 0]
        coord_short_prob = coord_probs[:, 1]
        
        # Worker probabilities
        worker_probs = F.softmax(worker_logits, dim=-1)  # [batch, workers, 3]
        
        # Worker votes
        worker_votes = worker_logits.argmax(dim=-1)  # [batch, workers]
        
        # Worker confidence (max probability)
        worker_conf = worker_probs.max(dim=-1)[0]  # [batch, workers]
        
        # Count confident votes
        confident_mask = worker_conf > self.threshold
        
        # Count confident LONG votes
        n_long = ((worker_votes == 0) & confident_mask).float().sum(dim=-1)
        
        # Count confident SHORT votes
        n_short = ((worker_votes == 1) & confident_mask).float().sum(dim=-1)
        
        # Consensus disagreement penalty
        # If workers say LONG but Coordinator Long prob is low
        long_consensus = (n_long >= self.consensus_min).float()
        long_penalty = long_consensus * F.relu(0.5 - coord_long_prob)
        
        # If workers say SHORT but Coordinator Short prob is low
        short_consensus = (n_short >= self.consensus_min).float()
        short_penalty = short_consensus * F.relu(0.5 - coord_short_prob)
        
        total_penalty = long_penalty + short_penalty
        return total_penalty.mean()


class AsymmetricCrossEntropyLoss(nn.Module):
    """
    Custom Loss for Sniper Mode (High Precision).
    Penalizes False Positives (acting on noise, target=Wait) heavier than 
    False Negatives (missing a trade, target=Long/Short).
    
    Args:
        weight_fp: Weight penalty when target is Wait (class 2)
        weight_fn: Weight penalty when target is Action (class 0 or 1)
        label_smoothing: CrossEntropy label smoothing
    """
    def __init__(self, weight_fp: float = 10.0, weight_fn: float = 1.0, label_smoothing: float = 0.1):
        super().__init__()
        # weights: [Long, Short, Wait] -> Target 2 is Wait
        self.weights = torch.tensor([weight_fn, weight_fn, weight_fp], dtype=torch.float32)
        self.label_smoothing = label_smoothing
        
    def forward(self, logits: torch.Tensor, targets: torch.Tensor, weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.weights.device != logits.device:
            self.weights = self.weights.to(logits.device)
            
        # reduction='none' so we can apply optional sample weights
        # epsilon padding added intrinsically by cross_entropy via Softmax
        loss = F.cross_entropy(logits, targets.long(), weight=self.weights, label_smoothing=self.label_smoothing, reduction='none')
        
        if weight is not None:
            loss = loss * weight
            
        return loss.mean()


class SwarmLoss(nn.Module):
    """
    Complete loss function for Swarm Intelligence training.
    
    Combines:
    1. AsymmetricCrossEntropyLoss for Coordinator (Sniper Mode)
    2. CrossEntropyLoss for Workers (auxiliary)
    3. ConsensusLoss for regularization
    
    Formula:
    L_total = L_coord + λ * L_workers + β * L_consensus
    
    Args:
        alpha: Focal loss alpha (default 0.75)
        gamma: Focal loss gamma (default 2.0)
        consensus_weight: Weight for consensus loss β (default 0.1)
        label_smoothing: Smoothing for worker loss (default 0.1)
    """
    
    def __init__(
        self,
        alpha: float = 0.75,
        gamma: float = 2.0,
        consensus_weight: float = 0.1,
        label_smoothing: float = 0.1,
        diversity_weight: float = 0.1 # 🌋 NEW: Penalty to avoid weight collapse
    ):
        super().__init__()
        
        # Learnable weight logits (Natural Dynamic Weighting)
        # Softmax of this tensor determines final weights.
        # Initialized to favor Coordinator (Index 0) at approx 85%
        self.weight_logits = nn.Parameter(torch.tensor([2.5, 0.0, 0.0]))
        
        # Sub-losses (Sniper Mode Asymmetric Loss)
        self.coord_loss = AsymmetricCrossEntropyLoss(weight_fp=10.0, weight_fn=1.0, label_smoothing=label_smoothing)
        self.focal_alt = AsymmetricFocalLoss(alpha=alpha, gamma=gamma) # Still keep for binary auxiliary if needed
        self.worker_loss = WorkerAuxiliaryLoss(
            num_classes=3, 
            label_smoothing=label_smoothing,
            weight_fp=10.0,
            weight_fn=1.0
        )
        self.consensus_loss = ConsensusLoss()
        self.diversity_weight = diversity_weight
        
    def forward(
        self,
        coord_logits: torch.Tensor,
        worker_logits: torch.Tensor,
        targets: torch.Tensor,
        weight: Optional[torch.Tensor] = None
    ) -> dict:
        """
        Compute all losses with automatic Natural Scaling.
        Targets: [Batch] with values {0: Long, 1: Short, 2: Neutral}
        """
        # 1. Compute individual losses
        
        # Coordinator is now 3-class {Long, Short, Neutral} mapped via Asymmetric Loss
        # coord_logits shape: [Batch, 3]
        # targets shape: [Batch]
        l_coord = self.coord_loss(coord_logits, targets, weight=weight)
             
        l_workers = self.worker_loss(worker_logits, targets.long())
        l_consensus = self.consensus_loss(coord_logits, worker_logits)
        
        # 2. Natural Dynamic Weighting (Softmax)
        weights = F.softmax(self.weight_logits, dim=0)
        
        # 3. Anchor Mechanism: Ensure Coordinator (Líder) always has strong voice
        # We enforce a minimum presence (e.g., mixing with a static anchor)
        # This prevents the weight from falling to zero if other tasks are "easier"
        coord_weight = 0.5 * weights[0] + 0.3 # At least 30%, up to 80%
        
        # 4. Total Weighted Loss
        # FIX: Include l_workers to allow gradients to flow to weight_logits[1]
        total_l = (
            coord_weight * l_coord + 
            weights[1] * l_workers * 0.5 + 
            weights[2] * l_consensus * 0.1
        )
        
        # 4. Weight Entropy Penalty (🌋 Phase 19)
        # Prevents weights from becoming "one-hot" (winner-take-all collapse)
        # Entropy is sum(-p * log(p)). To maximize it, we minimize -entropy.
        weight_entropy = - (weights * torch.log(weights + 1e-8)).sum()
        total_l = total_l - (self.diversity_weight * weight_entropy)
        
        return {
            "total": total_l,
            "coord": l_coord,
            "workers": l_workers,
            "consensus": l_consensus,
            "entropy": weight_entropy,
            "weights": [coord_weight.item(), weights[1].item(), weights[2].item()]
        }


# Quick test
if __name__ == "__main__":
    print("Testing SwarmLoss...")
    
    batch_size = 64
    num_workers = 5
    
    # Simulate outputs
    coord_logits = torch.randn(batch_size, 3)
    worker_logits = torch.randn(batch_size, num_workers, 3)
    targets = torch.randint(0, 3, (batch_size,)).float()
    
    # Create loss
    criterion = SwarmLoss(
        alpha=0.75,
        gamma=2.0,
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
