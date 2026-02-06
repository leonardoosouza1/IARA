"""
TRINITY BRAIN v8.5 [PHASE 1]
============================
The Unified Model Class.

Components:
1. Tactical: LNN (ModulatedIARA)
2. Structural: LLM Gating (Context Projection)
3. Safety: News Awareness Brake (Gradient-Safe)
"""

import torch
import torch.nn as nn
from src.brain.models.liquid_v8 import ModulatedIARA

class TrinityBrain(nn.Module):
    def __init__(self, lnn_core: ModulatedIARA, llm_hidden_dim: int):
        super().__init__()
        self.lnn = lnn_core
        
        # Project LLM context to LNN hidden dimension
        # LNN hidden dim is usually 128 (defined in config)
        # We need to know lnn_core.config.hidden_dim
        lnn_dim = lnn_core.config.hidden_dim
        
        self.context_proj = nn.Linear(llm_hidden_dim, lnn_dim)
        
        # Prepare Gate for Future News (Sentinel Interface)
        # Input: [Sentiment (-1 to 1), Impact (0 to 1)]
        # Output: 32 dim latent representation (reserved for future Phase 2 logic if needed)
        self.news_gate = nn.Linear(2, 32)

    def forward(self, market_data: torch.Tensor, llm_context: torch.Tensor, news_vector: torch.Tensor) -> torch.Tensor:
        """
        Args:
            market_data: [Batch, Seq, Features]
            llm_context: [Batch, LLM_Dim]
            news_vector: [Batch, 2] -> [Sentiment, Impact]
        """
        
        # 1. Modulation (The Dual Core Logic)
        # Project LLM embedding to LNN latent space
        modulation = self.context_proj(llm_context) 
        # modulation shape: [Batch, LNN_Hidden]
        
        # 2. Tactical Execution (Pass modulation to CfC)
        x = self.lnn(market_data, external_modulation=modulation)
        
        # 3. Emergency Brake (Gradient-Safe)
        # news_vector[:, 1] is Impact Score (0.0 to 1.0)
        # If impact is 1.0 (Thermo-Nuclear), we want to suppress output.
        # Formula: output * (1.0 - impact)
        
        impact_score = news_vector[:, 1].unsqueeze(-1) # [Batch, 1]
        
        # Ensure impact is within [0, 1] just in case
        impact_score = torch.clamp(impact_score, 0.0, 1.0)
        
        # Apply Brake
        # x is [Batch] (from ModulatedIARA)
        # impact_score is [Batch, 1]
        # We need data to be [Batch, 1] to avoid [Batch, Batch] broadcasting
        output = x.unsqueeze(-1) * (1.0 - impact_score)
        
        return output.squeeze(-1)
