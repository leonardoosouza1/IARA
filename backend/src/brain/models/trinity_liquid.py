import torch
import torch.nn as nn
from typing import Optional

class TrinityLiquidBrain(nn.Module):
    """
    I.A.R.A. v8.5 "Pure Liquid Trinity"
    ===================================
    The fusion module connecting the two liquid brains:
    1. Tactical Agent (LNN): Fast, recurrent, continuous-time execution.
    2. Strategic Agent (LFM): Slow, long-context, reasoning modulation.
    """
    def __init__(self, lnn_core: nn.Module, lfm_hidden_dim: int):
        super().__init__()
        self.lnn = lnn_core
        
        # Bridge Layer: Projects LFM thinking (text/strategy) to LNN input space (math/tactical)
        # Assuming lnn_core.input_size is available (defined in ModulatedIARA)
        self.bridge_layer = nn.Linear(lfm_hidden_dim, lnn_core.input_size)
        
        # News Gate: Modulation from Sentinel (Sentiment, Impact)
        # Input: 2 (Sentiment Score, Impact Score) -> Output: 32 (Hidden Modulation)
        # We might want to project this to match modulation dim or keep it separate.
        # For v8.5, we implement a simple brake/boost logic via gating.
        self.news_gate = nn.Linear(2, lnn_core.input_size) 
        
        # Initialize weights
        nn.init.xavier_uniform_(self.bridge_layer.weight)
        nn.init.zeros_(self.bridge_layer.bias)
        nn.init.xavier_uniform_(self.news_gate.weight)
        nn.init.zeros_(self.news_gate.bias)

    def forward(self, market_data: torch.Tensor, lfm_context_tensor: torch.Tensor, news_vector: torch.Tensor):
        """
        Args:
            market_data: [Batch, Seq, Feat] - The raw market ticks/candles.
            lfm_context_tensor: [Batch, LFM_Dim] - The 'Strategy Vector' from the LFM.
            news_vector: [Batch, 2] - Sentiment and Impact from Sentinel.
        """
        # 1. Modulation (Liquid to Liquid)
        # Project Strategy Vector to LNN Input Dimension to bias the ODE
        strategy_bias = self.bridge_layer(lfm_context_tensor) # [Batch, Hidden]
        
        # 2. News Modulation
        news_bias = self.news_gate(news_vector) # [Batch, Hidden]
        
        # Combine Modulations
        # We sum them up. Strategy directs the flow, News alters the gain.
        total_modulation = strategy_bias + news_bias
        
        # 3. Execution (CfC Cell with modulation)
        # The LNN is "primed" by the strategy of the LFM.
        # ModulatedIARA.forward accepts 'external_modulation'
        tactical_decision = self.lnn(market_data, external_modulation=total_modulation)
        
        # 4. Brake Logic (Safety)
        # If News Impact is high (close to 1.0) and Sentiment is Negative, we might want to suppress output.
        # v8.5 Logic: We let the LNN decide based on the modulation, but we can enforce a hard brake 
        # inside the trading engine if needed. Here we return the raw decision (logits/values).
        
        return tactical_decision
