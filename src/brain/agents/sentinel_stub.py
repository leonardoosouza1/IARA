from typing import Dict, Any

class SentinelStub:
    """
    A placeholder for the Sentinel (Watchdog) agent.
    
    Role:
    - Phase 1: Passive / Stubbed. Returns neutral scores to validate core loop.
    - Phase 2: Will be replaced by FinBERT implementation.
    """
    
    def __init__(self):
        print("🕊️ Sentinel Stub Initialized (Passive Mode)")

    def is_safe(self) -> bool:
        """
        Check if trading is allowed.
        Currently returns True (Passive).
        Future: Output of FinBERT or Risk Rules.
        """
        return True

    def analyze(self, text: str) -> Dict[str, float]:
        """
        Mimics the future FinBERT output.
        
        Args:
            text (str): Raw news headline or text.
            
        Returns:
            dict: {
                'sentiment': 0.0,  # Neutral (-1 to 1)
                'impact': 0.0      # Zero Impact (0 to 1) -> No Emergency Brake
            }
        """
        return {
            "sentiment": 0.0,
            "impact": 0.0
        }
