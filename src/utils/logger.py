
import json
import time
import logging
from pathlib import Path
from typing import Dict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# LOGS_DIR = Path(__file__).parent.parent.parent / "logs"
# Use absolute path relative to project root assuming src/utils location
LOGS_DIR = Path("logs")

class TelemetryLogger:
    """Real-time JSONL logger for training dynamics."""
    def __init__(self, log_dir: Path = LOGS_DIR):
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.log_path = log_dir / f"training_history_{timestamp}.jsonl"
        logger.info(f"Telemetry logging to: {self.log_path}")
        
    def log(self, entry: Dict):
        """Append a log entry to the JSONL file."""
        try:
            with open(self.log_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logger.error(f"Failed to write telemetry: {e}")
