"""
IARA API — Brain Lab Routes
Training orchestration and model management.
"""

import logging
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import desc

from src.database.core import get_db
from src.database.models import Trade, Signal, ModelVersion
from src.train.continuous_learner import ContinuousLearner
import threading
import os
import shutil
import subprocess
from pydantic import BaseModel
from datetime import datetime

router = APIRouter(prefix="/brain", tags=["Brain Lab"])
logger = logging.getLogger("api.brain")

# In-memory training status
training_state = {
    "is_training": False,
    "last_loss": None,
    "last_run": None,
    "message": "Idle",
    "progress": 0,  # 0 to 100
    "logs": [],     # Recent training logs
    "stop_requested": False,
}

# In-memory data fetching status
data_state = {
    "is_fetching": False,
    "message": "Idle",
    "logs": []
}

class TrainingProtocol(BaseModel):
    mode: str = "finetune" # "scratch", "finetune"
    base_model: Optional[str] = "active_model.pth"
    symbol: str = "BTC/USD"
    batch_size: int = 256
    epochs: int = 5
    learning_rate: float = 0.0001
    recent_months: int = 6
    workers: int = 8
    data_source: str = "journal" # "journal", "historical"
    dataset_name: Optional[str] = None

class FetchRequest(BaseModel):
    symbol: str = "BTC/USD"
    days: int = 30
    timeframe: str = "1Min"

def run_training_background(protocol: Dict[str, Any]):
    global training_state
    try:
        training_state["is_training"] = True
        training_state["progress"] = 0
        training_state["logs"] = []
        training_state["stop_requested"] = False
        training_state["message"] = f"Initializing {protocol['mode']} protocol..."
        
        def log_callback(msg: str):
            training_state["logs"].append(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")
            if len(training_state["logs"]) > 50:
                training_state["logs"].pop(0)
            training_state["message"] = msg
            
            # Parse progress % and loss from callback messages
            import re
            progress_match = re.search(r'(\d+)%', msg)
            if progress_match:
                training_state["progress"] = int(progress_match.group(1))
            loss_match = re.search(r'Loss:\s*([\d.]+)', msg)
            if loss_match:
                training_state["last_loss"] = float(loss_match.group(1))

        checkpoint_name = protocol.get("base_model")
        checkpoint_path = f"logs/models/{checkpoint_name}" if checkpoint_name else None
        
        learner = ContinuousLearner(checkpoint_path=checkpoint_path)
        log_callback(f"Compute device detected: {learner.device}")
        
        # Override learner params from protocol
        learner.epochs = protocol["epochs"]
        learner.batch_size = protocol["batch_size"]
        learner.learning_rate = protocol["learning_rate"]
        
        # Stop check callable for graceful interruption
        def should_stop():
            return training_state.get("stop_requested", False)
        
        # Handle Data Source
        if protocol.get("data_source") == "historical" and protocol.get("dataset_name"):
            log_callback(f"Switching to Historical Dataset: {protocol['dataset_name']}")
            if hasattr(learner, 'train_on_dataset'):
                learner.train_on_dataset(protocol['dataset_name'], callback=log_callback, stop_check=should_stop)
            else:
                log_callback("Error: train_on_dataset not implemented. Falling back.")
                learner.train_and_evaluate(callback=log_callback, stop_check=should_stop)
        else:
            learner.train_and_evaluate(callback=log_callback, stop_check=should_stop)
        
        if training_state["stop_requested"]:
            training_state["message"] = "⛔ Training stopped by user. Best model saved."
        else:
            training_state["message"] = "Training completed successfully"
        training_state["progress"] = 100
    except Exception as e:
        logger.error(f"Training failed: {e}")
        training_state["message"] = f"Error: {str(e)}"
    finally:
        training_state["is_training"] = False

def run_data_fetch_background(request: Dict[str, Any]):
    global data_state
    try:
        data_state["is_fetching"] = True
        data_state["logs"] = []
        
        def log_data(msg: str):
            data_state["logs"].append(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")
            if len(data_state["logs"]) > 50:
                data_state["logs"].pop(0)
            data_state["message"] = msg

        symbol = request["symbol"]
        days = request["days"]
        tf = request["timeframe"]
        
        log_data(f"🚀 Starting data fetch for {symbol}...")
        
        cmd_fetch = ["python3", "src/data/crypto_data_fetcher.py", "--symbol", symbol, "--days", str(days), "--timeframe", tf]
        result_fetch = subprocess.run(cmd_fetch, capture_output=True, text=True)
        for line in result_fetch.stdout.splitlines():
            if line.strip(): log_data(line)
        if result_fetch.returncode != 0:
            log_data(f"❌ Fetcher failed: {result_fetch.stderr}")
            return

        norm_symbol = symbol.replace("/", "")
        cmd_factory = ["python3", "src/data/feature_factory.py", "--symbol", norm_symbol, "--market-type", "crypto"]
        result_factory = subprocess.run(cmd_factory, capture_output=True, text=True)
        for line in result_factory.stdout.splitlines():
            if line.strip(): log_data(line)
        if result_factory.returncode != 0:
            log_data(f"❌ Factory failed: {result_factory.stderr}")
        else:
            log_data(f"✅ Data processing complete for {symbol}!")
            
    except Exception as e:
        logger.error(f"Data fetch failed: {e}")
        data_state["message"] = f"Error: {str(e)}"
    finally:
        data_state["is_fetching"] = False

@router.post("/train")
async def trigger_training(protocol: TrainingProtocol):
    global training_state
    if training_state["is_training"]:
        return {"message": "Training already in progress", "status": training_state}
    protocol_dict = protocol.dict()
    threading.Thread(target=run_training_background, args=(protocol_dict,)).start()
    return {"message": "Training started", "status": training_state}

@router.post("/stop")
async def stop_training():
    global training_state
    if not training_state["is_training"]:
        raise HTTPException(status_code=400, detail="No training in progress")
    training_state["stop_requested"] = True
    training_state["message"] = "⏳ Stop requested. Finishing current batch and saving best model..."
    return {"message": "Stop signal sent. Training will halt after current batch."}

@router.post("/data/fetch")
async def trigger_data_fetch(request: FetchRequest):
    global data_state
    if data_state["is_fetching"]:
        return {"message": "Data fetch already in progress", "status": data_state}
    request_dict = request.dict()
    threading.Thread(target=run_data_fetch_background, args=(request_dict,)).start()
    return {"message": "Data fetch initiated", "status": data_state}

@router.get("/data/status")
async def get_data_status():
    return data_state

@router.get("/data/inventory")
async def list_datasets():
    cache_dir = "data/cache"
    if not os.path.exists(cache_dir): return []
    datasets = []
    for symbol_dir in os.listdir(cache_dir):
        full_path = os.path.join(cache_dir, symbol_dir)
        if os.path.isdir(full_path):
            shard_dir = os.path.join(full_path, "train_shards")
            if os.path.exists(shard_dir):
                shards = [f for f in os.listdir(shard_dir) if f.endswith(".pt")]
                datasets.append({"symbol": symbol_dir, "shards": len(shards), "path": full_path})
    return datasets

@router.delete("/data/{symbol}")
async def delete_dataset(symbol: str):
    cache_dir = f"data/cache/{symbol}"
    raw_dir = f"data/raw/{symbol}"
    deleted = []
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
        deleted.append(cache_dir)
    if os.path.exists(raw_dir):
        shutil.rmtree(raw_dir)
        deleted.append(raw_dir)
    if not deleted: raise HTTPException(status_code=404, detail="Dataset not found")
    return {"message": f"Dataset {symbol} deleted", "paths": deleted}

@router.get("/models")
async def list_models():
    model_dir = "logs/models"
    if not os.path.exists(model_dir): return []
    models = []
    for file in os.listdir(model_dir):
        if file.endswith(".pth"):
            full_path = os.path.join(model_dir, file)
            stats = os.stat(full_path)
            models.append({
                "name": file,
                "size": f"{stats.st_size / (1024*1024):.2f} MB",
                "last_modified": datetime.fromtimestamp(stats.st_mtime).isoformat()
            })
    return models

@router.delete("/models/{model_name}")
async def delete_model(model_name: str):
    if model_name == "active_model.pth":
        raise HTTPException(status_code=400, detail="Cannot delete the active model directly")
    path = f"logs/models/{model_name}"
    if os.path.exists(path):
        os.remove(path)
        return {"message": f"Model {model_name} deleted"}
    raise HTTPException(status_code=404, detail="Model not found")

@router.post("/models/activate")
async def activate_model(request: Dict[str, str]):
    model_name = request.get("name")
    if not model_name: raise HTTPException(status_code=400, detail="Model name required")
    src = f"logs/models/{model_name}"
    dst = "logs/models/active_model.pth"
    if not os.path.exists(src): raise HTTPException(status_code=404, detail="Source model not found")
    shutil.copy2(src, dst)
    return {"message": f"Model {model_name} is now active"}

@router.get("/status")
async def get_training_status():
    return training_state

@router.get("/trades")
async def list_trades(limit: int = 50, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Trade).order_by(desc(Trade.timestamp)).limit(limit))
    return result.scalars().all()

@router.get("/signals")
async def list_signals(limit: int = 20, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Signal).order_by(desc(Signal.timestamp)).limit(limit))
    return result.scalars().all()
