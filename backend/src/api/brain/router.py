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
from fastapi.responses import PlainTextResponse

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
    "last_acc": None,
    "last_prec": None,
    "last_auc": None,
    "last_f1": None,
    "pressure": 1.0,
    "last_run": None,
    "message": "Idle",
    "progress": 0,  # 0 to 100
    "trauma_count": 0,
    "logs": [],     # Recent training logs
    "all_logs": [], # Full history for export
    "stop_requested": False,
    "current_epoch": 1,
    "micro_state": {},  # 🧠 Phase 20: Real-time internal brain state for "Opened Brain" dashboard
    "epoch_history": [],  # 📊 Phase 14 v2: Epoch comparison data for frontend dashboard
}

# In-memory data fetching status
data_state = {
    "is_fetching": False,
    "message": "Idle",
    "logs": []
}

backtest_state = {
    "is_running": False,
    "dataset": "",
    "threshold": 0.5,
    "message": "Idle",
    "logs": [],
    "results": None
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
    save_as: Optional[str] = "active_model.pth"

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
        training_state["all_logs"] = []
        training_state["history"] = []
        training_state["stop_requested"] = False
        training_state["micro_state"] = {}
        training_state["epoch_history"] = []
        training_state["message"] = f"Initializing {protocol['mode']} protocol..."
        
        def log_callback(msg: str):
            log_str = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
            training_state["all_logs"].append(log_str)
            training_state["logs"].append(log_str)
            if len(training_state["logs"]) > 50:
                training_state["logs"].pop(0)
            training_state["message"] = msg
            
            # Parse progress % and loss from callback messages
            import re
            epoch_match = re.search(r'Epoch\s*(\d+)', msg)
            if epoch_match:
                training_state["current_epoch"] = int(epoch_match.group(1))
            
            progress_match = re.search(r'(\d+)%', msg)
            if progress_match:
                training_state["progress"] = int(progress_match.group(1))
            loss_match = re.search(r'Loss:\s*([\d.]+)', msg)
            if loss_match:
                training_state["last_loss"] = float(loss_match.group(1))
            
            acc_match = re.search(r'Acc:\s*([\d.]+)', msg)
            if acc_match:
                training_state["last_acc"] = float(acc_match.group(1))
                
            prec_match = re.search(r'Prec:\s*([\d.]+)', msg)
            if prec_match:
                training_state["last_prec"] = float(prec_match.group(1))

            auc_match = re.search(r'AUC:\s*([\d.]+)', msg)
            if auc_match:
                training_state["last_auc"] = float(auc_match.group(1))

            f1_match = re.search(r'F1:\s*([\d.]+)', msg)
            if f1_match:
                training_state["last_f1"] = float(f1_match.group(1))

            pressure_match = re.search(r'Pressure:\s*([\d.]+)', msg)
            if pressure_match:
                training_state["pressure"] = float(pressure_match.group(1))

            trauma_match = re.search(r'Retraining on (\d+) (?:severe mistakes|persistent memories)', msg)
            if trauma_match:
                training_state["trauma_count"] = int(trauma_match.group(1))
            
            # 📊 Phase 14 v2: Parse epoch comparison rows (E1, E2, etc.)
            epoch_row_match = re.search(r'E(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)\s+(\d+)\s+([\d.]+)', msg)
            if epoch_row_match:
                epoch_num = int(epoch_row_match.group(1))
                epoch_entry = {
                    "epoch": epoch_num,
                    "loss": float(epoch_row_match.group(2)),
                    "acc": float(epoch_row_match.group(3)),
                    "prec": float(epoch_row_match.group(4)),
                    "auc": float(epoch_row_match.group(5)),
                    "f1": float(epoch_row_match.group(6)),
                    "trauma_size": int(epoch_row_match.group(7)),
                    "mastered": int(epoch_row_match.group(8)),
                    "lr": float(epoch_row_match.group(9)),
                }
                # Replace existing entry for same epoch or append
                existing = [i for i, e in enumerate(training_state["epoch_history"]) if e["epoch"] == epoch_num]
                if existing:
                    training_state["epoch_history"][existing[0]] = epoch_entry
                else:
                    training_state["epoch_history"].append(epoch_entry)
            
            # Parse trend line
            trend_match = re.search(r'Tend[êe]ncia:\s*(.+)', msg)
            if trend_match:
                training_state["epoch_trend"] = trend_match.group(1).strip()

            # Store in history if it's a metric update
            if loss_match:
                epoch_val = int(epoch_match.group(1)) if epoch_match else training_state["current_epoch"]
                progress_val = int(progress_match.group(1)) if progress_match else 0
                
                training_state["history"].append({
                    "time": datetime.now().strftime('%H:%M:%S'),
                    "epoch": epoch_val,
                    "progress": progress_val,
                    "loss": training_state["last_loss"],
                    "acc": training_state["last_acc"] or 0,
                    "prec": training_state["last_prec"] or 0,
                    "auc": training_state.get("last_auc", 0),
                    "f1": training_state.get("last_f1", 0),
                    "pressure": training_state.get("pressure", 1.0)
                })
                # Keep last 50 points to prevent memory leak
                if len(training_state["history"]) > 50:
                    training_state["history"].pop(0)

        checkpoint_name = protocol.get("base_model")
        checkpoint_path = f"logs/models/{checkpoint_name}" if checkpoint_name else None
        
        save_name = protocol.get("save_as", "active_model.pth")
        if not save_name.endswith('.pth'):
            save_name += '.pth'
        save_path = f"logs/models/{save_name}"

        learner = ContinuousLearner(checkpoint_path=checkpoint_path)
        learner.checkpoint_path = save_path # Override where it saves
        
        # 🧠 Phase 20: Wire Micro-State Callback
        def micro_state_callback(state: dict):
            training_state["micro_state"] = state
        learner._micro_state_callback = micro_state_callback
        
        log_callback(f"Compute device detected: {learner.device}")
        log_callback(f"Target Save Location: {save_name}")
        
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

@router.get("/export")
async def export_logs():
    return PlainTextResponse("\n".join(training_state.get("all_logs", [])))

@router.post("/data/fetch")
async def trigger_data_fetch(request: FetchRequest):
    global data_state
    if data_state["is_fetching"]:
        return {"message": "Data Pipeline already active", "status": data_state}
    request_dict = request.dict()
    threading.Thread(target=run_data_fetch_background, args=(request_dict,)).start()
    return {"message": "Data fetch initiated", "status": data_state}

class BacktestRequest(BaseModel):
    dataset: str = "BTCUSD"
    threshold: float = 0.50

def run_backtest_background(request: Dict[str, Any]):
    global backtest_state
    try:
        backtest_state["is_running"] = True
        backtest_state["dataset"] = request["dataset"]
        backtest_state["threshold"] = request["threshold"]
        backtest_state["logs"] = []
        backtest_state["results"] = None
        
        def log_cb(msg: str):
            backtest_state["logs"].append(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")
            if len(backtest_state["logs"]) > 100:
                backtest_state["logs"].pop(0)
            backtest_state["message"] = msg

        from src.train.backtester import Backtester
        bt = Backtester(model_path="logs/models/active_model_v10.pth")
        
        results = bt.evaluate_shards(
            dataset_name=request["dataset"],
            sniper_threshold=request["threshold"],
            callback=log_cb
        )
        
        backtest_state["results"] = results
        backtest_state["message"] = "Backtest completed."
            
    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        backtest_state["message"] = f"Error: {str(e)}"
    finally:
        backtest_state["is_running"] = False

@router.post("/backtest")
async def trigger_backtest(request: BacktestRequest):
    global backtest_state
    if backtest_state["is_running"]:
        return {"message": "Backtest already running", "status": backtest_state}
        
    threading.Thread(target=run_backtest_background, args=(request.dict(),)).start()
    return {"message": "Backtest initiated", "status": backtest_state}

@router.get("/backtest/status")
async def get_backtest_status():
    return backtest_state

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
    import torch
    model_dir = "logs/models"
    if not os.path.exists(model_dir): return []
    models = []
    for file in os.listdir(model_dir):
        if file.endswith(".pth"):
            full_path = os.path.join(model_dir, file)
            stats = os.stat(full_path)
            
            # Introspect checkpoint for dimensions
            meta = {"input_dim": "?", "worker_hidden": "?", "coord_hidden": "?", "num_params": "?"}
            try:
                state = torch.load(full_path, map_location="cpu", weights_only=False)
                # input_dim from first worker input_proj
                w = state.get("swarm.workers.M5.input_proj.weight")
                if w is not None and hasattr(w, 'shape'):
                    meta["input_dim"] = int(w.shape[1])
                    meta["worker_hidden"] = int(w.shape[0])
                # coord_hidden from policy_head
                ph = state.get("coordinator.policy_head.weight")
                if ph is not None and hasattr(ph, 'shape'):
                    meta["coord_hidden"] = int(ph.shape[1])
                # Count params
                meta["num_params"] = sum(v.numel() for v in state.values() if hasattr(v, 'numel'))
            except Exception:
                pass
            
            models.append({
                "name": file,
                "size": f"{stats.st_size / (1024*1024):.2f} MB",
                "last_modified": datetime.fromtimestamp(stats.st_mtime).isoformat(),
                **meta
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
