import torch
import torch.nn as nn
import torch.optim as optim
import time
import argparse
import sys
import os
import json
import csv
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.metrics import roc_auc_score

# Import Trinity Architecture
from src.brain.models.liquid_v8 import LiquidConfig, ModulatedIARA
from src.brain.models.trinity_liquid import TrinityLiquidBrain

# Constants
LOG_DIR = "logs"
SNAPSHOT_DIR = os.path.join(LOG_DIR, "snapshots")
MODELS_DIR = os.path.join(LOG_DIR, "models")
METRICS_FILE = os.path.join(LOG_DIR, "metrics.csv")
TELEMETRY_FILE = os.path.join(LOG_DIR, "telemetry_latest.json")
LEADERBOARD_FILE = os.path.join(LOG_DIR, "leaderboard.json")

# Global Data Cache
MARKET_DATA_CACHE = None

def load_and_cache_data():
    global MARKET_DATA_CACHE
    if MARKET_DATA_CACHE is None:
        # ABSOLUTE PATH for Docker
        csv_path = "/app/data/raw/EURUSD_synthetic.csv"
        
        if not os.path.exists(csv_path):
            # Also try relative path for local debug
            csv_path = "data/raw/EURUSD_synthetic.csv"
            
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"🔥 CRITICAL: Market Data NOT FOUND at {csv_path}. \n"
                                    "Please ensure data/raw/EURUSD_synthetic.csv exists and is mounted.\n"
                                    "Training CANNOT proceed without real data.")
            
        print(f"📂 Loading Market Data from {csv_path}...")
        try:
            df = pd.read_csv(csv_path)
            # Normalize column names
            df.columns = [c.lower().strip() for c in df.columns]
            
            if "date" in df.columns:
                df["timestamp"] = pd.to_datetime(df["date"])
            elif "time" in df.columns:
                df["timestamp"] = pd.to_datetime(df["time"])
            else:
                df["timestamp"] = pd.date_range(end=datetime.now(), periods=len(df), freq="15min")
                
            df = df.sort_values("timestamp").reset_index(drop=True)
            
            # Ensure required columns
            req_cols = ["open", "high", "low", "close"]
            for c in req_cols:
                if c not in df.columns:
                     raise ValueError(f"Missing column '{c}' in CSV.")
            
            # ----------------------------------------------------
            # FEATURE ENGINEERING (Give the IA "Glasses")
            # ----------------------------------------------------
            print("🧠 Pre-calculating Technical Indicators...")
            
            # 1. RSI (14)
            delta = df["close"].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / (loss + 1e-6)
            df["rsi"] = 100 - (100 / (1 + rs))
            df["rsi"] = df["rsi"].fillna(50) / 100.0 # Normalize 0-1
            
            # 2. MACD (12, 26, 9)
            ema12 = df["close"].ewm(span=12, adjust=False).mean()
            ema26 = df["close"].ewm(span=26, adjust=False).mean()
            df["macd"] = ema12 - ema26
            df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
            # Normalize robustness
            df["macd"] = (df["macd"] - df["macd"].mean()) / (df["macd"].std() + 1e-6)
            
            # 3. Bollinger Bands (20)
            sma20 = df["close"].rolling(window=20).mean()
            std20 = df["close"].rolling(window=20).std()
            df["bb_upper"] = (df["close"] - sma20) / (2 * std20 + 1e-6) # Z-Score from middle
            df["bb_upper"] = df["bb_upper"].fillna(0)
            
            # 4. Returns
            df["log_ret"] = np.log(df["close"] / df["close"].shift(1))
            df["log_ret"] = df["log_ret"].fillna(0)
            
            # 5. Volatility
            df["volatility"] = df["log_ret"].rolling(window=20).std().fillna(0)
            
            # Drop NaN from warmup
            df = df.iloc[30:].reset_index(drop=True)
                     
            MARKET_DATA_CACHE = df
        except Exception as e:
            raise RuntimeError(f"Failed to read CSV: {e}")
            
    return MARKET_DATA_CACHE

from torch.utils.data import TensorDataset, DataLoader

class CryptoDataset(TensorDataset):
    def __init__(self, x_path, y_path=None):
        self.x = torch.load(x_path)
        if isinstance(self.x, dict):
            # Handle potential dict wrapping
            self.y = self.x.get('y')
            self.x = self.x.get('X')
        else:
            self.y = None
            
        if self.y is None and y_path:
             self.y = torch.load(y_path)

        # Move to CPU first to avoid CUDA OOM during init if large
        self.x = self.x.float()
        self.y = self.y.float()
        
    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

def get_dataloaders(batch_size):
    # Load Cache
    print("📂 Loading Pre-computed Tensors from data/cache/...")
    train_ds = CryptoDataset("data/cache/train.pt")
    val_ds = CryptoDataset("data/cache/val.pt")
    
    print(f"   Train Size: {len(train_ds)} | Val Size: {len(val_ds)}")
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=True)
    
    return train_loader, val_loader

# Replaces generate_market_scenario for Training, 
# BUT we keep a helper for Visualization relying on the CSV.
def get_viz_scenario(seq_len=256):
    """
    Returns a single coherent slice from CSV for Dashboard visualization.
    Matches the input shape (18) by adding dummy features if needed or re-calculating.
    """
    df = load_and_cache_data()
    max_idx = len(df) - seq_len - 10
    start_idx = np.random.randint(0, max_idx)
    slice_df = df.iloc[start_idx : start_idx + seq_len].copy()
    
    # We need to recreate the 18-dim features to match the model
    # For viz purposes, we will try to match the V6 schema roughly or just pad
    # V6 (18 dims) usually: 
    # [NormClose, NormHigh, NormLow, RSI, MACD, Signal, BB_Up, BB_Low, CCI, ADX, STOCH, WILLR, ... ]
    # Since we can't easily reproduce exact V6 features on the fly without the factory code,
    # We will assume dimensions 0-5 are main price/techs and rest are advanced.
    
    # Hack: We generate the 6 basic we know, and pad the rest with zeros for the visualization inference.
    # The 'Brain' might give garbage expert curves on this dummy data, but the Hidden State heatmap will work.
    
    closes = slice_df["close"].values
    norm_close = (closes - closes.mean()) / (closes.std() + 1e-6)
    
    # We rely on what load_and_cache_data computes (RSI, MACD, etc)
    # Check lines 65+ in load_and_cache_data
    
    rsi = slice_df["rsi"].values
    macd = slice_df["macd"].values
    bb_pos = slice_df["bb_upper"].values
    log_ret = slice_df["log_ret"].values
    volat = slice_df["volatility"].values
    
    # 6 Real features
    base_feats = np.column_stack([norm_close, rsi, macd, bb_pos, log_ret, volat])
    
    # Pad to 18
    padding = np.zeros((seq_len, 12))
    full_feats = np.column_stack([base_feats, padding])
    
    inputs = torch.FloatTensor(full_feats).unsqueeze(0) # Batch 1
    llm_ctx = torch.randn(1, 32)
    news_vec = torch.zeros(1, 2)
    
    return inputs, slice_df, llm_ctx, news_vec

class ForensicRecorder:
    def __init__(self, model: TrinityLiquidBrain):
        self.model = model
        self.captured_data = {"feature_gates": []}
        self._register_hooks()

    def _register_hooks(self):
        def hook_feature_gate(module, input, output):
            self.captured_data["feature_gates"] = output.detach().cpu().numpy()
        self.model.lnn.feature_attention.gate[-1].register_forward_hook(hook_feature_gate)

    def capture_snapshot(self, inputs, ohlc_df, llm_ctx, news_vec, epoch):
        self.model.eval()
        with torch.no_grad():
            hidden_states = []
            def hook_cfc(module, input, output):
                hidden_states.append(output.detach().cpu().numpy())
            
            handle_cfc = self.model.lnn.cfc_layers[-1].register_forward_hook(hook_cfc)
            _ = self.model(inputs, llm_ctx, news_vec)
            
            feat_imp = self.captured_data["feature_gates"][0].mean(axis=0).tolist()
            handle_cfc.remove()
            
            hidden_stack = np.stack(hidden_states, axis=1)
            h_state = hidden_stack[0] # Sample 0
            
            h_tensor = torch.tensor(h_state).to(inputs.device) 
            long_curve = self.model.lnn.long_expert(h_tensor).detach().cpu().numpy().flatten().tolist()
            short_curve = self.model.lnn.short_expert(h_tensor).detach().cpu().numpy().flatten().tolist()
            
            snapshot = {
                "epoch": epoch,
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "ohlc": {
                   "open": ohlc_df["open"].tolist(),
                   "high": ohlc_df["high"].tolist(),
                   "low": ohlc_df["low"].tolist(),
                   "close": ohlc_df["close"].tolist(),
                   "time": ohlc_df["timestamp"].dt.strftime("%H:%M").tolist()
                },
                "feature_importance": feat_imp,
                "expert_conviction": {
                    "long": long_curve,
                    "short": short_curve
                },
                "hidden_activity": h_state.T.tolist(),
                "simulation_metadata": {"scenario": "REAL MARKET (EURUSD)", "volatility": "Real"}
            }
            
            path = os.path.join(SNAPSHOT_DIR, f"epoch_{epoch}.json")
            with open(path, "w") as f:
                json.dump(snapshot, f)
            
        self.model.train()

from torch.utils.data import WeightedRandomSampler

class SelectiveMemory:
    """
    v6.11 Selective Memory (Curriculum Learning).
    Tracks simple vs difficult samples across epochs to focus training.
    """
    def __init__(self, num_samples, alpha=0.7):
        self.alpha = alpha # EMA factor
        self.sample_loss_ema = torch.ones(num_samples)
        self.chronic_counter = torch.zeros(num_samples, dtype=torch.int16) # Persistent errors
        self.mastered_counter = torch.zeros(num_samples, dtype=torch.int16) # Persistent correct
        self.noise_mask = torch.zeros(num_samples, dtype=torch.bool) # Impossible samples
        
        # History for noise detection (Rolling window of 10 epochs)
        self.epoch_losses_history = torch.zeros(num_samples, 10)

    def analyze(self, model, dataset, device, batch_size=2048):
        """
        Deep Thought: Runs a fast evaluation pass to update sample difficulty.
        Returns: WeightedRandomSampler for the next epoch.
        """
        print("🧠 Deep Thought: Analyzing difficulty spectrum (Selective Memory)...")
        model.eval()
        losses = []
        accuracies = []
        
        # Fast Eval Loader (Sequential)
        eval_loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, 
            num_workers=0, pin_memory=True
        )
        
        criterion = nn.BCEWithLogitsLoss(reduction='none')
        
        total_batches = len(eval_loader)
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                for i, (inputs, targets) in enumerate(eval_loader):
                    if i % 10 == 0:
                        print(f"   [Deep Thought] Scanned {i}/{total_batches} batches...", end='\r')
                        
                    inputs = inputs.to(device)
                    targets = targets.float().to(device)
                    llm_ctx = torch.zeros(inputs.size(0), 32).to(device)
                    news_vec = torch.zeros(inputs.size(0), 2).to(device)
                    
                    outputs = model(inputs, llm_ctx, news_vec)
                    outputs = outputs.squeeze(-1)
                    
                    # 1. Per-Sample Loss
                    loss = criterion(outputs, targets)
                    losses.append(loss.cpu())
                    
                    # 2. Per-Sample Correctness (Mastery)
                    probs = torch.sigmoid(outputs)
                    preds = (probs > 0.5).float()
                    hard_targets = (targets > 0.5).float()
                    correct = (preds == hard_targets).float()
                    
                    # High Confidence?
                    confident = ((probs > 0.7) | (probs < 0.3)).float()
                    accuracies.append(correct * confident)
        
        print(f"   [Deep Thought] Processing stats for {len(losses) * batch_size} samples...") 
        current_losses = torch.cat(losses)
        current_mastered = torch.cat(accuracies)
        
        # --- UPDATE STATE ---
        # 1. EMA Loss
        if len(self.sample_loss_ema) != len(current_losses):
             # Resize if dataset changed (robustness)
             self.sample_loss_ema = torch.ones_like(current_losses)
             
        self.sample_loss_ema = (self.alpha * self.sample_loss_ema) + ((1 - self.alpha) * current_losses)
        
        # 2. Chronic (High Loss > 0.9)
        high_loss = current_losses > 0.9
        self.chronic_counter[high_loss] += 1
        self.chronic_counter[~high_loss] = 0
        
        # 3. Mastered (Confident Correct)
        mastered_mask = current_mastered > 0.5
        self.mastered_counter[mastered_mask] += 1
        self.mastered_counter[~mastered_mask] = 0
        
        # 4. Noise (Impossible > 1.2 for 10 epochs)
        self.epoch_losses_history = torch.roll(self.epoch_losses_history, -1, dims=1)
        self.epoch_losses_history[:, -1] = current_losses
        noise_candidates = (self.epoch_losses_history > 1.2).all(dim=1)
        self.noise_mask = self.noise_mask | noise_candidates
        
        # --- CALCULATE WEIGHTS ---
        weights = (self.sample_loss_ema + 0.05) ** 2
        
        # Boost Chronic
        weights[self.chronic_counter >= 3] *= 3.0
        
        # Boost Arrogant (Very high loss = Wrong + Confident)
        weights[current_losses > 1.5] *= 2.0
        
        # Suppress Mastered
        weights[self.mastered_counter >= 5] *= 0.05
        
        # Kill Noise
        weights[self.noise_mask] = 0.0
        
        # Stats
        n_chronic = (self.chronic_counter >= 3).sum().item()
        n_mastered = (self.mastered_counter >= 5).sum().item()
        n_noise = self.noise_mask.sum().item()
        
        print(f"   📊 Stats: {n_chronic} Chronic | {n_mastered} Mastered | {n_noise} Noise (Dropped)")
        torch.cuda.empty_cache()
        
        # Sampler
        sampler = WeightedRandomSampler(
            weights=weights, num_samples=len(weights), replacement=True
        )
        return sampler

class AsymmetricFocalLoss(nn.Module):
    """
    v6.7 Profit-Weighted Asymmetric Focal Loss.
    Ported from trainer_liquid.py
    """
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Targets must be 0-1 range.
        bce_loss = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.exp(-bce_loss) # prob of correct class
        
        # Focal term: (1 - pt)^gamma
        focal_term = (1 - pt) ** self.gamma
        
        # Alpha balancing: Higher weight for positive class (1)
        # targets is 0 or 1.
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        loss = alpha_t * focal_term * bce_loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

class Leaderboard:
    def __init__(self):
        self.history = []
        if os.path.exists(LEADERBOARD_FILE):
            try:
                with open(LEADERBOARD_FILE, "r") as f:
                    self.history = json.load(f)
            except: pass
            
    def update(self, epoch, loss, accuracy, auc, model_state):
        entry = {
            "epoch": epoch,
            "loss": loss,
            "accuracy": accuracy,
            "auc": auc,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        self.history.append(entry)
        self.history = sorted(self.history, key=lambda x: x["loss"])[:5] 
        
        with open(LEADERBOARD_FILE, "w") as f:
            json.dump(self.history, f, indent=2)
            
        if self.history[0]["epoch"] == epoch:
            print(f"🏆 New Best Model! Loss: {loss:.4f} | Acc: {accuracy:.2%} | AUC: {auc:.4f}")
            torch.save(model_state, os.path.join(MODELS_DIR, "best_model.pth"))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    # CRITICAL: Clean old snapshots
    if os.path.exists(SNAPSHOT_DIR):
        for f in os.listdir(SNAPSHOT_DIR):
            if f.endswith(".json"):
                 try: os.remove(os.path.join(SNAPSHOT_DIR, f))
                 except: pass
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    with open(METRICS_FILE, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["step", "epoch", "loss", "accuracy", "auc", "lr"])

    # 1. Init Model
    if args.cpu:
        device = "cpu"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        print("⚠️ CUDA not available. Falling back to CPU.")
        device = "cpu"

    print(f"🚀 Trinity Loop Initialized on {device.upper()}")
    if device == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    # Check Data Integrity BEFORE starting model
    try:
        load_and_cache_data() 
    except Exception as e:
        print(f"❌ DATA ERROR: {e}")
        sys.exit(1)

    # High Seq Len for Pattern Recognition
    # V8 Hybrid: Consuming V6 processed tensors which have 18 features
    INPUT_DIM = 18 
    liq_cfg = LiquidConfig(input_dim=INPUT_DIM, hidden_dim=128, max_seq_len=256)
    lnn = ModulatedIARA(liq_cfg)
    # V8.5 Pure Liquid: LFM Context is 4096 dim
    model = TrinityLiquidBrain(lnn, lfm_hidden_dim=4096).to(device)
    
    # ----------------------------------------------------
    # RESUME FROM CHAMPION (Transfer Learning)
    # ----------------------------------------------------
    best_model_path = os.path.join(MODELS_DIR, "best_model.pth")
    if os.path.exists(best_model_path):
        print(f"🏆 FOUND CHAMPION MODEL. Resuming training from: {best_model_path}")
        try:
            # Handle dimension mismatch if upgrading from 16 -> 18
            state_dict = torch.load(best_model_path, map_location=device)
            current_dict = model.state_dict()
            
            # Simple check: If first layer weight shape mismatches, skip it
            # But here we assume we are starting fresh or resuming compatible
            model.load_state_dict(state_dict, strict=False) 
            print("✅ Weights loaded (Relaxed/Strict)! Starting with high IQ base.")
        except Exception as e:
            print(f"⚠️ Failed to load champion model: {e}. Starting fresh.")
    else:
        print("🌱 No previous champion found. Starting fresh training.")

    recorder = ForensicRecorder(model)
    leaderboard = Leaderboard()
    
    # v6.9 Logic: AdamW + Asymmetric Focal Loss
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4) # Added weight_decay
    # V6 Exact Match: Gentle decay
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=15)
    criterion = AsymmetricFocalLoss(alpha=0.75, gamma=2.0)
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))
    
    # LOAD DATASETS
    # We access the underlying dataset directly for analysis
    print("📂 Loading Pre-computed Tensors from data/cache/...")
    train_ds = CryptoDataset("data/cache/train.pt")
    val_ds = CryptoDataset("data/cache/val.pt")
    print(f"   Train Size: {len(train_ds)} | Val Size: {len(val_ds)}")
    
    # Init Selective Memory
    memory = SelectiveMemory(len(train_ds))
    
    # Default Loader (Shuffle) for Epoch 1
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=True)
    
    steps_per_epoch = len(train_loader)
    print("------------------------------------------------")
    print(f"🔄 Training Loop Start: {steps_per_epoch} steps per epoch.")
    
    for epoch in range(1, args.epochs + 1):
        # ----------------------------------------
        # SELECTIVE MEMORY (Curriculum)
        # ----------------------------------------
        # From Epoch 2 onwards, we refine the curriculum based on previous performance
        if epoch > 1:
            sampler = memory.analyze(model, train_ds, device)
            train_loader = DataLoader(
                train_ds, 
                batch_size=args.batch_size, 
                sampler=sampler, # Use Sampler -> shuffle must be False
                shuffle=False, 
                drop_last=True
            )
        
        epoch_loss = 0
        epoch_acc = 0
        epoch_auc = 0
        
        all_targets = []
        all_probs = []
        
        model.train()
        
        # Iterate DataLoader
        for step, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device)
            targets = targets.float().to(device)
            
            # STABILITY FIX: Zero out LLM context instead of random noise
            # We want the LNN to learn market physics first.
            llm_ctx = torch.zeros(inputs.size(0), 4096).to(device)
            news_vec = torch.zeros(inputs.size(0), 2).to(device)
            
            optimizer.zero_grad()
            
            # AMP Forward
            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                outputs = model(inputs, llm_ctx, news_vec)
                outputs = outputs.squeeze(-1)
                
                # Loss: Asymmetric Focal Loss (Classification Logic)
                loss = criterion(outputs, targets)
            
            # AMP Backward
            scaler.scale(loss).backward()
            
            # Gradient Clipping (Unscale first)
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            
            # Metrics
            probs = torch.sigmoid(outputs)
            
            # Binary Accuracy (V6 Style for comparison)
            # Treat > 0.5 as Buy, <= 0.5 as Sell/Hold (Downside bias)
            preds = (probs > 0.5).float()
            
            # For Accuracy calc, we assume Hard targets (0 or 1). 
            # If target is 0.5, it counts as 0 for binary comparison
            hard_targets = (targets > 0.5).float() 
            
            acc = (preds == hard_targets).float().mean().item()
            epoch_acc += acc
            
            # Collect for AUC
            all_targets.extend(hard_targets.detach().cpu().numpy())
            all_probs.extend(probs.detach().cpu().numpy())
            
            # Telemetry (Sampled to avoid CSV spam)
            if step % 10 == 0:
                global_step = (epoch - 1) * steps_per_epoch + step
                current_lr = optimizer.param_groups[0]['lr']
                
                # Running AUC estimate (chunk)
                try:
                    batch_auc = roc_auc_score(hard_targets.detach().cpu().numpy(), probs.detach().cpu().numpy())
                except:
                    batch_auc = 0.5
                    
                with open(METRICS_FILE, "a", newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([global_step, epoch, loss.item(), acc, batch_auc, current_lr])
            
        avg_loss = epoch_loss / steps_per_epoch
        avg_acc = epoch_acc / steps_per_epoch
        
        try:
            avg_auc = roc_auc_score(all_targets, all_probs)
        except:
            avg_auc = 0.5
        
        # Adjust Learning Rate if plateau
        scheduler.step(avg_loss)
        
        print(f"Epoch {epoch}/{args.epochs} | Loss: {avg_loss:.4f} | Acc: {avg_acc:.2%} | AUC: {avg_auc:.4f}")
        
        # ----------------------------------------
        # SNAPSHOT (Visualization)
        # We need to construct a synthetic one because the DataLoader doesn't give OHLC
        # ----------------------------------------
        try:
             # Grab a random scenario from CSV for visualization
             print("   📸 Capturing Forensic Snapshot...")
             viz_inputs, viz_ohlc, v_ctx, v_news = get_viz_scenario()
             viz_inputs = viz_inputs.to(device)
             v_ctx = v_ctx.to(device)
             v_news = v_news.to(device)
             recorder.capture_snapshot(viz_inputs, viz_ohlc, v_ctx, v_news, epoch)
             print("   ✅ Snapshot Saved.")
        except Exception as e:
             print(f"⚠️ Snapshot Warning: {e}")
        
        print("   🏆 Updating Leaderboard...") 
        leaderboard.update(epoch, avg_loss, avg_acc, avg_auc, model.state_dict())
        print("   ✅ Leaderboard Updated.")
        
        with open(TELEMETRY_FILE, "w") as f:
            json.dump({
                "current_epoch": epoch,
                "total_epochs": args.epochs,
                "status": "TRAINING",
                "last_loss": avg_loss,
                "device": device
            }, f)
            
    with open(TELEMETRY_FILE, "w") as f:
        json.dump({"status": "COMPLETED"}, f)
    print("✅ Training Complete.")
 
if __name__ == "__main__":
    main()
