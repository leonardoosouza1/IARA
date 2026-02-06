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

# Import Trinity Architecture
from src.brain.models.liquid_v8 import LiquidConfig, ModulatedIARA
from src.brain.models.trinity_v8 import TrinityBrain

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

def generate_market_scenario(batch_size, seq_len, input_dim):
    """
    Samples a REAL market sequence from EURUSD_synthetic.csv.
    """
    df = load_and_cache_data() # Will CRASH if no data (Desired behavior)
    
    # Sample a random start point
    max_idx = len(df) - seq_len - 11 # -10 for target lookahead
    start_idx = np.random.randint(0, max_idx)
    
    # Extract Slice
    # We pick one coherent sequence for the forensic snapshot (Batch 0)
    slice_df = df.iloc[start_idx : start_idx + seq_len].copy()
    
    # Extract Features
    closes = slice_df["close"].values
    highs = slice_df["high"].values
    lows = slice_df["low"].values
    
    # Extract Features (Pre-calculated)
    # Norm Close
    closes = slice_df["close"].values
    norm_close = (closes - closes.mean()) / (closes.std() + 1e-6)
    
    # Technicals
    rsi = slice_df["rsi"].values
    macd = slice_df["macd"].values
    bb_pos = slice_df["bb_upper"].values
    log_ret = slice_df["log_ret"].values
    volat = slice_df["volatility"].values
    
    # Stack Real Features (6 dimensions)
    # [NormClose, RSI, MACD, BB_Pos, LogRet, Volatility]
    real_features = np.column_stack([norm_close, rsi, macd, bb_pos, log_ret, volat])
    
    # Fill remaining input dims (16 total) with subtle noise
    # We want the model to learn to IGNORE these channels (dims 6-15)
    remaining_dim = input_dim - 6
    if remaining_dim > 0:
        noise_feats = np.random.normal(0, 0.01, size=(seq_len, remaining_dim))
        sample_features = np.column_stack([real_features, noise_feats])
    else:
        sample_features = real_features[:, :input_dim]
    
    # Batch Replication
    batch_features = np.tile(sample_features, (batch_size, 1, 1))
    
    # Augmentation: Add subtle noise to prevent overfitting
    batch_features += np.random.normal(0, 0.005, size=batch_features.shape)
    
    # Targets: Future Return Direction with "Neutral Zone" (Smart Targets)
    # Look ahead 10 candles
    future_close = df.iloc[start_idx + seq_len + 10]["close"]
    current_close = closes[-1]
    
    price_change = (future_close - current_close) / current_close
    
    # Threshold for "Actionable Signal" (e.g., 0.01% move)
    # If price doesn't move enough? It's a HOLD (0.5)
    # This teaches the AI to NOT hallucinate trends in flat markets.
    NOISE_THRESHOLD = 0.0001 
    
    target_val = 0.5 # Default HOLD
    
    if price_change > NOISE_THRESHOLD:
        target_val = 1.0 # BUY
    elif price_change < -NOISE_THRESHOLD:
        target_val = 0.0 # SELL
        
    targets = torch.full((batch_size,), target_val)
    inputs_tensor = torch.FloatTensor(batch_features)
    targets_tensor = targets
    
    # OHLC for Dashboard
    return inputs_tensor, targets_tensor, slice_df

class ForensicRecorder:
    def __init__(self, model: TrinityBrain):
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
            
    def update(self, epoch, loss, accuracy, model_state):
        entry = {
            "epoch": epoch,
            "loss": loss,
            "accuracy": accuracy,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        self.history.append(entry)
        self.history = sorted(self.history, key=lambda x: x["loss"])[:5] 
        
        with open(LEADERBOARD_FILE, "w") as f:
            json.dump(self.history, f, indent=2)
            
        if self.history[0]["epoch"] == epoch:
            print(f"🏆 New Best Model! Loss: {loss:.4f} | Acc: {accuracy:.2%}")
            torch.save(model_state, os.path.join(MODELS_DIR, "best_model.pth"))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=64)
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
        writer.writerow(["step", "epoch", "loss", "accuracy", "lr"])

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
    liq_cfg = LiquidConfig(input_dim=16, hidden_dim=128, max_seq_len=256)
    lnn = ModulatedIARA(liq_cfg)
    model = TrinityBrain(lnn, llm_hidden_dim=32).to(device)
    
    # ----------------------------------------------------
    # RESUME FROM CHAMPION (Transfer Learning)
    # ----------------------------------------------------
    best_model_path = os.path.join(MODELS_DIR, "best_model.pth")
    if os.path.exists(best_model_path):
        print(f"🏆 FOUND CHAMPION MODEL. Resuming training from: {best_model_path}")
        try:
            model.load_state_dict(torch.load(best_model_path, map_location=device))
            print("✅ Weights loaded successfully! Starting with high IQ base.")
        except Exception as e:
            print(f"⚠️ Failed to load champion model: {e}. Starting fresh.")
    else:
        print("🌱 No previous champion found. Starting fresh training.")

    recorder = ForensicRecorder(model)
    leaderboard = Leaderboard()
    
    # v6.9 Logic: AdamW + Asymmetric Focal Loss
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4) # Added weight_decay
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    criterion = AsymmetricFocalLoss(alpha=0.75, gamma=2.0)
    
    steps_per_epoch = 50 
    
    print("------------------------------------------------")
    
    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0
        epoch_acc = 0
        
        for step in range(steps_per_epoch):
            # Generate REAL Market Data
            inputs, targets, ohlc_df = generate_market_scenario(args.batch_size, 256, 16)
            
            inputs = inputs.to(device)
            targets = targets.float().to(device) 
            
            llm_ctx = torch.randn(args.batch_size, 32).to(device)
            news_vec = torch.zeros(args.batch_size, 2).to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs, llm_ctx, news_vec)
            outputs = outputs.squeeze(-1)
            
            # Loss: Asymmetric Focal Loss (Classification Logic)
            # Outputs are Logits (unbounded), Targets are 0.0, 0.5, 1.0
            loss = criterion(outputs, targets) # No Sigmoid here, handled in BCEWithLogits
            
            loss.backward()
            
            # Gradient Clipping ( Prevent Explosion )
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            epoch_loss += loss.item()
            
            # Accuracy (3-Class: Buy, Sell, Hold)
            # Outputs are LOGITS, so we apply Sigmoid first for probability
            probs = torch.sigmoid(outputs)
            
            pred_class = torch.zeros_like(probs)
            pred_class[probs > 0.6] = 1.0  # Buy
            pred_class[probs < 0.4] = 0.0  # Sell
            pred_class[(probs >= 0.4) & (probs <= 0.6)] = 0.5 # Hold
            
            acc = (pred_class == targets).float().mean().item()
            epoch_acc += acc
            
            # Telemetry
            global_step = (epoch - 1) * steps_per_epoch + step
            current_lr = optimizer.param_groups[0]['lr']
            with open(METRICS_FILE, "a", newline='') as f:
                writer = csv.writer(f)
                writer.writerow([global_step, epoch, loss.item(), acc, current_lr])
                
            time.sleep(0.01) 
            
        avg_loss = epoch_loss / steps_per_epoch
        avg_acc = epoch_acc / steps_per_epoch
        
        # Adjust Learning Rate if plateau
        scheduler.step(avg_loss)
        current_lr = scheduler.optimizer.param_groups[0]['lr']
        
        print(f"Epoch {epoch}/{args.epochs} | Loss: {avg_loss:.4f} | Acc: {avg_acc:.2%}")
        
        recorder.capture_snapshot(inputs, ohlc_df, llm_ctx, news_vec, epoch)
        leaderboard.update(epoch, avg_loss, avg_acc, model.state_dict())
        
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
