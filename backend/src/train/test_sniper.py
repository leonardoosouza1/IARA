import os
import torch
import torch.nn.functional as F
from src.brain.models.liquid_v10 import SwarmBrain, WorkerConfig, CoordinatorConfig

def simulate_timeframes(x: torch.Tensor) -> dict:
    x_dict = {}
    x_dict["M5"] = x
    x_dict["M15"] = x[:, ::3, :]
    x_dict["M30"] = x[:, ::6, :]
    x_dict["H1"] = x[:, ::12, :]
    x_dict["H4"] = x[:, ::16, :]
    return x_dict

def test_sniper_mode():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing Sniper Mode on {device}...")
    
    # 1. Load Model
    input_dim = 14
    worker_cfg = WorkerConfig(input_dim=input_dim, hidden_dim=64)
    coord_cfg = CoordinatorConfig(num_workers=5, worker_dim=64, hidden_dim=256)
    model = SwarmBrain(worker_cfg, coord_cfg).to(device)
    
    checkpoint_path = "logs/models/madrugada.pth"
    if not os.path.exists(checkpoint_path):
        print(f"Error: {checkpoint_path} not found.")
        return
        
    state = torch.load(checkpoint_path, map_location=device)
    state_dict = state.get('model_state_dict', state)
    
    # Apply dynamic padding for any older layers if needed (just to be safe)
    for name, param in model.named_parameters():
        if name in state_dict and any(key in name for key in ["input_proj.weight", "cfc.x_proj.weight", "res_proj.weight"]):
            old_w = state_dict[name]
            if len(old_w.shape) == 2 and old_w.shape[1] < param.shape[1]:
                new_w = torch.zeros_like(param)
                new_w[:, :old_w.shape[1]] = old_w
                state_dict[name] = new_w
                
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    print("Model loaded successfully.")
    
    # 2. Load Validation Data (from BTCUSD cache)
    shard_dir = "data/cache/BTCUSD/val_shards"
    if not os.path.exists(shard_dir):
        print("No validation shards found. Using train shards for a quick test...")
        shard_dir = "data/cache/BTCUSD/train_shards"
        
    shards = sorted([f for f in os.listdir(shard_dir) if f.endswith(".pt")])
    if not shards:
        print("No shards available to test.")
        return
        
    print(f"Loaded {len(shards)} shards for testing. Processing first 5 shards...")
    
    total_samples = 0
    
    # Standard Metrics
    standard_correct = 0
    standard_trades = 0
    
    # Sniper Metrics
    sniper_correct = 0
    sniper_trades = 0
    sniper_threshold = 0.82
    
    batch_size = 512
    
    with torch.no_grad():
        for shard in shards[:5]:
            data = torch.load(os.path.join(shard_dir, shard), map_location='cpu')
            X = data['X']
            y = data['y']
            
            # Pad dataset to 14
            is_old_shard = (X.shape[-1] in [12, 22, 26])
            if is_old_shard:
                core_X = X[:, :, :12]
                if input_dim > 12:
                    padding = torch.zeros(X.shape[0], X.shape[1], input_dim - 12, dtype=X.dtype, device=X.device)
                    X = torch.cat([core_X, padding], dim=-1)
                else:
                    X = core_X
            else:
                if X.shape[-1] > input_dim:
                    X = X[:, :, :input_dim]
                elif X.shape[-1] < input_dim:
                    padding = torch.zeros(X.shape[0], X.shape[1], input_dim - X.shape[-1], dtype=X.dtype, device=X.device)
                    X = torch.cat([X, padding], dim=-1)
                    
            for i in range(0, len(X), batch_size):
                batch_X = X[i : i + batch_size].to(device)
                batch_y = y[i : i + batch_size].to(device)
                
                # Truncate sequence length if necessary (usually 128)
                if batch_X.shape[1] > 128:
                    batch_X = batch_X[:, -128:, :]
                    
                x_dict = simulate_timeframes(batch_X)
                logits, _, _, _ = model(x_dict)
                probs = F.softmax(logits, dim=1)
                
                # Targets: 0=Long, 1=Short, 2=Wait
                
                # 1. Standard mode (Argmax ignores confidence threshold)
                preds_argmax = torch.argmax(probs, dim=1)
                for pred, target in zip(preds_argmax, batch_y):
                    target = int(target.item())
                    pred = int(pred.item())
                    # Only score situations where the model wanted to act, OR where a trade existed
                    # For a true backtest precision we care if an ACTION was correct
                    if pred in [0, 1]:  # Model wanted to trade
                        standard_trades += 1
                        if pred == target:
                            standard_correct += 1
                            
                # 2. Sniper mode (>0.82 Confidence)
                max_probs, sniper_preds = torch.max(probs, dim=1)
                for prob, pred, target in zip(max_probs, sniper_preds, batch_y):
                    target = int(target.item())
                    pred = int(pred.item())
                    prob = float(prob.item())
                    
                    if pred in [0, 1] and prob >= sniper_threshold:  # Sniper trigger pulled
                        sniper_trades += 1
                        if pred == target:
                            sniper_correct += 1
                            
                total_samples += len(batch_X)
                
    print("\n" + "="*50)
    print("🎯 CRYPTO ARMOR (v10.0) - SNIPER MODE BENCHMARK")
    print("="*50)
    print(f"Total Market Snapshots Evaluated: {total_samples}")
    print("-" * 50)
    
    # Standard Results
    std_win_rate = (standard_correct / standard_trades * 100) if standard_trades > 0 else 0
    print(f"[STANDARD MODE] (Any Confidence > Margin)")
    print(f"Trades Taken: {standard_trades} / {total_samples}")
    print(f"Correct Hits: {standard_correct}")
    print(f"Win Rate:     {std_win_rate:.2f}%")
    print("-" * 50)
    
    # Sniper Results
    snp_win_rate = (sniper_correct / sniper_trades * 100) if sniper_trades > 0 else 0
    print(f"[SNIPER MODE] (Strict Confidence > {sniper_threshold*100}%)")
    print(f"Trades Taken: {sniper_trades} / {total_samples} (Filtering out Noise)")
    print(f"Correct Hits: {sniper_correct}")
    print(f"Win Rate:     {snp_win_rate:.2f}%")
    print("="*50 + "\n")

if __name__ == "__main__":
    test_sniper_mode()
