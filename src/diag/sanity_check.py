
import sys
import torch
import numpy as np
import polars as pl
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(".")
from src.model.iara_liquid import IARA, LiquidConfig
from src.train.trainer_liquid import MultiAssetDataset, TrainerConfig

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_checkpoint(path):
    checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
    config_dict = checkpoint['config']
    
    # Reconstruct Config
    # If config is object, convert to dict or use attributes
    if isinstance(config_dict, TrainerConfig):
        t_conf = config_dict
    else:
        # Fallback
        t_conf = TrainerConfig(**config_dict)
        
    liquid_config = LiquidConfig(
        input_dim=t_conf.input_dim,
        hidden_dim=t_conf.hidden_dim, 
        num_heads=t_conf.num_heads,
        dropout=t_conf.dropout,
        max_seq_len=t_conf.max_seq_len,
        num_cfc_layers=t_conf.num_cfc_layers
    )
    
    model = IARA(liquid_config).to(DEVICE)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    return model, t_conf

def sanity_check():
    print("🧠 Loading Interrupted Model...")
    model, config = load_checkpoint("checkpoints/interrupted_model.pth")
    
    print("📂 Loading Validation Data...")
    ds = MultiAssetDataset(seq_len=config.seq_len, stride=config.stride, augment=False)
    _, val_idxs = ds.get_chronological_split(config.val_split)
    
    # Take a subset for speed
    val_subset = torch.utils.data.Subset(ds, val_idxs[:2000]) 
    loader = torch.utils.data.DataLoader(val_subset, batch_size=256, shuffle=False)
    
    all_preds = []
    all_targets = []
    all_probs = []
    
    print("🔍 Running Inference...")
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            for X, y, _ in loader:
                X = X.to(DEVICE)
                logits, _ = model(X)
                probs = torch.sigmoid(logits).cpu().numpy()
                preds = (probs > 0.5).astype(float)
                
                all_probs.extend(probs)
                all_preds.extend(preds)
                all_targets.extend(y.numpy())
                
    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    print("\n" + "="*40)
    print("📊 DIAGNOSTIC RESULTS")
    print("="*40)
    
    # 1. Bias Check
    mean_prob = np.mean(all_probs)
    perc_up = np.mean(all_preds) * 100
    print(f"Mean Prediction Probability: {mean_prob:.4f}")
    print(f"Predicted 'UP' Ratio:      {perc_up:.2f}% (Target: 50% usually)")
    
    if perc_up > 65:
        print("⚠️  WARNING: Model is PERMABULL (Biased towards UP)")
    elif perc_up < 35:
        print("⚠️  WARNING: Model is PERMABEAR (Biased towards DOWN)")
    else:
        print("✅  Bias seems balanced.")
        
    # 2. Confusion Matrix
    cm = confusion_matrix(all_targets, all_preds)
    print("\nConfusion Matrix:")
    print(cm)
    print("TN  FP")
    print("FN  TP")
    
    # 3. High Confidence Check
    high_conf_mask = np.abs(all_probs - 0.5) > 0.1 # Prob > 0.6 or < 0.4
    if np.sum(high_conf_mask) > 0:
        hc_acc = np.mean(all_preds[high_conf_mask] == all_targets[high_conf_mask])
        print(f"\nHigh Confidence Accuracy (>60%): {hc_acc:.2%} ({np.sum(high_conf_mask)} samples)")
    else:
        print("\n❌ No High Confidence samples found.")

if __name__ == "__main__":
    sanity_check()
