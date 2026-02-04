
import logging
import argparse
from pathlib import Path
import sys
import torch
import torch.nn as nn
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import polars as pl
from tqdm import tqdm

# Add root to path
sys.path.append(".")

from src.model.iara_liquid import IARA, LiquidConfig
from src.train.trainer_liquid import MultiAssetDataset, TrainerConfig
from src.data.processor import Normalizer, NormalizerState

# Configure Headless Plotting
plt.switch_backend('Agg')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("BrainScan")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(checkpoint_path, input_dim=16):
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    state_dict = checkpoint['model']
    
    # Handle compiled model names
    if list(state_dict.keys())[0].startswith('_orig_mod.'):
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
        
    config = checkpoint.get('config')
    
    # Enforce v6.7 settings if config missing new fields
    liquid_config = LiquidConfig(
        input_dim=input_dim,
        hidden_dim=getattr(config, 'hidden_dim', 128),
        num_heads=getattr(config, 'num_heads', 4),
        dropout=getattr(config, 'dropout', 0.5),
        max_seq_len=getattr(config, 'max_seq_len', 256),
        num_cfc_layers=2 # v6.7 standard
    )
    
    model = IARA(liquid_config).to(DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    return model

def visualize(args):
    # 1. Load Normalizer for feature names
    normalizer = Normalizer.load("data/scaler.pkl")
    feature_names = normalizer.feature_names
    input_dim = len(feature_names)
    logger.info(f"Loaded {input_dim} features: {feature_names}")
    
    # 2. Load Model
    model = load_model(args.checkpoint, input_dim)
    logger.info("Model loaded.")
    
    # 3. Load Data Sample
    dataset = MultiAssetDataset(
        data_dir="data/processed",
        seq_len=256,
        stride=100, # Sparse sampling
        augment=False,
        target_col="close_log_ret"
    )
    
    # Pick a batch
    indices = range(0, min(len(dataset), args.batch_size))
    batch_tensors = [dataset[i][0] for i in indices]
    batch_tensor = torch.stack(batch_tensors).to(DEVICE) # [B, T, D]
    
    logger.info(f"Visualizing batch: {batch_tensor.shape}")
    
    # 4. Hook Mechanism
    # We want to capture FeatureAttention gates
    # FeatureAttention returns x * gate(x). We want 'gate(x)'.
    # A backward hook on gradients is one way, but forward hook on GATE is better.
    # Structure: model.feature_attention.gate (Sequential)
    
    attention_outputs = {}
    
    def hook_fn(module, input, output):
        # output is the gate value (sigmoid output)
        attention_outputs['gate'] = output.detach()
        
    # Register hook on the sigmoid layer of the specific attention module
    # model.feature_attention.gate is a Sequential. Last layer is Sigmoid.
    hook = model.feature_attention.gate[-1].register_forward_hook(hook_fn)
    
    # 5. Forward Pass
    with torch.no_grad():
        logits, _ = model(batch_tensor)
        
    hook.remove()
    
    # 6. Process Attention (Back-Projection)
    # Gate: [B, T, Hidden]
    # InputProj: [Hidden, Input] (Linear weight is [Out, In])
    # Importance = Gate @ Weight
    
    gate_values = attention_outputs['gate'] # [B, T, 128]
    proj_weight = model.input_proj.weight.abs() # [128, 16]
    
    # Broadcast matmul: [B, T, 128] x [128, 16] -> [B, T, 16]
    # We use torch.matmul which handles broadcasting
    feature_importance_raw = torch.matmul(gate_values, proj_weight) # [B, T, 16]
    
    # Normalize per step for heatmap
    # feature_importance = feature_importance_raw / feature_importance_raw.sum(dim=-1, keepdim=True)
    feature_importance = feature_importance_raw
    
    # 7. Visualization 1: Aggregated Importance (Global)
    # Sum over Batch and Time
    global_importance = feature_importance.mean(dim=(0, 1)).detach().cpu().numpy()
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x=feature_names, y=global_importance, palette="viridis")
    plt.title("Reference: IARA v6.7 Global Feature Importance (Gate Projection)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    Path("reports/figures").mkdir(parents=True, exist_ok=True)
    plt.savefig("reports/figures/feature_importance_global.png")
    logger.info("Saved reports/figures/feature_importance_global.png")
    
    # 8. Visualization 2: Temporal Dynamics (Best Sample)
    # Find sample with highest confidence (logit magnitude)
    conf = torch.sigmoid(logits).abs() - 0.5 # Distance from neutral
    best_idx = torch.argmax(conf).item()
    
    sample_imp = feature_importance[best_idx].detach().cpu().numpy().T # [Feat, Time]
    
    plt.figure(figsize=(14, 8))
    sns.heatmap(sample_imp, yticklabels=feature_names, cmap="magma", robust=True)
    plt.title(f"IARA v6.7 Brain Scan - Sample #{best_idx} (Conf: {torch.sigmoid(logits[best_idx]):.2f})")
    plt.xlabel("Time Steps (Past -> Now)")
    plt.ylabel("Features")
    plt.tight_layout()
    plt.savefig("reports/figures/brain_scan_sample.png")
    logger.info("Saved reports/figures/brain_scan_sample.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/latest.pth")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    
    if not Path(args.checkpoint).exists():
        logger.error(f"Checkpoint {args.checkpoint} not found. Train first!")
        exit(1)
        
    visualize(args)
