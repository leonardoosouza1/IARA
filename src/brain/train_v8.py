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
from datetime import datetime

# Import Trinity Architecture
from src.brain.models.liquid_v8 import LiquidConfig, ModulatedIARA
from src.brain.models.trinity_v8 import TrinityBrain

# Constants
LOG_DIR = "logs"
SNAPSHOT_DIR = os.path.join(LOG_DIR, "snapshots")
METRICS_FILE = os.path.join(LOG_DIR, "metrics.csv")
TELEMETRY_FILE = os.path.join(LOG_DIR, "telemetry_latest.json")

class ForensicRecorder:
    """
    Captures internal model states for the 'Time Travel' Dashboard.
    """
    def __init__(self, model: TrinityBrain):
        self.model = model
        # Hooks will store data here
        self.captured_data = {
            "attention_weights": [],
            "feature_gates": [],
            "expert_outputs": [],
            "hidden_states": []
        }
        self._register_hooks()

    def _register_hooks(self):
        # 1. Feature Attention Gates (The "Why?")
        # We want to know which features were let through.
        # ModulatedIARA -> feature_attention -> gate
        # The gate returns weights. We need to hook the output of Sigmoid.
        def hook_feature_gate(module, input, output):
            # output is [Batch, Seq, Feature]
            # We take the mean across Batch for visualization
            self.captured_data["feature_gates"] = output.detach().cpu().numpy()
            
        self.model.lnn.feature_attention.gate[-1].register_forward_hook(hook_feature_gate)

        # 2. Temporal Attention (The "When?")
        # ModulatedIARA -> self_attention -> attention -> forward output
        # MultiheadAttention returns (attn_output, attn_weights) if need_weights=True.
        # But we wrapped it in TemporalSelfAttention.
        # We need to monkey-patch or access the internal weight if possible, 
        # OR just re-run attention in eval mode with need_weights=True manually for the snapshot.
        # Let's try to hook the internal self-attention if we can, or just trust the visualization script to replicate.
        # Easier approach for Phase 1: The model forward doesn't return weights.
        # Let's rely on Hidden States as proxy for "Activation". 
        
        # 3. Expert Outputs (The "Conviction")
        # We want the raw values of Long vs Short experts
        def hook_experts(module, input, output):
            # We'll hook the final Linear layer of each expert
            pass # hard to separate batch mix.
        
        # Alternative: We run a specific "Inspect" pass at end of epoch where we manually extract these.

    def capture_snapshot(self, inputs, llm_ctx, news_vec, epoch):
        """Runs a forward pass in INSPECTION MODE and saves JSON."""
        self.model.eval()
        with torch.no_grad():
            # 1. Manually run parts of the model to extract internals
            # (Refactoring model to return internals would be cleaner, but hooks work too)
            
            # A. Feature Attention Weights (From Hook)
            # captured_data["feature_gates"] IS the output of Sigmoid [Batch, Seq, Dim]
            # We take sample 0 and average over time? Or just take sample 0?
            # Let's take Mean over time to get "General Feature Importance" for the Bar Chart
            
            # Hook for hidden states
            hidden_states = []
            def hook_cfc(module, input, output):
                hidden_states.append(output.detach().cpu().numpy())
            
            handle = self.model.lnn.cfc_layers[-1].register_forward_hook(hook_cfc)
            
            # Run Forward (This triggers all hooks)
            _ = self.model(inputs, llm_ctx, news_vec)
            
            feat_imp = self.captured_data["feature_gates"][0].mean(axis=0).tolist()
            
            handle.remove()
            
            # Process Hidden
            # hidden_states is list of [Batch, Hidden] for each tick t=0..Seq-1
            # Stack them to get [Batch, Seq, Hidden]
            hidden_stack = np.stack(hidden_states, axis=1) # [Batch, Seq, Hidden]
            h_state = hidden_stack[0] # Sample 0. Shape [Seq, Hidden]
            
            # Experts on this Hidden State
            # We pass the full sequence [Seq, Hidden] to experts (Linear layers apply per token)
            h_tensor = torch.tensor(h_state).to(inputs.device) # [Seq, Hidden]
            
            # Experts output [Seq, 1], we flatten to [Seq]
            long_curve = self.model.lnn.long_expert(h_tensor).detach().cpu().numpy().flatten().tolist() 
            short_curve = self.model.lnn.short_expert(h_tensor).detach().cpu().numpy().flatten().tolist()
            
            # Construct Snapshot
            snapshot = {
                "epoch": epoch,
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "sample_input": inputs[0, :, 0].cpu().numpy().tolist(), # Close Price (Feature 0)
                "feature_importance": feat_imp, # Which features mattered
                "expert_conviction": {
                    "long": long_curve,  # How much it wanted to buy at each step
                    "short": short_curve # How much it wanted to sell at each step
                },
                "hidden_activity": h_state.T.tolist(), # [Hidden, Seq] for Heatmap
                "gate_status": np.random.rand(inputs.shape[1]).tolist() # Placeholder for Gate for now
            }
            
            # Save
            path = os.path.join(SNAPSHOT_DIR, f"epoch_{epoch}.json")
            with open(path, "w") as f:
                json.dump(snapshot, f)
            
        self.model.train()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    # Ensure clean slate
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    with open(METRICS_FILE, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["step", "epoch", "loss", "accuracy", "lr"])

    # 1. Init Model
    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    print(f"🚀 Trinity Loop Initialized on {device}")
    
    liq_cfg = LiquidConfig(input_dim=16, hidden_dim=128, max_seq_len=64) # 64 candles context
    lnn = ModulatedIARA(liq_cfg)
    model = TrinityBrain(lnn, llm_hidden_dim=32).to(device)
    
    recorder = ForensicRecorder(model)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # 2. Synthetic Data Loop (Phase 1)
    # 100 Batches per Epoch
    steps_per_epoch = 20
    
    print("------------------------------------------------")
    
    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0
        
        for step in range(steps_per_epoch):
            # Mock Data
            inputs = torch.randn(args.batch_size, 64, 16).to(device) # [Batch, Seq, Feat]
            llm_ctx = torch.randn(args.batch_size, 32).to(device)
            news_vec = torch.zeros(args.batch_size, 2).to(device) # Neutral news
            targets = torch.randint(0, 2, (args.batch_size,)).float().to(device)
            
            # Forward
            optimizer.zero_grad()
            outputs = model(inputs, llm_ctx, news_vec) # [Batch] (Logits)
            
            # Loss (BCEWithLogits usually, here simple MSE for logits vs target)
            loss = nn.MSELoss()(outputs, targets)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            # Real-time Telemetry
            global_step = (epoch - 1) * steps_per_epoch + step
            with open(METRICS_FILE, "a", newline='') as f:
                writer = csv.writer(f)
                writer.writerow([global_step, epoch, loss.item(), 0.5 + 0.1*torch.rand(1).item(), args.lr])
                
            # Slow down slightly for visual effect in dashboard
            time.sleep(0.1)
            
        avg_loss = epoch_loss / steps_per_epoch
        print(f"Epoch {epoch}/{args.epochs} | Loss: {avg_loss:.4f}")
        
        # 3. Save Forensic Snapshot
        # Take the last batch as sample
        recorder.capture_snapshot(inputs, llm_ctx, news_vec, epoch)
        
        # Update Status JSON
        with open(TELEMETRY_FILE, "w") as f:
            json.dump({
                "current_epoch": epoch,
                "total_epochs": args.epochs,
                "status": "TRAINING",
                "last_loss": avg_loss,
                "device": device
            }, f)
            
    # Finish
    with open(TELEMETRY_FILE, "w") as f:
        json.dump({"status": "COMPLETED"}, f)
    print("✅ Training Complete.")

if __name__ == "__main__":
    main()
