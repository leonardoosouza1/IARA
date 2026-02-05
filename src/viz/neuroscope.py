import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path
import argparse
import sys
import logging

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.train.trainer_liquid import TrainerConfig, MultiAssetDataset
from src.model.iara_liquid import IARA
# from src.train.dataset import MultiAssetDataset # Moved to trainer_liquid

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Neuroscope")

class Neuroscope:
    def __init__(self, model_path, device='cuda'):
        self.device = device
        self.output_dir = Path("plots/neuroscope")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load Model
        self.model, self.config = self._load_model(model_path)
        
        # Trace Storage
        self.traces = {
            'cfc_states': [],    # List of [batch, hidden]
            'attention_weights': None
        }
        
    def _load_model(self, path):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        # Reconstruct Config
        config_data = checkpoint.get('config', None)
        
        if isinstance(config_data, TrainerConfig):
             config = config_data
        elif isinstance(config_data, dict):
             config = TrainerConfig(**config_data)
        else:
             config = TrainerConfig()
        
        # Override architecture params if stored directly
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint # Assume raw state dict
            
        # Initialize Model
        model = IARA(config).to(self.device)
        
        # Handle state dict mismatch (if any)
        try:
            model.load_state_dict(state_dict)
        except Exception as e:
            logger.warning(f"Strict load failed: {e}. Trying non-strict.")
            # Filter distinct keys if they contain 'module.' prefix (DataParallel)
            new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            model.load_state_dict(new_state_dict, strict=False)
            
        model.eval()
        return model, config
    
    def _hook_cfc(self, module, input, output):
        # Output is h_new: [batch, hidden_dim]
        # We append to list. Since this is called for every timestep, 
        # we will have seq_len entries per layer.
        self.traces['cfc_states'].append(output.detach().cpu())

    def _hook_attention(self, module, input, output):
        # Attention output is usually [batch, seq_len, hidden]. 
        # We want the weights ideally, but IARA's attention layer returns context tensor.
        # If we can't get weights easily without modifying code, we visualize the output context magnitude.
        # Ideally, we'd hook 'attn_output_weights' inside MultiheadAttention, but that's internal.
        self.traces['attention_weights'] = output.detach().cpu()

    def register_hooks(self):
        # Hook CfC Layers
        for i, layer in enumerate(self.model.cfc_layers):
            layer.register_forward_hook(self._hook_cfc)
            logger.info(f"Hooked CfC Layer {i}")
            
        # Hook Attention
        if hasattr(self.model, 'self_attention'):
            self.model.self_attention.register_forward_hook(self._hook_attention)
            logger.info("Hooked Self Attention")

    def run(self, ticker='EURUSD', limit=100):
        # Load Data
        dataset = MultiAssetDataset(
            seq_len=self.config.seq_len,
            stride=1,
            tickers=[ticker] if ticker else None,
            augment=False,
            target_lookahead=5
        )
        
        # Pick a random sample or specific one
        idx = np.random.randint(0, len(dataset))
        x, y, w = dataset[idx]
        
        # Add batch dim
        x = x.unsqueeze(0).to(self.device) # [1, seq, 16]
        
        # Handle Feature Mismatch (e.g. Model trained on 13 features but dataset has 16)
        if x.shape[-1] > self.config.input_dim:
            logger.warning(f"Input features {x.shape[-1]} > Model expects {self.config.input_dim}. Truncating.")
            x = x[:, :, :self.config.input_dim]
            
        logger.info(f"Analyzing sample {idx} from {ticker}...")
        
        # Run Inference
        with torch.no_grad():
            self.model(x)
            
        # Process Traces
        self.plot_cfc_activation()
        self.plot_attention_magnitude()
        self.plot_input_features(x.cpu())
        
        logger.info(f"Plots saved to {self.output_dir}")

    def plot_cfc_activation(self):
        # Traces contains (seq_len * num_layers) items
        # Each item is [1, hidden_dim]
        
        seq_len = self.config.seq_len
        num_layers = self.config.num_cfc_layers
        
        # Stack all: [total_steps, hidden]
        all_states = torch.stack(self.traces['cfc_states']).squeeze(1) # [steps, hidden]
        
        # Reshape to [layers, seq, hidden]
        # Because the loop is: for layer: for time: ...
        # The hooks fire in that order? 
        # Wait, the code is:
        # for layer in layers:
        #    for t in range(seq_len):
        #        cell(x, h)
        # So yes, the list is ordered: Layer0_t0, Layer0_t1... Layer0_tN, Layer1_t0...
        
        try:
            states_per_layer = all_states.view(num_layers, seq_len, -1)
        except RuntimeError:
            logger.error("Could not reshape traces. Did the sequence length match?")
            return

        # Plot Heatmap for each layer
        for i in range(num_layers):
            plt.figure(figsize=(15, 6))
            data = states_per_layer[i].T.numpy() # [neurons, time]
            
            # Normalize for better visualization (-1 to 1 usually)
            sns.heatmap(data, cmap='viridis', center=0, vmin=-1, vmax=1)
            plt.title(f"CfC Layer {i+1} - Neuron Activation (Liquid State)")
            plt.xlabel("Time Step (M5 Bars)")
            plt.ylabel("Neuron #")
            plt.savefig(self.output_dir / f"cfc_activity_layer_{i+1}.png")
            plt.close()

    def plot_attention_magnitude(self):
        if self.traces['attention_weights'] is None:
            return
            
        # attention output: [1, seq, hidden]
        attn = self.traces['attention_weights'].squeeze(0).T.numpy() # [hidden, seq]
        
        plt.figure(figsize=(15, 6))
        # Compute "Focus" as simple magnitude of attention context vector
        focus = np.linalg.norm(attn, axis=0) # [seq]
        
        plt.plot(focus, color='orange', linewidth=2)
        plt.fill_between(range(len(focus)), focus, color='orange', alpha=0.3)
        plt.title("Fractal Attention Magnitude (Temporal Focus)")
        plt.xlabel("Time Step (M5 Bars)")
        plt.ylabel("Attention Energy")
        plt.grid(True, alpha=0.3)
        plt.savefig(self.output_dir / "attention_magnitude.png")
        plt.close()

    def plot_input_features(self, x):
        # x: [1, seq, features]
        x = x.squeeze(0).numpy().T # [features, seq]
        
        plt.figure(figsize=(15, 8))
        sns.heatmap(x, cmap='coolwarm', center=0, robust=True)
        plt.title("Input Features (Normalized)")
        plt.xlabel("Time Step")
        plt.ylabel("Feature Index")
        plt.savefig(self.output_dir / "input_features.png")
        plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="checkpoints/best_model.pth")
    parser.add_argument("--limit", type=int, default=1)
    args = parser.parse_args()
    
    scope = Neuroscope(args.model)
    scope.register_hooks()
    scope.run()
