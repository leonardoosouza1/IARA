"""
Inference - Generate Market Predictions from Trained Saci-Nano Model
=====================================================================
Load a trained checkpoint and generate next token predictions.
"""

import argparse
import logging
from pathlib import Path
from typing import Optional

import torch
import numpy as np

from src.model.saci_nano import SaciNano, SaciNanoConfig
from src.data.tokenizer import FinancialTokenizer
from src.model.saci_nano import SaciNano, SaciNanoConfig
from src.data.tokenizer import FinancialTokenizer
from src.train.dataset import StructuredForexDataset, create_dataloader
from src.train.trainer import TrainerConfig
from src.train.test_direction import get_neutral_threshold, evaluate_direction

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Paths
CHECKPOINT_DIR = Path(__file__).parent.parent.parent / "checkpoints"
TOKENIZER_PATH = Path(__file__).parent.parent.parent / "data" / "tokenizer.pkl"
DATA_TOKENIZED_DIR = Path(__file__).parent.parent.parent / "data" / "tokenized"


def load_model(checkpoint_path: Path, device: torch.device) -> SaciNano:
    """Load model from checkpoint."""
    logger.info(f"Loading model from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    model_config = checkpoint['model_config']
    model = SaciNano(model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    epoch = checkpoint.get('epoch', 'unknown')
    logger.info(f"Loaded model from epoch {epoch}")
    logger.info(f"Model parameters: {model.count_parameters():,}")
    
    return model


def analyze_predictions(
    model: SaciNano,
    tokenizer: FinancialTokenizer,
    dataset: StructuredForexDataset,
    num_samples: int = 5,
    device: torch.device = torch.device("cpu")
):
    """
    Analyze model predictions on sample sequences.
    """
    print("\n" + "=" * 60)
    print("SACI-NANO PREDICTION ANALYSIS")
    print("=" * 60)
    
    print("=" * 60)
    
    # Use valid dataset (no shuffle)
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    
    for i, idx in enumerate(indices):
        inputs, targets = dataset[idx]
        
        # Add batch dim
        inputs_t = inputs.unsqueeze(0).to(device)
        targets_t = targets.unsqueeze(0).to(device)
        
        with torch.no_grad():
            logits, loss = model(inputs_t, targets_t)
            probs = torch.softmax(logits, dim=-1)
            predictions = logits.argmax(dim=-1)
        
        # Identify Target positions (where target != -100)
        # Note: targets tensor contains the LABELS for the Next step.
        # But we want to see what the model predicted for the "Price" token.
        # In our dataset logic, the label for the Price token input is Next_Feature (which is masked -100).
        # Wait, the label for the Feature[Last] input IS the Price token.
        
        # Let's find indices where the LABEL is valid (not -100)
        # These correspond to the moments we are predicting the Price.
        valid_indices = (targets_t[0] != -100).nonzero(as_tuple=True)[0]
        
        if len(valid_indices) == 0:
            print(f"\n--- Sample {i + 1} (No valid targets found in window) ---")
            continue

        # Calculate accuracy on Valid Targets Only
        valid_preds = predictions[0][valid_indices]
        valid_targets = targets_t[0][valid_indices]
        
        correct = (valid_preds == valid_targets).float()
        accuracy = correct.mean().item() * 100
        
        
        # Loss is already computed correctly by the model with ignore_index=-100
        
        print(f"\n--- Sample {i + 1} ---")
        print(f"  Loss: {loss.item():.4f}")
        print(f"  Exact Accuracy: {accuracy:.2f}%")
        
        # Calculate Directional Metrics for this sample
        target_col = "close_log_ret"
        try:
             neutral = get_neutral_threshold(tokenizer, target_col)
             
             pred_sign = torch.sign(valid_preds - neutral)
             targ_sign = torch.sign(valid_targets - neutral)
             sign_acc = (pred_sign == targ_sign).float().mean().item() * 100
             
             print(f"  Directional Acc: {sign_acc:.2f}% (Neutral ID: {neutral})")
        except:
             pass
        
        print(f"\n  Predictions (Last 5 valid targets):")
        for pos in valid_indices[-5:]:
            target_token = targets_t[0, pos].item()
            pred_token = predictions[0, pos].item()
            prob = probs[0, pos, pred_token].item()
            
            # The input that generated this prediction was inputs[pos]
            input_token = inputs_t[0, pos].item()
            
            input_feat, input_bin = tokenizer.decode_token(input_token)
            target_feat, target_bin = tokenizer.decode_token(target_token)
            pred_feat, pred_bin = tokenizer.decode_token(pred_token)
            
            # Use arrows for direction
            target_dir = "▲" if target_token > neutral else "▼"
            pred_dir = "▲" if pred_token > neutral else "▼"
            
            dir_match = "✓" if target_dir == pred_dir else "✗"
            
            match = "✓" if pred_token == target_token else "✗"
            print(f"    Input: {input_feat[:10]:>10} -> Target: {target_feat[:10]:>10}[{target_bin:3d}] {target_dir} | "
                  f"Pred: {pred_feat[:10]:>10}[{pred_bin:3d}] {pred_dir} (p={prob:.2f}) {dir_match}")


def generate_sequence(
    model: SaciNano,
    tokenizer: FinancialTokenizer,
    seed_tokens: torch.Tensor,
    num_tokens: int = 50,
    temperature: float = 0.8,
    top_k: int = 10,
    device: torch.device = torch.device("cpu")
):
    """
    Generate a sequence of tokens autoregressively.
    """
    print("\n" + "=" * 60)
    print("AUTOREGRESSIVE GENERATION")
    print("=" * 60)
    
    seed_tokens = seed_tokens.unsqueeze(0).to(device)
    
    with torch.no_grad():
        generated = model.generate(
            seed_tokens,
            max_new_tokens=num_tokens,
            temperature=temperature,
            top_k=top_k
        )
    
    print(f"\nSeed length: {seed_tokens.size(1)} tokens")
    print(f"Generated: {num_tokens} new tokens")
    print(f"Temperature: {temperature}, Top-k: {top_k}")
    
    # Decode generated tokens
    print("\nGenerated Token Sequence (decoded):")
    gen_tokens = generated[0, seed_tokens.size(1):].cpu().numpy()
    
    for i, token in enumerate(gen_tokens[:20]):  # Show first 20
        feat_name, bin_idx = tokenizer.decode_token(int(token))
        print(f"  [{i:3d}] {feat_name}: bin {bin_idx}")


def show_model_stats(model: SaciNano, checkpoint: dict):
    """Display model statistics."""
    print("\n" + "=" * 60)
    print("MODEL STATISTICS")
    print("=" * 60)
    
    config = model.config
    print(f"\n  Architecture:")
    print(f"    - Vocabulary Size: {config.vocab_size:,}")
    print(f"    - Model Dimension: {config.dim}")
    print(f"    - Layers: {config.n_layers}")
    print(f"    - Attention Heads: {config.n_heads}")
    print(f"    - KV Heads (GQA): {config.n_kv_heads}")
    print(f"    - Max Sequence Length: {config.max_seq_len}")
    
    print(f"\n  Training:")
    print(f"    - Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"    - Global Step: {checkpoint.get('global_step', 'N/A'):,}")
    print(f"    - Best Loss: {checkpoint.get('best_loss', 'N/A'):.4f}" if isinstance(checkpoint.get('best_loss'), float) else f"    - Best Loss: N/A")
    
    print(f"\n  Parameters: {model.count_parameters():,}")


def main():
    parser = argparse.ArgumentParser(description="Saci-Nano Inference")
    parser.add_argument("--checkpoint", "-c", type=str, default="latest.pth",
                       help="Checkpoint file name (in checkpoints/)")
    parser.add_argument("--num-samples", "-n", type=int, default=3,
                       help="Number of samples to analyze")
    parser.add_argument("--generate", "-g", type=int, default=0,
                       help="Number of tokens to generate (0 to skip)")
    parser.add_argument("--temperature", "-t", type=float, default=0.8,
                       help="Generation temperature")
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load tokenizer
    try:
        tokenizer = FinancialTokenizer.load(TOKENIZER_PATH)
        logger.info(f"Tokenizer loaded. Vocab size: {tokenizer.vocab_size}")
    except FileNotFoundError:
        logger.error("Tokenizer not found. Run the pipeline first.")
        return
    
    # Load checkpoint
    checkpoint_path = CHECKPOINT_DIR / args.checkpoint
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        available = list(CHECKPOINT_DIR.glob("*.pth"))
        if available:
            logger.info(f"Available checkpoints: {[p.name for p in available]}")
        return
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = load_model(checkpoint_path, device)
    
    # Show stats
    show_model_stats(model, checkpoint)
    
    # Load dataset for analysis
    try:
        # Use StructuredForexDataset with is_training=False to avoid shuffling during inference
        dataset = StructuredForexDataset(
            data_dir=DATA_TOKENIZED_DIR,
            seq_len=model.config.max_seq_len,
            is_training=False
        )
        logger.info(f"Dataset loaded: {len(dataset)} samples")
        
        # Analyze predictions
        analyze_predictions(model, tokenizer, dataset, args.num_samples, device)
        
        # Generate if requested
        if args.generate > 0:
            seed_tokens, _ = dataset[0]
            generate_sequence(
                model, tokenizer, seed_tokens[:64],  # Use first 64 as seed
                num_tokens=args.generate,
                temperature=args.temperature,
                device=device
            )
            
    except Exception as e:
        logger.error(f"Dataset error: {e}")
    
    print("\n" + "=" * 60)
    print("INFERENCE COMPLETE")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
