import torch
import sys

def main():
    path = '/app/checkpoints/best_model_v9_BTCUSD.pth'  # wait, backtest_v9.py uses args.model
    # Let's get it from sys.argv
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = '/app/logs/models/best_model_v9_BTCUSD.pth'
        
    try:
        ckpt = torch.load(path, map_location='cpu')
        print(f"Loaded checkpoint from: {path}")
        print("--- KEYS ---")
        for key in ckpt.keys():
            if 'proj' in key or 'macro_encoder' in key:
                print(f"{key}: {ckpt[key].shape}")
    except Exception as e:
        print(f"Error loading {path}: {e}")

if __name__ == "__main__":
    main()
