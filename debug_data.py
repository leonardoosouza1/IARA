import torch

try:
    print("Loading train.pt...")
    data = torch.load("data/cache/train.pt")
    
    if isinstance(data, dict):
        if 'X' in data:
            X = data['X']
            print(f"Found 'X' key with shape {X.shape}")
            
            print(f"Mean: {X.float().mean().item()}")
            print(f"Std: {X.float().std().item()}")
            print(f"Min: {X.min().item()}")
            print(f"Max: {X.max().item()}")
            
            # Check for NaNs
            if torch.isnan(X).any():
                print("⚠️ WARNING: NaNs found in X!")
            
            # Check for constant features
            std_per_feature = X.float().std(dim=0)
            print(f"Std per feature: {std_per_feature}")
            if (std_per_feature == 0).any():
                 print("⚠️ WARNING: Constant features found!")
        else:
            print("Could not find 'X' key")
            exit(1)
    else:
        print(f"Data Type: {type(data)}")
        exit(1)

except Exception as e:
    import traceback
    traceback.print_exc()
