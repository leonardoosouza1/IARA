import torch
import pandas as pd
import numpy as np
from pathlib import Path

# Load data
try:
    data_path = "data/cache/train.pt"
    if not Path(data_path).exists():
        # Try shards
        shard = list(Path("data/cache/train_shards").glob("*.pt"))[0]
        data_path = str(shard)

    print(f"Loading {data_path}...")
    data = torch.load(data_path, weights_only=False)
    
    X = data['X'].numpy()
    y = data['y'].numpy()
    
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    
    # Check for NaNs
    if np.isnan(X).any():
        print("⚠️ NaNs found in X!")
    if np.isnan(y).any():
        print("⚠️ NaNs found in y!")
        
    # Flatten X to 2D using the last timestep (most relevant for leakage)
    X_last = X[:, -1, :]
    print(f"Analyzing correlations on last timestep: {X_last.shape}")
    
    # Correlations
    df = pd.DataFrame(X_last)
    df['target'] = y
    
    corr = df.corr()['target'].sort_values(ascending=False)
    
    print("\nTop Correlations with Target:")
    print(corr.head(10))
    print(corr.tail(10))
    
    # Check for exact matches
    for col in df.columns[:-1]:
        if abs(corr[col]) > 0.95:
            print(f"\n🚨 POTENTIAL LEAKAGE DETECTED in Feature {col} (Corr: {corr[col]:.4f})")
            
except Exception as e:
    print(f"Error: {e}")
