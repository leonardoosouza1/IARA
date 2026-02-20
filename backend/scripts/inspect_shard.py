
import torch
import sys

try:
    path = "data/cache/train_shards/shard_0.pt"
    data = torch.load(path)
    X = data['X']
    print(f"Shard 0 Shape: {X.shape}")
    print(f"Feature Dim: {X.shape[-1]}")
except Exception as e:
    print(e)
