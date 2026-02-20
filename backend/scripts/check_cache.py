import torch
import os

try:
    path = "data/cache/train.pt"
    print(f"Loading {path}...")
    data = torch.load(path)
    if isinstance(data, dict):
       # Trainer liquid implies it saves datasets, or maybe raw tensors
       print("Data is Dict keys:", data.keys())
       if "features" in data:
           print("Features Shape:", data["features"].shape)
       if "targets" in data:
           print("Targets Shape:", data["targets"].shape)
    else:
       # Accessing TensorDataset internals if saved as whole object
       if hasattr(data, 'tensors'):
           print("Feature Tensor:", data.tensors[0].shape)
           print("Target Tensor:", data.tensors[1].shape)
       else:
           print("Data Type:", type(data))
except Exception as e:
    print(f"Error: {e}")
