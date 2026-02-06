import torch
import os

try:
    path = "data/cache/train.pt"
    data = torch.load(path)
    if 'X' in data:
        print("X Shape:", data['X'].shape)
    if 'y' in data:
        print("y Shape:", data['y'].shape)
        
    path_val = "data/cache/val.pt"
    data_val = torch.load(path_val)
    if 'X' in data_val:
        print("Val X Shape:", data_val['X'].shape)

except Exception as e:
    print(f"Error: {e}")
