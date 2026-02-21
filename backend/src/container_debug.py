
import torch
import os
from src.train.continuous_learner import ContinuousLearner
from src.brain.models.liquid_v9 import SwarmBrain
from src.brain.models.generator import ChallengerGenerator

def debug():
    print("--- Docker Container Debug ---")
    learner = ContinuousLearner()
    print(f"Device: {learner.device}")
    print(f"Learner workers hidden: {learner.worker_hidden}")
    
    # Check model dims
    print(f"Model input_dim (implicitly from workers): {learner.model.worker_config.input_dim}")
    
    # Check challenger dims
    print(f"Challenger input_dim: {learner.challenger.input_dim}")
    print(f"Challenger encoder[0] weight shape: {learner.challenger.encoder[0].weight.shape}")

    # Check a shard if exists
    shard_dir = "data/cache/EURUSD/train_shards"
    if os.path.exists(shard_dir):
        shards = [f for f in os.listdir(shard_dir) if f.endswith(".pt")]
        if shards:
            shard_path = os.path.join(shard_dir, shards[0])
            print(f"Checking shard: {shard_path}")
            data = torch.load(shard_path, map_location='cpu')
            X = data['X']
            print(f"Shard X shape: {X.shape}")
            if X.shape[-1] > 12:
                print(f"Warning: Shard has {X.shape[-1]} features, model expects 12.")
    else:
        print(f"Shard dir {shard_dir} not found.")

if __name__ == "__main__":
    debug()
