from src.train.continuous_learner import ContinuousLearner
from src.brain.models.liquid_v9 import SwarmBrain, CoordinatorConfig
from src.brain.models.workers import WorkerConfig
import torch

learner = ContinuousLearner()
print(f"Learner worker_hidden: {learner.worker_hidden}")
print(f"Learner coord_hidden: {learner.coord_hidden}")

model = learner.model
print(f"Model worker_dim in coordinator: {model.coordinator.config.worker_dim}")
print(f"Model hidden_dim in coordinator: {model.coordinator.config.hidden_dim}")

# Check actual layer shapes
fusion_weight_shape = model.coordinator.fusion.fusion[0].weight.shape
print(f"Fusion weight shape: {fusion_weight_shape}")
