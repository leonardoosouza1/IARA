# Research Log: GPU Optimization & Model Scaling (V9.7)

## Problem: Low GPU Utilization
Initial benchmarks of the IARA v9 architecture showed VRAM usage at ~1.2GB and GPU load at ~8% on an AMD Radeon RX 6750 XT (12GB). While efficient, this indicated the model was too small to leverage the hardware's parallel processing power effectively.

## Research & Hypothesis
By scaling the internal dimensions of the Swarm Intelligence components, we aimed to:
1. Increase the model's capacity to learn complex market nuances.
2. Maximize data throughput to reduce idle GPU cycles.
3. Improve convergence stability by using larger hidden states in Liquid Neural Networks (CfC).

## Optimization Workloads

### 1. Dimension Scaling
We doubled (and in some cases quadrupled) the hidden dimensions:
- **Workers**: Increased from 32 to 128 units.
- **Coordinator**: Increased from 128 to 512 units.
- **Impact**: Parameter count grew from ~300k to ~2.1M.

### 2. Throughput Maximization
We increased the window of attention and batch processing:
- **Sequence Length**: Increased from 64 to 256 candles (expanding the temporal horizon by 4x).
- **Batch Size**: Increased from 64 to 256 samples.
- **Impact**: VRAM usage increased from ~10% to ~40% (4.5GB+).

## Bypassing Constraints
- **Architecture Mismatch**: Resolved issues where `backtest_v9.py` and `auto_learn_v9.py` had hardcoded smaller dimensions. Implemented dynamic CLI arguments to sync training and evaluation environments.
- **Resource Allocation**: Optimized the Docker `shm_size` and ROCm overrides to ensure stable execution on AMD hardware.
- **Learning Continuity**: Fixed the gating and rollback logic to ensure that scaled models are correctly promoted and compared against previous baselines.

## Results
- **Memory**: Stabilized at ~4.5GB VRAM.
- **Learning**: AUC improved significantly due to the increased capacity (observed Val AUC ~0.74 in initial cycles).
- **Efficiency**: Throughput increased 16x (4x batch * 4x sequence) while maintaining acceptable step times (~0.3s/step).

---
*Status: V9.7 "Gladiator" Scaling Research Concluded.*
