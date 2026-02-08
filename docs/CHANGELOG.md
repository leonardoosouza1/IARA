# Changelog - SACI Project

## [V9.6 - Stability & Curriculum] - Sunday, Feb 8 2026 (Current)
### Curriculum Stabilization
- **Selective Memory v9.6**: 
    - **Neutral Initialization**: Starting memory at 0.6 Loss (Neutral) instead of 1.0 (Panic) to prevent extreme bias.
    - **Diversity Mix**: Guaranteed **10% random background** in every batch to prevent catastrophic forgetting.
    - **Coverage Boost**: Analysis coverage increased to **60,000 samples** per epoch (3x speedup).
    - **Softer Weights**: Reduced selection pressure from `^2.0` to `^1.5` for smoother convergence.
- **UI & Visualization**:
    - **Atomic Telemetry**: Fixed dashboard flickering by implementing atomic file write/replace operations.
    - **Fixed Chart Scaling**: Locked Training Dynamics axis (Loss/AUC) to [0, 1] for intuitive visual monitoring.

## [V9.5 - Self-Correcting Brain] - Saturday, Feb 7 2026
### Autonomous Resilience
- **Autonomous Rollback**: Brain can now sense "Confusion" (Drop > 5% in AUC) and auto-revert to the previous best state.
- **Dynamic Speed Reduction**: Automatically reduces Learning Rate by **20%** after a rollback to approach hard patterns more carefully.
- **Persistence Fix**: Ensured human-captured feedback persists across the entire training mission.

## [V9.0 - Swarm Intelligence] - Friday, Feb 7 2026
### The "Fractal Consensus" Architecture
- **Swarm Intelligence**: Replaced monolithic model with **5 Specialized Workers** (M5, M15, M30, H1, H4).
- **LightningWorker**: Ultra-light CfC cells (~32 units each) for each timeframe.
- **SwarmCoordinator**: Cross-Attention fusion + CfC refinement for final decision.
- **New Loss Function (SwarmLoss)**:
    - `AsymmetricFocalLoss` for Coordinator
    - `WorkerAuxiliaryLoss` forces each worker to predict independently
    - `ConsensusLoss` penalizes Coordinator contradicting strong worker consensus
- **Training Improvements**:
    - Ported `SelectiveMemory` curriculum learning from v6
    - Gradient Clipping: 0.5 (crucial for LNN stability)
    - Label Smoothing: 0.1
- **VRAM Estimate**: ~210-370MB (fits 12GB GPU easily)

### Files Created
- `src/brain/models/workers.py`: LightningWorker, WorkerSwarm
- `src/brain/models/liquid_v9.py`: SwarmBrain, SwarmCoordinator  
- `src/brain/losses.py`: SwarmLoss with 3 component losses
- `src/brain/train_v9.py`: Complete training loop with AMP

---

## [V7 - MLOps Era] - Thursday, Feb 5 2026
### Implemented
- **SACI Protocol V7**: Full MLOps Architecture.
    - **TimescaleDB**: Integrated for BrainState persistence (`src/trading/simulation.py`).
    - **MLflow**: Connected for experiment tracking (`SACI_Operations_V1`).
    - **MinIO**: Artifact storage configuration verified.
    - **Dashboard**: Connected to TimescaleDB for "Trade Hub" replay.
- **Trade Hub**: Automated Simulation (`src/run_simulation.py`) and Real-Time Dashboard.
- **Fixes**: MLflow 403 Security, Docker Dependency Injection, Dashboard DB Grid.

---

## [V6 - Liquid Era] - Wednesday, Feb 4 2026
### v6.11 - The "Golden" Dataset
- **Hierarchical Curation**: Implemented M1 -> M5 -> H1 trend consensus filtering.
- **Impact**: Reduced noise, balanced classes (50/50), AUC broke the 0.50 barrier.

### v6.9 - The "Hybrid" Engine
- **Crisis**: ROCm HIPBLAS errors during Data Loading.
- **Solution**: Feature Factory (CPU) + RAM Mapping.
- **Result**: 2x speedup (33ms -> 15ms/batch) and stability.

### v6.3 - Biological Learning
- **Selective Memory**: "Focus on Hard Examples" (Focal Loss).
- **Arrogant Error Penalty**: Feedback loop for high-confidence mistakes.

### v6.0 - The Initial Shift
- **Architecture**: Pivot to **Liquid Neural Networks (LNN)**.
- **Reason**: Transformers were too heavy; LSTMs vanished. Liquid (LTC/CfC) handles irregular series naturally.
- **Fractal Stride**: Expanded receptive field from 4h to 2 days efficiently.

---

## [V1 ~ V5 - The "Genesis" (MVP)] - Tuesday, Feb 3 2026
### v4.0 - Stabilization
- **Problem**: Mode Collapse (Predicting only "HOLD").
- **Fix**: Input Noise Injection + Wide Layers (512 units). Stable convergence achieved.

### v1.0 ~ v3.0 - Validation & Failures
- **Tests**:
    - **Bitnet**: Tested 1.58-bit quantization (failed to capture nuance).
    - **Vanilla LSTMs**: Failed due to exploding gradients.
    - **Transformers**: High compute, low signal (Overfitting).
- **Conclusion**: Financial data requires specialized architectures, not just "bigger" models.

## [V8.5 - Protocol Trinity] - Friday, Feb 6 2026
### The "Consciousness" Upgrade
- **Trinity Engine**: Replaced monolithic model with **Tri-Cameral Architecture** (Fast/Sentinel/Slow).
- **Dashboard v2.0**:
    - **Forensic Lab**: Added "Time Travel" slider to replay model states tick-by-tick.
    - **Brain MRI**: Visualizations for Feature Gating, Expert Conviction, and Neural Heatmaps.
    - **Hot Reload**: Fixed Python caching issues for instant UI updates.
- **Forensic Recorder**:
    - Implemented  to capture full-sequence hidden states ().
    - Fixed  by ensuring snapshot dimensions match the UI slider (64 ticks).
- **Infrastructure**:
    - **Defensive Coding**: Added robust type/length checks to validade legacy snapshots vs new ones.
    - **Batch Size**: Increased limits to 1024 for high-end GPUs.
