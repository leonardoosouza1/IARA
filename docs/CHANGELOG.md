# Changelog - SACI Project

## [V7 - MLOps Era] - Thursday, Feb 5 2026 (Current)
### Infrastructure (Planned)
- **SACI Protocol V7**: MLOps Architecture Pivot.
    - **TimescaleDB**: PostgreSQL specialized for Tick Data.
    - **MLflow**: Professional Experiment Tracking.
    - **MinIO**: Local "S3" for Artifact Storage.

### Implemented
- **Trade Hub**: Simulated Trading Engine (`src/trading/simulation.py`) and Real-Time Dashboard Tab.
- **Neuroscope**: Deep Explainability (Logic Log, Macro vs Micro Trend Viz).
- **Fixes**: Dashboard 18-feature vector alignment.

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
