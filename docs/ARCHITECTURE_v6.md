# I.A.R.A. v6.2 Architecture (Agile & Adaptive)

## 🧠 The "Agile" Philosophy
Version 6.2 shifts from "Brute Force" (Large Models) to "Agile Intelligence" (Lean Models + Smarter Data).

### 1. Model Compression (The scalpel)
- **Hidden Dimension**: Reduced from 512 to **64**.
- **Why?** Financial time-series are noisy. Large models tends to memorize noise (overfitting). A smaller CfC backbone forces the network to learn only the most robust, repeatable fractal patterns.
- **Result**: Faster convergence and higher validation AUC (0.60+).

### 2. High-Density Training Data
- **Stride Strategy**: Reduced from 8 to **2**.
- **Impact**: We now train on 97% of all available candles, rather than a 12% sample. This provides the volume needed for a small model to generalize effectively.

---

## 🏋️ Advanced Training Dynamics

The `Trainer` class has been upgraded with three SOTA mechanisms:

### A. Elephant Memory (Historical Hard Mining)
The trainer maintains a persistent memory of every sample's difficulty across epochs.
- **Formula**: `Memory = (0.7 * Old) + (0.3 * Current_Loss)`
- **Behavior**:
  - If the model gets a sample right *once* by luck, the memory keeps the weight high ("Prove it again").
  - If the model consistently fails, the weight skyrockets (Curriculum Learning).

### B. Auto-Rollback (Safety Net)
Prevents "Catastrophic Forgetting" during training.
- **Trigger**: When the Learning Rate Scheduler detects a plateau (no improvement for 5 epochs).
- **Action**: The trainer **automatically reloads** the absolute best model weights (`best_model.pth`) from RAM.
- **Effect**: Training resumes from the peak, not from a degraded state. This allows for essentially infinite training without regression.

### C. Fine-Tuning Capability
Allows "polishing" a pre-trained model.
- **CLI**: `--lr` argument allows reducing learning rate (e.g., `1e-5`).
- **Workflow**: `cp best_model.pth latest.pth` -> Resume with low LR.

---

## 🔮 Active Learning Inference

The signal generator is no longer static. It adapts to market regimes.

### Dynamic Thresholding
- **Bull Run (Easy)**: Threshold drops to **0.55** (Take more trades).
- **Bear Trap (Hard)**: Threshold rises to **0.70** (Defensive mode).
- **Mechanism**: The `ActiveLearner` module tracks the last 50 virtual trades. If accuracy drops, it tightens the criteria automatically.

---

## 📊 Performance Checkpoints
- **v6.1 (Legacy)**: AUC 0.50 (Stagnant).
- **v6.2 (Agile)**: AUC 0.60+ (Profitable Edge).
- **Deployment**: Confirmed on EURUSD M5.

---

## 🛡️ Robustness Features (The Shield)

### Data Augmentation
To prevent the model from memorizing specific price sequences, we inject controlled chaos during training:
- **Gaussian Noise**: Adds random jitter (`std=0.02`) to features.
- **Random Scaling**: Scales the entire sequence amplitude by `0.98x` to `1.02x`.
*Idea: "If you can recognize the pattern even when it's fuzzy, you truly understand it."*

### Class Balancing
Market data is often unbalanced.
- **Mechanism**: We calculate `pos_weight = Negatives / Positives` dynamically.
- **Effect**: The Loss Function penalizes missing a rare trade more than hitting a common noise event.

---

## 🧬 Neural Core (The Brain)
The underlying engine remains the robust **Liquid Neural Network**:
- **Inputs**: 15 normalized features -> Projected to 64 dims.
- **Fractal Window**: 4-Head Linear Attention (Captures multi-scale dependencies).
- **Liquid Body**: Closed-form Continuous (CfC) RNN.
  - *Why?* Solves the vanishing gradient problem in long sequences (256 steps).
  - *Behavior*: Adapts its time-constant based on input, modeling "Liquid" time.
- **Output**: Binary Classification (Sigmoid).

---

## ⚡ Hardware Optimizations
- **Mixed Precision (AMP)**: Uses float16 for math but float32 for master weights. 2x speedup.
- **Gradient Accumulation**: Simulates massive batches (1024) even on smaller VRAM.
- **Pin Memory**: Accelerates CPU->GPU data transfer.
