# I.A.R.A. v7.0 Architecture Documentation

## 1. Overview
Input-Aware Recurrent Agent (I.A.R.A.) v7.0 represents a significant evolution in time-series forecasting for high-frequency forex markets. Supported on a robust MLOps infrastructure (SACI Protocol V7), I.A.R.A. leverages **Liquid Time-Constant (LTC)** theory via **Closed-form Continuous (CfC)** cells to model hidden state dynamics.

## 2. Model Architecture
The model pipeline processes a sequence of market data ($T=256$) to predict the probability of a positive return in the near future.

### Data Flow
1.  **Input ($B, 256, 15$)**: Batch of 256 time-steps with 15 engineered features.
2.  **Projection**: Linear map to Hidden Dimension ($d=512$).
3.  **Learned Positional Encoding**: Adds learnable vectors to embed temporal order. Crucial for differentiation between "immediate past" and "distant past" in the 256-step window.
4.  **Temporal Self-Attention**:
    - 4-Head Attention Mechanism.
    - Captures long-range dependencies (e.g., support/resistance levels formed 3 hours ago).
    - *Optimization*: Feed-Forward network removed in v6.1 Lite for speed.
5.  **Feature Attention Gate**:
    - A learnable Gating Unit (Sigmoid) that suppresses irrelevant features dynamically per time-step.
    - Functions as a "noise filter" for indicators ensuring only valid signals pass through.
6.  **CfC Recurrent Backbone**:
    - **Closed-form Continuous Cell**.
    - Solves the differential equation of a liquid neural network in closed form.
    - **Symmetric tanh gating**: Ensures gradient flow remains stable even over 256 steps without vanishing gradients.
    - Clipping: Hidden states clipped to $[-1, 1]$ for stability.
7.  **Head**:
    - Layer Norm -> Dropout (0.5) -> Linear -> Output Logits.

## 3. Data Engineering Strategy

### Feature Set (15 Dimensions)
We use a robust set of indicators to capture multiple market facets.

| Category | Indicators | Purpose |
| :--- | :--- | :--- |
| **Momentum** | **RSI** (14) | Overbought/Oversold detection. |
| | **Stoch %K, %D** (14, 3) | **Critical** for v6.0. Filters fake breakouts/divergences. |
| **Trend** | **MACD** (Line, Signal, Hist) | Trend following and reversal strength. |
| | **ADX** (14) | Trend strength (distinguishes chop vs. trend). |
| **Volatility** | **Bollinger Bands** (%B, Width) | Volatility expansion/contraction (squeeze). |
| | **ATR** (14) | Absolute volatility measurement. |
| **Dynamics** | **Price Velocity** | rate-of-change (1st, derivative). |
| **Time** | **Sin/Cos encoding** | Hour of day & Day of week cyclic features. |

### Unified Normalization
Instead of scaling each asset individually, we use a **Global StandardScaler**:
- Fitted on **pooled** data from all major pairs.
- Preserves the *relative* volatility between assets (e.g., GBPUSD is naturally more volatile than EURUSD; individual scaling would destroy this signal).
- State is saved to `data/scaler.pkl` for consistent inference.

## 4. Training Methodology

### Objective
Binary Classification: **Will the price close higher $N$ steps from now?** (Log Return > 0).

### Key Components
- **Optimizer**: AdamW (Weight Decay 1e-4) for regularization.
- **Scheduler**: `ReduceLROnPlateau` (Halves LR if Val Loss stalls for 5 epochs).
- **Loss Function**: `BCEWithLogitsLoss` with `pos_weight`.
    - Automatically calculates class imbalance (Up vs Down) in the batch and adjusts weights to prevent bias towards the majority class.
- **Regularization**:
    - **Dropout 0.5**: Heavy dropout in the head to force distributed feature representation.
    - **Label Smoothing 0.1**: Targets are $0.1 / 0.9$ instead of $0 / 1$. Prevents the model from becoming "overconfident" on noisy data.
    - **Noise Injection**: Gaussian noise ($std=0.02$) added to input features during training to simulate market imperfections.

### Chronological Validation
We STRICTLY avoid random splitting.
- **Train**: First 80% of time-sorted data.
- **Val**: Last 20% of time-sorted data.
- This ensures the model is predicting the *future* and not interpolating missing points from the past, providing a realistic OOS (Out-of-Sample) metric.

## 5. MLOps Infrastructure (SACI Protocol V7)

The system operates on a containerized microservices architecture coordinated by `docker-compose-mlops.yml`.

### Components
1.  **SACI Brain (Container)**:
    - Runtime: Python 3.10+ (ROCm/CUDA).
    - Role: Training, Inference, and Simulation.
    - Integration: Logs metrics to MLflow, writes states to TimescaleDB.

2.  **TimescaleDB (Database)**:
    - Role: Primary persistent storage.
    - Tables: `market_candles` (Ohlcv), `brain_states` (Inference logs), `trades`.
    - Features: Hypertable partitioning for time-series efficiency.

3.  **MLflow (Tracking Server)**:
    - Role: Experimentation Lab.
    - Tracking: Hyperparameters (`seq_len`, `lr`), Metrics (`AUC`, `Loss`), Artifacts (Models).

4.  **MinIO (Object Storage)**:
    - Role: Artifact Vault.
    - Stores: Heavy binaries (Model `.pt` files, Parquet datasets) for MLflow.

5.  **Dashboard (Viz)**:
    - Role: User Interface.
    - Connectivity: Reads real-time simulation data from TimescaleDB (Port 8501).
