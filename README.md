# I.A.R.A. v6.1 (Fractal Attention-CfC)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-ROCm-orange)
![License](https://img.shields.io/badge/License-MIT-green)

**I.A.R.A. v6.1 (Fractal Attention-CfC)** is an advanced algorithmic trading system powered by **I.A.R.A. (Intelligent Algorithmic Recurrent Agent)**. The system utilizes **Liquid Neural Networks (LNN)** with **Closed-form Continuous (CfC)** cells and a novel **Fractal Attention** mechanism to predict short-term currency trends with high robustness against market noise and "fakeouts".

Designed for performance on **AMD RDNA2 GPUs (ROCm)** via Docker.

---

## ⚡ Key Features

### 🧠 I.A.R.A. v6.1 Architecture
A hybrid architecture combining the best of Transformers and Continuous-Time RNNs:
- **Learned Positional Encoding**: Captures precise temporal distances in long sequences (256 steps).
- **Fractal Attention**: Multi-Head Self-Attention (4 heads) coupled with a Feature Gating mechanism to filter irrelevant market noise.
- **CfC Backbone**: Symmetric Closed-form Continuous cells that model hidden states as continuous-time dynamic systems, offering superior stability over vanishing/exploding gradients.
- **Regulated Training**: High dropout (0.5), Gradient Clipping, and Label Smoothing to prevent overconfidence.

### 📊 Advanced Data Engineering
- **Multi-Asset Support**: Seamlessly trains on EURUSD, GBPUSD, USDJPY, AUDUSD, etc.
- **Unified Normalization**: `StandardScaler` fitted on pooled data from all assets to preserve relative volatility magnitudes.
- **15+ Technical Indicators**:
  - **Momentum**: RSI, Stochastic Oscillator (%K, %D).
  - **Trend**: MACD, ADX.
  - **Volatility**: Bollinger Bands (%B, Width), ATR.
  - **Dynamics**: Price Velocity (First derivative).
  - **Time**: Cyclical encoding of Hour and Day.

### 🛠️ Production Ready
- **Dockerized Environment**: Fully encapsulated build with ROCm support for AMD GPUs.
- **Polars Integration**: Ultra-fast data processing pipeline.
- **Inference Suit**: Tools for backtesting and signal generation with confidence metrics.

---

## 🚀 Quick Start

### 1. Prerequisites
- Docker & Docker Compose
- AMD GPU (configured for ROCm) *OR* NVIDIA GPU (adjust Dockerfile base image) *OR* CPU (slow but functional).

### 2. Installation
Clone the repository and build the container:
```bash
git clone https://github.com/your-repo/saci-trader.git
cd saci-trader

# Build the Saci Brain container
docker compose build
```

### 3. Usage
All commands are run inside the Docker container for consistency.

#### Start the Container
```bash
docker compose up -d
```

#### A. Data Processing
Process raw parquet files (Ohlc data) into model-ready features.
```bash
docker compose exec saci-brain python src/data/processor.py
```
*Input: `data/raw/*.parquet` | Output: `data/processed/*.parquet` + `data/scaler.pkl`*

#### B. Training
Train the I.A.R.A. model on processed data.
```bash
docker compose exec saci-brain python src/train/trainer_liquid.py \
    --epochs 200 \
    --batch-size 512 \
    --seq-len 256
```
*checkpoints are saved to `checkpoints/`*

#### C. Inference (Testing)
Evaluate the model's performance on a specific pair.
```bash
docker compose exec saci-brain python src/train/inference_liquid.py \
    --ticker EURUSD \
    --checkpoint best_model.pth \
    --limit 1000
```

---

## 📂 Project Structure

```
saci-trader/
├── checkpoints/       # Saved model weights
├── config/            # Configuration files
├── data/
│   ├── raw/           # Input market data (parquet)
│   └── processed/     # Feature-engineered data
├── docker/            # Docker configuration
├── docs/              # Detailed documentation
├── src/
│   ├── data/          # ETL & Feature Engineering (processor.py)
│   ├── model/         # I.A.R.A. Neural Network (iara_liquid.py)
│   └── train/         # Training & Inference loops
├── requirements.txt   # Python dependencies
└── docker-compose.yml # Service orchestration
```

---

## 🏗️ Technical Details
For a deep dive into the Model Architecture, Feature Logic, and Training Strategy, please read the [Architecture Documentation](docs/ARCHITECTURE.md).

---

## 📝 License
MIT License.
