# I.A.R.A. v8.5 "Trinity Engine" (Protocol Trinity)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-ROCm-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)

**I.A.R.A. "Trinity Engine" (Protocol Trinity)** is the next evolution of the SACI trading system. It moves beyond a single model to a **Tri-Cameral Architecture** inspired by the human brain (System 1 vs System 2) and MLOps best practices.

It features a **Liquid Neural Network (Fast System)** for instant pattern recognition, a **Watchdog Sentinel (Reflex System)** for risk gating, and a **Slow System Interface** (LLM) for strategic oversight.

---

## 🧠 The Trinity Architecture

The engine is composed of three distinct but integrated systems:

1.  **Fast System (The Liquid Brain)**:
    *   **Core**: `ModulatedIARA` (LNN + CfC + Feature Attention).
    *   **Role**: Processes market data tick-by-tick (milliseconds).
    *   **Features**: Time-Decay Memory, Fractal Attention, Expert Mixture (Long/Short Specialists).

2.  **Sentinel System (The Watchdog)**:
    *   **Core**: Rule-based & News-based Gating.
    *   **Role**: Blocks trade execution during "DEFCON 1" events (e.g., Breaking News, Extreme Volatility).
    *   **Mechanism**: Physically suppresses neural gradients and output logits when triggered.

3.  **Slow System (The Strategist)**:
    *   **Core**: LLM Integration (DeepSeek/Llama via `llama-cpp-python`).
    *   **Role**: Analyzes macroeconomic context and adjusts the Fast System's bias (e.g., "Bearish Regime").

---

## 🖥️ Mission Control v2.0 (Dashboard)

A professional grade Streamlit dashboard (`src/dashboard/`) replaces the old CLI logs.

### 1. 🚀 Training Lab Pro
*   **Live Monitor**: Real-time plots of Loss, Accuracy, and Learning Rate.
*   **Forensic Lab (Time Travel)**:
    *   **Brain MRI**: Visualize hidden state activations (`epoch_*.json`).
    *   **Decision Architecture**: See exactly *why* the model bought/sold (Long Expert vs Short Expert conviction).
    *   **Feature Heatmap**: Which indicators (RSI, Price, Volatility) triggered the decision?

### 2. 🎛️ Mission Control (Ops)
*   **DEFCON Gauge**: Real-time visualization of the Sentinel's threat level.
*   **Neural Override**: Manual "Kill Switch" to disable the AI during emergencies.
*   **News Feed**: Live streaming of financial headlines affecting the Sentinel.

---

## ⚡ Quick Start

### 1. Prerequisites
*   Docker & Docker Compose
*   AMD GPU (ROCm) *OR* NVIDIA GPU (CUDA)

### 2. Launch the Trinity Stack
The entire system (Training, Dashboard, Database, MLflow) runs in a unified Mesh.

```bash
# Start the MLOps Infrastructure
docker compose -f docker-compose-mlops.yml up -d
```

### 3. Access the Dashboard
Open **[http://localhost:8501](http://localhost:8501)** in your browser.

*   **Go to "Training Lab"**: Configure hyperparams (Batch Size 1024 supported) and click **"Launch Training"**.
*   **Watch in Real-Time**: Metrics update live. Use the **Forensic Lab** tab to inspect "Brain Snapshots" from completed epochs.

---

## 📂 Project Structure

```
saci-trader/
├── src/
│   ├── brain/                 # The Trinity Engine Core
│   │   ├── engine.py          # Orchestrator
│   │   ├── train_v8.py        # Forensic Training Loop
│   │   ├── models/            # Neural Architectures (trinity_v8.py, liquid_v8.py)
│   │   └── agents/            # Sentinel & Slow System Stubs
│   ├── dashboard/             # Streamlit App (Mission Control)
│   │   ├── app.py             # Router
│   │   └── pages/             # Interface Modules (training.py, mission_control.py)
│   └── data/                  # Feature Engineering
├── docker-compose-mlops.yml   # Production Orchestration
├── logs/                      # Telemetry & Snapshots (JSON/CSV)
└── docs/                      # Documentation
```

---

## 🛠️ Forensics & Debugging

If the Dashboard acts up:
1.  **Clear Snapshots**: `rm logs/snapshots/*.json` (Fixes dimension mismatch from old runs).
2.  **Restart Container**: `docker restart saci-brain` to force code reload (if hot-reload misses something).
3.  **Check Logs**: `docker logs -f saci-brain`.

---
*Maintained by the SACI Team.*
