# I.A.R.A. v8.5 Trinity Engine Architecture

## 1. Overview: The Tri-Cameral Brain
I.A.R.A. v8.5 "Trinity" represents a paradigm shift from a monolithic model to a **Tri-Cameral Architecture**, inspired by the breakdown of human cognition into **System 1 (Fast)**, **System 2 (Slow)**, and a **Reflexive Guard (Sentinel)**.

This design addresses the core limitation of previous versions: *The inability to stop trading during black swan events or regime shifts.*

---

## 2. System Architecture

### ⚡ System 1: The Fast Brain (Liquid Neural Network)
Responsible for millisecond-level execution. It is always "on" and looking for patterns.
*   **Core**: `ModulatedIARA` (LTC/CfC Backbone).
*   **Input**: 256-step Time Series (Technical Indicators).
*   **Enhancements**:
    *   **Fractal Attention**: Focuses on relevant historical windows.
    *   **Expert Mixture**: Two specialized "Experts" (Linear Layers) trained for Long and Short biases respectively.
    *   **Feature Gating**: A learnable attention mechanism that assigns importance to each input feature (e.g., ignoring RSI when Volume is low).

### 🛡️ System 2: The Sentinel (Watchdog)
A reflexive, rule-based inhibition system that overrides System 1.
*   **Role**: The "Amygdala" – instantaneous fear response.
*   **Triggers**:
    *   **News DEFCON**: Keyword scanning in real-time news feeds (War, Fed Rates, Disaster).
    *   **Volatility Spikes**: VIX or ATR explosions.
*   **Mechanism**: **Gradient Interlock**. It doesn't just "filter" trades; it physically zeroes out the output logits of the Neural Network, preventing the signal from ever reaching the execution layer.

### 🧠 System 3: The Strategist (Slow System)
A Large Language Model (LLM) operating asynchronously.
*   **Core**: Quantized LLM (DeepSeek/Llama via `llama-cpp-python`).
*   **Role**: Market Analyst.
*   **Function**:
    *   Reads macro reports and news summaries.
    *   Sets a global **"Regime Bias"** (e.g., `Bearish`).
    *   Modulates the Fast System's priors (e.g., "Harder to buy, easier to sell").

---

## 3. Data Flow & Tensor Shape

### Input Pipeline ($B, 256, 17$)
1.  **Market Data ($15$)**: OHLCV derived indicators (RSI, MACD, Bollinger, etc.).
2.  **Context Embeddings ($2$)**:
    *   `news_sentiment`: Float from System 3.
    *   `regime_code`: Categorical embedding of the market state.

### The Forward Pass
1.  **Feature Gate**: $X' = X \odot \sigma(W_g X)$
    *   *Filters out noise before it enters the expensive RNN.*
2.  **Liquid Backbone**: $H_t = \text{CfC}(X'_t, H_{t-1})$
    *   *Solves the continuous-time hidden state.*
3.  **Expert Heads**:
    *   $E_{long} = \text{Linear}(H_t)$
    *   $E_{short} = \text{Linear}(H_t)$
4.  **Conflict Resolution**: $Y = E_{long} - E_{short}$
    *   *Buying requires high Long conviction AND low Short conviction.*

---

## 4. Operational Dashboard (Mission Control v2.0)

The interface is no longer just a log viewer; it is an active component of the feedback loop.

### 🔬 Forensic Lab
Allows the human operator to inspect the "Brain's" internal state.
*   **Tick Slider**: Replay the 256-step inference window.
*   **Heatmaps**:
    *   **Feature Importance**: See which indicator triggered the trade.
    *   **Neural Activity**: Visualize CfC hidden state saturation.
*   **Conviction Plot**: Visual proof of *why* a trade was taken (Long/Short split).

### 🎛️ Ops Panel
*   **DEFCON Gauge**: Real-time visual of the Sentinel's status.
*   **Manual Override**: Physical "Kill Switch" to disconnect the execution engine.

---

## 5. MLOps Infrastructure

*   **Training**: PyTorch Lightning loop logic (custom implemented) with MLflow logging.
*   **Storage**:
    *   **TimescaleDB**: Stores "Brain Snapshots" (Tensors) for replay.
    *   **MinIO**: Stores model weights (`.pth`) and optimized ONNX exports.
*   **Telemetry**: Real-time CSV logging for the Dashboard Streamlit app.

---
*Architecture definition locked for V8.5 Protocol Trinity.*
