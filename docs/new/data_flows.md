# System Data Flows

## 1. Authentication Flow (Login)
```mermaid
sequenceDiagram
    participant User
    participant NextJS (Client)
    participant FastAPI (Auth)
    participant DB

    User->>NextJS: Enter Credentials
    NextJS->>FastAPI: POST /api/auth/login
    FastAPI->>DB: Validate User
    DB-->>FastAPI: User Valid
    FastAPI-->>NextJS: Set-Cookie: access_token (HttpOnly)
    NextJS->>User: Redirect to /dashboard
```

## 2. Real-time Market Data Flow
From Alpaca to the Chart (Zero Latency path).
```mermaid
sequenceDiagram
    participant Alpaca (WebSocket)
    participant MarketStream (Python Service)
    participant SocketManager (FastAPI)
    participant NextJS (Chart Component)

    Alpaca->>MarketStream: Trade Update (BTC @ 50k)
    MarketStream->>MarketStream: Normalize Data
    MarketStream->>SocketManager: emit('market_tick', payload)
    SocketManager-->>NextJS: Broadcast('market_tick')
    NextJS->>NextJS: Update Chart Series (Directly)
```

## 3. "The Brain" Decision Flow
How the FleetAdmiral broadcasts thoughts to God Mode.
```mermaid
sequenceDiagram
    participant Agent (TrinityEngine)
    participant LFM (Ollama)
    participant SocketManager
    participant NextJS (God Mode)

    Agent->>LFM: Query("Analyze Trend")
    LFM-->>Agent: "Trend is Bullish because..."
    Agent->>SocketManager: emit('brain_thought', text)
    SocketManager-->>NextJS: Broadcast('brain_thought')
    NextJS->>NextJS: Append to Terminal Log
```

## 4. Active Learning Loop
Self-correction mechanism.
```mermaid
sequenceDiagram
    participant Agent
    participant TradeMonitor
    participant DB (LearningSamples)

    Agent->>TradeMonitor: Open Trade (Snapshot Input)
    TradeMonitor->>TradeMonitor: Wait for Close
    TradeMonitor->>Agent: Trade Closed (PnL = -50$)
    Agent->>DB: Insert into learning_samples (Input, Label=0)
    Note over DB: Next Training Cycle will load this sample
```
