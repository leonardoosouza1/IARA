# Database Integration Strategy (TimescaleDB)

## Overview
The Brain needs persistence for:
1.  **Trade History**: Full audit trail of decisions.
2.  **Telemetry Snapshots**: High-frequency state recording for "God Mode" replay.
3.  **Active Learning**: Storing "Mistakes" (Input Tensor + Outcome) for retraining.

We use **PostgreSQL with TimescaleDB extension** via **Async SQLAlchemy** (`asyncpg` driver) for non-blocking I/O.

## 1. Schema Design

### `agents` Table
Identity of the IAs.
- `id`: UUID (Primary Key)
- `symbol`: String (BTC/USD)
- `market_type`: Enum (Crypto, Forex)
- `config`: JSONB (Strategy parameters)
- `status`: Enum (Active, Paused, Dead)
- `created_at`: Types.DateTime

### `trade_history` Table (Hypertable)
Partitioned by time.
- `time`: DateTime (PK)
- `agent_id`: UUID (FK)
- `side`: Enum (Buy, Sell)
- `price`: Decimal
- `size`: Decimal
- `pnl`: Decimal (Nullable, filled on close)
- `exit_price`: Decimal (Nullable)
- `reason`: Text (LFM reasoning summary)

### `brain_telemetry` Table (Hypertable)
High-frequency dumps of internal state (for Replay/Debug).
- `time`: DateTime (PK)
- `agent_id`: UUID (FK)
- `worker_votes`: JSONB (e.g., `{"M5": 0.8, "H1": 0.2}`)
- `consensus`: Float (0-1)
- `lfm_thought`: Text (Stream of consciousness)

### `learning_samples` Table
For Active Learning loop.
- `id`: UUID
- `input_snapshot`: ByteA (Serialized Tensor)
- `outcome_label`: Integer (0=Bad, 1=Good)
- `confidence`: Float

## 2. Async Implementation (`src/api/db`)

### Session Management
We use a singleton `AsyncSession` dependency for FastAPI routes.
```python
# src/api/dependencies.py
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with async_session_factory() as session:
        yield session
```

### High-Performance Ingestion
For generic telemetry (hundreds of inserts/sec):
- Use `COPY` command or batch inserts via `asyncpg` directly, bypassing ORM overhead for telemetry tables.
- ORM is reserved for `agents` config and critical `trade_history`.

## 3. Migration Strategy (Alembic)
- `alembic` configured for async migrations.
- `env.py` updated to use `async_engine`.
