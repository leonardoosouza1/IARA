# Backend API Architecture (FastAPI + Socket.IO)

## Overview
The backend transforms from a monolithic Streamlit script to a headless **FastAPI** service. It serves:
1.  **REST API**: Configuration, historical data, authentication.
2.  **Socket.IO Server**: Real-time bidirectional communication for live trading.

## Technology Stack
- **Framework**: FastAPI (high performance, async).
- **Real-time**: `python-socketio` (ASGI mode).
- **Validation**: Pydantic v2.
- **Database**: Async SQLAlchemy + asyncpg (TimescaleDB).
- **Task Queue**: Internal `asyncio` tasks for FleetAdmiral (no Celery needed yet).

---

## 1. REST API Endpoints (`/api/v1`)

### Authentication
- `POST /auth/login`: Returns `HttpOnly` cookie with JWT `access_token`.
- `POST /auth/logout`: Clears cookies.
- `GET /auth/me`: Returns current user user info.

### Fleet Management
- `GET /fleet/status`: Returns active agents and their health.
- `POST /fleet/start`: Spawns a new agent for a symbol.
- `POST /fleet/stop`: Halts an agent (graceful shutdown).
- `POST /fleet/killswitch`: Emergency stop all trading.

### Market Data (Historical)
- `GET /market/history/{symbol}?timeframe={tf}&limit={n}`: Returns OHLCV JSON for charting.
- `GET /market/snapshot/{symbol}`: Latest full book state.

### Brain Internals
- `GET /brain/zoo`: List available trained models.
- `GET /brain/active`: Get currently deployed model metadata.

---

## 2. Socket.IO Events (`/ws`)

### Namespaces
- `/live`: Public market data (ticks, candles).
- `/fleet`: Protected fleet events (brain thoughts, signals).

### Events (Server -> Client)
| Event Name | Namespace | Payload | Frequency |
| :--- | :--- | :--- | :--- |
| `market_tick` | `/live` | `{symbol, price, time}` | ~100ms |
| `candle_closed` | `/live` | `{symbol, o, h, l, c, v, time}` | Timeframe end |
| `brain_thought` | `/fleet` | `{agent_id, step, log_text}` | Real-time |
| `worker_votes` | `/fleet` | `{agent_id, M5_vote, H1_vote, ...}` | Per tick |
| `trade_update` | `/fleet` | `{agent_id, type, price, pnl}` | On exec |
| `fleet_heartbeat` | `/fleet` | `{active_agents_count, uptime}` | 5s |

### Events (Client -> Server)
| Event Name | Namespace | Payload | Action |
| :--- | :--- | :--- | :--- |
| `subscribe` | `/live` | `{symbol}` | Join room |
| `manual_override` | `/fleet` | `{agent_id, action: BUY/SELL}` | Force trade |

---

## 3. Project Structure (`src/api`)
```
src/api/
├── main.py            # ASGI Entrypoint
├── config.py          # Envs & Constants
├── dependencies.py    # DI (Db session, Current User)
├── events.py          # Socket.IO Event Handlers
├── auth/
│   ├── jwt.py         # Token logic
│   └── router.py      # Auth routes
├── fleet/
│   ├── router.py      # Fleet control routes
│   └── service.py     # FleetAdmiral Bridge
├── market/
│   └── router.py      # Historical data
└── schemas/           # Pydantic models
    ├── fleet.py
    └── market.py
```
