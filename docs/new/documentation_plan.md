# Professional System Architecture Documentation Plan

## Context
Refactoring the IARA trading system to a "Professional Grade" Client-Server architecture.
- **Backend**: Python (FastAPI + Socket.IO) keeping the core Brain logic.
- **Frontend**: Next.js 14 + Tailwind + Shadcn/UI for "God Mode" observability.

## Documentation Structure (`docs/new/`)

### 1. Backend Architecture (`docs/new/backend/`)
- **`api_design.md`**: REST Endpoints, Socket.IO Events (Paylod Specs, Namespaces), Service Layer (Patterns).
- **`security_protocol.md`**: JWT Implementation (Access/Refresh), HttpOnly Cookie strategy, CORS, Rate Limiting, and Input Validation (Pydantic).
- **`database_integration.md`**: Async SQLAlchemy/Asyncpg usage, Migration strategy (Alembic), and TimescaleDB optimization for high-frequency data.

### 2. Frontend Architecture (`docs/new/frontend/`)
- **`nextjs_structure.md`**: App Router layout, Server vs Client Components strategy, Middleware protection.
- **`state_management.md`**: Zustand (Client Global State) + React Query (Server State) strategy. Socket.IO integration pattern (`useSocket` hook).
- **`ui_ux_design.md`**: Design System (Tokens, Colors, Typography), Component Library (Shadcn), "God Mode" visualization specs (Charts, Terminal).

### 3. System Flows (`docs/new/`)
- **`data_flows.md`**: Sequence diagrams for:
  - Authentication (Login -> Cookie -> Socket Handshake).
  - Live Market Data (Alpaca -> FleetAdmiral -> Socket -> Frontend Chart).
  - Active Learning (Trade Error -> Snapshot -> DB).

## Key Design Decisions
1.  **Zero-Latency Bridge**: The `FleetAdmiral` will broadcast internal `BrainState` directly to the Socket.IO `BroadcastManager`, bypassing the database for live UI updates (Database is for persistence/history only).
2.  **Secure by Default**: No LocalStorage for tokens. Strict Content Security Policy (CSP).
3.  **Resilient Sockets**: Auto-reconnect logic with exponential backoff and "state reconciliation" upon reconnection (fetching missed candles).

---
## Implementation Roadmap reference
(See `task.md` Phase 6 & 7)
