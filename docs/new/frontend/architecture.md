# Frontend Architecture (Next.js 14)

## Overview
A "Mission Control" dashboard built with performance and observability as first principles.
Uses **Next.js 14 App Router** for hybrid rendering (Server Components for layout/auth, Client Components for live charts).

## Technology Stack
- **Framework**: Next.js 14
- **Language**: TypeScript (Strict Mode)
- **Styling**: Tailwind CSS + `clsx` + `tailwind-merge`
- **UI Library**: Shadcn/UI (Radix Primitives)
- **State**: Zustand (Global), React Query (Async/Server)
- **Socket**: `socket.io-client` with custom hooks.

---

## 1. Directory Structure (`src/web`)
```
src/web/
├── app/
│   ├── (auth)/        # Route Group for auth pages
│   │   └── login/
│   ├── (dashboard)/   # Protected Layout
│   │   ├── layout.tsx # Sidebar + Socket Provider
│   │   ├── page.tsx   # Fleet Overview
│   │   └── god-mode/
│   │       └── [id]/  # Dynamic Agent View
│   └── layout.tsx     # Root Layout (Fonts, Theory)
├── components/
│   ├── ui/            # Shadcn Primitives (Button, Card...)
│   ├── fleet/         # Business Components (AgentCard, BrainMRI)
│   └── charts/        # Lightweight-charts wrappers
├── lib/
│   ├── socket.ts      # Singleton Socket Client
│   ├── api.ts         # Axios/Fetch wrapper
│   └── utils.ts
├── hooks/
│   ├── use-socket.ts  # Subscription Hook
│   └── use-fleet.ts   # Zustand Store
└── types/             # TS Interfaces
```

## 2. State Management Strategy

### Server State (React Query)
Used for data that changes slowly or needs initial fetch.
- `useQuery(['fleet-status'])`: Fetches active agents list.
- `useQuery(['market-history', symbol])`: Fetches initial candles.

### Client State (Zustand)
Used for high-frequency ephemeral data.
- **`useFleetStore`**:
  - `agents`: Map of AgentID -> Data
  - `socketStatus`: Connected/Disconnected
  - `addTick(symbol, tick)`: Appends tick to local history.

### Real-time Bridge (`useSocket`)
Custom hook to prevent listeners duplication.
```typescript
useSocketEvent('market_tick', (data) => {
  useFleetStore.getState().addTick(data.symbol, data);
});
```

## 3. "God Mode" UI Components

### `RealTimeChart.tsx`
- Wraps `lightweight-charts`.
- Uses `useRef` to maintain chart instance.
- **Performance**: Updates via `series.update(tick)` direct API, NO React re-renders for data updates.

### `BrainMRI.tsx`
- Radar Chart (Recharts) showing Worker votes.
- Updates every second via `worker_votes` event.

### `ThoughtStream.tsx`
- Terminal-like scrolling log.
- Displays `brain_thought` events with syntax highlighting.

---

## 4. Design System (Dark Theme)
- **Background**: `bg-zinc-950` (Deep space).
- **Cards**: `bg-zinc-900/50` + `backdrop-blur` (Glassmorphism).
- **Accents**: 
  - Buy/Long: `text-emerald-400`
  - Sell/Short: `text-rose-400`
  - Neutral/Thinking: `text-blue-400`
- **Typography**: `Inter` (UI), `JetBrains Mono` (Data/Code).
