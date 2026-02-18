# User Interface & Design System

## Design Philosophy
"NASA Control Room meets Cyberpunk".
- **Density**: High data density, minimized whitespace.
- **Contrast**: High contrast for readability in dark environments.
- **Feedback**: Instant visual feedback for all actions (Toast, Pulse, color change).

## 1. Color Palette (Tailwind Config)

| Token | Color | Hex | Usage |
| :--- | :--- | :--- | :--- |
| `background` | Zinc 950 | `#09090b` | Main Page BG |
| `surface` | Zinc 900 | `#18181b` | Cards, Sidebar |
| `primary` | Indigo 500 | `#6366f1` | Main Actions |
| `long` | Emerald 500 | `#10b981` | Buy Signals, Profit |
| `short` | Rose 500 | `#f43f5e` | Sell Signals, Loss |
| `warning` | Amber 500 | `#f59e0b` | System Alerts |
| `text-muted` | Zinc 400 | `#a1a1aa` | Labels, secondary text |

## 2. Core Components (Shadcn/UI)

### `AgentCard`
A summary card for the Grid View.
- **Header**: Symbol (BTC/USD) + Status Indicator (Pulse Green/Red).
- **Body**: Sparkline chart (last 1h PnL).
- **Footer**: Current Position + PnL Value.

### `BrainMRI` (Radar Chart)
Visualizes the consensus of the Swarm.
- **Axes**: M5, M15, M30, H1, H4.
- **Value**: 0 (Bearish) to 1 (Bullish).
- **Overlay**: Filled polygon representing the "Shape of the Market".

### `LiveTerminal`
A retro-styled log viewer.
- **Font**: JetBrains Mono.
- **Colors**: Based on log level (Info=White, Warning=Yellow, Crit=Red).
- **Behavior**: Auto-scroll to bottom, pause on hover.

## 3. Layout Structure

### **Global Sidebar**
- **Brand**: I.A.R.A. Logo (Glitch effect on hover).
- **Nav**: Dashboard, Market, Brain, Settings.
- **Status Footer**: WebSocket Connection State (Green Dot / Red Dot).

### **God Mode Layout (Grid)**
```
[ Header: Symbol | Current Price | 24h Change ]
------------------------------------------------
[ Main Chart (70%)        ] [ Brain MRI (30%)  ]
[ (Candles + Orders)      ] [ (Worker Voters)  ]
------------------------------------------------
[ Thought Stream (50%)    ] [ Active Pos (25%) ]
[ (Log output)            ] [ (PnL, Entry)     ]
                            [------------------]
                            [ Manual Controls  ]
                            [ (Buy/Sell/Kill)  ]
```

## 4. UX Patterns
- **Optimistic UI**: When user clicks "Buy", show "Submitting..." immediately, then "Filled" on socket event.
- **Connection Loss**: Full-screen blur overlay "Reconnecting to Mainframe..." if socket drops.
- **Sound Effects** (Optional): Subtle blips for new orders/alerts.
