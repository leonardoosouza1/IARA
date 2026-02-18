// Core TypeScript interfaces for the IARA Fleet system

export interface Agent {
  id: string;
  symbol: string;
  marketType: "crypto" | "forex";
  status: "idle" | "active" | "thinking" | "stalking" | "in_trade" | "paused" | "error";
  modelName: string;
  pnl: number;
  winRate: number;
  totalTrades: number;
  currentPosition: Position | null;
  lastSignal: BrainSignal | null;
  uptime: number; // seconds
  config?: {
    pulse: string;
    label: string;
    color: string;
  };
}

export interface Position {
  side: "long" | "short";
  entryPrice: number;
  currentPrice: number;
  size: number;
  unrealizedPnl: number;
  entryTime: string;
}

export interface BrainSignal {
  timestamp: string;
  action: "buy" | "sell" | "hold";
  confidence: number;
  longScore: number;
  shortScore: number;
  workerVotes: WorkerVotes;
  lfmThought: string;
}

export interface WorkerVotes {
  M5: number;  // -1 to 1 (bearish to bullish)
  M15: number;
  M30: number;
  H1: number;
  H4: number;
  consensus: number; // 0 to 1
  [key: string]: number; // Allow additional fields from backend
}

export interface MarketTick {
  symbol: string;
  price: number;
  time: string;
  volume?: number;
}

export interface CandleData {
  time: number; // unix timestamp
  open: number;
  high: number;
  low: number;
  close: number;
  volume?: number;
}

export interface TradeRecord {
  id: string;
  agentId: string;
  symbol: string;
  side: "buy" | "sell";
  entryPrice: number;
  exitPrice: number | null;
  size: number;
  pnl: number | null;
  reason: string;
  openedAt: string;
  closedAt: string | null;
}

export interface FleetStatus {
  activeAgents: number;
  totalPnl: number;
  uptime: number;
  agents: Agent[];
}

export interface User {
  id: string;
  username: string;
  role: "admin" | "viewer";
}

export interface LoginCredentials {
  username: string;
  password: string;
}

export interface BrainThought {
  agentId: string;
  timestamp: string;
  step: string;
  text: string;
  level: "info" | "warning" | "critical" | "success";
}
