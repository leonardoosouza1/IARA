import { create } from "zustand";
import type { Agent, MarketTick, BrainThought, FleetStatus, CandleData } from "@/types";

interface FleetState {
  // Connection
  socketConnected: boolean;
  setSocketConnected: (connected: boolean) => void;

  // Fleet
  agents: Agent[];
  setAgents: (agents: Agent[]) => void;
  updateAgent: (id: string, patch: Partial<Agent>) => void;

  // Market Data (per symbol)
  ticks: Record<string, MarketTick[]>;
  candles: Record<string, CandleData[]>;
  addTick: (symbol: string, tick: MarketTick) => void;
  addCandle: (symbol: string, candle: CandleData) => void;
  setCandles: (symbol: string, candles: CandleData[]) => void;

  // Brain Thoughts (per agent)
  thoughts: Record<string, BrainThought[]>;
  addThought: (agentId: string, thought: BrainThought) => void;
  clearThoughts: (agentId: string) => void;

  // Fleet Status
  fleetStatus: FleetStatus | null;
  setFleetStatus: (status: FleetStatus) => void;
}

const MAX_TICKS = 500;
const MAX_THOUGHTS = 200;

export const useFleetStore = create<FleetState>((set) => ({
  // Connection
  socketConnected: false,
  setSocketConnected: (connected) => set({ socketConnected: connected }),

  // Fleet
  agents: [],
  setAgents: (agents) => set({ agents }),
  updateAgent: (id, patch) =>
    set((state) => ({
      agents: state.agents.map((a) =>
        a.id === id ? { ...a, ...patch } : a
      ),
    })),

  // Market Data
  ticks: {},
  candles: {},
  addTick: (symbol, tick) =>
    set((state) => ({
      ticks: {
        ...state.ticks,
        [symbol]: [...(state.ticks[symbol] || []).slice(-MAX_TICKS), tick],
      },
    })),
  addCandle: (symbol, candle) =>
    set((state) => ({
      candles: {
        ...state.candles,
        [symbol]: [...(state.candles[symbol] || []), candle],
      },
    })),
  setCandles: (symbol, candles) =>
    set((state) => ({
      candles: { ...state.candles, [symbol]: candles },
    })),

  // Brain Thoughts
  thoughts: {},
  addThought: (agentId, thought) =>
    set((state) => ({
      thoughts: {
        ...state.thoughts,
        [agentId]: [
          ...(state.thoughts[agentId] || []).slice(-MAX_THOUGHTS),
          thought,
        ],
      },
    })),
  clearThoughts: (agentId) =>
    set((state) => ({
      thoughts: { ...state.thoughts, [agentId]: [] },
    })),

  // Fleet Status
  fleetStatus: null,
  setFleetStatus: (status) => set({ fleetStatus: status }),
}));
