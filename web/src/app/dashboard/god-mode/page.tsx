"use client";

import Link from "next/link";
import { useFleetStore } from "@/hooks/use-fleet-store";
import { cn } from "@/lib/utils";
import type { Agent } from "@/types";

// Demo agents for development
const DEMO_AGENTS: Agent[] = [
  {
    id: "agent-btc-1", symbol: "BTC/USD", marketType: "crypto", status: "in_trade",
    modelName: "SwarmBrain v9", pnl: 324.50, winRate: 0.67, totalTrades: 12,
    currentPosition: { side: "long", entryPrice: 97500, currentPrice: 98200, size: 0.01, unrealizedPnl: 7.0, entryTime: "2026-02-17T10:00:00Z" },
    lastSignal: null, uptime: 7200,
  },
  {
    id: "agent-eth-1", symbol: "ETH/USD", marketType: "crypto", status: "thinking",
    modelName: "SwarmBrain v9", pnl: -42.30, winRate: 0.45, totalTrades: 8,
    currentPosition: null, lastSignal: null, uptime: 5400,
  },
  {
    id: "agent-eur-1", symbol: "EUR/USD", marketType: "forex", status: "stalking",
    modelName: "SwarmBrain v9", pnl: 89.20, winRate: 0.58, totalTrades: 19,
    currentPosition: null, lastSignal: null, uptime: 14400,
  },
];

export default function GodModeListPage() {
  const storeAgents = useFleetStore((s) => s.agents);
  const agents = storeAgents.length > 0 ? storeAgents : DEMO_AGENTS;

  return (
    <div className="p-6 space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-zinc-100 tracking-tight">God Mode</h1>
        <p className="text-sm text-zinc-500 mt-1">Select an agent to observe its brain in real-time</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
        {agents.map((agent) => (
          <Link key={agent.id} href={`/dashboard/god-mode/${agent.id}`}>
            <div className="group rounded-xl border border-zinc-800/60 bg-zinc-900/40 p-5 hover:border-indigo-500/30 hover:bg-zinc-900/70 transition-all duration-200 cursor-pointer">
              <div className="flex items-center gap-3 mb-3">
                <div className="h-3 w-3 rounded-full bg-emerald-500 animate-pulse" />
                <span className="text-lg font-bold text-zinc-100 group-hover:text-indigo-300 transition-colors">
                  {agent.symbol}
                </span>
                <span className="text-xs text-zinc-500 font-mono ml-auto">
                  {agent.status.toUpperCase()}
                </span>
              </div>
              <p className="text-xs text-zinc-500">
                Click to enter God Mode for this agent
              </p>
            </div>
          </Link>
        ))}
      </div>
    </div>
  );
}
