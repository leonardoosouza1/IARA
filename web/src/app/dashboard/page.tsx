"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { useFleetStore } from "@/hooks/use-fleet-store";
import { cn } from "@/lib/utils";
import {
  Activity,
  TrendingUp,
  TrendingDown,
  Zap,
  Brain,
  DollarSign,
  BarChart3,
  Clock,
} from "lucide-react";
import Link from "next/link";
import type { Agent } from "@/types";

function formatPnl(pnl: number): string {
  const sign = pnl >= 0 ? "+" : "";
  return `${sign}$${pnl.toFixed(2)}`;
}

function formatUptime(seconds: number): string {
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  return `${h}h ${m}m`;
}

const statusConfig: Record<Agent["status"], { color: string; label: string; pulse: boolean }> = {
  idle: { color: "text-zinc-400", label: "Idle", pulse: false },
  thinking: { color: "text-blue-400", label: "Thinking", pulse: true },
  stalking: { color: "text-amber-400", label: "Stalking", pulse: true },
  in_trade: { color: "text-emerald-400", label: "In Trade", pulse: true },
  paused: { color: "text-zinc-500", label: "Paused", pulse: false },
  error: { color: "text-red-400", label: "Error", pulse: true },
};

function AgentCard({ agent }: { agent: Agent }) {
  const config = statusConfig[agent.status];

  return (
    <Link href={`/dashboard/god-mode/${agent.id}`}>
      <Card className="bg-zinc-900/50 border-zinc-800/60 hover:border-indigo-500/30 hover:bg-zinc-900/80 transition-all duration-200 cursor-pointer group">
        <CardHeader className="pb-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="relative">
                <div
                  className={cn(
                    "h-10 w-10 rounded-lg flex items-center justify-center",
                    agent.marketType === "crypto"
                      ? "bg-amber-500/10 text-amber-400"
                      : "bg-blue-500/10 text-blue-400",
                  )}
                >
                  <DollarSign className="h-5 w-5" />
                </div>
                {config?.pulse && (
                  <span
                    className={cn(
                      "absolute -top-0.5 -right-0.5 h-2.5 w-2.5 rounded-full",
                      config?.color?.replace("text-", "bg-"),
                    )}
                  >
                    <span
                      className={cn(
                        "absolute inset-0 rounded-full animate-ping",
                        config?.color?.replace("text-", "bg-"),
                        "opacity-75",
                      )}
                    />
                  </span>
                )}
              </div>

              <div>
                <CardTitle className="text-base font-semibold text-zinc-100 group-hover:text-indigo-300 transition-colors">
                  {agent.symbol}
                </CardTitle>
                <div className="flex items-center gap-2 mt-0.5">
                  <Badge
                    variant="outline"
                    className="text-[10px] h-4 px-1.5 border-zinc-700 text-zinc-400"
                  >
                    {agent?.marketType?.toUpperCase()}
                  </Badge>
                  <span className={cn("text-xs font-medium", config?.color)}>
                    {config?.label}
                  </span>
                </div>
              </div>
            </div>
          </div>
        </CardHeader>

        <CardContent className="space-y-3">
          {/* PnL */}
          <div className="flex items-center justify-between">
            <span className="text-xs text-zinc-500">Session PnL</span>
            <span
              className={cn(
                "text-sm font-bold font-mono",
                agent.pnl >= 0 ? "text-emerald-400" : "text-red-400",
              )}
            >
              {formatPnl(agent.pnl)}
            </span>
          </div>

          {/* Stats Row */}
          <div className="grid grid-cols-3 gap-2">
            <div className="text-center p-2 rounded-md bg-zinc-800/40">
              <p className="text-[10px] text-zinc-500 uppercase tracking-wide">
                Trades
              </p>
              <p className="text-sm font-semibold text-zinc-200 font-mono">
                {agent.totalTrades}
              </p>
            </div>
            <div className="text-center p-2 rounded-md bg-zinc-800/40">
              <p className="text-[10px] text-zinc-500 uppercase tracking-wide">
                Win Rate
              </p>
              <p className="text-sm font-semibold text-zinc-200 font-mono">
                {(agent.winRate * 100).toFixed(0)}%
              </p>
            </div>
            <div className="text-center p-2 rounded-md bg-zinc-800/40">
              <p className="text-[10px] text-zinc-500 uppercase tracking-wide">
                Uptime
              </p>
              <p className="text-sm font-semibold text-zinc-200 font-mono">
                {formatUptime(agent.uptime)}
              </p>
            </div>
          </div>

          {/* Current Position */}
          {agent.currentPosition && (
            <div
              className={cn(
                "flex items-center justify-between rounded-lg px-3 py-2 text-sm",
                agent.currentPosition.side === "long"
                  ? "bg-emerald-500/10 border border-emerald-500/20"
                  : "bg-red-500/10 border border-red-500/20",
              )}
            >
              <div className="flex items-center gap-2">
                {agent.currentPosition.side === "long" ? (
                  <TrendingUp className="h-4 w-4 text-emerald-400" />
                ) : (
                  <TrendingDown className="h-4 w-4 text-red-400" />
                )}
                <span
                  className={cn(
                    "font-medium uppercase text-xs",
                    agent.currentPosition.side === "long"
                      ? "text-emerald-400"
                      : "text-red-400",
                  )}
                >
                  {agent.currentPosition.side}
                </span>
              </div>
              <span
                className={cn(
                  "font-mono text-xs font-semibold",
                  agent.currentPosition.unrealizedPnl >= 0
                    ? "text-emerald-400"
                    : "text-red-400",
                )}
              >
                {formatPnl(agent.currentPosition.unrealizedPnl)}
              </span>
            </div>
          )}
        </CardContent>
      </Card>
    </Link>
  );
}

// Demo agents for initial UI development
const DEMO_AGENTS: Agent[] = [
  {
    id: "agent-btc-1",
    symbol: "BTC/USD",
    marketType: "crypto",
    status: "in_trade",
    modelName: "SwarmBrain v9",
    pnl: 324.50,
    winRate: 0.67,
    totalTrades: 12,
    currentPosition: { side: "long", entryPrice: 97500, currentPrice: 98200, size: 0.01, unrealizedPnl: 7.0, entryTime: "2026-02-17T10:00:00Z" },
    lastSignal: null,
    uptime: 7200,
  },
  {
    id: "agent-eth-1",
    symbol: "ETH/USD",
    marketType: "crypto",
    status: "thinking",
    modelName: "SwarmBrain v9",
    pnl: -42.30,
    winRate: 0.45,
    totalTrades: 8,
    currentPosition: null,
    lastSignal: null,
    uptime: 5400,
  },
  {
    id: "agent-eur-1",
    symbol: "EUR/USD",
    marketType: "forex",
    status: "stalking",
    modelName: "SwarmBrain v9",
    pnl: 89.20,
    winRate: 0.58,
    totalTrades: 19,
    currentPosition: null,
    lastSignal: null,
    uptime: 14400,
  },
];

export default function FleetOverviewPage() {
  const storeAgents = useFleetStore((s) => s.agents);
  const agents = storeAgents.length > 0 ? storeAgents : DEMO_AGENTS;

  const totalPnl = agents.reduce((acc, a) => acc + a.pnl, 0);
  const activeCount = agents.filter((a) => a.status !== "idle" && a.status !== "paused").length;
  const inTradeCount = agents.filter((a) => a.status === "in_trade").length;

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-zinc-100 tracking-tight">
            Fleet Overview
          </h1>
          <p className="text-sm text-zinc-500 mt-1">
            Real-time status of all active agents
          </p>
        </div>
        <Badge
          variant="outline"
          className={cn(
            "text-sm px-3 py-1 font-mono",
            totalPnl >= 0
              ? "border-emerald-500/30 text-emerald-400 bg-emerald-500/5"
              : "border-red-500/30 text-red-400 bg-red-500/5",
          )}
        >
          Total PnL: {formatPnl(totalPnl)}
        </Badge>
      </div>

      {/* Stats Bar */}
      <div className="grid grid-cols-4 gap-4">
        <Card className="bg-zinc-900/40 border-zinc-800/50">
          <CardContent className="p-4 flex items-center gap-3">
            <div className="h-10 w-10 rounded-lg bg-indigo-500/10 flex items-center justify-center">
              <Brain className="h-5 w-5 text-indigo-400" />
            </div>
            <div>
              <p className="text-xs text-zinc-500">Active Agents</p>
              <p className="text-xl font-bold text-zinc-100 font-mono">
                {activeCount}
              </p>
            </div>
          </CardContent>
        </Card>

        <Card className="bg-zinc-900/40 border-zinc-800/50">
          <CardContent className="p-4 flex items-center gap-3">
            <div className="h-10 w-10 rounded-lg bg-emerald-500/10 flex items-center justify-center">
              <Activity className="h-5 w-5 text-emerald-400" />
            </div>
            <div>
              <p className="text-xs text-zinc-500">In Trade</p>
              <p className="text-xl font-bold text-zinc-100 font-mono">
                {inTradeCount}
              </p>
            </div>
          </CardContent>
        </Card>

        <Card className="bg-zinc-900/40 border-zinc-800/50">
          <CardContent className="p-4 flex items-center gap-3">
            <div className="h-10 w-10 rounded-lg bg-amber-500/10 flex items-center justify-center">
              <BarChart3 className="h-5 w-5 text-amber-400" />
            </div>
            <div>
              <p className="text-xs text-zinc-500">Total Trades</p>
              <p className="text-xl font-bold text-zinc-100 font-mono">
                {agents.reduce((a, b) => a + (b.totalTrades || 0), 0)}
              </p>
            </div>
          </CardContent>
        </Card>

        <Card className="bg-zinc-900/40 border-zinc-800/50">
          <CardContent className="p-4 flex items-center gap-3">
            <div className="h-10 w-10 rounded-lg bg-purple-500/10 flex items-center justify-center">
              <Zap className="h-5 w-5 text-purple-400" />
            </div>
            <div>
              <p className="text-xs text-zinc-500">Fleet PnL</p>
              <p
                className={cn(
                  "text-xl font-bold font-mono",
                  totalPnl >= 0 ? "text-emerald-400" : "text-red-400",
                )}
              >
                {formatPnl(totalPnl)}
              </p>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Agent Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
        {agents.map((agent) => (
          <AgentCard key={agent.id} agent={agent} />
        ))}
      </div>
    </div>
  );
}
