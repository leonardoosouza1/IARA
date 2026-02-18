"use client";

import { use } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import { RealTimeChart } from "@/components/charts/real-time-chart";
import { ThoughtStream } from "@/components/fleet/thought-stream";
import { BrainMRI } from "@/components/fleet/brain-mri";
import { useFleetStore } from "@/hooks/use-fleet-store";
import { cn } from "@/lib/utils";
import {
  ArrowLeft,
  TrendingUp,
  TrendingDown,
  Power,
  AlertTriangle,
  DollarSign,
  Crosshair,
} from "lucide-react";
import Link from "next/link";
import type { Agent, WorkerVotes } from "@/types";

// Demo agent for development
const DEMO_AGENT: Agent = {
  id: "agent-btc-1",
  symbol: "BTC/USD",
  marketType: "crypto",
  status: "in_trade",
  modelName: "SwarmBrain v9",
  pnl: 324.50,
  winRate: 0.67,
  totalTrades: 12,
  currentPosition: {
    side: "long",
    entryPrice: 97500,
    currentPrice: 98200,
    size: 0.01,
    unrealizedPnl: 7.0,
    entryTime: "2026-02-17T10:00:00Z",
  },
  lastSignal: {
    timestamp: "2026-02-17T19:00:06Z",
    action: "buy",
    confidence: 0.82,
    longScore: 0.85,
    shortScore: 0.12,
    workerVotes: { M5: 0.72, M15: 0.55, M30: 0.80, H1: 0.38, H4: 0.65, consensus: 0.78 },
    lfmThought: "Bullish structure intact. Price holding above key support.",
  },
  uptime: 7200,
};

const DEMO_VOTES: WorkerVotes = {
  M5: 0.72,
  M15: 0.55,
  M30: 0.80,
  H1: 0.38,
  H4: 0.65,
  consensus: 0.78,
};

function formatPnl(pnl: number): string {
  const sign = pnl >= 0 ? "+" : "";
  return `${sign}$${pnl.toFixed(2)}`;
}

export default function GodModePage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = use(params);
  const storeAgents = useFleetStore((s) => s.agents);
  const agent = storeAgents.find((a) => a.id === id) || DEMO_AGENT;
  const votes = agent.lastSignal?.workerVotes || DEMO_VOTES;

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="flex items-center justify-between px-6 py-4 border-b border-zinc-800/40 bg-zinc-900/30">
        <div className="flex items-center gap-4">
          <Link href="/dashboard">
            <Button variant="ghost" size="icon" className="text-zinc-400 hover:text-zinc-200">
              <ArrowLeft className="h-4 w-4" />
            </Button>
          </Link>

          <div className="flex items-center gap-3">
            <div className={cn(
              "h-10 w-10 rounded-lg flex items-center justify-center",
              agent.marketType === "crypto"
                ? "bg-amber-500/10 text-amber-400"
                : "bg-blue-500/10 text-blue-400"
            )}>
              <DollarSign className="h-5 w-5" />
            </div>

            <div>
              <h1 className="text-lg font-bold text-zinc-100">{agent.symbol}</h1>
              <div className="flex items-center gap-2">
                <Badge variant="outline" className="text-[10px] h-4 px-1.5 border-zinc-700 text-zinc-400">
                  {agent.marketType.toUpperCase()}
                </Badge>
                <Badge variant="outline" className="text-[10px] h-4 px-1.5 border-indigo-500/30 text-indigo-400">
                  {agent.modelName}
                </Badge>
              </div>
            </div>
          </div>
        </div>

        {/* Price + Actions */}
        <div className="flex items-center gap-4">
          <div className="text-right">
            <p className="text-lg font-bold font-mono text-zinc-100">
              {agent.currentPosition?.currentPrice?.toLocaleString() || "—"}
            </p>
            <p className={cn(
              "text-sm font-mono font-semibold",
              agent.pnl >= 0 ? "text-emerald-400" : "text-red-400"
            )}>
              {formatPnl(agent.pnl)}
            </p>
          </div>

          <Separator orientation="vertical" className="h-10 bg-zinc-800" />

          {/* Manual Controls */}
          <div className="flex gap-2">
            <Button size="sm" className="bg-emerald-600 hover:bg-emerald-700 text-white gap-1.5">
              <TrendingUp className="h-3.5 w-3.5" />
              Force BUY
            </Button>
            <Button size="sm" className="bg-red-600 hover:bg-red-700 text-white gap-1.5">
              <TrendingDown className="h-3.5 w-3.5" />
              Force SELL
            </Button>
            <Button size="sm" variant="outline" className="border-red-500/40 text-red-400 hover:bg-red-500/10 gap-1.5">
              <Power className="h-3.5 w-3.5" />
              Kill
            </Button>
          </div>
        </div>
      </div>

      {/* Main Grid */}
      <div className="flex-1 grid grid-cols-[1fr_320px] grid-rows-[1fr_1fr] gap-4 p-4 overflow-hidden">
        {/* Chart (Top Left - Dominant) */}
        <div className="row-span-1">
          <Card className="h-full bg-zinc-900/30 border-zinc-800/40">
            <CardHeader className="pb-2 pt-3 px-4">
              <div className="flex items-center justify-between">
                <CardTitle className="text-sm font-semibold text-zinc-300">
                  <Crosshair className="h-4 w-4 inline mr-2 text-indigo-400" />
                  Live Chart
                </CardTitle>
                {agent.currentPosition && (
                  <Badge className={cn(
                    "text-xs",
                    agent.currentPosition.side === "long"
                      ? "bg-emerald-500/10 text-emerald-400 border-emerald-500/20"
                      : "bg-red-500/10 text-red-400 border-red-500/20"
                  )}>
                    {agent.currentPosition.side.toUpperCase()} @ {agent.currentPosition.entryPrice.toLocaleString()}
                  </Badge>
                )}
              </div>
            </CardHeader>
            <CardContent className="p-2 pt-0">
              <RealTimeChart symbol={agent.symbol} height={280} />
            </CardContent>
          </Card>
        </div>

        {/* Brain MRI (Top Right) */}
        <div className="row-span-1">
          <Card className="h-full bg-zinc-900/30 border-zinc-800/40 flex flex-col">
            <CardHeader className="pb-2 pt-3 px-4">
              <CardTitle className="text-sm font-semibold text-zinc-300">
                🧠 Brain MRI — Worker Votes
              </CardTitle>
            </CardHeader>
            <CardContent className="flex-1 flex items-center justify-center p-4">
              <BrainMRI votes={votes} />
            </CardContent>
          </Card>
        </div>

        {/* Thought Stream (Bottom Left) */}
        <div className="row-span-1">
          <div className="h-full">
            <ThoughtStream agentId={agent.id} />
          </div>
        </div>

        {/* Position + Stats (Bottom Right) */}
        <div className="row-span-1">
          <Card className="h-full bg-zinc-900/30 border-zinc-800/40 flex flex-col">
            <CardHeader className="pb-2 pt-3 px-4">
              <CardTitle className="text-sm font-semibold text-zinc-300">
                📊 Position & Stats
              </CardTitle>
            </CardHeader>
            <CardContent className="flex-1 p-4 space-y-4 text-sm">
              {/* Active Position */}
              {agent.currentPosition ? (
                <div className={cn(
                  "rounded-lg p-3 space-y-2 border",
                  agent.currentPosition.side === "long"
                    ? "bg-emerald-500/5 border-emerald-500/20"
                    : "bg-red-500/5 border-red-500/20"
                )}>
                  <div className="flex justify-between">
                    <span className="text-zinc-500">Side</span>
                    <span className={cn(
                      "font-bold uppercase",
                      agent.currentPosition.side === "long" ? "text-emerald-400" : "text-red-400"
                    )}>
                      {agent.currentPosition.side}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-zinc-500">Entry</span>
                    <span className="text-zinc-200 font-mono">${agent.currentPosition.entryPrice.toLocaleString()}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-zinc-500">Current</span>
                    <span className="text-zinc-200 font-mono">${agent.currentPosition.currentPrice.toLocaleString()}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-zinc-500">Unrealized</span>
                    <span className={cn(
                      "font-bold font-mono",
                      agent.currentPosition.unrealizedPnl >= 0 ? "text-emerald-400" : "text-red-400"
                    )}>
                      {formatPnl(agent.currentPosition.unrealizedPnl)}
                    </span>
                  </div>
                </div>
              ) : (
                <div className="text-center py-4 text-zinc-500">
                  <AlertTriangle className="h-5 w-5 mx-auto mb-1 text-zinc-600" />
                  No active position
                </div>
              )}

              {/* Stats */}
              <Separator className="bg-zinc-800/40" />
              <div className="space-y-2">
                <div className="flex justify-between">
                  <span className="text-zinc-500">Total Trades</span>
                  <span className="text-zinc-200 font-mono">{agent.totalTrades}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-zinc-500">Win Rate</span>
                  <span className="text-zinc-200 font-mono">{(agent.winRate * 100).toFixed(0)}%</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-zinc-500">Session PnL</span>
                  <span className={cn(
                    "font-bold font-mono",
                    agent.pnl >= 0 ? "text-emerald-400" : "text-red-400"
                  )}>
                    {formatPnl(agent.pnl)}
                  </span>
                </div>
              </div>

              {/* Last Signal */}
              {agent.lastSignal && (
                <>
                  <Separator className="bg-zinc-800/40" />
                  <div className="space-y-1">
                    <p className="text-[10px] text-zinc-600 uppercase tracking-wider font-semibold">Last Signal</p>
                    <div className="flex justify-between">
                      <span className="text-zinc-500">Action</span>
                      <Badge className={cn(
                        "text-[10px]",
                        agent.lastSignal.action === "buy" ? "bg-emerald-500/10 text-emerald-400" : "bg-red-500/10 text-red-400"
                      )}>
                        {agent.lastSignal.action.toUpperCase()}
                      </Badge>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-zinc-500">Confidence</span>
                      <span className="text-zinc-200 font-mono">{(agent.lastSignal.confidence * 100).toFixed(0)}%</span>
                    </div>
                  </div>
                </>
              )}
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
