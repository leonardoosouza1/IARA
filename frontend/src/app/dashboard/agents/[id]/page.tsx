"use client";

import { use } from "react";
import { useRouter } from "next/navigation";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import { RealTimeChart } from "@/components/charts/real-time-chart";
import { ThoughtStream } from "@/components/fleet/thought-stream";
import { BrainMRI } from "@/components/fleet/brain-mri";
import { useFleetStore } from "@/hooks/use-fleet-store";
import { api } from "@/lib/api";
import { cn } from "@/lib/utils";
import {
  ArrowLeft,
  TrendingUp,
  TrendingDown,
  Power,
  AlertTriangle,
  DollarSign,
  Crosshair,
  Loader2,
} from "lucide-react";
import Link from "next/link";
import { toast } from "sonner";
import { getSocket } from "@/lib/socket";

function formatPnl(pnl: number): string {
  const sign = pnl >= 0 ? "+" : "";
  return `${sign}$${pnl.toFixed(2)}`;
}

export default function GodModePage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = use(params);
  const router = useRouter();

  // Direct selection from store - no need for useEffect sync
  const agent = useFleetStore((s) => s.agents.find((a) => a.id === id));
  const socketConnected = useFleetStore((s) => s.socketConnected);

  // Loading state: wait for socket
  const loading = !socketConnected;

  // Actions
  const handleAction = async (action: "buy" | "sell") => {
    if (!agent) return;
    try {
      // Direct WebSocket emit for real-time action
      getSocket().emit("manual_override", { agent_id: agent.id, action });
      toast.info(`Sending Force ${action.toUpperCase()} command...`);
    } catch {
      toast.error("Failed to send override command");
    }
  };

  const handleKill = async () => {
    if (!agent) return;
    try {
      await api.post("/api/v1/fleet/stop", { agent_id: agent.id });
      toast.warning(`Kill signals sent to ${agent.symbol}`);
      router.refresh();
    } catch {
      toast.error("Failed to stop agent");
    }
  };

  if (loading) {
    return (
      <div className="flex h-full items-center justify-center flex-col gap-4">
        <Loader2 className="h-8 w-8 animate-spin text-indigo-500" />
        <p className="text-zinc-500 text-sm">Connecting to SwarmBrain...</p>
      </div>
    );
  }

  if (!agent && socketConnected) {
    return (
      <div className="flex h-full items-center justify-center flex-col gap-4">
        <AlertTriangle className="h-12 w-12 text-amber-500" />
        <h2 className="text-xl font-bold text-zinc-100">Agent Not Found</h2>
        <p className="text-zinc-400">
          The agent <span className="font-mono text-zinc-300">{id}</span> is not
          active or does not exist.
        </p>
        <Link href="/dashboard/agents">
          <Button variant="outline">Return to Fleet</Button>
        </Link>
      </div>
    );
  }

  // Double check agent exists for TS
  if (!agent) return null;

  const votes = agent.lastSignal?.workerVotes || {
    M5: 0,
    M15: 0,
    M30: 0,
    H1: 0,
    H4: 0,
    consensus: 0,
  };
  const config = agent.config || {
    color: "indigo",
    label: agent.symbol,
    pulse: "none",
  };

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="flex items-center justify-between px-6 py-4 border-b border-zinc-800/40 bg-zinc-900/30">
        <div className="flex items-center gap-4">
          <Link href="/dashboard">
            <Button
              variant="ghost"
              size="icon"
              className="text-zinc-400 hover:text-zinc-200"
            >
              <ArrowLeft className="h-4 w-4" />
            </Button>
          </Link>

          <div className="flex items-center gap-3">
            <div
              className={cn(
                "h-10 w-10 rounded-lg flex items-center justify-center",
                `bg-${config.color}-500/10 text-${config.color}-400`,
              )}
            >
              <DollarSign className="h-5 w-5" />
            </div>

            <div>
              <h1 className="text-lg font-bold text-zinc-100 flex items-center gap-2">
                {agent.symbol}
                {agent.status === "thinking" && (
                  <span className="relative flex h-2 w-2">
                    <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-blue-400 opacity-75"></span>
                    <span className="relative inline-flex rounded-full h-2 w-2 bg-blue-500"></span>
                  </span>
                )}
              </h1>
              <div className="flex items-center gap-2">
                <Badge
                  variant="outline"
                  className={cn(
                    "text-[10px] h-4 px-1.5 border-zinc-700",
                    agent.status === "active" || agent.status === "in_trade"
                      ? "text-emerald-400 border-emerald-500/30"
                      : "text-zinc-400",
                  )}
                >
                  {agent.status.toUpperCase()}
                </Badge>
                <Badge
                  variant="outline"
                  className="text-[10px] h-4 px-1.5 border-indigo-500/30 text-indigo-400"
                >
                  {agent.modelName}
                </Badge>
              </div>
            </div>
          </div>
        </div>

        {/* Price + Actions */}
        <div className="flex items-center gap-4">
          <div className="text-right hidden sm:block">
            <p className="text-lg font-bold font-mono text-zinc-100">
              {agent.currentPosition?.currentPrice?.toLocaleString() || "—"}
            </p>
            <p
              className={cn(
                "text-sm font-mono font-semibold",
                agent.pnl >= 0 ? "text-emerald-400" : "text-red-400",
              )}
            >
              {formatPnl(agent.pnl)}
            </p>
          </div>

          <Separator
            orientation="vertical"
            className="h-10 bg-zinc-800 hidden sm:block"
          />

          {/* Manual Controls */}
          <div className="flex gap-2">
            <Button
              size="sm"
              onClick={() => handleAction("buy")}
              className="bg-emerald-600 hover:bg-emerald-700 text-white gap-1.5"
            >
              <TrendingUp className="h-3.5 w-3.5" />
              <span className="hidden sm:inline">Force BUY</span>
            </Button>
            <Button
              size="sm"
              onClick={() => handleAction("sell")}
              className="bg-red-600 hover:bg-red-700 text-white gap-1.5"
            >
              <TrendingDown className="h-3.5 w-3.5" />
              <span className="hidden sm:inline">Force SELL</span>
            </Button>
            <Button
              size="sm"
              variant="outline"
              onClick={handleKill}
              className="border-red-500/40 text-red-400 hover:bg-red-500/10 gap-1.5"
            >
              <Power className="h-3.5 w-3.5" />
              <span className="hidden sm:inline">Kill</span>
            </Button>
          </div>
        </div>
      </div>

      {/* Main Grid */}
      <div className="flex-1 grid grid-cols-1 lg:grid-cols-[1fr_320px] grid-rows-[min-content_1fr] lg:grid-rows-[1fr_1fr] gap-4 p-4 overflow-y-auto">
        {/* Chart (Top Left - Dominant) */}
        <div className="lg:row-span-1 h-[300px] lg:h-auto">
          <Card className="h-full bg-zinc-900/30 border-zinc-800/40">
            <CardHeader className="pb-2 pt-3 px-4">
              <div className="flex items-center justify-between">
                <CardTitle className="text-sm font-semibold text-zinc-300">
                  <Crosshair className="h-4 w-4 inline mr-2 text-indigo-400" />
                  Live Chart
                </CardTitle>
                {agent.currentPosition && (
                  <Badge
                    className={cn(
                      "text-xs",
                      agent.currentPosition.side === "long"
                        ? "bg-emerald-500/10 text-emerald-400 border-emerald-500/20"
                        : "bg-red-500/10 text-red-400 border-red-500/20",
                    )}
                  >
                    {agent.currentPosition.side.toUpperCase()} @{" "}
                    {agent.currentPosition.entryPrice.toLocaleString()}
                  </Badge>
                )}
              </div>
            </CardHeader>
            <CardContent className="p-2 pt-0 h-[calc(100%-3rem)]">
              <RealTimeChart symbol={agent.symbol} agent={agent} height={300} />
            </CardContent>
          </Card>
        </div>

        {/* Brain MRI (Top Right) */}
        <div className="lg:row-span-1 min-h-[250px]">
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
        <div className="lg:row-span-1 min-h-[300px]">
          <div className="h-full">
            <ThoughtStream agentId={agent.id} />
          </div>
        </div>

        {/* Position + Stats (Bottom Right) */}
        <div className="lg:row-span-1 min-h-[300px]">
          <Card className="h-full bg-zinc-900/30 border-zinc-800/40 flex flex-col">
            <CardHeader className="pb-2 pt-3 px-4">
              <CardTitle className="text-sm font-semibold text-zinc-300">
                📊 Position & Stats
              </CardTitle>
            </CardHeader>
            <CardContent className="flex-1 p-4 space-y-4 text-sm">
              {/* Active Position */}
              {agent.currentPosition ? (
                <div
                  className={cn(
                    "rounded-lg p-3 space-y-2 border",
                    agent.currentPosition.side === "long"
                      ? "bg-emerald-500/5 border-emerald-500/20"
                      : "bg-red-500/5 border-red-500/20",
                  )}
                >
                  <div className="flex justify-between">
                    <span className="text-zinc-500">Side</span>
                    <span
                      className={cn(
                        "font-bold uppercase",
                        agent.currentPosition.side === "long"
                          ? "text-emerald-400"
                          : "text-red-400",
                      )}
                    >
                      {agent.currentPosition.side}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-zinc-500">Entry</span>
                    <span className="text-zinc-200 font-mono">
                      ${agent.currentPosition.entryPrice.toLocaleString()}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-zinc-500">Current</span>
                    <span className="text-zinc-200 font-mono">
                      ${agent.currentPosition.currentPrice.toLocaleString()}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-zinc-500">Unrealized</span>
                    <span
                      className={cn(
                        "font-bold font-mono",
                        agent.currentPosition.unrealizedPnl >= 0
                          ? "text-emerald-400"
                          : "text-red-400",
                      )}
                    >
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
                  <span className="text-zinc-200 font-mono">
                    {agent.totalTrades}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-zinc-500">Win Rate</span>
                  <span className="text-zinc-200 font-mono">
                    {(agent.winRate * 100).toFixed(0)}%
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-zinc-500">Session PnL</span>
                  <span
                    className={cn(
                      "font-bold font-mono",
                      agent.pnl >= 0 ? "text-emerald-400" : "text-red-400",
                    )}
                  >
                    {formatPnl(agent.pnl)}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-zinc-500">Uptime</span>
                  <span className="text-zinc-400 font-mono text-xs">
                    {new Date(agent.uptime * 1000).toLocaleTimeString()}
                  </span>
                </div>
              </div>

              {/* Last Signal */}
              {agent.lastSignal && (
                <>
                  <Separator className="bg-zinc-800/40" />
                  <div className="space-y-1">
                    <p className="text-[10px] text-zinc-600 uppercase tracking-wider font-semibold">
                      Last Signal
                    </p>
                    <div className="flex justify-between">
                      <span className="text-zinc-500">Action</span>
                      <Badge
                        variant="secondary"
                        className={cn(
                          "text-[10px]",
                          agent.lastSignal.action === "buy"
                            ? "bg-emerald-500/10 text-emerald-400"
                            : agent.lastSignal.action === "sell"
                              ? "bg-red-500/10 text-red-400"
                              : "bg-zinc-500/10 text-zinc-400",
                        )}
                      >
                        {agent.lastSignal.action.toUpperCase()}
                      </Badge>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-zinc-500">Confidence</span>
                      <span className="text-zinc-200 font-mono">
                        {(agent.lastSignal.confidence * 100).toFixed(0)}%
                      </span>
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
