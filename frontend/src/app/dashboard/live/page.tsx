"use client";

import { useState, useEffect, useCallback } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Terminal, Activity, Crosshair, TrendingUp, Loader2, Zap, ShieldAlert } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import { api } from "@/lib/api";
import { toast } from "sonner";
import { SearchableSelect } from "@/components/ui/searchable-select";

interface LiveTrade {
  id: string;
  symbol: string;
  side: string;
  price: number;
  pnl: number;
  status: string;
  confidence: number;
  time: number;
}

interface LiveStatus {
  agent_id: string | null;
  is_live: boolean;
  engine_start: number;
  pnl: number;
  trades: LiveTrade[];
  current_inference: Record<string, unknown> | null;
}

interface ModelInfo {
  name: string;
  size: string;
  last_modified: string;
}

export default function LiveTradingPage() {
  const [liveStatus, setLiveStatus] = useState<LiveStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [models, setModels] = useState<ModelInfo[]>([]);
  
  const [agentConfig, setAgentConfig] = useState({
    symbol: "BTCUSDT",
    model_name: "active_model.pth"
  });

  const fetchLiveStatus = useCallback(async () => {
    try {
      const [statusRes, modelsRes] = await Promise.all([
         api.get<LiveStatus>("/api/v1/brain/live/status"),
         api.get<ModelInfo[]>("/api/v1/brain/models")
      ]);
      setLiveStatus(statusRes || null);
      if (modelsRes) setModels(modelsRes);
    } catch (error) {
      console.error("Failed to fetch live execution status:", error);
    } finally {
      if (loading) setLoading(false);
    }
  }, [loading]);

  useEffect(() => {
    fetchLiveStatus();
    const interval = setInterval(fetchLiveStatus, 1000); // 1s aggressive poll for live
    return () => clearInterval(interval);
  }, [fetchLiveStatus]);
  
  const handleStartAgent = async () => {
    try {
      await api.post("/api/v1/fleet/start", {
        symbol: agentConfig.symbol,
        market_type: "crypto",
        model_name: agentConfig.model_name
      });
      toast.success(`Agent starting for ${agentConfig.symbol}`);
      fetchLiveStatus();
    } catch (error) {
      const err = error as Error;
      toast.error(err.message || "Failed to start live agent");
    }
  };
  
  const handleStopAgent = async () => {
    try {
      await api.post("/api/v1/fleet/killswitch", {});
      toast.success("Killswitch activated. All agents stopped.");
      fetchLiveStatus();
    } catch (error) {
      const err = error as Error;
      toast.error(err.message || "Failed to stop agents");
    }
  };

  if (loading && !liveStatus) {
    return (
      <div className="flex h-full items-center justify-center">
        <Loader2 className="h-8 w-8 animate-spin text-rose-500" />
      </div>
    );
  }

  const isLive = liveStatus?.is_live ?? false;
  const pnl = liveStatus?.pnl ?? 0.0;
  const trades = liveStatus?.trades ?? [];
  const activeAgent = liveStatus?.agent_id;
  
  return (
    <div className="p-6 space-y-6 max-w-7xl mx-auto pb-20">
      <div className="flex flex-col md:flex-row md:items-end justify-between gap-4">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <Activity className={cn("h-5 w-5", isLive ? "text-rose-500 animate-pulse" : "text-zinc-500")} />
            <span className={cn("text-xs font-bold uppercase tracking-widest", isLive ? "text-rose-500" : "text-zinc-500")}>
              {isLive ? "LIVE EXECUTION: ACTIVE" : "ENGINE SECURED"}
            </span>
          </div>
          <h1 className="text-3xl font-black text-zinc-100 tracking-tight flex items-center gap-3">
            Trading Terminal
            {isLive && <Badge variant="destructive" className="font-mono text-[10px] animate-pulse">BINANCE TESTNET</Badge>}
          </h1>
          <p className="text-sm text-zinc-500 mt-1 max-w-xl">
            Monitor real-time generative signals from I.A.R.A and the Binance API high-frequency execution queue.
          </p>
        </div>
      </div>
      
      {/* Control Panel */}
      <Card className="bg-zinc-900/40 border-zinc-800/60 backdrop-blur-md">
         <CardContent className="p-4 flex flex-col md:flex-row items-end gap-6">
            <div className="flex-1 space-y-2">
              <label className="text-[10px] uppercase font-bold text-zinc-500">Asset Symbol</label>
              <input 
                 type="text" 
                 className="w-full bg-zinc-950 border border-zinc-800 rounded px-3 py-2 text-sm text-zinc-200 outline-none uppercase"
                 placeholder="BTCUSDT"
                 value={agentConfig.symbol}
                 onChange={(e) => setAgentConfig({...agentConfig, symbol: e.target.value.toUpperCase()})}
                 disabled={isLive}
              />
            </div>
            <div className="flex-1 space-y-2">
              <label className="text-[10px] uppercase font-bold text-zinc-500">Brain Origin (Weights)</label>
              <SearchableSelect
                options={models.map((m) => ({ value: m.name, label: m.name }))}
                value={agentConfig.model_name}
                onChange={(val) => setAgentConfig({ ...agentConfig, model_name: val })}
                placeholder="Select a Model"
                searchPlaceholder="Search weights..."
              />
            </div>
            <div className="flex gap-3">
               {!isLive ? (
                 <button onClick={handleStartAgent} className="px-6 py-2 bg-emerald-600 hover:bg-emerald-500 text-white text-sm font-bold rounded shadow-lg shadow-emerald-900/20 transition-all flex items-center gap-2">
                   <Zap className="h-4 w-4" /> Start Engine
                 </button>
               ) : (
                 <button onClick={handleStopAgent} className="px-6 py-2 bg-rose-600 hover:bg-rose-500 text-white text-sm font-bold rounded shadow-lg shadow-rose-900/20 transition-all flex items-center gap-2 animate-pulse">
                   <ShieldAlert className="h-4 w-4" /> Stop & Liquidate
                 </button>
               )}
            </div>
         </CardContent>
      </Card>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
         {/* Live PnL Card */}
         <Card className="bg-zinc-900/40 border-zinc-800/60 backdrop-blur-md">
            <CardHeader className="border-b border-zinc-800/60 py-4 pb-4">
              <CardTitle className="text-xs font-bold text-zinc-400 uppercase tracking-widest flex justify-between items-center">
                 <span>Active Session PnL</span>
                 <TrendingUp className="h-4 w-4 text-emerald-500" />
              </CardTitle>
            </CardHeader>
            <CardContent className="pt-6">
                <p className={cn("text-4xl font-black font-mono tracking-tighter", pnl >= 0 ? "text-emerald-400" : "text-rose-400")}>
                  {pnl >= 0 ? "+" : ""}${pnl.toFixed(2)}
                </p>
                <p className="text-xs text-zinc-500 mt-2">Since engine start</p>
            </CardContent>
         </Card>
         
         {/* Active Positions */}
         {/* <Card className="md:col-span-2 bg-zinc-900/40 border-zinc-800/60 backdrop-blur-md">
           <CardHeader className="border-b border-zinc-800/60 py-4 flex flex-row items-center justify-between">
              <CardTitle className="text-xs font-bold text-zinc-400 uppercase tracking-widest flex items-center gap-2">
                 <Crosshair className="h-4 w-4 text-indigo-400" />
                 Recent Trades
              </CardTitle>
              <Badge className="bg-emerald-500/20 text-emerald-400 hover:bg-emerald-500/30">
                {trades.length} Recorded
              </Badge>
           </CardHeader>
           <CardContent className="p-0">
              <div className="divide-y divide-zinc-800/60">
                {trades.filter((t: LiveTrade) => t.status === "OPEN").map((trade: LiveTrade) => (
                  <div key={trade.id} className="p-4 flex items-center justify-between hover:bg-zinc-800/20 transition-colors">
                     <div className="flex items-center gap-4">
                        <div className={cn("px-2 py-1 rounded text-xs font-black", trade.side === "LONG" ? "bg-emerald-500/20 text-emerald-400" : "bg-rose-500/20 text-rose-400")}>
                           {trade.side}
                        </div>
                        <div>
                          <p className="text-sm font-bold text-zinc-200">{trade.symbol}</p>
                          <p className="text-[10px] text-zinc-500 font-mono">Entry: ${trade.price.toLocaleString()}</p>
                        </div>
                     </div>
                     <div className="text-right">
                        <p className={cn("text-lg font-black font-mono", trade.pnl >= 0 ? "text-emerald-400" : "text-rose-400")}>
                           {trade.pnl >= 0 ? "+" : ""}{trade.pnl.toFixed(2)} USDT
                        </p>
                        <p className="text-[10px] text-zinc-500 flex items-center justify-end gap-1">
                           AI Confidence: {(trade.confidence * 100).toFixed(0)}%
                        </p>
                     </div>
                  </div>
                ))}
              </div>
           </CardContent>
         </Card> */}
      </div>
      
      {/* Event Terminal */}
      <Card className="bg-zinc-900/40 border-zinc-800/60 backdrop-blur-md">
        <CardHeader className="border-b border-zinc-800/60 py-3">
           <CardTitle className="text-xs font-bold text-zinc-400 uppercase tracking-widest flex items-center justify-between">
              <div className="flex items-center gap-2">
                 <Terminal className="h-4 w-4 text-zinc-500" />
                 Engine Event Log
              </div>
              {activeAgent && (
                 <span className="text-[10px] text-emerald-500 font-mono bg-emerald-500/10 px-2 py-0.5 rounded">
                   Target: {activeAgent}
                 </span>
              )}
           </CardTitle>
        </CardHeader>
        <CardContent className="p-0">
           <div className="h-64 bg-black/50 p-4 font-mono text-[10px] text-emerald-500 flex flex-col justify-end pb-4 space-y-1">
              <div className="opacity-50"><span className="text-zinc-600 mr-2">12:45:01</span> [INFO] WebSocket Binance Hub Initialized</div>
              <div className="opacity-70"><span className="text-zinc-600 mr-2">12:45:03</span> [INFO] Live Ingestor syncing M5 array...</div>
              <div><span className="text-zinc-600 mr-2">12:51:00</span> [WARN] Engine: Network ping &gt; 300ms. Aborting routine.</div>
              <div className="text-white font-bold bg-indigo-500/20 px-2 py-0.5 rounded -mx-2"><span className="text-zinc-500 mr-2 font-normal">13:00:00</span> [SIGNAL] IARA Triggered: LONG BTCUSDT (Confidence 88%)</div>
              <div className="text-emerald-400"><span className="text-zinc-600 mr-2">13:00:01</span> [EXEC] Binance API: Limit Order Accepted @ 61240.50</div>
           </div>
        </CardContent>
      </Card>
    </div>
  );
}
