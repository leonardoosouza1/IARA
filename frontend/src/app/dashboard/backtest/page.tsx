"use client";

import { useState, useEffect, useCallback, useRef } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { BarChart3, Terminal, Play, Loader2 } from "lucide-react";
import { api } from "@/lib/api";
import { toast } from "sonner";
import { cn } from "@/lib/utils";
import { SearchableSelect } from "@/components/ui/searchable-select";

interface BacktestStatus {
  is_running: boolean;
  dataset: string | null;
  threshold: number;
  message: string;
  logs: string[];
  results: {
    total_trades: number;
    win_rate: number;
    total_pnl: number;
    max_drawdown: number;
  } | null;
}

interface DatasetInfo {
  symbol: string;
  shards: number;
  path: string;
}

interface ModelInfo {
  name: string;
  size: string;
  last_modified: string;
}

export default function BacktestStudioPage() {
  const [datasets, setDatasets] = useState<DatasetInfo[]>([]);
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [backtestStatus, setBacktestStatus] = useState<BacktestStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const backtestLogsContainerRef = useRef<HTMLDivElement>(null);

  const [backtestProtocol, setBacktestProtocol] = useState({
    dataset: "",
    model_name: "active_model.pth",
    threshold: 0.82,
  });

  // Auto-scroll for Backtest Logs
  useEffect(() => {
    if (backtestLogsContainerRef.current) {
      backtestLogsContainerRef.current.scrollTop =
        backtestLogsContainerRef.current.scrollHeight;
    }
  }, [backtestStatus?.logs]);

  const fetchData = useCallback(async () => {
    try {
      const [datasetsRes, backtestRes, modelsRes] = await Promise.all([
        api.get<DatasetInfo[]>("/api/v1/brain/data/inventory"),
        api.get<BacktestStatus>("/api/v1/brain/backtest/status"),
        api.get<ModelInfo[]>("/api/v1/brain/models"),
      ]);
      setDatasets(datasetsRes || []);
      setBacktestStatus(backtestRes || null);
      setModels(modelsRes || []);
    } catch (error) {
      console.error("Failed to fetch backtest data:", error);
      if (!loading) toast.error("Failed to connect to Backtest services.");
    } finally {
      setLoading(false);
    }
  }, [loading]);

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 1000);
    return () => clearInterval(interval);
  }, [fetchData]);

  const handleRunBacktest = async () => {
    if (!backtestProtocol.dataset) {
      toast.error("Please select a dataset to backtest");
      return;
    }
    try {
      await api.post("/api/v1/brain/backtest", backtestProtocol);
      toast.success("Sniper Backtest initiated!");
      fetchData();
    } catch (error) {
      const err = error as Error;
      toast.error(err.message || "Failed to start backtester.");
    }
  };

  if (loading && !backtestStatus) {
    return (
      <div className="flex h-full items-center justify-center">
        <Loader2 className="h-8 w-8 animate-spin text-emerald-500" />
      </div>
    );
  }

  return (
    <div className="p-6 space-y-8 max-w-7xl mx-auto pb-20">
      <div className="flex flex-col md:flex-row md:items-end justify-between gap-4">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <BarChart3 className="h-5 w-5 text-emerald-500" />
            <span className="text-xs font-bold text-emerald-500 uppercase tracking-widest">
              Simulation Environment
            </span>
          </div>
          <h1 className="text-3xl font-black text-zinc-100 tracking-tight">
            Backtest Studio
          </h1>
          <p className="text-sm text-zinc-500 mt-1 max-w-xl">
            Simulate historical market conditions against the I.A.R.A. model
            using configurable entry thresholds.
          </p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6 mt-6">
        {/* Backtester Config */}
        <Card className="lg:col-span-1 bg-zinc-900/40 border-zinc-800/60 backdrop-blur-md">
          <CardHeader className="border-b border-zinc-800/60 py-4">
            <CardTitle className="text-sm font-bold text-zinc-300 flex items-center gap-2">
              <BarChart3 className="h-4 w-4 text-emerald-500" />
              Sniper Validation Lab
            </CardTitle>
          </CardHeader>
          <CardContent className="p-4 space-y-4">
            <div className="space-y-2">
              <label className="text-[10px] uppercase font-bold text-zinc-500">
                Validation Dataset
              </label>
              <SearchableSelect
                options={datasets.map((d) => ({
                  value: d.symbol,
                  label: d.symbol,
                }))}
                value={backtestProtocol.dataset}
                onChange={(val) =>
                  setBacktestProtocol({ ...backtestProtocol, dataset: val })
                }
                placeholder="-- Select Test Shard --"
                searchPlaceholder="Search Datasets..."
              />
            </div>

            <div className="space-y-2">
              <label className="text-[10px] uppercase font-bold text-zinc-500">
                Brain Model (Weights)
              </label>
              <SearchableSelect
                options={models.map((m) => ({ value: m.name, label: m.name }))}
                value={backtestProtocol.model_name}
                onChange={(val) =>
                  setBacktestProtocol({ ...backtestProtocol, model_name: val })
                }
                placeholder="Select a Model"
                searchPlaceholder="Search weights..."
              />
            </div>

            <div className="space-y-2">
              <label className="flex justify-between text-[10px] uppercase font-bold text-zinc-500">
                <span>Sniper Threshold</span>
                <span className="text-emerald-400">
                  {(backtestProtocol.threshold * 100)?.toFixed(0)}%
                </span>
              </label>
              <input
                type="range"
                min="0.5"
                max="0.99"
                step="0.01"
                className="w-full accent-emerald-500"
                value={backtestProtocol.threshold}
                onChange={(e) =>
                  setBacktestProtocol({
                    ...backtestProtocol,
                    threshold: parseFloat(e.target.value),
                  })
                }
              />
              <p className="text-[9px] text-zinc-600 leading-tight">
                Minimum AI confidence required to execute a simulated trade.
                Higher threshold = fewer trades, higher precision.
              </p>
            </div>

            <Button
              className="w-full mt-4 h-10 font-bold bg-emerald-600 hover:bg-emerald-500"
              disabled={backtestStatus?.is_running || !backtestProtocol.dataset}
              onClick={handleRunBacktest}
            >
              <Play className="mr-2 h-4 w-4" />
              {backtestStatus?.is_running
                ? "Backtesting..."
                : "Run Historical Simulation"}
            </Button>
          </CardContent>
        </Card>

        {/* Backtester Results Console */}
        <Card className="lg:col-span-3 bg-zinc-900/40 border-zinc-800/60 backdrop-blur-md">
          <CardHeader className="border-b border-zinc-800/60 py-4 flex flex-row justify-between items-center">
            <CardTitle className="text-sm font-bold text-zinc-300 flex items-center gap-2">
              <Terminal className="h-4 w-4 text-emerald-500" />
              Backtest Telemetry
            </CardTitle>
            {backtestStatus?.is_running && (
              <span className="text-[10px] font-mono font-bold text-emerald-500 animate-pulse">
                SIMULATING TRADES...
              </span>
            )}
          </CardHeader>
          <CardContent className="p-4 flex flex-col md:flex-row gap-4 h-[600px]">
            {/* Results Window (Moved to Left for better UX on Backtest) */}
            <div className="w-full md:w-64 bg-zinc-950/60 border border-zinc-800 rounded-lg p-4 flex flex-col gap-3">
              <p className="text-[10px] font-bold text-indigo-400 uppercase tracking-widest text-center border-b border-zinc-800 pb-2">
                Simulation Results
              </p>

              {backtestStatus?.results ? (
                <div className="space-y-4 mt-2">
                  <div>
                    <p className="text-[9px] uppercase font-bold text-zinc-500 mb-1">
                      Total Return (PnL)
                    </p>
                    <p
                      className={cn(
                        "text-2xl font-black font-mono tracking-tighter",
                        backtestStatus.results.total_pnl >= 0
                          ? "text-emerald-400"
                          : "text-rose-400",
                      )}
                    >
                      {backtestStatus.results.total_pnl >= 0 ? "+" : ""}
                      {backtestStatus.results.total_pnl?.toFixed(2)}%
                    </p>
                  </div>

                  <div className="flex justify-between items-center">
                    <div>
                      <p className="text-[9px] uppercase font-bold text-zinc-500">
                        Win Rate
                      </p>
                      <p className="text-sm font-bold text-zinc-300">
                        {(backtestStatus.results.win_rate * 100)?.toFixed(1)}%
                      </p>
                    </div>
                    <div className="text-right">
                      <p className="text-[9px] uppercase font-bold text-zinc-500">
                        Max Drawdown
                      </p>
                      <p className="text-sm font-bold text-rose-400">
                        {backtestStatus.results.max_drawdown?.toFixed(2)}%
                      </p>
                    </div>
                  </div>

                  <div className="border-t border-zinc-800 pt-3">
                    <p className="text-[9px] uppercase font-bold text-zinc-500 flex justify-between">
                      <span>Executed Trades</span>
                      <span className="text-zinc-300">
                        {backtestStatus.results.total_trades}
                      </span>
                    </p>
                  </div>
                </div>
              ) : (
                <div className="flex-1 flex items-center justify-center text-[10px] text-zinc-700 italic text-center">
                  Results will appear here.
                </div>
              )}
            </div>

            {/* Logs Window */}
            <div
              ref={backtestLogsContainerRef}
              className="flex-1 bg-black/60 rounded-lg p-4 font-mono text-[10px] overflow-y-auto space-y-1 border border-zinc-800 shadow-inner text-emerald-400/90 [scrollbar-width:none]"
            >
              {backtestStatus?.logs && backtestStatus.logs.length > 0 ? (
                backtestStatus.logs.map((log, i) => (
                  <div key={`bt-log-${i}`} className="text-emerald-500/80">
                    <span className="text-zinc-700 mr-2">$</span>
                    {log}
                  </div>
                ))
              ) : (
                <div className="text-zinc-700 italic text-xs">
                  Run a simulation to view historical execution logs...
                </div>
              )}
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
