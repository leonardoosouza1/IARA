"use client";

import { useState, useEffect, useCallback } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  FlaskConical,
  Play,
  Loader2,
  TrendingUp,
  TrendingDown,
  History,
  Activity,
  Award,
  Database,
  Download,
  Terminal,
  RefreshCw,
  Trash2,
  Trash,
  Star,
  Square,
} from "lucide-react";
import { api } from "@/lib/api";
import { toast } from "sonner";
import { cn } from "@/lib/utils";
import {
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  AreaChart,
  Area,
} from "recharts";

interface TrainingStatus {
  is_training: boolean;
  last_loss: number | null;
  last_acc: number | null;
  last_prec: number | null;
  last_run: string | null;
  message: string;
  progress: number;
  logs: string[];
}

interface DataStatus {
  is_fetching: boolean;
  message: string;
  logs: string[];
}

interface ModelInfo {
  name: string;
  size: string;
  last_modified: string;
}

interface DatasetInfo {
  symbol: string;
  shards: number;
  path: string;
}

interface TradeRecord {
  id: number;
  symbol: string;
  side: string;
  qty: number;
  price: number;
  pnl: number | null;
  timestamp: string;
}

interface MetricPoint {
  time: string;
  loss: number;
  acc: number;
  prec: number;
}

export default function BrainLabPage() {
  const [trainingStatus, setTrainingStatus] = useState<TrainingStatus | null>(
    null,
  );
  const [dataStatus, setDataStatus] = useState<DataStatus | null>(null);
  const [trades, setTrades] = useState<TradeRecord[]>([]);
  const [loading, setLoading] = useState(true);
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [datasets, setDatasets] = useState<DatasetInfo[]>([]);
  const [metricHistory, setMetricHistory] = useState<MetricPoint[]>([]);

  const [protocol, setProtocol] = useState({
    mode: "finetune",
    base_model: "active_model.pth",
    symbol: "BTC/USD",
    batch_size: 256,
    epochs: 5,
    learning_rate: 0.0001,
    data_source: "journal", // "journal" or "historical"
    dataset_name: "",
  });

  const [fetchRequest, setFetchRequest] = useState({
    symbol: "BTC/USD",
    days: 30,
    timeframe: "1Min",
  });

  const fetchData = useCallback(async () => {
    try {
      const [statusRes, tradesRes, modelsRes, dataStatusRes, datasetsRes] =
        await Promise.all([
          api.get<TrainingStatus>("/api/v1/brain/status"),
          api.get<TradeRecord[]>("/api/v1/brain/trades"),
          api.get<ModelInfo[]>("/api/v1/brain/models"),
          api.get<DataStatus>("/api/v1/brain/data/status"),
          api.get<DatasetInfo[]>("/api/v1/brain/data/inventory"),
        ]);
      setTrainingStatus(statusRes);
      setTrades(tradesRes || []);
      setModels(modelsRes || []);
      setDataStatus(dataStatusRes);
      setDatasets(datasetsRes || []);

      // Update Metric History for Chart
      if (statusRes.is_training && statusRes.last_loss !== null) {
        setMetricHistory((prev) => {
          const last = prev[prev.length - 1];
          // Only add if it's a new batch to avoid redundant points
          if (
            last?.loss === statusRes.last_loss &&
            last?.acc === statusRes.last_acc
          )
            return prev;

          const newHistory = [
            ...prev,
            {
              time: new Date().toLocaleTimeString([], {
                hour: "2-digit",
                minute: "2-digit",
                second: "2-digit",
              }),
              loss: statusRes.last_loss as number,
              acc: statusRes.last_acc || 0,
              prec: statusRes.last_prec || 0,
            },
          ];
          return newHistory.slice(-50); // Keep last 50 points
        });
      }

      // Auto-select first dataset if none selected and source is historical
      if (
        datasetsRes?.length > 0 &&
        protocol.data_source === "historical" &&
        !protocol.dataset_name
      ) {
        setProtocol((prev) => ({
          ...prev,
          dataset_name: datasetsRes[0].symbol,
        }));
      }
    } catch (error) {
      console.error("Failed to fetch brain lab data:", error);
      if (!loading) toast.error("Failed to connect to Brain Lab services.");
    } finally {
      setLoading(false);
    }
  }, [loading, protocol.data_source, protocol.dataset_name]);

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 5000);
    return () => clearInterval(interval);
  }, [fetchData]);

  const handleStartTraining = async () => {
    try {
      const payload = {
        ...protocol,
        base_model:
          protocol.base_model === "FRESH_RANDOM" ? null : protocol.base_model,
      };
      await api.post("/api/v1/brain/train", payload);
      toast.success("Training session initiated!");
      fetchData();
    } catch (error: any) {
      toast.error(error.message || "Failed to start training session.");
    }
  };

  const handleStopTraining = async () => {
    try {
      await api.post("/api/v1/brain/stop");
      toast.success("Stop signal sent. Saving best model...");
      fetchData();
    } catch (error: any) {
      toast.error(error.message || "Failed to stop training.");
    }
  };

  const handleFetchData = async () => {
    try {
      await api.post("/api/v1/brain/data/fetch", fetchRequest);
      toast.success("Data Pipeline initiated! Check logs below.");
      fetchData();
    } catch (error: any) {
      toast.error(error.message || "Failed to start data pipeline.");
    }
  };

  const handleDeleteDataset = async (symbol: string) => {
    if (!confirm(`Are you sure you want to delete dataset ${symbol}?`)) return;
    try {
      await api.delete(`/api/v1/brain/data/${symbol}`);
      toast.success("Dataset deleted.");
      fetchData();
    } catch (error: any) {
      toast.error(error.message || "Failed to delete dataset.");
    }
  };

  const handleDeleteModel = async (name: string) => {
    if (name === "active_model.pth") return;
    if (!confirm(`Are you sure you want to delete model ${name}?`)) return;
    try {
      await api.delete(`/api/v1/brain/models/${name}`);
      toast.success("Model deleted.");
      fetchData();
    } catch (error: any) {
      toast.error(error.message || "Failed to delete model.");
    }
  };

  const handleActivateModel = async (name: string) => {
    try {
      await api.post(`/api/v1/brain/models/activate`, { name });
      toast.success(`Model ${name} is now ACTIVE.`);
      fetchData();
    } catch (error: any) {
      toast.error(error.message || "Failed to activate model.");
    }
  };

  if (loading && !trainingStatus) {
    return (
      <div className="flex h-full items-center justify-center">
        <Loader2 className="h-8 w-8 animate-spin text-indigo-500" />
      </div>
    );
  }

  return (
    <div className="p-6 space-y-8 max-w-7xl mx-auto pb-20">
      <div className="flex flex-col md:flex-row md:items-end justify-between gap-4">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <FlaskConical className="h-5 w-5 text-indigo-500" />
            <span className="text-xs font-bold text-indigo-500 uppercase tracking-widest">
              Scientific Protocol v2.8
            </span>
          </div>
          <h1 className="text-3xl font-black text-zinc-100 tracking-tight">
            Brain Lab
          </h1>
          <p className="text-sm text-zinc-500 mt-1 max-w-xl">
            Configure training protocols and manage historical datasets. Train
            on live trade feedback or supervised market history.
          </p>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-4">
        <MetricCard
          icon={<Activity className="h-5 w-5" />}
          label="Training Loss"
          value={
            trainingStatus?.last_loss
              ? trainingStatus.last_loss.toFixed(4)
              : "0.0000"
          }
          color="indigo"
          subValue={trainingStatus?.is_training ? "Computing..." : "Idle"}
        />
        <MetricCard
          icon={<TrendingUp className="h-5 w-5" />}
          label="Precision (Avg)"
          value={
            trainingStatus?.last_prec
              ? `${(trainingStatus.last_prec * 100).toFixed(1)}%`
              : "0.0%"
          }
          color="emerald"
        />
        <MetricCard
          icon={<Award className="h-5 w-5" />}
          label="Accuracy"
          value={
            trainingStatus?.last_acc
              ? `${(trainingStatus.last_acc * 100).toFixed(1)}%`
              : "0.0%"
          }
          color="amber"
        />
        <MetricCard
          icon={<Database className="h-5 w-5" />}
          label="Datasets Ready"
          value={datasets.length.toString()}
          color="purple"
        />
        <MetricCard
          icon={<History className="h-5 w-5" />}
          label="Journal Size"
          value={trades.length.toString()}
          color="zinc"
        />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        {/* Protocol Designer */}
        <Card className="lg:col-span-1 bg-zinc-900/40 border-zinc-800/60 backdrop-blur-md">
          <CardHeader className="border-b border-zinc-800/60 py-4">
            <CardTitle className="text-sm font-bold text-zinc-300 flex items-center gap-2">
              <RefreshCw className="h-4 w-4 text-indigo-500" />
              Protocol Designer
            </CardTitle>
          </CardHeader>
          <CardContent className="p-4 space-y-4">
            <div className="space-y-2">
              <label className="text-[10px] uppercase font-bold text-zinc-500">
                Data Source
              </label>
              <div className="grid grid-cols-2 gap-2">
                <Button
                  size="sm"
                  variant={
                    protocol.data_source === "journal" ? "default" : "outline"
                  }
                  onClick={() =>
                    setProtocol({ ...protocol, data_source: "journal" })
                  }
                  className={cn(
                    "h-8 text-xs",
                    protocol.data_source === "journal"
                      ? "bg-indigo-600 hover:bg-indigo-500"
                      : "border-zinc-800",
                  )}
                >
                  Live Journal
                </Button>
                <Button
                  size="sm"
                  variant={
                    protocol.data_source === "historical"
                      ? "default"
                      : "outline"
                  }
                  onClick={() =>
                    setProtocol({ ...protocol, data_source: "historical" })
                  }
                  className={cn(
                    "h-8 text-xs",
                    protocol.data_source === "historical"
                      ? "bg-indigo-600 hover:bg-indigo-500"
                      : "border-zinc-800",
                  )}
                >
                  Historical
                </Button>
              </div>
            </div>

            {protocol.data_source === "historical" && (
              <div className="space-y-2 animate-in fade-in slide-in-from-top-1">
                <label className="text-[10px] uppercase font-bold text-zinc-500">
                  Select Dataset
                </label>
                <select
                  title="Source Dataset"
                  className="w-full bg-zinc-950 border border-zinc-800 rounded px-2 py-1.5 text-xs text-zinc-200 outline-none"
                  value={protocol.dataset_name}
                  onChange={(e) =>
                    setProtocol({ ...protocol, dataset_name: e.target.value })
                  }
                >
                  <option value="">-- Choose Dataset --</option>
                  {datasets.map((d) => (
                    <option key={d.symbol} value={d.symbol}>
                      {d.symbol} ({d.shards} shards)
                    </option>
                  ))}
                </select>
              </div>
            )}

            <div className="space-y-2">
              <label className="text-[10px] uppercase font-bold text-zinc-500">
                Base Model
              </label>
              <select
                title="Base Model"
                className="w-full bg-zinc-950 border border-zinc-800 rounded px-2 py-1.5 text-xs text-zinc-200 outline-none"
                value={protocol.base_model}
                onChange={(e) =>
                  setProtocol({ ...protocol, base_model: e.target.value })
                }
              >
                <option
                  value="FRESH_RANDOM"
                  className="text-indigo-400 font-bold"
                >
                  --- FRESH MODEL (RANDOM) ---
                </option>
                {models.map((m) => (
                  <option key={m.name} value={m.name}>
                    {m.name}
                  </option>
                ))}
              </select>
            </div>

            <div className="grid grid-cols-2 gap-3">
              <div className="space-y-1">
                <label className="text-[10px] uppercase font-bold text-zinc-500">
                  Batch Size
                </label>
                <input
                  type="number"
                  className="w-full bg-zinc-950 border border-zinc-800 rounded px-2 py-1.5 text-xs text-zinc-200"
                  value={protocol.batch_size}
                  onChange={(e) =>
                    setProtocol({
                      ...protocol,
                      batch_size: parseInt(e.target.value),
                    })
                  }
                />
              </div>
              <div className="space-y-1">
                <label className="text-[10px] uppercase font-bold text-zinc-500">
                  Epochs
                </label>
                <input
                  type="number"
                  className="w-full bg-zinc-950 border border-zinc-800 rounded px-2 py-1.5 text-xs text-zinc-200"
                  value={protocol.epochs}
                  onChange={(e) =>
                    setProtocol({
                      ...protocol,
                      epochs: parseInt(e.target.value),
                    })
                  }
                />
              </div>
            </div>

            <div className="space-y-1">
              <label className="text-[10px] uppercase font-bold text-zinc-500">
                Learning Rate
              </label>
              <input
                type="number"
                step="0.0001"
                className="w-full bg-zinc-950 border border-zinc-800 rounded px-2 py-1.5 text-xs text-zinc-200"
                value={protocol.learning_rate}
                onChange={(e) =>
                  setProtocol({
                    ...protocol,
                    learning_rate: parseFloat(e.target.value),
                  })
                }
              />
            </div>

            {trainingStatus?.is_training ? (
              <Button
                className="w-full mt-4 h-10 font-bold bg-red-600 hover:bg-red-500"
                onClick={handleStopTraining}
              >
                <Square className="mr-2 h-4 w-4" /> Stop & Save Best
              </Button>
            ) : (
              <Button
                className="w-full mt-4 h-10 font-bold bg-indigo-600 hover:bg-indigo-500"
                disabled={
                  protocol.data_source === "historical" &&
                  !protocol.dataset_name
                }
                onClick={handleStartTraining}
              >
                <Play className="mr-2 h-4 w-4" /> Execute Protocol
              </Button>
            )}

            <Button
              variant="ghost"
              className="w-full text-[10px] text-zinc-500 hover:text-indigo-400"
              onClick={() =>
                setProtocol({
                  ...protocol,
                  mode: "finetune",
                  base_model: "active_model.pth",
                  batch_size: 256,
                  epochs: 20,
                  learning_rate: 0.0001,
                })
              }
            >
              Restore &quot;v9&quot; Gladiator Presets
            </Button>
          </CardContent>
        </Card>

        {/* Live Status & Console */}
        <Card className="lg:col-span-3 bg-zinc-900/40 border-zinc-800/60 backdrop-blur-md">
          <CardHeader className="border-b border-zinc-800/60 py-4 flex flex-row justify-between items-center">
            <CardTitle className="text-sm font-bold text-zinc-300 flex items-center gap-2">
              <Activity className="h-4 w-4 text-emerald-500" />
              Live Telemetry
            </CardTitle>
            <div className="flex items-center gap-4">
              <span className="text-[10px] font-mono text-zinc-500">
                {trainingStatus?.is_training ? "COMPUTING" : "IDLE"}
              </span>
              <div className="w-32 h-1.5 bg-zinc-800 rounded-full overflow-hidden">
                <div
                  className="h-full bg-emerald-500 transition-all duration-500"
                  style={{ width: `${trainingStatus?.progress || 0}%` }}
                />
              </div>
            </div>
          </CardHeader>
          <CardContent className="p-4 flex flex-col h-[500px] gap-4">
            <div className="flex gap-4 h-1/2">
              {/* Visual Console */}
              <div className="flex-1 bg-black/60 rounded-lg p-4 font-mono text-[10px] overflow-y-auto space-y-1 border border-zinc-800 shadow-inner text-emerald-400/90 [scrollbar-width:none]">
                {trainingStatus?.logs && trainingStatus.logs.length > 0 ? (
                  trainingStatus.logs.map((log, i) => (
                    <div
                      key={i}
                      className={cn(
                        "transition-opacity duration-300",
                        i === trainingStatus.logs.length - 1
                          ? "text-emerald-400 font-bold"
                          : "text-emerald-500/40",
                      )}
                    >
                      <span className="text-zinc-700 mr-2">$</span>
                      {log}
                    </div>
                  ))
                ) : (
                  <div className="text-zinc-700 italic text-xs">
                    Waiting for protocol execution command...
                  </div>
                )}
              </div>

              {/* Swarm Insight (Macro/Micro) */}
              <div className="w-64 bg-zinc-950/40 border border-zinc-800 rounded-lg p-4 flex flex-col items-center justify-center space-y-4">
                <p className="text-[10px] font-bold text-zinc-500 uppercase tracking-widest text-center">
                  Swarm Thoughts
                </p>
                <div className="flex flex-col gap-2 w-full">
                  {["M5", "M15", "M30", "H1", "H4"].map((tf, i) => {
                    // Extract last vote from logs if available
                    const lastLog =
                      trainingStatus?.logs[trainingStatus.logs.length - 1] ||
                      "";
                    const swarmMatch = lastLog.match(/Swarm: ([L|S|W-]+)/);
                    const votes = swarmMatch
                      ? swarmMatch[1].split("-")
                      : ["W", "W", "W", "W", "W"];
                    const vote = votes[i] || "W";

                    return (
                      <div
                        key={tf}
                        className="flex items-center justify-between bg-black/20 px-3 py-1.5 rounded border border-zinc-800/50"
                      >
                        <span className="text-[10px] font-bold text-zinc-500">
                          {tf} Specialist
                        </span>
                        <Badge
                          className={cn(
                            "text-[9px] px-1.5 h-4 border-none",
                            vote === "L"
                              ? "bg-emerald-500/20 text-emerald-400"
                              : vote === "S"
                                ? "bg-rose-500/20 text-rose-400"
                                : "bg-zinc-800 text-zinc-500",
                          )}
                        >
                          {vote === "L"
                            ? "LONG"
                            : vote === "S"
                              ? "SHORT"
                              : "WAIT"}
                        </Badge>
                      </div>
                    );
                  })}
                </div>
                <div className="pt-2 border-t border-zinc-800 w-full text-center">
                  <p className="text-[9px] text-zinc-600 italic">
                    Coordinator Confidence
                  </p>
                  <div className="mt-1 h-1 w-full bg-zinc-900 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-indigo-500"
                      style={{
                        width: `${(trainingStatus?.last_acc || 0) * 100}%`,
                      }}
                    />
                  </div>
                </div>
              </div>
            </div>

            {/* Metrics Chart */}
            <div className="flex-1 min-h-0 bg-zinc-950/20 rounded-lg border border-zinc-800/50 p-2">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={metricHistory}>
                  <defs>
                    <linearGradient id="colorLoss" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#6366f1" stopOpacity={0.1} />
                      <stop offset="95%" stopColor="#6366f1" stopOpacity={0} />
                    </linearGradient>
                    <linearGradient id="colorAcc" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#10b981" stopOpacity={0.1} />
                      <stop offset="95%" stopColor="#10b981" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid
                    strokeDasharray="3 3"
                    stroke="#18181b"
                    vertical={false}
                  />
                  <XAxis dataKey="time" hide />
                  <YAxis yAxisId="left" hide domain={[0, "auto"]} />
                  <YAxis yAxisId="right" hide domain={[0, 1]} />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: "#09090b",
                      border: "1px solid #27272a",
                      borderRadius: "8px",
                      fontSize: "10px",
                    }}
                    itemStyle={{ padding: "0px" }}
                  />
                  <Area
                    yAxisId="left"
                    type="monotone"
                    dataKey="loss"
                    stroke="#6366f1"
                    strokeWidth={2}
                    fillOpacity={1}
                    fill="url(#colorLoss)"
                    name="Loss"
                  />
                  <Area
                    yAxisId="right"
                    type="monotone"
                    dataKey="acc"
                    stroke="#10b981"
                    strokeWidth={2}
                    fillOpacity={1}
                    fill="url(#colorAcc)"
                    name="Accuracy"
                  />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        {/* Dataset Manager */}
        <Card className="lg:col-span-2 bg-zinc-900/30 border-zinc-800/60 backdrop-blur-sm">
          <CardHeader className="border-b border-zinc-800/60 py-4">
            <CardTitle className="text-sm font-bold text-zinc-300 flex items-center gap-2">
              <Database className="h-4 w-4 text-amber-500" />
              Dataset Manager
            </CardTitle>
          </CardHeader>
          <CardContent className="p-4 flex flex-col md:flex-row gap-6">
            <div className="w-full md:w-1/2 space-y-4">
              <div className="p-3 rounded-lg bg-zinc-950/40 border border-zinc-800 space-y-3">
                <p className="text-[10px] uppercase font-bold text-zinc-400 mb-2">
                  Fetch New Data
                </p>

                <div className="space-y-1">
                  <label className="text-[9px] text-zinc-500 uppercase">
                    Symbol
                  </label>
                  <input
                    type="text"
                    className="w-full bg-zinc-900 border border-zinc-800 rounded px-2 py-1.5 text-xs text-zinc-200"
                    placeholder="BTC/USD"
                    value={fetchRequest.symbol}
                    onChange={(e) =>
                      setFetchRequest({
                        ...fetchRequest,
                        symbol: e.target.value,
                      })
                    }
                  />
                </div>

                <div className="grid grid-cols-2 gap-3">
                  <div className="space-y-1">
                    <label className="text-[9px] text-zinc-500 uppercase">
                      Days
                    </label>
                    <input
                      type="number"
                      className="w-full bg-zinc-900 border border-zinc-800 rounded px-2 py-1.5 text-xs text-zinc-200"
                      value={fetchRequest.days}
                      onChange={(e) =>
                        setFetchRequest({
                          ...fetchRequest,
                          days: parseInt(e.target.value),
                        })
                      }
                    />
                  </div>
                  <div className="space-y-1">
                    <label className="text-[9px] text-zinc-500 uppercase">
                      TF
                    </label>
                    <select
                      title="Timeframe"
                      className="w-full bg-zinc-900 border border-zinc-800 rounded px-2 py-1.5 text-xs text-zinc-200"
                      value={fetchRequest.timeframe}
                      onChange={(e) =>
                        setFetchRequest({
                          ...fetchRequest,
                          timeframe: e.target.value,
                        })
                      }
                    >
                      <option value="1Min">1 Min</option>
                      <option value="5Min">5 Min</option>
                      <option value="15Min">15 Min</option>
                      <option value="1Hour">1 Hour</option>
                    </select>
                  </div>
                </div>

                <Button
                  className="w-full h-9 bg-amber-600 hover:bg-amber-500 text-xs font-bold"
                  onClick={handleFetchData}
                  disabled={dataStatus?.is_fetching}
                >
                  {dataStatus?.is_fetching ? (
                    <>
                      <Loader2 className="mr-2 h-3 w-3 animate-spin" />{" "}
                      Downloading...
                    </>
                  ) : (
                    <>
                      <Download className="mr-2 h-3 w-3" /> Fetch & Process
                    </>
                  )}
                </Button>
              </div>

              {/* Data Fetch Console */}
              <div className="p-3 rounded-lg bg-black/40 border border-zinc-800 h-32 overflow-y-auto font-mono text-[9px] text-zinc-600 space-y-1 scrollbar-hide">
                <p className="text-zinc-600 mb-1 border-b border-zinc-800 pb-1 flex items-center gap-1">
                  <Terminal className="h-2.5 w-2.5" /> DATAPIPE_LOG
                </p>
                {dataStatus?.logs.map((log, i) => <div key={i}>{log}</div>) || (
                  <div className="italic text-zinc-800">No active process</div>
                )}
              </div>
            </div>

            <div className="w-full md:w-1/2 space-y-3">
              <p className="text-[10px] font-bold text-zinc-500 uppercase tracking-widest">
                Inventory
              </p>
              <div className="space-y-2 h-[280px] overflow-y-auto pr-1 scrollbar-hide">
                {datasets.length === 0 ? (
                  <div className="text-center py-10 border border-dashed border-zinc-800 rounded-lg text-zinc-700 text-xs italic">
                    Repository empty. Fetch data to begin.
                  </div>
                ) : (
                  datasets.map((d) => (
                    <div
                      key={d.symbol}
                      className="p-3 rounded-lg bg-zinc-950/40 border border-zinc-800 flex justify-between items-center group hover:border-amber-500/30 transition-colors"
                    >
                      <div>
                        <p className="text-xs font-bold text-zinc-200">
                          {d.symbol}
                        </p>
                        <p className="text-[9px] font-mono text-zinc-600">
                          {d.shards} structural shards
                        </p>
                      </div>
                      <div className="flex gap-2">
                        <button
                          onClick={() => handleDeleteDataset(d.symbol)}
                          className="p-1.5 rounded bg-zinc-900 border border-zinc-800 text-rose-500 hover:bg-rose-500/10 hover:border-rose-500/40 transition-all opacity-0 group-hover:opacity-100"
                          title="Purge Dataset"
                        >
                          <Trash2 className="h-3 w-3" />
                        </button>
                        <Badge
                          variant="outline"
                          className="text-[8px] border-zinc-800 text-zinc-500 group-hover:text-amber-500 group-hover:border-amber-500/20"
                        >
                          READY
                        </Badge>
                      </div>
                    </div>
                  ))
                )}
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Model Zoo */}
        <Card className="lg:col-span-1 bg-zinc-900/30 border-zinc-800/60 backdrop-blur-sm">
          <CardHeader className="border-b border-zinc-800/60 py-4">
            <CardTitle className="text-sm font-bold text-zinc-300 flex items-center gap-2">
              <Award className="h-4 w-4 text-emerald-500" />
              Model Zoo
            </CardTitle>
          </CardHeader>
          <CardContent className="p-4 h-[400px] overflow-y-auto scrollbar-hide">
            <div className="space-y-3">
              {models.length === 0 ? (
                <p className="text-xs text-zinc-600 text-center py-8">
                  No models found in vault.
                </p>
              ) : (
                models.map((m) => (
                  <div
                    key={m.name}
                    className={cn(
                      "p-3 rounded-lg border transition-all cursor-pointer group relative",
                      m.name === protocol.base_model
                        ? "bg-indigo-500/10 border-indigo-500/40 shadow-lg shadow-indigo-500/10"
                        : m.name === "active_model.pth"
                          ? "bg-emerald-500/5 border-emerald-500/20"
                          : "bg-zinc-950/40 border-zinc-800/60 hover:border-zinc-700",
                    )}
                    onClick={() =>
                      setProtocol({ ...protocol, base_model: m.name })
                    }
                  >
                    <div className="flex justify-between items-start mb-2">
                      <p className="text-xs font-bold text-zinc-200 truncate pr-2">
                        {m.name}
                      </p>
                      <div className="flex gap-1 items-center">
                        {m.name === "active_model.pth" && (
                          <Badge className="bg-emerald-500/20 text-emerald-400 border-none text-[8px] h-4">
                            PRODUCTION
                          </Badge>
                        )}
                        {m.name === protocol.base_model && (
                          <Badge className="bg-indigo-500 text-white border-none text-[8px] h-4">
                            SELECT
                          </Badge>
                        )}
                      </div>
                    </div>

                    <div className="flex justify-between items-end">
                      <div className="text-[10px] text-zinc-600 font-mono">
                        <p>{m.size}</p>
                        <p>{new Date(m.last_modified).toLocaleDateString()}</p>
                      </div>
                      <div className="flex gap-1">
                        {m.name !== "active_model.pth" && (
                          <>
                            <button
                              onClick={(e) => {
                                e.stopPropagation();
                                handleActivateModel(m.name);
                              }}
                              className="p-1 px-1.5 rounded bg-zinc-900 border border-zinc-800 text-emerald-500 hover:bg-emerald-500/10 opacity-0 group-hover:opacity-100"
                              title="Activate as Global Model"
                            >
                              <Star className="h-3 w-3 fill-emerald-500/20" />
                            </button>
                            <button
                              onClick={(e) => {
                                e.stopPropagation();
                                handleDeleteModel(m.name);
                              }}
                              className="p-1 px-1.5 rounded bg-zinc-900 border border-zinc-800 text-rose-500 hover:bg-rose-500/10 opacity-0 group-hover:opacity-100"
                              title="Delete Checkpoint"
                            >
                              <Trash className="h-3 w-3" />
                            </button>
                          </>
                        )}
                      </div>
                    </div>
                  </div>
                ))
              )}
            </div>
          </CardContent>
        </Card>

        {/* Trade Journal Mini */}
        <Card className="lg:col-span-1 bg-zinc-900/30 border-zinc-800/60 backdrop-blur-sm">
          <CardHeader className="border-b border-zinc-800/60 py-4">
            <CardTitle className="text-sm font-bold text-zinc-300 flex items-center gap-2">
              <History className="h-4 w-4 text-purple-500" />
              Recent Journal
            </CardTitle>
          </CardHeader>
          <CardContent className="p-4 h-[400px] overflow-y-auto scrollbar-hide">
            <div className="space-y-3">
              {trades.slice(0, 10).map((t) => (
                <div
                  key={t.id}
                  className="p-2 border-b border-zinc-800/40 flex justify-between items-center"
                >
                  <div>
                    <p className="text-[10px] font-bold text-zinc-300">
                      {t.symbol}
                    </p>
                    <p
                      className={cn(
                        "text-[8px] font-black",
                        t.side === "BUY" ? "text-emerald-500" : "text-rose-500",
                      )}
                    >
                      {t.side}
                    </p>
                  </div>
                  <p
                    className={cn(
                      "text-xs font-mono font-bold",
                      (t.pnl || 0) >= 0 ? "text-emerald-400" : "text-rose-400",
                    )}
                  >
                    {t.pnl
                      ? `${t.pnl > 0 ? "+" : ""}${t.pnl.toFixed(2)}`
                      : "0.00"}
                  </p>
                </div>
              ))}
              {trades.length === 0 && (
                <p className="text-xs text-zinc-700 italic text-center py-8">
                  No recent trades.
                </p>
              )}
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}

function MetricCard({
  icon,
  label,
  value,
  trend,
  color,
  subValue,
}: {
  icon: React.ReactNode;
  label: string;
  value: string;
  trend?: string;
  color: "indigo" | "purple" | "amber" | "emerald" | "zinc";
  subValue?: string;
}) {
  const colors = {
    indigo: "text-indigo-400 bg-indigo-500/10 border-indigo-500/20",
    purple: "text-purple-400 bg-purple-500/10 border-purple-500/20",
    amber: "text-amber-400 bg-amber-500/10 border-amber-500/20",
    emerald: "text-emerald-400 bg-emerald-500/10 border-emerald-500/20",
    zinc: "text-zinc-400 bg-zinc-500/10 border-zinc-500/20",
  };

  return (
    <Card className="bg-zinc-900/30 border-zinc-800/60 p-5 group hover:border-zinc-700 transition-all duration-300">
      <div className="flex items-center justify-between mb-4">
        <div className={cn("p-2 rounded-lg border", colors[color])}>{icon}</div>
        {(trend || subValue) && (
          <div
            className={cn(
              "text-[10px] font-black px-1.5 py-0.5 rounded flex items-center gap-1",
              trend?.startsWith("+") || !trend
                ? "bg-emerald-500/10 text-emerald-500"
                : "bg-rose-500/10 text-rose-500",
            )}
          >
            {trend &&
              (trend.startsWith("+") ? (
                <TrendingUp className="h-2.5 w-2.5" />
              ) : (
                <TrendingDown className="h-2.5 w-2.5" />
              ))}
            {trend || subValue}
          </div>
        )}
      </div>
      <p className="text-[10px] font-bold text-zinc-500 uppercase tracking-widest mb-1">
        {label}
      </p>
      <h3 className="text-2xl font-black text-zinc-100 font-mono tracking-tight">
        {value}
      </h3>
    </Card>
  );
}
