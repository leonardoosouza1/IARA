"use client";

import React, { useState, useEffect } from "react";
import Link from "next/link";
import { useFleetStore } from "@/hooks/use-fleet-store";
import { api } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Plus, Trash2, Power, Crosshair, Loader2 } from "lucide-react";
import { toast } from "sonner";
import { SearchableSelect, type Option } from "@/components/ui/searchable-select";

const MARKET_TYPES: Option[] = [
  { value: "crypto", label: "Crypto" },
  { value: "forex", label: "Forex" },
  { value: "stock", label: "Stock" },
];

const COMMON_SYMBOLS: Option[] = [
  // Crypto
  { value: "BTC/USD", label: "BTC/USD - Bitcoin" },
  { value: "ETH/USD", label: "ETH/USD - Ethereum" },
  { value: "SOL/USD", label: "SOL/USD - Solana" },
  { value: "XRP/USD", label: "XRP/USD - Ripple" },
  { value: "ADA/USD", label: "ADA/USD - Cardano" },
  { value: "DOT/USD", label: "DOT/USD - Polkadot" },
  { value: "DOGE/USD", label: "DOGE/USD - Dogecoin" },
  { value: "AVAX/USD", label: "AVAX/USD - Avalanche" },
  { value: "LINK/USD", label: "LINK/USD - Chainlink" },
  // Forex
  { value: "EUR/USD", label: "EUR/USD - Euro" },
  { value: "GBP/USD", label: "GBP/USD - British Pound" },
  { value: "USD/JPY", label: "USD/JPY - Japanese Yen" },
  { value: "AUD/USD", label: "AUD/USD - Australian Dollar" },
  { value: "USD/CAD", label: "USD/CAD - Canadian Dollar" },
  { value: "NZD/USD", label: "NZD/USD - New Zealand Dollar" },
  { value: "USD/CHF", label: "USD/CHF - Swiss Franc" },
];

interface AgentProfile {
  id: string;
  symbol: string;
  market_type: string;
  model_version: string;
  is_active: boolean;
}

export default function AgentsPage() {
  const [profiles, setProfiles] = useState<AgentProfile[]>([]);
  const [loading, setLoading] = useState(true);
  
  // Modal state
  const [showModal, setShowModal] = useState(false);
  const [newSymbol, setNewSymbol] = useState("ETH/USD");
  const [newMarketType, setNewMarketType] = useState("crypto");
  const [newModel, setNewModel] = useState("");
  const [availableModels, setAvailableModels] = useState<Option[]>([]);

  const storeAgents = useFleetStore((s) => s.agents);

  useEffect(() => {
    fetchProfiles();
    fetchModels();
  }, []);

  const fetchModels = async () => {
    try {
      const res = await api.get<Option[]>("/api/v1/agents/models");
      setAvailableModels(res);
      if (res.length > 0) {
        setNewModel(res[0].value);
      }
    } catch {
      toast.error("Failed to fetch available models");
    }
  };

  const fetchProfiles = async () => {
    try {
      const res = await api.get<{ data: AgentProfile[] } | AgentProfile[]>("/api/v1/agents");
      // Need to handle if it's returning the array directly vs an object with a data property
      if (Array.isArray(res)) {
        setProfiles(res);
      } else if (res && typeof res === 'object' && 'data' in res && Array.isArray(res.data)) {
        setProfiles(res.data);
      } else {
        setProfiles([]);
      }
    } catch {
      toast.error("Failed to fetch agent profiles");
    } finally {
      setLoading(false);
    }
  };

  const handleCreate = async () => {
    try {
      const id = "agent-" + newSymbol.toLowerCase().replace("/", "-");
      await api.post("/api/v1/agents", {
        id,
        symbol: newSymbol,
        market_type: newMarketType,
        model_version: newModel || "SwarmBrain v9.7" // Fallback
      });
      toast.success("Agent profile created!");
      setShowModal(false);
      fetchProfiles();
    } catch (e: unknown) {
      const err = e as { response?: { data?: { detail?: string } } };
      toast.error(err.response?.data?.detail || "Failed to create agent");
    }
  };

  const handleDelete = async (id: string) => {
    try {
      await api.delete(`/api/v1/agents/${id}`);
      toast.success("Agent profile deleted");
      fetchProfiles();
    } catch {
      toast.error("Failed to delete agent");
    }
  };

  const handleToggle = async (id: string, currentlyActive: boolean) => {
    try {
      await api.put(`/api/v1/agents/${id}/toggle`);
      toast.success(`Agent ${currentlyActive ? 'deactivated' : 'activated'} successfully.`);
      fetchProfiles();
    } catch {
      toast.error("Failed to toggle agent");
    }
  };

  if (loading) {
    return (
      <div className="p-6 flex items-center justify-center h-full">
        <Loader2 className="animate-spin h-8 w-8 text-indigo-500" />
      </div>
    );
  }

  return (
    <div className="p-6 space-y-6 relative">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-zinc-100 tracking-tight">Agents Panel</h1>
          <p className="text-sm text-zinc-500 mt-1">Manage and configure your dynamic trading fleet</p>
        </div>
        <Button 
          onClick={() => setShowModal(true)}
          className="bg-indigo-600 hover:bg-indigo-700 text-white"
        >
          <Plus className="mr-2 h-4 w-4" /> New Agent
        </Button>
      </div>

      {!loading && profiles?.length === 0 ? (
        <div className="flex flex-col items-center justify-center p-12 text-center rounded-xl border border-zinc-800/60 bg-zinc-900/20">
          <div className="flex h-16 w-16 items-center justify-center rounded-full bg-zinc-800/50 mb-4">
            <Crosshair className="h-8 w-8 text-zinc-500" />
          </div>
          <h3 className="text-xl font-bold text-zinc-300">No Agents Configured</h3>
          <p className="text-sm text-zinc-500 max-w-sm mt-2 mb-6">
            You don&apos;t have any trading agents set up yet. Create your first agent to start monitoring the market.
          </p>
          <Button 
            onClick={() => setShowModal(true)}
            className="bg-indigo-600 hover:bg-indigo-700 text-white"
          >
            <Plus className="mr-2 h-4 w-4" /> Create First Agent
          </Button>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
          {profiles?.map((profile) => {
            // Find live state if running
            const liveAgent = storeAgents.find(a => a.id === profile.id);
            const isLive = !!liveAgent;
            const status = isLive ? liveAgent.status : (profile.is_active ? "waiting" : "inactive");
            
            return (
              <div key={profile.id} className="group rounded-xl border border-zinc-800/60 bg-zinc-900/40 p-5 hover:border-indigo-500/30 transition-all duration-200">
                <div className="flex items-center justify-between mb-4">
                  <div className="flex items-center gap-3">
                    <div className={`h-3 w-3 rounded-full ${isLive ? 'bg-emerald-500 animate-pulse' : (profile.is_active ? 'bg-amber-500' : 'bg-zinc-600')}`} />
                    <span className="text-lg font-bold text-zinc-100">
                      {profile.symbol}
                    </span>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="text-xs text-zinc-500 font-mono uppercase border border-zinc-800 px-2 py-0.5 rounded-md">
                      {profile.market_type}
                    </span>
                  </div>
                </div>
                
                <div className="space-y-4">
                  <div className="flex justify-between items-center text-sm">
                    <span className="text-zinc-500">Model</span>
                    <span className="text-zinc-300 font-mono text-xs">{profile.model_version}</span>
                  </div>
                  
                  <div className="flex justify-between items-center text-sm">
                    <span className="text-zinc-500">Live Status</span>
                    <span className={`font-mono text-xs ${isLive ? 'text-emerald-400' : 'text-zinc-500'}`}>
                      {status.toUpperCase()}
                    </span>
                  </div>
                </div>

                <div className="mt-6 flex items-center justify-between gap-2 border-t border-zinc-800/60 pt-4">
                  <div className="flex gap-2">
                    <Button 
                      variant="outline" 
                      size="sm"
                      onClick={() => handleToggle(profile.id, profile.is_active)}
                      className={`border-zinc-800 ${profile.is_active ? 'text-amber-400 hover:text-amber-300' : 'text-emerald-400 hover:text-emerald-300'}`}
                    >
                      <Power className="h-4 w-4" />
                    </Button>
                    <Button 
                      variant="outline" 
                      size="sm"
                      onClick={() => handleDelete(profile.id)}
                      className="border-zinc-800 text-red-500 hover:text-red-400 hover:bg-red-500/10"
                    >
                      <Trash2 className="h-4 w-4" />
                    </Button>
                  </div>
                  
                  {isLive && (
                    <Link href={`/dashboard/agents/${profile.id}`}>
                      <Button variant="outline" size="sm" className="bg-indigo-500/10 text-indigo-400 border-indigo-500/30 hover:bg-indigo-500/20">
                        <Crosshair className="mr-2 h-4 w-4" /> Enter Agent
                      </Button>
                    </Link>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      )}

      {/* Tailwind Modal */}
      {showModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm">
          <div className="w-full max-w-md bg-zinc-950 border border-zinc-800 p-6 rounded-xl shadow-2xl">
            <h2 className="text-xl font-bold text-zinc-100 mb-4">Add New Agent</h2>
            
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-zinc-400 mb-1 flex items-center gap-2">
                  <span>Symbol Pair</span>
                </label>
                <SearchableSelect
                  options={COMMON_SYMBOLS}
                  value={newSymbol}
                  onChange={setNewSymbol}
                  placeholder="Select a symbol pair..."
                  searchPlaceholder="Search pair (e.g. BTC)..."
                />
              </div>
              
              <div>
                <label className="block text-sm font-medium text-zinc-400 mb-1">Market Type</label>
                <SearchableSelect
                  options={MARKET_TYPES}
                  value={newMarketType}
                  onChange={setNewMarketType}
                  placeholder="Select market type..."
                  searchPlaceholder="Search market type..."
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-zinc-400 mb-1">AI Model Checkpoint</label>
                <SearchableSelect
                  options={availableModels}
                  value={newModel}
                  onChange={setNewModel}
                  placeholder="Select trained AI model..."
                  searchPlaceholder="Search checkpoints (e.g. BTC)..."
                />
              </div>
            </div>

            <div className="mt-6 flex justify-end gap-3">
              <Button variant="ghost" onClick={() => setShowModal(false)} className="text-zinc-400">
                Cancel
              </Button>
              <Button onClick={handleCreate} className="bg-indigo-600 hover:bg-indigo-700 text-white">
                Create Agent
              </Button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
