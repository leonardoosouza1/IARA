"use client";

import { useEffect, useRef } from "react";
import { getSocket } from "@/lib/socket";
import { useFleetStore } from "@/hooks/use-fleet-store";
import { api } from "@/lib/api";
import type { MarketTick, CandleData, BrainThought, Agent } from "@/types";

export function useSocket() {
  const initialized = useRef(false);
  const {
    setSocketConnected,
    addTick,
    addCandle,
    addThought,
    updateAgent,
    setAgents,
  } = useFleetStore();

  useEffect(() => {
    if (initialized.current) return;
    initialized.current = true;

    const socket = getSocket();

    async function fetchFleetStatus() {
      try {
        const res = await api.get<{ agents: Agent[] }>("/api/v1/fleet/status");
        setAgents(res.agents);
      } catch (err) {
        console.error("Failed to fetch fleet status:", err);
      }
    }

    socket.on("connect", () => {
      setSocketConnected(true);
      fetchFleetStatus();
    });

    socket.on("disconnect", () => {
      setSocketConnected(false);
    });

    // Market Events
    socket.on("market_tick", (data: MarketTick) => {
      addTick(data.symbol, data);
    });

    socket.on("candle_closed", (data: CandleData & { symbol: string }) => {
      const { symbol, ...candle } = data;
      addCandle(symbol, candle);
    });

    // Brain Events
    socket.on("brain_thought", (data: BrainThought) => {
      addThought(data.agentId, data);
    });

    socket.on("agent_update", (data: Agent) => {
      updateAgent(data.id, data);
    });

    socket.on("fleet_sync", (agents: Agent[]) => {
      setAgents(agents);
    });

    socket.connect();

    return () => {
      socket.off("connect");
      socket.off("disconnect");
      socket.off("market_tick");
      socket.off("candle_closed");
      socket.off("brain_thought");
      socket.off("agent_update");
      socket.off("fleet_sync");
      socket.disconnect();
      initialized.current = false;
    };
  }, [
    setSocketConnected,
    addTick,
    addCandle,
    addThought,
    updateAgent,
    setAgents,
  ]);
}
