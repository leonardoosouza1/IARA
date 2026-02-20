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
    setCandles,
    addThought,
    updateAgent,
    setAgents,
    agents,
    socketConnected
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
      // Backend might send { symbol, ...candle } or just candle if room is specific
      // Assuming wrapper { symbol, ...candle } based on previous code
      const { symbol, ...candle } = data;
      addCandle(symbol, candle);
    });

    socket.on("historical_candles", (data: { symbol: string; candles: CandleData[] }) => {
      setCandles(data.symbol, data.candles);
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
      socket.off("historical_candles");
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
    setCandles,
    addThought,
    updateAgent,
    setAgents,
  ]);

  // Auto-subscribe to market data channels when agents are loaded
  useEffect(() => {
    const socket = getSocket();
    if (socketConnected && agents.length > 0) {
      agents.forEach((agent) => {
        // Emit subscribe to join the room for this symbol
        // Backend expects { symbol: "BTC/USD" }
        socket.emit("subscribe", { symbol: agent.symbol });

        // Request history for the chart (Fix for late joiners)
        socket.emit("request_history", { symbol: agent.symbol });

      });
    }
  }, [socketConnected, agents]);
}
