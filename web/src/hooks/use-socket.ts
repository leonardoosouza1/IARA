"use client";

import { useEffect, useRef } from "react";
import { getSocket } from "@/lib/socket";
import { useFleetStore } from "@/hooks/use-fleet-store";
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

    socket.on("connect", () => {
      setSocketConnected(true);
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
  }, [setSocketConnected, addTick, addCandle, addThought, updateAgent, setAgents]);
}
