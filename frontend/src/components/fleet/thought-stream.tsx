"use client";

import { useEffect, useRef } from "react";
import { cn } from "@/lib/utils";
import { useFleetStore } from "@/hooks/use-fleet-store";
import type { BrainThought } from "@/types";

interface ThoughtStreamProps {
  agentId: string;
}

const levelConfig: Record<BrainThought["level"], { color: string; prefix: string }> = {
  info: { color: "text-zinc-300", prefix: "INFO" },
  warning: { color: "text-amber-400", prefix: "WARN" },
  critical: { color: "text-red-400", prefix: "CRIT" },
  success: { color: "text-emerald-400", prefix: " OK " },
};

// Demo thoughts
const DEMO_THOUGHTS: BrainThought[] = [
  { agentId: "agent-btc-1", timestamp: "2026-02-17T19:00:01Z", step: "LFM_SCAN", text: "Checking support/resistance levels for BTC/USD...", level: "info" },
  { agentId: "agent-btc-1", timestamp: "2026-02-17T19:00:03Z", step: "TOOL_CALL", text: "MarketScanner.check_support_resistance(BTC/USD, 97500) → Support at 96800, Resistance at 98200", level: "success" },
  { agentId: "agent-btc-1", timestamp: "2026-02-17T19:00:05Z", step: "LFM_ANALYZE", text: "Price at 97500, near support. Trend is bullish on H1. Workers consensus: 0.78 (Strong Buy).", level: "info" },
  { agentId: "agent-btc-1", timestamp: "2026-02-17T19:00:06Z", step: "DECISION", text: "BUY Signal (Confidence: 0.82). Entry: 97500, TP: 98200, SL: 96800.", level: "success" },
  { agentId: "agent-btc-1", timestamp: "2026-02-17T19:00:08Z", step: "EXECUTOR", text: "Order submitted to Alpaca. Market BUY 0.01 BTC/USD.", level: "info" },
  { agentId: "agent-btc-1", timestamp: "2026-02-17T19:01:00Z", step: "LFM_SCAN", text: "Monitoring position. Current PnL: +$3.50. Distance to TP: $700.", level: "info" },
  { agentId: "agent-btc-1", timestamp: "2026-02-17T19:02:00Z", step: "SENTINEL", text: "Volatility spike detected. VIX-like metric above threshold.", level: "warning" },
  { agentId: "agent-btc-1", timestamp: "2026-02-17T19:02:05Z", step: "LFM_ANALYZE", text: "Holding position. Spike is temporary, trend structure intact.", level: "info" },
];

export function ThoughtStream({ agentId }: ThoughtStreamProps) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const autoScrollRef = useRef(true);
  const storeThoughts = useFleetStore((s) => s.thoughts[agentId]);
  const thoughts = (storeThoughts && storeThoughts.length > 0) ? storeThoughts : DEMO_THOUGHTS;

  // Auto-scroll
  useEffect(() => {
    if (autoScrollRef.current && scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [thoughts]);

  function handleScroll() {
    if (!scrollRef.current) return;
    const { scrollTop, scrollHeight, clientHeight } = scrollRef.current;
    autoScrollRef.current = scrollHeight - scrollTop - clientHeight < 50;
  }

  return (
    <div className="flex flex-col h-full rounded-lg border border-zinc-800/40 bg-zinc-950/80 overflow-hidden">
      {/* Terminal Header */}
      <div className="flex items-center gap-2 px-4 py-2 bg-zinc-900/80 border-b border-zinc-800/40">
        <div className="flex gap-1.5">
          <span className="h-2.5 w-2.5 rounded-full bg-red-500/80" />
          <span className="h-2.5 w-2.5 rounded-full bg-amber-500/80" />
          <span className="h-2.5 w-2.5 rounded-full bg-emerald-500/80" />
        </div>
        <span className="text-[10px] text-zinc-500 font-mono ml-2">brain_thought_stream</span>
      </div>

      {/* Log Content */}
      <div
        ref={scrollRef}
        onScroll={handleScroll}
        className="flex-1 overflow-auto p-3 font-mono text-xs leading-relaxed space-y-0.5"
      >
        {thoughts.map((t, i) => {
          const config = levelConfig[t.level];
          const time = new Date(t.timestamp).toLocaleTimeString("pt-BR", { hour12: false });

          return (
            <div key={i} className="flex gap-2 hover:bg-zinc-800/20 px-1 rounded">
              <span className="text-zinc-600 shrink-0">{time}</span>
              <span className={cn("font-bold shrink-0", config.color)}>
                [{config.prefix}]
              </span>
              <span className="text-zinc-500 shrink-0">{t.step}:</span>
              <span className={cn(config.color, "break-all")}>{t.text}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
