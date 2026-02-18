import { NextResponse } from "next/server";

// DEV MOCK: Return demo fleet data
export async function GET() {
  const agents = [
    {
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
    },
    {
      id: "agent-eth-1",
      symbol: "ETH/USD",
      marketType: "crypto",
      status: "thinking",
      modelName: "SwarmBrain v9",
      pnl: -42.30,
      winRate: 0.45,
      totalTrades: 8,
      currentPosition: null,
      lastSignal: null,
      uptime: 5400,
    },
    {
      id: "agent-eur-1",
      symbol: "EUR/USD",
      marketType: "forex",
      status: "stalking",
      modelName: "SwarmBrain v9",
      pnl: 89.20,
      winRate: 0.58,
      totalTrades: 19,
      currentPosition: null,
      lastSignal: null,
      uptime: 14400,
    },
  ];

  return NextResponse.json({ agents, totalPnl: 371.40, activeAgents: 3 });
}
