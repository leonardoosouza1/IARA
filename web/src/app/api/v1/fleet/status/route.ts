import { NextResponse } from "next/server";
import { cookies } from "next/headers";

const BACKEND_URL = process.env.BACKEND_URL || "http://localhost:8000";

/**
 * Proxy to the Python backend fleet status endpoint.
 * Forwards the auth cookie for authentication.
 */
export async function GET() {
  try {
    const cookieStore = await cookies();
    const token = cookieStore.get("access_token")?.value;

    const res = await fetch(`${BACKEND_URL}/api/v1/fleet/status`, {
      headers: {
        "Content-Type": "application/json",
        ...(token ? { Cookie: `access_token=${token}` } : {}),
      },
      cache: "no-store",
    });

    if (!res.ok) {
      const errorText = await res.text();
      console.error(`Backend fleet/status error: ${res.status} ${errorText}`);
      return NextResponse.json(
        { agents: [], totalPnl: 0, activeAgents: 0 },
        { status: res.status }
      );
    }

    const data = await res.json();

    // Transform snake_case backend response to camelCase frontend format
    const agents = (data.agents || []).map((a: Record<string, unknown>) => ({
      id: a.id,
      symbol: a.symbol,
      marketType: a.market_type,
      status: a.status,
      modelName: a.model_name,
      pnl: a.pnl ?? 0,
      winRate: a.win_rate ?? 0,
      totalTrades: a.total_trades ?? 0,
      currentPosition: a.current_position ?? null,
      lastSignal: a.last_signal
        ? {
          timestamp: (a.last_signal as Record<string, unknown>).timestamp,
          action: (a.last_signal as Record<string, unknown>).action,
          confidence: (a.last_signal as Record<string, unknown>).confidence,
          longScore: (a.last_signal as Record<string, unknown>).longScore,
          shortScore: (a.last_signal as Record<string, unknown>).shortScore,
          workerVotes: (a.last_signal as Record<string, unknown>).workerVotes,
          lfmThought: (a.last_signal as Record<string, unknown>).lfmThought,
        }
        : null,
      uptime: a.uptime ?? 0,
      config: a.config ?? undefined,
    }));

    return NextResponse.json({
      agents,
      totalPnl: data.total_pnl ?? 0,
      activeAgents: data.active_agents ?? 0,
    });
  } catch (error) {
    console.error("Failed to proxy fleet status:", error);
    return NextResponse.json(
      { agents: [], totalPnl: 0, activeAgents: 0 },
      { status: 502 }
    );
  }
}
