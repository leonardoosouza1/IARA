"use client";

import { cn } from "@/lib/utils";
import type { WorkerVotes } from "@/types";

interface BrainMRIProps {
  votes: WorkerVotes;
}

const WORKERS = ["M5", "M15", "M30", "H1", "H4"] as const;

// Demo votes for development
const DEMO_VOTES: WorkerVotes = {
  M5: 0.72,
  M15: 0.55,
  M30: 0.80,
  H1: 0.38,
  H4: 0.65,
  consensus: 0.78,
};

export function BrainMRI({ votes = DEMO_VOTES }: BrainMRIProps) {
  const size = 200;
  const center = size / 2;
  const maxRadius = 80;
  const levels = 4; // concentric circles

  // Calculate points on the pentagon
  function getPoint(index: number, value: number) {
    const angle = (Math.PI * 2 * index) / WORKERS.length - Math.PI / 2;
    const r = maxRadius * Math.abs(value);
    return {
      x: center + r * Math.cos(angle),
      y: center + r * Math.sin(angle),
    };
  }

  // Generate polygon path
  const points = WORKERS.map((w, i) => getPoint(i, votes[w]));
  const polygonPath = points.map((p) => `${p.x},${p.y}`).join(" ");

  return (
    <div className="flex flex-col items-center gap-3">
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} className="drop-shadow-lg">
        {/* Concentric pentagons (grid) */}
        {Array.from({ length: levels }, (_, lvl) => {
          const scale = (lvl + 1) / levels;
          const gridPoints = WORKERS.map((_, i) => {
            const angle = (Math.PI * 2 * i) / WORKERS.length - Math.PI / 2;
            const r = maxRadius * scale;
            return `${center + r * Math.cos(angle)},${center + r * Math.sin(angle)}`;
          }).join(" ");

          return (
            <polygon
              key={lvl}
              points={gridPoints}
              fill="none"
              stroke="rgba(255,255,255,0.06)"
              strokeWidth="1"
            />
          );
        })}

        {/* Axis lines */}
        {WORKERS.map((_, i) => {
          const angle = (Math.PI * 2 * i) / WORKERS.length - Math.PI / 2;
          return (
            <line
              key={i}
              x1={center}
              y1={center}
              x2={center + maxRadius * Math.cos(angle)}
              y2={center + maxRadius * Math.sin(angle)}
              stroke="rgba(255,255,255,0.06)"
              strokeWidth="1"
            />
          );
        })}

        {/* Data polygon */}
        <polygon
          points={polygonPath}
          fill="rgba(99, 102, 241, 0.15)"
          stroke="#6366f1"
          strokeWidth="2"
          className="drop-shadow-[0_0_8px_rgba(99,102,241,0.3)]"
        />

        {/* Data points */}
        {points.map((p, i) => (
          <circle
            key={i}
            cx={p.x}
            cy={p.y}
            r="4"
            fill={votes[WORKERS[i]] > 0.5 ? "#10b981" : "#f43f5e"}
            stroke="rgba(0,0,0,0.5)"
            strokeWidth="1"
            className="drop-shadow-md"
          />
        ))}

        {/* Labels */}
        {WORKERS.map((w, i) => {
          const angle = (Math.PI * 2 * i) / WORKERS.length - Math.PI / 2;
          const labelR = maxRadius + 18;
          const x = center + labelR * Math.cos(angle);
          const y = center + labelR * Math.sin(angle);

          return (
            <text
              key={w}
              x={x}
              y={y}
              textAnchor="middle"
              dominantBaseline="middle"
              className="fill-zinc-400 text-[10px] font-mono font-semibold"
            >
              {w}
            </text>
          );
        })}
      </svg>

      {/* Consensus Bar */}
      <div className="w-full max-w-[200px] space-y-1">
        <div className="flex justify-between text-[10px] font-mono">
          <span className="text-zinc-500">Consensus</span>
          <span className={cn(
            "font-bold",
            votes.consensus > 0.7 ? "text-emerald-400" : votes.consensus > 0.4 ? "text-amber-400" : "text-red-400"
          )}>
            {(votes.consensus * 100).toFixed(0)}%
          </span>
        </div>
        <div className="h-1.5 rounded-full bg-zinc-800 overflow-hidden">
          <div
            className={cn(
              "h-full rounded-full transition-all duration-500",
              votes.consensus > 0.7 ? "bg-emerald-500" : votes.consensus > 0.4 ? "bg-amber-500" : "bg-red-500"
            )}
            style={{ width: `${votes.consensus * 100}%` }}
          />
        </div>
      </div>
    </div>
  );
}
