"use client";

import { useRef, useEffect, useCallback } from "react";
import { createChart, IChartApi, ISeriesApi, CandlestickData, Time, ColorType, CandlestickSeries } from "lightweight-charts";
import { useFleetStore } from "@/hooks/use-fleet-store";

interface RealTimeChartProps {
  symbol: string;
  height?: number;
}

// Demo candle data for development
function generateDemoCandles(count: number): CandlestickData<Time>[] {
  const candles: CandlestickData<Time>[] = [];
  let price = 97000;
  const now = Math.floor(Date.now() / 1000);

  for (let i = count; i > 0; i--) {
    const open = price + (Math.random() - 0.5) * 200;
    const close = open + (Math.random() - 0.5) * 300;
    const high = Math.max(open, close) + Math.random() * 100;
    const low = Math.min(open, close) - Math.random() * 100;
    price = close;

    candles.push({
      time: (now - i * 300) as Time,
      open: parseFloat(open.toFixed(2)),
      high: parseFloat(high.toFixed(2)),
      low: parseFloat(low.toFixed(2)),
      close: parseFloat(close.toFixed(2)),
    });
  }
  return candles;
}

export function RealTimeChart({ symbol, height = 400 }: RealTimeChartProps) {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const seriesRef = useRef<ISeriesApi<"Candlestick"> | null>(null);
  const candles = useFleetStore((s) => s.candles[symbol]);

  const initChart = useCallback(() => {
    if (!chartContainerRef.current) return;



    const chart = createChart(chartContainerRef.current, {
      width: chartContainerRef.current.clientWidth,
      height,
      layout: {
        background: { type: ColorType.Solid, color: "transparent" },
        textColor: "#71717a",
        fontFamily: "'JetBrains Mono', monospace",
        fontSize: 11,
      },
      grid: {
        vertLines: { color: "rgba(255, 255, 255, 0.03)" },
        horzLines: { color: "rgba(255, 255, 255, 0.03)" },
      },
      crosshair: {
        vertLine: { color: "rgba(99, 102, 241, 0.3)", labelBackgroundColor: "#6366f1" },
        horzLine: { color: "rgba(99, 102, 241, 0.3)", labelBackgroundColor: "#6366f1" },
      },
      rightPriceScale: {
        borderColor: "rgba(255, 255, 255, 0.05)",
      },
      timeScale: {
        borderColor: "rgba(255, 255, 255, 0.05)",
        timeVisible: true,
        secondsVisible: false,
      },
    });

    const series = chart.addSeries(CandlestickSeries, {
      upColor: "#10b981",
      downColor: "#f43f5e",
      borderUpColor: "#10b981",
      borderDownColor: "#f43f5e",
      wickUpColor: "#10b981",
      wickDownColor: "#f43f5e",
    });

    // Load initial data (demo or from store)
    const initialData = candles?.map((c) => ({
      time: c.time as Time,
      open: c.open,
      high: c.high,
      low: c.low,
      close: c.close,
    })) || generateDemoCandles(100);

    series.setData(initialData);
    chart.timeScale().fitContent();

    chartRef.current = chart;
    seriesRef.current = series;

    // Resize observer
    const resizeObserver = new ResizeObserver((entries) => {
      for (const entry of entries) {
        chart.applyOptions({ width: entry.contentRect.width });
      }
    });
    resizeObserver.observe(chartContainerRef.current);

    return () => {
      resizeObserver.disconnect();
      chart.remove();
      chartRef.current = null;
      seriesRef.current = null;
    };
  }, [height, candles, symbol]);

  useEffect(() => {
    const cleanup = initChart();
    return cleanup;
    // Only re-initialize on mount or height change, not candle updates
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [height]);

  // Update chart with new candles (without re-initializing)
  useEffect(() => {
    if (!seriesRef.current || !candles?.length) return;

    const lastCandle = candles[candles.length - 1];
    seriesRef.current.update({
      time: lastCandle.time as Time,
      open: lastCandle.open,
      high: lastCandle.high,
      low: lastCandle.low,
      close: lastCandle.close,
    });
  }, [candles]);

  return (
    <div className="w-full rounded-lg overflow-hidden bg-zinc-900/30 border border-zinc-800/40">
      <div ref={chartContainerRef} className="w-full" />
    </div>
  );
}
