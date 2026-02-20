"use client";

import { useRef, useEffect, useCallback } from "react";
import { createChart, IChartApi, ISeriesApi, CandlestickData, Time, ColorType, CandlestickSeries } from "lightweight-charts";
import { useFleetStore } from "@/hooks/use-fleet-store";

interface RealTimeChartProps {
  symbol: string;
  height?: number;
}

export function RealTimeChart({ symbol, height = 400 }: RealTimeChartProps) {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const seriesRef = useRef<ISeriesApi<"Candlestick"> | null>(null);
  const candles = useFleetStore((s) => s.candles[symbol]);
  const ticks = useFleetStore((s) => s.ticks[symbol]);

  // Keep track of the current forming candle
  const currentBar = useRef<CandlestickData<Time> | null>(null);

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
        vertLine: {
          color: "rgba(99, 102, 241, 0.3)",
          labelBackgroundColor: "#6366f1",
        },
        horzLine: {
          color: "rgba(99, 102, 241, 0.3)",
          labelBackgroundColor: "#6366f1",
        },
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

    // Load initial data (from store only)
    const initialData =
      candles?.map((c) => ({
        time: c.time as Time,
        open: c.open,
        high: c.high,
        low: c.low,
        close: c.close,
      })) || [];

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
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [height, symbol]); 

  useEffect(() => {
    const cleanup = initChart();
    return cleanup;
  }, [initChart]);

  // Handle closed candles updates (Add new bar)
  useEffect(() => {
    if (!seriesRef.current || !candles?.length) return;

    const lastCandle = candles[candles.length - 1];
    // Update the series with the closed candle (this confirms it)
    seriesRef.current.update({
      time: lastCandle.time as Time,
      open: lastCandle.open,
      high: lastCandle.high,
      low: lastCandle.low,
      close: lastCandle.close,
    });

    // Reset current bar to allow next tick to start a new one
    currentBar.current = null;
  }, [candles]);

  // Handle real-time ticks (Update forming bar)
  useEffect(() => {
    if (!seriesRef.current || !ticks?.length || !candles?.length) return;

    const lastTick = ticks[ticks.length - 1];
    const lastClosedCandle = candles[candles.length - 1];

    // Determine time for the forming candle (e.g., last closed + 5s)
    const nextTime = ((lastClosedCandle.time as number) + 5) as Time;

    let bar = currentBar.current;

    if (!bar) {
      // Start new bar
      bar = {
        time: nextTime,
        open: lastClosedCandle.close,
        high: Math.max(lastClosedCandle.close, lastTick.price),
        low: Math.min(lastClosedCandle.close, lastTick.price),
        close: lastTick.price,
      };
    } else {
      // Update existing bar
      bar.high = Math.max(bar.high, lastTick.price);
      bar.low = Math.min(bar.low, lastTick.price);
      bar.close = lastTick.price;
    }

    currentBar.current = bar;
    seriesRef.current.update(bar);
  }, [ticks, candles]);

  return (
    <div className="w-full rounded-lg overflow-hidden bg-zinc-900/30 border border-zinc-800/40">
      <div ref={chartContainerRef} className="w-full" />
    </div>
  );
}
