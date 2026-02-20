"use client";

import { useRef, useEffect, useCallback, useState } from "react";
import {
  createChart,
  IChartApi,
  ISeriesApi,
  CandlestickData,
  Time,
  ColorType,
  CandlestickSeries,
  LineSeries,
  SeriesMarker,
} from "lightweight-charts";
import { useFleetStore } from "@/hooks/use-fleet-store";
import { SMA, RSI } from "technicalindicators";
import type { Agent } from "@/types";

interface RealTimeChartProps {
  symbol: string;
  agent?: Agent; // Pass agent to trace its trades
  height?: number;
}

export function RealTimeChart({
  symbol,
  agent,
  height = 400,
}: RealTimeChartProps) {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const rsiContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const rsiChartRef = useRef<IChartApi | null>(null);
  const seriesRef = useRef<ISeriesApi<"Candlestick"> | null>(null);
  const sma20Ref = useRef<ISeriesApi<"Line"> | null>(null);
  const sma200Ref = useRef<ISeriesApi<"Line"> | null>(null);
  const rsiSeriesRef = useRef<ISeriesApi<"Line"> | null>(null);
  const dataLoadedRef = useRef<boolean>(false);

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

    const rsiChart = createChart(rsiContainerRef.current!, {
      width: rsiContainerRef.current!.clientWidth,
      height: 100,
      layout: {
        background: { type: ColorType.Solid, color: "transparent" },
        textColor: "#71717a",
        fontFamily: "'JetBrains Mono', monospace",
        fontSize: 10,
      },
      grid: {
        vertLines: { color: "rgba(255, 255, 255, 0.03)" },
        horzLines: { color: "rgba(255, 255, 255, 0.03)" },
      },
      rightPriceScale: {
        borderColor: "rgba(255, 255, 255, 0.05)",
      },
      timeScale: {
        visible: false, // Hide time scale on RSI pane
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

    const sma20 = chart.addSeries(LineSeries, {
      color: "rgba(234, 179, 8, 0.7)", // Yellow
      lineWidth: 1 as any,
      crosshairMarkerVisible: false,
    });

    const sma200 = chart.addSeries(LineSeries, {
      color: "rgba(56, 189, 248, 0.7)", // Light Blue
      lineWidth: 2 as any,
      crosshairMarkerVisible: false,
    });

    const rsiSeries = rsiChart.addSeries(LineSeries, {
      color: "#a855f7",
      lineWidth: 1 as any,
      priceFormat: {
        type: "price",
        precision: 2,
        minMove: 0.01,
      },
    });

    // Reference lines for RSI
    rsiSeries.createPriceLine({
      price: 70,
      color: "rgba(239, 68, 68, 0.5)",
      lineWidth: 1,
      lineStyle: 2,
      axisLabelVisible: true,
      title: "OB",
    });
    rsiSeries.createPriceLine({
      price: 30,
      color: "rgba(34, 197, 94, 0.5)",
      lineWidth: 1,
      lineStyle: 2,
      axisLabelVisible: true,
      title: "OS",
    });

    chart.timeScale().fitContent();

    chartRef.current = chart;
    rsiChartRef.current = rsiChart;
    seriesRef.current = series;
    sma20Ref.current = sma20;
    sma200Ref.current = sma200;
    rsiSeriesRef.current = rsiSeries;

    // Sync subchart X-axis
    chart.timeScale().subscribeVisibleTimeRangeChange((range) => {
      if (range) {
        rsiChart
          .timeScale()
          .setVisibleLogicalRange(chart.timeScale().getVisibleLogicalRange()!);
      }
    });

    const resizeObserver = new ResizeObserver((entries) => {
      for (const entry of entries) {
        if (entry.target === chartContainerRef.current) {
          chart.applyOptions({ width: entry.contentRect.width });
        }
        if (entry.target === rsiContainerRef.current) {
          rsiChart.applyOptions({ width: entry.contentRect.width });
        }
      }
    });
    resizeObserver.observe(chartContainerRef.current);
    if (rsiContainerRef.current)
      resizeObserver.observe(rsiContainerRef.current);

    return () => {
      resizeObserver.disconnect();
      chart.remove();
      rsiChart.remove();
      chartRef.current = null;
      rsiChartRef.current = null;
      seriesRef.current = null;
      sma20Ref.current = null;
      sma200Ref.current = null;
      rsiSeriesRef.current = null;
      dataLoadedRef.current = false;
    };
  }, [height]);

  useEffect(() => {
    const cleanup = initChart();
    return cleanup;
  }, [initChart]);

  // Handle Historical & Closed Candles
  useEffect(() => {
    if (!seriesRef.current || !candles?.length) return;

    if (!dataLoadedRef.current && candles.length > 1) {
      // First time loading history
      const formattedData = candles.map((c) => ({
        time: c.time as Time,
        open: c.open,
        high: c.high,
        low: c.low,
        close: c.close,
      }));
      seriesRef.current.setData(formattedData);

      // --- Map Markers from Agent Signals ---
      if (agent?.lastSignal) {
        // Here we just map the latest signal as a demo marker.
        // In a real-world scenario with a full ledger, we would map all history trades.
        try {
          // find closest candle time to signal
          const signalTime = parseInt(agent.lastSignal.timestamp);
          if (!isNaN(signalTime)) {
            const closestCandle = formattedData.reduce((prev, curr) =>
              Math.abs((curr.time as number) - signalTime) <
              Math.abs((prev.time as number) - signalTime)
                ? curr
                : prev,
            );

            if (
              agent.lastSignal.action === "buy" ||
              agent.lastSignal.action === "sell"
            ) {
              const markers: any[] = [
                {
                  time: closestCandle.time,
                  position:
                    agent.lastSignal.action === "buy" ? "belowBar" : "aboveBar",
                  color:
                    agent.lastSignal.action === "buy" ? "#10b981" : "#f43f5e",
                  shape:
                    agent.lastSignal.action === "buy" ? "arrowUp" : "arrowDown",
                  text: agent.lastSignal.action.toUpperCase(),
                },
              ];
              // Type cast seriesRef as Candlestick has setMarkers
              (seriesRef.current as any).setMarkers(markers);
            }
          }
        } catch (e) {
          console.error("Marker mapping error", e);
        }
      }

      // --- Calculate Indicators ---
      const closePrices = formattedData.map((c) => c.close);

      const sma20Vals = SMA.calculate({ period: 20, values: closePrices });
      const sma20Data = sma20Vals.map((val, i) => ({
        time: formattedData[i + 19].time,
        value: val,
      }));
      sma20Ref.current?.setData(sma20Data);

      const sma200Vals = SMA.calculate({ period: 200, values: closePrices });
      const sma200Data = sma200Vals.map((val, i) => ({
        time: formattedData[i + 199].time,
        value: val,
      }));
      sma200Ref.current?.setData(sma200Data);

      const rsiVals = RSI.calculate({ period: 14, values: closePrices });
      const rsiData = rsiVals.map((val, i) => ({
        time: formattedData[i + 14].time, // Tech Indicators shifted by period
        value: val,
      }));
      rsiSeriesRef.current?.setData(rsiData);

      chartRef.current?.timeScale().fitContent();
      dataLoadedRef.current = true;
    } else if (dataLoadedRef.current) {
      // Just update the latest closed candle
      const lastCandle = candles[candles.length - 1];
      seriesRef.current.update({
        time: lastCandle.time as Time,
        open: lastCandle.open,
        high: lastCandle.high,
        low: lastCandle.low,
        close: lastCandle.close,
      });

      // We don't recalculate FULL array of SMA/RSI on every closed candle right now
      // for performance, but we could by pushing to an accumulator class.
      // Usually updating the last item is sufficient unless requested.
      // E.g., const lastPrice = candles.map(c => c.close) -> RSI.calculate...

      currentBar.current = null;
    }
  }, [candles, agent?.lastSignal]);

  // Handle Real-Time Ticks
  useEffect(() => {
    if (
      !seriesRef.current ||
      !ticks?.length ||
      !candles?.length ||
      !dataLoadedRef.current
    )
      return;

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
    <div className="w-full flex flex-col gap-[2px] rounded-lg overflow-hidden bg-zinc-900/30 border border-zinc-800/40">
      <div ref={chartContainerRef} className="w-full" />
      <div ref={rsiContainerRef} className="w-full" />
    </div>
  );
}
