"""
I.A.R.A. v11 — Hurst Exponent & Fractal Dimension
===================================================
Measures the "memory" of a time series using Rescaled Range (R/S) analysis.

Interpretation:
  H ≈ 0.5  → Random Walk (noise) → Skip trading
  H > 0.5  → Long Memory (trending) → Hold positions longer
  H < 0.5  → Anti-persistent (mean-reverting) → Scalp mode

Reference: Hurst, H.E. (1951). "Long-term storage capacity of reservoirs"
"""

import numpy as np
from typing import Tuple


def _rescaled_range(series: np.ndarray) -> float:
    """Compute the Rescaled Range (R/S) statistic for a time series."""
    n = len(series)
    if n < 2:
        return 0.0

    mean = np.mean(series)
    deviations = series - mean
    cumulative = np.cumsum(deviations)

    R = np.max(cumulative) - np.min(cumulative)
    S = np.std(series, ddof=1)

    if S < 1e-10:
        return 0.0

    return R / S


def compute_hurst(series: np.ndarray, min_window: int = 20, max_window: int = 200) -> float:
    """
    Compute the Hurst Exponent using Rescaled Range (R/S) analysis.

    Uses multiple window sizes and fits log(R/S) vs log(n) to extract H.

    Args:
        series: 1D array of prices or returns
        min_window: Minimum sub-series window size
        max_window: Maximum sub-series window size (clamped to len(series)//2)

    Returns:
        Hurst exponent H ∈ [0, 1]. Returns 0.5 if computation fails.
    """
    n = len(series)
    if n < min_window * 2:
        return 0.5  # Not enough data

    max_window = min(max_window, n // 2)
    if max_window <= min_window:
        max_window = min_window + 1

    # Generate logarithmically spaced window sizes
    window_sizes = np.unique(
        np.logspace(
            np.log10(min_window),
            np.log10(max_window),
            num=15,
            dtype=int,
        )
    )
    window_sizes = window_sizes[window_sizes >= min_window]

    if len(window_sizes) < 3:
        return 0.5

    log_n = []
    log_rs = []

    for w in window_sizes:
        # Split series into non-overlapping windows of size w
        num_windows = n // w
        if num_windows < 1:
            continue

        rs_values = []
        for i in range(num_windows):
            segment = series[i * w: (i + 1) * w]
            rs = _rescaled_range(segment)
            if rs > 0:
                rs_values.append(rs)

        if rs_values:
            avg_rs = np.mean(rs_values)
            log_n.append(np.log(w))
            log_rs.append(np.log(avg_rs))

    if len(log_n) < 3:
        return 0.5

    # Linear regression: log(R/S) = H * log(n) + c
    log_n = np.array(log_n)
    log_rs = np.array(log_rs)

    try:
        coeffs = np.polyfit(log_n, log_rs, 1)
        H = float(coeffs[0])
        # Clamp to valid range
        H = max(0.0, min(1.0, H))
        return H
    except (np.linalg.LinAlgError, ValueError):
        return 0.5


def compute_fractal_dimension(series: np.ndarray, **kwargs) -> float:
    """
    Compute the Fractal Dimension from Hurst Exponent.
    D = 2 - H

    A higher D means more "rough"/complex behavior.

    Returns:
        Fractal dimension D ∈ [1, 2]
    """
    H = compute_hurst(series, **kwargs)
    return 2.0 - H


def compute_hurst_rolling(
    series: np.ndarray,
    window: int = 200,
    step: int = 1,
) -> np.ndarray:
    """
    Compute rolling Hurst Exponent over a time series.

    Args:
        series: 1D array of close prices
        window: Rolling window size
        step: Step size for rolling computation

    Returns:
        np.ndarray of same length as series with Hurst values.
        Positions with insufficient data are filled with 0.5.
    """
    n = len(series)
    result = np.full(n, 0.5, dtype=np.float32)

    for i in range(window, n, step):
        segment = series[i - window: i]
        h = compute_hurst(segment, min_window=20, max_window=window // 2)
        result[i] = h

    # Forward-fill steps > 1
    if step > 1:
        for i in range(window, n):
            if result[i] == 0.5 and i > window:
                result[i] = result[i - 1]

    return result
