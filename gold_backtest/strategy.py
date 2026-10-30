"""Dual moving-average crossover trend/momentum signals (long-only)."""

from __future__ import annotations

import pandas as pd

# Default parameters (also used for optional in-sample selection)
DEFAULT_FAST = 50
DEFAULT_SLOW = 200


def dual_ma_signals(
    close: pd.Series,
    fast: int = DEFAULT_FAST,
    slow: int = DEFAULT_SLOW,
) -> pd.DataFrame:
    """
    Compute dual SMA crossover signals on close prices.

    Exact rules
    -----------
    1. Compute simple moving averages of ``close`` over ``fast`` and ``slow``
       bars (``fast`` < ``slow``).
    2. Desired position is **long (1)** when ``SMA_fast > SMA_slow``.
    3. Desired position is **flat (0)** when ``SMA_fast <= SMA_slow``.
    4. Until both SMAs are defined (first ``slow - 1`` bars), position is 0.
    5. Signal is based on the close of day *t*; fills use the next-bar open
       (see ``backtest.simulate``).

    This is a classic trend-following / momentum starter: stay long while the
    shorter average is above the longer average.
    """
    if fast < 1 or slow < 2 or fast >= slow:
        raise ValueError(f"Require 1 <= fast < slow; got fast={fast}, slow={slow}")

    out = pd.DataFrame(index=close.index)
    out["close"] = close.astype(float)
    out["sma_fast"] = out["close"].rolling(fast, min_periods=fast).mean()
    out["sma_slow"] = out["close"].rolling(slow, min_periods=slow).mean()
    out["signal"] = 0
    valid = out["sma_fast"].notna() & out["sma_slow"].notna()
    out.loc[valid & (out["sma_fast"] > out["sma_slow"]), "signal"] = 1
    out.loc[valid & (out["sma_fast"] <= out["sma_slow"]), "signal"] = 0
    return out


def candidate_params(
    fast_grid: tuple[int, ...] = (20, 50, 100),
    slow_grid: tuple[int, ...] = (100, 150, 200),
) -> list[tuple[int, int]]:
    """Valid (fast, slow) pairs with fast < slow."""
    return [(f, s) for f in fast_grid for s in slow_grid if f < s]
