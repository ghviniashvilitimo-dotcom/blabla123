"""Smoke tests that do not require network (synthetic OHLC)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from gold_backtest.backtest import Friction, buy_and_hold, simulate
from gold_backtest.metrics import summarize
from gold_backtest.strategy import dual_ma_signals
from gold_backtest.walkforward import chronological_split


def _synthetic(n: int = 600) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    idx = pd.bdate_range("2018-01-01", periods=n)
    # Mild upward drift with noise
    rets = rng.normal(0.0003, 0.01, size=n)
    close = 100 * np.cumprod(1 + rets)
    open_ = np.concatenate([[close[0]], close[:-1]])
    return pd.DataFrame(
        {"open": open_, "high": close * 1.01, "low": close * 0.99, "close": close},
        index=idx,
    )


def test_dual_ma_signal_bounds():
    ohlc = _synthetic()
    sig = dual_ma_signals(ohlc["close"], fast=20, slow=50)
    assert set(sig["signal"].unique()).issubset({0, 1})
    assert sig["signal"].iloc[:49].eq(0).all()


def test_simulate_and_bh():
    ohlc = _synthetic()
    fr = Friction(commission_bps=5, slippage_bps=5)
    strat = simulate(ohlc, fast=20, slow=50, friction=fr)
    bh = buy_and_hold(ohlc, friction=fr)
    s, b = summarize(strat), summarize(bh)
    assert s["trade_count"] >= 1
    assert b["trade_count"] == 1
    assert s["final_equity"] > 0
    assert b["final_equity"] > 0


def test_split_sizes():
    ohlc = _synthetic(800)
    train, test = chronological_split(ohlc, 0.7)
    assert len(train) + len(test) == len(ohlc)
    assert len(train) > len(test)
