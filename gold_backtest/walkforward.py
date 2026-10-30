"""Train/test split and simple walk-forward evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from gold_backtest.backtest import BacktestResult, Friction, buy_and_hold, simulate
from gold_backtest.metrics import summarize
from gold_backtest.strategy import candidate_params


@dataclass
class SplitResult:
    train_summary: dict[str, Any]
    test_summary: dict[str, Any]
    train_bh: dict[str, Any]
    test_bh: dict[str, Any]
    best_params: dict[str, int]
    train_result: BacktestResult
    test_result: BacktestResult
    test_bh_result: BacktestResult


def chronological_split(
    ohlc: pd.DataFrame,
    train_ratio: float = 0.7,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by calendar order (no shuffle)."""
    if not 0.1 <= train_ratio <= 0.9:
        raise ValueError("train_ratio should be between 0.1 and 0.9")
    n = len(ohlc)
    cut = int(n * train_ratio)
    if cut < 250 or n - cut < 100:
        raise ValueError(
            f"Split too small for MA warmup: train={cut}, test={n - cut}, n={n}"
        )
    return ohlc.iloc[:cut].copy(), ohlc.iloc[cut:].copy()


def select_params_on_train(
    train: pd.DataFrame,
    friction: Friction,
    initial_capital: float = 100_000.0,
) -> tuple[int, int, BacktestResult, dict[str, Any]]:
    """
    Choose (fast, slow) that maximizes Sharpe on the training window.

    This is intentional in-sample selection; report test metrics separately.
    """
    best: BacktestResult | None = None
    best_sharpe = float("-inf")
    best_fs = (50, 200)
    for fast, slow in candidate_params():
        # Need enough bars for slow MA
        if len(train) < slow + 5:
            continue
        res = simulate(
            train,
            fast=fast,
            slow=slow,
            friction=friction,
            initial_capital=initial_capital,
            label=f"train SMA({fast}/{slow})",
        )
        s = summarize(res)
        if s["sharpe"] > best_sharpe:
            best_sharpe = s["sharpe"]
            best = res
            best_fs = (fast, slow)
    if best is None:
        raise RuntimeError("No valid parameter set for training window")
    return best_fs[0], best_fs[1], best, summarize(best)


def run_train_test(
    ohlc: pd.DataFrame,
    friction: Friction | None = None,
    train_ratio: float = 0.7,
    initial_capital: float = 100_000.0,
    fixed_fast: int | None = None,
    fixed_slow: int | None = None,
) -> SplitResult:
    """
    Chronological train/test evaluation.

    - If ``fixed_fast`` / ``fixed_slow`` are set, use those on both windows
      (no in-sample search).
    - Otherwise select params by train Sharpe, then evaluate on test.
    """
    fr = friction or Friction()
    train, test = chronological_split(ohlc, train_ratio=train_ratio)

    if fixed_fast is not None and fixed_slow is not None:
        fast, slow = fixed_fast, fixed_slow
        train_res = simulate(
            train, fast, slow, friction=fr, initial_capital=initial_capital, label=f"train SMA({fast}/{slow})"
        )
        train_sum = summarize(train_res)
    else:
        fast, slow, train_res, train_sum = select_params_on_train(
            train, fr, initial_capital=initial_capital
        )

    test_res = simulate(
        test,
        fast=fast,
        slow=slow,
        friction=fr,
        initial_capital=initial_capital,
        label=f"test SMA({fast}/{slow})",
    )
    train_bh_res = buy_and_hold(train, friction=fr, initial_capital=initial_capital, label="train buy&hold")
    test_bh_res = buy_and_hold(test, friction=fr, initial_capital=initial_capital, label="test buy&hold")

    return SplitResult(
        train_summary=train_sum,
        test_summary=summarize(test_res),
        train_bh=summarize(train_bh_res),
        test_bh=summarize(test_bh_res),
        best_params={"fast": fast, "slow": slow},
        train_result=train_res,
        test_result=test_res,
        test_bh_result=test_bh_res,
    )


def run_walk_forward(
    ohlc: pd.DataFrame,
    friction: Friction | None = None,
    train_bars: int = 756,  # ~3 years
    test_bars: int = 252,  # ~1 year
    step_bars: int = 252,
    initial_capital: float = 100_000.0,
) -> list[dict[str, Any]]:
    """
    Rolling walk-forward: optimize on ``train_bars``, trade next ``test_bars``,
    then roll forward by ``step_bars``.
    """
    fr = friction or Friction()
    rows: list[dict[str, Any]] = []
    start = 0
    n = len(ohlc)
    fold = 0
    while start + train_bars + test_bars <= n:
        train = ohlc.iloc[start : start + train_bars]
        test = ohlc.iloc[start + train_bars : start + train_bars + test_bars]
        fast, slow, _, _ = select_params_on_train(train, fr, initial_capital=initial_capital)
        test_res = simulate(
            test,
            fast,
            slow,
            friction=fr,
            initial_capital=initial_capital,
            label=f"wf{fold} SMA({fast}/{slow})",
        )
        bh = buy_and_hold(test, friction=fr, initial_capital=initial_capital, label=f"wf{fold} bh")
        s = summarize(test_res)
        b = summarize(bh)
        rows.append(
            {
                "fold": fold,
                "fast": fast,
                "slow": slow,
                "test_start": s["start"],
                "test_end": s["end"],
                "strategy_cagr": s["cagr"],
                "strategy_sharpe": s["sharpe"],
                "strategy_total_return": s["total_return"],
                "strategy_max_dd": s["max_drawdown"],
                "bh_cagr": b["cagr"],
                "bh_total_return": b["total_return"],
                "beat_bh": s["cagr"] > b["cagr"],
            }
        )
        fold += 1
        start += step_bars
    return rows
