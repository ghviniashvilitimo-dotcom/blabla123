"""Performance metrics for equity curves and trade lists."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from gold_backtest.backtest import BacktestResult

TRADING_DAYS = 252


def _cagr(equity: pd.Series) -> float:
    if len(equity) < 2:
        return 0.0
    start, end = float(equity.iloc[0]), float(equity.iloc[-1])
    if start <= 0 or end <= 0:
        return 0.0
    days = (equity.index[-1] - equity.index[0]).days
    if days <= 0:
        return 0.0
    years = days / 365.25
    return (end / start) ** (1.0 / years) - 1.0


def _max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min()) if len(dd) else 0.0


def _sharpe(returns: pd.Series, rf: float = 0.0) -> float:
    if returns.std(ddof=0) == 0 or len(returns) < 2:
        return 0.0
    excess = returns - rf / TRADING_DAYS
    return float(np.sqrt(TRADING_DAYS) * excess.mean() / excess.std(ddof=0))


def _trade_stats(trades: pd.DataFrame) -> dict[str, Any]:
    if trades.empty:
        return {"trade_count": 0, "win_rate": float("nan"), "round_trips": 0}

    buys = trades[trades["side"] == "buy"].reset_index(drop=True)
    sells = trades[trades["side"] == "sell"].reset_index(drop=True)
    n = min(len(buys), len(sells))
    if n == 0:
        # Open position only (e.g. buy-and-hold or still in trade)
        return {
            "trade_count": int(len(trades)),
            "win_rate": float("nan"),
            "round_trips": 0,
        }

    pnls = []
    for i in range(n):
        # Approx P&L per share after fees already baked into fill prices/equity;
        # use price difference as trade outcome for win rate.
        entry = float(buys.loc[i, "price"])
        exit_ = float(sells.loc[i, "price"])
        pnls.append(exit_ - entry)

    wins = sum(1 for p in pnls if p > 0)
    return {
        "trade_count": int(len(trades)),
        "win_rate": wins / n if n else float("nan"),
        "round_trips": n,
    }


def summarize(result: BacktestResult) -> dict[str, Any]:
    eq = result.equity
    total_return = float(eq.iloc[-1] / eq.iloc[0] - 1.0) if len(eq) else 0.0
    trade_stats = _trade_stats(result.trades)
    start = eq.index[0].date().isoformat() if len(eq) else None
    end = eq.index[-1].date().isoformat() if len(eq) else None
    return {
        "label": result.label or f"SMA({result.params.get('fast')}/{result.params.get('slow')})",
        "start": start,
        "end": end,
        "total_return": total_return,
        "cagr": _cagr(eq),
        "max_drawdown": _max_drawdown(eq),
        "sharpe": _sharpe(result.returns),
        "win_rate": trade_stats["win_rate"],
        "trade_count": trade_stats["trade_count"],
        "round_trips": trade_stats["round_trips"],
        "final_equity": float(eq.iloc[-1]) if len(eq) else 0.0,
        "params": result.params,
    }


def format_summary(rows: list[dict[str, Any]]) -> str:
    """Pretty table for CLI output."""
    try:
        from tabulate import tabulate
    except ImportError:
        tabulate = None

    display = []
    for r in rows:
        wr = r["win_rate"]
        display.append(
            {
                "Label": r["label"],
                "Start": r["start"],
                "End": r["end"],
                "Total Ret": f"{r['total_return']:.2%}",
                "CAGR": f"{r['cagr']:.2%}",
                "Max DD": f"{r['max_drawdown']:.2%}",
                "Sharpe": f"{r['sharpe']:.2f}",
                "Win Rate": "n/a" if wr is None or (isinstance(wr, float) and math.isnan(wr)) else f"{wr:.1%}",
                "Trades": r["trade_count"],
            }
        )
    if tabulate:
        return tabulate(display, headers="keys", tablefmt="github")
    # Fallback
    lines = [" | ".join(display[0].keys())] if display else []
    for d in display:
        lines.append(" | ".join(str(v) for v in d.values()))
    return "\n".join(lines)


def beat_buy_and_hold(strategy: dict[str, Any], benchmark: dict[str, Any]) -> bool:
    """True if strategy CAGR exceeds buy-and-hold CAGR on the same window."""
    return float(strategy["cagr"]) > float(benchmark["cagr"])
