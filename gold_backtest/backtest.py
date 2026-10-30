"""Long-only backtest engine with commission and slippage."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from gold_backtest.strategy import dual_ma_signals

# Friction defaults (documented in README)
# Per-side proportional costs applied on each fill (entry or exit).
DEFAULT_COMMISSION_BPS = 5.0  # 5 bps = 0.05% of notional per side
DEFAULT_SLIPPAGE_BPS = 5.0  # 5 bps adverse price move per side
# Round-trip friction ≈ 20 bps (0.20%) of notional when both sides apply.


@dataclass(frozen=True)
class Friction:
    commission_bps: float = DEFAULT_COMMISSION_BPS
    slippage_bps: float = DEFAULT_SLIPPAGE_BPS

    @property
    def cost_rate(self) -> float:
        """Combined proportional cost per side (fraction of notional)."""
        return (self.commission_bps + self.slippage_bps) / 10_000.0


@dataclass
class BacktestResult:
    equity: pd.Series
    positions: pd.Series
    trades: pd.DataFrame
    returns: pd.Series
    params: dict
    friction: Friction
    label: str = ""


def simulate(
    ohlc: pd.DataFrame,
    fast: int,
    slow: int,
    friction: Friction | None = None,
    initial_capital: float = 100_000.0,
    label: str = "",
) -> BacktestResult:
    """
    Simulate long-only dual-MA strategy.

    Execution model
    ---------------
    - Signal at day *t* close → target position for the next session.
    - Fills at day *t+1* **open**, adjusted for slippage:
      - Buy: open * (1 + slippage_bps/1e4)
      - Sell: open * (1 - slippage_bps/1e4)
    - Commission: ``commission_bps`` of fill notional on every trade.
    - Fully invested when long (all equity in the asset); cash when flat.
    - No leverage, no shorting, no overnight borrow costs.
    """
    fr = friction or Friction()
    if "open" not in ohlc.columns or "close" not in ohlc.columns:
        raise ValueError("ohlc must include 'open' and 'close'")

    sig = dual_ma_signals(ohlc["close"], fast=fast, slow=slow)
    # Target position for next bar
    target = sig["signal"].shift(1).fillna(0).astype(int)

    opens = ohlc["open"].astype(float).values
    closes = ohlc["close"].astype(float).values
    dates = ohlc.index
    n = len(ohlc)

    cash = float(initial_capital)
    shares = 0.0
    equity = np.empty(n, dtype=float)
    position = np.zeros(n, dtype=int)
    trade_rows: list[dict] = []

    slip = fr.slippage_bps / 10_000.0
    comm = fr.commission_bps / 10_000.0

    prev_target = 0
    for i in range(n):
        tgt = int(target.iloc[i])
        if i > 0 and tgt != prev_target:
            if tgt == 1 and shares == 0.0:
                # Enter long at open + slippage
                fill = opens[i] * (1.0 + slip)
                if fill > 0 and cash > 0:
                    # Spend all cash; commission reduces shares acquired
                    notional = cash
                    fee = notional * comm
                    spend = notional - fee
                    shares = spend / fill
                    cash = 0.0
                    trade_rows.append(
                        {
                            "date": dates[i],
                            "side": "buy",
                            "price": fill,
                            "shares": shares,
                            "fee": fee,
                        }
                    )
            elif tgt == 0 and shares > 0.0:
                # Exit at open - slippage
                fill = opens[i] * (1.0 - slip)
                notional = shares * fill
                fee = notional * comm
                cash = notional - fee
                trade_rows.append(
                    {
                        "date": dates[i],
                        "side": "sell",
                        "price": fill,
                        "shares": shares,
                        "fee": fee,
                    }
                )
                shares = 0.0
        prev_target = tgt

        # Mark-to-market at close
        equity[i] = cash + shares * closes[i]
        position[i] = 1 if shares > 0 else 0

    eq = pd.Series(equity, index=dates, name="equity")
    pos = pd.Series(position, index=dates, name="position")
    rets = eq.pct_change().fillna(0.0)
    trades = pd.DataFrame(trade_rows)
    return BacktestResult(
        equity=eq,
        positions=pos,
        trades=trades,
        returns=rets,
        params={"fast": fast, "slow": slow, "initial_capital": initial_capital},
        friction=fr,
        label=label,
    )


def buy_and_hold(
    ohlc: pd.DataFrame,
    friction: Friction | None = None,
    initial_capital: float = 100_000.0,
    label: str = "buy_and_hold",
) -> BacktestResult:
    """Buy on first open, hold to end; same friction model on entry only."""
    fr = friction or Friction()
    slip = fr.slippage_bps / 10_000.0
    comm = fr.commission_bps / 10_000.0

    dates = ohlc.index
    opens = ohlc["open"].astype(float).values
    closes = ohlc["close"].astype(float).values
    n = len(ohlc)

    fill = opens[0] * (1.0 + slip)
    fee = initial_capital * comm
    shares = (initial_capital - fee) / fill
    equity = shares * closes
    eq = pd.Series(equity, index=dates, name="equity")
    trades = pd.DataFrame(
        [
            {
                "date": dates[0],
                "side": "buy",
                "price": fill,
                "shares": shares,
                "fee": fee,
            }
        ]
    )
    return BacktestResult(
        equity=eq,
        positions=pd.Series(1, index=dates, name="position"),
        trades=trades,
        returns=eq.pct_change().fillna(0.0),
        params={"fast": None, "slow": None, "initial_capital": initial_capital},
        friction=fr,
        label=label,
    )
