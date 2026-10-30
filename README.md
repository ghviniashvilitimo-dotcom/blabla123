# Gold trend / momentum backtester

Offline **long-only** backtest of a dual moving-average crossover on gold via the **GLD** ETF (Yahoo Finance daily OHLC). No live or paper trading.

## Strategy rules

1. Compute simple moving averages of the adjusted **close**: `SMA_fast` and `SMA_slow` (defaults **50 / 200**).
2. **Long** when `SMA_fast > SMA_slow`; **flat** (cash) when `SMA_fast <= SMA_slow`.
3. Signal is known at day *t* close; fills use the **next session open**.
4. Long-only, no leverage, fully invested when in the market.

Documented defaults (`--baseline`) use fixed 50/200. Without fixed params, the CLI selects `(fast, slow)` from a small grid by **training-window Sharpe**, then evaluates on the held-out test window.

## Friction assumptions

| Cost | Default | Notes |
|------|---------|--------|
| Commission | **5 bps** per side | Fraction of fill notional |
| Slippage | **5 bps** per side | Buy: open × (1 + slip); sell: open × (1 − slip) |

Approximate **round-trip** cost ≈ **20 bps** (0.20%) of notional. Override with `--commission-bps` / `--slippage-bps`.

## Data

- Source: Yahoo Finance via [`yfinance`](https://github.com/ranaroussi/yfinance) (free, public).
- Default ticker: `GLD`, start `2015-01-01` (~5+ years).
- CSV cache: `data/GLD_daily.csv` (created on first run; refresh with `--force-download`).

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
# Documented SMA(50/200) + chronological train/test (70/30)
python3 -m gold_backtest.cli --baseline

# Train-select params, report out-of-sample vs buy-and-hold
python3 -m gold_backtest.cli

# Also print rolling walk-forward folds
python3 -m gold_backtest.cli --walk-forward

# Custom params / dates
python3 -m gold_backtest.cli --fast 50 --slow 200 --start 2015-01-01 --json-out results.json
```

Equivalent: `python3 scripts/run_backtest.py …`

## Metrics

Reported for strategy and **buy-and-hold gold** on the same window:

- Total return, CAGR, max drawdown, Sharpe (daily, √252, rf=0)
- Win rate (closed round-trips), trade count

Primary honesty check: **out-of-sample (test) CAGR vs buy-and-hold**, not full-sample fit alone.

## Project layout

```
gold_backtest/
  data.py          # download + cache
  strategy.py      # dual SMA signals
  backtest.py      # simulation + friction
  metrics.py       # performance stats
  walkforward.py   # train/test + walk-forward
  cli.py           # Click CLI
scripts/run_backtest.py
data/              # cached OHLC (gitignored)
```

## Disclaimer

Research / education only. Past simulated performance is not indicative of future results. Not investment advice.
