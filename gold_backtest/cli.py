"""CLI entry point for the gold trend/momentum backtester."""

from __future__ import annotations

import json
from pathlib import Path

import click

from gold_backtest.backtest import Friction, buy_and_hold, simulate
from gold_backtest.data import DEFAULT_START, DEFAULT_TICKER, download_ohlc
from gold_backtest.metrics import beat_buy_and_hold, format_summary, summarize
from gold_backtest.strategy import DEFAULT_FAST, DEFAULT_SLOW
from gold_backtest.walkforward import run_train_test, run_walk_forward


@click.command()
@click.option("--ticker", default=DEFAULT_TICKER, show_default=True, help="Yahoo Finance ticker (GLD recommended).")
@click.option("--start", default=DEFAULT_START, show_default=True, help="History start date (YYYY-MM-DD).")
@click.option("--end", default=None, help="History end date (optional).")
@click.option("--fast", default=None, type=int, help="Fast SMA period. If omitted with --slow, params are selected on train.")
@click.option("--slow", default=None, type=int, help="Slow SMA period.")
@click.option("--train-ratio", default=0.7, show_default=True, type=float, help="Chronological train fraction.")
@click.option("--commission-bps", default=5.0, show_default=True, type=float, help="Commission per side in basis points.")
@click.option("--slippage-bps", default=5.0, show_default=True, type=float, help="Slippage per side in basis points.")
@click.option("--capital", default=100_000.0, show_default=True, type=float, help="Starting capital.")
@click.option("--cache-dir", default=None, type=click.Path(), help="OHLC cache directory.")
@click.option("--force-download", is_flag=True, help="Ignore cache and re-download.")
@click.option("--walk-forward", "do_wf", is_flag=True, help="Also run rolling walk-forward folds.")
@click.option("--json-out", default=None, type=click.Path(), help="Write full results JSON to this path.")
@click.option(
    "--baseline",
    is_flag=True,
    help="Use fixed default SMA(50/200) on full sample + train/test (no param search).",
)
def main(
    ticker: str,
    start: str,
    end: str | None,
    fast: int | None,
    slow: int | None,
    train_ratio: float,
    commission_bps: float,
    slippage_bps: float,
    capital: float,
    cache_dir: str | None,
    force_download: bool,
    do_wf: bool,
    json_out: str | None,
    baseline: bool,
) -> None:
    """
    Backtest a long-only dual moving-average crossover on gold (GLD).

    Downloads/caches daily OHLC, applies commission+slippage, and reports
    train/test (or fixed) metrics versus buy-and-hold.
    """
    cache = Path(cache_dir) if cache_dir else None
    click.echo(f"Loading {ticker} daily OHLC from {start}" + (f" to {end}" if end else "") + " ...")
    ohlc = download_ohlc(
        ticker=ticker,
        start=start,
        end=end,
        cache_dir=cache,
        force=force_download,
    )
    click.echo(f"Bars: {len(ohlc)}  ({ohlc.index[0].date()} → {ohlc.index[-1].date()})")

    friction = Friction(commission_bps=commission_bps, slippage_bps=slippage_bps)
    click.echo(
        f"Friction: commission={commission_bps} bps/side, "
        f"slippage={slippage_bps} bps/side "
        f"(~{(commission_bps + slippage_bps) * 2:.0f} bps round-trip)"
    )

    if (fast is None) ^ (slow is None):
        raise click.UsageError("Provide both --fast and --slow, or neither (or use --baseline).")

    use_fixed = baseline or (fast is not None and slow is not None)
    f_param = DEFAULT_FAST if baseline or fast is None else fast
    s_param = DEFAULT_SLOW if baseline or slow is None else slow

    # Full-sample snapshot with documented SMA(50/200) defaults (always shown)
    full_strat = simulate(
        ohlc,
        fast=DEFAULT_FAST,
        slow=DEFAULT_SLOW,
        friction=friction,
        initial_capital=capital,
        label=f"full SMA({DEFAULT_FAST}/{DEFAULT_SLOW})",
    )
    full_bh = buy_and_hold(ohlc, friction=friction, initial_capital=capital, label="full buy&hold")

    click.echo("\n=== Full sample (documented SMA defaults; in-sample) ===")
    click.echo(format_summary([summarize(full_strat), summarize(full_bh)]))

    if use_fixed:
        split = run_train_test(
            ohlc,
            friction=friction,
            train_ratio=train_ratio,
            initial_capital=capital,
            fixed_fast=f_param,
            fixed_slow=s_param,
        )
        click.echo(
            f"\n=== Train/test split ({train_ratio:.0%} / {1 - train_ratio:.0%}) "
            f"fixed SMA({f_param}/{s_param}) ==="
        )
    else:
        split = run_train_test(
            ohlc,
            friction=friction,
            train_ratio=train_ratio,
            initial_capital=capital,
        )
        click.echo(
            f"\n=== Train/test split ({train_ratio:.0%} / {1 - train_ratio:.0%}); "
            f"params selected on train → SMA({split.best_params['fast']}/{split.best_params['slow']}) ==="
        )

    click.echo("\n-- Train --")
    click.echo(format_summary([split.train_summary, split.train_bh]))
    click.echo("\n-- Test (out-of-sample) --")
    click.echo(format_summary([split.test_summary, split.test_bh]))

    won = beat_buy_and_hold(split.test_summary, split.test_bh)
    click.echo(
        "\nOut-of-sample verdict: "
        + (
            "strategy CAGR beat buy-and-hold gold."
            if won
            else "strategy CAGR did NOT beat buy-and-hold gold."
        )
    )

    payload: dict = {
        "ticker": ticker,
        "bars": len(ohlc),
        "range": [str(ohlc.index[0].date()), str(ohlc.index[-1].date())],
        "friction": {
            "commission_bps": commission_bps,
            "slippage_bps": slippage_bps,
        },
        "rules": {
            "type": "dual_sma_crossover_long_only",
            "long_when": "sma_fast > sma_slow",
            "flat_when": "sma_fast <= sma_slow",
            "fill": "next open +/- slippage; commission on notional",
        },
        "full_sample": [summarize(full_strat), summarize(full_bh)],
        "best_params": split.best_params,
        "train": [split.train_summary, split.train_bh],
        "test": [split.test_summary, split.test_bh],
        "test_beat_buy_and_hold": won,
    }

    if do_wf:
        wf = run_walk_forward(ohlc, friction=friction, initial_capital=capital)
        click.echo("\n=== Walk-forward folds (test windows) ===")
        if not wf:
            click.echo("Not enough history for default walk-forward windows.")
        else:
            for row in wf:
                mark = "YES" if row["beat_bh"] else "no"
                click.echo(
                    f"  fold {row['fold']}: SMA({row['fast']}/{row['slow']}) "
                    f"{row['test_start']}→{row['test_end']}  "
                    f"CAGR {row['strategy_cagr']:.2%} vs BH {row['bh_cagr']:.2%}  "
                    f"beat={mark}"
                )
            beats = sum(1 for r in wf if r["beat_bh"])
            click.echo(f"Walk-forward folds beating BH: {beats}/{len(wf)}")
        payload["walk_forward"] = wf

    if json_out:
        out = Path(json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        # Make JSON-safe
        def _clean(obj):
            if isinstance(obj, dict):
                return {k: _clean(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_clean(v) for v in obj]
            if isinstance(obj, float):
                if obj != obj:  # NaN
                    return None
                return obj
            return obj

        out.write_text(json.dumps(_clean(payload), indent=2, default=str))
        click.echo(f"\nWrote {out}")


if __name__ == "__main__":
    main()
