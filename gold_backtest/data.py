"""Download and cache daily gold (GLD) OHLC from Yahoo Finance."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf

DEFAULT_TICKER = "GLD"
DEFAULT_CACHE_DIR = Path(__file__).resolve().parent.parent / "data"
DEFAULT_START = "2015-01-01"


def cache_path(ticker: str = DEFAULT_TICKER, cache_dir: Path | None = None) -> Path:
    root = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
    root.mkdir(parents=True, exist_ok=True)
    return root / f"{ticker.upper()}_daily.csv"


def download_ohlc(
    ticker: str = DEFAULT_TICKER,
    start: str = DEFAULT_START,
    end: str | None = None,
    cache_dir: Path | None = None,
    force: bool = False,
) -> pd.DataFrame:
    """
    Fetch daily OHLC for ``ticker`` and cache as CSV.

    Uses Yahoo Finance via yfinance (free public source). Reuses the local
    cache unless ``force`` is True or the cache is missing.
    """
    path = cache_path(ticker, cache_dir)
    if path.exists() and not force:
        return load_cached(path)

    raw = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    if raw.empty:
        raise RuntimeError(f"No data returned for {ticker} from Yahoo Finance")

    # yfinance may return MultiIndex columns for a single ticker
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [c[0] for c in raw.columns]

    df = raw.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )
    keep = [c for c in ("open", "high", "low", "close", "volume") if c in df.columns]
    df = df[keep].copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.index.name = "date"
    df = df.sort_index().dropna(subset=["close"])
    df.to_csv(path)
    return df


def load_cached(path: Path | str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"], index_col="date")
    df = df.sort_index()
    if "close" not in df.columns:
        raise ValueError(f"Cached file missing 'close' column: {path}")
    return df
