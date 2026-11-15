"""Daily prices for a stock and the S&P 500 fund, cached on disk.

Prices come from Yahoo Finance (yfinance) by default; any `fetch(ticker, start, end) -> pandas.Series`
of adjusted closes can be passed instead, for tests or another provider.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import pandas as pd

from mosaic.config import SETTINGS

Fetch = Callable[[str, str, str], pd.Series]


def yahoo_fetch(ticker: str, start: str, end: str) -> pd.Series:     # pragma: no cover - needs internet
    import yfinance as yf
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    close = df["Close"]
    return (close.iloc[:, 0] if isinstance(close, pd.DataFrame) else close).dropna()


def load_closes(ticker: str, start: str, end: str, fetch: Fetch | None = None, cache_dir: str | Path | None = None) -> pd.Series:
    cache = Path(cache_dir or Path(SETTINGS.data_dir) / "prices") / f"{ticker}_{start}_{end}.csv"
    if cache.exists():
        s = pd.read_csv(cache, index_col=0, parse_dates=True).iloc[:, 0]
    else:
        s = (fetch or yahoo_fetch)(ticker, start, end)
        cache.parent.mkdir(parents=True, exist_ok=True)
        s.rename("close").to_csv(cache)
    s.index = pd.DatetimeIndex(s.index)
    return s.sort_index()


def price_table(ticker: str, start: str, end: str, fetch: Fetch | None = None, cache_dir=None) -> dict:
    """{"dates": [...], ticker: [...], "SPY": [...]} on the days both trade, in the shape the tools expect."""
    stock = load_closes(ticker, start, end, fetch, cache_dir)
    market = load_closes(SETTINGS.market_ticker, start, end, fetch, cache_dir)
    both = stock.index.intersection(market.index)
    dates = [datetime(d.year, d.month, d.day, tzinfo=timezone.utc) for d in both]
    return {"dates": dates, ticker: [float(x) for x in stock.loc[both]], "SPY": [float(x) for x in market.loc[both]]}
