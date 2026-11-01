"""The forecast definition: bucket thresholds and the realised bucket.

Full design §1: fall < -0.5 sigma, flat in between, gain > +0.5 sigma, where sigma is the
stock's trailing one-year volatility of daily log returns, scaled to 21 trading days.

Conventions used everywhere in the package:
- "as of" a date means at the start of that day: only closes on earlier days are known.
- Returns are log returns of adjusted closes, so the thresholds on the price are
  start_price * exp(-k sigma) and start_price * exp(+k sigma). The answer key reads the
  option-implied distribution at exactly these two prices.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime

import numpy as np
import pandas as pd

from .config import SETTINGS


def _as_date(d: date | datetime | pd.Timestamp) -> date:
    if isinstance(d, pd.Timestamp):
        return d.date()
    if isinstance(d, datetime):
        return d.date()
    return d


def prices_before(prices: pd.Series, as_of) -> pd.Series:
    """Closes strictly before the as-of date (what is known at the start of that day)."""
    idx = pd.DatetimeIndex(prices.index)
    mask = np.array([d < _as_date(as_of) for d in idx.date])
    return prices[mask].dropna()


@dataclass(frozen=True)
class Thresholds:
    as_of: date
    start_date: date        # date of the last close before as_of
    start_price: float
    sigma: float            # one-year volatility scaled to the horizon, in log-return units
    lower: float            # log-return threshold for "fall"
    upper: float            # log-return threshold for "gain"

    @property
    def lower_price(self) -> float:
        return self.start_price * math.exp(self.lower)

    @property
    def upper_price(self) -> float:
        return self.start_price * math.exp(self.upper)


def horizon_sigma(prices: pd.Series, as_of, lookback: int | None = None, horizon: int | None = None) -> float:
    """Trailing volatility of daily log returns over `lookback` days, scaled to `horizon` days."""
    lookback = lookback or SETTINGS.sigma_lookback_days
    horizon = horizon or SETTINGS.horizon_days
    past = prices_before(prices, as_of)
    r = np.diff(np.log(past.to_numpy(dtype=float)))[-lookback:]
    if len(r) < lookback // 2:
        raise ValueError(f"need at least {lookback // 2} daily returns before {as_of}, got {len(r)}")
    return float(r.std() * math.sqrt(horizon))


def thresholds(prices: pd.Series, as_of, k: float | None = None) -> Thresholds:
    k = SETTINGS.bucket_k if k is None else k
    past = prices_before(prices, as_of)
    if past.empty:
        raise ValueError(f"no prices before {as_of}")
    sigma = horizon_sigma(prices, as_of)
    return Thresholds(
        as_of=_as_date(as_of),
        start_date=pd.Timestamp(past.index[-1]).date(),
        start_price=float(past.iloc[-1]),
        sigma=sigma,
        lower=-k * sigma,
        upper=k * sigma,
    )


def bucket_of(log_return: float, th: Thresholds) -> str:
    """Strict inequalities: a return exactly on a threshold counts as flat."""
    if log_return < th.lower:
        return "fall"
    if log_return > th.upper:
        return "gain"
    return "flat"


def realised_return(prices: pd.Series, as_of, horizon: int | None = None) -> float | None:
    """Log return from the last close before as_of to the close `horizon` trading days later.

    Returns None when the price history does not reach that far yet.
    """
    horizon = horizon or SETTINGS.horizon_days
    past = prices_before(prices, as_of)
    if past.empty:
        return None
    idx = pd.DatetimeIndex(prices.index)
    after = prices[np.array([d >= _as_date(as_of) for d in idx.date])].dropna()
    if len(after) < horizon:
        return None
    return float(math.log(after.iloc[horizon - 1] / past.iloc[-1]))


def realised_bucket(prices: pd.Series, as_of, th: Thresholds | None = None) -> str | None:
    th = th or thresholds(prices, as_of)
    r = realised_return(prices, as_of)
    return None if r is None else bucket_of(r, th)
