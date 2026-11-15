"""Check a forecast request before any agent runs (Lab 6.1, before generation).

A request is refused, with the reasons, when the stock is not in the universe, the date is in the
future, there is less than a year of prices before the date, or a data source cannot be reached.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Callable

from mosaic.config import SETTINGS, UNIVERSE


class RequestRefused(ValueError):
    def __init__(self, reasons: list[str]):
        super().__init__("; ".join(reasons))
        self.reasons = reasons


def check_request(ticker: str, as_of: datetime, *, price_days_before: int | None = None,
                  probes: dict[str, Callable[[], bool]] | None = None, universe=None,
                  now: datetime | None = None) -> list[str]:
    """Reasons to refuse the request (empty list = fine).

    price_days_before: number of daily prices known before as_of (None skips the check).
    probes: {"SEC EDGAR": fn, ...}; each fn returns True if the source answers.
    """
    universe = UNIVERSE if universe is None else universe
    now = now or datetime.now(timezone.utc)
    if as_of.tzinfo is None:
        as_of = as_of.replace(tzinfo=timezone.utc)
    reasons = []
    if ticker not in universe:
        reasons.append(f"{ticker} is not one of the {len(universe)} stocks MosAIc covers")
    if as_of > now:
        reasons.append(f"{as_of:%Y-%m-%d} is in the future")
    if price_days_before is not None and price_days_before < SETTINGS.sigma_lookback_days:
        reasons.append(f"only {price_days_before} days of prices before {as_of:%Y-%m-%d}; "
                       f"the baseline needs {SETTINGS.sigma_lookback_days}")
    for name, probe in (probes or {}).items():
        try:
            ok = probe()
        except Exception:                               # a probe that crashes counts as unreachable
            ok = False
        if not ok:
            reasons.append(f"{name} cannot be reached")
    return reasons


def require(ticker: str, as_of: datetime, **kw) -> None:
    reasons = check_request(ticker, as_of, **kw)
    if reasons:
        raise RequestRefused(reasons)
