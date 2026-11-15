"""The baseline from prices and the earnings calendar (version C, chosen in the baseline comparison).

Recent moves (weights 0.94 per day) blended half and half with the past year, with results days left
out of both, so one big results-day jump does not make the next weeks look jumpy. When results are
expected within the month, the typical results-day move of the last four quarters is added.
Version A (recent moves only, results days included) is kept for comparison.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta

import numpy as np

from mosaic.config import SETTINGS
from mosaic.ledger import floor_and_normalise

FIXED_GUESS = {"fall": 30.85, "flat": 38.3, "gain": 30.85}


def norm_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def results_day_mask(dates: list[datetime], results_times: list[datetime]) -> np.ndarray:
    """True for the trading days that carried a results announcement's price move.

    A filing at or after 20:00 UTC (after the US close) moves the stock the next trading day.
    """
    days = np.array([np.datetime64(d.date(), "D") for d in dates])
    mask = np.zeros(len(days), dtype=bool)
    for a in results_times:
        side = "right" if a.hour >= SETTINGS.after_close_utc_hour else "left"
        k = int(np.searchsorted(days, np.datetime64(a.date(), "D"), side=side))
        if k < len(days):
            mask[k] = True
    return mask


def main_results(r202: list[datetime], reports: list[datetime], max_gap_days: int | None = None) -> list[datetime]:
    """One results announcement per quarter: the last item-2.02 filing within 75 days before each 10-Q or 10-K.

    Some companies also file item 2.02 for mid-quarter updates; those are dropped.
    """
    gap = timedelta(days=max_gap_days or SETTINGS.results_max_gap_days)
    r202 = sorted(set(r202))
    kept = set()
    for rep in reports:
        before = [a for a in r202 if rep - gap <= a <= rep]
        if before:
            kept.add(before[-1])
    last_rep = max(reports) if reports else None
    tail = [a for a in r202 if last_rep is None or a > last_rep]
    if tail:
        kept.add(tail[-1])
    return sorted(kept)


def results_dates(filings: list[dict]) -> list[datetime]:
    """Acceptance times of the quarterly results 8-Ks, from an EDGAR filing list."""
    from mosaic.edgar import parse_accepted
    r202 = [parse_accepted(r["acceptanceDateTime"]) for r in filings
            if r["form"] == "8-K" and "2.02" in str(r.get("items", ""))]
    reports = [parse_accepted(r["acceptanceDateTime"]) for r in filings if r["form"] in ("10-Q", "10-K")]
    return main_results(r202, reports)


def too_few_results(results: list[datetime], as_of: datetime, years: int = 2) -> bool:
    """Fewer than one results date per quarter over the last two years (e.g. XOM's incomplete SEC list)."""
    recent = [d for d in results if as_of - timedelta(days=365 * years) <= d < as_of]
    return len(recent) < 4 * years - 1


def baseline(dates: list[datetime], closes: list[float], as_of: datetime, results_times: list[datetime],
             results_in_window: bool, version: str | None = None) -> tuple[dict[str, float], dict]:
    """Fall / flat / gain in percent, from closes strictly before as_of. Returns (probabilities, details)."""
    version = version or SETTINGS.baseline_version
    keep_days = [i for i, d in enumerate(dates) if d < as_of]
    d = [dates[i] for i in keep_days]
    c = np.array([closes[i] for i in keep_days], dtype=float)
    info: dict = {"version": version}
    n = SETTINGS.sigma_lookback_days
    if len(c) < n // 2 + 1:
        info["fallback"] = f"only {len(c)} prices before {as_of:%Y-%m-%d}"
        return dict(FIXED_GUESS), info
    r = np.diff(np.log(c))[-n:]
    sigma_year = float(r.std() * math.sqrt(SETTINGS.horizon_days))
    w = SETTINGS.ewma_decay ** np.arange(len(r))[::-1]            # recent days weigh more
    past = sorted(t for t in results_times if t < as_of)
    on_results = results_day_mask(d, past)[1:][-n:]
    if version == "A":
        var_month = float(np.sum(w * r ** 2) / np.sum(w)) * SETTINGS.horizon_days
    else:
        keep = ~on_results
        recent = float(np.sum(w[keep] * r[keep] ** 2) / np.sum(w[keep]))
        year = float(np.mean(r[keep] ** 2))
        var_month = (SETTINGS.blend_recent * recent + (1 - SETTINGS.blend_recent) * year) * SETTINGS.horizon_days
    moves = np.abs(r[on_results])[-SETTINGS.results_jump_last:]
    if results_in_window and len(moves):
        move = float(np.mean(moves))
        var_month += move ** 2
        info["results_move"] = round(move, 4)
    s = math.sqrt(var_month)
    lower, upper = -SETTINGS.bucket_k * sigma_year, SETTINGS.bucket_k * sigma_year
    fall, gain = norm_cdf(lower / s), 1 - norm_cdf(upper / s)
    info.update(sigma_year=round(sigma_year, 4), sigma_month=round(s, 4), results_days_left_out=int(on_results.sum()))
    return floor_and_normalise({"fall": 100 * fall, "flat": 100 * (1 - fall - gain), "gain": 100 * gain}), info
