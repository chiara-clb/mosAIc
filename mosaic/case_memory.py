"""Case memory: what happened after similar past events, from SEC filings and prices.

For an event type (an 8-K item, a 10-Q, a 10-K, or news) seen some days ago, case memory counts how
often past events of that type, at the same delay, were followed by a large fall, a flat month or a
large gain over 21 trading days. With the price reaction, only past events where the stock reacted the
same way (from big fall to big rise, against the S&P 500) count. Small samples are pulled toward the
frequencies of all events, so a rare event type starts from "average" instead of an extreme.

Only windows that ended before the forecast date are used.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from mosaic.config import OUTCOMES, SETTINGS
from mosaic.ledger import shift_for

NEUTRAL = {"fall": 0.3085, "flat": 0.383, "gain": 0.3085}


def _arrays(s: pd.Series):
    s = s.dropna().sort_index()
    return np.array(pd.DatetimeIndex(s.index).values, dtype="datetime64[D]"), np.log(s.to_numpy(dtype=float))


class CaseMemory:
    def __init__(self, events: dict[str, list[tuple[str, datetime]]], prices: dict[str, pd.Series],
                 market: pd.Series | None = None):
        """events: {ticker: [(event type, accepted time), ...]}; prices: {ticker: closes}; market: index closes."""
        self.events = events
        self.px = {t: _arrays(s) for t, s in prices.items()}
        self.mkt = _arrays(market) if market is not None else None
        self._windows: dict = {}
        self._reactions: dict = {}

    def reaction(self, ticker: str, accepted: datetime, lag_days: int) -> dict | None:
        """How the stock moved against the market since a filing, up to the last close before accepted + lag_days."""
        key = (ticker, accepted, lag_days)
        if key in self._reactions:
            return self._reactions[key]
        out = None
        if ticker in self.px:
            dates, logp = self.px[ticker]
            side = "right" if accepted.hour >= SETTINGS.after_close_utc_hour else "left"
            i0 = int(np.searchsorted(dates, np.datetime64(accepted.date(), "D"), side=side))
            e = int(np.searchsorted(dates, np.datetime64(accepted.date() + timedelta(days=lag_days), "D"))) - 1
            if 2 <= i0 <= e < len(dates):
                stock = float(logp[e] - logp[i0 - 1])
                market = 0.0
                if self.mkt is not None:
                    md, ml = self.mkt
                    a = int(np.searchsorted(md, dates[i0 - 1], side="right")) - 1
                    b = int(np.searchsorted(md, dates[e], side="right")) - 1
                    if a >= 0 and b >= a:
                        market = float(ml[b] - ml[a])
                r = np.diff(logp[max(0, i0 - 1 - SETTINGS.sigma_lookback_days):i0])
                daily = float(r.std()) if len(r) > 20 else 0.0
                abnormal = stock - market
                z = abnormal / (daily * math.sqrt(e - i0 + 1)) if daily > 0 else 0.0
                bucket = SETTINGS.reaction_labels[int(np.searchsorted(SETTINGS.reaction_edges, z, side="right"))]
                out = {"abnormal_return": round(abnormal, 4), "z": round(z, 2), "bucket": bucket, "days": e - i0 + 1}
        self._reactions[key] = out
        return out

    def window(self, ticker: str, start) -> tuple[str | None, object]:
        """Bucket of the 21 trading days from `start`, and the date the window ends."""
        key = (ticker, start)
        if key not in self._windows:
            dates, logp = self.px[ticker]
            i = int(np.searchsorted(dates, np.datetime64(start, "D")))
            n, h = SETTINGS.sigma_lookback_days, SETTINGS.horizon_days
            if i < n // 2 + 1 or i + h - 1 >= len(dates):
                self._windows[key] = (None, None)
            else:
                r = np.diff(logp[max(0, i - 1 - n):i])
                sigma = float(r.std() * math.sqrt(h))
                ret = float(logp[i + h - 1] - logp[i - 1])
                k = SETTINGS.bucket_k
                b = "fall" if ret < -k * sigma else "gain" if ret > k * sigma else "flat"
                self._windows[key] = (b, pd.Timestamp(dates[i + h - 1]).date())
        return self._windows[key]

    def overall(self, as_of: datetime) -> dict[str, float]:
        """Share of all 21-day windows (weekly starts, all stocks) in each bucket, ended before as_of."""
        counts = dict.fromkeys(OUTCOMES, 0)
        for t, (dates, _) in self.px.items():
            for d in dates[SETTINGS.sigma_lookback_days::5]:
                b, end = self.window(t, pd.Timestamp(d).date())
                if b and end < as_of.date():
                    counts[b] += 1
        n = sum(counts.values())
        return {k: counts[k] / n for k in OUTCOMES} if n else dict(NEUTRAL)

    def distribution(self, event: str, lag_days: int, as_of: datetime, p0: dict | None = None,
                     reaction_bucket: str | None = None) -> tuple[dict, int, dict]:
        """Smoothed bucket frequencies after past events of this type ("*" = any), the number used, and p0."""
        p0 = p0 or self.overall(as_of)
        prior = SETTINGS.prior_strength
        counts, counts_r = dict.fromkeys(OUTCOMES, 0), dict.fromkeys(OUTCOMES, 0)
        for t, evs in self.events.items():
            for key, accepted in evs:
                if (event != "*" and key != event) or accepted >= as_of:
                    continue
                b, end = self.window(t, accepted.date() + timedelta(days=lag_days))
                if not b or end >= as_of.date():
                    continue
                counts[b] += 1
                if reaction_bucket:
                    rx = self.reaction(t, accepted, lag_days)
                    if rx and rx["bucket"] == reaction_bucket:
                        counts_r[b] += 1
        n = sum(counts.values())
        q = {k: (counts[k] + prior * p0[k]) / (n + prior) for k in OUTCOMES}
        if not reaction_bucket:
            return q, n, p0
        n_r = sum(counts_r.values())
        return {k: (counts_r[k] + prior * q[k]) / (n_r + prior) for k in OUTCOMES}, n_r, p0

    def score(self, effect: str, event: str, lag_days: int, as_of: datetime, reaction_bucket: str | None = None,
              size: str = "medium", p0: dict | None = None) -> tuple[float, int]:
        """0 to 1; 0.5 means the past gives no reason either way. Returns (score, events used)."""
        q, n, p0 = self.distribution(event, lag_days, as_of, p0, reaction_bucket)
        if effect == "none":
            return 0.5, n
        s = shift_for(effect, size)
        p1 = {k: max(p0[k] + s[k] / 100, 0.01) for k in OUTCOMES}
        z = sum(p1.values())
        gain = sum(q[k] * (math.log(p1[k] / z) - math.log(p0[k])) for k in OUTCOMES)
        return 0.5 + 0.5 * math.tanh(gain / SETTINGS.base_rate_scale), n

    def rates(self, event: str, lag_days: int, as_of: datetime, reaction_bucket: str | None = None) -> dict:
        """Scores for every effect, in the shape the case_memory tool returns."""
        p0 = self.overall(as_of)
        out = {}
        for effect in ("fall", "gain", "spread", "none"):
            out[effect], n = self.score(effect, event, lag_days, as_of, reaction_bucket, p0=p0)
        out["n"] = n
        return out
