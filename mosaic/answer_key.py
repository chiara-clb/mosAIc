"""The answer key: the market's fall / flat / gain probabilities implied by option prices.

Used only for evaluation. No agent module may import this file (a test checks it).

How it works, for one stock on one date:
1. Take the option quotes at the close of the last trading day before the as-of date.
2. For each expiry, turn mid prices of out-of-the-money options into implied volatilities,
   and draw a smooth curve through them across strikes (the "smile").
3. The market's probability that the stock ends below a price K is read from how call prices
   change with the strike: P(S_T < K) = 1 + e^{rT} dC/dK. dC/dK is taken numerically from
   Black-Scholes prices evaluated on the smooth smile, which keeps the skew in the answer.
4. Read that probability at the two bucket thresholds (buckets.Thresholds), and interpolate
   in time between the two expiries around the target date (21 trading days ahead).

Standard chain format (one row per option quote), which a provider loader must produce:
    quote_date, expiration, strike, right ("C" or "P"), bid, ask, underlying_price
Rates: `rate` is the continuously compounded risk-free rate, `div_yield` the dividend yield.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq
from scipy.stats import norm

from .buckets import Thresholds
from .config import SETTINGS

REQUIRED_COLUMNS = ("quote_date", "expiration", "strike", "right", "bid", "ask", "underlying_price")


@dataclass(frozen=True)
class ChainRules:
    min_quotes_per_side: int = 4        # usable out-of-the-money quotes below and above the forward
    max_rel_spread: float = 0.5         # (ask - bid) / mid
    max_expiry_gap_days: int = 10       # when only one expiry is usable, it must be this close to the target
    min_days_to_expiry: int = 5


@dataclass
class AnswerKey:
    status: str                          # "ok" or "unscorable"
    probabilities: dict[str, float] = field(default_factory=dict)   # percent, sums to 100
    reason: str = ""
    expiries_used: list[str] = field(default_factory=list)
    target_date: str = ""


# ---------- Black-Scholes pieces ----------

def bs_call(fwd: float, strike: float, t: float, vol: float, disc: float) -> float:
    if vol <= 0 or t <= 0:
        return disc * max(fwd - strike, 0.0)
    sd = vol * math.sqrt(t)
    d1 = (math.log(fwd / strike) + 0.5 * sd * sd) / sd
    return disc * (fwd * norm.cdf(d1) - strike * norm.cdf(d1 - sd))


def bs_put(fwd: float, strike: float, t: float, vol: float, disc: float) -> float:
    return bs_call(fwd, strike, t, vol, disc) - disc * (fwd - strike)


def implied_vol(price: float, fwd: float, strike: float, t: float, disc: float, right: str) -> float | None:
    f = bs_call if right == "C" else bs_put
    intrinsic = disc * max((fwd - strike) if right == "C" else (strike - fwd), 0.0)
    upper = disc * (fwd if right == "C" else strike)
    if not (intrinsic < price < upper):
        return None
    try:
        return brentq(lambda v: f(fwd, strike, t, v, disc) - price, 1e-4, 5.0, xtol=1e-8)
    except ValueError:
        return None


# ---------- One expiry ----------

@dataclass
class _Smile:
    t: float
    fwd: float
    disc: float
    curve: PchipInterpolator
    k_min: float
    k_max: float

    def vol(self, strike: float) -> float:
        x = math.log(strike / self.fwd)
        lo, hi = self.curve.x[0], self.curve.x[-1]
        return float(self.curve(min(max(x, lo), hi)))

    def cdf(self, strike: float) -> float:
        h = 1e-3 * strike
        up = bs_call(self.fwd, strike + h, self.t, self.vol(strike + h), self.disc)
        dn = bs_call(self.fwd, strike - h, self.t, self.vol(strike - h), self.disc)
        return 1.0 + (up - dn) / (2 * h) / self.disc


def _fit_smile(rows: pd.DataFrame, spot: float, t: float, rate: float, div_yield: float,
               rules: ChainRules) -> _Smile | str:
    disc = math.exp(-rate * t)
    fwd = spot * math.exp((rate - div_yield) * t)
    rows = rows.copy()
    rows["mid"] = (rows["bid"] + rows["ask"]) / 2
    ok = (rows["bid"] > 0) & (rows["ask"] > rows["bid"]) & ((rows["ask"] - rows["bid"]) / rows["mid"] <= rules.max_rel_spread)
    rows = rows[ok]
    otm = rows[((rows["right"] == "P") & (rows["strike"] < fwd)) | ((rows["right"] == "C") & (rows["strike"] >= fwd))]
    points = []
    for r in otm.itertuples():
        v = implied_vol(r.mid, fwd, r.strike, t, disc, r.right)
        if v is not None:
            points.append((math.log(r.strike / fwd), v, r.strike))
    below = sum(1 for p in points if p[2] < fwd)
    above = len(points) - below
    if below < rules.min_quotes_per_side or above < rules.min_quotes_per_side:
        return f"too few usable quotes ({below} below, {above} above the forward)"
    points.sort()
    xs = np.array([p[0] for p in points])
    vs = np.array([p[1] for p in points])
    keep = np.concatenate([[True], np.diff(xs) > 1e-9])
    curve = PchipInterpolator(xs[keep], vs[keep], extrapolate=False)
    return _Smile(t=t, fwd=fwd, disc=disc, curve=curve,
                  k_min=min(p[2] for p in points), k_max=max(p[2] for p in points))


def _probs_for_smile(sm: _Smile, th: Thresholds) -> dict[str, float] | str:
    if sm.k_min > th.lower_price or sm.k_max < th.upper_price:
        return "strikes do not cover both thresholds"
    lo, hi = sm.cdf(th.lower_price), sm.cdf(th.upper_price)
    if not (-1e-6 <= lo <= hi <= 1 + 1e-6):
        return f"implied distribution not valid (P(below lower)={lo:.3f}, P(below upper)={hi:.3f})"
    lo, hi = min(max(lo, 0.0), 1.0), min(max(hi, 0.0), 1.0)
    return {"fall": lo, "flat": hi - lo, "gain": 1.0 - hi}


# ---------- Public API ----------

def target_date(quote_date: date, horizon: int | None = None) -> date:
    """Approximate date of the 21st trading day after the as-of date (business days, no holidays)."""
    horizon = horizon or SETTINGS.horizon_days
    return (pd.Timestamp(quote_date) + pd.offsets.BDay(horizon)).date()


def validate_chain(chain: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in chain.columns]
    if missing:
        raise ValueError(f"option chain is missing columns: {missing}")


def implied_bucket_probabilities(chain: pd.DataFrame, th: Thresholds, rate: float = 0.0,
                                 div_yield: float = 0.0, target: date | None = None,
                                 rules: ChainRules | None = None) -> AnswerKey:
    """Market-implied fall / flat / gain probabilities (percent) for the thresholds `th`."""
    rules = rules or ChainRules()
    validate_chain(chain)
    chain = chain.copy()
    chain["quote_date"] = pd.to_datetime(chain["quote_date"]).dt.date
    chain["expiration"] = pd.to_datetime(chain["expiration"]).dt.date
    chain["right"] = chain["right"].str.upper().str[0]

    quote_days = sorted(d for d in chain["quote_date"].unique() if d < th.as_of)
    if not quote_days:
        return AnswerKey("unscorable", reason="no quotes before the as-of date")
    qd = quote_days[-1]
    chain = chain[chain["quote_date"] == qd]
    tgt = target or target_date(qd)
    spot = float(chain["underlying_price"].iloc[0])

    fitted: list[tuple[date, dict[str, float]]] = []
    reasons = []
    for exp, rows in chain.groupby("expiration"):
        days = (exp - qd).days
        if days < rules.min_days_to_expiry:
            continue
        sm = _fit_smile(rows, spot, days / 365.0, rate, div_yield, rules)
        if isinstance(sm, str):
            reasons.append(f"{exp}: {sm}")
            continue
        p = _probs_for_smile(sm, th)
        if isinstance(p, str):
            reasons.append(f"{exp}: {p}")
            continue
        fitted.append((exp, p))

    before = [f for f in fitted if f[0] <= tgt]
    after = [f for f in fitted if f[0] >= tgt]
    if before and after:
        (e0, p0), (e1, p1) = before[-1], after[0]
        w = 0.0 if e1 == e0 else (tgt - e0).days / (e1 - e0).days
        probs = {k: (1 - w) * p0[k] + w * p1[k] for k in p0}
        used = sorted({str(e0), str(e1)})
    elif fitted:
        e, p = min(fitted, key=lambda f: abs((f[0] - tgt).days))
        if abs((e - tgt).days) > rules.max_expiry_gap_days:
            return AnswerKey("unscorable", reason=f"nearest usable expiry {e} is too far from {tgt}",
                             target_date=str(tgt))
        probs, used = p, [str(e)]
    else:
        return AnswerKey("unscorable", reason="; ".join(reasons) or "no usable expiry", target_date=str(tgt))

    total = sum(probs.values())
    pct = {k: 100.0 * v / total for k, v in probs.items()}
    return AnswerKey("ok", probabilities=pct, expiries_used=used, target_date=str(tgt))


def log_unscorable(path: str | Path, ticker: str, as_of, key: AnswerKey) -> None:
    """Append an unscorable date to a JSONL log, so dropped dates are counted (CP1.1 feedback)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps({"ticker": ticker, "as_of": str(as_of), **asdict(key)}) + "\n")
