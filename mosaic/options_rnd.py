"""The option market's distribution of the next-day move after a results release (baseline B3).

Ported from the earnings-release prototype (MosAIc v2, quant/options_rnd.py). For one release:

1. For each expiry after the release day t0: discount D = exp(-RATE * days / 365), and forward
   F = median of K + (C - P) / D over the ~10 call/put pairs nearest the money (put-call parity:
   FirstRate files have no stock price column).
2. Out-of-the-money mid quotes (puts below F, calls above), bid > 0 and relative spread at most
   MAX_REL_SPREAD, inverted with Black-76 to total implied volatility s = sigma * sqrt(T).
3. Smile: a smoothing spline of s against log-moneyness k = log(K / F), flat beyond the quoted strikes.
4. Front expiry: call prices on a fine grid from the smile, and the distribution of x = log(S_T / F)
   from P(S_T <= K) = 1 + dC/dK / D (Breeden-Litzenberger), made non-decreasing.
5. One day: if the front expiry is the next session, that distribution is used as it is. Otherwise its
   variance V_T = V_event + n_T * s2, with s2 the ordinary daily variance (slope of at-the-money variance
   against sessions over the later expiries), so V_1 = V_T - (n_T - 1) * s2, and the distribution is
   rescaled around its median by sqrt(V_1 / V_T).
6. B3 = (P(x < -tau), P(-tau <= x <= tau), P(x > tau)).

Used only to show and score the market's view. No agent module imports this file.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import optimize, stats
from scipy.interpolate import make_smoothing_spline

MAX_REL_SPREAD = 0.5
RATE = 0.04                 # annual rate for discounting; over a few days its effect is about 0.05%
MIN_QUOTES = 6              # usable out-of-the-money quotes needed to fit a smile
MAX_LATER_SESSIONS = 30     # later expiries used for the daily variance: up to this many sessions after t0
GRID = 4001


def black(F: float, K, s, D: float, call) -> np.ndarray:
    """Black-76 price with total volatility s = sigma * sqrt(T)."""
    K, s = np.asarray(K, float), np.maximum(np.asarray(s, float), 1e-8)
    d1 = (np.log(F / K) + 0.5 * s ** 2) / s
    d2 = d1 - s
    c = D * (F * stats.norm.cdf(d1) - K * stats.norm.cdf(d2))
    p = D * (K * stats.norm.cdf(-d2) - F * stats.norm.cdf(-d1))
    return np.where(call, c, p)


def implied_s(price: float, F: float, K: float, D: float, call: bool) -> float:
    """Total implied volatility, or nan when the price is outside no-arbitrage bounds."""
    intrinsic = D * max(F - K, 0.0) if call else D * max(K - F, 0.0)
    upper = D * F if call else D * K
    if not (intrinsic < price < upper):
        return math.nan
    try:
        return optimize.brentq(lambda s: float(black(F, K, s, D, call)) - price, 1e-6, 5.0, xtol=1e-10)
    except ValueError:
        return math.nan


@dataclass
class Smile:
    expiry: pd.Timestamp
    sessions: int                # trading sessions in (t0, expiry]
    F: float
    D: float
    k: np.ndarray                # log-moneyness of the quotes used
    s: np.ndarray                # their total implied volatility
    n_quotes: int
    spline: object = field(repr=False, default=None)

    def s_at(self, k) -> np.ndarray:
        k = np.clip(np.asarray(k, float), self.k[0], self.k[-1])
        return np.maximum(self.spline(k), 1e-4)

    @property
    def atm_w(self) -> float:
        return float(self.s_at(0.0) ** 2)


def parity_forward(calls: pd.Series, puts: pd.Series, D: float) -> float:
    both = calls.index.intersection(puts.index)
    diff = (calls[both] - puts[both]).sort_index()
    if len(diff) < 1:
        raise ValueError("no call/put pairs")
    near = diff.abs().sort_values().index[:10]
    return float(np.median(np.array(near, float) + diff[near].to_numpy(float) / D))


def fit_smile(q: pd.DataFrame, expiry: pd.Timestamp, sessions: int, t0: pd.Timestamp) -> Smile:
    """Smile for one expiry from its quotes (columns strike, right "C"/"P", bid, ask)."""
    q = q[(q["bid"] > 0) & (q["ask"] > q["bid"])].copy()
    q["mid"] = (q["bid"] + q["ask"]) / 2
    calls = q[q["right"] == "C"].groupby("strike")["mid"].mean()
    puts = q[q["right"] == "P"].groupby("strike")["mid"].mean()
    D = math.exp(-RATE * (expiry - t0).days / 365)
    F = parity_forward(calls, puts, D)
    q = q[(q["ask"] - q["bid"]) / q["mid"] <= MAX_REL_SPREAD]
    otm = q[((q["right"] == "P") & (q["strike"] < F)) | ((q["right"] == "C") & (q["strike"] >= F))]
    otm = otm.groupby("strike").agg(mid=("mid", "mean"), right=("right", "first")).reset_index()
    s = np.array([implied_s(m, F, K, D, r == "C") for K, m, r in otm[["strike", "mid", "right"]].itertuples(index=False)])
    k = np.log(otm["strike"].to_numpy(float) / F)
    ok = np.isfinite(s)
    k, s = k[ok], s[ok]
    if len(k) < MIN_QUOTES:
        raise ValueError(f"only {len(k)} usable out-of-the-money quotes for {expiry.date()}")
    order = np.argsort(k)
    k, s = k[order], s[order]
    return Smile(expiry, sessions, F, D, k, s, len(k), make_smoothing_spline(k, s))


@dataclass
class Distribution:
    """The distribution of a log return x, as a cumulative probability on a grid."""
    x: np.ndarray
    F: np.ndarray
    repair: float = 0.0

    def cdf(self, v) -> np.ndarray:
        return np.interp(v, self.x, self.F, left=0.0, right=1.0)

    @property
    def median(self) -> float:
        return float(np.interp(0.5, self.F, self.x))

    def rescale(self, scale: float) -> "Distribution":
        m = self.median
        return Distribution(m + (self.x - m) * scale, self.F.copy(), self.repair)

    def buckets(self, tau: float) -> np.ndarray:
        fall, below_gain = float(self.cdf(-tau)), float(self.cdf(tau))
        return np.array([fall, below_gain - fall, 1.0 - below_gain])


def rnd(sm: Smile) -> Distribution:
    """Breeden-Litzenberger distribution of x = log(S_T / F) for one expiry."""
    width = max(8 * math.sqrt(sm.atm_w), 0.5, abs(sm.k[0]) + 0.1, abs(sm.k[-1]) + 0.1)
    x = np.linspace(-width, width, GRID)
    K = sm.F * np.exp(x)
    C = black(sm.F, K, sm.s_at(x), sm.D, np.ones_like(x, bool))
    cdf = np.clip(1.0 + np.gradient(C, K) / sm.D, 0.0, 1.0)
    mono = np.maximum.accumulate(cdf)
    repair = float(np.abs(mono - cdf).max())
    mono[0], mono[-1] = 0.0, 1.0
    return Distribution(x, mono, repair)


def sessions_between(t0: pd.Timestamp, day: pd.Timestamp, sessions: pd.DatetimeIndex) -> int:
    return int(((sessions > t0) & (sessions <= day)).sum())


def daily_variance(smiles: list[Smile]) -> tuple[float | None, str]:
    """Ordinary daily implied variance: slope of at-the-money variance against sessions (two expiries or more)."""
    if len(smiles) < 2:
        return None, "fewer than two later expiries"
    n = np.array([s.sessions for s in smiles], float)
    w = np.array([s.atm_w for s in smiles])
    if np.ptp(n) == 0:
        return None, "later expiries on one day"
    slope = float(np.polyfit(n, w, 1)[0])
    if slope <= 0:
        return None, "non-positive slope"
    return slope, f"slope over {len(smiles)} expiries"


@dataclass
class OptionsResult:
    probs: np.ndarray            # (fall, flat, gain), fractions
    diagnostics: dict


def next_day_distribution(chain: pd.DataFrame, t0: pd.Timestamp, t1: pd.Timestamp, tau: float,
                          sessions: pd.DatetimeIndex, fallback_var: float) -> OptionsResult:
    """B3 for one release. `chain`: quotes at the close on t0 (strike, expiration, right, bid, ask)."""
    chain = chain.assign(expiration=pd.to_datetime(chain["expiration"]))
    exps = sorted(e for e in chain["expiration"].unique() if pd.Timestamp(e) > t0)
    exps = [pd.Timestamp(e) for e in exps]
    if not exps or exps[0] < t1:
        raise ValueError("no expiry on or after the next session")
    front_exp = exps[0]
    n_T = sessions_between(t0, front_exp, sessions)
    front = fit_smile(chain[chain["expiration"] == front_exp], front_exp, n_T, t0)
    later = []
    for e in exps[1:]:
        n = sessions_between(t0, e, sessions)
        if n > MAX_LATER_SESSIONS:
            break
        try:
            later.append(fit_smile(chain[chain["expiration"] == e], e, n, t0))
        except ValueError:
            continue
    dist_T, V_T = rnd(front), front.atm_w
    diag = {"front_expiry": front_exp.date().isoformat(), "sessions_to_front": n_T, "forward": front.F,
            "front_quotes": front.n_quotes, "later_expiries": len(later)}
    if n_T == 1:
        dist1, path = dist_T, "front expiry is the next session"
    else:
        s2, how = daily_variance(later)
        if s2 is None:
            s2, how = fallback_var, f"historical daily variance ({how})"
        V_1 = max(V_T - (n_T - 1) * s2, 0.25 * V_T)
        dist1 = dist_T.rescale(math.sqrt(V_1 / V_T))
        path = f"converted from a {n_T}-session expiry ({how})"
    diag["path"] = path
    return OptionsResult(dist1.buckets(tau), diag)
