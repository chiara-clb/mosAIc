"""Grading forecasts (design review 3.4).

- Main: log loss against the market's implied probabilities (lower is better). It ranks
  forecasts exactly like KL divergence, which is log loss minus a part that depends only on
  the market's numbers.
- Shown alongside: total variation distance, the percentage points that would have to move
  to match the market (easy to read).
- Secondary: Brier score against the bucket that actually happened (noisy: one draw a month).
- Comparisons between two versions of the agent use confidence intervals resampled by week,
  because forecasts made in the same week tend to be wrong together.

Probabilities are dicts in percent, e.g. {"fall": 36, "flat": 38, "gain": 26}.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import OUTCOMES

EPS = 1e-6


def _frac(p: dict[str, float]) -> np.ndarray:
    v = np.array([float(p[k]) for k in OUTCOMES]) / 100.0
    s = v.sum()
    if not math.isclose(s, 1.0, abs_tol=1e-3):
        raise ValueError(f"probabilities must add up to 100, got {100 * s:.2f}")
    return v / s


def log_loss_implied(forecast: dict, implied: dict) -> float:
    p, q = _frac(forecast), _frac(implied)
    return float(-np.sum(q * np.log(np.clip(p, EPS, 1.0))))


def entropy(probs: dict) -> float:
    q = _frac(probs)
    return float(-np.sum(q * np.log(np.clip(q, EPS, 1.0))))


def kl_divergence(forecast: dict, implied: dict) -> float:
    return log_loss_implied(forecast, implied) - entropy(implied)


def total_variation(forecast: dict, implied: dict) -> float:
    """In percentage points."""
    return float(50.0 * np.sum(np.abs(_frac(forecast) - _frac(implied))))


def brier(forecast: dict, outcome: str) -> float:
    """Multi-class Brier score, from 0 (certain and right) to 2 (certain and wrong)."""
    if outcome not in OUTCOMES:
        raise ValueError(f"unknown outcome {outcome!r}")
    o = np.array([1.0 if k == outcome else 0.0 for k in OUTCOMES])
    return float(np.sum((_frac(forecast) - o) ** 2))


METRICS = {
    "log_loss_implied": lambda r: log_loss_implied(r["forecast"], r["implied"]) if r.get("implied") else None,
    "tvd_implied": lambda r: total_variation(r["forecast"], r["implied"]) if r.get("implied") else None,
    "brier_outcome": lambda r: brier(r["forecast"], r["outcome"]) if r.get("outcome") else None,
}


def score_records(records: list[dict]) -> pd.DataFrame:
    """One row per forecast record with every metric that can be computed.

    Each record needs: id, variant, as_of, forecast; optional implied (dict) and outcome (str).
    """
    rows = []
    for r in records:
        row = {"id": r["id"], "variant": r["variant"], "as_of": pd.Timestamp(r["as_of"])}
        for name, f in METRICS.items():
            row[name] = f(r)
        rows.append(row)
    return pd.DataFrame(rows)


@dataclass
class Comparison:
    metric: str
    n_forecasts: int
    n_weeks: int
    mean_difference: float      # variant_a minus variant_b; negative = a is better (lower)
    ci_low: float
    ci_high: float


def compare(scores: pd.DataFrame, variant_a: str, variant_b: str, metric: str = "log_loss_implied",
            n_boot: int = 2000, seed: int = 0, level: float = 0.95) -> Comparison:
    """Paired difference between two variants on the same forecasts, CI resampled by week."""
    a = scores[scores["variant"] == variant_a].set_index("id")
    b = scores[scores["variant"] == variant_b].set_index("id")
    both = a[[metric, "as_of"]].join(b[[metric]], rsuffix="_b", how="inner").dropna()
    if both.empty:
        raise ValueError("no forecasts scored for both variants")
    both["diff"] = both[metric] - both[f"{metric}_b"]
    both["week"] = both["as_of"].dt.to_period("W")
    weeks = [g["diff"].to_numpy() for _, g in both.groupby("week")]
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(weeks), len(weeks))
        sample = np.concatenate([weeks[i] for i in pick])
        means.append(sample.mean())
    alpha = (1 - level) / 2
    return Comparison(metric=metric, n_forecasts=len(both), n_weeks=len(weeks),
                      mean_difference=float(both["diff"].mean()),
                      ci_low=float(np.quantile(means, alpha)), ci_high=float(np.quantile(means, 1 - alpha)))
