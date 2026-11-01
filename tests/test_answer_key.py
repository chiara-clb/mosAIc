import json
import math
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from mosaic.answer_key import (
    AnswerKey, implied_bucket_probabilities, log_unscorable, target_date, validate_chain,
)
from mosaic.buckets import Thresholds

QD = date(2026, 3, 2)          # quote date: last close before the as-of date
AS_OF = date(2026, 3, 3)
SPOT = 100.0
RATE = 0.04


def mixture_chain(expiry_days, components, strikes=None, spread=0.01, quote_date=QD, spot=SPOT, rate=RATE):
    """Exact option prices for a mixture of lognormals (weight, forward multiplier, vol).

    The forward multipliers are rescaled so the mixture's mean equals the forward price.
    Returns the chain and a function giving the exact P(S_T < K).
    """
    t = expiry_days / 365
    fwd = spot * math.exp(rate * t)
    disc = math.exp(-rate * t)
    w = np.array([c[0] for c in components], float)
    m = np.array([c[1] for c in components], float)
    m = m / np.dot(w, m)
    vols = np.array([c[2] for c in components], float)
    fwds = fwd * m

    def d2(k):
        sd = vols * math.sqrt(t)
        return (np.log(fwds / k) - 0.5 * sd ** 2) / sd

    def call(k):
        sd = vols * math.sqrt(t)
        d1 = d2(k) + sd
        return disc * float(np.dot(w, fwds * norm.cdf(d1) - k * norm.cdf(d1 - sd)))

    def cdf(k):
        return float(np.dot(w, norm.cdf(-d2(k))))

    strikes = strikes if strikes is not None else np.arange(70, 131, 2.5)
    rows = []
    exp = quote_date + timedelta(days=expiry_days)
    for k in strikes:
        c = call(k)
        p = c - disc * (fwd - k)
        for right, px in (("C", c), ("P", p)):
            half = spread * px / 2
            rows.append({"quote_date": quote_date, "expiration": exp, "strike": float(k), "right": right,
                         "bid": px - half, "ask": px + half, "underlying_price": spot})
    return pd.DataFrame(rows), cdf


def thresholds(sigma=0.08):
    return Thresholds(as_of=AS_OF, start_date=QD, start_price=SPOT, sigma=sigma,
                      lower=-0.5 * sigma, upper=0.5 * sigma)


def expected_probs(cdf, th):
    lo, hi = cdf(th.lower_price), cdf(th.upper_price)
    return {"fall": 100 * lo, "flat": 100 * (hi - lo), "gain": 100 * (1 - hi)}


def days_to_target():
    return (target_date(QD) - QD).days


def test_flat_volatility_matches_black_scholes():
    th = thresholds()
    chain, cdf = mixture_chain(days_to_target(), [(1.0, 1.0, 0.25)])
    key = implied_bucket_probabilities(chain, th, rate=RATE)
    assert key.status == "ok"
    for k, v in expected_probs(cdf, th).items():
        assert key.probabilities[k] == pytest.approx(v, abs=0.3)


def test_skewed_distribution_is_recovered():
    # 85%: calm; 15%: a crash scenario. Gives a strong downside skew.
    th = thresholds()
    chain, cdf = mixture_chain(days_to_target(), [(0.85, 1.03, 0.18), (0.15, 0.83, 0.40)])
    key = implied_bucket_probabilities(chain, th, rate=RATE)
    exp = expected_probs(cdf, th)
    assert key.status == "ok"
    for k in exp:
        assert key.probabilities[k] == pytest.approx(exp[k], abs=0.6)
    # Ignoring the skew (one lognormal with the same overall volatility) would be clearly wrong,
    # so the match above shows the smile is being used.
    total_vol = math.sqrt(0.85 * 0.18 ** 2 + 0.15 * 0.40 ** 2)
    _, flat_cdf = mixture_chain(days_to_target(), [(1.0, 1.0, total_vol)])
    naive = expected_probs(flat_cdf, th)
    assert max(abs(naive[k] - exp[k]) for k in exp) > 3


def test_probabilities_sum_to_100():
    chain, _ = mixture_chain(days_to_target(), [(1.0, 1.0, 0.3)])
    key = implied_bucket_probabilities(chain, thresholds(), rate=RATE)
    assert sum(key.probabilities.values()) == pytest.approx(100.0)
    assert all(v >= 0 for v in key.probabilities.values())


def test_interpolates_between_two_expiries():
    th = thresholds()
    n = days_to_target()
    early, _ = mixture_chain(n - 7, [(1.0, 1.0, 0.25)])
    late, _ = mixture_chain(n + 7, [(1.0, 1.0, 0.25)])
    both = implied_bucket_probabilities(pd.concat([early, late]), th, rate=RATE)
    only_early = implied_bucket_probabilities(early, th, rate=RATE)
    only_late = implied_bucket_probabilities(late, th, rate=RATE)
    assert len(both.expiries_used) == 2
    # more time, more spread: the flat bucket shrinks with the expiry
    assert only_late.probabilities["flat"] < both.probabilities["flat"] < only_early.probabilities["flat"]


def test_single_expiry_too_far_is_unscorable():
    chain, _ = mixture_chain(days_to_target() + 30, [(1.0, 1.0, 0.25)])
    key = implied_bucket_probabilities(chain, thresholds(), rate=RATE)
    assert key.status == "unscorable" and "too far" in key.reason


def test_thin_chain_is_unscorable():
    chain, _ = mixture_chain(days_to_target(), [(1.0, 1.0, 0.25)], strikes=np.array([95.0, 100.0, 105.0]))
    key = implied_bucket_probabilities(chain, thresholds(), rate=RATE)
    assert key.status == "unscorable" and "too few" in key.reason


def test_strikes_not_covering_thresholds_is_unscorable():
    chain, _ = mixture_chain(days_to_target(), [(1.0, 1.0, 0.25)], strikes=np.arange(98, 103, 0.5))
    key = implied_bucket_probabilities(chain, thresholds(sigma=0.3), rate=RATE)
    assert key.status == "unscorable" and "cover" in key.reason


def test_wide_spreads_are_not_used():
    chain, _ = mixture_chain(days_to_target(), [(1.0, 1.0, 0.25)], spread=1.2)
    key = implied_bucket_probabilities(chain, thresholds(), rate=RATE)
    assert key.status == "unscorable"


def test_quotes_on_or_after_as_of_are_ignored():
    th = thresholds()
    chain, _ = mixture_chain(days_to_target(), [(1.0, 1.0, 0.25)])
    later, _ = mixture_chain(days_to_target(), [(1.0, 1.0, 0.80)], quote_date=AS_OF)
    clean = implied_bucket_probabilities(chain, th, rate=RATE)
    mixed = implied_bucket_probabilities(pd.concat([chain, later]), th, rate=RATE)
    assert mixed.probabilities == clean.probabilities


def test_no_quotes_before_as_of():
    chain, _ = mixture_chain(days_to_target(), [(1.0, 1.0, 0.25)], quote_date=AS_OF)
    assert implied_bucket_probabilities(chain, thresholds(), rate=RATE).status == "unscorable"


def test_missing_columns_raise():
    with pytest.raises(ValueError):
        validate_chain(pd.DataFrame({"strike": [100]}))


def test_unscorable_dates_are_logged(tmp_path):
    path = tmp_path / "unscorable.jsonl"
    log_unscorable(path, "AAPL", AS_OF, AnswerKey("unscorable", reason="too few usable quotes"))
    rec = json.loads(path.read_text().strip())
    assert rec["ticker"] == "AAPL" and rec["reason"] == "too few usable quotes"
