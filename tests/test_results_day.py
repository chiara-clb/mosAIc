"""Results-day baselines on synthetic data with a known answer."""

import math
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from mosaic.options_rnd import black, next_day_distribution
from mosaic.results_day import buckets_normal, reaction_session, results_day, rows

UTC = timezone.utc
T0 = date(2026, 7, 30)


def after_close(d: date) -> datetime:
    return datetime(d.year, d.month, d.day, 21, 30, tzinfo=UTC)         # after the 16:00 close, summer or winter


def synthetic_prices(release_days: list[date], jump: float = 0.08, daily: float = 0.012, seed: int = 0):
    """Calm daily moves, plus a move of +/- jump on the session after each release."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2023-06-01", "2026-08-14")
    r = rng.normal(0, daily, len(idx))
    for k, d in enumerate(release_days):
        nxt = idx[idx > pd.Timestamp(d)][0]
        r[idx.get_loc(nxt)] = jump if k % 2 else -jump
    return pd.Series(100 * np.exp(np.cumsum(r)), index=idx)


PAST = [T0 - timedelta(days=91 * k) for k in range(9, 0, -1)]            # nine earlier releases


def chain_for(vol_1d: float, t0: date, expiries: list[date], spot: float):
    """Quotes at the t0 close whose front expiry carries exactly a 1-day volatility of vol_1d."""
    rows_ = []
    for e in expiries:
        s = vol_1d * math.sqrt(1)                                        # total vol, the same for each strike
        for k in np.arange(0.70, 1.31, 0.01):
            K = round(spot * k, 2)
            for right, call in (("C", True), ("P", False)):
                mid = float(black(spot, K, s, 1.0, call))
                if mid > 0.02:
                    rows_.append({"quote_date": t0, "strike": K, "expiration": e, "right": right,
                                  "bid": mid * 0.995, "ask": mid * 1.005})
    return pd.DataFrame(rows_)


def test_after_close_releases_react_the_next_session():
    sessions = pd.bdate_range("2026-07-27", "2026-08-07")
    assert reaction_session(after_close(T0), sessions) == pd.Timestamp("2026-07-31")
    morning = datetime(2026, 7, 30, 11, 0, tzinfo=UTC)                   # 07:00 New York: same session
    assert reaction_session(morning, sessions) == pd.Timestamp("2026-07-30")


def test_baselines_one_and_two_use_daily_and_results_day_volatility():
    closes = synthetic_prices(PAST + [T0])
    rd = results_day("AMZN", after_close(T0), closes, [after_close(d) for d in PAST], chain=None)
    assert len(rd.past_moves) == 8
    assert rd.J == pytest.approx(0.08, abs=0.002)                       # the planted results-day jumps
    assert rd.tau == pytest.approx(0.5 * rd.J)
    b2 = rd.baselines["B2"]
    assert b2["flat"] == pytest.approx(100 * (2 * stats.norm.cdf(0.5) - 1), abs=0.2)     # about 38%
    b1 = rd.baselines["B1"]
    assert b1["flat"] > 90                                               # ordinary days: tau is many sigmas
    assert rd.baselines["B3"] is None and any("no quotes" in n for n in rd.notes)
    assert rd.actual["outcome"] in ("fall", "gain")                     # the planted jump on t1


def test_baseline_three_reads_the_options_at_the_close():
    closes = synthetic_prices(PAST + [T0])
    spot = float(closes.loc[pd.Timestamp(T0)])
    chain = chain_for(0.06, T0, [date(2026, 7, 31)], spot)               # front expiry is the next session
    rd = results_day("AMZN", after_close(T0), closes, [after_close(d) for d in PAST], chain)
    want = buckets_normal(0.06, rd.tau) * 100
    got = rd.baselines["B3"]
    assert got["fall"] == pytest.approx(want[0], abs=2.0)                # lognormal skew: within 2 points
    assert got["flat"] == pytest.approx(want[1], abs=2.0)
    assert any("next session" in n for n in rd.notes)


def test_a_later_front_expiry_is_converted_to_one_day():
    t0, t1 = pd.Timestamp("2026-07-30"), pd.Timestamp("2026-07-31")
    sessions = pd.bdate_range("2026-07-01", "2026-09-30")
    spot, event, daily = 100.0, 0.06, 0.015
    rows_ = []
    for e in (date(2026, 8, 7), date(2026, 8, 14), date(2026, 8, 21)):      # 6, 11, 16 sessions away
        n = int(((sessions > t0) & (sessions <= pd.Timestamp(e))).sum())
        s = math.sqrt(event ** 2 + (n - 1) * daily ** 2)                 # the event day plus ordinary days
        for k in np.arange(0.70, 1.31, 0.01):
            K = round(spot * k, 2)
            for right, call in (("C", True), ("P", False)):
                mid = float(black(spot, K, s, 1.0, call))
                if mid > 0.02:
                    rows_.append({"strike": K, "expiration": e, "right": right, "bid": mid * 0.995, "ask": mid * 1.005})
    res = next_day_distribution(pd.DataFrame(rows_), t0, t1, 0.03, sessions, fallback_var=daily ** 2)
    want = buckets_normal(event, 0.03)
    assert res.probs[1] == pytest.approx(want[1], abs=0.03)              # the event's own spread is recovered
    assert "converted" in res.diagnostics["path"]


def test_a_release_before_the_close_is_refused():
    closes = synthetic_prices(PAST + [T0])
    morning = datetime(2026, 7, 30, 11, 0, tzinfo=UTC)
    with pytest.raises(ValueError, match="before the close"):
        results_day("AMZN", morning, closes, [after_close(d) for d in PAST], None)


def test_rows_show_the_chance_each_baseline_gave_to_what_happened():
    closes = synthetic_prices(PAST + [T0])
    rd = results_day("AMZN", after_close(T0), closes, [after_close(d) for d in PAST], None)
    table = rows(rd)
    assert [r["Baseline"][:2] for r in table] == ["B1", "B2", "B3"]
    assert table[1]["Chance given to what happened %"] == rd.baselines["B2"][rd.actual["outcome"]]


def test_only_after_close_releases_can_be_chosen_newest_first():
    from mosaic.results_day import release_choices
    releases = [after_close(date(2026, 4, 29)), datetime(2026, 7, 30, 12, 0, tzinfo=UTC),    # 08:00 New York
                after_close(date(2026, 7, 30)), after_close(date(2022, 7, 28))]
    choices = release_choices(releases, option_quarters={(2026, 3)}, since=date(2023, 1, 1))
    assert [c["day"] for c in choices] == [date(2026, 7, 30), date(2026, 4, 29)]
    assert choices[0]["options"] and not choices[1]["options"]
    assert choices[0]["label"] == "2026-07-30 · 17:30 New York · options available"
