import math
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from tests.notebook.core import load_baseline_core

ns = load_baseline_core()
UTC = timezone.utc


def make_stock(n=1200, daily=0.01, jump=0.08, every=63, first=100, seed=0, hour=21, calm_after=None):
    """Synthetic stock: daily moves of size `daily`, a results jump every `every` days.

    Results are filed at `hour` UTC on day e; with hour >= 20 the jump lands on day e+1.
    calm_after: optional (days, factor) making the first `days` after each jump calmer.
    """
    rng = np.random.default_rng(seed)
    r = rng.normal(0, daily, n)
    idx = pd.bdate_range("2015-01-01", periods=n)
    results = []
    for e in range(first, n - 2, every):
        land = e + 1 if hour >= 20 else e
        r[land] = jump * (1 if rng.uniform() > 0.5 else -1)
        if calm_after:
            days, f = calm_after
            r[land + 1:land + 1 + days] *= f
        results.append(idx[e].to_pydatetime().replace(tzinfo=UTC, hour=hour))
    px = pd.Series(100 * np.exp(np.cumsum(r)), index=idx)
    return ns["Stock"]("XYZ", px, results), idx, results


def at(idx, k):
    return idx[k].to_pydatetime().replace(tzinfo=UTC)


def test_results_after_close_move_the_next_day():
    s, idx, res = make_stock(hour=21)
    assert s.reaction_pos[0] == 101
    s2, _, _ = make_stock(hour=14)
    assert s2.reaction_pos[0] == 100


def test_position_and_known_prices():
    s, idx, _ = make_stock()
    assert s.position(at(idx, 500)) == 500
    assert s.can_forecast(500) and not s.can_forecast(100) and not s.can_forecast(len(idx) - 5)


def test_year_without_results_days_ignores_the_jumps():
    s, idx, _ = make_stock(daily=0.01, jump=0.2)
    i = 800
    assert math.sqrt(s.year_ex_var(i)) == pytest.approx(0.01, rel=0.15)
    assert s.sigma_year(i) / math.sqrt(21) > 0.012          # thresholds keep the jumps


def test_a_is_fooled_by_the_jump_for_weeks_and_b_is_not():
    s, idx, _ = make_stock(jump=0.12)
    e = s.reaction_pos[8]
    gaps = [baselines_at(s, idx, e + k)["B"]["flat"] - baselines_at(s, idx, e + k)["A"]["flat"] for k in (2, 20, 40, 60)]
    assert gaps[0] > 5                                   # A fooled by the fresh jump
    assert gaps[0] > gaps[1] > gaps[2] > gaps[3] > 0     # and the effect fades only slowly


def baselines_at(s, idx, k):
    return ns["baselines"](s, at(idx, k))


def test_blend_pulls_toward_the_past_year():
    s, idx, _ = make_stock(daily=0.01, jump=0.0)
    i = 900
    # make the last month unusually calm
    s.ret[i - 25:i] *= 0.2
    s.cum_sq = np.concatenate([[0.0], np.cumsum(np.where(~s.reaction & ~np.isnan(s.ret), s.ret, 0.0) ** 2)])
    v = ns["core_vars"](s, i)
    assert v["B"] < v["C"] < s.year_ex_var(i) * 21


def test_results_jump_added_only_when_results_are_due():
    s, idx, res = make_stock(jump=0.1)
    due = res[8] + timedelta(days=80)           # next results about 11 days later
    not_due = res[8] + timedelta(days=10)
    assert s.earnings_in_window(due) and not s.earnings_in_window(not_due)
    b_due, b_not = ns["baselines"](s, due), ns["baselines"](s, not_due)
    assert b_due["B"]["flat"] < b_not["B"]["flat"]
    assert s.typical_move(s.position(due)) == pytest.approx(0.1)


def test_probabilities_add_to_100_and_fixed_guess_is_fixed():
    s, idx, _ = make_stock()
    out = baselines_at(s, idx, 700)
    assert set(out) == {"A", "B", "C", "Fixed guess"}
    for p in out.values():
        assert sum(p.values()) == pytest.approx(100)
    assert out["Fixed guess"] == ns["FIXED_GUESS"]


def test_no_look_ahead():
    s, idx, _ = make_stock()
    before = baselines_at(s, idx, 700)
    s.ret[700:] *= 10
    s.logp[700:] += 3
    assert baselines_at(s, idx, 700) == before


def test_realised_bucket():
    s, idx, _ = make_stock(jump=0.0)
    s.logp[700:] += 0.5
    assert s.realised_bucket(700) == "gain"
