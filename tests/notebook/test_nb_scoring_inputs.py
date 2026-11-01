import math
from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from tests.notebook.core import load_core
from tests.notebook.helpers import UTC, chunk, dt, unit

ns = load_core()


# ---------- ledger ----------

@pytest.mark.parametrize("effect", ["fall", "gain", "spread", "none"])
@pytest.mark.parametrize("size", ["small", "medium", "large"])
def test_every_line_sums_to_zero(effect, size):
    assert sum(ns["shift_for"](effect, size).values()) == pytest.approx(0)


def test_wider_spread_moves_flat_to_both_tails():
    s = ns["shift_for"]("spread", "medium")
    assert s == {"fall": 3, "flat": -6, "gain": 3}


def test_checkpoint_example_line():
    s = ns["shift_for"]("fall", "medium")
    assert (s["fall"], s["flat"], s["gain"]) == pytest.approx((6, -2, -4))


def test_ledger_applies_and_caps():
    base = {"fall": 30, "flat": 40, "gain": 30}
    final, f = ns["apply_ledger"](base, [{"effect": "fall", "size": "medium"}])
    assert final == pytest.approx({"fall": 36, "flat": 38, "gain": 26}) and f == 1
    big = [{"effect": "gain", "size": "large"}] * 3          # +27 on gain -> scaled to +15
    final, f = ns["apply_ledger"](base, big)
    assert f == pytest.approx(15 / 27) and final["gain"] == pytest.approx(45)


def test_ledger_keeps_every_outcome_possible():
    final, _ = ns["apply_ledger"]({"fall": 2, "flat": 49, "gain": 49}, [{"effect": "gain", "size": "large"}])
    assert min(final.values()) >= 0.99 and sum(final.values()) == pytest.approx(100, abs=0.02)


def test_floor_holds_after_rescaling():
    out = ns["floor_and_normalise"]({"fall": 0.0, "flat": 30.0, "gain": 90.0})
    assert out["fall"] == 1.0 and sum(out.values()) == pytest.approx(100)


def test_total_shift_orders_ledgers():
    ts = ns["total_shift"]
    assert ts([]) == 0 == ts([{"effect": "none", "size": "small"}])
    assert ts([{"effect": "fall", "size": "small"}]) < ts([{"effect": "fall", "size": "large"}])


# ---------- case memory ----------

def synthetic_market(effect, n_days=1600, every=60, first=400, seed=0):
    """Prices with an engineered move after each event, and the event list."""
    rng = np.random.default_rng(seed)
    logp = np.cumsum(rng.normal(0, 0.008, n_days))
    idx = pd.bdate_range("2019-01-01", periods=n_days)
    events = []
    for e in range(first, n_days - 40, every):
        if effect == "fall":                      # gradual 25% fall over the 21 days after the event
            for k in range(1, 22):
                logp[e + k:] -= 0.0125
        elif effect == "jump":                    # one-day 30% jump right after the event, then nothing
            logp[e + 1:] += math.log(1.3)
        events.append(("8-K 5.02", idx[e].to_pydatetime().replace(tzinfo=UTC, hour=21)))
    return pd.Series(100 * np.exp(logp), index=idx), events, idx


def case_for(effect, **kw):
    px, events, idx = synthetic_market(effect, **kw)
    return ns["CaseMemory"]({"XYZ": events}, {"XYZ": px}), events, idx


def test_history_supports_the_right_direction():
    case, events, idx = case_for("fall")
    as_of = idx[-1].to_pydatetime().replace(tzinfo=UTC)
    fall, d = case.score("fall", "medium", "8-K 5.02", 0, as_of)
    gain, _ = case.score("gain", "medium", "8-K 5.02", 0, as_of)
    none, _ = case.score("none", "medium", "8-K 5.02", 0, as_of)
    assert d["n_events"] > 15
    assert fall > 0.6 > 0.5 == none > gain


def test_size_is_checked_too():
    case, _, idx = case_for("fall")
    as_of = idx[-1].to_pydatetime().replace(tzinfo=UTC)
    small, _ = case.score("fall", "small", "8-K 5.02", 0, as_of)
    large, _ = case.score("fall", "large", "8-K 5.02", 0, as_of)
    assert large > small


def test_unknown_event_type_favours_no_effect():
    case, _, idx = case_for("fall")
    as_of = idx[-1].to_pydatetime().replace(tzinfo=UTC)
    s, d = case.score("fall", "medium", "8-K 1.05", 0, as_of)
    assert d["n_events"] == 0 and s < 0.5


def test_small_samples_say_less():
    case, events, idx = case_for("fall")
    few_as_of = (events[1][1] + timedelta(days=60))
    many_as_of = idx[-1].to_pydatetime().replace(tzinfo=UTC)
    few, d_few = case.score("fall", "medium", "8-K 5.02", 0, few_as_of)
    many, d_many = case.score("fall", "medium", "8-K 5.02", 0, many_as_of)
    assert d_few["n_events"] < d_many["n_events"]
    assert abs(few - 0.5) < abs(many - 0.5)


def test_only_windows_that_ended_before_the_forecast_date():
    case, events, idx = case_for("fall")
    e = events[5][1]
    just_after = e + timedelta(days=3)            # event known, outcome not yet
    _, n_before, _ = case.distribution("8-K 5.02", 0, just_after)
    _, n_later, _ = case.distribution("8-K 5.02", 0, e + timedelta(days=40))
    assert n_later == n_before + 1


def test_timing_lag_treats_old_jumps_as_priced_in():
    case, _, idx = case_for("jump")
    as_of = idx[-1].to_pydatetime().replace(tzinfo=UTC)
    q0, _, _ = case.distribution("8-K 5.02", 0, as_of)
    q5, _, _ = case.distribution("8-K 5.02", 5, as_of)
    p0 = case.overall(as_of)
    assert q0["gain"] > p0["gain"] + 0.25          # jump inside the window: gains
    assert q5["gain"] < q0["gain"] - 0.4           # jump already priced in:
    assert q5["flat"] > q0["flat"] + 0.4           # the month after looks flat


def test_overall_frequencies_look_like_a_normal_prior():
    case, _, idx = case_for("none")
    p0 = case.overall(idx[-1].to_pydatetime().replace(tzinfo=UTC))
    assert sum(p0.values()) == pytest.approx(1)
    assert 0.25 < p0["flat"] < 0.55


# ---------- novelty ----------

def nov_index(items, vecs):
    idx = ns["Index"](items, np.array(vecs, dtype=np.float32))
    return idx


def item_at(idx, i):
    return dict(idx.chunks[i], row=i)


def test_repeated_wording_scores_low_and_new_wording_high():
    prev = chunk("p", filing="Q1", accepted="2026-05-01", section="Item 1. Legal Proceedings")
    same = chunk("s", filing="Q2", accepted="2026-07-31", section="Item 1. Legal Proceedings")
    idx = nov_index([prev, same], [unit(1, 0), unit(1, 0)])
    for ch in idx.chunks:
        ch["section_key"] = ns["section_key"](ch["section"])
    nov, p = ns["novelty"](idx, item_at(idx, 1))
    assert nov == pytest.approx(0, abs=1e-6) and p == "Q1"
    idx.emb[1] = unit(0, 1)
    assert ns["novelty"](idx, item_at(idx, 1))[0] == pytest.approx(1)


def test_10k_counts_as_previous_report_for_10q():
    k = chunk("k", filing="K1", form="10-K", accepted="2026-02-01", section="Item 3. Legal Proceedings")
    q = chunk("q", filing="Q1", form="10-Q", accepted="2026-05-01", section="Item 1. Legal Proceedings")
    idx = nov_index([k, q], [unit(1, 0), unit(1, 0.1)])
    for ch in idx.chunks:
        ch["section_key"] = ns["section_key"](ch["section"])
    nov, p = ns["novelty"](idx, item_at(idx, 1))
    assert p == "K1" and nov < 0.01


def test_only_the_latest_earlier_filing_same_section_counts():
    older = chunk("o", filing="Q0", accepted="2026-02-01", section="Item 1A. Risk Factors")
    other = chunk("x", filing="Q1", accepted="2026-05-01", section="Item 2. Management's Discussion")
    prev = chunk("p", filing="Q1", accepted="2026-05-01", section="Item 1A. Risk Factors")
    later = chunk("l", filing="Q3", accepted="2026-10-01", section="Item 1A. Risk Factors")
    cur = chunk("c", filing="Q2", accepted="2026-07-31", section="Item 1A. Risk Factors")
    idx = nov_index([older, other, prev, later, cur], [unit(0, 1), unit(1, 0), unit(0.6, 0.8), unit(1, 0), unit(1, 0)])
    for ch in idx.chunks:
        ch["section_key"] = ns["section_key"](ch["section"])
    nov, p = ns["novelty"](idx, item_at(idx, 4))
    assert p == "Q1" and nov == pytest.approx(0.4, abs=1e-5)


def test_8k_compares_with_same_item_only():
    results = chunk("r", filing="E1", form="8-K", event="8-K 2.02", accepted="2026-05-01", section="Press release")
    mgmt = chunk("m", filing="E2", form="8-K", event="8-K 5.02", accepted="2026-06-01", section="Press release")
    cur = chunk("c", filing="E3", form="8-K", event="8-K 2.02", accepted="2026-07-30", section="Press release")
    idx = nov_index([results, mgmt, cur], [unit(1, 0), unit(1, 0), unit(0.8, 0.6)])
    for ch in idx.chunks:
        ch["section_key"] = ns["section_key"](ch["section"])
    nov, p = ns["novelty"](idx, item_at(idx, 2))
    assert p == "E1" and nov == pytest.approx(0.2, abs=1e-5)


def test_no_previous_filing_counts_as_new():
    idx = nov_index([chunk("c")], [unit(1, 0)])
    assert ns["novelty"](idx, item_at(idx, 0)) == (1.0, None)


def test_no_effect_reading_gets_the_opposite_novelty():
    assert ns["reading_novelty"]("none", 0.1) == pytest.approx(0.9)
    assert ns["reading_novelty"]("fall", 0.1) == pytest.approx(0.1)
