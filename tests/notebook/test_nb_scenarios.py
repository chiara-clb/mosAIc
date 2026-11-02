from datetime import timedelta

import pandas as pd
import pytest

from tests.notebook.core import load_core
from tests.notebook.helpers import dt

ns = load_core(RANDOM_PER_TICKER=4)
DAYS = pd.bdate_range("2024-06-03", "2026-12-31")


def row(form, items, when):
    return {"form": form, "items": items, "acceptanceDateTime": when + "T21:00:00.000Z"}


def quarterly_filings():
    rows = []
    for d in pd.date_range("2024-07-30", "2026-10-30", freq="91D"):
        day = d.strftime("%Y-%m-%d")
        rows.append(row("8-K", "2.02,9.01", day))
        rows.append(row("10-Q", "", (d + timedelta(days=3)).strftime("%Y-%m-%d")))
    rows.append(row("8-K", "5.02", "2025-06-10"))      # management change
    rows.append(row("8-K", "8.01", "2025-06-20"))      # generic: not an event
    rows.append(row("8-K", "1.01,2.02", "2025-07-01"))  # carries 2.02: counted as results, not as an event
    return rows


@pytest.fixture(scope="module")
def plan():
    return ns["scenario_dates"](quarterly_filings(), DAYS, "2025-01-01", "2026-09-25")


def test_every_scenario_is_present(plan):
    assert {sc for _, sc in plan} == set(ns["SCENARIOS"])


def test_dates_are_unique_sorted_midnights_inside_the_window(plan):
    dates = [d for d, _ in plan]
    assert dates == sorted(set(dates))
    assert all(d.hour == 0 and dt("2025-01-01") <= d <= dt("2026-09-25") for d in dates)


def test_event_dates_come_from_event_items_only(plan):
    events = [d for d, sc in plan if sc == "event"]
    assert events == [dt("2025-06-12")]


def test_after_results_is_two_days_after_each_results_filing(plan):
    after = [d for d, sc in plan if sc == "after results"]
    assert all(d - timedelta(days=2) in {dt(r["acceptanceDateTime"][:10]) for r in quarterly_filings()} for d in after)
    assert len(after) >= 6


def test_results_coming_has_next_results_inside_the_month(plan):
    results = ns["results_dates"](quarterly_filings())
    for d, sc in plan:
        if sc != "results coming":
            continue
        nxt = min(r for r in results if r > d)
        trading_after = DAYS[DAYS >= pd.Timestamp(d.date())]
        assert pd.Timestamp(nxt.date()) <= trading_after[20] + timedelta(days=1)


def test_mid_quarter_is_far_from_results(plan):
    results = ns["results_dates"](quarterly_filings())
    for d, sc in plan:
        if sc == "mid-quarter":
            last = max(r for r in results if r < d)
            assert (d - last).days > 30


def test_random_dates_are_reproducible_mondays(plan):
    rnd = [d for d, sc in plan if sc == "random"]
    assert 1 <= len(rnd) <= 4 and all(d.weekday() == 0 for d in rnd)
    again = ns["scenario_dates"](quarterly_filings(), DAYS, "2025-01-01", "2026-09-25")
    assert again == plan


def test_collisions_keep_the_first_scenario_in_priority_order():
    rows = [row("8-K", "2.02", "2025-03-03"), row("10-Q", "", "2025-03-05"), row("8-K", "5.02", "2025-03-03")]
    plan = dict(ns["scenario_dates"](rows, DAYS, "2025-01-01", "2026-09-25"))
    assert plan[dt("2025-03-05")] == "event"


def test_results_by_scenario_table():
    recs = []
    for sc, outcome in (("event", "fall"), ("event", "flat"), ("random", "gain")):
        p = {"fall": 30, "flat": 40, "gain": 30}
        recs.append({"scenario": sc, "outcome": outcome, "prepared": [1, 2],
                     "tot": {"ledger": []}, "variants": {"baseline": p, "ToT beam 3": p}})
    t = ns["results_by_scenario"](recs)
    assert list(t.index) == ["event", "random"]
    assert t.loc["event", "forecasts"] == 2 and t.loc["event", "evidence items"] == 2
    assert t.loc["random", "baseline"] == pytest.approx(ns["brier"]({"fall": 30, "flat": 40, "gain": 30}, "gain"))
