from datetime import datetime, timezone

import numpy as np
import pandas as pd

from mosaic.case_memory import CaseMemory

UTC = timezone.utc


def closes(r, start="2020-01-01"):
    days = pd.bdate_range(start, periods=len(r) + 1)
    return pd.Series(100 * np.exp(np.concatenate([[0], np.cumsum(r)])), index=days)


def world(drop_after_event=True, seed=0):
    """One stock: after every '8-K 2.06' (impairment) the next month falls sharply."""
    rng = np.random.default_rng(seed)
    r = rng.normal(0, 0.01, 1500)
    days = pd.bdate_range("2020-01-01", periods=1501)
    events = []
    for i in range(300, 1400, 63):
        events.append(("8-K 2.06", datetime(days[i].year, days[i].month, days[i].day, 12, tzinfo=UTC)))
        if drop_after_event:
            r[i + 1:i + 21] -= 0.006
    return CaseMemory({"ACME": events}, {"ACME": closes(r)}, market=closes(rng.normal(0, 0.008, 1500))), days


def test_falls_after_an_event_type_raise_the_fall_score():
    cm, days = world()
    as_of = datetime(days[-1].year, days[-1].month, days[-1].day, tzinfo=UTC)
    rates = cm.rates("8-K 2.06", 1, as_of)
    assert rates["fall"] > 0.7 and rates["gain"] < 0.3 and rates["none"] == 0.5 and rates["n"] >= 10


def test_only_windows_ended_before_the_forecast_date_count():
    cm, days = world()
    early = datetime(days[400].year, days[400].month, days[400].day, tzinfo=UTC)
    late = datetime(days[-1].year, days[-1].month, days[-1].day, tzinfo=UTC)
    assert cm.rates("8-K 2.06", 1, early)["n"] < cm.rates("8-K 2.06", 1, late)["n"]


def test_with_no_past_cases_no_effect_reading_beats_no_effect():
    cm, days = world()
    as_of = datetime(days[-1].year, days[-1].month, days[-1].day, tzinfo=UTC)
    rates = cm.rates("8-K 1.05", 1, as_of)
    assert rates["n"] == 0 and rates["none"] == 0.5
    assert all(rates[e] < 0.5 for e in ("fall", "gain", "spread"))


def test_price_reaction_is_measured_against_the_market():
    rng = np.random.default_rng(1)
    market_r = rng.normal(0, 0.005, 400)
    stock_r = market_r.copy()
    stock_r[300:305] += 0.02                                   # company-specific rise after the filing
    cm = CaseMemory({}, {"ACME": closes(stock_r)}, market=closes(market_r))
    day = pd.bdate_range("2020-01-01", periods=401)[300]
    rx = cm.reaction("ACME", datetime(day.year, day.month, day.day, 12, tzinfo=UTC), 7)
    assert rx["bucket"] in ("rise", "big rise") and rx["abnormal_return"] > 0.05

    same = CaseMemory({}, {"ACME": closes(market_r * 2)}, market=closes(market_r * 2))
    rx2 = same.reaction("ACME", datetime(day.year, day.month, day.day, 12, tzinfo=UTC), 7)
    assert rx2["bucket"] == "small"                            # a market-wide move is not a company reaction
