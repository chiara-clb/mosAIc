from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from mosaic.baseline import (FIXED_GUESS, baseline, main_results, results_dates, results_day_mask,
                             too_few_results)

UTC = timezone.utc
AS_OF = datetime(2025, 6, 2, tzinfo=UTC)


def series(n=400, sigma=0.01, jumps=(), seed=0):
    rng = np.random.default_rng(seed)
    dates, d = [], AS_OF - timedelta(days=1)
    while len(dates) < n:
        if d.weekday() < 5:
            dates.append(d)
        d -= timedelta(days=1)
    dates = dates[::-1]
    r = rng.normal(0, sigma, n)
    for day_index, size in jumps:
        r[day_index] = size
    return dates, list(100 * np.exp(np.cumsum(r)))


def test_steady_volatility_gives_about_31_38_31():
    dates, closes = series()
    p, info = baseline(dates, closes, AS_OF, [], False)
    assert p["fall"] == pytest.approx(30.85, abs=2.5) and p["gain"] == pytest.approx(30.85, abs=2.5)
    assert abs(sum(p.values()) - 100) < 0.05 and info["version"] == "C"


def test_a_fresh_results_day_jump_does_not_inflate_version_c():
    dates, closes = series(jumps=[(390, 0.12)])
    results = [dates[390].replace(hour=12)]
    c, _ = baseline(dates, closes, AS_OF, results, False)
    a, _ = baseline(dates, closes, AS_OF, results, False, version="A")
    assert c["flat"] > a["flat"] + 3                          # A expects a jumpy month, C does not


def test_results_in_the_window_widen_both_tails():
    dates, closes = series(jumps=[(300, 0.06), (237, -0.05)])
    results = [dates[300].replace(hour=12), dates[237].replace(hour=12)]
    calm, _ = baseline(dates, closes, AS_OF, results, False)
    due, info = baseline(dates, closes, AS_OF, results, True)
    assert due["flat"] < calm["flat"] and "results_move" in info


def test_only_prices_before_the_forecast_date_are_used():
    dates, closes = series()
    later = dates + [AS_OF + timedelta(days=1)]
    p1, _ = baseline(dates, closes, AS_OF, [], False)
    p2, _ = baseline(later, closes + [closes[-1] * 3], AS_OF, [], False)
    assert p1 == p2


def test_too_little_history_falls_back_to_the_fixed_guess():
    dates, closes = series(n=50)
    p, info = baseline(dates, closes, AS_OF, [], False)
    assert p == FIXED_GUESS and "fallback" in info


def test_after_close_filings_move_the_next_day():
    dates, _ = series(n=10)
    mid_day = dates[5].replace(hour=15)
    after_close = dates[5].replace(hour=21)
    assert results_day_mask(dates, [mid_day]).nonzero()[0].tolist() == [5]
    assert results_day_mask(dates, [after_close]).nonzero()[0].tolist() == [6]


def test_one_results_announcement_per_quarter():
    q_report = datetime(2025, 5, 2, tzinfo=UTC)
    r202 = [datetime(2025, 3, 10, tzinfo=UTC), datetime(2025, 4, 30, tzinfo=UTC), datetime(2025, 6, 20, tzinfo=UTC)]
    assert main_results(r202, [q_report]) == [datetime(2025, 4, 30, tzinfo=UTC), datetime(2025, 6, 20, tzinfo=UTC)]


def test_results_dates_from_an_edgar_list_and_the_too_few_check():
    rows = [{"form": "8-K", "items": "2.02,9.01", "acceptanceDateTime": "2025-04-29T20:10:00.000Z"},
            {"form": "8-K", "items": "5.02", "acceptanceDateTime": "2025-05-28T21:05:00.000Z"},
            {"form": "10-Q", "items": "", "acceptanceDateTime": "2025-05-09T20:30:00.000Z"}]
    got = results_dates(rows)
    assert got == [datetime(2025, 4, 29, 20, 10, tzinfo=UTC)]
    assert too_few_results(got, AS_OF)
