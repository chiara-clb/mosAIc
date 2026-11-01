from datetime import timedelta

import numpy as np
import pytest

from tests.notebook.core import load_baseline_core
from tests.notebook.test_bl_versions import UTC, at, make_stock

ns = load_baseline_core()


@pytest.fixture(scope="module")
def calm_world():
    """Three stocks whose first 25 days after each results jump move at half their usual size."""
    stocks = []
    for seed in range(3):
        s, idx, _ = make_stock(n=2000, seed=seed, jump=0.08, calm_after=(25, 0.5))
        stocks.append(s)
    return stocks, idx, ns["CalmProfile"](stocks)


def test_months_right_after_results_are_calmer(calm_world):
    stocks, idx, calm = calm_world
    end = at(idx, len(idx) - 1)
    early, late = calm.factor(2, end), calm.factor(50, end)
    assert early < 0.8
    assert early < late


def test_no_history_means_no_correction(calm_world):
    stocks, idx, calm = calm_world
    assert calm.factor(2, at(idx, 300)) == 1.0       # no window has ended yet
    assert calm.factor(None, at(idx, 1900)) == 1.0


def test_only_windows_ended_before_the_forecast_date(calm_world):
    stocks, idx, calm = calm_world
    ends, _, _ = calm.tables[calm.bin_of(2)]
    as_of = at(idx, 1500)
    n = int(np.searchsorted(ends, np.datetime64(as_of.date(), "D")))
    assert n > 0 and all(e < np.datetime64(as_of.date(), "D") for e in ends[:n])
    assert all(e >= np.datetime64(as_of.date(), "D") for e in ends[n:])


def test_bins():
    assert ns["CalmProfile"].bin_of(1) == 0 and ns["CalmProfile"].bin_of(5) == 0
    assert ns["CalmProfile"].bin_of(6) == 1 and ns["CalmProfile"].bin_of(200) == 5


def test_version_d_expects_a_calmer_month_after_results(calm_world):
    stocks, idx, calm = calm_world
    s = stocks[0]
    e = [t for t in s.reaction_pos if t > 1700][0]
    out = ns["baselines"](s, at(idx, e + 2), calm=calm)
    assert out["D"]["flat"] > out["C"]["flat"] + 3
    assert sum(out["D"].values()) == pytest.approx(100)


def test_summary_table(calm_world):
    stocks, idx, calm = calm_world
    t = calm.summary(at(idx, 1999))
    assert list(t.columns) == ["days since results", "past windows", "factor"]
    assert len(t) == len(ns["CALM_BINS"]) and t["past windows"].iloc[0] > 50
