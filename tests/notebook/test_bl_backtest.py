"""Dry run of the baseline comparison on synthetic stocks, plus unit tests of the scoring."""

import math

import numpy as np
import pandas as pd
import pytest

from tests.notebook.core import load_baseline_core
from tests.notebook.test_bl_versions import UTC, at, make_stock

ns = load_baseline_core(N_BOOT=200)


def test_scores_by_hand():
    p = {"fall": 30, "flat": 40, "gain": 30}
    assert ns["brier"](p, "flat") == pytest.approx(0.09 + 0.36 + 0.09)
    assert ns["log_loss"](p, "gain") == pytest.approx(-math.log(0.3))
    assert math.isfinite(ns["log_loss"]({"fall": 0, "flat": 50, "gain": 50}, "fall"))


def test_weekly_dates_are_mondays():
    ds = ns["weekly_dates"]("2026-01-01", "2026-02-01")
    assert len(ds) == 4 and all(d.weekday() == 0 for d in ds)


def test_phase_labels():
    assert ns["phase_label"](3) == "1-5" and ns["phase_label"](70) == "64+" and ns["phase_label"](None) == "no results yet"


@pytest.fixture(scope="module")
def backtest():
    stocks = []
    for seed in range(3):
        s, idx, _ = make_stock(n=1600, seed=seed, jump=0.08, calm_after=(25, 0.5))
        s.ticker = f"S{seed}"
        stocks.append(s)
    calm = ns["CalmProfile"](stocks)
    dates = ns["weekly_dates"](str(idx[300].date()), str(idx[-30].date()))
    return ns["run_backtest"](stocks, calm, dates), stocks


def test_backtest_has_every_version_for_every_forecast(backtest):
    bt, _ = backtest
    per = bt.groupby(["ticker", "as_of"])["version"].nunique()
    assert (per == len(ns["VERSIONS"])).all()
    assert len(per) > 150
    total = bt["fall"] + bt["flat"] + bt["gain"]
    assert np.allclose(total, 100)


def test_in_a_calm_after_results_world_b_and_d_beat_a_right_after_results(backtest):
    bt, _ = backtest
    after = bt[bt["phase"] == "1-5"]
    means = after.groupby("version")["brier"].mean()
    assert means["B"] < means["A"]
    assert means["D"] < means["C"]


def test_compare_versions_table(backtest):
    bt, _ = backtest
    t = ns["compare_versions"](bt)
    assert list(t.index) == list(ns["VERSIONS"])
    assert t.loc["Fixed guess", "vs Fixed guess"] == 0
    lo, hi = t.loc["D", "95% interval"]
    assert lo <= t.loc["D", "vs Fixed guess"] <= hi


def test_calibration_table(backtest):
    bt, _ = backtest
    cal = ns["calibration"](bt)
    assert list(cal.columns[:2]) == ["forecasts", "actual % flat"]
    assert set(ns["VERSIONS"]) <= set(cal.columns)
    assert (cal["Fixed guess"] == 38.3).all()
    # the world is calm right after results, and D should expect more calm there than A
    assert cal.loc["1-5", "D"] > cal.loc["1-5", "A"]
