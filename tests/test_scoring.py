import math

import numpy as np
import pandas as pd
import pytest

from mosaic.scoring import (
    brier, compare, entropy, kl_divergence, log_loss_implied, score_records, total_variation,
)

MARKET = {"fall": 33, "flat": 40, "gain": 27}


def test_log_loss_by_hand():
    f = {"fall": 36, "flat": 38, "gain": 26}
    expected = -(0.33 * math.log(0.36) + 0.40 * math.log(0.38) + 0.27 * math.log(0.26))
    assert log_loss_implied(f, MARKET) == pytest.approx(expected)


def test_matching_the_market_is_best():
    exact = log_loss_implied(MARKET, MARKET)
    assert exact == pytest.approx(entropy(MARKET))
    assert kl_divergence(MARKET, MARKET) == pytest.approx(0.0)
    assert log_loss_implied({"fall": 31, "flat": 38, "gain": 31}, MARKET) > exact


def test_kl_and_log_loss_rank_forecasts_the_same_way():
    fs = [{"fall": 30, "flat": 40, "gain": 30}, {"fall": 36, "flat": 38, "gain": 26}, {"fall": 20, "flat": 60, "gain": 20}]
    by_ll = sorted(range(3), key=lambda i: log_loss_implied(fs[i], MARKET))
    by_kl = sorted(range(3), key=lambda i: kl_divergence(fs[i], MARKET))
    assert by_ll == by_kl


def test_total_variation_in_points():
    assert total_variation({"fall": 36, "flat": 38, "gain": 26}, MARKET) == pytest.approx(3.0)
    assert total_variation(MARKET, MARKET) == 0


def test_brier_by_hand():
    assert brier({"fall": 100, "flat": 0, "gain": 0}, "fall") == 0
    assert brier({"fall": 100, "flat": 0, "gain": 0}, "gain") == 2
    third = 100 / 3
    assert brier({"fall": third, "flat": third, "gain": third}, "flat") == pytest.approx(2 / 3)


def test_zero_probability_does_not_crash():
    assert math.isfinite(log_loss_implied({"fall": 0, "flat": 50, "gain": 50}, MARKET))


def test_probabilities_must_add_to_100():
    with pytest.raises(ValueError):
        total_variation({"fall": 50, "flat": 50, "gain": 50}, MARKET)
    with pytest.raises(ValueError):
        brier(MARKET, "sideways")


def test_score_records_skips_missing_labels():
    df = score_records([
        {"id": "a", "variant": "agent", "as_of": "2026-03-02", "forecast": MARKET, "implied": MARKET, "outcome": "flat"},
        {"id": "b", "variant": "agent", "as_of": "2026-03-09", "forecast": MARKET, "implied": None, "outcome": None},
    ])
    assert df.loc[0, "tvd_implied"] == 0 and df.loc[0, "brier_outcome"] > 0
    assert pd.isna(df.loc[1, "log_loss_implied"]) and pd.isna(df.loc[1, "brier_outcome"])


def _records(n_weeks=20, per_week=5, better_by=0.0, seed=0):
    rng = np.random.default_rng(seed)
    recs = []
    dates = pd.date_range("2026-01-05", periods=n_weeks, freq="W-MON")
    for w, d in enumerate(dates):
        for j in range(per_week):
            implied = {"fall": 30 + rng.uniform(-5, 5), "flat": 0, "gain": 30 + rng.uniform(-5, 5)}
            implied["flat"] = 100 - implied["fall"] - implied["gain"]
            noise = rng.uniform(-6, 6)
            base = {"fall": implied["fall"] + noise, "flat": implied["flat"], "gain": implied["gain"] - noise}
            k = better_by
            agent = {"fall": implied["fall"] + noise * (1 - k), "flat": implied["flat"], "gain": implied["gain"] - noise * (1 - k)}
            fid = f"{w}-{j}"
            recs.append({"id": fid, "variant": "baseline", "as_of": d, "forecast": base, "implied": implied})
            recs.append({"id": fid, "variant": "agent", "as_of": d, "forecast": agent, "implied": implied})
    return recs


def test_compare_identical_variants_gives_zero():
    recs = [dict(r, variant="copy") for r in _records() if r["variant"] == "baseline"] + _records()
    c = compare(score_records(recs), "copy", "baseline")
    assert c.mean_difference == 0 and c.ci_low == 0 and c.ci_high == 0


def test_compare_detects_a_better_agent_and_counts_weeks():
    c = compare(score_records(_records(better_by=0.8)), "agent", "baseline")
    assert c.mean_difference < 0 and c.ci_high < 0
    assert c.n_weeks == 20 and c.n_forecasts == 100


def test_compare_is_reproducible():
    s = score_records(_records(better_by=0.3))
    assert compare(s, "agent", "baseline", seed=1) == compare(s, "agent", "baseline", seed=1)
