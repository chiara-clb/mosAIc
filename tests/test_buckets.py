import math
from datetime import date

import pandas as pd
import pytest

from mosaic.buckets import (
    bucket_of, horizon_sigma, prices_before, realised_bucket, realised_return, thresholds,
)


def test_sigma_matches_known_daily_volatility(make_px):
    px = make_px(n=600, daily_sigma=0.01, seed=1)
    as_of = px.index[500]
    s = horizon_sigma(px, as_of)
    assert s == pytest.approx(0.01 * math.sqrt(21), rel=0.12)


def test_only_prices_before_as_of_are_used(prices):
    as_of = prices.index[300]
    before = thresholds(prices, as_of)
    shocked = prices.copy()
    shocked.iloc[300:] *= 3.0          # a crash or spike on and after the as-of date
    after = thresholds(shocked, as_of)
    assert before == after
    assert before.start_date == prices.index[299].date()


def test_prices_before_is_strict(prices):
    as_of = prices.index[10]
    assert prices_before(prices, as_of).index[-1] == prices.index[9]


def test_thresholds_are_symmetric_half_sigma(prices):
    th = thresholds(prices, prices.index[300])
    assert th.lower == pytest.approx(-0.5 * th.sigma)
    assert th.upper == pytest.approx(0.5 * th.sigma)
    assert th.lower_price < th.start_price < th.upper_price
    assert th.upper_price == pytest.approx(th.start_price * math.exp(th.upper))


def test_bucket_edges_count_as_flat(prices):
    th = thresholds(prices, prices.index[300])
    assert bucket_of(th.lower, th) == "flat"
    assert bucket_of(th.upper, th) == "flat"
    assert bucket_of(th.lower - 1e-9, th) == "fall"
    assert bucket_of(th.upper + 1e-9, th) == "gain"


def test_realised_return_uses_21st_trading_day(prices):
    i = 300
    r = realised_return(prices, prices.index[i])
    assert r == pytest.approx(math.log(prices.iloc[i + 20] / prices.iloc[i - 1]))


def test_realised_bucket_none_when_future_not_available(prices):
    assert realised_bucket(prices, prices.index[-5]) is None


def test_realised_bucket_detects_a_big_gain(prices):
    i = 300
    px = prices.copy()
    px.iloc[i:] *= 1.5
    assert realised_bucket(px, px.index[i]) == "gain"


def test_too_little_history_raises(make_px):
    px = make_px(n=50)
    with pytest.raises(ValueError):
        thresholds(px, px.index[-1])


def test_accepts_plain_dates_and_timestamps(prices):
    d = prices.index[300]
    assert thresholds(prices, d.date()) == thresholds(prices, pd.Timestamp(d)) == thresholds(prices, d.to_pydatetime())
    assert isinstance(thresholds(prices, d).as_of, date)
