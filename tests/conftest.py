import math

import numpy as np
import pandas as pd
import pytest


def make_prices(n=400, daily_sigma=0.01, start="2025-01-02", seed=0, start_price=100.0):
    """Synthetic adjusted closes on business days with a known daily volatility."""
    rng = np.random.default_rng(seed)
    r = rng.normal(0.0, daily_sigma, n - 1)
    px = start_price * np.exp(np.concatenate([[0.0], np.cumsum(r)]))
    idx = pd.bdate_range(start, periods=n)
    return pd.Series(px, index=idx, name="close")


@pytest.fixture
def prices():
    return make_prices()


@pytest.fixture
def make_px():
    return make_prices


def close_to(a, b, tol):
    return math.isclose(a, b, abs_tol=tol)
