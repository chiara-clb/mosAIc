import pandas as pd

from mosaic.prices import load_closes, price_table

CALLS = []


def fake_fetch(ticker, start, end):
    CALLS.append(ticker)
    days = pd.bdate_range("2025-01-01", "2025-01-10")
    if ticker == "SPY":
        days = days.drop(days[2])                         # a day the fund is missing
    return pd.Series(range(1, len(days) + 1), index=days, dtype=float)


def test_price_table_aligns_the_stock_and_the_fund(tmp_path):
    t = price_table("ACME", "2025-01-01", "2025-01-11", fake_fetch, tmp_path)
    assert len(t["dates"]) == len(t["ACME"]) == len(t["SPY"]) == 7
    assert all(d.tzinfo is not None for d in t["dates"])


def test_prices_are_cached(tmp_path):
    CALLS.clear()
    a = load_closes("ACME", "2025-01-01", "2025-01-11", fake_fetch, tmp_path)
    b = load_closes("ACME", "2025-01-01", "2025-01-11", fake_fetch, tmp_path)
    assert CALLS == ["ACME"] and list(a) == list(b)
