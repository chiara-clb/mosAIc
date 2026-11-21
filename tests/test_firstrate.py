"""The FirstRate loader, tested on synthetic files in FirstRate's format (the real data is licensed)."""

import ast
import math
import zipfile
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from mosaic import firstrate as fr
from mosaic.answer_key import bs_call, bs_put
from mosaic.buckets import thresholds

SPOT, VOL = 100.0, 0.30


def firstrate_lines(quote_days, spot=SPOT, vol=VOL, expiry_days=(10, 31, 59, 94), strikes=None):
    """Rows in FirstRate's layout: date, strike, expiry, c/p, last, bid, ask, then 9 more columns."""
    strikes = strikes or [float(k) for k in range(60, 145, 5)]
    lines = []
    for qd in quote_days:
        for dte in expiry_days:
            exp = qd + timedelta(days=dte)
            t = dte / 365
            for right, f in (("c", bs_call), ("p", bs_put)):
                for k in strikes:
                    mid = f(spot, k, t, vol, 1.0)
                    bid, ask = 0.99 * mid, 1.01 * mid + 0.001
                    lines.append(f"{qd},{k:g},{exp},{right},{mid:.2f},{bid:.4f},{ask:.4f},"
                                 f"0.29,0.31,100,10,0.5,0.01,0.1,-0.01,0.01")
    return lines


def write(path: Path, lines):
    path.write_text("\n".join(lines) + "\n")
    return path


def prices_until(last_day: date, spot=SPOT, n=300):
    idx = pd.bdate_range(end=pd.Timestamp(last_day), periods=n)
    rng = np.random.default_rng(1)
    walk = np.exp(rng.normal(0, VOL / math.sqrt(252), n).cumsum())
    return pd.Series(spot * walk / walk[-1], index=idx)


QD = date(2026, 3, 4)


def test_reads_the_sixteen_columns(tmp_path):
    raw = fr.read_firstrate(write(tmp_path / "AAPL_2026_q1_option_chain.txt", firstrate_lines([QD])))
    assert list(raw.columns) == fr.COLUMNS
    assert set(raw["right"]) == {"C", "P"}
    assert raw["quote_date"].iloc[0] == QD
    assert isinstance(raw["expiration"].iloc[0], date)


def test_reads_a_zipped_file(tmp_path):
    z = tmp_path / "AAPL_2026_q1_option_chain.zip"
    with zipfile.ZipFile(z, "w") as f:
        f.writestr("AAPL_2026_q1_option_chain.txt", "\n".join(firstrate_lines([QD])))
    assert len(fr.read_firstrate(z)) == len(firstrate_lines([QD]))


def test_refuses_a_file_that_is_not_an_option_chain(tmp_path):
    bad = write(tmp_path / "x.txt", ["2026-03-04,100,2026-04-17,x,1,1,1,0,0,0,0,0,0,0,0,0"])
    with pytest.raises(ValueError, match="neither calls nor puts"):
        fr.read_firstrate(bad)


def test_stock_price_comes_back_from_put_call_parity(tmp_path):
    raw = fr.read_firstrate(write(tmp_path / "AAPL_2026_q1_option_chain.txt", firstrate_lines([QD])))
    assert fr.implied_spot(raw) == pytest.approx(SPOT, abs=0.05)


def test_parity_ignores_strikes_with_no_bid():
    raw = pd.DataFrame({
        "quote_date": [QD] * 4, "expiration": [QD + timedelta(days=30)] * 4,
        "right": ["C", "P", "C", "P"], "strike": [100.0, 100.0, 50.0, 50.0],
        "bid": [4.0, 3.9, 0.0, 0.0], "ask": [4.2, 4.1, 99.0, 0.1],
    })
    assert fr.parity_forward(raw) == pytest.approx(100.1)


def test_closes_win_over_parity_when_given(tmp_path):
    raw = fr.read_firstrate(write(tmp_path / "AAPL_2026_q1_option_chain.txt", firstrate_lines([QD])))
    chain = fr.to_chain(raw, closes=pd.Series({pd.Timestamp(QD): 101.5}))
    assert set(chain["underlying_price"]) == {101.5}
    assert set(fr.REQUIRED_COLUMNS) <= set(chain.columns)


def test_finds_files_by_stock_and_quarter(tmp_path):
    write(tmp_path / "AAPL_2026_q1_option_chain.txt", ["x"])
    write(tmp_path / "AAPL_2026_q2_option_chain.txt", ["x"])
    write(tmp_path / "MSFT_2026_q1_option_chain.txt", ["x"])
    assert set(fr.find_files(tmp_path, "aapl")) == {(2026, 1), (2026, 2)}


def test_finds_and_reads_a_stock_inside_a_zip_of_many_stocks(tmp_path):
    days = [date(2026, 7, 13), date(2026, 7, 14)]
    big = tmp_path / "2026_q3_option_chain_cr7mic.zip"
    with zipfile.ZipFile(big, "w") as z:
        z.writestr("2026_q3/AAPL_2026_q3_option_chain.txt", "\n".join(firstrate_lines(days)))
        z.writestr("2026_q3/MSFT_2026_q3_option_chain.txt", "\n".join(firstrate_lines(days, spot=400)))
    files = fr.find_files(tmp_path, "AAPL")
    assert files == {(2026, 3): (big, "2026_q3/AAPL_2026_q3_option_chain.txt")}
    chain = fr.chain_before(tmp_path, "AAPL", date(2026, 7, 15))
    assert sorted(set(chain["quote_date"])) == days
    assert chain["underlying_price"].iloc[0] == pytest.approx(SPOT, abs=0.05)


def test_reads_a_zipped_stock_inside_a_zip_of_many_stocks(tmp_path):
    inner = tmp_path / "inner.zip"
    with zipfile.ZipFile(inner, "w") as z:
        z.writestr("AAPL_2026_q3_option_chain.txt", "\n".join(firstrate_lines([date(2026, 7, 14)])))
    big = tmp_path / "all" / "2026_q3_option_chain.zip"
    big.parent.mkdir()
    with zipfile.ZipFile(big, "w") as z:
        z.write(inner, "AAPL_2026_q3_option_chain.zip")
    inner.unlink()
    files = fr.find_files(tmp_path, "AAPL")
    assert files[(2026, 3)] == (big, "AAPL_2026_q3_option_chain.zip")
    assert len(fr.read_firstrate(*files[(2026, 3)])) == len(firstrate_lines([date(2026, 7, 14)]))


def test_an_unzipped_copy_wins_over_one_inside_a_zip(tmp_path):
    with zipfile.ZipFile(tmp_path / "2026_q3_all.zip", "w") as z:
        z.writestr("AAPL_2026_q3_option_chain.txt", "x")
    plain = write(tmp_path / "AAPL_2026_q3_option_chain.txt", ["x"])
    assert fr.find_files(tmp_path, "AAPL")[(2026, 3)] == (plain, None)


def test_chain_before_reads_across_a_quarter_boundary(tmp_path):
    q1_days = [date(2026, 3, 30), date(2026, 3, 31)]
    write(tmp_path / "AAPL_2026_q1_option_chain.txt", firstrate_lines(q1_days))
    write(tmp_path / "AAPL_2026_q2_option_chain.txt", firstrate_lines([date(2026, 4, 1), date(2026, 4, 2)]))
    chain = fr.chain_before(tmp_path, "AAPL", date(2026, 4, 2))
    assert sorted(set(chain["quote_date"])) == q1_days + [date(2026, 4, 1)]     # nothing on or after the date


def test_answer_key_matches_the_textbook_answer_for_a_flat_smile(tmp_path):
    days = [QD - timedelta(days=1), QD]
    write(tmp_path / "AAPL_2026_q1_option_chain.txt", firstrate_lines(days))
    prices = prices_until(QD)
    as_of = QD + timedelta(days=1)
    key = fr.make_answer_key(tmp_path, lambda t, a: prices)
    got = key("AAPL", as_of)

    th = thresholds(prices, as_of)
    target = (pd.Timestamp(QD) + pd.offsets.BDay(21)).date()
    sd = VOL * math.sqrt((target - QD).days / 365)
    def p_below(k):        # lognormal with zero rate, forward = spot
        return norm.cdf((math.log(k / SPOT) + 0.5 * sd * sd) / sd)
    want_fall = 100 * p_below(th.lower_price)
    want_gain = 100 * (1 - p_below(th.upper_price))
    assert got["fall"] == pytest.approx(want_fall, abs=1.0)
    assert got["gain"] == pytest.approx(want_gain, abs=1.0)
    assert sum(got.values()) == pytest.approx(100)


def test_answer_key_returns_none_and_says_why(tmp_path):
    log = []
    key = fr.make_answer_key(tmp_path, lambda t, a: prices_until(QD), unscorable_log=log)
    assert key("AAPL", QD + timedelta(days=1)) is None
    assert log[0]["reason"].startswith("no option file")

    write(tmp_path / "AAPL_2026_q1_option_chain.txt",
          firstrate_lines([QD], strikes=[95.0, 100.0, 105.0]))          # too few strikes
    assert key("AAPL", QD + timedelta(days=1)) is None
    assert "too few usable quotes" in log[1]["reason"]


AGENT_MODULES = ["agents", "tools", "graph", "llm", "roles", "world", "live", "handoffs",
                 "ledger", "risk", "report", "scenarios", "views", "cli"]


@pytest.mark.parametrize("name", AGENT_MODULES)
def test_no_agent_module_imports_the_answer_key(name):
    path = Path(fr.__file__).parent / f"{name}.py"
    if not path.exists():
        pytest.skip(f"{name}.py not present")
    imported = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.ImportFrom):
            imported.add((node.module or "").split(".")[-1])
            imported.update(a.name for a in node.names)
        elif isinstance(node, ast.Import):
            imported.update(a.name.split(".")[-1] for a in node.names)
    assert not imported & {"answer_key", "firstrate", "options_rnd", "results_day"}
