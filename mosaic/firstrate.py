"""Load FirstRate Data option files into the standard chain format used by the answer key.

Used only for evaluation. No agent module may import this file.

FirstRate's end-of-day option files have no header and 16 columns, in this order:
    trade date, strike, expiry date, call/put ("c" or "p"), last trade price, bid, ask,
    bid implied volatility, ask implied volatility, open interest, volume,
    delta, gamma, vega, theta, rho
Files are named like ZUMZ_2025_q1_option_chain.txt, one per stock and quarter, and may come zipped.

There is no column for the stock price, which the answer key needs. It is worked out from the
option prices themselves (put-call parity): for a strike K, call minus put equals the discounted
forward price minus the discounted strike, so the forward is K + (call - put) / discount.
The estimate takes the median over the strikes nearest the money, on the expiry closest to one month.
If daily closes are passed in, they are used instead.
"""

from __future__ import annotations

import io
import math
import re
import zipfile
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Callable

import pandas as pd

from .answer_key import REQUIRED_COLUMNS, implied_bucket_probabilities
from .buckets import thresholds

COLUMNS = ["quote_date", "strike", "expiration", "right", "last", "bid", "ask",
           "iv_bid", "iv_ask", "open_interest", "volume", "delta", "gamma", "vega", "theta", "rho"]
FILE_PATTERN = re.compile(r"^(?P<ticker>[A-Z.\-]+)_(?P<year>\d{4})_q(?P<q>[1-4])_option_chain", re.I)


# ---------- Reading ----------

def _read_text(source) -> pd.DataFrame:
    return pd.read_csv(source, header=None, names=COLUMNS, usecols=range(len(COLUMNS)),
                       dtype={"right": str}, skipinitialspace=True)


def read_firstrate(path: str | Path) -> pd.DataFrame:
    """Read one FirstRate file (.txt or .csv, or a .zip holding them) as a raw table."""
    path = Path(path)
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as z:
            names = [n for n in z.namelist() if n.lower().endswith((".txt", ".csv"))]
            if not names:
                raise ValueError(f"{path.name} holds no .txt or .csv file")
            parts = [_read_text(io.BytesIO(z.read(n))) for n in sorted(names)]
        raw = pd.concat(parts, ignore_index=True)
    else:
        raw = _read_text(path)
    raw["quote_date"] = pd.to_datetime(raw["quote_date"]).dt.date
    raw["expiration"] = pd.to_datetime(raw["expiration"]).dt.date
    raw["right"] = raw["right"].str.strip().str.upper().str[0]
    bad = ~raw["right"].isin(["C", "P"])
    if bad.any():
        raise ValueError(f"{path.name}: {int(bad.sum())} rows are neither calls nor puts; is this a FirstRate option file?")
    return raw


# ---------- The stock price from put-call parity ----------

def parity_forward(rows: pd.DataFrame, rate: float = 0.0, n_strikes: int = 3) -> float | None:
    """Forward price implied by one expiry's calls and puts at the same strikes."""
    days = (rows["expiration"].iloc[0] - rows["quote_date"].iloc[0]).days
    disc = math.exp(-rate * days / 365.0)
    quoted = rows[(rows["bid"] > 0) & (rows["ask"] >= rows["bid"])].copy()
    quoted["mid"] = (quoted["bid"] + quoted["ask"]) / 2
    calls = quoted[quoted["right"] == "C"].set_index("strike")["mid"]
    puts = quoted[quoted["right"] == "P"].set_index("strike")["mid"]
    both = calls.index.intersection(puts.index)
    if len(both) == 0:
        return None
    gap = (calls[both] - puts[both])
    nearest = gap.abs().sort_values().index[:n_strikes]          # strikes nearest the money
    forwards = [k + gap[k] / disc for k in nearest]
    return float(pd.Series(forwards).median())


def implied_spot(day_rows: pd.DataFrame, rate: float = 0.0, target_days: int = 30,
                 min_days: int = 5) -> float | None:
    """Stock price on one quote date, from the expiry closest to `target_days` that gives a forward."""
    qd = day_rows["quote_date"].iloc[0]
    expiries = sorted(e for e in day_rows["expiration"].unique() if (e - qd).days >= min_days)
    for exp in sorted(expiries, key=lambda e: abs((e - qd).days - target_days)):
        fwd = parity_forward(day_rows[day_rows["expiration"] == exp], rate)
        if fwd is not None and fwd > 0:
            return fwd * math.exp(-rate * (exp - qd).days / 365.0)
    return None


def to_chain(raw: pd.DataFrame, closes: pd.Series | None = None, rate: float = 0.0) -> pd.DataFrame:
    """Raw FirstRate rows -> the answer key's chain format, with an underlying price for each quote date.

    `closes`: optional daily closes indexed by date; when a quote date has a close it is used,
    otherwise the price comes from put-call parity. Quote dates with neither are dropped.
    """
    close_by_day = {}
    if closes is not None and len(closes):
        close_by_day = {pd.Timestamp(d).date(): float(v) for d, v in closes.dropna().items()}
    spots = {}
    for qd, day_rows in raw.groupby("quote_date"):
        spot = close_by_day.get(qd) or implied_spot(day_rows, rate)
        if spot is not None:
            spots[qd] = spot
    chain = raw[raw["quote_date"].isin(spots)].copy()
    chain["underlying_price"] = chain["quote_date"].map(spots)
    return chain[list(REQUIRED_COLUMNS) + ["volume", "open_interest"]].reset_index(drop=True)


# ---------- Finding the files for a stock and date ----------

def quarter_of(d: date) -> tuple[int, int]:
    return d.year, (d.month - 1) // 3 + 1


def find_files(directory: str | Path, ticker: str) -> dict[tuple[int, int], Path]:
    """FirstRate files for one stock in `directory`, keyed by (year, quarter)."""
    found = {}
    for p in Path(directory).rglob("*"):
        m = FILE_PATTERN.match(p.name)
        if p.is_file() and m and m["ticker"].upper() == ticker.upper():
            key = (int(m["year"]), int(m["q"]))
            if key not in found or p.suffix.lower() != ".zip":     # prefer an unzipped copy
                found[key] = p
    return found


def chain_before(directory: str | Path, ticker: str, as_of, closes: pd.Series | None = None,
                 rate: float = 0.0, look_back_days: int = 10) -> pd.DataFrame | None:
    """Option quotes for the last few trading days before `as_of`, in the chain format, or None."""
    as_of = as_of.date() if isinstance(as_of, (datetime, pd.Timestamp)) else as_of
    first = as_of - timedelta(days=look_back_days)
    files = find_files(directory, ticker)
    needed = sorted({quarter_of(first), quarter_of(as_of - timedelta(days=1))})
    parts = []
    for key in needed:
        if key in files:
            raw = read_firstrate(files[key])
            parts.append(raw[(raw["quote_date"] >= first) & (raw["quote_date"] < as_of)])
    if not parts:
        return None
    raw = pd.concat(parts, ignore_index=True)
    if raw.empty:
        return None
    return to_chain(raw, closes=closes, rate=rate)


# ---------- The hook for the evaluation runner ----------

def make_answer_key(directory: str | Path, price_source: Callable, rate: float = 0.0,
                    use_closes: bool = True, unscorable_log: list | None = None) -> Callable:
    """Return answer_key(ticker, as_of) -> {"fall", "flat", "gain"} in percent, or None.

    `price_source(ticker, as_of)` gives daily closes; they set the fall and gain thresholds and,
    with `use_closes`, the stock price for the option maths. Dates that cannot be scored return
    None and, if `unscorable_log` is a list, are appended to it with the reason.
    """
    def answer_key(ticker: str, as_of) -> dict | None:
        prices = price_source(ticker, as_of)
        chain = chain_before(directory, ticker, as_of, closes=prices if use_closes else None, rate=rate)
        reason = None
        if chain is None:
            reason = "no option file or no quotes before the date"
        else:
            key = implied_bucket_probabilities(chain, thresholds(prices, as_of), rate=rate)
            if key.status == "ok":
                return dict(key.probabilities)
            reason = key.reason
        if unscorable_log is not None:
            unscorable_log.append({"ticker": ticker, "as_of": str(as_of), "reason": reason})
        return None
    return answer_key
