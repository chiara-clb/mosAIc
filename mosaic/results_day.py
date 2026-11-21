"""Results day: three baselines for the next-day move after an after-close results release.

A first step towards the earnings-release direction (see the earnings-release prototype, MosAIc v2).
For a stock that reports after the close on day t0, the question is the move from the close on t0 to
the close on the next session t1, in three buckets:

- **fall**: below -tau, **flat**: between -tau and +tau, **gain**: above +tau,
- with tau = 0.5 * J and J the root-mean-square of the previous 8 results-day moves.

The three baselines, all known before the release:

- **B1**: one year of ordinary daily moves (a normal distribution with that volatility).
- **B2**: the previous 8 results-day moves (a normal distribution with standard deviation J).
- **B3**: the option market at the close on t0 (mosaic.options_rnd).

No model is called. The actual move is shown next to them when it is already known.

    python -m mosaic.results_day AMZN 2026-07-30 --options-dir data/firstrate
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from scipy import stats

K = 0.5                      # tau = K * J
N_PAST = 8                   # past results used for J
LOOKBACK = 252               # sessions of daily moves for B1
NEW_YORK = ZoneInfo("America/New_York")
CLOSE = time(16, 0)
OUTCOMES = ("fall", "flat", "gain")


def reaction_session(released: datetime, sessions: pd.DatetimeIndex) -> pd.Timestamp | None:
    """The first session whose 16:00 New York close comes after the release."""
    local = released.astimezone(NEW_YORK) if released.tzinfo else released
    day = pd.Timestamp(local.date())
    i = sessions.searchsorted(day)
    if i < len(sessions) and sessions[i] == day and local.time() < CLOSE:
        return sessions[i]
    j = sessions.searchsorted(day, side="right")
    return sessions[j] if j < len(sessions) else None


def buckets_normal(sd: float, tau: float) -> np.ndarray:
    d = stats.norm(scale=sd)
    fall, below_gain = float(d.cdf(-tau)), float(d.cdf(tau))
    return np.array([fall, below_gain - fall, 1.0 - below_gain])


@dataclass
class ResultsDay:
    ticker: str
    release: datetime                     # acceptance time of the results 8-K
    t0: date                              # close before the release
    t1: date | None                       # the session that reacts
    tau: float                            # barrier, as a log return
    J: float
    sigma_d: float
    past_moves: list[tuple[str, float]]   # (date, move) of the previous results days
    baselines: dict[str, dict[str, float] | None]   # B1, B2, B3 -> {"fall","flat","gain"} in percent
    actual: dict | None = None            # {"move": r, "outcome": ...} once known
    notes: list[str] = field(default_factory=list)


def _pct(p: np.ndarray) -> dict[str, float]:
    return {k: round(float(v) * 100, 1) for k, v in zip(OUTCOMES, p)}


def results_day(ticker: str, release: datetime, closes: pd.Series, past_releases: list[datetime],
                chain: pd.DataFrame | None) -> ResultsDay:
    """The three baselines for one after-close release.

    closes: daily closes indexed by date (their dates are the trading sessions), including t1 if known.
    past_releases: earlier results acceptance times. chain: option quotes at the close on t0, or None.
    """
    closes = closes.sort_index()
    sessions = pd.DatetimeIndex(pd.to_datetime(closes.index).normalize())
    closes.index = sessions
    local = release.astimezone(NEW_YORK)
    if local.time() < CLOSE:
        raise ValueError(f"{ticker} released at {local:%H:%M} New York time, before the close: "
                         "this view is for after-close releases")
    t0 = pd.Timestamp(local.date())
    if t0 not in sessions:
        raise ValueError(f"no close for {ticker} on {t0.date()}")
    nxt = sessions[sessions > t0]
    t1 = nxt[0] if len(nxt) else None
    notes = []

    moves = []
    for r in sorted(p for p in past_releases if p < release - timedelta(days=20)):
        s = reaction_session(r, sessions)
        if s is None or s not in sessions or sessions.get_loc(s) == 0:
            continue
        i = sessions.get_loc(s)
        moves.append((s.date().isoformat(), float(math.log(closes.iloc[i] / closes.iloc[i - 1]))))
    moves = moves[-N_PAST:]
    if len(moves) < 4:
        raise ValueError(f"only {len(moves)} earlier results days with prices")
    if len(moves) < N_PAST:
        notes.append(f"J from {len(moves)} earlier results days, not {N_PAST}")
    J = float(math.sqrt(np.mean(np.square([m for _, m in moves]))))
    tau = K * J

    daily = np.diff(np.log(closes.loc[:t0].to_numpy(float)))[-LOOKBACK:]
    sigma_d = float(daily.std(ddof=1))
    if len(daily) < LOOKBACK:
        notes.append(f"daily volatility from {len(daily)} moves, not {LOOKBACK}")

    base = {"B1": _pct(buckets_normal(sigma_d, tau)), "B2": _pct(buckets_normal(J, tau)), "B3": None}
    if chain is not None and len(chain) and t1 is not None:
        from mosaic.options_rnd import next_day_distribution
        try:
            res = next_day_distribution(chain, t0, t1, tau, sessions.append(
                pd.bdate_range(sessions[-1] + pd.Timedelta(days=1), periods=40)), sigma_d ** 2)
            base["B3"] = _pct(res.probs)
            notes.append("options: " + res.diagnostics["path"])
        except ValueError as e:
            notes.append(f"options: {e}")
    elif chain is None or not len(chain):
        notes.append("options: no quotes at the close on the release day")

    actual = None
    if t1 is not None:
        r = float(math.log(closes.loc[t1] / closes.loc[t0]))
        actual = {"move": r, "outcome": "gain" if r > tau else "fall" if r < -tau else "flat"}
    return ResultsDay(ticker, release, t0.date(), t1.date() if t1 is not None else None, tau, J, sigma_d,
                      moves, base, actual, notes)


def rows(rd: ResultsDay) -> list[dict]:
    """One row per baseline, for a table."""
    names = {"B1": "B1 · one year of daily moves", "B2": "B2 · previous results days",
             "B3": "B3 · options at the close"}
    out = []
    for key, label in names.items():
        p = rd.baselines.get(key)
        row = {"Baseline": label}
        row.update({"Fall %": p["fall"], "Flat %": p["flat"], "Gain %": p["gain"]} if p else
                   {"Fall %": None, "Flat %": None, "Gain %": None})
        if p and rd.actual:
            row["Chance given to what happened %"] = p[rd.actual["outcome"]]
        out.append(row)
    return out


# ---------------------------------------------------------------- which releases can be chosen

def after_close_releases(releases: list[datetime], since: date | None = None) -> list[datetime]:
    """The results releases made after the 16:00 New York close, newest first (optionally from `since`)."""
    out = [r for r in releases if r.astimezone(NEW_YORK).time() >= CLOSE]
    if since:
        out = [r for r in out if r.astimezone(NEW_YORK).date() >= since]
    return sorted(out, reverse=True)


def quarter_reported(release_day: date) -> str:
    """The calendar quarter a release reports: the last quarter to end at least two weeks before it."""
    d = release_day - timedelta(days=14)
    q = (d.month - 1) // 3                       # quarters fully ended before d's quarter began
    year = d.year if q else d.year - 1
    return f"Q{q if q else 4} {year}"


def release_choices(releases: list[datetime], option_quarters: set[tuple[int, int]] | None = None,
                    since: date | None = None) -> list[dict]:
    """One entry per after-close release, for a picker: its day, time, and whether option prices cover it."""
    out = []
    for r in after_close_releases(releases, since):
        local = r.astimezone(NEW_YORK)
        d = local.date()
        has_options = option_quarters is not None and (d.year, (d.month - 1) // 3 + 1) in option_quarters
        q = quarter_reported(d)
        label = f"{q} results · released {d:%d %b %Y}" + (" · options available" if has_options else "")
        out.append({"day": d, "release": r, "options": has_options, "label": label, "quarter": q})
    return out


def live_release_choices(ticker: str, options_dir: str | None = None, get=None) -> list[dict]:   # pragma: no cover
    """The after-close results releases of a stock since 2023, from its SEC filings."""
    from mosaic.baseline import results_dates
    from mosaic.edgar import list_filings, sec_getter, ticker_to_cik
    get = get or sec_getter()
    _, filings = list_filings(ticker_to_cik(get)[ticker.upper()], get)
    quarters = None
    if options_dir:
        from mosaic.firstrate import find_files
        quarters = set(find_files(options_dir, ticker))
    return release_choices(results_dates(filings), quarters, since=date(2023, 1, 1))


# ---------------------------------------------------------------- live data

def load_live(ticker: str, release_day: date, options_dir: str | None = None, sec_get=None,
              price_fetch=None) -> ResultsDay:                      # pragma: no cover - needs internet
    """Fetch SEC results dates, prices and (if given) the FirstRate chain, then compute the baselines."""
    from mosaic.baseline import results_dates
    from mosaic.edgar import list_filings, sec_getter, ticker_to_cik
    from mosaic.prices import load_closes

    get = sec_get or sec_getter()
    _, filings = list_filings(ticker_to_cik(get)[ticker.upper()], get)
    releases = results_dates(filings)
    match = [r for r in releases if r.astimezone(NEW_YORK).date() == release_day]
    if not match:
        raise ValueError(f"no results 8-K from {ticker} dated {release_day} in its SEC filings")
    release = match[0]
    start = (release_day - timedelta(days=365 * 3)).isoformat()
    end = (release_day + timedelta(days=10)).isoformat()
    closes = load_closes(ticker.upper(), start, end, price_fetch)
    chain = None
    if options_dir:
        from mosaic.firstrate import chain_before
        quotes = chain_before(options_dir, ticker, datetime.combine(release_day + timedelta(days=1), time(),
                                                                   tzinfo=timezone.utc), look_back_days=1)
        if quotes is not None:
            chain = quotes[quotes["quote_date"] == release_day]
    return results_day(ticker.upper(), release, closes, releases, chain)


def main(argv=None) -> int:                                        # pragma: no cover - needs internet
    p = argparse.ArgumentParser(prog="mosaic.results_day", description=__doc__.split("\n\n")[0])
    p.add_argument("ticker")
    p.add_argument("release_day", help="the day of the after-close release, YYYY-MM-DD")
    p.add_argument("--options-dir", default=None)
    p.add_argument("--read", action="store_true", help="also run the release reader (Claude, needs ANTHROPIC_API_KEY)")
    p.add_argument("--consensus-dir", default="data/consensus")
    p.add_argument("--model", default="claude-haiku-5-5")
    a = p.parse_args(argv)
    rd = load_live(a.ticker, date.fromisoformat(a.release_day), a.options_dir)
    print(f"{rd.ticker}: results released {rd.release.astimezone(NEW_YORK):%Y-%m-%d %H:%M} New York time")
    print(f"Next-day move from the {rd.t0} close: a large move is beyond ±{rd.tau * 100:.1f}% "
          f"(half of J = {rd.J * 100:.1f}%, the size of the last {len(rd.past_moves)} results-day moves)")
    print(pd.DataFrame(rows(rd)).to_string(index=False))
    if rd.actual:
        print(f"Actual: {rd.actual['move'] * 100:+.1f}% on {rd.t1} → {rd.actual['outcome']}")
    for n in rd.notes:
        print("Note:", n)
    if a.read:
        from mosaic.llm import make_model
        from mosaic.release_reader import fetch_release, ledger_rows, load_consensus, read_release
        consensus = load_consensus(a.consensus_dir, rd.ticker, rd.t0)
        r = read_release(fetch_release(rd.ticker, rd.release), rd.baselines, rd.tau, consensus,
                         make_model("anthropic", model_name=a.model), rd.ticker)
        print(f"\nRelease reader: {r.status} ({len(consensus)} consensus values, {r.retries} retries)")
        if r.read:
            print(r.read)
        if r.forecast:
            print("MosAIc after the release:", r.forecast)
        print(pd.DataFrame(ledger_rows(r)).to_string(index=False) if r.ledger else "No ledger.")
        for f in r.flags:
            print("Flag:", f)
    return 0


if __name__ == "__main__":
    sys.exit(main())
