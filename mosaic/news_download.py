"""Download GDELT headlines slowly for a few stocks, so forecasts can run on saved news.

GDELT refuses calls that come too fast. This asks for one day of headlines at a time, waits `--spacing`
seconds between calls (longer after a refusal) and saves every answer in the same cache the forecasts
read (data/gdelt_cache.json). It covers every day from a week before `--start` to `--end`, so a forecast
on any date in the range finds its news, plus the tone timeline for each Wednesday and each `--also` date.
Stopping and restarting is safe: saved days are skipped.

    python -m mosaic.news_download --tickers AMZN,GOOGL,NVDA --start 2026-07-01 --end 2026-08-19 \
        --also 2026-07-08,2026-07-30 --spacing 15

Afterwards, run with MOSAIC_NEWS_OFFLINE=1 to use only what was saved.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, date, datetime, timedelta

from mosaic.config import SETTINGS
from mosaic.news import Gdelt, day_params, news_query, tone_params


def news_days_between(start: date, end: date) -> list[tuple[datetime, datetime]]:
    """Whole UTC days a forecast from `start` to `end` may read: from `news_fetch_days` before start to the
    day before end."""
    first = start - timedelta(days=SETTINGS.news_fetch_days)
    out, d = [], first
    while d < end:
        s = datetime(d.year, d.month, d.day, tzinfo=UTC)
        out.append((s, s + timedelta(hours=23, minutes=59, seconds=59)))
        d += timedelta(days=1)
    return out


def tone_dates(start: date, end: date, also: list[date] = ()) -> list[datetime]:
    """Each Wednesday from start to end, plus the extra dates, as forecast times (midnight UTC)."""
    d = start + timedelta(days=(2 - start.weekday()) % 7)
    days = set(also)
    while d <= end:
        days.add(d)
        d += timedelta(days=7)
    return [datetime(x.year, x.month, x.day, tzinfo=UTC) for x in sorted(days)]


def download(tickers: list[str], start: date, end: date, gdelt: Gdelt, also: list[date] = (), log=print) -> list[dict]:
    """Ask GDELT for every needed call that is not saved yet. One summary row per stock."""
    rows = []
    for ticker in tickers:
        q = news_query(ticker, ticker)
        days = news_days_between(start, end)
        articles, failed = 0, 0
        for n, (s, e) in enumerate(days, 1):
            params = day_params(q, s, e)
            saved = json.dumps(params, sort_keys=True) in gdelt.cache
            before = len(gdelt.errors)
            got = len((gdelt(params) or {}).get("articles", []))
            failed += len(gdelt.errors) > before
            articles += got
            status = "saved" if saved else ("failed, run again later" if len(gdelt.errors) > before else "downloaded")
            log(f"{ticker} {s:%Y-%m-%d} ({n} of {len(days)}): {got} headlines, {status}")
        for t in tone_dates(start, end, also):
            before = len(gdelt.errors)
            gdelt(tone_params(q, t))
            failed += len(gdelt.errors) > before
        rows.append({"ticker": ticker, "days": len(days), "headlines": articles, "failed calls": failed})
    return rows


def main(argv=None) -> None:  # pragma: no cover - needs internet
    p = argparse.ArgumentParser(prog="mosaic.news_download", description=__doc__.split("\n\n")[0])
    p.add_argument("--tickers", required=True, help="e.g. AMZN,GOOGL,NVDA")
    p.add_argument("--start", required=True, help="first forecast date, YYYY-MM-DD")
    p.add_argument("--end", required=True, help="last forecast date, YYYY-MM-DD")
    p.add_argument("--also", default="", help="other forecast dates for the tone timeline, e.g. 2026-07-30")
    p.add_argument("--spacing", type=float, default=15.0, help="seconds between calls (GDELT asks for at least 5)")
    a = p.parse_args(argv)
    tickers = [t.strip().upper() for t in a.tickers.split(",") if t.strip()]
    also = [date.fromisoformat(x.strip()) for x in a.also.split(",") if x.strip()]
    gdelt = Gdelt(spacing=a.spacing, offline=False)
    gdelt.progress = print
    rows = download(tickers, date.fromisoformat(a.start), date.fromisoformat(a.end), gdelt, also)
    print("\nSummary")
    for r in rows:
        print(f"  {r['ticker']}: {r['headlines']} headlines over {r['days']} days, {r['failed calls']} failed calls")
    if any(r["failed calls"] for r in rows):
        print("Some calls failed. Run the same command again later: saved days are skipped.")


if __name__ == "__main__":  # pragma: no cover
    main()
