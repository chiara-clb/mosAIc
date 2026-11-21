"""Download and cache the data for an evaluation run, without calling any model.

For each stock and Wednesday it builds the same live world the evaluation uses (SEC filings, prices,
GDELT news), so everything lands in data/. It also checks the option-based answer key, and writes a
coverage table: how many filings and news stories each forecast will have. Run it before the
evaluation, so the paid run is fast and you can see in advance which dates have no evidence.

    python -m mosaic.prefetch --tickers AAPL,NKE,INTC,MSFT,JPM,XOM --start 2026-07-01 --end 2026-10-15 \
        --options-dir data/firstrate
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd

from mosaic.config import SETTINGS
from mosaic.evaluate import DEFAULT_TICKERS, OUT_DIR, UNIVERSE, default_price_source, forecast_dates


def coverage_row(ticker: str, as_of: datetime, world, gdelt_errors: int, key: dict | None) -> dict:
    fresh = as_of - timedelta(days=SETTINGS.fresh_days)
    filings = {(f["form"], f["filed"]) for f in world.filings if fresh <= f["filed"] < as_of}      # one per filing
    return {"ticker": ticker, "as_of": f"{as_of:%Y-%m-%d}",
            "filing_chunks_30d": sum(1 for f in world.filings if fresh <= f["filed"] < as_of),
            "filings_30d": len(filings),
            "news_stories": len(world.news),
            "gdelt_errors": gdelt_errors,
            "answer_key": "ok" if key else "missing"}


def parse_pairs(text: str) -> list[tuple[str, datetime]]:
    """"AAPL:2026-08-05,INTC:2026-07-29" -> [("AAPL", datetime), ...]."""
    pairs = []
    for part in filter(None, (p.strip() for p in text.split(","))):
        ticker, _, day = part.partition(":")
        d = datetime.fromisoformat(day.strip())
        pairs.append((ticker.strip().upper(), datetime(d.year, d.month, d.day, tzinfo=UTC)))
    return pairs


def prefetch(tickers, dates, *, world_factory, answer_key=None, out: Path | None = None, log=print,
             pairs: list | None = None) -> pd.DataFrame:
    """Every stock x every date, or exactly the (stock, date) `pairs` when given."""
    rows = []
    for t, d in pairs or [(t, d) for t in tickers for d in dates]:
        try:
            world, errors = world_factory(t, d)
            key = answer_key(t, d) if answer_key else None
            row = coverage_row(t, d, world, errors, key)
        except Exception as e:                      # keep going; the row says what failed
            row = {"ticker": t, "as_of": f"{d:%Y-%m-%d}", "error": f"{type(e).__name__}: {e}"}
        rows.append(row)
        log(f"{t} {d:%Y-%m-%d}: " + ", ".join(f"{k}={v}" for k, v in row.items() if k not in ("ticker", "as_of")))
        if out:
            out.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(out, index=False)
    return pd.DataFrame(rows)


def live_factory():                                     # pragma: no cover - needs internet
    from mosaic.index import sentence_transformer_embed
    from mosaic.live import live_world
    from mosaic.news import Gdelt

    embed, gdelt = sentence_transformer_embed(), Gdelt()

    def build(ticker, as_of):
        before = len(gdelt.errors)
        w = live_world(ticker, as_of, gdelt=gdelt, embed=embed)
        return w, len(gdelt.errors) - before
    return build


def main(argv=None) -> int:                             # pragma: no cover - needs internet
    p = argparse.ArgumentParser(prog="mosaic.prefetch", description=__doc__.split("\n\n")[0])
    p.add_argument("--tickers", default=",".join(DEFAULT_TICKERS))
    p.add_argument("--start", default="2026-07-01")
    p.add_argument("--end", default=None)
    p.add_argument("--options-dir", default=None)
    p.add_argument("--dates", default=None, help='only these, e.g. "AAPL:2026-08-05,INTC:2026-07-29"')
    p.add_argument("--out", default=str(Path(OUT_DIR) / "coverage.csv"))
    a = p.parse_args(argv)
    tickers = sorted(UNIVERSE) if a.tickers.lower() == "all" else [t.strip().upper() for t in a.tickers.split(",")]
    pairs = parse_pairs(a.dates) if a.dates else None
    dates = [] if pairs else forecast_dates(a.start, a.end or datetime.now(UTC).date().isoformat())
    key = None
    if a.options_dir:
        from mosaic.firstrate import make_answer_key
        key = make_answer_key(a.options_dir, default_price_source)
    n = len(pairs) if pairs else len(tickers) * len(dates)
    print(f"{n} stock-dates. News: {SETTINGS.news_fetch_days} days each, one call per day, paced to GDELT's limits.")
    table = prefetch(tickers, dates, world_factory=live_factory(), answer_key=key, out=Path(a.out), pairs=pairs)
    if "news_stories" in table:
        print(f"\nDates with no filing and no news: {int(((table.filings_30d == 0) & (table.news_stories == 0)).sum())} of {len(table)}")
    print(f"Coverage table: {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
