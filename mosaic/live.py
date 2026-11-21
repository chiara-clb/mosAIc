"""The live world: a real stock on a real date, from SEC EDGAR, prices and GDELT news.

It has the same shape as the demo world, so the same tools and agents run on it. The document search
uses the embedding index, case memory counts past events from SEC and prices, and each passage gets its
novelty and price reaction when an agent first reads it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable

from mosaic.baseline import results_dates
from mosaic.case_memory import CaseMemory
from mosaic.config import SETTINGS
from mosaic.edgar import build_corpus, event_key, list_filings, parse_accepted, sec_getter, ticker_to_cik
from mosaic.index import Index, sentence_transformer_embed
from mosaic.news import Gdelt, news_items
from mosaic.novelty import novelty
from mosaic.prices import load_closes
from mosaic.tools import clean_extract
from mosaic.world import World


@dataclass
class LiveWorld(World):
    enrich: Callable[[dict], dict] | None = None
    _cache: dict = field(default_factory=dict)

    def item(self, item_id: str) -> dict | None:
        if item_id not in self._cache:
            it = super().item(item_id)
            self._cache[item_id] = self.enrich(it) if it is not None and self.enrich else it
        return self._cache[item_id]


def _hits(results: list[dict]) -> list[dict]:
    out = []
    for r in results:
        extract, removed = clean_extract(r["text"])
        out.append({"id": r["id"], "kind": "filing", "score": r["score"], "extract": extract,
                    "header": r["header"] + (" | instruction-like text removed" if removed else "")})
    return out


def live_world(ticker: str, as_of: datetime, *, sec_get=None, gdelt: Gdelt | None = None, embed=None,
               price_fetch=None, cache_dir=None, history_days: int = 200, price_years: int = 6,
               progress=None) -> LiveWorld:
    """Build the world for a real stock. `progress(message)` is called at each stage, for a live display."""
    loading: list[str] = []                # kept in the world, so the run's log shows how the data was loaded

    def say(message: str) -> None:
        loading.append(message)
        if progress:
            progress(message)
    get = sec_get or sec_getter()
    embed = embed or sentence_transformer_embed()
    if gdelt is None:                      # saved Wayback news when it covers this stock and date, else GDELT
        from mosaic.news_archive import ArchiveNews
        archive = ArchiveNews()
        gdelt = archive if archive.covers(ticker, as_of) else Gdelt()
    say("Listing SEC filings")
    cik = ticker_to_cik(get)[ticker]
    company, rows = list_filings(cik, get)
    say(f"Reading {company}'s filings from the last {history_days} days")

    # Documents from the last 200 days (enough for the previous 10-Q or 10-K, for novelty)
    corpus = build_corpus(ticker, company, cik, rows, as_of - timedelta(days=history_days), as_of, get)
    say(f"Indexing {len(corpus)} passages for search")
    index = Index(corpus, embed)
    say("Loading prices")

    # Prices and case memory (this stock's own past events)
    start, end = (as_of - timedelta(days=365 * price_years)).date().isoformat(), as_of.date().isoformat()
    stock = load_closes(ticker, start, end, price_fetch, cache_dir)
    market = load_closes(SETTINGS.market_ticker, start, end, price_fetch, cache_dir)
    events = {ticker: [(event_key(r["form"], r.get("items", "")), parse_accepted(r["acceptanceDateTime"])) for r in rows]}
    cm = CaseMemory(events, {ticker: stock}, market=market)
    both = stock.index.intersection(market.index)
    both = both[both < as_of.strftime("%Y-%m-%d")]                 # closes before the forecast day only
    prices = {"dates": [datetime(d.year, d.month, d.day, tzinfo=as_of.tzinfo) for d in both],
              ticker: [float(x) for x in stock.loc[both]], "SPY": [float(x) for x in market.loc[both]]}

    def reaction(when: datetime) -> str | None:
        rx = cm.reaction(ticker, when, max((as_of - when).days, 1))
        return rx["bucket"] if rx else None

    # News stories, with their price reaction
    news = news_items(company, ticker, as_of, gdelt, embed, say, filings=corpus)
    say(f"News: {len(news)} stories found")
    for n in news:
        n["reaction"] = reaction(n["first_seen"])

    filings = [{"id": c["id"], "form": c["form"], "item": c["event"], "event": c["event"], "section": c["section"],
                "section_key": c["section_key"], "filed": c["accepted"], "text": c["text"], "header": c["header"],
                "chunk": c} for c in corpus]

    def enrich(it: dict) -> dict:
        if "chunk" not in it or "novelty" in it:
            return it
        nov, prev = novelty(corpus, it["chunk"], embed)
        return {**it, "novelty": nov, "previous_filing": prev, "reaction": reaction(it["filed"])}

    def search_documents(query: str, as_of: datetime, k: int, list_all: bool) -> list[dict]:
        since = as_of - timedelta(days=SETTINGS.fresh_days)
        found = index.filing_candidates(ticker, as_of, since) if list_all else index.search(query, ticker, as_of, since, k)["results"]
        return _hits(found)

    def case_rates(event: str, reaction_bucket: str | None, lag_days: int | None) -> dict:
        return cm.rates("*" if event == "news" else event, lag_days or 1, as_of, reaction_bucket)

    return LiveWorld(ticker=ticker, company=company, as_of=as_of, prices=prices, filings=filings, news=news,
                     case_memory={}, earnings_dates=results_dates(rows), enrich=enrich,
                     extra={"search_documents": search_documents, "case_rates": case_rates, "index": index,
                            "loading": loading})
