"""The live world on saved SEC pages, fake prices and fake GDELT replies (no network)."""

import json
import zlib
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from mosaic import minigraph
from mosaic.graph import run_forecast
from mosaic.live import live_world
from mosaic.llm import make_model
from mosaic.news import Gdelt
from mosaic.tools import ToolBox
from mosaic.trace import leakage_audit

UTC = timezone.utc
AS_OF = datetime(2025, 6, 2, tzinfo=UTC)
FILLER = " ".join(f"detail{i}" for i in range(60))
ENGINE = (minigraph.StateGraph, minigraph.START, minigraph.END, "built-in runner")


def embed(texts):
    out = np.zeros((len(texts), 4096), dtype=np.float32)
    for i, t in enumerate(texts):
        for w in t.lower().replace("|", " ").split():
            out[i, zlib.crc32(w.encode()) % 4096] += 1
    n = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.where(n == 0, 1, n)


SUB = {"name": "Acme Robotics Inc.", "filings": {"files": [], "recent": {
    "form": ["8-K", "8-K", "10-Q", "8-K"],
    "filingDate": ["2025-06-04", "2025-05-28", "2025-05-09", "2025-04-29"],
    "acceptanceDateTime": ["2025-06-04T11:00:00.000Z", "2025-05-28T21:05:00.000Z", "2025-05-09T20:30:00.000Z",
                           "2025-04-29T20:10:00.000Z"],
    "accessionNumber": ["A-4", "A-3", "A-2", "A-1"],
    "primaryDocument": ["merger.htm", "cfo.htm", "q.htm", "results.htm"],
    "items": ["1.01", "5.02,9.01", "", "2.02"]}}}
BASE = "https://www.sec.gov/Archives/edgar/data/42/"
PAGES = {
    "https://www.sec.gov/files/company_tickers.json": json.dumps({"0": {"ticker": "ACME", "cik_str": 42}}),
    "https://data.sec.gov/submissions/CIK0000000042.json": json.dumps(SUB),
    BASE + "A3/index.json": json.dumps({"directory": {"item": [{"name": "cfo.htm"}]}}),
    BASE + "A4/index.json": json.dumps({"directory": {"item": []}}),
    BASE + "A1/index.json": json.dumps({"directory": {"item": []}}),
    BASE + "A3/cfo.htm": "<p>Item 5.02 Departure of Certain Officers</p><p>Jane Doe will resign as chief financial "
                         "officer and no interim chief financial officer has been named. " + FILLER + "</p>",
    BASE + "A2/q.htm": "<p>Item 1. Legal Proceedings</p><p>The company is involved in legal proceedings. " + FILLER + "</p>",
    BASE + "A1/results.htm": "<p>Item 2.02 Results of Operations</p><p>Acme issued a press release with results. " + FILLER + "</p>",
    BASE + "A4/merger.htm": "<p>Item 1.01 Entry into a Material Definitive Agreement</p><p>Acme agreed to be acquired. " + FILLER + "</p>",
}


class Resp:
    def __init__(self, body):
        self.text = body

    def json(self):
        return json.loads(self.text)


def sec_get(url):
    return Resp(PAGES[url])


def price_fetch(ticker, start, end):
    days = pd.bdate_range(start, end)
    rng = np.random.default_rng(0 if ticker == "ACME" else 1)
    return pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(days)))), index=days)


def gdelt(tmp_path):
    arts = {"articles": [
        {"title": "Acme Robotics wins warehouse automation contract", "seendate": "20250522T150000Z", "domain": "a"},
        {"title": "Acme Robotics wins big warehouse automation contract", "seendate": "20250522T160000Z", "domain": "b"},
        {"title": "Acme Robotics agrees to be acquired", "seendate": "20250604T120000Z", "domain": "c"},
    ]}
    return Gdelt(tmp_path / "g.json", http_get=lambda p: arts if p["mode"] == "artlist" else {}, spacing=0)


def build(tmp_path):
    return live_world("ACME", AS_OF, sec_get=sec_get, gdelt=gdelt(tmp_path), embed=embed,
                      price_fetch=price_fetch, cache_dir=tmp_path / "px")


def test_live_world_has_dated_chunks_news_prices_and_results(tmp_path):
    w = build(tmp_path)
    assert w.company == "Acme Robotics Inc." and {f["form"] for f in w.filings} == {"8-K", "10-Q"}
    assert all(f["filed"] < AS_OF for f in w.filings)                      # the merger 8-K is after the date
    assert len(w.news) == 1 and w.news[0]["first_seen"] < AS_OF
    assert w.earnings_dates == [datetime(2025, 4, 29, 20, 10, tzinfo=UTC)]
    assert max(w.prices["dates"]) < AS_OF


def test_live_tools_search_the_index_and_enrich_passages(tmp_path):
    box = ToolBox(build(tmp_path))
    hits = box.call("search_documents", "research", query="chief financial officer resign")
    assert hits and "Item 5.02" in hits[0]["header"] and "text" not in hits[0]
    every = box.call("search_documents", "research", query="", list_all=True)
    assert len(every) == 2                                                   # CFO 8-K and 10-Q in the window
    item = box.passage(hits[0]["id"])
    assert 0 <= item["novelty"] <= 1 and "reaction" in item
    rates = box.call("case_memory", "synthesis", event=item["event"], reaction=item["reaction"], lag_days=5)
    assert set(rates) >= {"fall", "gain", "spread", "none", "n"}


def test_a_full_forecast_runs_on_the_live_world_without_leakage(tmp_path):
    state, run = run_forecast(build(tmp_path), make_model("scripted"), engine=ENGINE)
    assert state["final"]["status"] in ("accepted", "baseline") and state["evidence"]
    assert leakage_audit(run.tools.trace) == []


def test_live_world_reports_each_stage(tmp_path):
    said = []
    live_world("ACME", AS_OF, sec_get=sec_get, gdelt=gdelt(tmp_path), embed=embed,
               price_fetch=price_fetch, cache_dir=tmp_path / "px", progress=said.append)
    assert said[0] == "Listing SEC filings"
    assert any(s.startswith("Indexing") for s in said)
    assert sum(s.startswith("News: ") and "days, downloading)" in s for s in said) == 14
    assert said[-1].startswith("News: ") and said[-1].endswith("stories found")
