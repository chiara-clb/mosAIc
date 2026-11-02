import math
from datetime import timedelta

import numpy as np
import pytest

from tests.notebook.core import load_core
from tests.notebook.helpers import chunk, dt, prices, unit

ns = load_core()


# ---------- filings text ----------

FILING_HTML = """<html><body><ix:header>hidden xbrl</ix:header>
<p>Table of contents</p><p>Item 1A. Risk Factors 12</p>
<p>Item 1A. Risk Factors</p><p>""" + "risk " * 60 + """</p>
<p>Item 3. Legal Proceedings</p><p>""" + "lawsuit " * 60 + "</p></body></html>"


def test_html_to_text_drops_hidden_xbrl():
    text = ns["html_to_text"](FILING_HTML)
    assert "hidden xbrl" not in text and "Legal Proceedings" in text


def test_sections_skip_contents_entries():
    secs = ns["split_sections"](ns["html_to_text"](FILING_HTML), "Main document")
    assert [s[0] for s in secs] == ["Item 1A. Risk Factors", "Item 3. Legal Proceedings"]
    assert ns["section_key"](secs[1][0]) == "legal proceedings"


def test_section_key_matches_10q_and_10k_titles():
    sk = ns["section_key"]
    assert sk("Item 1. Legal Proceedings") == sk("Item 3. Legal Proceedings") == "legal proceedings"
    assert sk("Item 1. Financial Statements") != sk("Item 1. Legal Proceedings")
    assert sk("Press release") == "press release"
    assert sk("Item 7") == "item 7"


def test_chunks_have_size_and_overlap():
    words = [f"w{i}" for i in range(400)]
    chunks = ns["chunk_words"](" ".join(words))
    assert all(len(c.split()) <= 150 for c in chunks)
    assert chunks[1].split()[0] == "w120"           # 150 - 30 overlap


def test_event_key_uses_first_real_item():
    ek = ns["event_key"]
    assert ek("8-K", "2.02,9.01") == "8-K 2.02"
    assert ek("8-K", "9.01") == "8-K other"
    assert ek("8-K", "5.02,7.01,9.01") == "8-K 5.02"
    assert ek("10-Q", "") == "10-Q"


# ---------- search and evidence selection ----------

def make_index(chunks, vecs):
    return ns["Index"](chunks, np.array(vecs, dtype=np.float32))


def test_mmr_drops_near_copies():
    # realistic relevance gaps: Checkpoint 3 saw top-5 cosine scores within about 0.06
    emb = np.array([unit(1, 0, 0), unit(1, 0.01, 0), unit(0.7, 0.7, 0)])
    picked = ns["mmr_select"]([0.50, 0.495, 0.45], emb, 2, 0.7)
    assert picked == [0, 2]


def test_search_respects_dates_and_ticker():
    chunks = [chunk("old", accepted="2026-05-01"), chunk("ok", accepted="2026-07-20"),
              chunk("late", accepted="2026-08-01"), chunk("other", ticker="NKE", accepted="2026-07-20")]
    idx = make_index(chunks, [unit(1, 0)] * 4)
    as_of = dt("2026-08-01")
    res = ns["search_index"](idx, unit(1, 0), "AAPL", as_of, since=as_of - timedelta(days=30))
    assert [r["id"] for r in res["results"]] == ["ok"]


def test_search_reports_nothing_found_below_threshold():
    idx = make_index([chunk("a", accepted="2026-07-20")], [unit(0, 1)])
    res = ns["search_index"](idx, unit(1, 0), "AAPL", dt("2026-08-01"))
    assert res["results"] == [] and "nothing relevant" in res["status"]


def _res(*items):
    return {"status": "ok", "results": [dict(c, row=row, score=s) for c, row, s in items]}


def test_evidence_pools_searches_and_keeps_best_score():
    a = chunk("a", filing="F1")
    out = ns["select_evidence"](make_index([a], [unit(1, 0)]), [_res((a, 0, 0.4)), _res((a, 0, 0.7))])
    assert len(out) == 1 and out[0]["score"] == 0.7


def test_evidence_keeps_one_passage_per_filing():
    a, b, c = chunk("a", filing="F1"), chunk("b", filing="F1"), chunk("c", filing="F2")
    idx = make_index([a, b, c], [unit(1, 0, 0), unit(0, 1, 0), unit(0, 0, 1)])
    out = ns["select_evidence"](idx, [_res((a, 0, 0.5), (b, 1, 0.8), (c, 2, 0.6))])
    assert sorted(r["id"] for r in out) == ["b", "c"]


def test_evidence_at_most_five_diverse_and_newest_first():
    chunks, vecs = [], []
    for i in range(8):
        chunks.append(chunk(f"c{i}", filing=f"F{i}", accepted=f"2026-07-{10 + i}"))
        vecs.append(unit(1, 0.001 * i, 0) if i < 4 else unit(0.2, 1, i * 0.3))
    idx = make_index(chunks, vecs)
    res = _res(*[(c, i, 0.50 - 0.005 * i) for i, c in enumerate(chunks)])
    out = ns["select_evidence"](idx, [res])
    assert len(out) == 5
    assert sum(r["id"] in {"c0", "c1", "c2", "c3"} for r in out) < 4      # near-copies thinned out
    assert [r["accepted"] for r in out] == sorted([r["accepted"] for r in out], reverse=True)


def test_evidence_empty_when_nothing_found():
    assert ns["select_evidence"](make_index([], np.zeros((0, 2))), [{"status": "x", "results": []}]) == []


# ---------- prices, buckets and baseline ----------

def test_thresholds_use_only_earlier_prices():
    px = prices()
    as_of = px.index[400].to_pydatetime().replace(tzinfo=dt("2026-01-01").tzinfo)
    before = ns["thresholds"](px, as_of)
    shocked = px.copy()
    shocked.iloc[400:] *= 5
    assert ns["thresholds"](shocked, as_of) == before
    assert before["upper"] == pytest.approx(0.5 * before["sigma"])


def test_realised_bucket_and_missing_future():
    px = prices()
    as_of = px.index[400].to_pydatetime().replace(tzinfo=dt("2026-01-01").tzinfo)
    up = px.copy()
    up.iloc[400:] *= 1.5
    assert ns["realised_bucket"](up, as_of) == "gain"
    late = px.index[-5].to_pydatetime().replace(tzinfo=as_of.tzinfo)
    assert ns["realised_bucket"](px, late) is None


def test_baseline_sums_to_100_and_earnings_widen_tails():
    px = prices(n=700)
    as_of = px.index[600].to_pydatetime().replace(tzinfo=dt("2026-01-01").tzinfo)
    base, info = ns["baseline_probabilities"](px, as_of, [])
    assert sum(base.values()) == pytest.approx(100, abs=0.05) and not info["earnings_in_window"]
    last_results = as_of - timedelta(days=80)          # next results expected in about 11 days
    with_e, info_e = ns["baseline_probabilities"](px, as_of, [last_results - timedelta(days=91 * k) for k in range(4)])
    assert info_e["earnings_in_window"]
    assert with_e["flat"] < base["flat"]


def test_baseline_falls_back_without_prices():
    base, info = ns["baseline_probabilities"](prices(n=20), dt("2023-02-01"), [])
    assert base == {"fall": 30.85, "flat": 38.3, "gain": 30.85} and "fallback" in info


def test_results_dates_finds_item_202():
    rows = [{"form": "8-K", "items": "2.02,9.01", "acceptanceDateTime": "2026-07-30T20:30:00.000Z"},
            {"form": "8-K", "items": "5.02", "acceptanceDateTime": "2026-06-01T20:30:00.000Z"},
            {"form": "10-Q", "items": "", "acceptanceDateTime": "2026-07-31T20:30:00.000Z"}]
    assert ns["results_dates"](rows) == [dt("2026-07-30T20:30:00")]


def test_results_dates_skip_interim_updates():
    rows = [{"form": "8-K", "items": "2.02", "acceptanceDateTime": "2025-04-08T21:00:00.000Z"},   # interim update
            {"form": "8-K", "items": "2.02,9.01", "acceptanceDateTime": "2025-05-02T21:00:00.000Z"},
            {"form": "10-Q", "items": "", "acceptanceDateTime": "2025-05-07T21:00:00.000Z"}]
    assert ns["results_dates"](rows) == [dt("2025-05-02T21:00:00")]
