import zlib
from datetime import datetime, timedelta, timezone

import numpy as np

from mosaic.index import Index, is_boilerplate, mmr_select

UTC = timezone.utc
AS_OF = datetime(2025, 6, 2, tzinfo=UTC)


def hash_embed(texts):
    """A tiny deterministic bag-of-words embedding for tests."""
    out = np.zeros((len(texts), 4096), dtype=np.float32)
    for i, t in enumerate(texts):
        for w in t.lower().replace(",", " ").replace(";", " ").split():
            out[i, zlib.crc32(w.encode()) % 4096] += 1
    n = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.where(n == 0, 1, n)


def chunk(cid, filing, days_before, text, form="8-K", event="8-K 5.02", ticker="ACME"):
    return {"id": cid, "ticker": ticker, "company": "Acme", "form": form, "section": "s", "section_key": "s",
            "event": event, "accepted": AS_OF - timedelta(days=days_before), "filing": filing, "text": text,
            "header": f"Acme | {form}"}


CHUNKS = [
    chunk("a", "f1", 5, "departure of certain officers chief financial officer resign"),
    chunk("b", "f1", 5, "signature pursuant to the requirements of the securities exchange act"),
    chunk("c", "f2", 20, "legal proceedings ordinary course of business", form="10-Q", event="10-Q"),
    chunk("d", "f3", 40, "chief financial officer resign old news"),
    chunk("late", "f4", -1, "chief financial officer resign merger"),
    chunk("other", "f5", 3, "chief financial officer resign", ticker="BETA"),
]


def make():
    return Index(CHUNKS, hash_embed)


def test_search_applies_the_date_window_and_the_stock():
    res = make().search("chief financial officer resign", "ACME", AS_OF, AS_OF - timedelta(days=30))
    ids = [r["id"] for r in res["results"]]
    assert "late" not in ids and "d" not in ids and "other" not in ids and ids[0] == "a"


def test_search_reports_nothing_relevant_and_no_documents():
    idx = make()
    assert idx.search("zebra volcano", "ACME", AS_OF, AS_OF - timedelta(days=30))["status"].startswith("nothing relevant")
    assert idx.search("x", "ACME", AS_OF - timedelta(days=60))["status"] == "no documents in this date range"


def test_every_filing_in_the_window_is_a_candidate_without_boilerplate():
    cands = make().filing_candidates("ACME", AS_OF, AS_OF - timedelta(days=30))
    assert {c["filing"] for c in cands} == {"f1", "f2"} and {c["id"] for c in cands} == {"a", "c"}
    assert is_boilerplate(CHUNKS[1]["text"])


def test_select_evidence_one_per_filing_newest_first():
    idx = make()
    cands = idx.filing_candidates("ACME", AS_OF, AS_OF - timedelta(days=30))
    cands.append({**CHUNKS[0], "row": 0, "score": 0.9})            # same filing found again by a search
    chosen = idx.select_evidence(cands)
    assert [c["filing"] for c in chosen] == ["f1", "f2"] and chosen[0]["score"] == 0.9


def test_mmr_drops_near_copies():
    emb = np.array([[1, 0], [1, 0], [0, 1]], dtype=float)
    assert mmr_select([0.9, 0.89, 0.5], emb, 2, 0.7) == [0, 2]
