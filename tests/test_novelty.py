import zlib
from datetime import datetime, timedelta, timezone

import numpy as np

from mosaic.novelty import novelty, previous_filing_chunks, strip_heading

T0 = datetime(2025, 1, 1, tzinfo=timezone.utc)


def embed(texts):
    out = np.zeros((len(texts), 4096), dtype=np.float32)
    for i, t in enumerate(texts):
        for w in t.lower().split():
            out[i, zlib.crc32(w.encode()) % 4096] += 1
    n = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.where(n == 0, 1, n)


def c(filing, days, text, form="10-Q", key="legal proceedings", event="10-Q", section="Item 1. Legal Proceedings"):
    return {"ticker": "ACME", "form": form, "filing": filing, "accepted": T0 + timedelta(days=days), "text": text,
            "section_key": key, "event": event, "section": section}


LEGAL = "from time to time the company is involved in legal proceedings arising in the ordinary course"


def test_repeated_wording_scores_low_and_new_wording_high():
    chunks = [c("q1", 0, LEGAL), c("q2", 90, LEGAL), c("q3", 180, "a jury awarded damages of two billion dollars")]
    assert novelty(chunks, chunks[1], embed)[0] == 0.0
    assert novelty(chunks, chunks[2], embed) == (1.0, "q2")


def test_a_10k_counts_as_the_previous_report_for_a_10q():
    chunks = [c("k", 0, LEGAL, form="10-K", section="Item 3. Legal Proceedings"), c("q", 60, LEGAL)]
    assert [x["filing"] for x in previous_filing_chunks(chunks, chunks[1])] == ["k"]


def test_8ks_are_compared_only_with_the_same_item():
    a = c("a", 0, "officer change", form="8-K", key="departure of certain officers", event="8-K 5.02")
    b = c("b", 10, "other news", form="8-K", key="departure of certain officers", event="8-K 8.01")
    assert previous_filing_chunks([a, b], b) == []
    assert novelty([a, b], b, embed) == (1.0, None)


def test_headings_are_removed_before_comparing():
    s = "Item 5.02 Departure of Directors or Certain Officers"
    assert strip_heading(s + " On April 24 the CFO resigned.", s) == "On April 24 the CFO resigned."
    assert strip_heading("Item 5. 02 Something happened.") == "Something happened."
