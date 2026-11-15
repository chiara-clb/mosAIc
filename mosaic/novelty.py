"""Novelty: is a passage new, compared with the same section of the company's previous filing?

The previous filing is the last 10-Q or 10-K for a report section (sections are matched by title, since
the two forms number their items differently) or the last 8-K with the same item. Similarity of 0.5 or
less counts as fully new, 0.95 or more as fully repeated. Section headings are removed before comparing,
because two unrelated officer changes share the same long item title.
"""

from __future__ import annotations

import re

import numpy as np

from mosaic.config import SETTINGS

def strip_heading(text: str, section: str = "") -> str:
    """Drop a leading item heading (and the section title, if the text starts with it)."""
    t = text.strip()
    if section and t.lower().startswith(section.lower()):
        t = t[len(section):]
    t = re.sub(r"^\s*item\s+\d+[a-z]?(\.\s?\d{2})?\.?\s*", "", t, flags=re.I)
    return t.strip()


def previous_filing_chunks(chunks: list[dict], item: dict) -> list[dict]:
    family = ("10-Q", "10-K") if item["form"] in ("10-Q", "10-K") else ("8-K",)
    rows = [c for c in chunks if c["ticker"] == item["ticker"] and c["form"] in family and c["filing"] != item["filing"]
            and c["section_key"] == item["section_key"] and c["accepted"] < item["accepted"]
            and (c["form"] != "8-K" or c["event"] == item["event"])]
    if not rows:
        return []
    latest = max(c["accepted"] for c in rows)
    prev = next(c["filing"] for c in rows if c["accepted"] == latest)
    return [c for c in rows if c["filing"] == prev]


def novelty(chunks: list[dict], item: dict, embed) -> tuple[float, str | None]:
    """(0 = repeated, 1 = new; previous filing id or None when there is nothing to compare with)."""
    prev = previous_filing_chunks(chunks, item)
    if not prev:
        return 1.0, None
    vecs = embed([strip_heading(item["text"], item["section"])] + [strip_heading(c["text"], c["section"]) for c in prev])
    sim = float(np.max(vecs[1:] @ vecs[0]))
    lo, hi = SETTINGS.novelty_sim_range
    return float(min(max((hi - sim) / (hi - lo), 0.0), 1.0)), prev[0]["filing"]
