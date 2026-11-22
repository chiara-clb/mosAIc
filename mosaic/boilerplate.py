"""Spot quotes that are boilerplate: text that is really in the source but says nothing about the business.

The critic checks that a quote exists in its source. That is not enough: a 47-forecast run found that about
half the lines the tree chose quoted exhibit references, vote counts, section headings or standard risk
definitions. A reading resting on such a quote is pruned like a made-up one.
"""

from __future__ import annotations

import re

# Phrases that only appear in legal and filing machinery, never in news about the business.
PHRASES = (
    "incorporated herein by reference",
    "incorporated by reference",
    "filed as exhibit",
    "filed herewith",
    "furnished herewith",
    "qualified in its entirety",
    "refer to note",
    "see note",
    "(registrant)",
    "exact name of registrant",
    "registration statement on form",
    "pursuant to the requirements of",
    "indicate by check mark",
    "check the appropriate box",
    "forward-looking statements",
    "number of shares",
    "broker non-votes",
    "votes cast",
    "is the risk of loss",
    "is the risk that",
)

ITEM_HEADING = re.compile(r"^\s*item\s+\d+(\.\d+)?\b", re.IGNORECASE)        # "ITEM 2.03 CREATION OF ..."
NUMBER = re.compile(r"^[A-Z]{0,2}[$€£]?\(?\d[\d,.]*\)?%?$")                  # 1,399,727,580  C$4,750  (5,855)  26%


def numeric_share(quote: str) -> float:
    """Share of the words that are numbers, ignoring stray punctuation."""
    words = [w for w in quote.split() if re.search(r"\w", w)]
    return sum(bool(NUMBER.match(w)) for w in words) / len(words) if words else 0.0


def is_boilerplate(quote: str | None) -> bool:
    """True when the quote is filing machinery rather than content: a cross-reference, a legal formula, a
    section heading, a vote tally or table row, or a standard risk definition."""
    q = " ".join((quote or "").split())
    if not q:
        return False
    low = q.lower()
    if any(p in low for p in PHRASES) or ITEM_HEADING.match(q):
        return True
    words = [w for w in q.split() if re.search(r"\w", w)]
    return len(words) >= 3 and numeric_share(q) >= 0.4                       # a row of numbers with a label


def reason(quote: str | None) -> str:
    return "the quote is boilerplate (a reference, heading, vote count or standard wording), not news about the business"
