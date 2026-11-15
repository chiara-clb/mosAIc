"""The document index: embeddings, search with the forecast-date rule, MMR, and evidence candidates.

Each chunk is embedded as "header [SEP] text". Search keeps only chunks of the stock accepted in the
30 days before the forecast date, takes the 20 closest and reranks them to 5 with MMR, so the five do
not repeat each other. Every filing in the window is also a candidate on its own: its passage closest
to the filing's own subject, skipping signature pages and form text.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Callable

import numpy as np

from mosaic.config import SETTINGS

Embed = Callable[[list[str]], np.ndarray]          # texts -> normalised rows

ITEM_TITLES = {
    "1.01": "Entry into a Material Definitive Agreement",
    "1.02": "Termination of a Material Definitive Agreement",
    "1.05": "Material Cybersecurity Incidents",
    "2.01": "Completion of Acquisition or Disposition of Assets",
    "2.02": "Results of Operations and Financial Condition, quarterly results, revenue and outlook",
    "2.03": "Creation of a Direct Financial Obligation",
    "2.05": "Costs Associated with Exit or Disposal Activities, restructuring",
    "2.06": "Material Impairments",
    "5.02": "Departure of Directors or Certain Officers; Election of Directors; Appointment of Certain Officers",
    "5.07": "Submission of Matters to a Vote of Security Holders",
    "7.01": "Regulation FD Disclosure",
    "8.01": "Other Events",
}
REPORT_TOPICS = "results of operations, outlook and guidance, risk factors, legal proceedings"
BOILERPLATE = re.compile(
    r"pursuant to the requirements of the securities exchange act|check the appropriate box|"
    r"securities registered pursuant to section 12|cover page interactive data|inline xbrl|"
    r"indicate by check mark|exhibit index|forward-looking statements", re.I)


def sentence_transformer_embed(model_name: str | None = None) -> Embed:       # pragma: no cover - downloads a model
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name or SETTINGS.embed_model)
    return lambda texts: np.asarray(model.encode(texts, normalize_embeddings=True), dtype=np.float32)


def is_boilerplate(text: str) -> bool:
    return bool(BOILERPLATE.search(text))


def mmr_select(rel, cand_emb: np.ndarray, k: int, lam: float) -> list[int]:
    """Maximal Marginal Relevance: k items that are relevant (rel) but not near-copies of each other."""
    rel = np.asarray(rel, dtype=float)
    selected, remaining = [], list(range(len(rel)))
    while remaining and len(selected) < k:
        if not selected:
            best = max(remaining, key=lambda i: rel[i])
        else:
            sims = cand_emb[remaining] @ cand_emb[selected].T
            scores = lam * rel[remaining] - (1 - lam) * sims.max(axis=1)
            best = remaining[int(np.argmax(scores))]
        selected.append(best)
        remaining.remove(best)
    return selected


class Index:
    def __init__(self, chunks: list[dict], embed: Embed, emb: np.ndarray | None = None):
        self.chunks, self.embed = chunks, embed
        self.emb = emb if emb is not None else (embed([c["header"] + " [SEP] " + c["text"] for c in chunks])
                                                 if chunks else np.zeros((0, 1), dtype=np.float32))
        self.accepted = [c["accepted"] for c in chunks]
        self.tickers = np.array([c["ticker"] for c in chunks])

    def rows(self, ticker: str, before: datetime, since: datetime | None = None) -> np.ndarray:
        return np.array([i for i in range(len(self.chunks)) if self.tickers[i] == ticker and self.accepted[i] < before
                         and (since is None or self.accepted[i] >= since)], dtype=int)

    def search(self, query: str, ticker: str, before: datetime, since: datetime | None = None,
               k: int | None = None) -> dict:
        """The forecast-date rule is applied here: nothing accepted at or after `before` is ever scored."""
        k = k or SETTINGS.k_final
        rows = self.rows(ticker, before, since)
        if len(rows) == 0:
            return {"status": "no documents in this date range", "results": []}
        q = self.embed([query])[0]
        sims = self.emb[rows] @ q
        order = rows[np.argsort(-sims)[:SETTINGS.k_candidates]]
        rel = self.emb[order] @ q
        if float(rel[0]) < SETTINGS.min_score:
            return {"status": f"nothing relevant found (best score {float(rel[0]):.2f})", "results": []}
        picked = mmr_select(rel, self.emb[order], k, SETTINGS.mmr_lambda)
        return {"status": "ok", "results": [{**self.chunks[order[j]], "row": int(order[j]),
                                             "score": round(float(rel[j]), 3)} for j in picked]}

    def filing_candidates(self, ticker: str, before: datetime, since: datetime) -> list[dict]:
        """One passage per filing in the window: the one closest to the filing's own subject."""
        by_filing: dict[str, list[int]] = {}
        for i in self.rows(ticker, before, since):
            if not is_boilerplate(self.chunks[i]["text"]):
                by_filing.setdefault(self.chunks[i]["filing"], []).append(int(i))
        out, refs = [], {}
        for rows in by_filing.values():
            ref = reference_text(self.chunks[rows[0]])
            if ref not in refs:
                refs[ref] = self.embed([ref])[0]
            sims = self.emb[rows] @ refs[ref]
            j = int(np.argmax(sims))
            if sims[j] >= SETTINGS.filing_min_score:
                out.append({**self.chunks[rows[j]], "row": rows[j], "score": round(float(sims[j]), 3)})
        return out

    def select_evidence(self, candidates: list[dict], k: int | None = None) -> list[dict]:
        """Pool candidates, keep the best passage per filing, MMR to k, newest first."""
        k = k or SETTINGS.k_final
        per_filing: dict[str, dict] = {}
        for r in candidates:
            if r["filing"] not in per_filing or r["score"] > per_filing[r["filing"]]["score"]:
                per_filing[r["filing"]] = r
        cands = sorted(per_filing.values(), key=lambda r: -r["score"])
        if not cands:
            return []
        picked = mmr_select([r["score"] for r in cands], self.emb[[r["row"] for r in cands]], k, SETTINGS.mmr_lambda)
        return sorted((cands[j] for j in picked), key=lambda r: (r["accepted"], r["score"]), reverse=True)


def reference_text(chunk: dict) -> str:
    if chunk["form"] in ("10-Q", "10-K"):
        return REPORT_TOPICS
    return ITEM_TITLES.get(chunk["event"].replace("8-K ", ""), "company announcement")
