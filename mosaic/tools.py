"""The tools, behind one MCP-style door.

Agents never call a tool function directly. `ToolBox.call(tool, caller, **args)`:
1. refuses a tool that is not on the caller's role card,
2. returns only data published before the forecast date (each tool applies this),
3. logs every call in the run trace, and raises an error when a tool is switched off
   instead of returning an empty result.
"""

from __future__ import annotations

import math
import re
from datetime import timedelta

from mosaic.config import SETTINGS
from mosaic.roles import can_use
from mosaic.trace import Trace
from mosaic.world import World, published


class ToolError(RuntimeError):
    """A tool failed or was switched off. The caller must handle it."""


class RoleError(PermissionError):
    """An agent asked for a tool that is not on its role card."""


INJECTION_PATTERNS = [r"ignore (all )?(your )?previous instructions", r"note to ai", r"you are an ai",
                      r"call \w+ with", r"system prompt"]
STOPWORDS = set("the a an and or of to in on for with by is are was were be been its it this that as at from "
                "not has have had any will which under into than other".split())


def words(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z]+", text.lower()) if len(w) > 2 and w not in STOPWORDS]


def similarity(a: str, b: str) -> float:
    """Cosine similarity of word counts. Enough for the demo world; the live index uses embeddings."""
    ca: dict[str, int] = {}
    cb: dict[str, int] = {}
    for w in words(a):
        ca[w] = ca.get(w, 0) + 1
    for w in words(b):
        cb[w] = cb.get(w, 0) + 1
    dot = sum(v * cb.get(k, 0) for k, v in ca.items())
    na, nb = math.sqrt(sum(v * v for v in ca.values())), math.sqrt(sum(v * v for v in cb.values()))
    return dot / (na * nb) if na and nb else 0.0


def clean_extract(text: str, n_words: int = SETTINGS.research_extract_words) -> tuple[str, bool]:
    """First n words, with instruction-like sentences removed. Returns (extract, whether anything was removed)."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    kept = [s for s in sentences if not any(re.search(p, s, re.I) for p in INJECTION_PATTERNS)]
    ws = " ".join(kept).split()
    return " ".join(ws[:n_words]) + (" ..." if len(ws) > n_words else ""), len(kept) < len(sentences)


class ToolBox:
    """The tools for one forecast run: one world, one forecast date, one trace."""

    def __init__(self, world: World, trace: Trace | None = None, failures=(), extra_roles: dict | None = None):
        self.world = world
        self.as_of = world.as_of
        self.trace = trace or Trace(run_id=f"{world.ticker}-{world.as_of:%Y-%m-%d}", as_of=world.as_of)
        self.failures = set(failures)
        self.extra_roles = extra_roles or {}
        self.branch_log: list[dict] = []
        self.off: set[str] = set()                         # guardrails switched off, for the ablation only
        self.tools = {"market_data": self.market_data, "search_documents": self.search_documents,
                      "search_news": self.search_news, "case_memory": self.case_memory,
                      "branch_log": self.log_branch, "get_passage": self.get_passage}

    # --- the door ---------------------------------------------------------------------------
    def call(self, tool: str, caller: str, **kwargs):
        args = {k: v for k, v in kwargs.items() if k != "record"}
        if tool not in self.tools:
            self.trace.add(caller, "tool_call", tool=tool, args=args, status="unknown tool")
            raise ValueError(f"unknown tool {tool!r}")
        if not can_use(caller, tool, self.extra_roles):
            if "role_check" not in self.off:
                self.trace.add(caller, "tool_call", tool=tool, args=args, status="refused: not in role")
                raise RoleError(f"{caller} may not use {tool}")
        if tool in self.failures:
            self.trace.add(caller, "tool_call", tool=tool, args=args, status="error")
            raise ToolError(f"{tool} is unavailable")
        out = self.tools[tool](**kwargs)
        self.trace.add(caller, "tool_call", tool=tool, args=args, status="ok")
        return out

    @property
    def calls(self) -> list[dict]:
        return self.trace.of_kind("tool_call")

    # --- the forecast-date rule -------------------------------------------------------------
    def visible(self, item: dict, since=None) -> bool:
        t = published(item)
        before = t < self.as_of or "date_rule" in self.off
        return before and (since is None or t >= since)

    def passage(self, item_id: str) -> dict | None:
        """Code-side lookup of a full item, never offered to the research model. None if not yet published."""
        it = self.world.item(item_id)
        return it if it is not None and self.visible(it) else None

    # --- the tools --------------------------------------------------------------------------
    def market_data(self, kind: str):
        w = self.world
        if kind == "prices":
            keep = [i for i, d in enumerate(w.prices["dates"]) if d < self.as_of]
            return {"dates": [w.prices["dates"][i] for i in keep], "stock": [w.prices[w.ticker][i] for i in keep],
                    "market": [w.prices["SPY"][i] for i in keep]}
        if kind == "earnings":
            past = sorted(d for d in w.earnings_dates if d < self.as_of)
            last = past[-1] if past else None
            return {"last_results": last, "next_estimate": last + timedelta(days=91) if last else None,
                    "past_results": past}
        raise ValueError(f"unknown market data {kind!r}")

    def search_documents(self, query: str, k: int = SETTINGS.k_final, list_all: bool = False) -> list[dict]:
        """Filings from the last 30 days matching the query. list_all=True returns every one, at the cut-off score."""
        hook = self.world.extra.get("search_documents")       # the live world searches its embedding index
        if hook:
            return hook(query=query, as_of=self.as_of, k=k, list_all=list_all)
        since = self.as_of - timedelta(days=SETTINGS.fresh_days)
        hits = []
        for f in self.world.filings:
            if not self.visible(f, since):
                continue
            score = SETTINGS.min_score if list_all else similarity(query, f["section"] + " " + f["text"])
            if score >= SETTINGS.min_score:
                extract, removed = clean_extract(f["text"])
                header = f"{f['form']} | {f['section']} | filed {f['filed']:%Y-%m-%d}"
                hits.append({"id": f["id"], "kind": "filing", "score": round(score, 3), "extract": extract,
                             "header": header + (" | instruction-like text removed" if removed else "")})
        return sorted(hits, key=lambda h: -h["score"])[:k]

    def search_news(self, query: str, k: int = SETTINGS.k_final, days: int = SETTINGS.news_days) -> list[dict]:
        since = self.as_of - timedelta(days=days)
        hits = []
        for s in self.world.news:
            if not self.visible(s, since):
                continue
            text = " | ".join(s["headlines"])
            score = similarity(query, text)
            if score >= SETTINGS.min_score:
                hits.append({"id": s["id"], "kind": "news", "score": round(score, 3), "extract": clean_extract(text)[0],
                             "header": f"News | {len(s['headlines'])} articles | first seen "
                                       f"{s['first_seen']:%Y-%m-%d} | tone {s['tone']:+.1f}"})
        return sorted(hits, key=lambda h: -h["score"])[:k]

    def case_memory(self, event: str, reaction: str | None, lag_days: int | None = None) -> dict:
        hook = self.world.extra.get("case_rates")             # the live world counts past events from SEC and prices
        if hook:
            return dict(hook(event, reaction, lag_days), matched=f"{event} / {reaction}")
        cm = self.world.case_memory
        key = (event, reaction)
        return dict(cm[key], matched=f"{event} / {reaction}") if key in cm else dict(cm["*"], matched="all events")

    def log_branch(self, record: dict) -> int:
        self.branch_log.append(record)
        return len(self.branch_log) - 1

    def get_passage(self, item_id: str) -> dict | None:
        it = self.passage(item_id)
        if it is None:
            return None
        return {"id": it["id"], "published": published(it),
                "text": it["text"] if "text" in it else " | ".join(it["headlines"])}
