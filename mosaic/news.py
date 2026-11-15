"""News headlines from GDELT (free): fetched, filtered, grouped into stories, with the tone trend.

Two lessons from the validation run are built in: exact copies of a headline (the same wire story
republished) count once, and only headlines that name the company or its ticker are kept, since GDELT
also returns market round-ups that mention the company in passing. Headlines about options activity are
dropped, because option prices are the answer key.
"""

from __future__ import annotations

import json
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

import numpy as np

from mosaic.config import SETTINGS
from mosaic.edgar import ANSWER_KEY_LEAK

GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
SUFFIXES = r"\b(inc|incorporated|corp|corporation|co|company|ltd|plc|holdings?|group|the|sa|nv|ag)\b"


def news_name(company: str, ticker: str, overrides: dict | None = None) -> str:
    """Search name: an override if given, else the company name without legal suffixes."""
    if overrides and ticker in overrides:
        return overrides[ticker]
    name = re.sub(r"[,.]|/\w+/", " ", company)
    name = re.sub(SUFFIXES, " ", name, flags=re.I)
    return " ".join(name.split()) or ticker


def news_query(company: str, ticker: str) -> str:
    return f'"{news_name(company, ticker)}" (shares OR stock OR earnings OR analyst) sourcelang:english'


def parse_articles(payload: dict | None, as_of: datetime, company: str, ticker: str) -> list[dict]:
    """GDELT article list -> headlines first seen before as_of that name the company, without copies or option talk."""
    name_words = [w for w in news_name(company, ticker).lower().split() if len(w) > 2]
    out, seen_titles = [], set()
    for a in (payload or {}).get("articles", []):
        try:
            seen = datetime.strptime(a["seendate"], "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
        except (KeyError, ValueError):
            continue
        title = " ".join(str(a.get("title", "")).split())
        low = title.lower()
        names_it = bool(re.search(rf"\b{re.escape(ticker.lower())}\b", low)) or (name_words and name_words[0] in low)
        if seen >= as_of or not title or ANSWER_KEY_LEAK.search(title) or not names_it or low in seen_titles:
            continue
        seen_titles.add(low)
        out.append({"title": title, "seen": seen, "domain": a.get("domain", ""), "url": a.get("url", "")})
    return sorted(out, key=lambda a: a["seen"])


def tone_shift(payload: dict | None, as_of: datetime) -> dict | None:
    """Average tone over the last 14 days and over the 60 days before that."""
    points = []
    for series in (payload or {}).get("timeline", []):
        for d in series.get("data", []):
            try:
                points.append((datetime.strptime(d["date"], "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc), float(d["value"])))
            except (KeyError, ValueError):
                continue
    recent_start = as_of - timedelta(days=SETTINGS.news_days)
    base_start = recent_start - timedelta(days=SETTINGS.tone_base_days)
    recent = [v for w, v in points if recent_start <= w < as_of]
    before = [v for w, v in points if base_start <= w < recent_start]
    if not recent:
        return None
    return {"recent": round(float(np.mean(recent)), 2), "before": round(float(np.mean(before)), 2) if before else None}


def build_stories(articles: list[dict], embed: Callable[[list[str]], np.ndarray]) -> list[dict]:
    """Group headlines into stories; a story needs at least two different headlines."""
    stories: list[dict] = []
    vecs = embed([a["title"] for a in articles]) if articles else []
    for a, v in zip(articles, vecs):
        for s in stories:
            c = s["sum"] / np.linalg.norm(s["sum"])
            if float(c @ v) >= SETTINGS.news_story_sim:
                s["articles"].append(a)
                s["sum"] = s["sum"] + v
                break
        else:
            stories.append({"articles": [a], "sum": np.array(v, dtype=float)})
    out = [{"articles": s["articles"], "first_seen": s["articles"][0]["seen"],
            "domains": sorted({a["domain"] for a in s["articles"] if a["domain"]})}
           for s in stories if len(s["articles"]) >= SETTINGS.news_min_articles]
    return sorted(out, key=lambda s: -len(s["articles"]))


def stories_to_items(stories: list[dict], ticker: str, tone: dict | None) -> list[dict]:
    """Stories as world news items, largest first, at most news_max_stories."""
    items = []
    for k, s in enumerate(stories[:SETTINGS.news_max_stories]):
        items.append({"id": f"{ticker}-NEWS-{s['first_seen']:%Y%m%d}-{k}", "first_seen": s["first_seen"],
                      "headlines": [a["title"] for a in s["articles"][:5]], "articles": len(s["articles"]),
                      "domains": s["domains"], "tone": (tone or {}).get("recent", 0.0), "tone_before": (tone or {}).get("before"),
                      "event": "news", "reaction": None, "novelty": SETTINGS.news_novelty})
    return items


class Gdelt:
    """GDELT DOC API calls, cached on disk and spaced 5 seconds apart, as GDELT asks."""

    def __init__(self, cache_file: str | Path | None = None, http_get: Callable[[dict], dict] | None = None,
                 spacing: float | None = None):
        self.cache_file = Path(cache_file or Path(SETTINGS.data_dir) / "gdelt_cache.json")
        self.cache = json.loads(self.cache_file.read_text()) if self.cache_file.exists() else {}
        self.http_get = http_get or self._requests_get
        self.spacing = SETTINGS.gdelt_spacing_seconds if spacing is None else spacing
        self._last = 0.0

    @staticmethod
    def _requests_get(params: dict) -> dict:                 # pragma: no cover - needs internet
        import requests
        r = requests.get(GDELT_URL, params=params, timeout=60)
        return r.json() if r.text.strip().startswith("{") else {}

    def __call__(self, params: dict) -> dict:
        key = json.dumps(params, sort_keys=True)
        if key in self.cache:
            return self.cache[key]
        wait = self.spacing - (time.time() - self._last)
        if wait > 0:
            time.sleep(wait)
        self._last = time.time()
        try:
            payload = self.http_get(params)
        except Exception:
            payload = {}
        self.cache[key] = payload
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        self.cache_file.write_text(json.dumps(self.cache))
        return payload

    def fetch(self, company: str, ticker: str, as_of: datetime) -> tuple[dict, dict]:
        q, fmt = news_query(company, ticker), "%Y%m%d%H%M%S"
        arts = self({"query": q, "mode": "artlist", "format": "json", "maxrecords": 250, "sort": "datedesc",
                     "startdatetime": (as_of - timedelta(days=SETTINGS.news_days)).strftime(fmt),
                     "enddatetime": as_of.strftime(fmt)})
        tone = self({"query": q, "mode": "timelinetone", "format": "json",
                     "startdatetime": (as_of - timedelta(days=SETTINGS.news_days + SETTINGS.tone_base_days)).strftime(fmt),
                     "enddatetime": as_of.strftime(fmt)})
        return arts, tone


def news_items(company: str, ticker: str, as_of: datetime, gdelt: Gdelt, embed) -> list[dict]:
    """The news stories for one forecast, ready to put in the world."""
    arts, tone = gdelt.fetch(company, ticker, as_of)
    return stories_to_items(build_stories(parse_articles(arts, as_of, company, ticker), embed), ticker,
                            tone_shift(tone, as_of))
