"""News headlines from GDELT (free): fetched, filtered, grouped into stories, with the tone trend.

Two lessons from the validation run are built in: exact copies of a headline (the same wire story
republished) count once, and only headlines that name the company or its ticker are kept, since GDELT
also returns market round-ups that mention the company in passing. Headlines about options activity are
dropped, because option prices are the answer key.
"""

from __future__ import annotations

import json
import os
import re
import sys
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


# The names headlines actually use for each stock in the universe. Without them the filter keeps only
# headlines with the first word of the legal name, which misses "Google" for Alphabet and is far too broad
# for "Johnson & Johnson" ("johnson") or American Tower ("american").
NEWS_NAMES = {
    "AAPL": ["Apple"], "AMT": ["American Tower"], "AMZN": ["Amazon", "AWS"], "BA": ["Boeing"],
    "CAT": ["Caterpillar"], "CVX": ["Chevron"], "DUK": ["Duke Energy"], "FCX": ["Freeport-McMoRan", "Freeport"],
    "GOOGL": ["Alphabet", "Google", "YouTube"], "GS": ["Goldman Sachs", "Goldman"], "INTC": ["Intel"],
    "JNJ": ["Johnson & Johnson", "J&J"], "JPM": ["JPMorgan", "JP Morgan", "Jamie Dimon"], "KO": ["Coca-Cola", "Coke"],
    "LIN": ["Linde"], "LLY": ["Eli Lilly", "Lilly"], "LMT": ["Lockheed Martin", "Lockheed"],
    "META": ["Meta", "Facebook", "Instagram", "WhatsApp"], "MSFT": ["Microsoft", "Azure"], "NEE": ["NextEra"],
    "NFLX": ["Netflix"], "NKE": ["Nike"], "NVDA": ["Nvidia"], "PG": ["Procter & Gamble", "P&G"], "PLD": ["Prologis"],
    "TSLA": ["Tesla"], "UNH": ["UnitedHealth", "Optum"], "V": ["Visa"], "WMT": ["Walmart"], "XOM": ["Exxon", "ExxonMobil"],
}


def news_terms(company: str, ticker: str) -> list[str]:
    """The names a headline may use for the company: the list above, or the cleaned legal name."""
    return NEWS_NAMES.get(ticker.upper()) or [news_name(company, ticker)]


def news_query(company: str, ticker: str) -> str:
    terms = news_terms(company, ticker)
    names = f'"{terms[0]}"' if len(terms) == 1 else "(" + " OR ".join(f'"{t}"' for t in terms) + ")"
    return f"{names} (shares OR stock OR earnings OR analyst) sourcelang:english"


def names_company(title: str, company: str, ticker: str) -> bool:
    """True if the headline names the company: its ticker, one of its usual names, or (for stocks outside
    the list) the first word of its legal name."""
    low = title.lower()
    if len(ticker) > 1 and re.search(rf"\b{re.escape(ticker.lower())}\b", low):     # "V" alone is too common
        return True
    if ticker.upper() in NEWS_NAMES:
        return any(re.search(rf"(?<![a-z]){re.escape(t.lower())}(?![a-z])", low) for t in NEWS_NAMES[ticker.upper()])
    words = [w for w in news_name(company, ticker).lower().split() if len(w) > 2]
    return bool(words) and words[0] in low


def parse_articles(payload: dict | None, as_of: datetime, company: str, ticker: str) -> list[dict]:
    """GDELT article list -> headlines first seen before as_of that name the company, without copies or option talk."""
    out, seen_titles = [], set()
    for a in (payload or {}).get("articles", []):
        try:
            seen = datetime.strptime(a["seendate"], "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
        except (KeyError, ValueError):
            continue
        title = " ".join(str(a.get("title", "")).split())
        low = title.lower()
        names_it = names_company(title, company, ticker)
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
                 spacing: float | None = None, retries: int = 1, offline: bool | None = None):
        # Offline: answer only from the saved calls and never ask GDELT (MOSAIC_NEWS_OFFLINE=1), so a
        # rerun sees the same news as the run that saved them, without waiting on refusals.
        self.offline = os.environ.get("MOSAIC_NEWS_OFFLINE") == "1" if offline is None else offline
        self.cache_file = Path(cache_file or Path(SETTINGS.data_dir) / "gdelt_cache.json")
        self.cache = self._load()
        self.http_get = http_get or self._requests_get
        self.spacing = SETTINGS.gdelt_spacing_seconds if spacing is None else spacing
        self.retries = retries
        self._pace = self.spacing                 # current seconds between calls; grows after refusals
        self.errors: list[str] = []          # failed calls in this session, so a run can say news was missing
        self.progress = None                  # optional callable(message), told about refusals and waits
        self._last = 0.0

    def _load(self) -> dict:
        try:
            return json.loads(self.cache_file.read_text()) if self.cache_file.exists() else {}
        except (json.JSONDecodeError, OSError):
            return {}

    def _save(self) -> None:
        """Merge with what is on disk (another program may have added answers), then replace the file in one step."""
        self.cache = {**self._load(), **self.cache}
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.cache_file.with_name(f"{self.cache_file.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(self.cache))
        os.replace(tmp, self.cache_file)

    @staticmethod
    def _requests_get(params: dict) -> dict:                 # pragma: no cover - needs internet
        import requests
        r = requests.get(GDELT_URL, params=params, timeout=60)
        r.raise_for_status()
        text = r.text.strip()
        if not text:
            return {}                                        # GDELT's answer when nothing matches
        if not text.startswith("{"):
            raise RuntimeError(f"GDELT did not return data: {text[:120]}")   # e.g. "Please limit requests..."
        return r.json()

    def __call__(self, params: dict) -> dict:
        """One GDELT call. Answers are cached on disk; failures are not, so the next run tries again."""
        key = json.dumps(params, sort_keys=True)
        if key in self.cache:
            return self.cache[key]
        if self.offline:
            return {}                                        # not saved: no news, as when GDELT refused it
        error = ""
        for attempt in range(self.retries + 1):
            # The pace adapts: after a refusal every call waits longer (up to 40 s), and it speeds up again
            # as calls succeed. A refused call is retried once after 60 s, then skipped for a later run.
            wait = (self._pace if attempt == 0 else max(60.0 * attempt, self._pace)) - (time.time() - self._last)
            if attempt and self.progress:
                self.progress(f"   GDELT refused ({error[:60]}); waiting {max(wait, 0):.0f} s, "
                              f"attempt {attempt + 1} of {self.retries + 1}")
            if wait > 0 and self.spacing > 0:
                time.sleep(wait)
            self._last = time.time()
            try:
                payload = self.http_get(params)
            except Exception as e:                           # network error or a refusal
                error = f"{type(e).__name__}: {e}"
                self._pace = min(self._pace * 2, 40.0)
                continue
            self._pace = max(self._pace * 0.8, self.spacing)
            self.cache[key] = payload
            self._save()
            return payload
        self.errors.append(f"{params.get('mode')}: {error}")
        if self.progress:
            self.progress("   Skipped: GDELT kept refusing. A later run will try this day again.")
        print(f"Warning: GDELT news could not be fetched ({error}). The forecast will have no news.", file=sys.stderr)
        return {}

    def fetch(self, company: str, ticker: str, as_of: datetime, progress=None) -> tuple[dict, dict]:
        """Headlines from the last `news_fetch_days` days before as_of, asked for one day at a time.

        GDELT returns at most 250 articles per call, newest first. For a widely covered company one
        call over two weeks only reaches back a day or so, so each day is a separate call. Days are
        whole UTC days, so the calls are cached and shared by forecasts on nearby dates.
        """
        q, fmt = news_query(company, ticker), "%Y%m%d%H%M%S"
        articles, seen_urls = [], set()
        days = news_days(as_of)
        self.progress = progress
        for n, (start, end) in enumerate(days, 1):
            params = {"query": q, "mode": "artlist", "format": "json", "maxrecords": 250, "sort": "datedesc",
                      "startdatetime": start.strftime(fmt), "enddatetime": end.strftime(fmt)}
            if progress:
                saved = json.dumps(params, sort_keys=True) in self.cache
                progress(f"News: {start:%d %b} ({n} of {len(days)} days, {'saved' if saved else 'downloading'})")
            day = self(params)
            for a in (day or {}).get("articles", []):
                key = (a.get("url"), a.get("title"), a.get("seendate"))       # the same article returned twice
                if key not in seen_urls:
                    seen_urls.add(key)
                    articles.append(a)
        tone = self({"query": q, "mode": "timelinetone", "format": "json",
                     "startdatetime": (as_of - timedelta(days=SETTINGS.news_days + SETTINGS.tone_base_days)).strftime(fmt),
                     "enddatetime": as_of.strftime(fmt)})
        return {"articles": articles}, tone


def news_days(as_of: datetime, n: int | None = None) -> list[tuple[datetime, datetime]]:
    """Whole UTC days before as_of's date: (00:00:00, 23:59:59) for each of the last `n` days.

    By default `news_fetch_days` (7), set with MOSAIC_NEWS_FETCH_DAYS, never more than `news_days`.
    """
    n = min(n or SETTINGS.news_fetch_days, SETTINGS.news_days)
    last = as_of.astimezone(timezone.utc).date() if as_of.tzinfo else as_of.date()
    days = []
    for k in range(n, 0, -1):
        d = last - timedelta(days=k)
        start = datetime(d.year, d.month, d.day, tzinfo=timezone.utc)
        days.append((start, start + timedelta(hours=23, minutes=59, seconds=59)))
    return days


def story_novelty(story: dict, earlier_texts: list[str], embed) -> float:
    """How new a story is: 1 minus its closest match among earlier headlines and filings, rescaled like filings.

    Similarity of 0.5 or less counts as fully new, 0.95 or more as already known. With nothing earlier to
    compare with, the answer is unknown and stays at news_novelty (0.5).
    """
    if not earlier_texts:
        return SETTINGS.news_novelty
    titles = [a["title"] for a in story["articles"]]
    vecs = np.asarray(embed(titles + earlier_texts), dtype=float)
    centre = vecs[:len(titles)].mean(axis=0)
    centre = centre / (np.linalg.norm(centre) or 1.0)
    sim = float(np.max(vecs[len(titles):] @ centre))
    lo, hi = SETTINGS.novelty_sim_range
    return float(min(max((hi - sim) / (hi - lo), 0.0), 1.0))


def news_items(company: str, ticker: str, as_of: datetime, gdelt: Gdelt, embed, progress=None,
               filings: list[dict] | None = None) -> list[dict]:
    """The news stories for one forecast, ready to put in the world.

    `filings`: the company's filing passages ({"accepted", "text"}); each story's novelty is measured
    against them and against the headlines seen before it.
    """
    arts, tone = gdelt.fetch(company, ticker, as_of, progress)
    kept = parse_articles(arts, as_of, company, ticker)
    stories = build_stories(kept, embed)
    items = stories_to_items(stories, ticker, tone_shift(tone, as_of))
    for item, story in zip(items, stories):
        start = story["first_seen"]
        mine = {a["title"] for a in story["articles"]}
        earlier = [a["title"] for a in kept if a["seen"] < start and a["title"] not in mine]
        earlier += [" ".join(f["text"].split()[:60]) for f in filings or []
                    if start - timedelta(days=SETTINGS.fresh_days) <= f["accepted"] < start]
        item["novelty"] = story_novelty(story, earlier, embed)
    return items
