"""News saved earlier from the Wayback Machine, used instead of GDELT when it covers the stock and date.

The folder (data/news by default) holds one file per stock and results date, events/<TICKER>_<date>.json,
listing the articles found in the month before the results release, and archive/<id>.json with each
article's text. Each article is dated by the moment the Wayback Machine captured it, which is never
earlier than when it was online, so the forecast-date rule still holds.

The reader returns articles in the same shape as GDELT, so they go through the same filters and story
grouping (news.parse_articles and news.build_stories).
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from mosaic.config import SETTINGS


def _utc(text: str) -> datetime:
    d = datetime.fromisoformat(text)
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)


class ArchiveNews:
    """Same `fetch` as news.Gdelt, reading the saved files. `errors` stays empty: there is no network."""

    def __init__(self, folder: str | Path | None = None):
        self.folder = Path(folder or Path(SETTINGS.data_dir) / "news")
        self.errors: list[str] = []

    def _events(self, ticker: str) -> list[dict]:
        out = []
        for f in sorted((self.folder / "events").glob(f"{ticker.upper()}_*.json")):
            try:
                out.append(json.loads(f.read_text()))
            except (json.JSONDecodeError, OSError):
                continue
        return out

    def covers(self, ticker: str, as_of: datetime) -> bool:
        """True if a saved file spans the news window before as_of."""
        need = as_of - timedelta(days=SETTINGS.news_fetch_days)
        return any(_utc(e["window_start_utc"]) <= need and as_of <= _utc(e["cutoff_utc"]) + timedelta(days=1)
                   for e in self._events(ticker))

    def fetch(self, company: str, ticker: str, as_of: datetime, progress=None) -> tuple[dict, dict]:
        since = as_of - timedelta(days=SETTINGS.news_days)
        articles, seen = [], set()
        for e in self._events(ticker):
            for a in (e.get("stock") or {}).get("articles", []):
                when = datetime.strptime(a["capture_ts"], "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
                if not (since <= when < as_of) or a["url"] in seen:
                    continue
                seen.add(a["url"])
                title = a.get("title", "").split(" | ")[0].strip()        # drop " | The Motley Fool" and the like
                articles.append({"title": title, "seendate": f"{when:%Y%m%dT%H%M%SZ}", "domain": a.get("domain", ""),
                                 "url": a["url"]})
        if progress:
            progress(f"News: {len(articles)} saved articles from the archive")
        return {"articles": articles}, {}
