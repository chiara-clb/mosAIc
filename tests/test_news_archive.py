import json
from datetime import datetime, timezone

from mosaic.news import build_stories, parse_articles
from mosaic.news_archive import ArchiveNews

UTC = timezone.utc


def write_event(folder, ticker="GOOGL", articles=None):
    (folder / "events").mkdir(parents=True, exist_ok=True)
    articles = articles or [
        {"url": "u1", "title": "Alphabet shares rise ahead of Q2 results | The Motley Fool", "domain": "fool.com",
         "capture_ts": "20260720120000"},
        {"url": "u2", "title": "Alphabet stock climbs before Q2 results", "domain": "cnbc.com", "capture_ts": "20260721090000"},
        {"url": "u3", "title": "Alphabet results day: what to watch", "domain": "x.com", "capture_ts": "20260722150000"},
        {"url": "u1", "title": "duplicate", "domain": "fool.com", "capture_ts": "20260720130000"},
    ]
    (folder / "events" / f"{ticker}_2026-07-22.json").write_text(json.dumps({
        "window_start_utc": "2026-06-22T20:00:00+00:00", "cutoff_utc": "2026-07-22T20:00:00+00:00",
        "stock": {"articles": articles}, "market": {"articles": []}}))


def test_covers_only_dates_inside_a_saved_window(tmp_path):
    write_event(tmp_path)
    a = ArchiveNews(tmp_path)
    assert a.covers("GOOGL", datetime(2026, 7, 22, tzinfo=UTC))
    assert a.covers("googl", datetime(2026, 7, 23, tzinfo=UTC))           # the morning after results
    assert not a.covers("GOOGL", datetime(2026, 8, 5, tzinfo=UTC))
    assert not a.covers("AAPL", datetime(2026, 7, 22, tzinfo=UTC))


def test_fetch_keeps_only_articles_captured_before_the_date(tmp_path):
    write_event(tmp_path)
    arts, tone = ArchiveNews(tmp_path).fetch("Alphabet Inc.", "GOOGL", datetime(2026, 7, 22, tzinfo=UTC))
    titles = [a["title"] for a in arts["articles"]]
    assert titles == ["Alphabet shares rise ahead of Q2 results", "Alphabet stock climbs before Q2 results"]
    assert tone == {}                                                     # no tone in the archive


def test_archive_articles_go_through_the_same_filters_and_stories(tmp_path):
    write_event(tmp_path)
    as_of = datetime(2026, 7, 23, tzinfo=UTC)
    arts, _ = ArchiveNews(tmp_path).fetch("Alphabet Inc.", "GOOGL", as_of)
    kept = parse_articles(arts, as_of, "Alphabet Inc.", "GOOGL")
    assert len(kept) == 3

    def embed(texts):
        import numpy as np
        return np.array([[1.0, 0.0] if "Q2" in t else [0.0, 1.0] for t in texts])
    assert len(build_stories(kept, embed)) == 1                           # the two "Q2 results" headlines
