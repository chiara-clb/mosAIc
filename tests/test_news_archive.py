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


# --- GDELT and the archive together ---------------------------------------------------------------

from mosaic.news import Gdelt  # noqa: E402
from mosaic.news_archive import CombinedNews, news_source  # noqa: E402


def gdelt_with(tmp_path, titles):
    def http(params):
        if params["mode"] != "artlist":
            return {"timeline": [{"data": []}]}
        return {"articles": [{"title": t, "seendate": params["startdatetime"][:8] + "T120000Z", "url": u,
                              "domain": "gdelt.example"} for u, t in titles]}
    return Gdelt(tmp_path / "g.json", http_get=http, spacing=0, offline=False)


def test_combined_keeps_both_sources_each_link_once_and_gdelt_tone(tmp_path):
    write_event(tmp_path)
    g = gdelt_with(tmp_path, [("u1", "Alphabet shares rise ahead of Q2 results"), ("g9", "Google cloud deal")])
    arts, tone = CombinedNews(g, ArchiveNews(tmp_path)).fetch("Alphabet Inc.", "GOOGL", datetime(2026, 7, 22, tzinfo=UTC))
    urls = [a["url"] for a in arts["articles"]]
    assert sorted(set(urls)) == ["g9", "u1", "u2"] and len(urls) == len(set(urls))
    assert tone == {"timeline": [{"data": []}]}


def test_combined_keeps_the_date_rule(tmp_path):
    write_event(tmp_path)
    g = gdelt_with(tmp_path, [])
    arts, _ = CombinedNews(g, ArchiveNews(tmp_path)).fetch("Alphabet Inc.", "GOOGL", datetime(2026, 7, 21, tzinfo=UTC))
    assert [a["url"] for a in arts["articles"]] == ["u1"]          # u2 and u3 were captured after the date


def test_source_is_combined_only_where_the_archive_covers(tmp_path):
    write_event(tmp_path)
    g, a = gdelt_with(tmp_path, []), ArchiveNews(tmp_path)
    assert isinstance(news_source("GOOGL", datetime(2026, 7, 22, tzinfo=UTC), g, a), CombinedNews)
    assert news_source("GOOGL", datetime(2026, 8, 19, tzinfo=UTC), g, a) is g
    assert news_source("AAPL", datetime(2026, 7, 22, tzinfo=UTC), g, a) is g
