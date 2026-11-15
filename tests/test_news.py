import zlib
from datetime import datetime, timezone

import numpy as np

from mosaic.news import Gdelt, build_stories, news_items, news_name, parse_articles, tone_shift

UTC = timezone.utc
AS_OF = datetime(2026, 3, 9, tzinfo=UTC)


def embed(texts):
    out = np.zeros((len(texts), 4096), dtype=np.float32)
    for i, t in enumerate(texts):
        for w in t.lower().split():
            out[i, zlib.crc32(w.encode()) % 4096] += 1
    return out / np.linalg.norm(out, axis=1, keepdims=True)


def art(title, day, hour=12):
    return {"title": title, "seendate": f"202603{day:02d}T{hour:02d}0000Z", "domain": "news.example", "url": "u"}


ARTICLES = {"articles": [
    art("Intel board shakeup as Yeary steps down", 6),
    art("Intel board shakeup as Yeary steps down", 6, 13),           # an exact copy
    art("Intel board shakeup: Yeary steps down as chair", 6, 14),
    art("Bitcoin heads for worst week since February", 6),           # does not name the company
    art("Intel call options volume surges ahead of event", 7),        # option talk
    art("Intel names new chair after board shakeup", 9, 15),          # after the forecast date
]}


def test_name_cleaning():
    assert news_name("Intel Corporation", "INTC") == "Intel"
    assert news_name("Alphabet Inc.", "GOOGL", {"GOOGL": "Alphabet OR Google"}) == "Alphabet OR Google"


def test_headlines_are_filtered_and_copies_count_once():
    titles = [a["title"] for a in parse_articles(ARTICLES, AS_OF, "Intel Corporation", "INTC")]
    assert titles == ["Intel board shakeup as Yeary steps down", "Intel board shakeup: Yeary steps down as chair"]


def test_stories_need_two_different_headlines():
    arts = parse_articles(ARTICLES, AS_OF, "Intel Corporation", "INTC")
    stories = build_stories(arts, embed)
    assert len(stories) == 1 and len(stories[0]["articles"]) == 2
    assert build_stories(arts[:1], embed) == []


def test_tone_compares_the_last_14_days_with_the_60_before():
    payload = {"timeline": [{"data": [{"date": "20260301T000000Z", "value": "-3"},
                                      {"date": "20260115T000000Z", "value": "1"},
                                      {"date": "20260310T000000Z", "value": "9"}]}]}
    assert tone_shift(payload, AS_OF) == {"recent": -3.0, "before": 1.0}
    assert tone_shift({}, AS_OF) is None


def test_gdelt_calls_are_cached(tmp_path):
    calls = []

    def http(params):
        calls.append(params["mode"])
        return ARTICLES if params["mode"] == "artlist" else {"timeline": []}

    g = Gdelt(tmp_path / "cache.json", http_get=http, spacing=0)
    items = news_items("Intel Corporation", "INTC", AS_OF, g, embed)
    assert len(items) == 1 and items[0]["event"] == "news" and items[0]["first_seen"] < AS_OF
    Gdelt(tmp_path / "cache.json", http_get=http, spacing=0).fetch("Intel Corporation", "INTC", AS_OF)
    assert calls == ["artlist", "timelinetone"]                       # the second run used the cache
