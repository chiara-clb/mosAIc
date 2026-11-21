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
    assert calls == ["artlist"] * 7 + ["timelinetone"]                # the second run used the cache


def test_a_refused_gdelt_call_is_retried_and_not_cached(tmp_path, capsys):
    calls = []

    def refuse_then_answer(params):
        calls.append(params["mode"])
        if len(calls) == 1:
            raise RuntimeError("GDELT did not return data: Please limit requests to one every 5 seconds")
        return ARTICLES if params["mode"] == "artlist" else {"timeline": []}

    g = Gdelt(tmp_path / "cache.json", http_get=refuse_then_answer, spacing=0)
    assert len(news_items("Intel Corporation", "INTC", AS_OF, g, embed)) == 1      # the retry worked
    assert g.errors == []


def test_a_gdelt_failure_is_reported_and_tried_again_next_run(tmp_path, capsys):
    def down(params):
        raise ConnectionError("no network")

    g = Gdelt(tmp_path / "cache.json", http_get=down, spacing=0, retries=1)
    assert news_items("Intel Corporation", "INTC", AS_OF, g, embed) == []
    assert len(g.errors) == 8 and "no network" in g.errors[0]
    assert "GDELT news could not be fetched" in capsys.readouterr().err

    calls = []
    def up(params):
        calls.append(params["mode"])
        return ARTICLES if params["mode"] == "artlist" else {"timeline": []}

    g2 = Gdelt(tmp_path / "cache.json", http_get=up, spacing=0)
    assert len(news_items("Intel Corporation", "INTC", AS_OF, g2, embed)) == 1
    assert calls == ["artlist"] * 7 + ["timelinetone"]             # the failure was not cached


def test_news_is_asked_for_one_whole_day_at_a_time(tmp_path):
    from mosaic.news import news_days
    asked = []

    def http(params):
        if params["mode"] == "artlist":
            asked.append((params["startdatetime"], params["enddatetime"]))
            day = params["startdatetime"][:8]
            return {"articles": [{"url": f"u{day}", "title": f"Intel shares {day}", "seendate": day + "T120000Z"},
                                 {"url": "same-everywhere", "title": "Intel repeat", "seendate": "20260705T130000Z"}]}
        return {}

    as_of = datetime(2026, 7, 15, tzinfo=timezone.utc)
    arts, _ = Gdelt(tmp_path / "c.json", http_get=http, spacing=0).fetch("Intel Corporation", "INTC", as_of)
    assert asked[0] == ("20260708000000", "20260708235959")
    assert asked[-1] == ("20260714000000", "20260714235959")          # nothing from the forecast day itself
    assert len(asked) == 7 and len(news_days(as_of)) == 7 and len(news_days(as_of, 14)) == 14
    assert len(news_days(as_of, 30)) == 14                             # never more than the search window
    assert len(arts["articles"]) == 8                                  # 7 days + one article seen every day, once


def test_two_programs_sharing_the_cache_keep_each_others_answers(tmp_path):
    f = tmp_path / "cache.json"
    a = Gdelt(f, http_get=lambda p: {"from": "a"}, spacing=0)
    b = Gdelt(f, http_get=lambda p: {"from": "b"}, spacing=0)      # both opened before either wrote
    a({"q": 1})
    b({"q": 2})
    a({"q": 3})
    fresh = Gdelt(f, http_get=lambda p: {"from": "new"}, spacing=0)
    assert fresh({"q": 1}) == {"from": "a"} and fresh({"q": 2}) == {"from": "b"} and fresh({"q": 3}) == {"from": "a"}
    assert not list(tmp_path.glob("*.tmp"))


def test_a_broken_cache_file_is_treated_as_empty(tmp_path):
    f = tmp_path / "cache.json"
    f.write_text('{"half written')
    assert Gdelt(f, http_get=lambda p: {"ok": 1}, spacing=0)({"q": 1}) == {"ok": 1}


def test_refusals_and_waits_are_reported_live(tmp_path):
    said, calls = [], []

    def refuse_once(params):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("429 Too Many Requests")
        return {"articles": []}

    g = Gdelt(tmp_path / "c.json", http_get=refuse_once, spacing=0)
    g.fetch("Intel Corporation", "INTC", AS_OF, progress=said.append)
    assert said[0].startswith("News: ") and said[0].endswith("downloading)")
    assert said[1].startswith("   GDELT refused (RuntimeError: 429") and "attempt 2 of 2" in said[1]
    said.clear()
    g.fetch("Intel Corporation", "INTC", AS_OF, progress=said.append)
    assert all(s.endswith("saved)") for s in said)                       # second time, all from the cache


def test_the_pace_slows_after_a_refusal_and_recovers(tmp_path):
    calls = []

    def flaky(params):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("429")
        return {}

    g = Gdelt(tmp_path / "c.json", http_get=flaky, spacing=0)      # spacing 0: no real sleeping in the test
    g._pace = 5.0
    g({"q": 1})
    assert g._pace == 8.0                     # doubled to 10 on the refusal, then x0.8 on the success


def test_story_novelty_compares_with_earlier_headlines_and_filings():
    from mosaic.news import story_novelty

    def embed(texts):                       # two topics: contracts and the chief executive
        return np.array([[1.0, 0.0] if "contract" in t else [0.0, 1.0] for t in texts])

    story = {"articles": [{"title": "Intel wins contract"}, {"title": "Intel contract with retailer"}]}
    assert story_novelty(story, [], embed) == 0.5                                    # nothing to compare: unknown
    assert story_novelty(story, ["Intel CEO steps down"], embed) == 1.0              # a new topic
    assert story_novelty(story, ["8-K: Intel signs a contract"], embed) == 0.0       # already in a filing


def test_news_items_measure_novelty_against_filings_before_the_story(tmp_path):
    def embed(texts):
        return np.array([[1.0, 0.0] if "board" in t.lower() else [0.0, 1.0] for t in texts])

    def http(params):
        return ARTICLES if params["mode"] == "artlist" else {"timeline": []}

    filing = {"accepted": datetime(2026, 3, 1, tzinfo=UTC), "text": "Item 5.02 The board accepted the resignation"}
    g = Gdelt(tmp_path / "c.json", http_get=http, spacing=0)
    items = news_items("Intel Corporation", "INTC", AS_OF, g, embed, filings=[filing])
    assert items and items[0]["novelty"] == 0.0              # the board story was already in the 8-K


def test_headlines_count_when_they_use_any_usual_name_of_the_company():
    from mosaic.news import names_company, news_query
    assert names_company("Google is making it easier to switch from iPhone", "Alphabet Inc.", "GOOGL")
    assert names_company("Instagram adds a new feature", "Meta Platforms, Inc.", "META")
    assert not names_company("Johnson Controls beats estimates", "Johnson & Johnson", "JNJ")      # not just "johnson"
    assert names_company("J&J wins talc appeal", "Johnson & Johnson", "JNJ")
    assert not names_company("Apple v. Samsung goes to court", "Visa Inc.", "V")                   # a lone "V" is not Visa
    assert names_company("Visa shares rise", "Visa Inc.", "V")
    assert names_company("Acme Robotics wins a deal", "Acme Robotics Inc.", "ACME")                # outside the list
    assert news_query("Apple Inc.", "AAPL").startswith('"Apple" (shares')                         # cached queries unchanged
    assert news_query("Alphabet Inc.", "GOOGL").startswith('("Alphabet" OR "Google" OR "YouTube")')


def test_offline_news_answers_from_the_saved_calls_and_never_asks_gdelt(tmp_path):
    from mosaic.news import Gdelt
    asked = []
    saved = Gdelt(cache_file=tmp_path / "c.json", http_get=lambda p: asked.append(p) or {"articles": [1]}, spacing=0)
    saved({"query": "a"})                                         # saved by an earlier online run
    off = Gdelt(cache_file=tmp_path / "c.json", http_get=lambda p: asked.append(p) or {}, spacing=0, offline=True)
    assert off({"query": "a"}) == {"articles": [1]} and off({"query": "b"}) == {}
    assert len(asked) == 1 and off.errors == []


def test_offline_news_can_be_set_from_the_environment(monkeypatch, tmp_path):
    from mosaic.news import Gdelt
    monkeypatch.setenv("MOSAIC_NEWS_OFFLINE", "1")
    assert Gdelt(cache_file=tmp_path / "c.json").offline is True
    monkeypatch.delenv("MOSAIC_NEWS_OFFLINE")
    assert Gdelt(cache_file=tmp_path / "c.json").offline is False
