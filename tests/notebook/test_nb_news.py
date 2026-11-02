import numpy as np
import pytest

from tests.notebook.core import load_core
from tests.notebook.helpers import chunk, dt, unit

ns = load_core()
AS_OF = dt("2026-08-01")


def art(title, when, domain="reuters.com"):
    return {"title": title, "seendate": when, "domain": domain, "url": f"https://{domain}/{abs(hash(title))}"}


def test_search_names_drop_legal_suffixes():
    assert ns["news_name"]("Apple Inc.", "AAPL") == "Apple"
    assert ns["news_name"]("NIKE, Inc.", "NKE") == "NIKE"
    assert ns["news_name"]("INTEL CORP", "INTC") == "INTEL"
    assert ns["news_name"]("Exxon Mobil Corporation", "XOM") == "Exxon Mobil"
    assert 'sourcelang:english' in ns["news_query"]("Apple Inc.", "AAPL")


def test_articles_before_the_forecast_date_only_and_no_option_talk():
    payload = {"articles": [art("Apple shares fall after lawsuit", "20260730T120000Z"),
                            art("Apple stock: options traders bet on a big move", "20260730T130000Z"),
                            art("Apple tops estimates", "20260801T090000Z"),            # after the forecast date
                            art("", "20260729T120000Z"),
                            {"title": "bad date", "seendate": "yesterday"}]}
    out = ns["parse_articles"](payload, AS_OF)
    assert [a["title"] for a in out] == ["Apple shares fall after lawsuit"]
    assert ns["parse_articles"](None, AS_OF) == [] == ns["parse_articles"]({}, AS_OF)


def test_tone_shift_compares_recent_with_before():
    payload = {"timeline": [{"series": "Average Tone", "data": [
        {"date": "20260725T000000Z", "value": -3.0}, {"date": "20260728T000000Z", "value": -1.0},
        {"date": "20260601T000000Z", "value": 0.5}, {"date": "20260802T000000Z", "value": 9.0}]}]}
    assert ns["tone_shift"](payload, AS_OF) == {"recent": -2.0, "before": 0.5}
    assert ns["tone_shift"]({}, AS_OF) is None
    assert "-2.0" in ns["tone_text"]({"recent": -2.0, "before": 0.5})


def embed_by_topic(title):
    t = title.lower()
    if "lawsuit" in t or "sued" in t:
        return unit(1, 0, 0)
    if "cfo" in t:
        return unit(0, 1, 0)
    return unit(0, 0, 1)


def articles():
    return ns["parse_articles"]({"articles": [
        art("Apple sued over App Store fees", "20260720T100000Z"),
        art("New lawsuit hits Apple", "20260721T100000Z", "bloomberg.com"),
        art("Apple lawsuit could cost billions", "20260722T100000Z"),
        art("Apple names new CFO", "20260729T200000Z"),
        art("Apple CFO change surprises analysts", "20260730T100000Z", "wsj.com"),
        art("Apple launches new colours", "20260725T100000Z")]}, AS_OF)


def test_stories_group_headlines_and_need_two_articles():
    stories = ns["build_stories"](articles(), embed_by_topic)
    assert [len(s["articles"]) for s in stories] == [3, 2]       # the single product headline is dropped
    assert stories[0]["first_seen"] == dt("2026-07-20T10:00:00")
    assert "bloomberg.com" in stories[0]["domains"]


def test_same_story_rule_links_coverage_and_keeps_other_stories():
    cfo_filing = dict(chunk("AAPL-8-K-cfo", form="8-K", event="8-K 5.02", filing="E1",
                            accepted="2026-07-29T20:30:00", text="Board appointed a new CFO."), row=0, score=0.4)
    idx = ns["Index"]([cfo_filing], np.array([unit(0, 1, 0)], dtype=np.float32))
    stories = ns["build_stories"](articles(), embed_by_topic)
    items, linked = ns["apply_news"](idx, [cfo_filing], stories, "AAPL", "Apple Inc.", {"recent": -2, "before": 0})
    assert list(linked) == ["AAPL-8-K-cfo"] and linked["AAPL-8-K-cfo"]["articles"] == 2
    assert len(items) == 1 and items[0]["form"] == "News" and "lawsuit" in items[0]["text"].lower()
    assert items[0]["accepted"] < AS_OF.isoformat()


def test_far_apart_stories_are_not_linked():
    old = dict(chunk("old", form="8-K", event="8-K 5.02", filing="E0", accepted="2026-07-01T20:30:00"), row=0, score=0.4)
    idx = ns["Index"]([old], np.array([unit(0, 1, 0)], dtype=np.float32))
    items, linked = ns["apply_news"](idx, [old], ns["build_stories"](articles(), embed_by_topic), "AAPL", "Apple", None)
    assert linked == {} and len(items) == 2


def test_at_most_two_news_items():
    many = []
    for k in range(4):
        v = np.zeros(4)
        v[k] = 1
        many.append({"articles": [{"title": f"story {k}", "seen": dt("2026-07-20"), "domain": "x", "url": "u"}] * 3,
                     "first_seen": dt("2026-07-20"), "vec": v, "domains": ["x"]})
    idx = ns["Index"]([], np.zeros((0, 4), dtype=np.float32))
    items, _ = ns["apply_news"](idx, [], many, "AAPL", "Apple", None)
    assert len(items) == ns["NEWS_MAX_STORIES"] == 2


def test_news_items_have_neutral_novelty_and_coverage_in_the_prompt():
    item = {"form": "News", "id": "n1"}
    assert ns["novelty"](None, item) == (0.5, None)
    filing = dict(chunk("f1", accepted="2026-07-29"), row=0,
                  news={"articles": 12, "headlines": ["Apple names new CFO"], "tone": {"recent": -1.5, "before": 0.2}})
    msg = ns["item_message"](filing, AS_OF)
    assert "News coverage: 12 articles" in msg and '"Apple names new CFO"' in msg and "-1.5" in msg
