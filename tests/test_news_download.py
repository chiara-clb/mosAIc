"""The slow news downloader: the calls are exactly the ones a forecast makes, saved days are skipped,
failures are counted, and the spacing is the one asked for."""

from datetime import UTC, date, datetime

import mosaic.news as news
from mosaic.news import Gdelt, news_days
from mosaic.news_download import download, news_days_between, tone_dates


def fake(calls, refuse=()):
    def http(params):
        calls.append(params)
        if params["mode"] == "artlist" and params["startdatetime"][:8] in refuse:
            raise RuntimeError("Please limit requests")
        return {"articles": [{"title": "Amazon news", "seendate": params["startdatetime"]}]} \
            if params["mode"] == "artlist" else {"timeline": []}
    return http


def test_days_cover_the_week_before_the_first_date_up_to_the_last():
    days = news_days_between(date(2026, 7, 1), date(2026, 7, 8))
    assert days[0][0] == datetime(2026, 6, 24, tzinfo=UTC) and days[-1][0] == datetime(2026, 7, 7, tzinfo=UTC)
    assert len(days) == 14


def test_tone_dates_are_wednesdays_plus_extra_dates():
    got = tone_dates(date(2026, 7, 1), date(2026, 7, 15), [date(2026, 7, 30)])
    assert [d.day for d in got] == [1, 8, 15, 30]


def test_a_forecast_after_the_download_needs_no_new_call(tmp_path):
    calls = []
    g = Gdelt(tmp_path / "c.json", http_get=fake(calls), spacing=0, offline=False)
    download(["AMZN"], date(2026, 7, 1), date(2026, 7, 15), g, log=lambda m: None)
    offline = Gdelt(tmp_path / "c.json", http_get=fake(asked := []), spacing=0, offline=True)
    arts, tone = offline.fetch("Amazon.com, Inc.", "AMZN", datetime(2026, 7, 8, tzinfo=UTC))
    assert len(arts["articles"]) == len(news_days(datetime(2026, 7, 8, tzinfo=UTC))) and tone == {"timeline": []}
    assert asked == []


def test_saved_days_are_skipped_on_a_second_run(tmp_path):
    calls = []
    g = Gdelt(tmp_path / "c.json", http_get=fake(calls), spacing=0, offline=False)
    download(["AMZN"], date(2026, 7, 1), date(2026, 7, 8), g, log=lambda m: None)
    first = len(calls)
    g2 = Gdelt(tmp_path / "c.json", http_get=fake(calls), spacing=0, offline=False)
    rows = download(["AMZN"], date(2026, 7, 1), date(2026, 7, 8), g2, log=lambda m: None)
    assert first == 14 + 2 and len(calls) == first          # 14 days and 2 Wednesdays, nothing asked again
    assert rows == [{"ticker": "AMZN", "days": 14, "headlines": 14, "failed calls": 0}]


def test_refused_days_are_counted_and_not_saved(tmp_path):
    calls = []
    g = Gdelt(tmp_path / "c.json", http_get=fake(calls, refuse={"20260630"}), spacing=0, retries=0, offline=False)
    lines = []
    rows = download(["AMZN"], date(2026, 7, 1), date(2026, 7, 1), g, log=lines.append)
    assert rows[0]["failed calls"] == 1 and any("failed, run again later" in x for x in lines)
    g2 = Gdelt(tmp_path / "c.json", http_get=fake(again := []), spacing=0, offline=False)
    download(["AMZN"], date(2026, 7, 1), date(2026, 7, 1), g2, log=lambda m: None)
    assert [c["startdatetime"][:8] for c in again] == ["20260630"]          # only the refused day is asked again


def test_calls_wait_the_spacing_asked_for(tmp_path, monkeypatch):
    waits = []
    monkeypatch.setattr(news.time, "sleep", waits.append)
    g = Gdelt(tmp_path / "c.json", http_get=fake([]), spacing=15, offline=False)
    download(["NVDA"], date(2026, 7, 1), date(2026, 7, 1), g, log=lambda m: None)
    assert len(waits) >= 7 and all(w >= 14 for w in waits)
