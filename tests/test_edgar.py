import json
from datetime import datetime, timezone

from mosaic import edgar

UTC = timezone.utc
FILLER = " ".join(["word"] * 60)

PAGES = {
    "https://data.sec.gov/submissions/CIK0000000042.json": json.dumps({
        "name": "Acme Robotics Inc.",
        "filings": {"recent": {
            "form": ["8-K", "10-Q", "S-8"],
            "filingDate": ["2025-05-28", "2025-05-09", "2025-05-01"],
            "acceptanceDateTime": ["2025-05-28T21:05:00.000Z", "2025-05-09T20:30:00.000Z", "2025-05-01T12:00:00.000Z"],
            "accessionNumber": ["0000042-25-000010", "0000042-25-000009", "0000042-25-000008"],
            "primaryDocument": ["acme-8k.htm", "acme-10q.htm", "s8.htm"],
            "items": ["5.02,9.01", "", ""]},
            "files": []}}),
    "https://www.sec.gov/Archives/edgar/data/42/000004225000010/index.json": json.dumps(
        {"directory": {"item": [{"name": "acme-8k.htm"}, {"name": "ex99-1.htm"}, {"name": "logo.jpg"}]}}),
    "https://www.sec.gov/Archives/edgar/data/42/000004225000010/acme-8k.htm":
        "<html><body><p>Item 5.</p><p>02 Departure of Certain Officers</p><p>On May 27, 2025, Jane Doe "
        "notified the Company of her decision to resign as Chief Financial Officer. " + FILLER + "</p>"
        "<script>var x=1;</script></body></html>",
    "https://www.sec.gov/Archives/edgar/data/42/000004225000010/ex99-1.htm":
        "<html><body><p>Acme announces CFO transition. Options traders priced an implied move of 8%. " + FILLER +
        "</p></body></html>",
    "https://www.sec.gov/Archives/edgar/data/42/000004225000009/acme-10q.htm":
        "<html><body><p>Item 1. Legal Proceedings</p><p>From time to time the Company is involved in legal "
        "proceedings. " + FILLER + "</p><p>Item 1A. Risk Factors</p><p>Risks remain. " + FILLER + "</p></body></html>",
}


class Resp:
    def __init__(self, body):
        self.text = body

    def json(self):
        return json.loads(self.text)


def get(url):
    return Resp(PAGES[url])


def test_list_filings_keeps_only_reports_and_8ks():
    name, rows = edgar.list_filings(42, get)
    assert name == "Acme Robotics Inc." and [r["form"] for r in rows] == ["8-K", "10-Q"]


def test_item_numbers_split_by_html_are_rejoined():
    assert edgar.clean_title("Item 5. 02 Departure of Certain Officers") == "Item 5.02 Departure of Certain Officers"
    assert edgar.section_key("Item 5.02 Departure of Certain Officers") == "departure of certain officers"
    assert edgar.section_key("Item 1A. Risk Factors") == "risk factors"


def test_8k_brings_its_press_release_and_drops_answer_key_sentences():
    _, rows = edgar.list_filings(42, get)
    chunks = edgar.filing_chunks("ACME", "Acme Robotics Inc.", 42, rows[0], get)
    sections = {c["section"] for c in chunks}
    assert sections == {"Item 5.02 Departure of Certain Officers", "Press release"}
    assert all(c["event"] == "8-K 5.02" for c in chunks)
    assert all(c["accepted"] == datetime(2025, 5, 28, 21, 5, tzinfo=UTC) for c in chunks)
    text = " ".join(c["text"] for c in chunks)
    assert "implied move" not in text and "CFO transition" in text and "var x" not in text


def test_build_corpus_respects_the_window_and_keeps_sections():
    _, rows = edgar.list_filings(42, get)
    corpus = edgar.build_corpus("ACME", "Acme Robotics Inc.", 42, rows,
                                datetime(2025, 5, 1, tzinfo=UTC), datetime(2025, 5, 20, tzinfo=UTC), get)
    assert {c["form"] for c in corpus} == {"10-Q"}
    assert {c["section_key"] for c in corpus} == {"legal proceedings", "risk factors"}
    assert len({c["id"] for c in corpus}) == len(corpus)


def test_chunking_and_event_keys():
    chunks = edgar.chunk_words(" ".join(f"w{i}" for i in range(300)), size=150, overlap=30)
    assert len(chunks) == 3 and chunks[1].split()[0] == "w120"
    assert edgar.chunk_words("too short") == []
    assert edgar.event_key("8-K", "9.01") == "8-K other" and edgar.event_key("10-K", "") == "10-K"


def test_sec_getter_needs_a_contact():
    import pytest
    with pytest.raises(ValueError):
        edgar.sec_getter(user_agent="")


def test_sec_pages_are_saved_and_read_back(tmp_path):
    import os
    import time as _time
    from mosaic.edgar import cached_getter

    calls = []

    class Page:
        def __init__(self, text):
            self.text = text

    def get(url):
        calls.append(url)
        return Page('{"n": %d}' % len(calls))

    g = cached_getter(get, tmp_path, max_age_days=7)
    doc = "https://www.sec.gov/Archives/edgar/data/320193/x/index.json"
    lst = "https://data.sec.gov/submissions/CIK0000320193.json"
    assert g(doc).json() == {"n": 1}
    assert g(doc).json() == {"n": 1}                 # second time from disk
    assert g(lst).json() == {"n": 2}
    assert g(lst).text == '{"n": 2}'
    assert len(calls) == 2

    old = _time.time() - 8 * 86400                    # a week later
    for f in tmp_path.iterdir():
        os.utime(f, (old, old))
    assert g(doc).json() == {"n": 1}                 # filed documents never change
    assert g(lst).json() == {"n": 3}                 # the list of filings is fetched again


def test_press_releases_are_found_by_type_when_the_file_name_does_not_say():
    base = "https://www.sec.gov/Archives/edgar/data/1045810/000104581026000099"
    pages = {
        base + "/index.json": json.dumps({"directory": {"item": [
            {"name": "nvda-20260826.htm"}, {"name": "q2fy27pr.htm"}, {"name": "q2fy27cfocommentary.htm"}]}}),
        base + "/0001045810-26-000099-index.htm":
            "<table><tr><th>Seq</th><th>Description</th><th>Document</th><th>Type</th></tr>"
            "<tr><td>1</td><td>8-K</td><td><a href='/Archives/edgar/data/1045810/000104581026000099/nvda-20260826.htm'>"
            "nvda-20260826.htm</a></td><td>8-K</td></tr>"
            "<tr><td>2</td><td>PRESS RELEASE</td><td><a href='/Archives/edgar/data/1045810/000104581026000099/"
            "q2fy27pr.htm'>q2fy27pr.htm</a></td><td>EX-99.1</td></tr>"
            "<tr><td>3</td><td>CFO COMMENTARY</td><td><a href='/Archives/edgar/data/1045810/000104581026000099/"
            "q2fy27cfocommentary.htm'>q2fy27cfocommentary.htm</a></td><td>EX-99.2</td></tr></table>",
    }
    urls = edgar.exhibit_urls(1045810, "0001045810-26-000099", lambda u: Resp(pages[u]))
    assert urls == [base + "/q2fy27pr.htm", base + "/q2fy27cfocommentary.htm"]


def predecessor_get(url):
    """Two registrants for one ticker: the new holding company and the older company with the history."""
    def sub(name, forms, dates, accessions, docs):
        return json.dumps({"name": name, "filings": {"recent": {
            "form": forms, "filingDate": [d[:10] for d in dates], "acceptanceDateTime": dates,
            "accessionNumber": accessions, "primaryDocument": docs, "items": [""] * len(forms)}, "files": []}})
    pages = {
        "https://www.sec.gov/files/company_tickers.json": json.dumps({"0": {"cik_str": 2115436, "ticker": "XOM",
                                                                            "title": "ExxonMobil Holdings Corp"}}),
        "https://data.sec.gov/submissions/CIK0002115436.json": sub(
            "ExxonMobil Holdings Corp", ["8-K", "10-Q"], ["2026-07-07T12:00:00.000Z", "2026-08-03T20:30:00.000Z"],
            ["0002115436-26-000001", "0000034088-26-000099"], ["new-8k.htm", "q.htm"]),
        "https://data.sec.gov/submissions/CIK0000034088.json": sub(
            "EXXON MOBIL CORP", ["10-K", "10-Q"], ["2026-02-18T21:00:00.000Z", "2026-08-03T20:30:00.000Z"],
            ["0000034088-26-000010", "0000034088-26-000099"], ["k.htm", "q.htm"]),
    }
    return Resp(pages[url])


def test_a_ticker_that_moved_keeps_its_predecessors_history():
    name, rows = edgar.company_filings("xom", predecessor_get)
    assert name == "ExxonMobil Holdings Corp"
    assert [r["accessionNumber"] for r in rows] == ["0000034088-26-000010", "0002115436-26-000001",
                                                    "0000034088-26-000099"]          # oldest first, no duplicate
    assert rows[0]["cik"] == 34088 and rows[1]["cik"] == 2115436


def test_a_predecessors_documents_are_fetched_from_its_own_folder():
    seen = []
    row = {"form": "10-K", "acceptanceDateTime": "2026-02-18T21:00:00.000Z", "accessionNumber": "0000034088-26-000010",
           "primaryDocument": "k.htm", "items": "", "cik": 34088}
    edgar.filing_chunks("XOM", "ExxonMobil Holdings Corp", 2115436, row,
                        lambda url: seen.append(url) or Resp("<html><body><p>Item 7. Results " + FILLER + "</p></body></html>"))
    assert seen and all("/edgar/data/34088/" in u for u in seen)


def test_other_tickers_are_unchanged():
    name, rows = edgar.company_filings("ACME", lambda url: Resp(json.dumps({"0": {"cik_str": 42, "ticker": "ACME"}}))
                                       if url.endswith("company_tickers.json") else get(url))
    assert name == "Acme Robotics Inc." and [r["form"] for r in rows] == ["10-Q", "8-K"] and {r["cik"] for r in rows} == {42}
