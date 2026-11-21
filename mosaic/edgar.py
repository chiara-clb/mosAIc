"""SEC EDGAR: list a company's filings, download them, and cut them into dated, searchable chunks.

Every chunk keeps the time the filing was accepted by the SEC, so search can apply the forecast-date
rule. 8-Ks also bring their press-release exhibits (Exhibit 99), where the actual numbers are.
Sentences that quote option-implied volatility or moves are dropped, since those are the answer key.

Network access goes through one `get(url)` function, so everything can be tested with saved pages.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from datetime import datetime, timezone
from typing import Callable

from mosaic.config import SETTINGS

FORMS = {"10-Q", "10-K", "8-K"}
KEYS = ["form", "filingDate", "acceptanceDateTime", "accessionNumber", "primaryDocument", "items"]
ANSWER_KEY_LEAK = re.compile(r"implied vol|implied move|options? (traders|market|activity|volume|pricing)|"
                             r"call options|put options", re.I)
ITEM_RE = re.compile(r"\n\s*(item\s+\d+[a-z]?\.?)\s*([^\n]{0,80})", re.I)


class SavedPage:
    """A page read back from the disk cache, with the two things the code uses from a response."""

    def __init__(self, text: str):
        self.text = text

    def json(self):
        return json.loads(self.text)


def cached_getter(get: Callable[[str], object], cache_dir: str | Path | None = None,
                  max_age_days: float = 7.0) -> Callable[[str], object]:
    """Wrap `get` so every page is saved under data/sec and read from there next time.

    Filed documents (URLs under /Archives/) never change, so they are kept for good. Lists of
    filings and the ticker table do change, so they are fetched again after `max_age_days`.
    """
    root = Path(cache_dir or Path(SETTINGS.data_dir) / "sec")

    def cached(url: str):
        path = root / (hashlib.sha1(url.encode()).hexdigest() + ".txt")
        if path.exists():
            age_days = (time.time() - path.stat().st_mtime) / 86400
            if "/Archives/" in url or age_days < max_age_days:
                return SavedPage(path.read_text(encoding="utf-8"))
        r = get(url)
        root.mkdir(parents=True, exist_ok=True)
        path.write_text(r.text, encoding="utf-8")
        return SavedPage(r.text)
    return cached


def sec_getter(user_agent: str | None = None, pause: float = 0.15, cache_dir: str | Path | None = None,
               use_cache: bool = True) -> Callable[[str], object]:
    """A polite SEC client (well under 10 requests a second), with pages saved on disk.

    The SEC asks for a contact in the User-Agent.
    """
    import requests

    ua = user_agent or SETTINGS.sec_user_agent
    if not ua:
        raise ValueError("Set MOSAIC_SEC_USER_AGENT to 'Your Name your.email@example.com' (SEC rule)")
    headers = {"User-Agent": ua, "Accept-Encoding": "gzip, deflate"}

    def get(url: str):
        time.sleep(pause)
        r = requests.get(url, headers=headers, timeout=30)
        r.raise_for_status()
        return r
    return cached_getter(get, cache_dir) if use_cache else get


# --- listing filings ---------------------------------------------------------------------------

def ticker_to_cik(get) -> dict[str, int]:
    data = get("https://www.sec.gov/files/company_tickers.json").json()
    return {v["ticker"].upper(): int(v["cik_str"]) for v in data.values()}


def _rows(block: dict) -> list[dict]:
    n = len(block["form"])
    return [{k: block.get(k, [""] * n)[i] for k in KEYS} for i in range(n)]


def list_filings(cik: int, get) -> tuple[str, list[dict]]:
    """(company name, every 10-Q, 10-K and 8-K), including the older pages EDGAR keeps in separate files."""
    sub = get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json").json()
    rows = _rows(sub["filings"]["recent"])
    for page in sub["filings"].get("files", []):
        rows += _rows(get(f"https://data.sec.gov/submissions/{page['name']}").json())
    return sub.get("name", ""), [r for r in rows if r["form"] in FORMS]


def filing_base(cik: int, accession: str) -> str:
    return f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession.replace('-', '')}"


def exhibit_urls(cik: int, accession: str, get) -> list[str]:
    """The press-release exhibits (EX-99.x) of a filing.

    Most companies name the file after the exhibit (ex99-1.htm). Some do not (Nvidia: q2fy27pr.htm), so when
    no name says so, the exhibit types are read from the filing's index page.
    """
    base = filing_base(cik, accession)
    idx = get(base + "/index.json").json()
    names = [it["name"] for it in idx["directory"]["item"]]
    by_name = [base + "/" + n for n in names
               if re.search(r"ex-?99", n, re.I) and n.lower().endswith((".htm", ".html", ".txt"))]
    return by_name or exhibits_by_type(base, accession, get)


def exhibits_by_type(base: str, accession: str, get) -> list[str]:
    """Documents whose type is EX-99.x in the filing's index page ({accession}-index.htm)."""
    from bs4 import BeautifulSoup
    try:
        html = get(f"{base}/{accession}-index.htm").text
    except Exception:
        return []
    out = []
    for row in BeautifulSoup(html, "html.parser").find_all("tr"):
        cells = [c.get_text(" ", strip=True) for c in row.find_all("td")]
        link = row.find("a", href=True)
        if link and any(re.fullmatch(r"EX-99(\.\d+)?", c, re.I) for c in cells):
            name = link["href"].rsplit("/", 1)[-1]
            if name.lower().endswith((".htm", ".html", ".txt")):
                out.append(f"{base}/{name}")
    return out


# --- turning a document into chunks ------------------------------------------------------------

def parse_accepted(s: str) -> datetime:
    """e.g. "2024-08-01T16:30:42.000Z" -> timezone-aware UTC datetime."""
    return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc)


def html_to_text(html: str) -> str:
    from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
    import warnings
    warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style"]):
        tag.decompose()
    for tag in soup.find_all(lambda t: t.name and t.name.lower() == "ix:header"):
        tag.decompose()                                  # hidden XBRL data in modern filings
    text = soup.get_text("\n")
    text = re.sub(r"[ \t\xa0]+", " ", text)
    return re.sub(r"\n\s*\n+", "\n", text).strip()


def clean_title(title: str) -> str:
    """'Item 5. 02 Departure...' -> 'Item 5.02 Departure...' (HTML often splits the item number)."""
    title = re.sub(r"\s+", " ", title).strip()
    return re.sub(r"(?i)(item\s+\d+)\.\s+(\d{2})\b", r"\1.\2", title)[:80]


def split_sections(text: str, label: str) -> list[tuple[str, str]]:
    if label == "Press release":
        return [("Press release", text)]
    text = "\n" + text
    matches = list(ITEM_RE.finditer(text))
    if not matches:
        return [("Full text", text.strip())]
    out = []
    for j, m in enumerate(matches):
        end = matches[j + 1].start() if j + 1 < len(matches) else len(text)
        body = text[m.start():end]
        if len(body.split()) > 40:                       # skips table-of-contents entries
            out.append((clean_title(m.group(1) + " " + m.group(2)), body))
    return out or [("Full text", text.strip())]


def section_key(section: str) -> str:
    """'Item 1A. Risk Factors' -> 'risk factors'. Titles match across 10-Q and 10-K; item numbers do not."""
    title = re.sub(r"^\s*item\s+\d+[a-z]?(\.\d{2})?\.?", "", section.strip(), flags=re.I)
    words = re.findall(r"[a-z]+", title.lower())
    if words:
        return " ".join(words[:4])
    m = re.match(r"\s*item\s+(\d+[a-z]?)", section, re.I)
    return f"item {m.group(1).lower()}" if m else section.strip().lower()


def drop_answer_key_sentences(text: str) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return " ".join(s for s in sentences if not ANSWER_KEY_LEAK.search(s))


def chunk_words(text: str, size: int | None = None, overlap: int | None = None) -> list[str]:
    size = size or SETTINGS.chunk_words
    overlap = SETTINGS.overlap_words if overlap is None else overlap
    words, step, out = text.split(), size - overlap, []
    for i in range(0, max(len(words) - overlap, 1), step):
        piece = words[i:i + size]
        if len(piece) >= 30:
            out.append(" ".join(piece))
    return out


def event_key(form: str, items) -> str:
    """Event type for case memory: the 8-K item number, or the report type."""
    if form != "8-K":
        return form
    codes = [c.strip() for c in str(items).split(",") if c.strip() and c.strip() != "9.01"]   # 9.01 = exhibits only
    return "8-K " + (codes[0] if codes else "other")


def filing_chunks(ticker: str, company: str, cik: int, row: dict, get) -> list[dict]:
    """All chunks of one filing (main document, plus press releases for an 8-K)."""
    accepted = parse_accepted(row["acceptanceDateTime"])
    docs = [(filing_base(cik, row["accessionNumber"]) + "/" + row["primaryDocument"], "Main document")]
    if row["form"] == "8-K":
        docs += [(u, "Press release") for u in exhibit_urls(cik, row["accessionNumber"], get)]
    out = []
    for url, label in docs:
        text = drop_answer_key_sentences(html_to_text(get(url).text))
        for section, body in split_sections(text, label):
            for k, piece in enumerate(chunk_words(body)):
                out.append({"id": f"{ticker}-{row['accessionNumber']}-{label[0]}-{len(out)}",
                            "ticker": ticker, "company": company, "form": row["form"], "section": section,
                            "section_key": section_key(section), "event": event_key(row["form"], row.get("items", "")),
                            "accepted": accepted, "filing": row["accessionNumber"], "url": url, "text": piece,
                            "header": f"{company} | {row['form']} | {section} | {accepted.date()}"})
    return out


def build_corpus(ticker: str, company: str, cik: int, rows: list[dict], since: datetime, until: datetime, get) -> list[dict]:
    """Chunks of every filing accepted in [since, until)."""
    corpus = []
    for r in rows:
        if since <= parse_accepted(r["acceptanceDateTime"]) < until:
            corpus += filing_chunks(ticker, company, cik, r, get)
    return corpus
