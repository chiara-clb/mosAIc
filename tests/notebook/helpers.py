"""Small fixtures shared by the notebook tests."""

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

UTC = timezone.utc


def dt(s):
    return datetime.fromisoformat(s).replace(tzinfo=UTC)


def chunk(cid, ticker="AAPL", accepted="2026-07-31", filing="F1", form="10-Q", section="Item 3. Legal Proceedings",
          event="10-Q", text="text"):
    return {"id": cid, "ticker": ticker, "company": ticker, "form": form, "section": section,
            "section_key": section.split(".")[0].lower(), "event": event,
            "accepted": dt(accepted).isoformat(), "filing": filing, "url": f"https://x/{filing}/doc.htm",
            "text": text, "header": f"{ticker} | {form} | {section}"}


def unit(*v):
    v = np.array(v, dtype=float)
    return v / np.linalg.norm(v)


def prices(n=700, daily_sigma=0.01, start="2023-01-02", seed=0, drift=0.0):
    rng = np.random.default_rng(seed)
    r = rng.normal(drift, daily_sigma, n - 1)
    px = 100 * np.exp(np.concatenate([[0.0], np.cumsum(r)]))
    return pd.Series(px, index=pd.bdate_range(start, periods=n))
