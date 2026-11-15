"""The fixed shapes agents hand to each other, checked at the boundary."""

from __future__ import annotations

from mosaic.config import OUTCOMES, SETTINGS
from mosaic.ledger import EFFECTS

ROUTES = ("finalise", "research", "synthesis", "fallback")


def problems_baseline(b: dict) -> list[str]:
    out = [f"missing {k}" for k in OUTCOMES if k not in b]
    if not out:
        if abs(sum(b[k] for k in OUTCOMES) - 100) > 0.05:
            out.append("does not sum to 100")
        out += [f"{k} below {SETTINGS.min_probability}%" for k in OUTCOMES if b[k] < SETTINGS.min_probability - 1e-9]
    return out


def problems_evidence(e: dict) -> list[str]:
    out = [f"missing {k}" for k in ("id", "kind", "header", "published", "extract", "full_text") if not e.get(k)]
    if e.get("kind") not in ("filing", "news"):
        out.append("kind must be filing or news")
    return out


def problems_reading(r: dict) -> list[str]:
    out = []
    if r.get("effect") not in EFFECTS:
        out.append(f"unknown effect {r.get('effect')!r}")
    elif r["effect"] != "none" and r.get("size") not in SETTINGS.strength_points:
        out.append(f"unknown size {r.get('size')!r}")
    if r.get("effect") != "none" and not str(r.get("quote", "")).strip():
        out.append("no quote")
    return out


def problems_line(line: dict) -> list[str]:
    return problems_reading(line) + [f"missing {k}" for k in ("source", "shift") if k not in line]


def problems_verdict(v: dict) -> list[str]:
    out = [] if isinstance(v.get("ok"), bool) else ["ok must be true or false"]
    if v.get("route") not in ROUTES:
        out.append(f"unknown route {v.get('route')!r}")
    return out


CHECKS = {"baseline": problems_baseline, "evidence": problems_evidence, "reading": problems_reading,
          "line": problems_line, "verdict": problems_verdict}


def handoff(kind: str, obj: dict) -> dict:
    """Check a handoff; raise instead of passing on a malformed object."""
    p = CHECKS[kind](obj)
    if p:
        raise ValueError(f"bad {kind} handoff: {'; '.join(p)}")
    return obj
