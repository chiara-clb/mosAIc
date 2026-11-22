"""Rescore saved Tree of Thought readings with other scorers, without new model calls.

`python -m mosaic.evaluate ... --variants "baseline,ToT"` saves every reading the tree chose from
(readings.jsonl). This module reruns the same beam search on those readings with a different score for
each reading, applies the same ledger rule, and scores the result against the option-implied
probabilities and against what happened.

The scorers were fixed before any result was seen (22 Nov 2026):

- current:      what MosAIc uses: half the stock's own base rate, half novelty (a check: it must
                reproduce the run)
- confidence:   the model's own confidence in the reading (E)
- reaction:     agreement with how the stock moved, against the market, since the filing (B): a
                reading that goes with the reaction scores higher, "no effect" scores higher after a
                small reaction. Past prices only, never option prices
- pooled:       like current, but the base rate comes from the same kind of event across all 30
                stocks instead of this stock alone (A)
- pooled + reaction, pooled + reaction + confidence, current + confidence: the plain averages

The weeks are split in two: the July forecasts are for choosing, the August ones for checking. A
scorer counts as better only if it also wins in August, and every scorer tried is reported.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from mosaic.agents import beam_search, readings_of
from mosaic.config import OUTCOMES
from mosaic.ledger import apply_ledger
from mosaic.scoring import brier, log_loss_implied

SPLIT = "2026-08-01"            # before: choosing; from: checking

# How well each effect goes with the stock's reaction since the filing (B). No reaction: no view.
REACTION_FIT = {
    "big fall": {"fall": 0.80, "gain": 0.20, "spread": 0.65, "none": 0.35},
    "fall":     {"fall": 0.65, "gain": 0.35, "spread": 0.50, "none": 0.50},
    "small":    {"fall": 0.40, "gain": 0.40, "spread": 0.40, "none": 0.65},
    "rise":     {"fall": 0.35, "gain": 0.65, "spread": 0.50, "none": 0.50},
    "big rise": {"fall": 0.20, "gain": 0.80, "spread": 0.65, "none": 0.35},
}

Scorer = Callable[[dict, dict, dict], float]       # (record, item, reading) -> score from 0 to 1


def novelty_part(item: dict, reading: dict) -> float:
    nov = item.get("novelty")
    nov = 0.5 if nov is None else float(nov)
    return 1 - nov if reading["effect"] == "none" else nov


def current(rec, item, r) -> float:
    return float(r["score"])


def confidence(rec, item, r) -> float:
    c = r.get("confidence")
    return 0.5 if r["effect"] == "none" or c is None else float(min(max(c, 0.0), 1.0))


def reaction(rec, item, r) -> float:
    fit = REACTION_FIT.get(item.get("reaction_bucket") or "")
    return 0.5 if fit is None else fit[r["effect"]]


def make_pooled(base_rate: Callable[[dict, dict, dict], float]) -> Scorer:
    """`base_rate(record, item, reading)` gives the pooled base rate; novelty is kept as in `current`."""
    def pooled(rec, item, r) -> float:
        return 0.5 * base_rate(rec, item, r) + 0.5 * novelty_part(item, r)
    return pooled


def mean_of(*scorers: Scorer) -> Scorer:
    def combined(rec, item, r) -> float:
        return float(np.mean([s(rec, item, r) for s in scorers]))
    return combined


def scorers(pooled_base_rate=None) -> dict[str, Scorer]:
    out = {"current": current, "confidence": confidence, "reaction": reaction,
           "current + confidence": mean_of(current, confidence)}
    if pooled_base_rate is not None:
        pooled = make_pooled(pooled_base_rate)
        out.update({"pooled": pooled, "pooled + reaction": mean_of(pooled, reaction),
                    "pooled + reaction + confidence": mean_of(pooled, reaction, confidence)})
    return out


# --- rerunning the tree ---------------------------------------------------------------------------

def rerun(rec: dict, scorer: Scorer) -> dict:
    """The forecast the tree would have made with this scorer. A run that ended in the baseline stays there."""
    base = rec["baseline"]
    if not rec["items"] or str(rec.get("status") or "").startswith("baseline"):
        return dict(base)
    prepared = [{"item": it["id"], "readings": [{**r, "score": scorer(rec, it, r)} for r in it["readings"]]}
                for it in rec["items"]]
    choice, _ = beam_search(prepared, base)
    if not choice:
        return dict(base)
    final, _ = apply_ledger(base, readings_of(prepared, choice))
    return final


def scores_for(rec: dict, probs: dict) -> dict:
    row = {"brier_outcome": np.nan, "log_loss_outcome": np.nan, "log_loss_implied": np.nan}
    if rec.get("outcome") in OUTCOMES:
        row["brier_outcome"] = brier(probs, rec["outcome"])
        row["log_loss_outcome"] = -math.log(max(probs[rec["outcome"]] / 100, 1e-6))
    if rec.get("implied"):
        row["log_loss_implied"] = log_loss_implied(probs, rec["implied"])
    return row


def rescore(records: list[dict], scorer_set: dict[str, Scorer], split: str = SPLIT) -> pd.DataFrame:
    """One row per forecast and method: the baseline, the tree as run, and each scorer."""
    rows = []
    for rec in records:
        half = "choose (July)" if rec["as_of"] < split else "check (August)"
        key = {"ticker": rec["ticker"], "as_of": rec["as_of"], "half": half}
        methods = {"baseline": dict(rec["baseline"]), "ToT as run": dict(rec["probabilities"])}
        for name, s in scorer_set.items():
            methods[f"ToT, {name}"] = rerun(rec, s)
        for name, probs in methods.items():
            moved = max(abs(probs[k] - rec["baseline"][k]) for k in OUTCOMES)
            rows.append({**key, "method": name, **{k: round(probs[k], 2) for k in OUTCOMES},
                         "moved": round(moved, 2), **scores_for(rec, probs)})
    return pd.DataFrame(rows)


def summary(table: pd.DataFrame) -> pd.DataFrame:
    """Mean scores per method, overall and per half, with the difference from the baseline."""
    out = []
    order = list(dict.fromkeys(table["method"]))
    for half in [None, "choose (July)", "check (August)"]:
        t = table if half is None else table[table["half"] == half]
        base = t[t["method"] == "baseline"].set_index(["ticker", "as_of"])["log_loss_implied"]
        for m in order:
            x = t[t["method"] == m]
            diff = (x.set_index(["ticker", "as_of"])["log_loss_implied"] - base).mean()
            out.append({"half": half or "all", "method": m, "forecasts": len(x),
                        "moved from baseline": int((x["moved"] > 0.05).sum()),
                        "log_loss_implied": round(x["log_loss_implied"].mean(), 4),
                        "vs baseline": round(diff, 4),
                        "log_loss_outcome": round(x["log_loss_outcome"].mean(), 4),
                        "brier_outcome": round(x["brier_outcome"].mean(), 4)})
    return pd.DataFrame(out)


def load(path: Path) -> list[dict]:
    """Saved readings, keeping the last record for each stock and date."""
    recs: dict = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            if r.get("baseline") and r.get("probabilities"):
                recs[(r["ticker"], r["as_of"])] = r
    return list(recs.values())


# --- pooled base rates (needs internet the first time; filings and prices are cached) -------------

def pooled_memory(tickers, start: str, end: str, sec_get=None, price_fetch=None):  # pragma: no cover - internet
    """One case memory over all the stocks' filings and prices."""
    from mosaic.case_memory import CaseMemory
    from mosaic.config import SETTINGS
    from mosaic.edgar import company_filings, event_key, parse_accepted, sec_getter
    from mosaic.prices import load_closes
    get = sec_get or sec_getter()
    events, prices = {}, {}
    for t in tickers:
        _, rows = company_filings(t, get)
        events[t] = [(event_key(r["form"], r.get("items", "")), parse_accepted(r["acceptanceDateTime"])) for r in rows]
        prices[t] = load_closes(t, start, end, price_fetch)
        print(f"  {t}: {len(events[t])} filings, {len(prices[t])} prices")
    market = load_closes(SETTINGS.market_ticker, start, end, price_fetch)
    return CaseMemory(events, prices, market=market)


def pooled_base_rate(memory) -> Callable[[dict, dict, dict], float]:
    """Base rate from the pooled memory, the same way MosAIc scores a reading (news counts as any event)."""
    cache: dict = {}

    def rate(rec, item, r) -> float:
        as_of = datetime.fromisoformat(rec["as_of"]).replace(tzinfo=timezone.utc)
        event = "*" if item.get("event") == "news" else item.get("event") or "*"
        key = (r["effect"], r.get("size"), event, item.get("lag_days") or 1, rec["as_of"], item.get("reaction_bucket"))
        if key not in cache:
            cache[key], _ = memory.score(r["effect"], event, item.get("lag_days") or 1, as_of,
                                         item.get("reaction_bucket"), r.get("size") or "medium")
        return cache[key]
    return rate


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Rescore saved Tree of Thought readings with other scorers.")
    p.add_argument("folder", help="an evaluation folder with readings.jsonl")
    p.add_argument("--pooled", action="store_true", help="also try base rates pooled across all 30 stocks")
    p.add_argument("--split", default=SPLIT)
    a = p.parse_args(argv)
    folder = Path(a.folder)
    records = load(folder / "readings.jsonl")
    print(f"{len(records)} forecasts with saved readings.")
    base_rate = None
    if a.pooled:
        from mosaic.config import UNIVERSE
        print("Building the pooled case memory (all 30 stocks):")
        base_rate = pooled_base_rate(pooled_memory(sorted(UNIVERSE), "2014-01-01", max(r["as_of"] for r in records)))
    table = rescore(records, scorers(base_rate), a.split)
    s = summary(table)
    table.to_csv(folder / "rescored.csv", index=False)
    s.to_csv(folder / "rescored_summary.csv", index=False)
    same = table[table["method"] == "ToT, current"].set_index(["ticker", "as_of"])[list(OUTCOMES)]
    run = table[table["method"] == "ToT as run"].set_index(["ticker", "as_of"])[list(OUTCOMES)]
    print(f"Check: the current scorer reproduces the run on {int(((same - run).abs().max(axis=1) < 0.05).sum())} "
          f"of {len(run)} forecasts.")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(s.to_string(index=False))
    print(f"Written to {folder}/rescored.csv and rescored_summary.csv")


if __name__ == "__main__":                                       # pragma: no cover
    main()
