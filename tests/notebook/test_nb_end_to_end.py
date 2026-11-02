"""Dry run of the notebook's whole flow with synthetic filings, prices and a stand-in model."""

import json
from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from tests.notebook.core import load_core
from tests.notebook.helpers import UTC, dt

ns = load_core(START_DATE="2025-01-01", END_DATE="2026-09-25")
DIM = 16
RNG = np.random.default_rng(7)
QUERY_VEC = {q: v / np.linalg.norm(v) for q, v in zip(ns["QUERIES"], RNG.normal(size=(3, DIM)))}


def near(v, noise=0.6):
    w = v + RNG.normal(scale=noise, size=DIM) / np.sqrt(DIM)
    return w / np.linalg.norm(w)


def synthetic_world():
    days = pd.bdate_range("2020-01-01", "2026-10-30")
    logp = np.cumsum(RNG.normal(0.0003, 0.012, len(days)))
    px = pd.Series(100 * np.exp(logp), index=days)
    filings, chunks, vecs = [], [], []
    q = list(QUERY_VEC.values())
    d = dt("2020-02-01")
    n = 0
    while d < dt("2026-10-01"):
        for form, items, offset, section, qv in (("8-K", "2.02,9.01", 0, "Press release", q[0]),
                                                 ("10-Q", "", 1, "Item 1. Legal Proceedings", q[1]),
                                                 ("8-K", "5.02", 20, "Press release", q[2])):
            acc = d + timedelta(days=offset, hours=21)
            acc_s = acc.strftime("%Y-%m-%dT%H:%M:%S.000Z")
            filings.append({"form": form, "items": items, "acceptanceDateTime": acc_s, "accessionNumber": f"A{n}"})
            if acc >= dt("2024-06-01"):
                for k in range(3):
                    chunks.append({"id": f"XYZ-{form}-{n}-{k}", "ticker": "XYZ", "company": "XYZ Corp", "form": form,
                                   "section": section, "section_key": ns["section_key"](section),
                                   "event": ns["event_key"](form, items), "accepted": acc.isoformat(),
                                   "filing": f"A{n}", "url": f"https://x/A{n}/d.htm",
                                   "text": f"{section} text {n}-{k} legal proceedings" if form == "10-Q" else f"news {n}-{k}",
                                   "header": f"XYZ | {form} | {section}"})
                    vecs.append(near(qv))
            n += 1
        d += timedelta(days=91)
    return px, filings, chunks, np.array(vecs, dtype=np.float32)


def fake_chat(messages, max_tokens):
    system = messages[0]["content"]
    if "TWO most plausible" in system:
        text = messages[1]["content"]
        if "Legal Proceedings" in text:
            return json.dumps([{"effect": "fall", "size": "medium", "reason": "legal proceedings"},
                               {"effect": "spread", "size": "small", "reason": "legal proceedings"}])
        return json.dumps([{"effect": "gain", "size": "small", "reason": "news"},
                           {"effect": "none", "size": "small", "reason": "routine"}])
    if "check whether a claim" in system:
        return "yes"
    return '{"fall": 25, "flat": 45, "gain": 30}'


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    px, filings, chunks, emb = synthetic_world()
    index = ns["Index"](chunks, emb)
    events = {"XYZ": [(ns["event_key"](r["form"], r["items"]), ns["parse_accepted"](r["acceptanceDateTime"]))
                      for r in filings]}
    case = ns["CaseMemory"](events, {"XYZ": px})
    log = tmp_path_factory.mktemp("logs") / "branch_log.jsonl"
    dates = ns["forecast_dates"](filings, "2025-01-01", "2026-09-25")
    plan = ns["scenario_dates"](filings, px.index, "2025-01-01", "2026-09-25")
    records = []
    for d, sc in plan:
        rec = ns["run_forecast"]("XYZ", d, fake_chat, index, case, px, filings, QUERY_VEC.__getitem__,
                                 company="XYZ Corp", branch_log=str(log))
        rec["scenario"] = sc
        records.append(rec)
    return records, log, dates


def test_forecast_dates_are_two_days_after_results(run):
    records, _, dates = run
    assert 6 <= len(dates) <= 8
    assert all(d.hour == 0 for d in dates)


def test_each_forecast_has_evidence_variants_and_outcome(run):
    records, _, _ = run
    for r in records:
        assert len(r["evidence"]) <= 5
        if r["scenario"] in ("after results", "event"):        # a filing two days earlier is always there
            assert len(r["evidence"]) >= 1
        assert set(r["variants"]) == {"baseline", "prompt only", "single pass", "ToT beam 1", "ToT beam 3"}
        assert all(sum(v.values()) == pytest.approx(100, abs=0.05) for v in r["variants"].values())
        assert r["outcome"] in ("fall", "flat", "gain")
        assert r["calls_tot"] <= ns["MAX_MODEL_CALLS"]


def test_budget_matches_checkpoint_4_estimate(run):
    records, _, _ = run
    for r in records:
        effects = sum(1 for e in r["prepared"] for x in e["readings"] if x["effect"] != "none")
        assert r["calls_tot"] == len(r["prepared"]) + effects <= 15


def test_branch_log_has_every_node(run):
    records, log, _ = run
    lines = [json.loads(l) for l in log.read_text().splitlines()]
    assert len(lines) == sum(len(r["tot"]["nodes"]) for r in records)
    assert {l["forecast"] for l in lines} == {r["id"] for r in records}


def test_analysis_tables_run(run):
    records, _, _ = run
    table = ns["results_table"](records)
    assert list(table.index) == ["baseline", "prompt only", "single pass", "ToT beam 1", "ToT beam 3"]
    assert table.loc["baseline", "vs baseline"] == 0
    sw = ns["sweep"](records)
    assert len(sw) == 9
    assert sw[(sw["beam width"] == 3) & (sw["margin"] == 0.3)]["ledgers different from (3, 0.3)"].iloc[0] == 0
    svs = ns["search_vs_scoring"](records)
    assert len(svs) == len(records)
    assert ns["calls_table"](records)["model calls (ToT)"].max() <= 15


def test_tree_view_and_case_lookup(run):
    records, _, _ = run
    rec = ns["find_record"](records, "legal proceedings")
    assert rec is not None
    text = ns["print_tree"](rec)
    assert "★" in text and "level 1" in text


def test_no_leakage(run):
    records, _, _ = run
    assert ns["leakage_audit"](records) == []
    for r in records:
        for e in r["evidence"]:
            assert r["as_of"] - timedelta(days=30) <= ns["datetime"].fromisoformat(e["accepted"]) < r["as_of"]


def test_slim_records_are_json_serialisable(run):
    records, _, _ = run
    r = dict(records[0])
    keep = {k: v for k, v in r.items() if k not in ("prepared", "tot", "tot1", "single", "exhaustive")}
    json.dumps(keep, default=str)


def test_every_scenario_runs_and_is_reported(run):
    records, _, _ = run
    assert {r["scenario"] for r in records} == set(ns["SCENARIOS"])
    table = ns["results_by_scenario"](records)
    assert set(table.index) == set(ns["SCENARIOS"])
    assert table["forecasts"].sum() == len(records)


def test_evidence_check_explains_empty_forecasts(run):
    records, _, _ = run
    ec = ns["evidence_check"](records)
    assert len(ec) == len(records)
    scores = ec.filter(like="best score")
    for i, r in ec.iterrows():
        if r["items"] == 0:
            no_filings = r["chunks in window"] == 0
            nothing_matched = all(isinstance(v, str) for v in scores.loc[i])
            assert no_filings or nothing_matched


def test_novelty_examples_pair_passages_with_previous_filing():
    import numpy as np
    from tests.notebook.helpers import chunk, unit
    a = chunk("p", filing="Q1", accepted="2026-05-01", section="Item 1A. Risk Factors", text="old wording")
    b = chunk("c", filing="Q2", accepted="2026-07-31", section="Item 1A. Risk Factors", text="new wording")
    idx = ns["Index"]([a, b], np.array([unit(1, 0), unit(0.9, 0.436)], dtype=np.float32))
    for ch in idx.chunks:
        ch["section_key"] = ns["section_key"](ch["section"])
    item = dict(idx.chunks[1], row=1)
    rec = {"prepared": [{"item": item, "novelty": 0.11, "previous_filing": "Q1"}]}
    low, high = ns["novelty_examples"]([rec], idx)
    assert len(low) == 1 and high == []
    assert "new wording" in low[0]["passage"] and "old wording" in low[0]["closest in previous filing"]
    assert low[0]["similarity"] == pytest.approx(0.9, abs=0.01)
