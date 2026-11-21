"""Results day through the graph: synthetic prices, options and release, scripted model."""

import json
from datetime import date, timedelta

import pandas as pd
import pytest

from mosaic.graph import graph_engine
from mosaic.llm import Model
from mosaic.results_graph import apply_results_review, results_world, run_results_day
from mosaic.views import forecast_rows, ledger_rows, progress_line, review_summary, run_facts, text_log
from tests.test_release_reader import CONSENSUS, RELEASE, good
from tests.test_results_day import PAST, T0, after_close, chain_for, synthetic_prices

ENGINES = [None]
try:
    from mosaic.minigraph import END, START, StateGraph
    ENGINES.append((StateGraph, START, END, "built-in runner"))
except ImportError:                                           # pragma: no cover
    pass


def world(chain=True, consensus=CONSENSUS, text=RELEASE):
    closes = synthetic_prices(PAST + [T0])
    upto_t0 = closes.loc[:pd.Timestamp(T0)]                    # the forecast is made right after the release
    spot = float(upto_t0.iloc[-1])
    ch = chain_for(0.06, T0, [date(2026, 7, 31)], spot) if chain else None
    return results_world("ACME", "Acme Robotics Inc.", after_close(T0), upto_t0,
                         [after_close(d) for d in PAST + [T0]], text, ch, consensus)


def scripted(*replies):
    queue = list(replies)
    return Model(lambda messages, max_tokens: json.dumps(queue.pop(0)), budget=10, name="scripted")


@pytest.mark.parametrize("engine", ENGINES)
def test_the_full_flow_reads_checks_and_forecasts(engine):
    events = []
    state, run = run_results_day(world(), scripted(good()), engine=engine, on_event=events.append)
    steps = [t["node"] for t in state["trace"]]
    assert steps[0] == "coordinator" and set(steps[1:3]) == {"quant", "fetch_release"}
    assert steps[3:] == ["reader", "critic", "finalise", "risk_gate", steps[-1]]
    final = state["final"]
    assert final["status"].startswith("accepted") and final["probabilities"]["gain"] > final["baseline"]["gain"]
    assert final["review"] in ("flag", "hold") and any("high impact" in t for t in final["triggers"])
    assert run_facts(state, run)["leakage"] == "clean"
    tools = [(c["node"], c["tool"], c["status"]) for c in run.tools.calls]
    assert ("release_fetch", "get_passage", "ok") in tools and ("release_critic", "get_passage", "ok") in tools
    assert any(progress_line(e) == "Release reader: working…" for e in events)


def test_the_critic_sends_a_made_up_quote_back_with_reasons():
    bad = good()
    bad["ledger"][0]["citation"] = "Diluted EPS was a record $1.70"
    state, run = run_results_day(world(), scripted(bad, good()))
    steps = [t["node"] for t in state["trace"]]
    assert steps.count("reader") == 2 and state["retries"] == 1
    assert any("troubled run" in t for t in state["final"]["triggers"])
    assert "sent back" in next(t["note"] for t in state["trace"] if t["node"] == "critic")


def test_lines_that_fail_twice_are_struck_and_the_rest_kept():
    bad = good()
    bad["ledger"][0]["citation"] = "the stock will triple"
    state, _ = run_results_day(world(), scripted(bad, bad, bad))
    assert state["final"]["status"].startswith("accepted") and len(state["final"]["ledger"]) == 1
    assert "struck by the critic" in state["final"]["why"][0]


def test_no_options_means_the_reader_has_nothing_to_adjust():
    state, run = run_results_day(world(chain=False), scripted(good()))
    assert state["final"]["status"].startswith("baseline") and run.model.calls == 0


def test_missing_consensus_is_a_warning_sign():
    state, _ = run_results_day(world(consensus=[]), scripted(good(strength="weak")))
    assert any("no consensus" in t for t in state["final"]["triggers"])


def test_hidden_instructions_in_the_release_trigger_nothing():
    state, run = run_results_day(world(), scripted(good()))
    assert all(c["status"] == "ok" for c in run.tools.calls)     # the reader has no tools; nothing was refused
    assert "IGNORE PREVIOUS INSTRUCTIONS" in run.tools.world.filings[0]["text"]


def test_the_release_is_not_visible_before_it_was_published():
    w = world()
    w.as_of = after_close(T0) - timedelta(hours=1)               # a forecast made before the release
    state, run = run_results_day(w, scripted(good()))
    assert state["final"]["status"].startswith("baseline")


def test_the_analyst_review_shows_the_agent_next_to_the_analyst():
    state, run = run_results_day(world(), scripted(good()))
    final = state["final"]
    reviewed = apply_results_review(final, {0: "strike", 1: "weak"})
    assert review_summary(reviewed) == "Reviewed by the analyst: 0 kept, 1 struck, 1 resized"
    assert set(forecast_rows(reviewed)[0]) >= {"Baseline %", "Agent %", "Analyst %"}
    rows = ledger_rows(state, reviewed)
    assert [r["Analyst decision"] for r in rows] == ["struck", "resized to weak"]
    assert "REVIEWED BY THE ANALYST" in text_log(state, run, reviewed)


def test_every_field_a_node_writes_is_declared_in_the_state():
    """LangGraph keeps only declared fields; the built-in runner keeps all, so check it here."""
    import mosaic.results_graph as G
    state, _ = run_results_day(world(), scripted(good()))
    declared = set(G.ResultsState.__annotations__)
    assert set(state) <= declared | {"verdict"} and "verdict" in declared


def test_what_happened_is_measured_from_the_release_close_to_the_next():
    from mosaic.results_graph import actual_move
    closes = synthetic_prices(PAST + [T0])
    a = actual_move(closes, T0, 0.04)
    assert a["day"] == date(2026, 7, 31) and a["outcome"] in ("gain", "fall")
    assert actual_move(closes.loc[:pd.Timestamp(T0)], T0, 0.04) is None        # not known yet
