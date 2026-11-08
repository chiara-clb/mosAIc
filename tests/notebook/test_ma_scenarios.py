"""Checkpoint 5 notebook: every scenario takes the path the design promises."""

import pytest

from tests.notebook import fake_langgraph as lg
from tests.notebook.core import load_multi_agent_core

NS = load_multi_agent_core()
GRAPH = NS["build_graph"](lg.StateGraph, lg.START, lg.END)


@pytest.mark.parametrize("name", list(NS["SCENARIOS"]))
def test_scenario_meets_its_expectation(name):
    rows, _ = NS["run_scenarios"](GRAPH, [name])
    assert rows[0]["as expected"] == "yes", rows[0]


def test_no_scenario_publishes_made_up_quotes_injected_calls_or_late_data():
    rows, _ = NS["run_scenarios"](GRAPH)
    assert len(rows) == 8
    for r in rows:
        assert not r["made-up quote published"] and r["injected tool calls"] == 0 and not r["late data used"], r
        assert r["model calls"] <= NS["MAX_MODEL_CALLS"]


def test_fallback_scenarios_report_the_baseline_and_why():
    _, states = NS["run_scenarios"](GRAPH, ["5 retries used up", "6 tool down"])
    for st in states.values():
        assert st["final"]["probabilities"] == st["final"]["baseline"] and st["final"]["why"]
    assert any("retries" in w for w in states["5 retries used up"]["final"]["why"])
    assert any("search_documents" in w for w in states["6 tool down"]["final"]["why"])


def test_hidden_instructions_are_refused_at_the_tool_door():
    _, states = NS["run_scenarios"](GRAPH, ["7 hidden instructions"])
    refused = [e for e in NS["CTX"]["tool_log"] if e["status"] == "refused: not in role"]
    assert refused and refused[0]["caller"] == "synthesis" and refused[0]["tool"] == "search_news"


def test_check_expectations_reports_misses():
    s = {"status": "baseline", "retries": 0, "path": "a → b"}
    assert NS["check_expectations"](s, {"status": "accepted", "via": "c"}) == \
        ["status = 'baseline', expected 'accepted'", "path lacks 'c'"]
