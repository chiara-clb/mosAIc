import pytest

from mosaic import minigraph
from mosaic.risk import ReviewLog, apply_review
from mosaic.scenarios import SCENARIOS, check_expectations, run_scenario, run_scenarios

ENGINE = (minigraph.StateGraph, minigraph.START, minigraph.END, "built-in runner")


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_scenario_takes_the_expected_path(name):
    _, _, row = run_scenario(name, engine=ENGINE)
    assert row["as expected"] == "yes", row


def test_no_scenario_publishes_made_up_quotes_injected_calls_or_late_data():
    rows = run_scenarios(engine=ENGINE)
    assert len(rows) == 9
    for r in rows:
        assert not r["made-up quote published"] and r["injected tool calls"] == 0 and not r["late data used"], r


def test_hidden_instructions_are_refused_and_flagged():
    state, run, _ = run_scenario("7 hidden instructions", engine=ENGINE)
    refused = [c for c in run.tools.calls if c["status"] == "refused: not in role"]
    assert refused and refused[0]["node"] == "synthesis"
    assert any("refused tool call" in t for t in state["final"]["triggers"])


def test_held_forecast_can_be_reviewed():
    state, _, _ = run_scenario("9 rare event, big move", engine=ENGINE)
    log = ReviewLog()
    reviewed = apply_review(state, {"F1": ("resize", "small")}, log)
    base = state["final"]["baseline"]
    moved = lambda p: max(abs(p[k] - base[k]) for k in base)
    assert moved(reviewed["probabilities"]) < moved(state["final"]["probabilities"])
    assert log.summary() == {"5.02": {"keep": 0, "resize": 1, "strike": 0}, "news": {"keep": 1, "resize": 0, "strike": 0}}


def test_check_expectations_reports_misses():
    s = {"status": "baseline", "retries": 0, "path": "a → b"}
    assert check_expectations(s, {"status": "accepted", "via": "c"}) == \
        ["status = 'baseline', expected 'accepted'", "path lacks 'c'"]
