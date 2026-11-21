from mosaic import minigraph
from mosaic.risk import apply_review
from mosaic.scenarios import run_scenario
from mosaic.views import evidence_rows, forecast_rows, ledger_rows, run_facts, step_rows, tool_rows, tot_rows

ENGINE = (minigraph.StateGraph, minigraph.START, minigraph.END, "built-in runner")


def test_tables_from_a_normal_run():
    state, run, _ = run_scenario("1 normal", engine=ENGINE)
    f = forecast_rows(state["final"])
    assert [r["Outcome"] for r in f] == ["Large fall", "Flat", "Large gain"]
    assert abs(sum(r["Change (points)"] for r in f)) < 0.2
    assert [(r["Source"], r["Published"]) for r in ledger_rows(state)] == [("F1", "2025-05-28"), ("S2", "2025-05-22")]
    assert {r["Id"] for r in evidence_rows(state)} == {"F1", "F2", "F3", "S2"}
    tot = tot_rows(state)
    assert len([r for r in tot if r["Chosen"]]) == 4 and {r["Item"] for r in tot} == {"F1", "F2", "F3", "S2"}
    assert step_rows(state)[0]["Step"] == "coordinator" and tool_rows(run)
    facts = run_facts(state, run)
    assert facts["review"] == "publish" and facts["leakage"] == "clean"


def test_reviewed_ledger_and_a_baseline_run():
    state, _, _ = run_scenario("9 rare event, big move", engine=ENGINE)
    reviewed = apply_review(state, {"F1": "strike"})
    assert [r["Source"] for r in ledger_rows(state, reviewed)] == ["S2"]
    base_state, base_run, _ = run_scenario("6 tool down", engine=ENGINE)
    assert ledger_rows(base_state) == [] and tot_rows(base_state) == []
    assert run_facts(base_state, base_run)["why"]
