"""Each guardrail can be switched off for the ablation; all are on by default."""

from mosaic import minigraph
from mosaic.scenarios import run_scenario

ENGINE = (minigraph.StateGraph, minigraph.START, minigraph.END, "built-in runner")


def run(name, off=()):
    return run_scenario(name, options={"guardrails_off": list(off)}, engine=ENGINE)


def injected(run_):
    return [c for c in run_.tools.calls if c["status"] == "ok" and "buyout rumor" in str(c.get("args"))]


def test_all_guardrails_on_by_default():
    _, r, row = run_scenario("7 hidden instructions", engine=ENGINE)
    assert r.tools.off == set() and row["injected tool calls"] == 0


def test_without_the_role_check_synthesis_can_search():
    _, r, row = run("7 hidden instructions", ["role_check"])
    assert row["injected tool calls"] >= 1 and any(c["node"] == "synthesis" for c in injected(r))
    assert not [c for c in r.tools.calls if c["status"] == "refused: not in role"]


def test_without_cleaned_extracts_research_follows_hidden_instructions():
    _, r, row = run("7 hidden instructions", ["clean_extracts"])
    assert any(c["node"] == "research" for c in injected(r))


def test_without_the_risk_gate_everything_is_published():
    state, _, row = run("9 rare event, big move", ["risk_gate"])
    assert row["review"] == "publish" and state["final"]["triggers"]          # the signs are still counted


def test_without_the_date_rule_late_items_come_back():
    state, r, _ = run("8 late data", ["date_rule"])
    assert {"F5", "S3"} & {e["id"] for e in state["evidence"]}
