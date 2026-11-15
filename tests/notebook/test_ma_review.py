"""Checkpoint 5 notebook: the Checkpoint 6 risk gate and the analyst's review."""

import pytest

from tests.notebook import fake_langgraph as lg
from tests.notebook.core import load_multi_agent_core

NS = load_multi_agent_core()
GRAPH = NS["build_graph"](lg.StateGraph, lg.START, lg.END)
BASE = {"fall": 32.0, "flat": 36.0, "gain": 32.0}


def line(source="F1", effect="gain", size="small", n=200, confidence=None):
    return {"source": source, "effect": effect, "size": size, "quote": "q", "reason": "r",
            "shift": NS["shift_for"](effect, size), "n": n, "confidence": confidence}


def state(lines, status="accepted", retries=0, ambiguous=False):
    probs, _ = NS["apply_ledger"](BASE, lines)
    return {"final": {"status": status, "baseline": BASE, "probabilities": probs if status != "baseline" else BASE,
                      "ledger": lines if status != "baseline" else []},
            "retries": retries, "tot": {"ambiguous": ambiguous},
            "evidence": [{"id": "F1", "event": "5.02"}, {"id": "S2", "event": "news"}]}


def level(st):
    return NS["risk_gate"](st)["final"]["review"]


def test_no_trigger_publishes():
    assert NS["risk_triggers"](state([line("F1", "spread"), line("S2", "gain")])) == []
    assert level(state([line("F1", "spread"), line("S2", "gain")])) == "publish"


@pytest.mark.parametrize("st,word", [
    (state([line("F1", "gain", "large"), line("S2", "gain", "large")]), "high impact"),
    (state([line("F1", "spread", n=8), line("S2", "gain")]), "thin evidence"),
    (state([line("F1", "gain", "medium")]), "one source"),
    (state([line("F1", "spread"), line("S2", "gain")], ambiguous=True), "ambiguity"),
    (state([line("F1", "spread", confidence=0.3), line("S2", "gain")]), "low confidence"),
    (state([line("F1", "spread"), line("S2", "gain")], retries=1), "sent work back"),
    (state([], status="baseline"), "baseline used"),
])
def test_each_trigger_alone_flags(st, word):
    t = NS["risk_triggers"](st)
    assert len(t) == 1 and word in t[0], t
    assert level(st) == "flag"


def test_two_triggers_hold():
    st = state([line("F1", "spread", "large", n=8), line("S2", "gain", "large")])
    assert len(NS["risk_triggers"](st)) == 2 and level(st) == "hold"
    assert NS["hold"]({**st, "final": NS["risk_gate"](st)["final"]})["final"]["status"].endswith("held for review")


def test_router_follows_the_level():
    for lv in ("publish", "flag", "hold"):
        assert NS["route_after_risk"]({"final": {"review": lv}}) == lv


def test_analyst_review_keeps_resizes_and_strikes():
    NS["REVIEW_LOG"].clear()
    st = state([line("F1", "spread", "large"), line("S2", "gain", "large")])
    out = NS["apply_review"](st, {"F1": "strike", "S2": ("resize", "small")})
    assert [l["source"] for l in out["ledger"]] == ["S2"] and out["ledger"][0]["size"] == "small"
    assert out["probabilities"] == NS["apply_ledger"](BASE, [line("S2", "gain", "small")])[0]
    assert out["status"] == "reviewed by analyst"
    assert NS["review_feedback"]() == {"5.02": {"keep": 0, "resize": 0, "strike": 1},
                                       "news": {"keep": 0, "resize": 1, "strike": 0}}
    with pytest.raises(ValueError):
        NS["apply_review"](st, {"F1": ("resize", "huge")})


def test_held_scenario_end_to_end():
    rows, states = NS["run_scenarios"](GRAPH, ["9 rare event, big move"])
    st = states["9 rare event, big move"]
    assert rows[0]["review"] == "hold" and st["trace"][-1]["node"] == "hold"
    reviewed = NS["apply_review"](st, {"F1": ("resize", "small")})
    assert max(abs(reviewed["probabilities"][k] - st["final"]["baseline"][k]) for k in NS["OUTCOMES"]) < \
        max(abs(st["final"]["probabilities"][k] - st["final"]["baseline"][k]) for k in NS["OUTCOMES"])


def test_synthesis_reports_ambiguity_only_for_close_opposite_ledgers():
    prepared = [{"readings": [{"effect": "gain", "size": "small", "score": 0.6},
                              {"effect": "fall", "size": "small", "score": 0.58},
                              {"effect": "none", "size": None, "score": 0.1}]}]
    nodes = [{"choice": (0,), "depth": 1, "status": "chosen", "score": 0.6},
             {"choice": (1,), "depth": 1, "status": "kept", "score": 0.58}]
    assert NS["is_ambiguous"](prepared, nodes)
    nodes[1]["score"] = 0.3
    assert not NS["is_ambiguous"](prepared, nodes)
