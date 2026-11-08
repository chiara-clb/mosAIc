"""Checkpoint 5 notebook: the LangGraph flow, run on an offline stand-in."""

import pytest

from tests.notebook import fake_langgraph as lg
from tests.notebook.core import load_multi_agent_core

NS = load_multi_agent_core()


def graph():
    return NS["build_graph"](lg.StateGraph, lg.START, lg.END)


def test_fake_graph_joins_and_reducers():
    from typing import Annotated, TypedDict
    import operator

    class S(TypedDict, total=False):
        log: Annotated[list, operator.add]
        x: int

    g = lg.StateGraph(S)
    g.add_node("a", lambda s: {"log": ["a"]})
    g.add_node("b", lambda s: {"log": ["b"]})
    g.add_node("c", lambda s: {"log": ["c"], "x": 1})
    g.add_node("d", lambda s: {"log": ["d"]})
    g.add_edge(lg.START, "a")
    g.add_edge("a", "b")
    g.add_edge("a", "d")
    g.add_edge(["b", "d"], "c")
    g.add_edge("c", lg.END)
    c = g.compile()
    out = c.invoke({"log": []})
    assert out["log"] == ["a", "b", "d", "c"] and c.steps == [["a"], ["b", "d"], ["c"]]

    bad = lg.StateGraph(S)
    bad.add_node("a", lambda s: {})
    bad.add_node("b", lambda s: {"x": 1})
    bad.add_node("d", lambda s: {"x": 2})
    bad.add_edge(lg.START, "a")
    bad.add_edge("a", "b")
    bad.add_edge("a", "d")
    with pytest.raises(lg.InvalidUpdateError):
        bad.compile().invoke({})


def test_quant_and_research_run_in_the_same_step_and_synthesis_waits():
    g = graph()
    state, summary = NS["run_forecast"](g)
    assert g.steps[:3] == [["coordinator"], ["quant", "research"], ["synthesis"]]
    assert summary["path"] == "coordinator → quant → research → synthesis → critic → finalise"
    assert summary["status"] == "accepted" and summary["items right"] == "4/4"
    assert summary["model calls"] <= NS["MAX_MODEL_CALLS"]


def test_every_route_out_of_the_critic_is_wired():
    g = graph()
    fn, mapping = g.g.cond["critic"]
    assert set(mapping) == set(NS["ROUTES"]) and fn is NS["route_after_critic"]
    assert ("research_more", "synthesis") in g.g.edges
    assert {("finalise", lg.END), ("fallback", lg.END)} <= set(g.g.edges)


def test_latency_counts_the_parallel_branch_once():
    _, s = NS["run_forecast"](graph())
    serial = s["model calls"] * NS["SECONDS_PER_MODEL_CALL"] + s["tool calls"] * NS["SECONDS_PER_TOOL_CALL"]
    assert s["est. seconds"] < serial


def test_running_out_of_model_calls_ends_in_the_baseline():
    budget = NS["MAX_MODEL_CALLS"]
    NS["MAX_MODEL_CALLS"] = 6
    try:
        state, summary = NS["run_forecast"](graph(), chat=NS["scripted_chat"])
    finally:
        NS["MAX_MODEL_CALLS"] = budget
    assert summary["status"] == "baseline" and summary["model calls"] <= 6
    assert any("budget" in w for w in state["final"]["why"])
