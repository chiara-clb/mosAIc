from mosaic import minigraph
from mosaic.graph import build_graph, graph_engine, run_forecast
from mosaic.llm import make_model
from mosaic.trace import leakage_audit
from mosaic.world import make_demo_world

ENGINE = (minigraph.StateGraph, minigraph.START, minigraph.END, "built-in runner")


def forecast(**kw):
    model = kw.pop("model", None) or make_model("scripted", kw.pop("script", None))
    return run_forecast(make_demo_world(), model, engine=ENGINE, **kw)


def test_quant_and_research_run_in_the_same_step_and_synthesis_waits():
    state, run = forecast()
    steps = [t["node"] for t in state["trace"]]
    assert steps == ["coordinator", "quant", "research", "synthesis", "critic", "finalise", "risk_gate", "publish"]
    g = build_graph(run, ENGINE)
    g.invoke({"request": state["request"], "trace": [], "error": []})
    assert g.steps[:3] == [["coordinator"], ["quant", "research"], ["synthesis"]]


def test_every_route_is_wired():
    _, run = forecast()
    g = build_graph(run, ENGINE).g
    assert set(g.cond["critic"][1]) == {"finalise", "research", "synthesis", "fallback"}
    assert ("research_more", "synthesis") in g.edges
    assert {("finalise", "risk_gate"), ("fallback", "risk_gate")} <= set(g.edges)
    assert g.cond["risk_gate"][1] == {"publish": "publish", "flag": "flag", "hold": "hold"}


def test_the_run_trace_records_steps_calls_and_sources_and_passes_the_audit():
    state, run = forecast()
    t = run.tools.trace
    assert t.events[0]["kind"] == "start" and t.events[-1]["kind"] == "end"
    assert [e["node"] for e in t.of_kind("step")] == [s["node"] for s in state["trace"]]
    assert t.of_kind("tool_call") and t.of_kind("source_used")
    assert leakage_audit(t) == []


def test_running_out_of_model_calls_ends_in_the_baseline():
    state, run = forecast(model=make_model("scripted", budget=6))
    assert state["final"]["status"] == "baseline" and run.model.calls <= 6
    assert any("budget" in w for w in state["final"]["why"])


def test_engine_falls_back_to_the_built_in_runner():
    assert graph_engine()[3] in ("langgraph", "built-in runner")


def test_a_research_memo_lets_later_runs_reuse_the_first_research():
    memo = {}
    first, run1 = forecast(research_memo=memo)
    second, run2 = forecast(research_memo=memo)
    research_calls = lambda run: [e for e in run.tools.trace.events if e["node"] == "research" and e["kind"] == "model_call"]
    assert research_calls(run1) and not research_calls(run2)
    assert any(e["kind"] == "reused" for e in run2.tools.trace.events)
    ids = lambda s: sorted(e["id"] for e in s["evidence"])
    assert ids(first) == ids(second)
    assert run2.model.calls < run1.model.calls


def test_without_a_memo_every_run_does_its_own_research():
    _, run1 = forecast()
    _, run2 = forecast()
    assert all(any(e["node"] == "research" and e["kind"] == "model_call" for e in r.tools.trace.events) for r in (run1, run2))
