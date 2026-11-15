"""The flow between the agents, as a LangGraph graph (Checkpoint 5), and `run_forecast`.

- coordinator -> quant and coordinator -> research: quant and research run in parallel.
- [quant, research] -> synthesis: synthesis waits for both.
- critic -> finalise | research_more | synthesis | fallback: the critic's verdict picks the next step.
- finalise and fallback -> risk gate -> publish | flag | hold (Checkpoint 6).

LangGraph is used when installed. Otherwise a small built-in runner with the same behaviour is used
(mosaic/minigraph.py), so the demo runs anywhere.
"""

from __future__ import annotations

import operator
from functools import partial
from typing import Annotated, TypedDict

from mosaic import agents as A
from mosaic.llm import BudgetError, Model
from mosaic.risk import review_level, risk_triggers
from mosaic.tools import ToolBox
from mosaic.trace import Trace
from mosaic.world import World


class ForecastState(TypedDict, total=False):
    request: dict
    plan: dict
    baseline: dict
    evidence: list
    ledger: list
    forecast: dict
    tot: dict
    verdict: dict
    feedback: list
    hints: list
    retries: int
    final: dict
    error: Annotated[list, operator.add]     # quant and research run in parallel, so these two
    trace: Annotated[list, operator.add]     # fields merge both nodes' writes


def graph_engine():
    """(StateGraph, START, END, name): LangGraph if installed, otherwise the built-in runner."""
    try:
        from langgraph.graph import END, START, StateGraph
        return StateGraph, START, END, "langgraph"
    except ImportError:
        from mosaic.minigraph import END, START, StateGraph
        return StateGraph, START, END, "built-in runner"


def as_node(name: str, fn, run: A.Run):
    """Run one agent, then add a line to the state's trace and the run trace."""
    def node(state):
        m0, t0 = run.model.calls, len(run.tools.calls)
        try:
            out = dict(fn(state, run) or {})
        except BudgetError as exc:                    # out of model calls: stop cleanly and report the baseline
            out = {"error": [f"{name}: {exc}"], "_note": str(exc)}
        note = out.pop("_note", "")
        step = {"node": name, "model_calls": run.model.calls - m0, "tool_calls": len(run.tools.calls) - t0, "note": note}
        run.tools.trace.add(name, "step", **{k: v for k, v in step.items() if k != "node"})
        out["trace"] = [step]
        return out
    node.__name__ = name
    return node


def risk_gate(state: dict, run: A.Run) -> dict:
    t = risk_triggers(state, run.tools.calls)
    level = review_level(t)
    return {"final": {**state["final"], "review": level, "triggers": t},
            "_note": f"{level}: " + ("; ".join(t) if t else "no trigger")}


def route_after_risk(state: dict) -> str:
    return state["final"]["review"]


def publish(state: dict, run: A.Run) -> dict:
    return {"_note": "published automatically"}


def flag(state: dict, run: A.Run) -> dict:
    return {"_note": "published with a flag for the analyst"}


def hold(state: dict, run: A.Run) -> dict:
    return {"final": {**state["final"], "status": state["final"]["status"] + ", held for review"},
            "_note": "held until the analyst reviews it"}


NODES = {"coordinator": A.coordinator, "quant": A.quant, "research": A.research, "synthesis": A.synthesis,
         "critic": A.critic, "research_more": A.research, "finalise": A.finalise, "fallback": A.fallback,
         "risk_gate": risk_gate, "publish": publish, "flag": flag, "hold": hold}


def build_graph(run: A.Run, engine=None):
    StateGraph, START, END, _ = engine or graph_engine()
    g = StateGraph(ForecastState)
    for name, fn in NODES.items():
        g.add_node(name, as_node(name, fn, run))
    g.add_edge(START, "coordinator")
    g.add_edge("coordinator", "quant")                 # parallel fan-out
    g.add_edge("coordinator", "research")
    g.add_edge(["quant", "research"], "synthesis")     # waits for both
    g.add_edge("synthesis", "critic")
    g.add_conditional_edges("critic", A.route_after_critic,
                            {"finalise": "finalise", "research": "research_more",
                             "synthesis": "synthesis", "fallback": "fallback"})
    g.add_edge("research_more", "synthesis")
    g.add_edge("finalise", "risk_gate")
    g.add_edge("fallback", "risk_gate")
    g.add_conditional_edges("risk_gate", route_after_risk, {"publish": "publish", "flag": "flag", "hold": "hold"})
    for end in ("publish", "flag", "hold"):
        g.add_edge(end, END)
    return g.compile()


def run_forecast(world: World, model: Model, failures=(), options: dict | None = None, engine=None):
    """One forecast, start to finish. Returns (final state, run)."""
    trace = Trace(run_id=f"{world.ticker}-{world.as_of:%Y-%m-%d}", as_of=world.as_of)
    run = A.Run(ToolBox(world, trace=trace, failures=failures), model,
                {"protocol": "two_way", "readings": "tot", **(options or {})})
    trace.add("run", "start", ticker=world.ticker, as_of=world.as_of.isoformat(), model=model.name,
              engine=(engine or graph_engine())[3], options=run.options)
    graph = build_graph(run, engine)
    state = graph.invoke({"request": {"ticker": world.ticker, "company": world.company, "as_of": world.as_of},
                          "trace": [], "error": []}, {"recursion_limit": 50})
    trace.add("run", "end", status=state["final"]["status"], review=state["final"].get("review"))
    return state, run
