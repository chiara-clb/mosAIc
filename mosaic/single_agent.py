"""The single-agent comparison: one model with every tool and the full passages, checking its own quotes.

It answers "why not one agent?". Same model, prompts and Tree of Thought as the design, but there is no
separate critic and no risk gate, and the model reads whole passages while holding the search tools.
"""

from __future__ import annotations

from mosaic import agents as A
from mosaic.graph import ForecastState, as_node, graph_engine
from mosaic.llm import Model
from mosaic.tools import ToolBox
from mosaic.trace import Trace
from mosaic.world import World

ANALYST_ROLE = {"analyst": {
    "role": "Analyst (single agent)",
    "goal": "Research the company, turn the evidence into ledger lines and check them.",
    "backstory": "You do the whole job alone.",
    "tools": ["search_documents", "search_news", "case_memory", "branch_log", "get_passage"],
    "kind": "llm", "sees": "full passages"}}


def analyst(state: dict, run: A.Run) -> dict:
    found = A.research(state, run, agent="analyst", full_text=True)
    if "error" in found:
        return found
    out = A.synthesis({**state, **found}, run, agent="analyst", judge="analyst")
    return {"evidence": found["evidence"], **out, "_note": found["_note"] + " | " + out["_note"]}


def build_single_agent_graph(run: A.Run, engine=None):
    StateGraph, START, END, _ = engine or graph_engine()
    g = StateGraph(ForecastState)
    for name, fn in {"coordinator": A.coordinator, "quant": A.quant, "analyst": analyst,
                     "finalise": A.finalise, "fallback": A.fallback}.items():
        g.add_node(name, as_node(name, fn, run))
    g.add_edge(START, "coordinator")
    g.add_edge("coordinator", "quant")
    g.add_edge("quant", "analyst")
    g.add_conditional_edges("analyst", lambda s: "fallback" if s.get("error") else "finalise",
                            {"finalise": "finalise", "fallback": "fallback"})
    g.add_edge("finalise", END)
    g.add_edge("fallback", END)
    return g.compile()


def run_single_agent(world: World, model: Model, failures=(), options: dict | None = None, engine=None):
    trace = Trace(run_id=f"{world.ticker}-{world.as_of:%Y-%m-%d}", as_of=world.as_of)
    run = A.Run(ToolBox(world, trace=trace, failures=failures, extra_roles=ANALYST_ROLE), model,
                {"protocol": "two_way", "readings": "tot", **(options or {})})
    trace.add("run", "start", ticker=world.ticker, model=model.name, single_agent=True)
    state = build_single_agent_graph(run, engine).invoke(
        {"request": {"ticker": world.ticker, "company": world.company, "as_of": world.as_of},
         "trace": [], "error": []}, {"recursion_limit": 50})
    return state, run
