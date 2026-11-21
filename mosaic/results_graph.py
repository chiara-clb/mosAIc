"""Results day as a LangGraph flow: the agents forecast the next-day move right after a results release.

The same structure as the monthly forecast (mosaic/graph.py), with the same tool door, wrapper, risk gate
and publish / flag / hold nodes:

    coordinator -> quant and coordinator -> research        (in parallel)
    [quant, research] -> reader                            (waits for both)
    reader -> critic -> finalise | reader | fallback       (sent back with reasons, at most twice)
    finalise and fallback -> risk gate -> publish | flag | hold

- **coordinator** (code): confirms through `market_data` that the release came after the close.
- **quant** (code): B1, B2 and B3 from prices before the release and the option quotes at the close.
- **research** (code): fetches the press release through `get_passage`, under the forecast-date rule.
- **reader** (agent, Claude): reads the release against the baselines and the consensus, writes a ledger.
- **critic** (code): every quote must be in the release; failing lines go back with reasons, at most
  twice; after that they are struck, and if nothing is left the options baseline is reported.
- **finalise** (code): the ledger tilts B3 into the forecast.

The forecast moment is just after the release: tools return prices up to that day's close and documents
published before that moment. The next day's close is only used to show what happened.
"""

from __future__ import annotations

import json
import operator
from datetime import datetime, timedelta, timezone
from typing import Annotated, TypedDict

import pandas as pd

from mosaic import agents as A
from mosaic.graph import as_node, flag, graph_engine, hold, publish, risk_gate, route_after_risk
from mosaic.llm import Model, agent_messages, parse_json
from mosaic.release_reader import (ROLE as READER_ROLE, TASK as READER_TASK, apply_ledger, norm_quote,
                                   problems_with)
from mosaic.tools import ToolBox, ToolError
from mosaic.trace import Trace
from mosaic.world import World

MAX_RETRIES = 2
CONFIDENCE = {"low": 0.35, "medium": 0.6, "high": 0.8}     # the agent's word, as a number for the risk gate
FORECAST_DELAY = timedelta(minutes=15)                     # the forecast is made this long after the release

ROLES = {
    **READER_ROLE,
    "release_research": {
        "role": "Release researcher",
        "goal": "Fetch the results press release and the analysts' consensus for the reader.",
        "backstory": "You collect documents; you do not judge them.",
        "tools": ["get_passage"], "kind": "code", "sees": "the release, published before the forecast moment"},
    "release_critic": {
        "role": "Critic",
        "goal": "Check that every quote in the ledger is in the release, word for word, and every field is valid. "
                "Accept, or send back with the reasons.",
        "backstory": "You are the last check before an analyst sees the forecast.",
        "tools": ["get_passage"], "kind": "code", "sees": "the ledger and the release"},
}


class ResultsState(TypedDict, total=False):
    request: dict
    plan: dict
    baseline: dict
    evidence: list
    release_text: str
    consensus: list
    reply: dict
    verdict: str
    ledger: list
    feedback: list
    retries: int
    struck: list
    final: dict
    error: Annotated[list, operator.add]
    trace: Annotated[list, operator.add]


# ---------------------------------------------------------------- the nodes

def coordinator(state: dict, run: A.Run) -> dict:
    req = state["request"]
    info = run.tools.call("market_data", "coordinator", kind="earnings")
    release = req["release"]
    if info["last_results"] != release:
        return {"plan": {"ok": False}, "error": ["coordinator: the release is not the latest results before now"],
                "_note": "the release is not the latest results before the forecast moment"}
    return {"plan": {"ok": True, "past_results": len(info["past_results"]) - 1},
            "_note": f"results released {release:%Y-%m-%d %H:%M} UTC, after the close; "
                     f"{len(info['past_results']) - 1} earlier results on file"}


def quant(state: dict, run: A.Run) -> dict:
    from mosaic.results_day import results_day
    req, w = state["request"], run.tools.world
    px = run.tools.call("market_data", "quant", kind="prices")
    closes = pd.Series(px["stock"], index=pd.to_datetime([d.date() for d in px["dates"]]))
    past = run.tools.call("market_data", "quant", kind="earnings")["past_results"]
    rd = results_day(req["ticker"], req["release"], closes, past, w.extra.get("option_chain"))
    base = {"B1": rd.baselines["B1"], "B2": rd.baselines["B2"], "B3": rd.baselines["B3"], "tau": rd.tau,
            "J": rd.J, "sigma_d": rd.sigma_d, "t0": rd.t0, "past_moves": rd.past_moves, "notes": rd.notes}
    b3 = rd.baselines["B3"]
    return {"baseline": base, "_note": f"tau ±{rd.tau * 100:.1f}%; B3 " +
            (" / ".join(f"{k} {v:.0f}%" for k, v in b3.items()) if b3 else "missing: " + "; ".join(rd.notes))}


def research(state: dict, run: A.Run) -> dict:
    req, w = state["request"], run.tools.world
    try:
        doc = run.tools.call("get_passage", "release_research", item_id=req["release_id"])
    except ToolError as exc:
        return {"error": [f"research: {exc}"], "_note": str(exc)}
    if doc is None:
        return {"error": ["research: the release is not available before the forecast moment"],
                "_note": "release not available"}
    run.tools.trace.add("release_research", "source_used", source=doc["id"], published=doc["published"])
    consensus = w.extra.get("consensus") or []
    evidence = [{"id": doc["id"], "kind": "release", "published": doc["published"],
                 "header": f"{req['company']} | 8-K | Exhibit 99.1 press release", "covered_by": [],
                 "novelty": 1.0, "reaction": "-", "event": "8-K 2.02"}]
    return {"evidence": evidence, "release_text": doc["text"], "consensus": consensus,
            "_note": f"press release ({len(doc['text'].split())} words); {len(consensus)} consensus values"}


def reader(state: dict, run: A.Run) -> dict:
    base = state.get("baseline") or {}
    if not base.get("B3") or not state.get("release_text"):
        return {"reply": None, "_note": "nothing to read: " + ("no options baseline" if not base.get("B3")
                                                               else "no release")}
    words = state["release_text"].split()
    payload = {"ticker": state["request"]["ticker"], "tau_percent": round(base["tau"] * 100, 2),
               "baselines_percent": {k: base[k] for k in ("B1", "B2", "B3")},
               "consensus": state.get("consensus") or "not available", "release": " ".join(words[:6000])}
    messages = agent_messages("release_reader", READER_TASK, payload, run.tools.extra_roles)
    if state.get("feedback"):
        messages += [{"role": "assistant", "content": json.dumps(state.get("reply"))},
                     {"role": "user", "content": "Code checks failed:\n- " + "\n- ".join(state["feedback"])
                      + "\nReturn a corrected JSON object."}]
    raw = run.model(messages, 1200)
    reply = parse_json(raw)
    run.tools.trace.add("release_reader", "model_call", task="read the release", max_tokens=1200,
                        reply=str(raw)[:3000], parsed=reply is not None)
    lines = len((reply or {}).get("ledger") or [])
    return {"reply": reply, "_note": f"{lines} ledger line(s); confidence {(reply or {}).get('confidence', '?')}"}


def critic(state: dict, run: A.Run) -> dict:
    req, reply = state["request"], state.get("reply")
    if reply is None and not state.get("release_text"):
        return {"verdict": "fallback", "_note": "no reading to check"}
    doc = run.tools.call("get_passage", "release_critic", item_id=req["release_id"])
    norm = norm_quote(doc["text"] if doc else "")
    problems = problems_with(reply, norm)
    retries = state.get("retries", 0)
    if not problems:
        return {"verdict": "accept", "ledger": reply["ledger"][:6], "feedback": [], "_note": "accepted"}
    if retries < MAX_RETRIES and reply is not None:
        return {"verdict": "retry", "feedback": problems, "retries": retries + 1,
                "_note": "sent back: " + "; ".join(problems)[:300]}
    good = [x for x in (reply or {}).get("ledger", [])[:6]
            if isinstance(x, dict) and not problems_with({"ledger": [x]}, norm)]
    struck = [x for x in (reply or {}).get("ledger", [])[:6] if x not in good]
    if good:
        return {"verdict": "accept", "ledger": good, "struck": struck, "feedback": [],
                "_note": f"struck {len(struck)} line(s) that still failed; kept {len(good)}"}
    return {"verdict": "fallback", "_note": "every line failed after " + str(retries) + " retries"}


def route_after_critic(state: dict) -> str:
    return {"accept": "finalise", "retry": "reader"}.get(state.get("verdict"), "fallback")


def _lines(state: dict) -> list[dict]:
    """The ledger in MosAIc's line format, so the risk gate and the views can read it."""
    conf = CONFIDENCE.get(str((state.get("reply") or {}).get("confidence")), None)
    rid = state["request"]["release_id"]
    return [{"source": rid, "effect": f"{x['direction']}, {x['magnitude']}", "size": x["strength"],
             "quote": x.get("citation", ""), "reason": x.get("factor", ""), "confidence": conf, "n": None,
             "direction": x["direction"], "magnitude": x["magnitude"], "strength": x["strength"]}
            for x in state.get("ledger") or []]


def finalise(state: dict, run: A.Run) -> dict:
    b3 = state["baseline"]["B3"]
    lines = _lines(state)
    probs = apply_ledger(b3, lines)
    why = []
    if state.get("struck"):
        why.append(f"{len(state['struck'])} ledger line(s) struck by the critic: quote not in the release")
    extra = []
    if not state.get("consensus"):
        extra.append("no consensus estimates: beat or miss judged from the release alone")
    final = {"status": "accepted", "baseline": b3, "probabilities": probs, "ledger": lines, "why": why,
             "extra_triggers": extra, "read": (state.get("reply") or {}).get("read", "")}
    return {"final": final, "_note": "forecast: " + " / ".join(f"{k} {v:.0f}%" for k, v in probs.items())}


def fallback(state: dict, run: A.Run) -> dict:
    b3 = (state.get("baseline") or {}).get("B3")
    if not b3:
        b3 = (state.get("baseline") or {}).get("B2") or {"fall": 33.3, "flat": 33.4, "gain": 33.3}
    reasons = (state.get("error") or []) + ["the ledger failed the checks" if state.get("reply") else "no reading"]
    return {"final": {"status": "baseline", "baseline": b3, "probabilities": dict(b3), "ledger": [], "why": reasons,
                      "extra_triggers": []},
            "_note": "options baseline reported: " + "; ".join(reasons)[:200]}


def results_risk_gate(state: dict, run: A.Run) -> dict:
    """MosAIc's risk gate, plus the results-day warning signs that finalise attached."""
    out = risk_gate(state, run)
    extra = state["final"].get("extra_triggers") or []
    if extra:
        from mosaic.risk import review_level
        t = out["final"]["triggers"] + extra
        out["final"] = {**out["final"], "triggers": t, "review": review_level(t)}
        out["_note"] = f"{out['final']['review']}: " + "; ".join(t)
    return out


NODES = {"coordinator": coordinator, "quant": quant, "research": research, "reader": reader, "critic": critic,
         "finalise": finalise, "fallback": fallback, "risk_gate": results_risk_gate,
         "publish": publish, "flag": flag, "hold": hold}


def build_results_graph(run: A.Run, engine=None):
    StateGraph, START, END, _ = engine or graph_engine()
    g = StateGraph(ResultsState)
    for name, fn in NODES.items():
        g.add_node(name, as_node(name, fn, run))
    g.add_edge(START, "coordinator")
    g.add_edge("coordinator", "quant")                  # parallel fan-out
    g.add_edge("coordinator", "research")
    g.add_edge(["quant", "research"], "reader")         # waits for both
    g.add_edge("reader", "critic")
    g.add_conditional_edges("critic", route_after_critic,
                            {"finalise": "finalise", "reader": "reader", "fallback": "fallback"})
    g.add_edge("finalise", "risk_gate")
    g.add_edge("fallback", "risk_gate")
    g.add_conditional_edges("risk_gate", route_after_risk, {"publish": "publish", "flag": "flag", "hold": "hold"})
    for end in ("publish", "flag", "hold"):
        g.add_edge(end, END)
    return g.compile()


# ---------------------------------------------------------------- one run

def results_world(ticker: str, company: str, release: datetime, closes: pd.Series, past_releases: list[datetime],
                  release_text: str, chain=None, consensus: list | None = None) -> World:
    """The world the agents see: prices up to the release day's close, the release, the option quotes."""
    as_of = release + FORECAST_DELAY
    idx = pd.to_datetime(closes.index)
    dates = [datetime(d.year, d.month, d.day, tzinfo=timezone.utc) for d in idx]
    values = [float(v) for v in closes]
    item = {"id": f"{ticker}-RELEASE-{release.astimezone(timezone.utc):%Y%m%d}", "text": release_text,
            "filed": release, "form": "8-K", "section": "Press release (Exhibit 99.1)", "event": "8-K 2.02"}
    return World(ticker=ticker, company=company, as_of=as_of,
                 prices={"dates": dates, ticker: values, "SPY": values},   # no market index is needed here
                 filings=[item], news=[], case_memory={}, earnings_dates=sorted(past_releases),
                 extra={"option_chain": chain, "consensus": consensus or []})


def run_results_day(world: World, model: Model, engine=None, on_event=None, failures=()):
    """One results-day forecast through the graph. Returns (final state, run)."""
    release_item = world.filings[0]
    trace = Trace(run_id=f"{world.ticker}-results-{release_item['filed']:%Y-%m-%d}", as_of=world.as_of)
    if on_event:
        trace.listeners.append(on_event)
    run = A.Run(ToolBox(world, trace=trace, failures=failures, extra_roles=ROLES), model, {"kind": "results_day"})
    trace.add("run", "start", ticker=world.ticker, as_of=world.as_of.isoformat(), model=model.name,
              engine=(engine or graph_engine())[3], options=run.options)
    graph = build_results_graph(run, engine)
    state = graph.invoke({"request": {"ticker": world.ticker, "company": world.company, "as_of": world.as_of,
                                      "release": release_item["filed"], "release_id": release_item["id"]},
                          "trace": [], "error": [], "retries": 0}, {"recursion_limit": 40})
    trace.add("run", "end", status=state["final"]["status"], review=state["final"].get("review"))
    return state, run


# ---------------------------------------------------------------- the analyst's review

STRENGTHS = ("weak", "moderate", "strong")


def apply_results_review(final: dict, decisions: dict[int, str]) -> dict:
    """Analyst decisions per ledger line (by position): "keep", "strike" or a strength to resize to."""
    kept, record = [], []
    for i, line in enumerate(final["ledger"]):
        d = decisions.get(i, "keep")
        if d not in ("keep", "strike") + STRENGTHS:
            raise ValueError(f"unknown decision for line {i}: {d!r}")
        action = "resize" if d in STRENGTHS and d != line["strength"] else ("keep" if d in STRENGTHS else d)
        size = d if d in STRENGTHS else line["strength"]
        record.append({"source": f"{line['source']} #{i + 1}", "effect": line["effect"], "agent_size": line["strength"],
                       "action": action, "analyst_size": None if action == "strike" else size})
        if action != "strike":
            kept.append({**line, "strength": size, "size": size})
    probs = apply_ledger(final["baseline"], kept)
    agent_ledger = [{**l, "source": f"{l['source']} #{i + 1}"} for i, l in enumerate(final["ledger"])]
    return {**final, "ledger": kept, "probabilities": probs, "status": "reviewed by analyst",
            "agent_probabilities": final["probabilities"], "agent_ledger": agent_ledger, "review_record": record}
