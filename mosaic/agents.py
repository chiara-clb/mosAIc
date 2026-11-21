"""The five agents, plus the final and fallback steps.

Each agent is a function agent(state, run) -> dict of the state fields it changes. `run` holds what one
forecast uses: the tool box (world, forecast date, trace), the model and the options being compared.
The graph (mosaic/graph.py) decides the order.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import timedelta

import numpy as np

from mosaic.baseline import baseline
from mosaic.config import OUTCOMES, SETTINGS
from mosaic.handoffs import handoff, problems_baseline, problems_reading
from mosaic.ledger import apply_ledger, direction, floor_and_normalise, shift_for, total_shift
from mosaic.llm import Model, agent_messages, parse_json
from mosaic.tools import RoleError, ToolBox, ToolError, similarity, words
from mosaic.world import published


@dataclass
class Run:
    tools: ToolBox
    model: Model
    options: dict = field(default_factory=lambda: {"protocol": "two_way", "readings": "tot"})

    def ask(self, agent: str, task: str, payload: dict, max_tokens: int = 300) -> dict | None:
        raw = self.model(agent_messages(agent, task, payload, self.tools.extra_roles), max_tokens)
        parsed = parse_json(raw)
        self.tools.trace.add(agent, "model_call", task=task[:70], max_tokens=max_tokens,
                             reply=str(raw)[:3000], parsed=parsed is not None)
        return parsed


AMBIGUITY_GAP = 0.05          # best two ledgers point opposite ways and score this close


# --- coordinator and quant ---------------------------------------------------------------------

def coordinator(state: dict, run: Run) -> dict:
    e = run.tools.call("market_data", "coordinator", kind="earnings")
    as_of = state["request"]["as_of"]
    window_end = as_of + timedelta(days=round(SETTINGS.horizon_days * 7 / 5))
    due = bool(e["next_estimate"] and as_of <= e["next_estimate"] <= window_end)
    return {"plan": {"results_in_window": due, "next_results_estimate": e["next_estimate"]},
            "retries": 0, "feedback": [], "hints": [],
            "_note": f"results {'due' if due else 'not due'} within the month"}


def quant(state: dict, run: Run) -> dict:
    try:
        px = run.tools.call("market_data", "quant", kind="prices")
        cal = run.tools.call("market_data", "quant", kind="earnings")
    except ToolError as exc:
        return {"error": [f"quant: {exc}"], "_note": "market data unavailable"}
    base, info = baseline(px["dates"], px["stock"], state["request"]["as_of"], cal["past_results"],
                          state["plan"]["results_in_window"])
    return {"baseline": handoff("baseline", base), "_note": f"baseline {base} ({info.get('fallback') or 'version ' + info['version']})"}


# --- research ----------------------------------------------------------------------------------

RESEARCH_TASK = ("Decide the next step. Reply {\"thought\": ..., \"action\": \"search_documents\" | \"search_news\" | "
                 "\"finish\", \"query\": ...}. Search for events that could move the stock. If a search found "
                 "nothing, rephrase it. If 'hints' is not empty, search for them first.")


def topic_similarity(a: str, b: str, company: str) -> float:
    """Similarity without the company's own name, which every passage shares."""
    name = set(words(company))
    strip = lambda t: " ".join(w for w in words(t) if w not in name)
    return similarity(strip(a), strip(b))


def same_story(items: list[dict], company: str) -> list[dict]:
    """News published within a few days of a filing it reports on becomes coverage of that filing."""
    filings = [i for i in items if i["kind"] == "filing"]
    kept = []
    for it in items:
        if it["kind"] == "news":
            match = next((f for f in filings
                          if abs((it["published"] - f["published"]).total_seconds()) <= SETTINGS.news_link_days * 86400
                          and topic_similarity(it["full_text"], f["full_text"], company) >= 0.2), None)
            if match is not None:
                match["coverage"].append({"id": it["id"], "headlines": it["full_text"]})
                continue
        kept.append(it)
    return kept


def _evidence_item(hit: dict, full: dict, as_of) -> dict:
    return {**hit, "published": published(full), "coverage": [], "lag_days": max((as_of - published(full)).days, 1),
            "full_text": full.get("text") or " | ".join(full["headlines"]),
            "event": full["event"], "reaction": full["reaction"], "novelty": full["novelty"]}


def research(state: dict, run: Run, agent: str = "research", full_text: bool = False) -> dict:
    req, tools = state["request"], run.tools
    hints = list(state.get("hints") or [])
    pool = {e["id"]: {**e, "coverage": []} for e in state.get("evidence") or []}
    observations, notes = [], []
    for _ in range(SETTINGS.max_research_steps):
        payload = {"question": f"What new information could move {req['company']} ({req['ticker']}) "
                               f"over the next {SETTINGS.horizon_days} trading days?",
                   "forecast_date": req["as_of"].date(), "observations": observations, "hints": hints}
        a = run.ask(agent, RESEARCH_TASK, payload, 150) or {"action": "finish"}
        if a.get("action") not in ("search_documents", "search_news"):
            break
        query = str(a.get("query") or "").strip()
        if not query or (a["action"], query.lower()) in {(o["tool"], o["query"].lower()) for o in observations}:
            observations.append({"tool": a["action"], "query": query, "results": [],
                                 "note": "Empty or repeated query, not run. Use different words or another tool."})
            notes.append(f"{a['action']}('{query}') -> refused: {'empty' if not query else 'repeated'}")
            continue
        try:
            hits = tools.call(a["action"], agent, query=query)
        except ToolError as exc:
            return {"error": [f"research: {exc}"], "_note": f"{a['action']} unavailable"}
        shown = [{k: h[k] for k in ("id", "header", "score", "extract")} for h in hits]
        if full_text or "clean_extracts" in tools.off:  # single agent, or the ablation without cleaned extracts
            shown = [{**r, "text": tools.get_passage(r["id"])["text"]} for r in shown]
        observations.append({"tool": a["action"], "query": query, "results": shown})
        notes.append(f"{a['action']}('{query}') -> {len(hits)}")
        for h in hits:
            if h["id"] not in pool or h["score"] > pool[h["id"]]["score"]:
                pool[h["id"]] = _evidence_item(h, tools.passage(h["id"]), req["as_of"])   # code adds the full passage
    searched = any(o["tool"] == "search_documents" and "note" not in o for o in observations)
    if not searched and not state.get("evidence"):
        try:                                            # safety net: the model never searched the filings
            hits = tools.call("search_documents", agent, query="", list_all=True)
        except ToolError as exc:
            return {"error": [f"research: {exc}"], "_note": "search_documents unavailable"}
        notes.append(f"no filing search by the model; code listed {len(hits)} recent filings")
        for h in hits:
            pool.setdefault(h["id"], _evidence_item(h, tools.passage(h["id"]), req["as_of"]))
    items = same_story(sorted(pool.values(), key=lambda e: -e["score"]), req["company"])[:SETTINGS.k_final]
    items = [handoff("evidence", e) for e in sorted(items, key=lambda e: e["published"], reverse=True)]
    for e in items:                                     # for the leakage audit
        tools.trace.used(agent, e["id"], e["published"])
        for c in e["coverage"]:
            tools.trace.used(agent, c["id"], published(tools.passage(c["id"])))
    return {"evidence": items, "hints": [], "_note": "; ".join(notes) + f" => {[e['id'] for e in items]}"}


# --- synthesis (Tree of Thought) ---------------------------------------------------------------

SYNTHESIS_TASK = ("Give two different readings of this evidence item for the stock over the next month. Reply "
                  "{\"readings\": [{\"effect\": \"fall\" | \"gain\" | \"spread\" | \"none\", \"size\": \"small\" | "
                  "\"medium\" | \"large\", \"quote\": exact words from the passage, \"reason\": one sentence, "
                  "\"confidence\": your probability from 0 to 1 that this reading is right}, ...]}. "
                  "'spread' means more likely to move a lot in either direction. If 'feedback' names this item, fix it.")
JUDGE_TASK = "Does the passage contain the quote and support the claim? Reply {\"supported\": true | false}."
NO_EFFECT = {"effect": "none", "size": None, "quote": "", "reason": "Already known or not material.", "ok": True}
WEIGHTS = {"reading": 0.8, "consistency": 0.2}


def read_item(item: dict, feedback: list, run: Run, agent: str = "synthesis", judge: str = "critic") -> dict:
    """The model's readings of one item plus "no effect"; each quote checked by the critic's model; base rate and novelty."""
    payload = {"item": {"id": item["id"], "header": item["header"], "text": item["full_text"],
                        "price_reaction": item["reaction"], "coverage": [c["headlines"] for c in item.get("coverage", [])]},
               "feedback": feedback}
    reply = run.ask(agent, SYNTHESIS_TASK, payload, 400) or {}
    blocked = None
    if isinstance(reply.get("tool_request"), dict):         # the model tried to act on something it read
        req = reply["tool_request"]
        try:
            run.tools.call(req.get("tool", ""), agent, query=req.get("query", ""))
        except (RoleError, ValueError) as exc:
            blocked = str(exc)
    readings = [dict(r) for r in reply.get("readings", [])[:2] if isinstance(r, dict) and not problems_reading(r)]
    for r in readings:
        if r["effect"] == "none":
            r["ok"] = True
            continue
        v = run.ask(judge, JUDGE_TASK, {"passage": item["full_text"], "quote": r["quote"], "claim": r.get("reason", "")}, 30) or {}
        r["ok"] = bool(v.get("supported"))
    readings.append(dict(NO_EFFECT))
    cm = run.tools.call("case_memory", agent, event=item["event"], reaction=item["reaction"], lag_days=item.get("lag_days"))
    for r in readings:
        r["source"] = item["id"]
        r["base_rate"] = cm[r["effect"]]
        r["novelty"] = item["novelty"] if r["effect"] != "none" else 1 - item["novelty"]
        r["score"] = 0.5 * r["base_rate"] + 0.5 * r["novelty"]
        r["n"] = cm["n"]
    return {"item": item["id"], "readings": readings, "blocked": blocked}


def readings_of(prepared: list[dict], choice: tuple) -> list[dict]:
    return [prepared[d]["readings"][i] for d, i in enumerate(choice)]


def node_score(prepared: list[dict], choice: tuple, base: dict) -> float:
    rs = readings_of(prepared, choice)
    _, factor = apply_ledger(base, rs)
    return WEIGHTS["reading"] * float(np.mean([r["score"] for r in rs])) + WEIGHTS["consistency"] * factor


def beam_search(prepared: list[dict], base: dict, width: int = SETTINGS.beam_width,
                margin: float = SETTINGS.prune_margin) -> tuple[tuple, list[dict]]:
    """Keep the best `width` partial ledgers per level; drop nodes more than `margin` below the best."""
    frontier, nodes = [()], []
    for d, entry in enumerate(prepared):
        children = []
        for parent in frontier:
            for i, r in enumerate(entry["readings"]):
                n = {"choice": parent + (i,), "depth": d + 1, "status": "", "score": None}
                nodes.append(n)
                if not r["ok"]:
                    n["status"] = "pruned: failed grounding"
                    continue
                n["score"] = node_score(prepared, n["choice"], base)
                n["shift"] = total_shift(readings_of(prepared, n["choice"]))
                children.append(n)
        children.sort(key=lambda n: (-round(n["score"], 9), round(n["shift"], 9)))
        kept = children[:width]
        for n in children[width:]:
            n["status"] = f"pruned: outside the top {width}"
        for n in kept[1:]:
            if n["score"] < kept[0]["score"] - margin:
                n["status"] = f"pruned: more than {margin} below the best"
        kept = [n for n in kept if not n["status"]]
        if width >= 2:                                  # the all-"no effect" node stays when there are 2+ slots
            neutral = next((n for n in children if all(r["effect"] == "none" for r in readings_of(prepared, n["choice"]))), None)
            if neutral is not None and neutral not in kept:
                if len(kept) >= width:
                    kept[-1]["status"] = "pruned: replaced by the no-effect node"
                    kept = kept[:-1]
                neutral["status"] = ""
                kept.append(neutral)
        for n in kept:
            n["status"] = "kept"
        frontier = [n["choice"] for n in kept] or frontier
    best = min((n for n in nodes if n["status"] == "kept" and n["depth"] == len(prepared)),
               key=lambda n: (-round(n["score"], 9), round(n["shift"], 9)), default=None)
    if best:
        best["status"] = "chosen"
    return (best["choice"] if best else ()), nodes


def is_ambiguous(prepared: list[dict], nodes: list[dict]) -> bool:
    """The best two complete ledgers lean opposite ways and score almost the same."""
    done = sorted((n for n in nodes if n["status"] in ("kept", "chosen") and n["depth"] == len(prepared)),
                  key=lambda n: -n["score"])
    if len(done) < 2:
        return False
    d1, d2 = (direction(readings_of(prepared, n["choice"])) for n in done[:2])
    return d1 * d2 < 0 and done[0]["score"] - done[1]["score"] < AMBIGUITY_GAP


def synthesis(state: dict, run: Run, agent: str = "synthesis", judge: str = "critic") -> dict:
    if state.get("error"):
        return {"_note": "skipped: " + "; ".join(state["error"])}
    base, feedback = state["baseline"], state.get("feedback") or []
    prepared = [read_item(it, feedback, run, agent, judge) for it in state["evidence"][:SETTINGS.k_final]]
    if run.options.get("readings") == "single":         # single pass: first reading that passed its check
        choice = tuple(next(i for i, r in enumerate(p["readings"]) if r["ok"]) for p in prepared)
        nodes: list[dict] = []
    else:
        choice, nodes = beam_search(prepared, base)
    chosen = readings_of(prepared, choice)
    run.tools.call("branch_log", agent, record={"nodes": nodes, "choice": choice,
                                                "readings": [(r["source"], r["effect"], r["size"]) for r in chosen]})
    final, _ = apply_ledger(base, chosen)
    lines = [handoff("line", {"source": r["source"], "effect": r["effect"], "size": r["size"], "quote": r["quote"],
                              "reason": r.get("reason", ""), "shift": shift_for(r["effect"], r["size"]),
                              "n": r.get("n"), "confidence": r.get("confidence")})
             for r in chosen if r["effect"] != "none"]
    blocked = [p["blocked"] for p in prepared if p["blocked"]]
    note = ", ".join(f"{r['source']}: {r['effect']}" + (f" {r['size']}" if r["size"] else "") for r in chosen)
    return {"ledger": lines, "forecast": final,
            "tot": {"nodes": len(nodes), "prepared": prepared, "choice": choice, "blocked": blocked,
                    "ambiguous": is_ambiguous(prepared, nodes)},
            "_note": note + (f" | blocked tool request: {blocked}" if blocked else "")}


# --- critic ------------------------------------------------------------------------------------

EVENT_WORDS = r"\b(resign\w*|step(s|ping)? down|appoint\w*|results|merger|acquir\w*|agreement|settle\w*)\b"


def norm(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def critic(state: dict, run: Run) -> dict:
    if state.get("error"):
        return {"verdict": {"ok": False, "route": "fallback", "reasons": state["error"]}, "_note": "tool failure"}
    evidence = {e["id"]: e for e in state["evidence"]}
    reasons, hints, to_research = [], [], False
    for e in evidence.values():                       # news about a company event needs its filing
        if e["kind"] == "news" and re.search(EVENT_WORDS, e["full_text"], re.I):
            near = [f for f in evidence.values() if f["kind"] == "filing"
                    and abs((f["published"] - e["published"]).total_seconds()) <= SETTINGS.news_link_days * 86400]
            if not near:
                headline = e["full_text"].split(" | ")[0]
                reasons.append(f"{e['id']}: the news reports a company event ('{headline}') but the filing "
                               f"behind it is not in the evidence")
                hints.append(headline)
                to_research = True
    for line in state.get("ledger") or []:            # sources, dates, quotes
        p = run.tools.call("get_passage", "critic", item_id=line["source"])
        if p is None or line["source"] not in evidence:
            reasons.append(f"{line['source']}: cited source is not in the evidence or was published after the forecast date")
        elif norm(line["quote"]) not in norm(p["text"]):
            reasons.append(f"{line['source']}: the quote '{line['quote']}' does not appear in the passage")
    fc, base = state.get("forecast") or {}, state.get("baseline") or {}
    reasons += [f"forecast {p}" for p in problems_baseline(fc)] if fc else []
    if fc and base and max(abs(fc[k] - base[k]) for k in OUTCOMES) > SETTINGS.max_total_shift + 0.05:
        reasons.append("forecast moves more than the cap away from the baseline")

    retries = state.get("retries", 0)
    protocol = run.options.get("protocol", "two_way")
    if protocol == "one_way":                         # no feedback loop: whatever arrives is published
        v = {"ok": True, "route": "finalise", "reasons": [], "unchecked": reasons}
        return {"verdict": handoff("verdict", v), "_note": "one-way: not checked" + (f" ({len(reasons)} problems)" if reasons else "")}
    if not reasons:
        return {"verdict": handoff("verdict", {"ok": True, "route": "finalise", "reasons": []}), "_note": "accepted"}
    if retries >= SETTINGS.critic_retries:
        v = {"ok": False, "route": "fallback", "reasons": reasons + [f"still failing after {retries} retries"]}
        return {"verdict": handoff("verdict", v), "_note": "retries used up"}
    route = "research" if to_research else "synthesis"
    feedback = ["The critic rejected the forecast."] if protocol == "two_way_no_reason" else reasons
    hints = [] if protocol == "two_way_no_reason" else hints
    v = {"ok": False, "route": route, "reasons": reasons}
    return {"verdict": handoff("verdict", v), "retries": retries + 1, "feedback": feedback, "hints": hints,
            "_note": f"rejected -> {route}: " + " | ".join(reasons)}


def route_after_critic(state: dict) -> str:
    return state["verdict"]["route"]


# --- the two endings ---------------------------------------------------------------------------

def finalise(state: dict, run: Run) -> dict:
    return {"final": {"status": "accepted", "probabilities": state["forecast"], "ledger": state["ledger"],
                      "baseline": state["baseline"]}, "_note": "forecast published"}


def fallback(state: dict, run: Run) -> dict:
    base = state.get("baseline") or floor_and_normalise({"fall": 30.85, "flat": 38.3, "gain": 30.85})
    why = (state.get("verdict") or {}).get("reasons") or state.get("error") or []
    return {"final": {"status": "baseline", "probabilities": base, "ledger": [], "baseline": base, "why": why},
            "_note": "baseline reported: " + "; ".join(why)[:200]}
