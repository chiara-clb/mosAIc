"""Plain tables from a finished run, for the Streamlit app (and anything else that shows a forecast).

Kept separate from the app so they can be tested without Streamlit.
"""

from __future__ import annotations

from mosaic.config import OUTCOMES
from mosaic.trace import leakage_audit

LABELS = {"fall": "Large fall", "flat": "Flat", "gain": "Large gain"}


def forecast_rows(final: dict) -> list[dict]:
    """One row per outcome: baseline, forecast and the change in points.

    After an analyst review: baseline, the agent's forecast, the analyst's forecast, and each change.
    """
    if "agent_probabilities" in final:
        a, b, r = final["agent_probabilities"], final["baseline"], final["probabilities"]
        return [{"Outcome": LABELS[k], "Baseline %": round(b[k], 1), "Agent %": round(a[k], 1),
                 "Analyst %": round(r[k], 1), "Agent change": round(a[k] - b[k], 1),
                 "Analyst change": round(r[k] - a[k], 1)} for k in OUTCOMES]
    return [{"Outcome": LABELS[k], "Baseline %": round(final["baseline"][k], 1),
             "Forecast %": round(final["probabilities"][k], 1),
             "Change (points)": round(final["probabilities"][k] - final["baseline"][k], 1)} for k in OUTCOMES]


def review_summary(final: dict) -> str | None:
    """"Reviewed by the analyst: 1 kept, 1 struck, 0 resized", or None if not reviewed."""
    rec = final.get("review_record")
    if rec is None:
        return None
    n = {a: sum(r["action"] == a for r in rec) for a in ("keep", "strike", "resize")}
    return f"Reviewed by the analyst: {n['keep']} kept, {n['strike']} struck, {n['resize']} resized"


def ledger_rows(state: dict, final: dict | None = None) -> list[dict]:
    """The ledger lines. After an analyst review, every line the agent proposed, with the analyst's decision."""
    final = final or state["final"]
    dates = {e["id"]: e["published"] for e in state.get("evidence") or []}
    decided = {r["source"]: r for r in final.get("review_record") or []}
    rows = []
    for x in final.get("agent_ledger") or final["ledger"]:
        row = {"Source": x["source"], "Published": f"{dates[x['source']]:%Y-%m-%d}" if x["source"] in dates else "-",
               "Effect": x["effect"], "Size": x["size"]}
        if decided:
            r = decided.get(x["source"], {"action": "keep"})
            row["Size"] = f"{x['size']} (agent)"
            row["Analyst decision"] = {"keep": "kept", "strike": "struck"}.get(r["action"]) or f"resized to {r['analyst_size']}"
        row.update({"Quote": x["quote"], "Reason": x.get("reason", ""), "Past cases": x.get("n")})
        rows.append(row)
    return rows


def evidence_rows(state: dict) -> list[dict]:
    return [{"Id": e["id"], "Kind": e["kind"], "Published": f"{e['published']:%Y-%m-%d}", "Header": e["header"],
             "News on the same event": ", ".join(c["id"] for c in e.get("coverage") or []),
             "Novelty": round(e.get("novelty", 0), 2), "Price reaction": e.get("reaction") or "-"}
            for e in state.get("evidence") or []]


def tot_rows(state: dict) -> list[dict]:
    """Every reading the Tree of Thought considered, with its scores, and which one was chosen."""
    tot = state.get("tot") or {}
    rows = []
    for depth, p in enumerate(tot.get("prepared") or []):
        chosen = tot["choice"][depth] if depth < len(tot.get("choice") or ()) else None
        for i, r in enumerate(p["readings"]):
            rows.append({"Item": p["item"], "Reading": r["effect"] + (f" {r['size']}" if r.get("size") else ""),
                         "Quote check": "passed" if r["ok"] else "failed",
                         "Base rate": round(r["base_rate"], 2), "Novelty": round(r["novelty"], 2),
                         "Score": round(r["score"], 2), "Chosen": "yes" if i == chosen else ""})
    return rows


def step_rows(state: dict) -> list[dict]:
    return [{"Step": t["node"], "Model calls": t["model_calls"], "Tool calls": t["tool_calls"], "Note": t["note"]}
            for t in state.get("trace") or []]


def tool_rows(run) -> list[dict]:
    return [{"Agent": c["node"], "Tool": c["tool"], "Status": c["status"],
             "Arguments": ", ".join(f"{k}={v}" for k, v in (c.get("args") or {}).items())} for c in run.tools.calls]


def run_facts(state: dict, run) -> dict:
    leaks = leakage_audit(run.tools.trace)
    final = state["final"]
    return {"status": final["status"], "review": final.get("review", "-"), "triggers": final.get("triggers") or [],
            "why": final.get("why") or [], "retries": state.get("retries", 0), "model": run.model.name,
            "model_calls": run.model.calls, "tool_calls": len(run.tools.calls),
            "leakage": "clean" if not leaks else f"{len(leaks)} sources published too late"}


STEP_NAMES = {"coordinator": "Coordinator", "quant": "Quant agent", "research": "Research agent",
              "join": "Waiting for quant and research", "synthesis": "Synthesis agent (Tree of Thought)",
              "critic": "Critic", "finalise": "Finalising the forecast", "fallback": "Falling back to the baseline",
              "risk_gate": "Risk gate", "publish": "Published", "flag": "Published with a flag",
              "hold": "Held for the analyst"}


def _short(value, limit: int = 60) -> str:
    text = str(value)
    return text if len(text) <= limit else text[:limit - 1] + "…"


def progress_line(event: dict) -> str | None:
    """One readable line for a live event, or None for events not worth showing."""
    node, kind = event.get("node", ""), event.get("kind")
    who = STEP_NAMES.get(node, node)
    if kind == "progress":
        if event.get("message") != "started":
            return event.get("message")
        return None if node in ("publish", "flag", "hold") else f"{who}: working…"
    if kind == "tool_call":
        args = ", ".join(f"{k}={_short(v, 40)!s}" for k, v in (event.get("args") or {}).items())
        status = event.get("status", "")
        mark = "" if status == "ok" else f"  ({status})"
        return f"   {who} → {event.get('tool')}({args}){mark}"
    if kind == "step":
        note = f": {_short(event['note'], 90)}" if event.get("note") else ""
        calls = f" [{event.get('model_calls', 0)} model calls]" if event.get("model_calls") else ""
        if node in ("publish", "flag", "hold"):
            return f"✓ {who}{note}"
        return f"✓ {who} done{calls}{note}"
    return None


def _table(rows: list[dict]) -> list[str]:
    if not rows:
        return ["  (none)"]
    return ["  " + " | ".join(f"{k}: {v}" for k, v in r.items()) for r in rows]


def text_log(state: dict, run, final: dict | None = None) -> str:
    """The whole run as plain text, to copy and paste: result, ledger, evidence, Tree of Thought, every event.

    `final`: the analyst-reviewed forecast, if there is one.
    """
    req, final, facts = state["request"], final or state["final"], run_facts(state, run)
    out = [f"MosAIc run {run.tools.trace.run_id}: {req['company']} ({req['ticker']}) as of {req['as_of']:%Y-%m-%d}",
           f"Model: {facts['model']}, {facts['model_calls']} calls | tool calls: {facts['tool_calls']} | "
           f"retries: {facts['retries']} | leakage audit: {facts['leakage']}",
           f"Status: {final['status']} | risk gate: {facts['review']}" +
           (f" ({'; '.join(facts['triggers'])})" if facts["triggers"] else ""),
           "", "FORECAST"] + _table(forecast_rows(final))
    if review_summary(final):
        out += ["", review_summary(final).upper()]
    out += ["", "LEDGER"] + _table(ledger_rows(state, final))
    out += ["", "EVIDENCE"] + _table(evidence_rows(state))
    out += ["", "TREE OF THOUGHT"] + _table(tot_rows(state))
    out += ["", "EVENTS"]
    for e in run.tools.trace.events:
        line = progress_line(e)
        if line:
            out.append(line)
        if e["kind"] == "model_call":
            mark = "" if e.get("parsed") else "  ** reply was not valid JSON **"
            out.append(f"   {STEP_NAMES.get(e['node'], e['node'])} model reply (limit {e.get('max_tokens')} tokens){mark}:")
            out += ["      " + l for l in str(e.get("reply", "")).splitlines() or [""]]
        elif e["kind"] == "source_used":
            out.append(f"   {STEP_NAMES.get(e['node'], e['node'])} used {e.get('source')} (published {e.get('published')})")
    if state.get("error"):
        out += ["", "ERRORS"] + ["  " + str(x) for x in state["error"]]
    return "\n".join(out) + "\n"
