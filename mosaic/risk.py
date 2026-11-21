"""The risk gate and the analyst's review (Checkpoint 6).

After a forecast is finalised (or the baseline is reported), the gate counts warning signs, all in code.
No sign: publish. One: publish with a flag. Two or more: hold for the analyst, who can keep, resize or
strike each ledger line. Every decision is logged by event type, so repeated corrections show where the
agent needs work.
"""

from __future__ import annotations

from mosaic.config import OUTCOMES, SETTINGS
from mosaic.ledger import apply_ledger, shift_for


def risk_triggers(state: dict, tool_calls: list[dict] | None = None) -> list[str]:
    final, triggers = state["final"], []
    base, probs, lines = final["baseline"], final["probabilities"], final["ledger"]
    moved = max(abs(probs[k] - base[k]) for k in OUTCOMES)
    if moved > SETTINGS.risk_high_impact:
        triggers.append(f"high impact: an outcome moved {moved:.1f} points")
    rare = [line["source"] for line in lines if line.get("n") is not None and line["n"] < SETTINGS.risk_rare_cases]
    if rare:
        triggers.append(f"thin evidence: few past cases behind {', '.join(rare)}")
    elif len(lines) == 1 and moved > SETTINGS.risk_single_source:
        triggers.append(f"thin evidence: a {moved:.1f}-point move rests on one source")
    if (state.get("tot") or {}).get("ambiguous"):
        triggers.append("ambiguity: the best two ledgers lean opposite ways")
    unsure = [line["source"] for line in lines
              if line.get("confidence") is not None and line["confidence"] < SETTINGS.risk_low_confidence]
    if unsure:
        triggers.append(f"low confidence: the model doubts {', '.join(unsure)}")
    if state.get("retries", 0) or final["status"] == "baseline":
        triggers.append("troubled run: " + ("baseline used after a failure" if final["status"] == "baseline"
                                            else f"the critic sent work back {state['retries']} time(s)"))
    refused = [c for c in tool_calls or [] if c.get("status") == "refused: not in role"]
    if refused:                                       # the model tried to act on something it read (Lab 6.2)
        who = sorted({f"{c['node']} asked for {c['tool']}" for c in refused})
        triggers.append("refused tool call: " + "; ".join(who))
    return triggers


def review_level(triggers: list[str]) -> str:
    return "publish" if not triggers else "flag" if len(triggers) == 1 else "hold"


class ReviewLog:
    """Every analyst decision, for the feedback loop."""

    def __init__(self):
        self.entries: list[dict] = []

    def summary(self) -> dict[str, dict[str, int]]:
        """Actions by event type: where the agent most often needs correcting."""
        out: dict[str, dict[str, int]] = {}
        for r in self.entries:
            row = out.setdefault(r["event"], {"keep": 0, "resize": 0, "strike": 0})
            row[r["action"]] += 1
        return out


def apply_review(state: dict, decisions: dict, log: ReviewLog | None = None) -> dict:
    """Analyst decisions per source: "keep", "strike" or ("resize", size). Returns the reviewed final forecast."""
    final = state["final"]
    events = {e["id"]: e.get("event", "?") for e in state.get("evidence") or []}
    kept, record = [], []
    for line in final["ledger"]:
        d = decisions.get(line["source"], "keep")
        action, size = (d[0], d[1]) if isinstance(d, tuple) else (d, line["size"])
        if action not in ("keep", "strike", "resize") or (action == "resize" and size not in SETTINGS.strength_points):
            raise ValueError(f"unknown decision for {line['source']}: {d!r}")
        if log is not None:
            log.entries.append({"source": line["source"], "event": events.get(line["source"], "?"),
                                "effect": line["effect"], "action": action, "from": line["size"],
                                "to": size if action == "resize" else None})
        record.append({"source": line["source"], "effect": line["effect"], "agent_size": line["size"],
                       "action": action, "analyst_size": None if action == "strike" else size})
        if action != "strike":
            kept.append({**line, "size": size, "shift": shift_for(line["effect"], size)})
    probs, _ = apply_ledger(final["baseline"], kept)
    return {**final, "ledger": kept, "probabilities": probs, "status": "reviewed by analyst",
            "agent_probabilities": final["probabilities"], "agent_ledger": final["ledger"], "review_record": record}
