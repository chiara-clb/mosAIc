"""A forecast as a short Markdown report: probabilities, ledger, review decision and how the run went."""

from __future__ import annotations

from mosaic.config import OUTCOMES
from mosaic.trace import leakage_audit


def _probs(p: dict) -> str:
    return " | ".join(f"{p[k]:.1f}%" for k in OUTCOMES)


def render_markdown(state: dict, run) -> str:
    final, req = state["final"], state["request"]
    lines = [f"# {req['company']} ({req['ticker']}), forecast as of {req['as_of']:%Y-%m-%d}", "",
             "Probability of each outcome over the next 21 trading days.", "",
             "| | Large fall | Flat | Large gain |", "| --- | --- | --- | --- |",
             f"| Baseline (prices only) | {_probs(final['baseline'])} |",
             "| **Forecast** | " + " | ".join(f"**{final['probabilities'][k]:.1f}%**" for k in OUTCOMES) + " |", "",
             f"**Status:** {final['status']}. **Review:** {final.get('review', '-')}"
             + (f" ({'; '.join(final['triggers'])})" if final.get("triggers") else "") + "."]
    if final.get("why"):
        lines += ["", "**Why the baseline was reported:** " + "; ".join(final["why"])]
    lines += ["", "## Ledger", ""]
    if final["ledger"]:
        dates = {e["id"]: e["published"] for e in state.get("evidence") or []}
        when = lambda src: f"{dates[src]:%Y-%m-%d}" if src in dates else "-"
        lines += ["| Source | Published | Effect | Size | Quote | Reason |", "| --- | --- | --- | --- | --- | --- |"]
        lines += [f"| {x['source']} | {when(x['source'])} | {x['effect']} | {x['size']} | \"{x['quote']}\" | {x.get('reason', '')} |"
                  for x in final["ledger"]]
    else:
        lines.append("No adjustments: the forecast is the baseline.")
    evidence = state.get("evidence") or []
    if evidence:
        lines += ["", "## Evidence read", ""]
        for e in evidence:
            cov = f" (covered by news {', '.join(c['id'] for c in e['coverage'])})" if e.get("coverage") else ""
            lines.append(f"- **{e['id']}** {e['header']}{cov}")
    leaks = leakage_audit(run.tools.trace)
    lines += ["", "## How the run went", "",
              "Path: " + " → ".join(t["node"] for t in state["trace"]), "",
              f"Model: {run.model.name}, {run.model.calls} calls. Tool calls: {len(run.tools.calls)}. "
              f"Leakage audit: {'clean' if not leaks else f'{len(leaks)} sources published too late'}.", ""]
    return "\n".join(lines)
