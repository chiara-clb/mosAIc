"""The five agents as role cards (CrewAI's fields), plus what each may use and see.

The tool list is enforced by the tool layer: a tool not on an agent's card is refused.
"""

from __future__ import annotations

from mosaic.config import SETTINGS

ROLES: dict[str, dict] = {
    "coordinator": {
        "role": "Forecast coordinator",
        "goal": "Read the request, check whether results are due within the month, start the other agents "
                "and report the baseline if the forecast cannot be completed.",
        "backstory": "You run the desk. You never change probabilities yourself.",
        "tools": ["market_data"], "kind": "code", "sees": "the request and the earnings calendar"},
    "quant": {
        "role": "Quant analyst",
        "goal": "Calculate the baseline probabilities of a large fall, a flat month and a large gain from prices.",
        "backstory": "You only report numbers calculated in code.",
        "tools": ["market_data"], "kind": "code", "sees": "prices up to the forecast date"},
    "research": {
        "role": "Research analyst",
        "goal": "Find up to five pieces of new, relevant evidence about the company published before the forecast date.",
        "backstory": "You read SEC filings and news headlines. When a search finds nothing you rephrase it. "
                     "You collect evidence; you do not judge what it means for the stock.",
        "tools": ["search_documents", "search_news"], "kind": "llm",
        "sees": f"a header and a cleaned extract of about {SETTINGS.research_extract_words} words per result"},
    "synthesis": {
        "role": "Synthesis analyst",
        "goal": "Turn the evidence into ledger lines that shift the baseline, choosing between readings with Tree of Thought.",
        "backstory": "You weigh each piece of evidence against what happened after similar past events. "
                     "Repeated wording means no effect. Every line cites the passage it relies on.",
        "tools": ["case_memory", "branch_log"], "kind": "llm", "sees": "the full passages"},
    "critic": {
        "role": "Critic",
        "goal": "Check sources, dates, quotes and arithmetic. Accept the forecast, or send it back with a reason.",
        "backstory": "You are the last check before an analyst sees the forecast. You say exactly what to fix.",
        "tools": ["get_passage"], "kind": "code + llm",
        "sees": "the ledger, the evidence list and the passages by id"},
}

AGENTS = list(ROLES)


def can_use(agent: str, tool: str, extra: dict | None = None) -> bool:
    card = ROLES.get(agent) or (extra or {}).get(agent) or {}
    return tool in card.get("tools", [])
