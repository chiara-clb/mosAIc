"""CrewAI at runtime: the role cards become CrewAI agents, and every model call goes through them.

LangGraph still decides the order of the steps (mosaic/graph.py). CrewAI holds the agents: each
model-backed step is answered by the CrewAI agent with that role, goal and backstory. No tools are
attached on the CrewAI side, so every tool call still goes through MosAIc's own door, which checks the
caller's role and applies the forecast-date rule.

    pip install -e ".[crew,llm]"
    python -m mosaic.cli --demo --backend crewai --model claude-haiku-5-5

CrewAI calls used: crewai.LLM(model=..., temperature=..., max_tokens=...), crewai.Agent(role=..., goal=...,
backstory=..., llm=..., tools=[], allow_delegation=False, verbose=False), then Agent.kickoff(prompt)
(CrewAI 0.108 and later) or, on older versions, Task(description=..., expected_output=..., agent=...) run by
Crew(agents=[...], tasks=[...], process=Process.sequential).kickoff(). The reply text is read from `.raw`.
"""

from __future__ import annotations

import re

from mosaic.config import SETTINGS
from mosaic.roles import ROLES

INSTALL_HINT = "CrewAI is not installed. Install it with: pip install -e '.[crew]'"


def load_crewai():
    try:
        import crewai
    except ImportError as exc:
        raise ImportError(INSTALL_HINT) from exc
    if crewai is None:                                 # a module explicitly blocked in sys.modules
        raise ImportError(INSTALL_HINT)
    return crewai


def crew_llm(model_name: str | None = None, crewai=None, temperature: float | None = None, max_tokens: int = 400):
    """CrewAI's LLM wrapper for a Claude model. Bare names get the "anthropic/" prefix LiteLLM expects."""
    crewai = crewai or load_crewai()
    name = model_name or SETTINGS.llm_model
    if "/" not in name:
        name = f"anthropic/{name}"
    from mosaic.llm import NEW_CLAUDE
    if NEW_CLAUDE.search(name):       # Claude 5: no temperature, and thinking off so it does not eat max_tokens
        return crewai.LLM(model=name, max_tokens=max_tokens, thinking={"type": "disabled"})
    t = SETTINGS.llm_temperature if temperature is None else temperature
    return crewai.LLM(model=name, temperature=t, max_tokens=max_tokens)


def crew_agents(roles: dict | None = None, llm=None, crewai=None) -> dict:
    """One CrewAI agent per role card, with no tools on the CrewAI side."""
    crewai = crewai or load_crewai()
    agents = {}
    for name, card in (roles or ROLES).items():
        agents[name] = crewai.Agent(role=card["role"], goal=card["goal"], backstory=card["backstory"], llm=llm,
                                    tools=[], allow_delegation=False, verbose=False)
    return agents


def _text(out) -> str:
    raw = getattr(out, "raw", None)
    return str(raw if raw is not None else out).strip()


def ask_agent(agent, prompt: str, crewai=None) -> str:
    """One completion from one CrewAI agent: Agent.kickoff when available, else a one-task Crew."""
    if hasattr(agent, "kickoff"):
        return _text(agent.kickoff(prompt))
    crewai = crewai or load_crewai()
    task = crewai.Task(description=prompt, expected_output="One JSON object, as described in the task.", agent=agent)
    crew = crewai.Crew(agents=[agent], tasks=[task], process=crewai.Process.sequential, verbose=False)
    return _text(crew.kickoff())


def crewai_chat(model_name: str | None = None, crewai=None, roles: dict | None = None):
    """A chat function (messages, max_tokens) -> text that answers through the CrewAI agent named in the prompt."""
    crewai = crewai or load_crewai()
    agents = crew_agents(roles, crew_llm(model_name, crewai), crewai)

    def chat(messages: list[dict], max_tokens: int = 300) -> str:
        m = re.match(r"Agent: (\w+)", messages[0]["content"])
        if not m or m.group(1) not in agents:
            raise ValueError(f"no CrewAI agent for {m.group(1) if m else 'this prompt'!r}")
        prompt = "\n\n".join(msg["content"] for msg in messages)
        return ask_agent(agents[m.group(1)], prompt, crewai)

    chat.agents = agents
    return chat
