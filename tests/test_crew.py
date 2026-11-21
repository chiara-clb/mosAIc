"""CrewAI at runtime, tested with a fake `crewai` module (the real package is not installed here)."""

import sys
import types

import pytest

from mosaic import minigraph
from mosaic.crew import INSTALL_HINT, crew_agents, crew_llm, crewai_chat
from mosaic.llm import agent_messages, make_model, scripted_chat
from mosaic.roles import ROLES
from mosaic.scenarios import run_scenario

ENGINE = (minigraph.StateGraph, minigraph.START, minigraph.END, "built-in runner")


def fake_crewai(with_kickoff=True, answer=None):
    """A stand-in for crewai: records what agents are built with and what they are asked."""
    mod = types.ModuleType("crewai")
    mod.prompts = []
    respond = answer or (lambda prompt: scripted_chat()([{"role": "user", "content": prompt}], 300))

    class LLM:
        def __init__(self, model, **kw):
            self.model, self.kw = model, kw

    class Output:
        def __init__(self, raw):
            self.raw = raw

    class Agent:
        def __init__(self, role, goal, backstory, llm=None, tools=None, allow_delegation=True, verbose=True, **kw):
            self.role, self.goal, self.backstory, self.llm = role, goal, backstory, llm
            self.tools, self.allow_delegation = tools, allow_delegation

    if with_kickoff:
        def kickoff(self, prompt):
            mod.prompts.append((self.role, prompt))
            return Output(respond(prompt))
        Agent.kickoff = kickoff

    class Task:
        def __init__(self, description, expected_output, agent):
            self.description, self.agent = description, agent

    class Process:
        sequential = "sequential"

    class Crew:
        def __init__(self, agents, tasks, process, verbose=False):
            self.tasks = tasks

        def kickoff(self):
            t = self.tasks[0]
            mod.prompts.append((t.agent.role, t.description))
            return Output(respond(t.description))

    mod.LLM, mod.Agent, mod.Task, mod.Process, mod.Crew = LLM, Agent, Task, Process, Crew
    return mod


def test_one_agent_per_role_card_and_no_tools_on_the_crewai_side():
    cw = fake_crewai()
    agents = crew_agents(llm="LLM", crewai=cw)
    assert list(agents) == list(ROLES)
    for name, a in agents.items():
        assert (a.role, a.goal, a.backstory) == (ROLES[name]["role"], ROLES[name]["goal"], ROLES[name]["backstory"])
        assert a.tools == [] and a.allow_delegation is False and a.llm == "LLM"


def test_claude_model_names_get_the_provider_prefix():
    llm = crew_llm("claude-haiku-5-5", fake_crewai())
    assert llm.model == "anthropic/claude-haiku-5-5" and llm.kw["temperature"] == 0.0
    assert crew_llm("anthropic/claude-sonnet-5-5", fake_crewai()).model == "anthropic/claude-sonnet-5-5"


@pytest.mark.parametrize("with_kickoff", [True, False])
def test_calls_are_routed_to_the_agent_named_in_the_prompt(with_kickoff):
    cw = fake_crewai(with_kickoff, answer=lambda prompt: '{"supported": true}')
    chat = crewai_chat("claude-haiku-5-5", crewai=cw)
    out = chat(agent_messages("critic", "Check the quote.", {"passage": "p", "quote": "q"}), 30)
    assert out == '{"supported": true}'
    role, prompt = cw.prompts[-1]
    assert role == ROLES["critic"]["role"] and "Check the quote." in prompt and '"quote": "q"' in prompt


def test_unknown_agent_is_an_error():
    chat = crewai_chat(crewai=fake_crewai())
    with pytest.raises(ValueError):
        chat([{"role": "system", "content": "Agent: trader\n"}, {"role": "user", "content": "INPUT:\n{}"}])


def test_a_full_demo_forecast_runs_through_crewai(monkeypatch):
    cw = fake_crewai()
    monkeypatch.setitem(sys.modules, "crewai", cw)
    state, run, row = run_scenario("1 normal", model_factory=lambda script: make_model("crewai", model_name="claude-haiku-5-5"),
                                   engine=ENGINE)
    assert run.model.name == "crewai:claude-haiku-5-5" and row["status"] == "accepted" and row["items right"] == "4/4"
    asked = {role for role, _ in cw.prompts}
    assert asked == {ROLES["research"]["role"], ROLES["synthesis"]["role"], ROLES["critic"]["role"]}


def test_clear_message_when_crewai_is_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "crewai", None)
    with pytest.raises(ImportError, match="pip install"):
        make_model("crewai")
    assert "crew" in INSTALL_HINT


def test_with_the_real_package_if_installed():
    crewai = pytest.importorskip("crewai")
    agents = crew_agents(llm=crew_llm("claude-haiku-5-5", crewai), crewai=crewai)
    assert set(agents) == set(ROLES) and agents["research"].role == ROLES["research"]["role"]


def test_cli_runs_the_demo_through_crewai(monkeypatch, tmp_path, capsys):
    from mosaic.cli import main
    monkeypatch.setitem(sys.modules, "crewai", fake_crewai())
    assert main(["--demo", "--backend", "crewai", "--model", "claude-haiku-5-5", "--out", str(tmp_path)]) == 0
    assert "Model backend: crewai" in capsys.readouterr().out
    report = (tmp_path / "1_normal" / "ACME-2025-06-02" / "report.md").read_text()
    assert "crewai:claude-haiku-5-5" in report


def test_app_offers_the_crewai_backend():
    from pathlib import Path
    app = (Path(__file__).resolve().parents[1] / "app.py").read_text()
    assert '"crewai"]' in app
