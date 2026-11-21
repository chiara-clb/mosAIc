from mosaic import minigraph
from mosaic.llm import DEFAULT_SCRIPT, make_model
from mosaic.scenarios import FABRICATED_F1, score_run
from mosaic.single_agent import run_single_agent
from mosaic.world import make_demo_world

ENGINE = (minigraph.StateGraph, minigraph.START, minigraph.END, "built-in runner")


def test_single_agent_runs_end_to_end():
    state, run = run_single_agent(make_demo_world(), make_model("scripted"), engine=ENGINE)
    assert [t["node"] for t in state["trace"]] == ["coordinator", "quant", "analyst", "finalise"]
    assert state["final"]["status"] == "accepted" and "review" not in state["final"]


def test_single_agent_follows_hidden_instructions():
    state, run = run_single_agent(make_demo_world(), make_model("scripted", {"gullible": True}), engine=ENGINE)
    assert score_run(state, run)["injected tool calls"] >= 1


def test_single_agent_publishes_a_made_up_quote():
    script = {"readings": {**DEFAULT_SCRIPT["readings"], "F1": FABRICATED_F1}, "lenient_judge": True}
    state, run = run_single_agent(make_demo_world(), make_model("scripted", script), engine=ENGINE)
    assert score_run(state, run)["made-up quote published"]
