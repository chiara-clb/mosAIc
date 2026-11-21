import pytest

from mosaic import minigraph
from mosaic.ablation import ABLATION, COMPARISON, run_one, write_results

ENGINE = (minigraph.StateGraph, minigraph.START, minigraph.END, "built-in runner")


@pytest.fixture(scope="module")
def tables(tmp_path_factory):
    out = tmp_path_factory.mktemp("results")
    comparison, ablation = write_results(out, engine=ENGINE)
    return comparison, ablation, out


def test_files_are_written(tables):
    _, _, out = tables
    assert {p.name for p in out.iterdir()} == {"comparison.csv", "ablation.csv", "scenario_detail.csv", "ablation.md"}
    assert "## The design with one guardrail switched off" in (out / "ablation.md").read_text()


def test_the_design_lets_nothing_through(tables):
    comparison, ablation, _ = tables
    for row in (comparison.loc["Critic with reasons (the design)"], ablation.loc["All guardrails on (the design)"]):
        assert row["made-up quotes published"] == 0 and row["injected tool calls"] == 0 and row["late data used"] == 0
        assert row["held"] == 1


def test_each_switched_off_guardrail_lets_its_failure_through(tables):
    _, ab, _ = tables
    assert ab.loc["No critic loop", "made-up quotes published"] >= 1
    assert ab.loc["No cleaned extracts", "injected tool calls"] >= 1
    assert ab.loc["No role check on tools", "injected tool calls"] >= 1
    assert ab.loc["No risk gate", "held"] == 0 and ab.loc["No risk gate", "flagged"] == 0
    assert ab.loc["No forecast-date rule", "late data used"] >= 1


def test_the_comparison_shows_the_trade_offs(tables):
    c, _, _ = tables
    right = lambda name: int(c.loc[name, "items right"].split("/")[0])
    assert right("One-way with Tree of Thought") > right("Linear pipeline (one-way, first reading)")
    assert c.loc["Critic rejects without reasons", "baseline reported"] > c.loc["Critic with reasons (the design)", "baseline reported"]
    assert c.loc["Single agent with all tools", "injected tool calls"] >= 1
    assert c.loc["Single agent with all tools", "made-up quotes published"] >= 1
    assert c.loc["One-way with Tree of Thought", "model calls (mean)"] < c.loc["Critic with reasons (the design)", "model calls (mean)"]


def test_scenario_three_without_the_critic_publishes_the_made_up_quote():
    row = run_one("3 wrong reading", ABLATION["No critic loop"], ENGINE)
    assert row["made-up quote published"] and row["status"] == "accepted"
    assert set(COMPARISON) and set(ABLATION)
