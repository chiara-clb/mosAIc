"""Checkpoint 5 notebook: the comparison shows the trade-offs the write-up claims."""

import pytest

from tests.notebook import fake_langgraph as lg
from tests.notebook.core import load_multi_agent_core

NS = load_multi_agent_core()
NS["pd"] = pytest.importorskip("pandas")
GRAPH = NS["build_graph"](lg.StateGraph, lg.START, lg.END)
SINGLE = NS["build_single_agent_graph"](lg.StateGraph, lg.START, lg.END)
TABLE, DETAIL = NS["compare_configs"](GRAPH, SINGLE)
DESIGN = "Two-way with reasons (design)"


def right(name):
    return int(TABLE.loc[name, "items right"].split("/")[0])


def status(name, scenario):
    d = DETAIL[(DETAIL.configuration == name) & (DETAIL.scenario == scenario)]
    return d.iloc[0]


def test_the_design_publishes_no_made_up_quote_and_no_injected_call():
    assert TABLE.loc[DESIGN, "made-up quotes published"] == 0
    assert TABLE.loc[DESIGN, "injected tool calls"] == 0
    assert right(DESIGN) == TABLE["items right"].map(lambda s: int(s.split("/")[0])).max()


def test_one_way_publishes_the_made_up_quote_and_misses_the_filing():
    assert TABLE.loc["One-way + ToT", "made-up quotes published"] >= 2
    assert status("One-way + ToT", "4 missing filing")["items right"] != "4/4"
    assert TABLE.loc["One-way + ToT", "model calls (mean)"] < TABLE.loc[DESIGN, "model calls (mean)"]


def test_rejections_without_reasons_end_in_the_baseline():
    assert TABLE.loc["Two-way, no reasons", "baseline reported"] > TABLE.loc[DESIGN, "baseline reported"]
    assert status("Two-way, no reasons", "3 wrong reading")["status"] == "baseline"


def test_tot_beats_the_first_reading():
    assert right("One-way + ToT") > right("Linear pipeline")


def test_the_single_agent_follows_hidden_instructions_and_publishes_made_up_quotes():
    assert status("Single agent", "7 hidden instructions")["injected"] >= 1
    assert status("Single agent", "3 wrong reading")["made-up quote"]


def test_options_are_reset_after_the_comparison():
    assert NS["OPTIONS"] == {"protocol": "two_way", "readings": "tot"}
