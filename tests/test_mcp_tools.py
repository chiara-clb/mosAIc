"""The agents' tool door with every call made over MCP."""

import logging

import pytest

pytest.importorskip("mcp")

from mosaic import minigraph, scenarios  # noqa: E402
from mosaic.graph import run_forecast  # noqa: E402
from mosaic.llm import make_model  # noqa: E402
from mosaic.mcp_tools import McpToolBox, run_forecast_mcp  # noqa: E402
from mosaic.tools import RoleError, ToolBox, ToolError  # noqa: E402
from mosaic.world import make_demo_world  # noqa: E402

ENGINE = (minigraph.StateGraph, minigraph.START, minigraph.END, "built-in runner")


@pytest.fixture(autouse=True)
def quiet_server_errors():
    logging.disable(logging.ERROR)
    yield
    logging.disable(logging.NOTSET)


def test_mcp_tool_box_matches_direct_tool_box():
    world = make_demo_world()
    direct = ToolBox(world)
    with McpToolBox(world) as remote:
        for kind in ("prices", "earnings"):
            assert remote.call("market_data", "quant", kind=kind) == direct.call("market_data", "quant", kind=kind)
        q = {"query": "chief financial officer", "list_all": True}
        assert remote.call("search_documents", "research", **q) == direct.call("search_documents", "research", **q)
        assert remote.call("get_passage", "critic", item_id="F1") == direct.call("get_passage", "critic", item_id="F1")
        assert remote.calls[-1]["status"] == "ok"


def test_role_refusal_happens_at_the_door():
    with McpToolBox(make_demo_world()) as box:
        with pytest.raises(RoleError):
            box.call("search_news", "synthesis", query="anything")
        assert box.calls[-1]["status"] == "refused: not in role"


def test_server_error_becomes_tool_error():
    with McpToolBox(make_demo_world()) as box:
        with pytest.raises(ToolError):
            box.call("get_passage", "critic", item_id=["not", "a", "string"])


def test_demo_forecast_over_mcp_matches_direct():
    s1, r1 = run_forecast(make_demo_world(), make_model("scripted"), engine=ENGINE)
    s2, r2 = run_forecast_mcp(make_demo_world(), make_model("scripted"), engine=ENGINE)
    for key in ("status", "probabilities", "ledger", "review", "baseline"):
        assert s1["final"][key] == s2["final"][key], key
    assert [(c["node"], c["tool"], c["status"]) for c in r1.tools.calls] == \
           [(c["node"], c["tool"], c["status"]) for c in r2.tools.calls]


@pytest.mark.parametrize("name", list(scenarios.SCENARIOS))
def test_scenarios_over_mcp_match_direct(name):
    sc = scenarios.SCENARIOS[name]
    world = (lambda: sc["world"]() if sc.get("world") else make_demo_world())
    direct, _ = run_forecast(world(), make_model("scripted", sc["script"]), failures=sc.get("failures", ()), engine=ENGINE)
    remote, _ = run_forecast_mcp(world(), make_model("scripted", sc["script"]), failures=sc.get("failures", ()),
                                 engine=ENGINE)
    for key in ("status", "probabilities", "ledger", "review"):
        assert direct["final"][key] == remote["final"][key], key
