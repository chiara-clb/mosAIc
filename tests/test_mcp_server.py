"""The MCP server: six tools, the forecast-date rule, JSON results."""

import asyncio
import json
import logging
from datetime import datetime

import pytest

pytest.importorskip("mcp")
from mcp import Client  # noqa: E402

from mosaic.mcp_server import TOOL_NAMES, build_server, to_jsonable  # noqa: E402
from mosaic.world import make_demo_world  # noqa: E402

@pytest.fixture(autouse=True)
def quiet_server_errors():
    logging.disable(logging.ERROR)
    yield
    logging.disable(logging.NOTSET)


def call_all(world):
    async def go():
        async with Client(build_server(world)) as c:
            names = [t.name for t in (await c.list_tools()).tools]
            calls = {
                "market_data": {"kind": "prices"},
                "search_documents": {"query": "chief financial officer", "list_all": True},
                "search_news": {"query": "Acme Robotics merger CFO contract", "k": 10},
                "case_memory": {"event": "5.02", "reaction": "small"},
                "get_passage": {"item_id": "F1"},
                "branch_log": {"record": {"item": "F1", "reading": "spread"}},
            }
            out = {}
            for name, args in calls.items():
                r = await c.call_tool(name, args)
                assert not r.is_error, (name, r.content)
                out[name] = json.loads(r.content[0].text)
            return names, out
    return asyncio.run(go())


def test_server_lists_six_tools_and_each_answers():
    names, out = call_all(make_demo_world())
    assert sorted(names) == sorted(TOOL_NAMES)
    assert out["market_data"]["dates"] and isinstance(out["market_data"]["dates"][0], str)
    assert out["search_documents"] and out["search_news"]
    assert out["case_memory"] is not None
    assert out["get_passage"]["id"] == "F1"
    assert out["branch_log"] is not None


def test_nothing_late_comes_back_over_mcp():
    world = make_demo_world()
    _, out = call_all(world)
    ids = {r.get("id") for r in out["search_documents"]} | {r.get("id") for r in out["search_news"]}
    assert "F5" not in ids and "S3" not in ids
    assert all(datetime.fromisoformat(d) < world.as_of for d in out["market_data"]["dates"])

    async def late_passage():
        async with Client(build_server(world)) as c:
            return json.loads((await c.call_tool("get_passage", {"item_id": "F5"})).content[0].text)
    assert asyncio.run(late_passage()) is None


def test_json_conversion_keeps_structure():
    d = datetime(2025, 6, 1, 12, 0)
    assert to_jsonable({"a": (d, 1), "b": [{"c": d}]}) == {"a": [d.isoformat(), 1], "b": [{"c": d.isoformat()}]}
