"""MosAIc's tools as a real MCP server (official `mcp` package).

One server serves one world (a company and a forecast date). It exposes the six tools the agents use:
market_data, search_documents, search_news, case_memory, get_passage and branch_log. The forecast-date
rule is applied inside each tool, exactly as in the direct tool box. The role check and the run trace
stay on the client side, at the agents' tool door (mosaic/mcp_tools.py).

Results are JSON text; dates travel as ISO strings.

    python -m mosaic.mcp_server --demo                              # the demo company, over stdio
    python -m mosaic.mcp_server --ticker AAPL --as-of 2026-07-15    # a real stock (live data)
"""

from __future__ import annotations

import argparse
import json
from datetime import date, datetime, timezone

from mcp.server.mcpserver import MCPServer

from mosaic.tools import ToolBox
from mosaic.world import World, make_demo_world

TOOL_NAMES = ("market_data", "search_documents", "search_news", "case_memory", "get_passage", "branch_log")


def to_jsonable(obj):
    """Datetimes to ISO strings, tuples to lists, recursively."""
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    return obj


def _dump(obj) -> str:
    return json.dumps(to_jsonable(obj))


def build_server(world: World, name: str = "mosaic-tools") -> MCPServer:
    """An MCP server for one world. Tools run on a plain tool box that applies the forecast-date rule."""
    box = ToolBox(world)
    server = MCPServer(name, instructions=f"Tools for {world.company} ({world.ticker}); only data published "
                                         f"before {world.as_of:%Y-%m-%d} is returned.")

    @server.tool(description="Prices before the forecast date (kind='prices') or the results calendar (kind='earnings').")
    def market_data(kind: str) -> str:
        return _dump(box.market_data(kind))

    @server.tool(description="Filings from the 30 days before the forecast date that match the query. "
                             "list_all=true returns every one.")
    def search_documents(query: str, k: int = 5, list_all: bool = False) -> str:
        return _dump(box.search_documents(query, k=k, list_all=list_all))

    @server.tool(description="News stories first seen in the 14 days before the forecast date that match the query.")
    def search_news(query: str, k: int = 5, days: int = 14) -> str:
        return _dump(box.search_news(query, k=k, days=days))

    @server.tool(description="How often past events of this type (and price reaction) were followed by each kind of move.")
    def case_memory(event: str, reaction: str | None = None, lag_days: int | None = None) -> str:
        return _dump(box.case_memory(event, reaction, lag_days))

    @server.tool(description="The full text of one passage by id, or null if it was published on or after the forecast date.")
    def get_passage(item_id: str) -> str:
        return _dump(box.get_passage(item_id))

    @server.tool(description="Store one Tree of Thought record in the branch log; returns its position.")
    def branch_log(record: dict) -> str:
        return _dump(box.log_branch(record))

    server.box = box                                   # for tests and inspection
    return server


def main(argv=None) -> None:                           # pragma: no cover - runs a stdio server
    p = argparse.ArgumentParser(prog="mosaic.mcp_server", description="MosAIc tools as an MCP server (stdio).")
    p.add_argument("--demo", action="store_true", help="serve the synthetic demo company")
    p.add_argument("--ticker", help="a real stock, e.g. AAPL")
    p.add_argument("--as-of", help="forecast date, YYYY-MM-DD")
    args = p.parse_args(argv)
    if args.demo:
        world = make_demo_world()
    elif args.ticker and args.as_of:
        from mosaic.live import live_world
        world = live_world(args.ticker.upper(), datetime.strptime(args.as_of, "%Y-%m-%d").replace(tzinfo=timezone.utc))
    else:
        p.error("use --demo, or --ticker and --as-of")
    build_server(world).run("stdio")


if __name__ == "__main__":                             # pragma: no cover
    main()
