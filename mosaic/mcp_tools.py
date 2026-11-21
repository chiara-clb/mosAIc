"""The agents' tool door, with every tool call going through an MCP client to the MCP server.

`McpToolBox` keeps the door of the direct tool box unchanged: the role check, the switch that turns a
tool off and the run trace all happen here, before anything is sent. Only the execution moves: each
allowed call becomes an MCP `tools/call` to the server built by `mosaic.mcp_server.build_server`.

The agents are synchronous and the MCP client is asynchronous, so the client lives in its own event
loop on a background thread for the length of the run.
"""

from __future__ import annotations

import asyncio
import json
import threading
from datetime import datetime

from mcp import Client

from mosaic.mcp_server import build_server, to_jsonable
from mosaic.tools import ToolBox, ToolError
from mosaic.trace import Trace
from mosaic.world import World


def _dt(value):
    return datetime.fromisoformat(value) if isinstance(value, str) else value


def from_wire(tool: str, args: dict, result):
    """Turn ISO strings back into datetimes where the agents need them."""
    if tool == "market_data" and isinstance(result, dict):
        if args.get("kind") == "prices":
            result["dates"] = [_dt(d) for d in result["dates"]]
        elif args.get("kind") == "earnings":
            for k in ("last_results", "next_estimate"):
                result[k] = _dt(result.get(k))
            result["past_results"] = [_dt(d) for d in result.get("past_results", [])]
    if tool == "get_passage" and isinstance(result, dict):
        result["published"] = _dt(result["published"])
    return result


class McpBridge:
    """A synchronous handle on an MCP client session, running in a background event loop.

    One task in that loop opens the session and keeps it open until `close()`, because the client's
    context must be entered and left by the same task.
    """

    def __init__(self, server, timeout: float = 120.0):
        self.timeout = timeout
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever, daemon=True, name="mcp-client")
        self.thread.start()
        self.client = None
        self._ready = threading.Event()
        self._error = None
        self._stop = None
        self._session = asyncio.run_coroutine_threadsafe(self._hold(server), self.loop)
        if not self._ready.wait(timeout) or self._error:
            self.close()
            raise ToolError(f"could not open the MCP session: {self._error}")

    async def _hold(self, server):
        self._stop = asyncio.Event()
        try:
            async with Client(server) as client:
                self.client = client
                self._ready.set()
                await self._stop.wait()
        except BaseException as exc:          # report a failed start to the waiting thread
            self._error = exc
            self._ready.set()
            raise

    def _run(self, coro):
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result(self.timeout)

    def list_tools(self) -> list[str]:
        return [t.name for t in self._run(self.client.list_tools()).tools]

    def call(self, name: str, arguments: dict):
        result = self._run(self.client.call_tool(name, to_jsonable(arguments)))
        text = result.content[0].text if result.content else "null"
        if result.is_error:
            raise ToolError(f"{name} failed on the MCP server: {text}")
        return json.loads(text)

    def close(self) -> None:
        if not self.loop.is_running():
            return
        if self._stop is not None:
            self.loop.call_soon_threadsafe(self._stop.set)
        try:
            self._session.result(self.timeout)
        except BaseException:
            pass
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(timeout=5)


class McpToolBox(ToolBox):
    """Same door as ToolBox; tool execution goes through MCP."""

    def __init__(self, world: World, trace: Trace | None = None, failures=(), extra_roles: dict | None = None,
                 server=None):
        super().__init__(world, trace=trace, failures=failures, extra_roles=extra_roles)
        self.server = server or build_server(world)
        self.bridge = McpBridge(self.server)
        names = ("market_data", "search_documents", "search_news", "case_memory", "get_passage")
        self.tools = {n: self._remote(n) for n in names}
        self.tools["branch_log"] = self._remote_branch_log

    def _remote(self, name: str):
        def call(**kwargs):
            return from_wire(name, kwargs, self.bridge.call(name, kwargs))
        call.__name__ = name
        return call

    def _remote_branch_log(self, record: dict) -> int:
        position = self.bridge.call("branch_log", {"record": record})
        self.branch_log.append(record)                 # local copy, as the direct tool box keeps
        return position

    def close(self) -> None:
        self.bridge.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def run_forecast_mcp(world: World, model, failures=(), options: dict | None = None, engine=None):
    """`run_forecast`, with every tool call made over MCP. Returns (final state, run)."""
    from mosaic import agents as A
    from mosaic.graph import build_graph, graph_engine

    trace = Trace(run_id=f"{world.ticker}-{world.as_of:%Y-%m-%d}", as_of=world.as_of)
    box = McpToolBox(world, trace=trace, failures=failures)
    try:
        run = A.Run(box, model, {"protocol": "two_way", "readings": "tot", **(options or {})})
        trace.add("run", "start", ticker=world.ticker, as_of=world.as_of.isoformat(), model=model.name,
                  engine=(engine or graph_engine())[3], options=run.options, tools="mcp")
        state = build_graph(run, engine).invoke(
            {"request": {"ticker": world.ticker, "company": world.company, "as_of": world.as_of},
             "trace": [], "error": []}, {"recursion_limit": 50})
        trace.add("run", "end", status=state["final"]["status"], review=state["final"].get("review"))
    finally:
        box.close()
    return state, run
