"""Checkpoint 5 notebook: MCP-style tools."""

from datetime import timedelta

import pytest

from tests.notebook.core import load_multi_agent_core

NS = load_multi_agent_core()


@pytest.fixture(autouse=True)
def fresh_run():
    NS["new_run"]()


def call(tool, caller, **kw):
    return NS["mcp_call"](tool, caller, **kw)


def test_nothing_published_after_the_forecast_date_comes_back():
    hits = call("search_documents", "research", query="merger agreement acquired cash")
    assert "F5" not in {h["id"] for h in hits}
    news = call("search_news", "research", query="Acme merger talks sale")
    assert "S3" not in {h["id"] for h in news}
    assert call("get_passage", "critic", item_id="F5") is None
    prices = call("market_data", "quant", kind="prices")
    assert max(prices["dates"]) < NS["AS_OF"]


def test_earnings_calendar_does_not_reveal_the_real_next_date():
    e = call("market_data", "coordinator", kind="earnings")
    assert e["last_results"] < NS["AS_OF"]
    assert e["next_estimate"] == e["last_results"] + timedelta(days=91)


def test_search_respects_the_30_day_window_and_cut_off():
    hits = call("search_documents", "research", query="first quarter results press release")
    assert "F4" not in {h["id"] for h in hits}                     # 29 April: outside the window
    assert call("search_documents", "research", query="zebra quantum volcano") == []
    cfo = call("search_documents", "research", query="chief financial officer resign")
    assert cfo[0]["id"] == "F1" and cfo[0]["score"] >= NS["MIN_SCORE"]


def test_research_sees_cleaned_extracts_only():
    hits = call("search_documents", "research", query="headquarters move Austin Texas")
    f3 = next(h for h in hits if h["id"] == "F3")
    assert "ignore all previous instructions" not in f3["extract"].lower()
    assert "instruction-like text removed" in f3["header"]
    assert "text" not in f3 and len(f3["extract"].split()) <= NS["EXTRACT_WORDS"] + 1


def test_tools_refuse_agents_without_the_role():
    with pytest.raises(NS["RoleError"]):
        call("search_news", "synthesis", query="acme buyout rumor")
    with pytest.raises(NS["RoleError"]):
        call("search_documents", "critic", query="cfo")
    assert NS["CTX"]["tool_log"][-1]["status"] == "refused: not in role"


def test_switched_off_tool_raises_and_is_logged():
    NS["new_run"](failures={"search_documents"})
    with pytest.raises(NS["ToolError"]):
        call("search_documents", "research", query="cfo")
    assert NS["CTX"]["tool_log"][-1]["status"] == "error"


def test_case_memory_falls_back_to_all_events():
    assert call("case_memory", "synthesis", event="5.02", reaction="small")["matched"] == "5.02 / small"
    assert call("case_memory", "synthesis", event="2.06", reaction="big fall")["matched"] == "all events"


def test_branch_log_and_tool_log_record_calls():
    i = call("branch_log", "synthesis", record={"node": 1})
    assert i == 0 and NS["CTX"]["branch_log"] == [{"node": 1}]
    assert [e["tool"] for e in NS["CTX"]["tool_log"]] == ["branch_log"]


def test_clean_extract_keeps_ordinary_text():
    text = "Revenue rose 10%. Margins were stable."
    assert NS["clean_extract"](text) == (text, False)
