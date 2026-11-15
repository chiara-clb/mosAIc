from datetime import timedelta

import pytest

from mosaic.config import SETTINGS
from mosaic.roles import AGENTS, ROLES
from mosaic.tools import RoleError, ToolBox, ToolError, clean_extract
from mosaic.world import DEMO_AS_OF, INJECTION, make_demo_world, make_prices, truth_of


@pytest.fixture
def box():
    return ToolBox(make_demo_world())


# --- roles and world -------------------------------------------------------------------------

def test_five_agents_and_only_research_searches():
    assert AGENTS == ["coordinator", "quant", "research", "synthesis", "critic"]
    searchers = [a for a, r in ROLES.items() if {"search_documents", "search_news"} & set(r["tools"])]
    assert searchers == ["research"]
    assert ROLES["quant"]["kind"] == "code" and ROLES["synthesis"]["kind"] == "llm"


def test_demo_world_has_its_traps():
    w = make_demo_world()
    late = {x["id"] for x in w.filings + w.news if (x.get("filed") or x.get("first_seen")) >= w.as_of}
    assert late == {"F5", "S3"}
    assert INJECTION in w.item("F3")["text"]
    assert truth_of(w, "F1") == ("spread", "small")


def test_prices_end_before_the_forecast_date_and_repeat():
    p = make_prices()
    assert p["dates"][-1] < DEMO_AS_OF and len(p["ACME"]) == 400
    assert make_prices()["ACME"] == p["ACME"]


# --- the forecast-date rule ------------------------------------------------------------------

def test_nothing_published_after_the_forecast_date_comes_back(box):
    assert "F5" not in {h["id"] for h in box.call("search_documents", "research", query="merger agreement acquired cash")}
    assert "S3" not in {h["id"] for h in box.call("search_news", "research", query="Acme merger talks sale")}
    assert box.call("get_passage", "critic", item_id="F5") is None
    assert max(box.call("market_data", "quant", kind="prices")["dates"]) < box.as_of


def test_earnings_calendar_gives_only_an_estimate_of_the_next_date(box):
    e = box.call("market_data", "coordinator", kind="earnings")
    assert e["last_results"] < box.as_of and e["next_estimate"] == e["last_results"] + timedelta(days=91)


def test_search_window_cut_off_and_list_mode(box):
    assert "F4" not in {h["id"] for h in box.call("search_documents", "research", query="first quarter results")}
    assert box.call("search_documents", "research", query="zebra quantum volcano") == []
    cfo = box.call("search_documents", "research", query="chief financial officer resign")
    assert cfo[0]["id"] == "F1"
    every = box.call("search_documents", "research", query="", list_all=True)
    assert {h["id"] for h in every} == {"F1", "F2", "F3"}
    assert all(h["score"] == SETTINGS.min_score for h in every)


# --- what research sees ----------------------------------------------------------------------

def test_research_sees_cleaned_extracts_only(box):
    f3 = next(h for h in box.call("search_documents", "research", query="headquarters move Austin Texas") if h["id"] == "F3")
    assert "ignore all previous instructions" not in f3["extract"].lower()
    assert "instruction-like text removed" in f3["header"]
    assert "text" not in f3 and len(f3["extract"].split()) <= SETTINGS.research_extract_words + 1


def test_clean_extract_keeps_ordinary_text():
    assert clean_extract("Revenue rose 10%. Margins were stable.") == ("Revenue rose 10%. Margins were stable.", False)


# --- the door --------------------------------------------------------------------------------

def test_tools_refuse_agents_without_the_role(box):
    with pytest.raises(RoleError):
        box.call("search_news", "synthesis", query="acme buyout rumor")
    assert box.calls[-1]["status"] == "refused: not in role" and box.calls[-1]["node"] == "synthesis"


def test_switched_off_tool_raises_and_is_logged():
    box = ToolBox(make_demo_world(), failures={"search_documents"})
    with pytest.raises(ToolError):
        box.call("search_documents", "research", query="cfo")
    assert box.calls[-1]["status"] == "error"


def test_unknown_tool_is_rejected(box):
    with pytest.raises(ValueError):
        box.call("send_email", "research")


def test_every_call_is_in_the_trace(box):
    box.call("case_memory", "synthesis", event="5.02", reaction="small")
    box.call("branch_log", "synthesis", record={"node": 1})
    assert [c["tool"] for c in box.calls] == ["case_memory", "branch_log"]
    assert box.branch_log == [{"node": 1}]


def test_case_memory_falls_back_to_all_events(box):
    assert box.call("case_memory", "synthesis", event="5.02", reaction="small")["matched"] == "5.02 / small"
    assert box.call("case_memory", "synthesis", event="2.06", reaction="big fall")["matched"] == "all events"


def test_extra_roles_can_be_registered():
    box = ToolBox(make_demo_world(), extra_roles={"analyst": {"tools": ["search_news"]}})
    assert box.call("search_news", "analyst", query="Acme Robotics contract deal")
