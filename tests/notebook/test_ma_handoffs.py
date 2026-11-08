"""Checkpoint 5 notebook: handoff checks, model budget and the scripted model."""

import json

import pytest

from tests.notebook.core import load_multi_agent_core

NS = load_multi_agent_core()


def test_good_handoffs_pass():
    NS["handoff"]("baseline", {"fall": 30.85, "flat": 38.3, "gain": 30.85})
    NS["handoff"]("reading", {"effect": "none", "size": None, "quote": "", "reason": ""})
    NS["handoff"]("line", {"source": "F1", "effect": "spread", "size": "small", "quote": "x", "reason": "", "shift": {}})
    NS["handoff"]("verdict", {"ok": True, "route": "finalise", "reasons": []})


@pytest.mark.parametrize("kind,obj", [
    ("baseline", {"fall": 50, "flat": 50, "gain": 10}),
    ("baseline", {"fall": 0.5, "flat": 49.5, "gain": 50}),
    ("reading", {"effect": "crash", "size": "small", "quote": "x"}),
    ("reading", {"effect": "fall", "size": "huge", "quote": "x"}),
    ("reading", {"effect": "fall", "size": "small", "quote": " "}),
    ("evidence", {"id": "F1", "kind": "tweet", "header": "h", "published": "d", "extract": "e", "full_text": "t"}),
    ("verdict", {"ok": "yes", "route": "somewhere"}),
])
def test_bad_handoffs_stop_at_the_boundary(kind, obj):
    with pytest.raises(ValueError):
        NS["handoff"](kind, obj)


def test_model_counts_calls_and_enforces_the_budget():
    m = NS["Model"](lambda msgs, n: "{}", budget=2)
    m([]), m([])
    assert m.calls == 2
    with pytest.raises(NS["BudgetError"]):
        m([])


def test_parse_json_tolerates_fences_and_text():
    assert NS["parse_json"]('Sure!\n```json\n{"a": 1}\n```') == {"a": 1}
    assert NS["parse_json"]("no json here") is None


def test_messages_carry_the_role_card():
    msgs = NS["agent_messages"]("research", "Pick the next search.", {"x": 1})
    assert msgs[0]["content"].startswith("Agent: research\nYou are the Research analyst")
    assert json.loads(msgs[1]["content"].split("INPUT:\n", 1)[1]) == {"x": 1}


def scripted(agent, payload, **script):
    NS["new_run"](script=script)
    return json.loads(NS["scripted_chat"](NS["agent_messages"](agent, "", payload)))


def test_scripted_research_follows_queries_then_news_then_finishes():
    s = NS["DEFAULT_SCRIPT"]
    obs = []
    for q in s["queries"]:
        a = scripted("research", {"observations": obs})
        assert (a["action"], a["query"]) == ("search_documents", q)
        obs.append({"tool": "search_documents", "query": q, "results": [{"id": "x"}]})
    assert scripted("research", {"observations": obs})["action"] == "search_news"
    obs.append({"tool": "search_news", "query": s["news_query"], "results": [{"id": "x"}]})
    assert scripted("research", {"observations": obs})["action"] == "finish"


def test_scripted_research_rephrases_and_follows_hints():
    obs = [{"tool": "search_documents", "query": "q1", "results": []}]
    a = scripted("research", {"observations": obs}, queries=["q1"], rephrase={"q1": "q2"})
    assert a["query"] == "q2"
    a = scripted("research", {"observations": [], "hints": ["find the filing"]})
    assert (a["action"], a["query"]) == ("search_documents", "find the filing")


def test_scripted_synthesis_uses_feedback_and_can_be_gullible():
    item = {"id": "F1", "text": "plain text"}
    first = scripted("synthesis", {"item": item, "feedback": []})["readings"][0]
    assert first["effect"] == "fall"
    fixed = [{"effect": "none", "size": None, "quote": "", "reason": "fixed"}]
    after = scripted("synthesis", {"item": item, "feedback": ["F1: quote not found"]}, readings_after_feedback={"F1": fixed})
    assert after["readings"] == fixed
    bad = {"id": "F3", "text": NS["INJECTION"]}
    assert "tool_request" in scripted("synthesis", {"item": bad, "feedback": []}, gullible=True)
    assert "tool_request" not in scripted("synthesis", {"item": bad, "feedback": []})


def test_scripted_judge_strict_unless_lenient():
    p = {"passage": "The CFO resigned.", "quote": "was fired"}
    assert scripted("critic", p) == {"supported": False}
    assert scripted("critic", p, lenient_judge=True) == {"supported": True}
