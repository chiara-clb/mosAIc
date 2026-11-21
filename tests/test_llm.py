import json
from types import SimpleNamespace

import pytest

from mosaic.llm import (DEFAULT_SCRIPT, BudgetError, Model, agent_messages, anthropic_chat, make_model,
                        parse_json, scripted_chat)
from mosaic.world import INJECTION


def test_model_counts_calls_and_enforces_the_budget():
    m = Model(lambda msgs, n: "{}", budget=2)
    m([]), m([])
    assert m.calls == 2
    with pytest.raises(BudgetError):
        m([])


def test_default_budget_comes_from_settings():
    from mosaic.config import SETTINGS
    assert Model(lambda m, n: "").budget == SETTINGS.run_model_budget == 40


def test_parse_json_tolerates_fences_and_text():
    assert parse_json('Sure!\n```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json("no json here") is None and parse_json("{broken") is None and parse_json(None) is None


def test_messages_carry_the_role_card():
    msgs = agent_messages("research", "Pick the next search.", {"x": 1})
    assert msgs[0]["content"].startswith("Agent: research\nYou are the Research analyst")
    assert json.loads(msgs[1]["content"].split("INPUT:\n", 1)[1]) == {"x": 1}


def ask(agent, payload, **script):
    return json.loads(scripted_chat(script)(agent_messages(agent, "", payload), 100))


def test_scripted_research_follows_queries_then_news_then_finishes():
    obs = []
    for q in DEFAULT_SCRIPT["queries"]:
        a = ask("research", {"observations": obs})
        assert (a["action"], a["query"]) == ("search_documents", q)
        obs.append({"tool": "search_documents", "query": q, "results": [{"id": "x"}]})
    assert ask("research", {"observations": obs})["action"] == "search_news"
    obs.append({"tool": "search_news", "query": DEFAULT_SCRIPT["news_query"], "results": [{"id": "x"}]})
    assert ask("research", {"observations": obs})["action"] == "finish"


def test_scripted_research_rephrases_and_follows_hints():
    obs = [{"tool": "search_documents", "query": "q1", "results": []}]
    assert ask("research", {"observations": obs}, queries=["q1"], rephrase={"q1": "q2"})["query"] == "q2"
    a = ask("research", {"observations": [], "hints": ["find the filing"]})
    assert (a["action"], a["query"]) == ("search_documents", "find the filing")


def test_scripted_synthesis_uses_feedback_and_can_be_gullible():
    item = {"id": "F1", "text": "plain text"}
    assert ask("synthesis", {"item": item, "feedback": []})["readings"][0]["effect"] == "fall"
    fixed = [{"effect": "none", "size": None, "quote": "", "reason": "fixed"}]
    after = ask("synthesis", {"item": item, "feedback": ["F1: quote not found"]}, readings_after_feedback={"F1": fixed})
    assert after["readings"] == fixed
    bad = {"id": "F3", "text": INJECTION}
    assert "tool_request" in ask("synthesis", {"item": bad, "feedback": []}, gullible=True)
    assert "tool_request" not in ask("synthesis", {"item": bad, "feedback": []})


def test_scripted_judge_is_strict_unless_lenient():
    p = {"passage": "The CFO resigned.", "quote": "was fired"}
    assert ask("critic", p) == {"supported": False}
    assert ask("critic", p, lenient_judge=True) == {"supported": True}


def test_anthropic_backend_sends_system_separately():
    sent = {}

    class FakeMessages:
        def create(self, **kw):
            sent.update(kw)
            return SimpleNamespace(content=[SimpleNamespace(text=' {"ok": true} ')])

    chat = anthropic_chat("claude-haiku-5-5", client=SimpleNamespace(messages=FakeMessages()))
    out = chat(agent_messages("critic", "Check.", {"q": 1}), 50)
    assert out == '{"ok": true}'
    assert sent["model"] == "claude-haiku-5-5" and sent["max_tokens"] == 50
    assert "temperature" not in sent and sent["extra_body"] == {"thinking": {"type": "disabled"}}
    assert sent["system"].startswith("Agent: critic") and [m["role"] for m in sent["messages"]] == ["user"]


def test_make_model_scripted_and_unknown():
    m = make_model("scripted", script={"lenient_judge": True})
    assert m.name == "scripted" and json.loads(m(agent_messages("critic", "", {"passage": "a", "quote": "b"})))["supported"]
    with pytest.raises(ValueError):
        make_model("gpt")


def test_anthropic_chat_works_with_a_package_that_has_no_temperature():
    seen = []

    class NoTemperature:
        def create(self, *, model, system, messages, max_tokens):
            seen.append(model)
            return SimpleNamespace(content=[SimpleNamespace(text="ok")])

    chat = anthropic_chat("claude-haiku-5-5", client=SimpleNamespace(messages=NoTemperature()))
    assert chat([{"role": "user", "content": "hi"}]) == "ok"
    assert chat([{"role": "user", "content": "hi"}]) == "ok"
    assert len(seen) == 2


def test_older_claude_models_keep_a_temperature():
    sent = {}

    class FakeMessages:
        def create(self, **kw):
            sent.update(kw)
            return SimpleNamespace(content=[SimpleNamespace(text="{}")])

    anthropic_chat("claude-3-5-haiku-latest", client=SimpleNamespace(messages=FakeMessages()))([{"role": "user", "content": "x"}])
    assert sent["temperature"] == 0.0 and "extra_body" not in sent


def test_a_reply_cut_off_by_the_limit_is_asked_again_with_more_room():
    limits = []

    class Thinker:
        def create(self, **kw):
            limits.append(kw["max_tokens"])
            if len(limits) == 1:          # all the room went on hidden thinking
                return SimpleNamespace(content=[SimpleNamespace(type="thinking")], stop_reason="max_tokens")
            return SimpleNamespace(content=[SimpleNamespace(text='{"readings": []}')], stop_reason="end_turn")

    out = anthropic_chat("claude-haiku-5-5", client=SimpleNamespace(messages=Thinker()))([{"role": "user", "content": "x"}], 400)
    assert out == '{"readings": []}' and limits == [400, 1600]
