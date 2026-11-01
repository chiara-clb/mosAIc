import json

import pytest

from tests.notebook.core import load_core
from tests.notebook.helpers import chunk, dt

ns = load_core()
AS_OF = dt("2026-08-01")


def scripted(*replies):
    """A stand-in chat function that returns the given replies in order and records the prompts."""
    log = []

    def chat(messages, max_tokens):
        log.append(messages)
        return replies[len(log) - 1]
    return ns["Model"](chat), log


def item():
    return dict(chunk("AAPL-10-Q-1", accepted="2026-07-31", text="Apple is subject to legal proceedings."), row=0)


def test_model_counts_calls():
    m, _ = scripted("a", "b")
    m([]), m([])
    assert m.calls == 2


@pytest.mark.parametrize("reply,expected", [
    ('[{"a": 1}]', [{"a": 1}]),
    ('```json\n[{"a": 1}]\n```', [{"a": 1}]),
    ('Here you go: [{"a": 1}] Hope this helps', [{"a": 1}]),
    ("no json here", None),
    ("[{broken", None),
])
def test_parse_json(reply, expected):
    assert ns["parse_json"](reply) == expected


def test_two_readings_plus_no_effect_most_plausible_first():
    reply = json.dumps([{"effect": "spread", "size": "small", "reason": "subject to legal proceedings"},
                        {"effect": "fall", "size": "medium", "reason": "lawsuit risk"}])
    m, log = scripted(reply)
    readings, _ = ns["propose_readings"](m, item(), AS_OF)
    assert [r["effect"] for r in readings] == ["spread", "fall", "none"]
    assert all(r["source"] == "AAPL-10-Q-1" for r in readings)
    assert m.calls == 1
    assert "1 days before the forecast date" in log[0][1]["content"]


def test_model_may_include_no_effect_itself():
    reply = json.dumps([{"effect": "none", "size": "small", "reason": "standard wording"},
                        {"effect": "fall", "size": "small", "reason": "lawsuit"}])
    readings, _ = ns["propose_readings"](scripted(reply)[0], item(), AS_OF)
    assert [r["effect"] for r in readings] == ["none", "fall"]


def test_invalid_and_duplicate_readings_are_dropped():
    reply = json.dumps([{"effect": "fall", "size": "huge", "reason": "x"},
                        {"effect": "gain", "size": "large", "reason": "y"},
                        {"effect": "gain", "size": "small", "reason": "z"},
                        {"effect": "sideways", "size": "small", "reason": "w"}])
    readings, _ = ns["propose_readings"](scripted(reply)[0], item(), AS_OF)
    assert [(r["effect"], r["size"]) for r in readings] == [("gain", "large"), ("none", "small")]


def test_unreadable_reply_leaves_only_no_effect():
    readings, raw = ns["propose_readings"](scripted("I think the stock will go up.")[0], item(), AS_OF)
    assert [r["effect"] for r in readings] == ["none"] and raw.startswith("I think")


@pytest.mark.parametrize("reply,ok", [("yes", True), ("Yes.", True), ("YES, it says so", True),
                                      ("no", False), ("No, it doesn't", False), ("", False), ("maybe", False)])
def test_grounding_reads_the_first_word(reply, ok):
    m, _ = scripted(reply)
    assert ns["grounded"](m, item(), {"effect": "fall", "reason": "claim"})[0] is ok


def test_no_effect_needs_no_grounding_call():
    m, _ = scripted()
    assert ns["grounded"](m, item(), {"effect": "none", "reason": "x"}) == (True, None)
    assert m.calls == 0


def test_date_rule_window():
    it = item()
    assert ns["date_ok"](it, AS_OF)
    assert not ns["date_ok"](it, dt("2026-07-31"))               # same day: not before the forecast date
    assert not ns["date_ok"](it, dt("2026-09-15"))               # older than 30 days
