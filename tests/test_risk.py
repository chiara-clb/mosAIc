import pytest

from mosaic.ledger import apply_ledger, shift_for
from mosaic.risk import ReviewLog, apply_review, review_level, risk_triggers

BASE = {"fall": 32.0, "flat": 36.0, "gain": 32.0}


def line(source="F1", effect="gain", size="small", n=200, confidence=None):
    return {"source": source, "effect": effect, "size": size, "quote": "q", "reason": "r",
            "shift": shift_for(effect, size), "n": n, "confidence": confidence}


def state(lines, status="accepted", retries=0, ambiguous=False):
    probs, _ = apply_ledger(BASE, lines)
    return {"final": {"status": status, "baseline": BASE, "probabilities": probs if status != "baseline" else BASE,
                      "ledger": lines if status != "baseline" else []},
            "retries": retries, "tot": {"ambiguous": ambiguous},
            "evidence": [{"id": "F1", "event": "5.02"}, {"id": "S2", "event": "news"}]}


def test_no_trigger_publishes():
    assert risk_triggers(state([line("F1", "spread"), line("S2", "gain")])) == []
    assert review_level([]) == "publish"


@pytest.mark.parametrize("st,word", [
    (state([line("F1", "gain", "large"), line("S2", "gain", "large")]), "high impact"),
    (state([line("F1", "spread", n=8), line("S2", "gain")]), "thin evidence"),
    (state([line("F1", "gain", "medium")]), "one source"),
    (state([line("F1", "spread"), line("S2", "gain")], ambiguous=True), "ambiguity"),
    (state([line("F1", "spread", confidence=0.3), line("S2", "gain")]), "low confidence"),
    (state([line("F1", "spread"), line("S2", "gain")], retries=1), "sent work back"),
    (state([], status="baseline"), "baseline used"),
])
def test_each_trigger_alone_flags(st, word):
    t = risk_triggers(st)
    assert len(t) == 1 and word in t[0], t
    assert review_level(t) == "flag"


def test_a_refused_tool_call_is_a_trigger():
    calls = [{"node": "synthesis", "tool": "search_news", "status": "refused: not in role"},
             {"node": "research", "tool": "search_news", "status": "ok"}]
    t = risk_triggers(state([line("F1", "spread"), line("S2", "gain")]), calls)
    assert t == ["refused tool call: synthesis asked for search_news"]


def test_two_triggers_hold():
    t = risk_triggers(state([line("F1", "spread", "large", n=8), line("S2", "gain", "large")]))
    assert len(t) == 2 and review_level(t) == "hold"


def test_analyst_review_keeps_resizes_and_strikes():
    log = ReviewLog()
    st = state([line("F1", "spread", "large"), line("S2", "gain", "large")])
    out = apply_review(st, {"F1": "strike", "S2": ("resize", "small")}, log)
    assert [x["source"] for x in out["ledger"]] == ["S2"] and out["ledger"][0]["size"] == "small"
    assert out["probabilities"] == apply_ledger(BASE, [line("S2", "gain", "small")])[0]
    assert out["status"] == "reviewed by analyst"
    assert log.summary() == {"5.02": {"keep": 0, "resize": 0, "strike": 1}, "news": {"keep": 0, "resize": 1, "strike": 0}}


def test_bad_decisions_are_rejected():
    st = state([line("F1", "spread", "large")])
    with pytest.raises(ValueError):
        apply_review(st, {"F1": ("resize", "huge")})
    with pytest.raises(ValueError):
        apply_review(st, {"F1": "delete"})
