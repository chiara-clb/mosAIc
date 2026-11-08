"""Checkpoint 5 notebook: each agent on its own."""

import pytest

from tests.notebook.core import load_multi_agent_core

NS = load_multi_agent_core()


def start(script=None, failures=(), **options):
    NS["OPTIONS"].update({"protocol": "two_way", "readings": "tot", **options})
    sent = []

    def spy(messages, max_tokens):
        sent.append(messages)
        return NS["scripted_chat"](messages, max_tokens)

    ctx = NS["new_run"](script=script, failures=failures, model=NS["Model"](spy))
    w = ctx["world"]
    state = {"request": {"ticker": w["ticker"], "company": w["company"], "as_of": w["as_of"]}, "trace": [], "error": []}
    return state, sent


def step(state, name):
    out = NS["as_node"](name, NS["NODES"][name])(state)
    for k, v in out.items():
        state[k] = state.get(k, []) + v if k in ("trace", "error") else v
    return state


def through(state, *names):
    for n in names:
        step(state, n)
    return state


def test_coordinator_uses_only_the_estimated_next_results_date():
    s = through(start()[0], "coordinator")
    assert s["plan"]["results_in_window"] is False and s["retries"] == 0
    assert s["trace"][-1]["node"] == "coordinator" and s["trace"][-1]["tool_calls"] == 1


def test_quant_baseline_is_symmetric_and_widens_with_results():
    s = through(start()[0], "coordinator", "quant")
    b = s["baseline"]
    assert b["fall"] == b["gain"] and abs(sum(b.values()) - 100) < 0.05 and s["trace"][-1]["model_calls"] == 0
    s2 = start()[0]
    s2["plan"] = {"results_in_window": True}
    step(s2, "quant")
    assert s2["baseline"]["flat"] < b["flat"]


def test_research_collects_evidence_and_applies_the_same_story_rule():
    s, sent = start()
    through(s, "coordinator", "research")
    ids = [e["id"] for e in s["evidence"]]
    assert set(ids) == {"F1", "F2", "F3", "S2"} and len(ids) <= NS["K_FINAL"]
    f1 = next(e for e in s["evidence"] if e["id"] == "F1")
    assert [c["id"] for c in f1["coverage"]] == ["S1"]          # S1 is coverage of F1, not its own item
    assert not {"F4", "F5", "S3"} & set(ids)
    research_prompts = " ".join(m[-1]["content"] for m in sent if "Agent: research" in m[0]["content"])
    assert "ignore all previous instructions" not in research_prompts.lower()   # extracts were cleaned
    assert "full_text" not in research_prompts                                # the model never saw full passages


def test_research_rephrases_an_empty_search():
    script = {"queries": ["zebra quantum volcano"], "rephrase": {"zebra quantum volcano": "chief financial officer resignation"}}
    s = through(start(script)[0], "coordinator", "research")
    assert "-> 0" in s["trace"][-1]["note"] and "F1" in [e["id"] for e in s["evidence"]]


def test_research_reports_a_tool_failure():
    s = through(start(failures={"search_documents"})[0], "coordinator", "research")
    assert s["error"] and "search_documents" in s["error"][0]


def ready(**kw):
    s, sent = start(**kw)
    return through(s, "coordinator", "quant", "research"), sent


def chosen(s):
    prepared, choice = s["tot"]["prepared"], s["tot"]["choice"]
    return {p["item"]: p["readings"][i]["effect"] for p, i in zip(prepared, choice)}


def test_tot_picks_the_careful_readings():
    s = through(ready()[0], "synthesis")
    assert chosen(s) == {"F1": "spread", "F2": "none", "F3": "none", "S2": "gain"}
    assert {l["source"] for l in s["ledger"]} == {"F1", "S2"}
    assert abs(sum(s["forecast"].values()) - 100) < 0.05


def test_single_pass_takes_the_first_reading():
    s = through(ready(readings="single")[0], "synthesis")
    assert chosen(s) == {"F1": "fall", "F2": "fall", "F3": "gain", "S2": "gain"}


def test_synthesis_cannot_act_on_hidden_instructions():
    s = through(ready(script={"gullible": True})[0], "synthesis")
    assert s["tot"]["blocked"] and "synthesis may not use search_news" in s["tot"]["blocked"][0]
    assert not [e for e in NS["CTX"]["tool_log"] if e["caller"] == "synthesis" and e["tool"] == "search_news" and e["status"] == "ok"]


def test_critic_accepts_a_clean_forecast():
    s = through(ready()[0], "synthesis", "critic")
    assert s["verdict"] == {"ok": True, "route": "finalise", "reasons": []}
    assert NS["route_after_critic"](s) == "finalise"


FABRICATED = [{"effect": "spread", "size": "small", "quote": "the Board expects further senior departures",
               "reason": "More exits may follow."},
              {"effect": "fall", "size": "medium", "quote": "her decision to resign as Chief Financial Officer",
               "reason": "Losing the CFO is bad news."}]


def fabricated_script():
    readings = dict(NS["DEFAULT_SCRIPT"]["readings"], F1=FABRICATED)
    return {"readings": readings, "lenient_judge": True}


def test_critic_sends_a_made_up_quote_back_to_synthesis_with_the_reason():
    s = through(ready(script=fabricated_script())[0], "synthesis", "critic")
    v = s["verdict"]
    assert (v["ok"], v["route"]) == (False, "synthesis") and s["retries"] == 1
    assert "F1" in s["feedback"][0] and "does not appear" in s["feedback"][0]


def test_critic_sends_missing_filings_back_to_research_with_a_hint():
    s, _ = start(script={"queries": ["legal proceedings lawsuits"], "news_query": "Acme Robotics CFO finance chief contract"})
    through(s, "coordinator", "quant", "research", "synthesis", "critic")
    assert s["verdict"]["route"] == "research" and s["hints"]
    step(s, "research_more")
    assert "F1" in [e["id"] for e in s["evidence"]]


def test_critic_falls_back_when_retries_are_used_up():
    s = through(ready(script=fabricated_script())[0], "synthesis")
    s["retries"] = NS["MAX_RETRIES"]
    step(s, "critic")
    assert s["verdict"]["route"] == "fallback"
    step(s, "fallback")
    assert s["final"]["status"] == "baseline" and s["final"]["probabilities"] == s["baseline"]


def test_one_way_publishes_without_checking_and_no_reason_hides_the_problem():
    s = through(ready(script=fabricated_script(), protocol="one_way")[0], "synthesis", "critic")
    assert s["verdict"]["route"] == "finalise" and s["verdict"]["unchecked"]
    s = through(ready(script=fabricated_script(), protocol="two_way_no_reason")[0], "synthesis", "critic")
    assert s["feedback"] == ["The critic rejected the forecast."]


def test_critic_routes_tool_failures_to_the_fallback():
    s = through(start(failures={"search_documents"})[0], "coordinator", "quant", "research", "synthesis", "critic")
    assert s["verdict"]["route"] == "fallback" and "skipped" in s["trace"][-2]["note"]


def lazy_chat(replies):
    """A stand-in for a weak model: returns the given research replies in order, then finishes."""
    it = iter(replies)

    def chat(messages, max_tokens):
        if "Agent: research" in messages[0]["content"]:
            return __import__("json").dumps(next(it, {"action": "finish"}))
        return NS["scripted_chat"](messages, max_tokens)
    return chat


def test_empty_and_repeated_queries_are_not_run():
    s, _ = start()
    NS["CTX"]["model"] = NS["Model"](lazy_chat([
        {"action": "search_documents", "query": ""},
        {"action": "search_news", "query": "Acme Robotics contract deal"},
        {"action": "search_news", "query": "acme robotics contract deal "},
        {"action": "search_documents", "query": "chief financial officer resignation"}]))
    through(s, "coordinator", "research")
    note = s["trace"][-1]["note"]
    assert "refused: empty" in note and "refused: repeated" in note
    runs = [e for e in NS["CTX"]["tool_log"] if e["tool"].startswith("search")]
    assert len(runs) == 2                                    # only the two real queries reached a tool
    assert "no filing search" not in note


def test_code_lists_recent_filings_when_the_model_never_searches_them():
    s, _ = start()
    NS["CTX"]["model"] = NS["Model"](lazy_chat([{"action": "search_news", "query": "Acme Robotics contract deal"}]))
    through(s, "coordinator", "research")
    ids = {e["id"] for e in s["evidence"]}
    assert {"F1", "F2", "F3"} <= ids and not {"F4", "F5"} & ids   # the window and the date rule still apply
    assert "code listed 3 recent filings" in s["trace"][-1]["note"]
