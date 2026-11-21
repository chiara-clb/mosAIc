import json

from mosaic import agents as A
from mosaic.config import SETTINGS
from mosaic.llm import DEFAULT_SCRIPT, Model, scripted_chat
from mosaic.tools import ToolBox
from mosaic.world import make_demo_world

STEPS = {"coordinator": A.coordinator, "quant": A.quant, "research": A.research, "synthesis": A.synthesis,
         "critic": A.critic, "finalise": A.finalise, "fallback": A.fallback}


def start(script=None, failures=(), chat=None, **options):
    sent = []
    base_chat = chat or scripted_chat(script)

    def spy(messages, max_tokens):
        sent.append(messages)
        return base_chat(messages, max_tokens)

    world = make_demo_world()
    run = A.Run(ToolBox(world, failures=failures), Model(spy), {"protocol": "two_way", "readings": "tot", **options})
    state = {"request": {"ticker": world.ticker, "company": world.company, "as_of": world.as_of}, "error": []}
    return state, run, sent


def through(state, run, *names):
    for n in names:
        out = dict(STEPS[n](state, run))
        out.pop("_note", None)
        for k, v in out.items():
            state[k] = state.get(k, []) + v if k == "error" else v
    return state


def ready(**kw):
    s, run, sent = start(**kw)
    return through(s, run, "coordinator", "quant", "research"), run, sent


def chosen(s):
    return {p["item"]: p["readings"][i]["effect"] for p, i in zip(s["tot"]["prepared"], s["tot"]["choice"])}


FABRICATED = [{"effect": "spread", "size": "small", "quote": "the Board expects further senior departures",
               "reason": "More exits may follow."},
              {"effect": "fall", "size": "medium", "quote": "her decision to resign as Chief Financial Officer",
               "reason": "Losing the CFO is bad news."}]


def fabricated_script():
    return {"readings": dict(DEFAULT_SCRIPT["readings"], F1=FABRICATED), "lenient_judge": True}


# --- coordinator and quant ---------------------------------------------------------------------

def test_coordinator_uses_only_the_estimated_next_results_date():
    s, run, _ = start()
    through(s, run, "coordinator")
    assert s["plan"]["results_in_window"] is False and s["retries"] == 0


def test_results_a_little_later_than_estimated_still_count_as_due():
    from datetime import timedelta
    s, run, _ = start()
    as_of = s["request"]["as_of"]
    for days_late, due in ((1, True), (13, True), (20, False)):
        run.tools.world.earnings_dates = [as_of - timedelta(days=91 + days_late)]     # estimate falls before as_of
        through(s, run, "coordinator")
        assert s["plan"]["results_in_window"] is due, days_late


def test_quant_baseline_is_symmetric_and_widens_with_results():
    s, run, _ = start()
    through(s, run, "coordinator", "quant")
    b = s["baseline"]
    assert b["fall"] == b["gain"] and abs(sum(b.values()) - 100) < 0.05 and run.model.calls == 0
    s2, run2, _ = start()
    s2["plan"] = {"results_in_window": True}
    through(s2, run2, "quant")
    assert s2["baseline"]["flat"] < b["flat"]


# --- research ----------------------------------------------------------------------------------

def test_research_collects_evidence_and_applies_the_same_story_rule():
    s, run, sent = start()
    through(s, run, "coordinator", "research")
    ids = [e["id"] for e in s["evidence"]]
    assert set(ids) == {"F1", "F2", "F3", "S2"} and len(ids) <= SETTINGS.k_final
    assert [c["id"] for c in next(e for e in s["evidence"] if e["id"] == "F1")["coverage"]] == ["S1"]
    prompts = " ".join(m[-1]["content"] for m in sent if "Agent: research" in m[0]["content"])
    assert "ignore all previous instructions" not in prompts.lower() and "full_text" not in prompts


def test_research_records_its_sources_for_the_leakage_audit():
    from mosaic.trace import leakage_audit
    s, run, _ = start()
    through(s, run, "coordinator", "research")
    used = {e["source"] for e in run.tools.trace.of_kind("source_used")}
    assert {"F1", "F2", "F3", "S2", "S1"} <= used and leakage_audit(run.tools.trace) == []


def test_research_rephrases_an_empty_search():
    s, run, _ = start({"queries": ["zebra quantum volcano"], "rephrase": {"zebra quantum volcano": "chief financial officer resignation"}})
    through(s, run, "coordinator", "research")
    assert "F1" in [e["id"] for e in s["evidence"]]


def test_research_reports_a_tool_failure():
    s, run, _ = start(failures={"search_documents"})
    through(s, run, "coordinator", "research")
    assert s["error"] and "search_documents" in s["error"][0]


def lazy(replies):
    it = iter(replies)
    fallback_chat = scripted_chat()

    def chat(messages, max_tokens):
        if "Agent: research" in messages[0]["content"]:
            return json.dumps(next(it, {"action": "finish"}))
        return fallback_chat(messages, max_tokens)
    return chat


def test_empty_and_repeated_queries_are_not_run():
    s, run, _ = start(chat=lazy([{"action": "search_documents", "query": ""},
                                 {"action": "search_news", "query": "Acme Robotics contract deal"},
                                 {"action": "search_news", "query": "acme robotics contract deal "},
                                 {"action": "search_documents", "query": "chief financial officer resignation"}]))
    through(s, run, "coordinator", "research")
    assert len([c for c in run.tools.calls if c["tool"].startswith("search")]) == 2


def test_code_lists_recent_filings_when_the_model_never_searches_them():
    s, run, _ = start(chat=lazy([{"action": "search_news", "query": "Acme Robotics contract deal"}]))
    through(s, run, "coordinator", "research")
    ids = {e["id"] for e in s["evidence"]}
    assert {"F1", "F2", "F3"} <= ids and not {"F4", "F5"} & ids


# --- synthesis ---------------------------------------------------------------------------------

def test_tot_picks_the_careful_readings():
    s, run, _ = ready()
    through(s, run, "synthesis")
    assert chosen(s) == {"F1": "spread", "F2": "none", "F3": "none", "S2": "gain"}
    assert {line["source"] for line in s["ledger"]} == {"F1", "S2"}
    assert run.tools.branch_log and abs(sum(s["forecast"].values()) - 100) < 0.05


def test_single_pass_takes_the_first_reading():
    s, run, _ = ready(readings="single")
    through(s, run, "synthesis")
    assert chosen(s) == {"F1": "fall", "F2": "fall", "F3": "gain", "S2": "gain"}


def test_synthesis_cannot_act_on_hidden_instructions():
    s, run, _ = ready(script={"gullible": True})
    through(s, run, "synthesis")
    assert s["tot"]["blocked"] and "synthesis may not use search_news" in s["tot"]["blocked"][0]
    assert not [c for c in run.tools.calls if c["tool"] == "search_news" and c["node"] == "synthesis" and c["status"] == "ok"]


def test_ambiguity_only_for_close_opposite_ledgers():
    prepared = [{"readings": [{"effect": "gain", "size": "small"}, {"effect": "fall", "size": "small"}]}]
    nodes = [{"choice": (0,), "depth": 1, "status": "chosen", "score": 0.6},
             {"choice": (1,), "depth": 1, "status": "kept", "score": 0.58}]
    assert A.is_ambiguous(prepared, nodes)
    nodes[1]["score"] = 0.3
    assert not A.is_ambiguous(prepared, nodes)


# --- critic ------------------------------------------------------------------------------------

def test_critic_accepts_a_clean_forecast():
    s, run, _ = ready()
    through(s, run, "synthesis", "critic")
    assert s["verdict"] == {"ok": True, "route": "finalise", "reasons": []}


def test_critic_sends_a_made_up_quote_back_to_synthesis_with_the_reason():
    s, run, _ = ready(script=fabricated_script())
    through(s, run, "synthesis", "critic")
    assert (s["verdict"]["route"], s["retries"]) == ("synthesis", 1)
    assert "F1" in s["feedback"][0] and "does not appear" in s["feedback"][0]


def test_critic_sends_missing_filings_back_to_research_with_a_hint():
    s, run, _ = start({"queries": ["legal proceedings lawsuits"], "news_query": "Acme Robotics CFO finance chief contract"})
    through(s, run, "coordinator", "quant", "research", "synthesis", "critic")
    assert s["verdict"]["route"] == "research" and s["hints"]
    through(s, run, "research")
    assert "F1" in [e["id"] for e in s["evidence"]]


def test_critic_falls_back_when_retries_are_used_up():
    s, run, _ = ready(script=fabricated_script())
    through(s, run, "synthesis")
    s["retries"] = SETTINGS.critic_retries
    through(s, run, "critic", "fallback")
    assert s["verdict"]["route"] == "fallback"
    assert s["final"]["status"] == "baseline" and s["final"]["probabilities"] == s["baseline"]


def test_one_way_and_no_reason_protocols():
    s, run, _ = ready(script=fabricated_script(), protocol="one_way")
    through(s, run, "synthesis", "critic")
    assert s["verdict"]["route"] == "finalise" and s["verdict"]["unchecked"]
    s, run, _ = ready(script=fabricated_script(), protocol="two_way_no_reason")
    through(s, run, "synthesis", "critic")
    assert s["feedback"] == ["The critic rejected the forecast."]


def test_tool_failures_go_to_the_fallback():
    s, run, _ = start(failures={"search_documents"})
    through(s, run, "coordinator", "quant", "research", "synthesis", "critic")
    assert s["verdict"]["route"] == "fallback"


def test_one_passage_per_filing_reaches_the_evidence():
    from types import SimpleNamespace
    full = {"A-0": {"chunk": {"filing": "0001-26-1"}}, "A-1": {"chunk": {"filing": "0001-26-1"}},
            "B-0": {"chunk": {"filing": "0001-26-2"}}, "NEWS-0": {}, "F1": {}}
    tools = SimpleNamespace(passage=lambda i: full.get(i))
    items = [{"id": i} for i in ("A-1", "NEWS-0", "A-0", "B-0", "F1")]          # best first
    assert [e["id"] for e in A.one_per_filing(items, tools)] == ["A-1", "NEWS-0", "B-0", "F1"]


def test_the_kept_passage_is_the_filings_own_subject_not_an_exhibit_list():
    from datetime import datetime, timezone
    from types import SimpleNamespace
    when = datetime(2026, 7, 9, 20, 46, tzinfo=timezone.utc)

    def item(i, text):
        return {"id": i, "text": text, "filed": when, "event": "8-K 8.01", "reaction": "fall", "novelty": 1.0,
                "chunk": {"filing": "0001-26-1"}}
    full = {"A-4": item("A-4", "Item 9.01 Exhibits. 4.1 Form of Note."),
            "A-0": item("A-0", "Item 8.01 Other Events. Amazon closed the sale of $24.9 billion of notes.")}
    index = SimpleNamespace(filing_candidates=lambda t, before, since: [
        {"filing": "0001-26-1", "id": "A-0", "header": "8-K | Item 8.01", "text": full["A-0"]["text"]}])
    tools = SimpleNamespace(world=SimpleNamespace(extra={"index": index}, ticker="AMZN"), passage=lambda i: full.get(i))
    items = [{"id": "A-4", "score": 0.6, "kind": "filing", "header": "8-K | Item 9.01", "extract": "Exhibits"}]
    kept = A.one_per_filing(items, tools, datetime(2026, 7, 30, tzinfo=timezone.utc))
    assert [e["id"] for e in kept] == ["A-0"] and "24.9 billion" in kept[0]["full_text"]


def test_the_quote_check_asks_about_grounding_not_the_forecast():
    assert "grounding only" in A.JUDGE_TASK and "Do not judge" in A.JUDGE_TASK
