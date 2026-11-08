"""Checkpoint 5 notebook: settings, roles and the synthetic world."""

from tests.notebook.core import load_multi_agent_core

NS = load_multi_agent_core()


def test_settings_match_the_design():
    assert NS["MODE"] == "scripted"
    assert (NS["BEAM_WIDTH"], NS["PRUNE_MARGIN"], NS["MAX_TOTAL_SHIFT"]) == (3, 0.3, 15.0)
    assert NS["STRENGTH_POINTS"] == {"small": 3, "medium": 6, "large": 9}
    assert (NS["MAX_RETRIES"], NS["MAX_RESEARCH_STEPS"], NS["K_FINAL"]) == (2, 4, 5)
    assert NS["MIN_SCORE"] == 0.20 and NS["FRESH_DAYS"] == 30


def test_five_agents_with_separate_tools():
    roles = NS["ROLES"]
    assert list(roles) == ["coordinator", "quant", "research", "synthesis", "critic"]
    for r in roles.values():
        assert {"role", "goal", "backstory", "tools", "kind", "sees"} <= set(r)
    searchers = [a for a, r in roles.items() if {"search_documents", "search_news"} & set(r["tools"])]
    assert searchers == ["research"]                     # only research can search
    assert roles["quant"]["kind"] == "code"              # numbers come from code
    assert roles["synthesis"]["kind"] == "llm" and not {"search_documents", "search_news"} & set(roles["synthesis"]["tools"])
    assert len(NS["role_table"]()) == 5


def test_world_has_the_three_traps():
    w = NS["make_world"]()
    as_of = w["as_of"]
    late = [f for f in w["filings"] + w["news"] if f.get("filed", f.get("first_seen")) >= as_of]
    assert {x["id"] for x in late} == {"F5", "S3"}
    assert NS["INJECTION"] in next(f["text"] for f in w["filings"] if f["id"] == "F3")
    old = [f for f in w["filings"] if f["filed"] < as_of - __import__("datetime").timedelta(days=NS["FRESH_DAYS"])]
    assert [f["id"] for f in old] == ["F4"]


def test_prices_end_before_the_forecast_date():
    p = NS["make_prices"]()
    assert p["dates"][-1] < NS["AS_OF"] and len(p["dates"]) == len(p["ACME"]) == len(p["SPY"]) == 400
    assert all(d.weekday() < 5 for d in p["dates"])
    assert NS["make_prices"]()["ACME"] == p["ACME"]      # same seed, same prices


def test_truth_lookup_and_case_memory_fallback():
    w = NS["WORLD"]
    assert NS["truth_of"](w, "F1") == ("spread", "small")
    assert NS["truth_of"](w, "F2") == ("none", None)
    assert "*" in w["case_memory"]
