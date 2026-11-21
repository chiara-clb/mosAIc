from mosaic import minigraph
from mosaic.risk import apply_review
from mosaic.scenarios import run_scenario
from mosaic.views import evidence_rows, forecast_rows, ledger_rows, run_facts, step_rows, tool_rows, tot_rows

ENGINE = (minigraph.StateGraph, minigraph.START, minigraph.END, "built-in runner")


def test_tables_from_a_normal_run():
    state, run, _ = run_scenario("1 normal", engine=ENGINE)
    f = forecast_rows(state["final"])
    assert [r["Outcome"] for r in f] == ["Large fall", "Flat", "Large gain"]
    assert abs(sum(r["Change (points)"] for r in f)) < 0.2
    assert [(r["Source"], r["Published"]) for r in ledger_rows(state)] == [("F1", "2025-05-28"), ("S2", "2025-05-22")]
    assert {r["Id"] for r in evidence_rows(state)} == {"F1", "F2", "F3", "S2"}
    tot = tot_rows(state)
    assert len([r for r in tot if r["Chosen"]]) == 4 and {r["Item"] for r in tot} == {"F1", "F2", "F3", "S2"}
    assert step_rows(state)[0]["Step"] == "coordinator" and tool_rows(run)
    facts = run_facts(state, run)
    assert facts["review"] == "publish" and facts["leakage"] == "clean"


def test_reviewed_ledger_and_a_baseline_run():
    state, _, _ = run_scenario("9 rare event, big move", engine=ENGINE)
    reviewed = apply_review(state, {"F1": "strike"})
    rows = ledger_rows(state, reviewed)
    assert [(r["Source"], r["Analyst decision"]) for r in rows] == [("F1", "struck"), ("S2", "kept")]
    assert [x["source"] for x in reviewed["ledger"]] == ["S2"]           # only kept lines count in the forecast
    base_state, base_run, _ = run_scenario("6 tool down", engine=ENGINE)
    assert ledger_rows(base_state) == [] and tot_rows(base_state) == []
    assert run_facts(base_state, base_run)["why"]


def test_progress_lines_follow_the_run_live():
    from mosaic.graph import run_forecast
    from mosaic.llm import make_model
    from mosaic.views import progress_line
    from mosaic.world import make_demo_world

    lines = []
    run_forecast(make_demo_world(), make_model("scripted"), on_event=lambda e: lines.append(progress_line(e)))
    lines = [l for l in lines if l]
    assert lines[0] == "Coordinator: working…"
    assert any(l.startswith("   Research agent → search_documents(") for l in lines)
    assert any(l.startswith("✓ Critic done") for l in lines)
    assert lines.index("Synthesis agent (Tree of Thought): working…") > lines.index("Research agent: working…")
    assert lines[-1].startswith("✓ ") and ("Published" in lines[-1] or "Held" in lines[-1])


def test_a_broken_display_does_not_stop_the_forecast():
    from mosaic.graph import run_forecast
    from mosaic.llm import make_model
    from mosaic.world import make_demo_world

    def broken(event):
        raise RuntimeError("screen closed")

    state, _ = run_forecast(make_demo_world(), make_model("scripted"), on_event=broken)
    assert state["final"]["status"] == "accepted"


def test_refused_tool_calls_are_marked():
    from mosaic.views import progress_line
    line = progress_line({"node": "synthesis", "kind": "tool_call", "tool": "search_news", "args": {"query": "x"},
                          "status": "refused: not in role"})
    assert line == "   Synthesis agent (Tree of Thought) → search_news(query=x)  (refused: not in role)"


def test_the_text_log_has_everything_including_raw_model_replies():
    from mosaic.graph import run_forecast
    from mosaic.llm import make_model
    from mosaic.views import text_log
    from mosaic.world import make_demo_world

    state, run = run_forecast(make_demo_world(), make_model("scripted"))
    log = text_log(state, run)
    for heading in ("FORECAST", "LEDGER", "EVIDENCE", "TREE OF THOUGHT", "EVENTS"):
        assert f"\n{heading}\n" in log
    assert "Synthesis agent (Tree of Thought) model reply (limit 400 tokens):" in log
    assert '"readings"' in log                                     # the raw JSON the model sent
    assert log.count("model reply") == run.model.calls


def test_a_reviewed_forecast_shows_the_agent_and_the_analyst_side_by_side():
    from mosaic.graph import run_forecast
    from mosaic.llm import make_model
    from mosaic.risk import apply_review
    from mosaic.views import forecast_rows, ledger_rows, review_summary, text_log
    from mosaic.world import make_demo_world

    state, run = run_forecast(make_demo_world(), make_model("scripted"))
    first, second = [x["source"] for x in state["final"]["ledger"]][:2]
    reviewed = apply_review(state, {first: "strike", second: ("resize", "large")})

    assert review_summary(state["final"]) is None
    assert review_summary(reviewed) == "Reviewed by the analyst: 0 kept, 1 struck, 1 resized"
    row = forecast_rows(reviewed)[0]
    assert set(row) == {"Outcome", "Baseline %", "Agent %", "Analyst %", "Agent change", "Analyst change"}
    assert row["Agent %"] == round(state["final"]["probabilities"]["fall"], 1)
    lines = {r["Source"]: r for r in ledger_rows(state, reviewed)}
    assert lines[first]["Analyst decision"] == "struck"                    # struck lines stay visible
    assert lines[second]["Analyst decision"] == "resized to large" and lines[second]["Size"].endswith("(agent)")
    assert "REVIEWED BY THE ANALYST" in text_log(state, run, reviewed)
