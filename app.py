"""MosAIc in the browser.

    pip install -e ".[ui]"
    streamlit run app.py
"""

from __future__ import annotations

import os
from datetime import date, datetime, timezone

import pandas as pd
import streamlit as st

from mosaic.config import SETTINGS, UNIVERSE
from mosaic.graph import graph_engine, run_forecast
from mosaic.llm import make_model
from mosaic.request import check_request
from mosaic.risk import ReviewLog, apply_review
from mosaic.scenarios import SCENARIOS, run_scenario
from mosaic.views import (evidence_rows, forecast_rows, ledger_rows, progress_line, review_summary, run_facts, step_rows,
                          text_log, tool_rows, tot_rows)

st.set_page_config(page_title="MosAIc", layout="wide")
st.title("MosAIc")
st.caption("Probability of a large fall, a flat month or a large gain over the next 21 trading days, "
           "with a cited ledger of every adjustment.")

def plain_table(rows: list[dict]) -> None:
    """A static table: shows long text in full and never takes over the mouse wheel, so the page scrolls."""
    if rows:
        st.table(pd.DataFrame(rows).set_index(list(rows[0])[0]))
    else:
        st.write("(none)")


def outcome_chart(table: pd.DataFrame):
    """Baseline and forecast side by side for each outcome, in the order fall, flat, gain."""
    import altair as alt
    series = [c for c in ("Baseline %", "Forecast %", "Agent %", "Analyst %") if c in table.columns]
    long = table[series].reset_index().melt("Outcome", var_name="Series", value_name="Percent")
    order = list(table.index)
    return (alt.Chart(long).mark_bar()
            .encode(x=alt.X("Outcome:N", sort=order, title=None, axis=alt.Axis(labelAngle=0)),
                    xOffset=alt.XOffset("Series:N", sort=series),
                    y=alt.Y("Percent:Q", title="%"),
                    color=alt.Color("Series:N", sort=series, legend=alt.Legend(orient="bottom", title=None)),
                    tooltip=["Outcome", "Series", alt.Tooltip("Percent:Q", format=".1f")])
            .properties(height=260))


# --- what to run --------------------------------------------------------------------------------
with st.sidebar:
    st.header("Forecast")
    mode = st.radio("Company", ["Demo company (synthetic)", "Real stock"])
    backend = st.selectbox("Model", ["scripted", "anthropic", "crewai"],
                           help="scripted: answers written in advance, no key needed. anthropic: Claude, needs "
                                "ANTHROPIC_API_KEY. crewai: Claude through the agents' CrewAI role cards.")
    model_name = st.text_input("Claude model", "claude-haiku-5-5") if backend in ("anthropic", "crewai") else None
    if mode.startswith("Demo"):
        scenario = st.selectbox("Scenario", list(SCENARIOS), help="Each scenario plants one problem for the guardrails.")
        st.caption("Planted: " + SCENARIOS[scenario]["plant"])
    else:
        ticker = st.selectbox("Stock", sorted(UNIVERSE))
        as_of = st.date_input("Forecast date (start of day)", date(2026, 7, 15))
        if not SETTINGS.sec_user_agent:
            st.warning("Set MOSAIC_SEC_USER_AGENT to 'Your Name your.email@example.com' before running a real stock.")
    if backend in ("anthropic", "crewai") and not os.environ.get("ANTHROPIC_API_KEY"):
        st.warning("ANTHROPIC_API_KEY is not set.")
    go = st.button("Run forecast", type="primary")
    st.caption(f"Graph engine: {graph_engine()[3]}")

if go:
    with st.status("The agents are working…", expanded=True) as status:
        shown = st.empty()
        lines: list[str] = []

        def show(line: str | None) -> None:
            if line:
                lines.append(line)
                shown.code("\n".join(lines[-25:]), language=None)      # the latest 25 lines, live

        on_event = lambda event: show(progress_line(event))
        try:
            if mode.startswith("Demo"):
                factory = None if backend == "scripted" else (lambda script: make_model(backend, model_name=model_name))
                state, run, _ = run_scenario(scenario, model_factory=factory, on_event=on_event)
            else:
                when = datetime(as_of.year, as_of.month, as_of.day, tzinfo=timezone.utc)
                reasons = check_request(ticker, when)
                if reasons:
                    status.update(label="Request refused", state="error")
                    st.error("Request refused: " + "; ".join(reasons))
                    st.stop()
                from mosaic.live import live_world
                show("Loading the data (saved data is reused, new data is downloaded)")
                world = live_world(ticker, when, progress=show)
                state, run = run_forecast(world, make_model(backend, model_name=model_name), on_event=on_event)
        except Exception as exc:                                    # show the reason instead of a stack trace
            status.update(label="The run failed", state="error")
            st.error(f"The run failed: {exc}")
            st.stop()
        status.update(label=f"Done: {state['final']['status']}, {state['final'].get('review', '')}",
                      state="complete", expanded=False)
    st.session_state.update(state=state, run=run, reviewed=None)

if "state" not in st.session_state:
    st.info("Choose a company and press **Run forecast**. The demo company needs no setup.")
    st.stop()

state, run = st.session_state["state"], st.session_state["run"]
facts = run_facts(state, run)
final = st.session_state.get("reviewed") or state["final"]
req = state["request"]

# --- the forecast -------------------------------------------------------------------------------
st.subheader(f"{req['company']} ({req['ticker']}), as of {req['as_of']:%Y-%m-%d}")
colour = {"publish": "green", "flag": "orange", "hold": "red"}.get(facts["review"], "gray")
st.markdown(f"**Status:** {final['status']}  ·  **Risk gate:** :{colour}[{facts['review']}]")
for t in facts["triggers"]:
    st.markdown(f"- {t}")
for w in facts["why"]:
    st.warning("Baseline reported: " + w)
if review_summary(final):
    st.success(f"**{review_summary(final)}.** The forecast below is the analyst's; the agent's estimate is shown next to it.")
    if st.button("Undo review"):
        st.session_state["reviewed"] = None
        st.rerun()
elif facts["review"] == "hold":
    st.info("Held for the analyst: review each ledger line in **Analyst review** at the bottom of the page.")

left, right = st.columns([3, 2])
table = pd.DataFrame(forecast_rows(final)).set_index("Outcome")
left.table(table)
try:
    right.altair_chart(outcome_chart(table), use_container_width=True)
except Exception:                                              # an older Altair: plain side-by-side bars
    right.bar_chart(table[[c for c in table.columns if c.endswith("%")]], stack=False)

st.markdown("#### Ledger")
rows = ledger_rows(state, final)
if rows:
    st.table(pd.DataFrame(rows).set_index("Source"))      # a plain table wraps long quotes and reasons
else:
    st.write("No adjustments: the forecast is the baseline.")

# --- how the agents got there ------------------------------------------------------------------
tabs = st.tabs(["Evidence", "Tree of Thought", "Steps", "Tool calls", "Full log"])
with tabs[0]:
    plain_table(evidence_rows(state))
with tabs[1]:
    st.caption("Every reading considered for each item. Quote check by the critic's model; score = base rate and novelty.")
    plain_table(tot_rows(state))
with tabs[2]:
    plain_table(step_rows(state))
with tabs[3]:
    plain_table(tool_rows(run))
with tabs[4]:
    log = text_log(state, run, final)
    st.caption("Everything about this run as plain text, including the model's raw replies. "
               "Use the copy icon at the top right of the box, or download it.")
    st.download_button("Download log", log, file_name=f"{run.tools.trace.run_id}.log.txt")
    st.code(log, language=None)

st.caption(f"Model: {facts['model']}, {facts['model_calls']} calls · Tool calls: {facts['tool_calls']} · "
           f"Retries: {facts['retries']} · Leakage audit: {facts['leakage']}")

# --- the analyst's review -----------------------------------------------------------------------
if state["final"]["ledger"]:
    with st.expander("Analyst review: keep, resize or strike each line",
                     expanded=facts["review"] == "hold" and not st.session_state.get("reviewed")):
        decisions = {}
        for line in state["final"]["ledger"]:
            choice = st.selectbox(f"{line['source']}: {line['effect']} {line['size']}",
                                  ["keep", "strike", "resize to small", "resize to medium", "resize to large"],
                                  key=f"review-{line['source']}")
            decisions[line["source"]] = ("resize", choice.split()[-1]) if choice.startswith("resize") else choice
        if st.button("Apply review"):
            log = st.session_state.setdefault("review_log", ReviewLog())
            st.session_state["reviewed"] = apply_review(state, decisions, log)
            st.rerun()
        if st.session_state.get("review_log"):
            st.caption(f"Analyst decisions so far, by event type: {st.session_state['review_log'].summary()}")
