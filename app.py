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
from mosaic.views import evidence_rows, forecast_rows, ledger_rows, run_facts, step_rows, tool_rows, tot_rows

st.set_page_config(page_title="MosAIc", layout="wide")
st.title("MosAIc")
st.caption("Probability of a large fall, a flat month or a large gain over the next 21 trading days, "
           "with a cited ledger of every adjustment.")

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
    with st.spinner("The agents are working..."):
        try:
            if mode.startswith("Demo"):
                factory = None if backend == "scripted" else (lambda script: make_model(backend, model_name=model_name))
                state, run, _ = run_scenario(scenario, model_factory=factory)
            else:
                when = datetime(as_of.year, as_of.month, as_of.day, tzinfo=timezone.utc)
                reasons = check_request(ticker, when)
                if reasons:
                    st.error("Request refused: " + "; ".join(reasons))
                    st.stop()
                from mosaic.live import live_world
                state, run = run_forecast(live_world(ticker, when), make_model(backend, model_name=model_name))
        except Exception as exc:                                    # show the reason instead of a stack trace
            st.error(f"The run failed: {exc}")
            st.stop()
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

left, right = st.columns([3, 2])
table = pd.DataFrame(forecast_rows(final)).set_index("Outcome")
left.dataframe(table, use_container_width=True)
right.bar_chart(table[["Baseline %", "Forecast %"]])

st.markdown("#### Ledger")
rows = ledger_rows(state, final)
if rows:
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
else:
    st.write("No adjustments: the forecast is the baseline.")

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

# --- how the agents got there ------------------------------------------------------------------
tabs = st.tabs(["Evidence", "Tree of Thought", "Steps", "Tool calls"])
with tabs[0]:
    st.dataframe(pd.DataFrame(evidence_rows(state)), use_container_width=True, hide_index=True)
with tabs[1]:
    st.caption("Every reading considered for each item. Quote check by the critic's model; score = base rate and novelty.")
    st.dataframe(pd.DataFrame(tot_rows(state)), use_container_width=True, hide_index=True)
with tabs[2]:
    st.dataframe(pd.DataFrame(step_rows(state)), use_container_width=True, hide_index=True)
with tabs[3]:
    st.dataframe(pd.DataFrame(tool_rows(run)), use_container_width=True, hide_index=True)

st.caption(f"Model: {facts['model']}, {facts['model_calls']} calls · Tool calls: {facts['tool_calls']} · "
           f"Retries: {facts['retries']} · Leakage audit: {facts['leakage']}")
