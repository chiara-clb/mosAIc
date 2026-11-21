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

def plain_table(rows: list[dict], empty: str = "(none)") -> None:
    """A static table: shows long text in full and never takes over the mouse wheel, so the page scrolls."""
    if not rows:
        st.write(empty)
        return
    df = pd.DataFrame(rows).set_index(list(rows[0])[0])
    for c in df.columns:                                       # floats as short text, not 31.0000
        if pd.api.types.is_float_dtype(df[c]):
            df[c] = df[c].map(lambda v: "" if pd.isna(v) else f"{v:.2f}".rstrip("0").rstrip("."))
    st.table(df)


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


@st.cache_data(show_spinner="Reading the stock's results releases from the SEC…")
def release_choices_cached(ticker: str, options_dir: str) -> list[dict]:
    from mosaic.results_day import live_release_choices
    return live_release_choices(ticker, options_dir or None)


# --- what to run --------------------------------------------------------------------------------
with st.sidebar:
    st.header("Forecast")
    mode = st.radio("Company", ["Demo company (synthetic)", "Real stock", "Results day (next-day move)"],
                    help="Results day: the three baselines for the move after an after-close results release, "
                         "a first step towards forecasting right after results.")
    if mode.startswith("Results"):
        rd_ticker = st.selectbox("Stock", sorted(UNIVERSE), index=sorted(UNIVERSE).index("AMZN"))
        rd_options = st.text_input("Folder of FirstRate option files", "data/firstrate")
        rd_day, listed = None, True
        try:
            choices = release_choices_cached(rd_ticker, rd_options)
        except Exception as exc:                                 # no SEC access: say so instead of a stack trace
            choices, listed = [], False
            st.error(f"Could not list {rd_ticker}'s results releases: {exc}")
        if choices:
            with_options = [i for i, c in enumerate(choices) if c["options"]]
            pick = st.selectbox("Results release (after the close)", range(len(choices)),
                                index=with_options[0] if with_options else 0,
                                format_func=lambda i: choices[i]["label"],
                                help="The stock's results releases made after the 16:00 New York close, from its "
                                     "SEC filings. B3 needs option prices for that day.")
            rd_day = choices[pick]["day"]
        elif listed:
            st.info(f"{rd_ticker} has no results released after the close since 2023 (it may report before the "
                    "market opens). This view covers after-close releases.")
        rd_consensus = st.text_input("Folder of consensus estimates", "data/consensus")
        rd_model = st.text_input("Claude model for the release reader", "claude-haiku-5-5")
        if not os.environ.get("ANTHROPIC_API_KEY"):
            st.warning("ANTHROPIC_API_KEY is not set: the release reader needs it.")
        rd_go = st.button("Run the results-day forecast", type="primary", disabled=rd_day is None)
        go = False
    else:
        rd_go = False
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
        if backend in ("anthropic", "crewai") and not os.environ.get("ANTHROPIC_API_KEY"):
            st.warning("ANTHROPIC_API_KEY is not set.")
        go = st.button("Run forecast", type="primary")
    if not mode.startswith("Demo") and not SETTINGS.sec_user_agent:
        st.warning("Set MOSAIC_SEC_USER_AGENT to 'Your Name your.email@example.com' before using a real stock.")
    st.caption(f"Graph engine: {graph_engine()[3]}")

# --- results day: the agents forecast the next-day move right after a results release ----------
if mode.startswith("Results"):
    from mosaic.results_day import NEW_YORK
    st.subheader("Results day: how much will the stock move the day after results?")
    with st.expander("How this works", expanded=False):
        st.markdown(
            "The forecast is made a few minutes after a release published after the 16:00 New York close. The "
            "question is the move from that close to the next day's close: a large fall, flat, or a large gain, "
            "where large means beyond half the size of the stock's last eight results-day moves.\n\n"
            "| Step | Who | What it does |\n|---|---|---|\n"
            "| Coordinator | code | Confirms through the tools that the release came after the close |\n"
            "| Quant | code | B1 from a year of daily moves, B2 from past results-day moves, B3 from the option "
            "prices at the close |\n"
            "| Fetch release | code | Gets the press release through the tool door, under the forecast-date rule, "
            "with the consensus values |\n"
            "| Release reader | **agent (Claude)** | Reads the release against the baselines and the consensus, writes a "
            "ledger with exact quotes. No tools |\n"
            "| Critic | code | Every quote must be in the release; failing lines go back with reasons, at most twice, "
            "then are struck |\n"
            "| Finalise | code | The ledger tilts B3 into MosAIc's forecast |\n"
            "| Risk gate | code | Publishes, flags or holds for the analyst, as in the monthly forecast |\n\n"
            "The agents never see the next day's price; it is shown below only to compare.")
    if rd_go and rd_day is not None:
        from mosaic.llm import make_model
        from mosaic.results_graph import live_results_case, run_results_day
        st.session_state.update(rd_state=None, rd_reviewed=None, rd_error=None)
        with st.status("The results-day agents are working…", expanded=True) as status:
            shown = st.empty()
            lines: list[str] = []

            def show(line: str | None) -> None:
                if line:
                    lines.append(line)
                    shown.code("\n".join(lines[-25:]), language=None)

            try:
                show("Loading the release, prices, option quotes and consensus (saved data is reused)")
                world, closes = live_results_case(rd_ticker, rd_day, rd_options or None, rd_consensus or None)
                state, run = run_results_day(world, make_model("anthropic", model_name=rd_model),
                                             on_event=lambda e: show(progress_line(e)))
                st.session_state.update(rd_state=state, rd_run=run, rd_closes=closes)
                status.update(label=f"Done: {state['final']['status']}, {state['final'].get('review', '')}",
                              state="complete", expanded=False)
            except Exception as exc:                            # keep the reason on screen
                st.session_state["rd_error"] = f"{type(exc).__name__}: {exc}"
                status.update(label="The run failed", state="error")
    if st.session_state.get("rd_error"):
        st.error("The results-day run failed: " + st.session_state["rd_error"])
    rstate = st.session_state.get("rd_state")
    if rstate is None:
        st.info("Choose a stock and one of its results releases, for example AMZN, NVDA or GOOGL Q2 2026, and press "
                "**Run the results-day forecast**.")
        st.stop()
    from mosaic.results_graph import STRENGTHS, actual_move, apply_results_review
    rrun = st.session_state["rd_run"]
    base = rstate.get("baseline") or {}
    rfinal = st.session_state.get("rd_reviewed") or rstate["final"]
    rfacts = run_facts(rstate, rrun)
    req = rstate["request"]
    st.markdown(f"**{req['company']} ({req['ticker']})** reported at "
                f"**{req['release'].astimezone(NEW_YORK):%H:%M} New York time** on {req['release'].astimezone(NEW_YORK):%Y-%m-%d}."
                + (f" A large move is beyond **±{base['tau'] * 100:.1f}%** (half of {base['J'] * 100:.1f}%, the typical "
                   f"size of its last {len(base['past_moves'])} results-day moves)." if base.get("tau") else ""))
    colour = {"publish": "green", "flag": "orange", "hold": "red"}.get(rfacts["review"], "gray")
    st.markdown(f"**Status:** {rfinal['status']}  ·  **Risk gate:** :{colour}[{rfacts['review']}]")
    for t in rfacts["triggers"]:
        st.markdown(f"- {t}")
    for w in rfinal.get("why") or []:
        st.warning(w)
    if review_summary(rfinal):
        st.success(f"**{review_summary(rfinal)}.** The forecast shown is the analyst's; the agent's is next to it.")
    elif rfacts["review"] == "hold":
        st.info("Held for the analyst: open the **Analyst review** tab to keep, resize or strike each line.")
    st.caption(f"Model: {rfacts['model']}, {rfacts['model_calls']} calls · Tool calls: {rfacts['tool_calls']} · "
               f"Retries: {rfacts['retries']} · Leakage audit: {rfacts['leakage']}")

    names = ["Forecast and ledger", "Analyst review", "Evidence", "Steps", "Tool calls", "Full log"]
    tabs = dict(zip(names, st.tabs(names)))
    with tabs["Forecast and ledger"]:
        rows = []
        labels = {"B1": "B1 · one year of daily moves", "B2": "B2 · previous results days",
                  "B3": "B3 · options at the close"}
        for key, label in labels.items():
            p = base.get(key)
            rows.append({"Forecast": label, "Fall %": p and p["fall"], "Flat %": p and p["flat"], "Gain %": p and p["gain"]})
        if rfinal.get("agent_probabilities"):
            a = rfinal["agent_probabilities"]
            rows.append({"Forecast": "MosAIc · agent", "Fall %": a["fall"], "Flat %": a["flat"], "Gain %": a["gain"]})
            rows.append({"Forecast": "MosAIc · after the analyst's review", "Fall %": rfinal["probabilities"]["fall"],
                         "Flat %": rfinal["probabilities"]["flat"], "Gain %": rfinal["probabilities"]["gain"]})
        elif rfinal["status"].startswith("accepted"):
            p = rfinal["probabilities"]
            rows.append({"Forecast": "MosAIc · after reading the release", "Fall %": p["fall"], "Flat %": p["flat"],
                         "Gain %": p["gain"]})
        actual = actual_move(st.session_state["rd_closes"], req["release"].astimezone(NEW_YORK).date(),
                             base["tau"]) if base.get("tau") else None
        if actual:
            for r in rows:
                v = r.get({"fall": "Fall %", "flat": "Flat %", "gain": "Gain %"}[actual["outcome"]])
                r["Chance given to what happened %"] = v
        table = pd.DataFrame(rows).set_index("Forecast")
        left, right = st.columns([3, 2])
        left.table(table.map(lambda v: "–" if v is None or pd.isna(v) else f"{v:.1f}"))
        chart = table[["Fall %", "Flat %", "Gain %"]].dropna()
        if len(chart):
            chart.index = [i.split(" ·")[0] if i.startswith("B") else i.replace("MosAIc · ", "MosAIc ") for i in chart.index]
            chart.columns = ["Large fall", "Flat", "Large gain"]
            try:
                import altair as alt
                long = chart.reset_index(names="Forecast").melt("Forecast", var_name="Outcome", value_name="Percent")
                right.altair_chart(
                    alt.Chart(long).mark_bar()
                    .encode(x=alt.X("Outcome:N", sort=list(chart.columns), title=None, axis=alt.Axis(labelAngle=0)),
                            xOffset=alt.XOffset("Forecast:N", sort=list(chart.index)),
                            y=alt.Y("Percent:Q", title="%", scale=alt.Scale(domain=[0, 100])),
                            color=alt.Color("Forecast:N", sort=list(chart.index),
                                            legend=alt.Legend(orient="bottom", title=None)),
                            tooltip=["Forecast", "Outcome", alt.Tooltip("Percent:Q", format=".1f")])
                    .properties(height=280), use_container_width=True)
            except Exception:
                right.bar_chart(chart.T, stack=False)
        if actual:
            st.success(f"What happened (never shown to the agents): **{actual['move'] * 100:+.1f}%** on "
                       f"{actual['day']}, a **{actual['outcome']}**.")
        if rfinal.get("read"):
            st.markdown(f"**The reader's view:** {rfinal['read']}")
        st.markdown("#### Ledger")
        plain_table(ledger_rows(rstate, rfinal), empty="No ledger: the options baseline is reported.")
        for n in base.get("notes") or []:
            st.caption("Note: " + n)
    with tabs["Analyst review"]:
        if not rstate["final"]["ledger"]:
            st.write("Nothing to review: the options baseline is reported.")
        else:
            if review_summary(rfinal) and st.button("Undo review", key="rd-undo"):
                st.session_state["rd_reviewed"] = None
                st.rerun()
            decisions = {}
            for i, line in enumerate(rstate["final"]["ledger"]):
                choice = st.selectbox(f"#{i + 1} {line['reason'][:70]} ({line['effect']}, {line['strength']})",
                                      ["keep", "strike"] + [f"resize to {s}" for s in STRENGTHS], key=f"rd-review-{i}")
                decisions[i] = choice.split()[-1] if choice.startswith("resize") else choice
            if st.button("Apply review", key="rd-apply"):
                st.session_state["rd_reviewed"] = apply_results_review(rstate["final"], decisions)
                st.rerun()
    with tabs["Evidence"]:
        plain_table(evidence_rows(rstate))
        cons = rstate.get("consensus") or []
        st.caption(f"Consensus values given to the reader: {len(cons)} (values only, never the articles they came from).")
        if cons:
            plain_table([{"Metric": c["metric"], "Provider": c["provider"], "Value": c["value"], "Unit": c["unit"],
                          "Basis": c["basis"]} for c in cons])
    with tabs["Steps"]:
        plain_table(step_rows(rstate))
    with tabs["Tool calls"]:
        plain_table(tool_rows(rrun))
    with tabs["Full log"]:
        from mosaic.results_graph import after_lines
        log = text_log(rstate, rrun, rfinal, after_lines(rfinal, actual, base))
        st.download_button("Download log", log, file_name=f"{rrun.tools.trace.run_id}.log.txt")
        st.code(log, language=None)
    st.stop()

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

# --- the result: a short header, then everything in tabs -------------------------------------------
st.subheader(f"{req['company']} ({req['ticker']}), as of {req['as_of']:%Y-%m-%d}")
colour = {"publish": "green", "flag": "orange", "hold": "red"}.get(facts["review"], "gray")
st.markdown(f"**Status:** {final['status']}  ·  **Risk gate:** :{colour}[{facts['review']}]")
for t in facts["triggers"]:
    st.markdown(f"- {t}")
for w in facts["why"]:
    st.warning("Baseline reported: " + w)
if review_summary(final):
    st.success(f"**{review_summary(final)}.** The forecast shown is the analyst's; the agent's estimate is next to it.")
elif facts["review"] == "hold":
    st.info("Held for the analyst: open the **Analyst review** tab to keep, resize or strike each line.")
st.caption(f"Model: {facts['model']}, {facts['model_calls']} calls · Tool calls: {facts['tool_calls']} · "
           f"Retries: {facts['retries']} · Leakage audit: {facts['leakage']}")

names = ["Forecast and ledger", "Analyst review", "Evidence", "Tree of Thought", "Steps", "Tool calls", "Full log"]
tabs = dict(zip(names, st.tabs(names)))

with tabs["Forecast and ledger"]:
    left, right = st.columns([3, 2])
    table = pd.DataFrame(forecast_rows(final)).set_index("Outcome")
    left.table(table.map(lambda v: f"{v:.1f}"))                # one decimal, as text
    try:
        right.altair_chart(outcome_chart(table), use_container_width=True)
    except Exception:                                          # an older Altair: plain side-by-side bars
        right.bar_chart(table[[c for c in table.columns if c.endswith("%")]], stack=False)
    st.markdown("#### Ledger")
    plain_table(ledger_rows(state, final), empty="No adjustments: the forecast is the baseline.")

with tabs["Analyst review"]:
    if not state["final"]["ledger"]:
        st.write("Nothing to review: the forecast is the baseline.")
    else:
        if review_summary(final):
            st.success(review_summary(final))
            if st.button("Undo review"):
                st.session_state["reviewed"] = None
                st.rerun()
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

with tabs["Evidence"]:
    plain_table(evidence_rows(state))
with tabs["Tree of Thought"]:
    st.caption("Every reading considered for each item. Quote check by the critic's model; score = base rate and novelty.")
    plain_table(tot_rows(state))
with tabs["Steps"]:
    plain_table(step_rows(state))
with tabs["Tool calls"]:
    plain_table(tool_rows(run))
with tabs["Full log"]:
    log = text_log(state, run, final)
    st.caption("Everything about this run as plain text, including the model's raw replies. "
               "Use the copy icon at the top right of the box, or download it.")
    st.download_button("Download log", log, file_name=f"{run.tools.trace.run_id}.log.txt")
    st.code(log, language=None)
