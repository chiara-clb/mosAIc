"""Evaluation on real stocks: several variants on the same data, scored against what happened.

For each stock and each weekly forecast date (Wednesdays, after the model's knowledge cutoff), the runner
builds the world once and produces five forecasts:

- baseline:            prices only (the quant agent's answer)
- prompt only:         the model alone, with no data and no baseline
- baseline + memory:   the model adjusts the baseline from what it already knows, with no evidence
- single pass:         the full system, first reading of each item
- ToT:                 the full system as designed (Tree of Thought, critic with reasons)
- 5 single passes:     (only when asked for) one research, then the single pass five times, averaged;
                       the fair check of whether Tree of Thought adds more than repeating and averaging

Before forecasting, a memory check asks the model what the stock did over the following month. If it
claims to know, the date is flagged (or dropped), because then the model could be remembering the answer.

Each forecast is scored against the realised bucket (Brier score and log loss), with intervals resampled
by week, and, when an answer key is given (option-implied probabilities), by log loss against it.

Results are written to results/evaluation/ after every stock and date, so an interrupted run can resume.

    python -m mosaic.evaluate --backend anthropic --model claude-haiku-5-5 --tickers AAPL,NKE \\
        --start 2026-07-01 --end 2026-10-15 --resume
"""

from __future__ import annotations

import argparse
import os
import csv
import json
import math
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from mosaic.config import OUTCOMES, SETTINGS, UNIVERSE
from mosaic.graph import run_forecast
from mosaic.ledger import floor_and_normalise
from mosaic.llm import Chat, Model, parse_json
from mosaic.scoring import brier, compare, log_loss_implied

UTC = timezone.utc
DEFAULT_TICKERS = ["AAPL", "NKE", "INTC", "MSFT", "JPM", "XOM"]
VARIANTS = ["baseline", "prompt only", "baseline + memory", "single pass", "ToT"]
AVERAGED = "5 single passes"
ALL_VARIANTS = VARIANTS[:4] + [AVERAGED] + VARIANTS[4:]    # the default run keeps VARIANTS
REPEATS = 5
FIXED_GUESS = {"fall": 30.85, "flat": 38.3, "gain": 30.85}
CUTOFF = "2026-07-01"                     # after the Claude models' reliable knowledge cutoff (June 2026)
OUT_DIR = Path("results") / "evaluation"
FIELDS = ["ticker", "as_of", "variant", "fall", "flat", "gain", "outcome", "status", "review", "model_calls",
          "memory_answer", "memory_knows", "implied_fall", "implied_flat", "implied_gain", "error"]
CAL_BINS = [0, 20, 30, 40, 50, 101]       # percent

WorldFactory = Callable[[str, datetime], object]
PriceSource = Callable[[str, datetime], pd.Series]
AnswerKey = Callable[[str, datetime], dict | None]


# --- dates --------------------------------------------------------------------------------------

def forecast_dates(start: str, end: str, today: date | None = None, weekday: int = 2,
                   horizon: int | None = None) -> list[datetime]:
    """Weekly dates (Wednesday by default) in [start, end] whose 21 trading days have all passed by `today`."""
    horizon = horizon or SETTINGS.horizon_days
    today = today or datetime.now(UTC).date()
    d = date.fromisoformat(start)
    d += timedelta(days=(weekday - d.weekday()) % 7)
    out = []
    while d <= date.fromisoformat(end):
        last_day = (pd.Timestamp(d) + pd.offsets.BDay(horizon)).date()      # weekends only; holidays add slack
        if last_day < today:
            out.append(datetime(d.year, d.month, d.day, tzinfo=UTC))
        d += timedelta(days=7)
    return out


# --- the model-only variants --------------------------------------------------------------------

MEMORY_PROMPT = ("Do you remember how the stock of {company} ({ticker}) moved over the month after {date}? "
                 "Answer from memory only. Reply with ONLY a JSON object: "
                 "{{\"answer\": \"fell a lot\" | \"flat\" | \"rose a lot\" | \"I don't know\"}}")
PROMPT_ONLY_SYSTEM = ("You forecast a stock's return over the next 21 trading days.\n"
                      "Give the probability (in percent) of a large fall, a roughly flat month and a large gain,\n"
                      "where large means more than half of the stock's typical monthly move over the past year.\n"
                      "Reply with ONLY a JSON object: {\"fall\": number, \"flat\": number, \"gain\": number}")
BASE_MEMORY_SYSTEM = ("You adjust a baseline forecast of a stock's return over the next 21 trading days, using only "
                      "what you already know about the company. You have no documents. A baseline from price history "
                      "gives the probability (in percent) of a large fall, a roughly flat month and a large gain. "
                      "Move each probability by at most 15 points, and only if you have a reason.\n"
                      "Reply with ONLY a JSON object: {\"fall\": number, \"flat\": number, \"gain\": number, \"reason\": text}")


def read_probs(reply: str | None) -> dict | None:
    d = parse_json(reply) or {}
    try:
        p = {k: max(float(d[k]), 0.0) for k in OUTCOMES}
    except (KeyError, TypeError, ValueError):
        return None
    return floor_and_normalise(p) if sum(p.values()) > 0 else None


def memory_check(model: Model, company: str, ticker: str, as_of: datetime) -> tuple[str, bool]:
    """(answer, whether the model claims to know). Unreadable replies count as not knowing."""
    reply = model([{"role": "user", "content": MEMORY_PROMPT.format(company=company, ticker=ticker, date=f"{as_of:%Y-%m-%d}")}],
                  max_tokens=40)
    answer = str((parse_json(reply) or {}).get("answer", "I don't know")).strip()
    knows = answer.lower() in ("fell a lot", "flat", "rose a lot")
    return answer, knows


def prompt_only(model: Model, company: str, ticker: str, as_of: datetime) -> tuple[dict, str]:
    reply = model([{"role": "system", "content": PROMPT_ONLY_SYSTEM},
                   {"role": "user", "content": f"Company: {company} ({ticker}). Forecast date: {as_of:%Y-%m-%d}."}],
                  max_tokens=60)
    p = read_probs(reply)
    return (p, "ok") if p else (dict(FIXED_GUESS), "unreadable reply, fixed guess used")


def baseline_plus_memory(model: Model, company: str, ticker: str, as_of: datetime, base: dict) -> tuple[dict, str]:
    reply = model([{"role": "system", "content": BASE_MEMORY_SYSTEM},
                   {"role": "user", "content": f"Company: {company} ({ticker}). Forecast date: {as_of:%Y-%m-%d}. "
                                               f"Baseline: fall {base['fall']}, flat {base['flat']}, gain {base['gain']}."}],
                  max_tokens=150)
    p = read_probs(reply)
    if not p:
        return dict(base), "unreadable reply, baseline used"
    biggest = max(abs(p[k] - base[k]) for k in OUTCOMES)
    if biggest > SETTINGS.max_total_shift:                  # same cap as the ledger
        f = SETTINGS.max_total_shift / biggest
        p = floor_and_normalise({k: base[k] + f * (p[k] - base[k]) for k in OUTCOMES})
    return p, "ok"


# --- outcomes -----------------------------------------------------------------------------------

def default_price_source(ticker: str, as_of: datetime) -> pd.Series:     # pragma: no cover - needs internet
    from mosaic.prices import load_closes
    end = min(as_of.date() + timedelta(days=45), datetime.now(UTC).date())
    return load_closes(ticker, (as_of - timedelta(days=420)).date().isoformat(), end.isoformat())


def realised_outcome(prices: pd.Series, as_of: datetime) -> str | None:
    from mosaic.buckets import realised_bucket
    try:
        return realised_bucket(prices, as_of)
    except ValueError:
        return None


# --- one stock and date -------------------------------------------------------------------------

def default_world_factory(ticker: str, as_of: datetime):           # pragma: no cover - needs internet
    from mosaic.live import live_world
    return live_world(ticker, as_of)


def evaluate_one(ticker: str, as_of: datetime, chat: Chat, model_name: str, *, world_factory: WorldFactory,
                 price_source: PriceSource, answer_key: AnswerKey | None = None, variants=None,
                 memory: str = "flag", engine=None) -> list[dict]:
    """All variant rows for one stock and date."""
    variants = variants or VARIANTS
    base_row = {"ticker": ticker, "as_of": f"{as_of:%Y-%m-%d}", "memory_answer": "", "memory_knows": "", "error": ""}
    try:
        world = world_factory(ticker, as_of)
        if memory != "off":
            answer, knows = memory_check(Model(chat, 3, model_name), world.company, ticker, as_of)
            base_row.update(memory_answer=answer, memory_knows=knows)
            if knows and memory == "drop":
                return [{**base_row, "variant": v, "status": "dropped: the model remembers this month"} for v in variants]
        outcome = realised_outcome(price_source(ticker, as_of), as_of)
        implied = answer_key(ticker, as_of) if answer_key else None
        if implied:
            base_row.update({f"implied_{k}": implied[k] for k in OUTCOMES})
        rows, base = [], None

        def add(variant, probs, status, review="", calls=0):
            rows.append({**base_row, "variant": variant, **{k: round(probs[k], 2) for k in OUTCOMES},
                         "outcome": outcome or "", "status": status, "review": review, "model_calls": calls})

        # The baseline comes from any full run; Tree of Thought runs for it only if nothing else does.
        other_run = any(v in variants for v in ("single pass", AVERAGED))
        need_tot = "ToT" in variants or (not other_run and any(v in variants for v in ("baseline", "baseline + memory")))
        if need_tot:
            m = Model(chat, None, model_name)
            state, _ = run_forecast(world, m, options={"protocol": "two_way", "readings": "tot"}, engine=engine)
            base = state["final"]["baseline"]
            if "ToT" in variants:
                add("ToT", state["final"]["probabilities"], state["final"]["status"], state["final"].get("review", ""), m.calls)
        if "single pass" in variants:
            m = Model(chat, None, model_name)
            state, _ = run_forecast(world, m, options={"protocol": "two_way", "readings": "single"}, engine=engine)
            base = base or state["final"]["baseline"]
            add("single pass", state["final"]["probabilities"], state["final"]["status"], state["final"].get("review", ""), m.calls)
        if AVERAGED in variants:
            memo, finals, calls = {}, [], 0
            for _ in range(REPEATS):
                m = Model(chat, None, model_name)
                state, _ = run_forecast(world, m, options={"protocol": "two_way", "readings": "single"},
                                        engine=engine, research_memo=memo)
                finals.append(state["final"])
                calls += m.calls
            base = base or finals[0]["baseline"]
            probs = {k: float(np.mean([f["probabilities"][k] for f in finals])) for k in OUTCOMES}
            kinds = sorted({f["status"].split(",")[0] for f in finals})
            held = sum(f.get("review") == "hold" for f in finals)
            add(AVERAGED, probs, f"average of {REPEATS} ({', '.join(kinds)})", f"{held} of {REPEATS} held", calls)
        if "baseline" in variants:
            add("baseline", base, "ok")
        if "prompt only" in variants:
            m = Model(chat, 3, model_name)
            p, status = prompt_only(m, world.company, ticker, as_of)
            add("prompt only", p, status, calls=m.calls)
        if "baseline + memory" in variants:
            m = Model(chat, 3, model_name)
            p, status = baseline_plus_memory(m, world.company, ticker, as_of, base)
            add("baseline + memory", p, status, calls=m.calls)
        order = {v: i for i, v in enumerate(ALL_VARIANTS)}
        return sorted(rows, key=lambda r: order.get(r["variant"], 99))
    except Exception as exc:                                    # one bad stock or date must not stop the run
        return [{**base_row, "variant": v, "status": "error", "error": f"{type(exc).__name__}: {exc}"} for v in variants]


# --- writing and resuming -----------------------------------------------------------------------

def done_keys(path: Path) -> set[tuple[str, str]]:
    """(ticker, date) pairs already written, so a run can resume."""
    if not path.exists():
        return set()
    with path.open(newline="", encoding="utf-8") as f:
        return {(r["ticker"], r["as_of"]) for r in csv.DictReader(f)}


def append_rows(path: Path, rows: list[dict]) -> None:
    new = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerows(rows)


def run(tickers, dates, chat: Chat, model_name: str, out_dir: Path = OUT_DIR, *, world_factory: WorldFactory = default_world_factory,
        price_source: PriceSource = default_price_source, answer_key: AnswerKey | None = None, variants=None,
        memory: str = "flag", resume: bool = True, limit: int | None = None, engine=None, log=print) -> pd.DataFrame:
    """Evaluate every stock and date, writing after each one. Returns the forecasts table."""
    path = Path(out_dir) / "forecasts.csv"
    if not resume and path.exists():
        path.unlink()
    done = done_keys(path)
    todo = [(t, d) for t in tickers for d in dates if (t, f"{d:%Y-%m-%d}") not in done]
    if limit:
        todo = todo[:limit]
    log(f"{len(done)} stock-dates already done, {len(todo)} to run.")
    for i, (t, d) in enumerate(todo, 1):
        rows = evaluate_one(t, d, chat, model_name, world_factory=world_factory, price_source=price_source,
                            answer_key=answer_key, variants=variants, memory=memory, engine=engine)
        append_rows(path, rows)
        status = rows[0]["status"] if rows[0]["status"].startswith(("error", "dropped")) else "ok"
        log(f"[{i}/{len(todo)}] {t} {d:%Y-%m-%d}: {status}" + (f" ({rows[0]['error']})" if rows[0].get("error") else ""))
    table = pd.read_csv(path, dtype={"as_of": str}) if path.exists() else pd.DataFrame(columns=FIELDS)
    write_reports(table, Path(out_dir))
    return table


# --- scoring ------------------------------------------------------------------------------------

def probs_of(row) -> dict:
    return {k: float(row[k]) for k in OUTCOMES}


def scored(table: pd.DataFrame, include_flagged: bool = True) -> pd.DataFrame:
    """Rows that can be scored, with Brier score and log loss against the outcome (and the answer key if present)."""
    t = table.copy()
    t = t[t["outcome"].isin(OUTCOMES) & t["fall"].notna()]
    if not include_flagged:
        t = t[t["memory_knows"].astype(str) != "True"]
    t["id"] = t["ticker"] + "-" + t["as_of"].astype(str)
    t["brier_outcome"] = [brier(probs_of(r), r["outcome"]) for _, r in t.iterrows()]
    t["log_loss_outcome"] = [-math.log(max(probs_of(r)[r["outcome"]] / 100, 1e-6)) for _, r in t.iterrows()]
    has_key = t["implied_fall"].notna() if "implied_fall" in t else pd.Series(False, index=t.index)
    t["log_loss_implied"] = [log_loss_implied(probs_of(r), {k: float(r[f"implied_{k}"]) for k in OUTCOMES}) if h else np.nan
                             for (_, r), h in zip(t.iterrows(), has_key)]
    t["as_of"] = pd.to_datetime(t["as_of"])
    return t


def summary_table(table: pd.DataFrame, include_flagged: bool = True, n_boot: int = 2000) -> pd.DataFrame:
    s = scored(table, include_flagged)
    metrics = ["log_loss_implied", "log_loss_outcome", "brier_outcome"]
    rows = []
    for v in [v for v in ALL_VARIANTS if v in set(s["variant"])]:
        part = s[s["variant"] == v]
        row = {"variant": v, "forecasts": len(part), "model calls (mean)": round(float(part["model_calls"].fillna(0).mean()), 1)}
        for m in metrics:
            row[m] = round(float(part[m].mean()), 4) if part[m].notna().any() else np.nan
        main = "log_loss_implied" if part["log_loss_implied"].notna().any() else "brier_outcome"
        if v != "baseline" and "baseline" in set(s["variant"]):
            try:
                c = compare(s, v, "baseline", metric=main, n_boot=n_boot)
                row[f"vs baseline ({main})"] = round(c.mean_difference, 4)
                row["95% interval"] = f"{c.ci_low:+.4f} to {c.ci_high:+.4f}"
            except ValueError:
                pass
        rows.append(row)
    return pd.DataFrame(rows)


def calibration_table(table: pd.DataFrame, include_flagged: bool = True) -> pd.DataFrame:
    """For each variant and outcome: forecasts grouped by stated probability, and how often the outcome happened."""
    s = scored(table, include_flagged)
    rows = []
    for v in [v for v in ALL_VARIANTS if v in set(s["variant"])]:
        part = s[s["variant"] == v]
        for k in OUTCOMES:
            bins = pd.cut(part[k], CAL_BINS, right=False)
            for b, g in part.groupby(bins, observed=True):
                rows.append({"variant": v, "outcome": k, "stated": f"{int(b.left)}-{min(int(b.right), 100)}%",
                             "forecasts": len(g), "mean stated %": round(float(g[k].mean()), 1),
                             "happened %": round(100 * float((g["outcome"] == k).mean()), 1)})
    return pd.DataFrame(rows)


def write_reports(table: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if table.empty or not table["outcome"].isin(OUTCOMES).any():
        (out_dir / "summary.md").write_text("# Evaluation\n\nNo scorable forecasts yet.\n", encoding="utf-8")
        return
    summ = summary_table(table)
    clean = summary_table(table, include_flagged=False)
    cal = calibration_table(table)
    summ.to_csv(out_dir / "summary.csv", index=False)
    cal.to_csv(out_dir / "calibration.csv", index=False)
    s = table.drop_duplicates(["ticker", "as_of"])
    flagged = int((s["memory_knows"].astype(str) == "True").sum())
    errors = int(s["status"].astype(str).str.startswith("error").sum())
    lines = ["# Evaluation on real stocks", "",
             f"{len(s)} stock-dates ({s['ticker'].nunique()} stocks, {s['as_of'].min()} to {s['as_of'].max()}). "
             f"Errors: {errors}. The model claimed to remember the month after {flagged} of them.", "",
             "Lower is better for every score. Log loss against the implied probabilities is the main score when the "
             "answer key is loaded; otherwise forecasts are scored against what actually happened (noisy).", "",
             "## All forecasts", "", summ.astype(object).fillna("").to_markdown(index=False), ""]
    if flagged:
        lines += ["## Without the dates the model claims to remember", "", clean.astype(object).fillna("").to_markdown(index=False), ""]
    lines += ["## Calibration (how often each outcome happened, by stated probability)", "",
              cal.to_markdown(index=False), ""]
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def with_plain_answers(chat: Chat, memory: str = "I don't know", probs: dict | None = None) -> Chat:
    """Wrap an agent-only chat (the scripted one) so it also answers the evaluator's own prompts."""
    probs = probs or FIXED_GUESS

    def wrapped(messages: list[dict], max_tokens: int = 300) -> str:
        if messages[0]["content"].startswith("Agent:"):
            return chat(messages, max_tokens)
        if "Do you remember" in messages[-1]["content"]:
            return json.dumps({"answer": memory})
        return json.dumps({**probs, "reason": "scripted"})
    return wrapped


# --- command line -------------------------------------------------------------------------------

def model_label(name: str, thinking: bool) -> str:
    """The model name as written in the results, saying whether a Claude 5 model was thinking."""
    from mosaic.llm import NEW_CLAUDE
    return f"{name} (thinking {'on' if thinking else 'off'})" if NEW_CLAUDE.search(name) else name


def make_chat(backend: str, model_name: str | None, thinking: bool | None = None) -> tuple[Chat, str]:
    """One chat function for the whole run (a local model is loaded once)."""
    from mosaic.llm import anthropic_chat, hf_chat, scripted_chat
    if backend == "anthropic":
        name = model_name or SETTINGS.llm_model
        thinking = SETTINGS.llm_thinking if thinking is None else thinking
        return anthropic_chat(name, thinking=thinking), model_label(name, thinking)
    if backend == "hf":                                       # pragma: no cover - needs a GPU
        name = model_name or "Qwen/Qwen2.5-3B-Instruct"
        return hf_chat(name), name
    return with_plain_answers(scripted_chat()), "scripted"


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="mosaic.evaluate", description=__doc__.split("\n\n")[0])
    p.add_argument("--backend", default="anthropic", choices=["anthropic", "hf", "scripted"])
    p.add_argument("--model", default=None)
    p.add_argument("--news-offline", action="store_true",
                   help="use only news already saved on disk; never call GDELT")
    p.add_argument("--thinking", choices=["on", "off"], default=None,
                   help="Claude 5 thinking (default: on, or MOSAIC_LLM_THINKING)")
    p.add_argument("--tickers", default=",".join(DEFAULT_TICKERS), help='comma-separated, or "all" for the 30 stocks')
    p.add_argument("--start", default=CUTOFF)
    p.add_argument("--end", default=None, help="last forecast date (default: today)")
    p.add_argument("--variants", default=",".join(VARIANTS))
    p.add_argument("--memory", default="flag", choices=["flag", "drop", "off"])
    p.add_argument("--resume", action="store_true", help="keep the rows already written and skip them")
    p.add_argument("--limit", type=int, default=None, help="run at most this many stock-dates")
    p.add_argument("--out", default=str(OUT_DIR))
    p.add_argument("--options-dir", default=None,
                   help="folder with FirstRate option files; adds the option-implied answer key to the scores")
    return p.parse_args(argv)


def main(argv=None) -> int:
    a = parse_args(argv)
    tickers = sorted(UNIVERSE) if a.tickers.strip().lower() == "all" else [t.strip().upper() for t in a.tickers.split(",") if t.strip()]
    unknown = [t for t in tickers if t not in UNIVERSE]
    if unknown:
        print(f"Not in the universe: {', '.join(unknown)}")
        return 2
    variants = [v.strip() for v in a.variants.split(",") if v.strip()]
    if bad := [v for v in variants if v not in ALL_VARIANTS]:
        print(f"Unknown variants: {bad}. Choose from: {', '.join(ALL_VARIANTS)}")
        return 2
    dates = forecast_dates(a.start, a.end or datetime.now(UTC).date().isoformat())
    if not dates:
        print("No forecast dates whose 21 trading days have already passed.")
        return 2
    if a.news_offline:
        os.environ["MOSAIC_NEWS_OFFLINE"] = "1"              # read by every GDELT client built for this run
    chat, name = make_chat(a.backend, a.model, None if a.thinking is None else a.thinking == "on")
    print(f"{len(tickers)} stocks x {len(dates)} dates ({dates[0]:%Y-%m-%d} to {dates[-1]:%Y-%m-%d}), model {name}.")
    answer_key, unscorable = None, []
    if a.options_dir:
        from .firstrate import make_answer_key       # evaluation only, never seen by the agents
        answer_key = make_answer_key(a.options_dir, default_price_source, unscorable_log=unscorable)
    table = run(tickers, dates, chat, name, Path(a.out), variants=variants, memory=a.memory, resume=a.resume,
                limit=a.limit, answer_key=answer_key)
    if unscorable:
        pd.DataFrame(unscorable).to_csv(Path(a.out) / "unscorable.csv", index=False)
        print(f"{len(unscorable)} stock-dates had no usable option prices (see unscorable.csv).")
    if table["outcome"].isin(OUTCOMES).any():
        print(summary_table(table).to_string(index=False))
    print(f"Results in {a.out}/ (forecasts.csv, summary.csv, calibration.csv, summary.md)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
