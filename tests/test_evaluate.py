"""The evaluation runner, fully offline: fake worlds, the scripted model, fake prices and a fake answer key."""

import json
import math
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from mosaic import evaluate as E
from mosaic.llm import Model, scripted_chat
from mosaic.world import make_demo_world

UTC = timezone.utc


def fake_world(ticker, as_of):
    return make_demo_world()        # its own stock, date and documents; the runner only needs a world to forecast on


def fake_prices(ticker, as_of):
    """A deterministic random walk from 400 days before to 60 days after the date."""
    days = pd.bdate_range(as_of.date() - timedelta(days=400), as_of.date() + timedelta(days=60))
    rng = np.random.default_rng(abs(hash((ticker, as_of.date().isoformat()))) % 2**32)
    return pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.015, len(days)))), index=days)


def fake_key(ticker, as_of):
    return {"fall": 28.0, "flat": 42.0, "gain": 30.0}


def chat(memory="I don't know", probs=None):
    return E.with_plain_answers(scripted_chat(), memory=memory, probs=probs or {"fall": 20, "flat": 30, "gain": 50})


DATES = [datetime(2026, 7, 1, tzinfo=UTC), datetime(2026, 7, 8, tzinfo=UTC)]


# --- dates ---

def test_forecast_dates_are_wednesdays_whose_month_has_passed():
    d = E.forecast_dates("2026-07-01", "2026-10-15", today=date(2026, 9, 1))
    assert all(x.weekday() == 2 for x in d)
    assert d[0].date() == date(2026, 7, 1)
    assert d[-1].date() == date(2026, 7, 29)     # 21 business days later is Aug 27, before Sep 1; Aug 5 + 21 = Sep 3
    assert E.forecast_dates("2026-07-02", "2026-07-07", today=date(2027, 1, 1)) == []


# --- the model-only variants ---

def test_read_probs_normalises_and_rejects_bad_replies():
    p = E.read_probs('{"fall": 20, "flat": 20, "gain": 60}')
    assert p == pytest.approx({"fall": 20, "flat": 20, "gain": 60}, abs=0.01)
    assert E.read_probs("no idea") is None
    assert E.read_probs('{"fall": "x"}') is None


def test_memory_check_reads_the_answer():
    for answer, knows in [("rose a lot", True), ("I don't know", False)]:
        m = Model(chat(memory=answer), 3)
        assert E.memory_check(m, "Apple", "AAPL", DATES[0]) == (answer, knows)
    assert E.memory_check(Model(lambda m, t: "hmm", 3), "Apple", "AAPL", DATES[0]) == ("I don't know", False)


def test_prompt_only_falls_back_to_the_fixed_guess():
    p, status = E.prompt_only(Model(lambda m, t: "not json", 3), "Apple", "AAPL", DATES[0])
    assert p == E.FIXED_GUESS and "fixed guess" in status


def test_baseline_plus_memory_is_capped():
    base = {"fall": 30.0, "flat": 40.0, "gain": 30.0}
    reply = json.dumps({"fall": 5, "flat": 15, "gain": 80})          # gain +50: scaled down to +15
    p, _ = E.baseline_plus_memory(Model(lambda m, t: reply, 3), "Apple", "AAPL", DATES[0], base)
    assert max(abs(p[k] - base[k]) for k in base) <= 15.0 + 0.5
    assert p["gain"] > base["gain"]
    p, status = E.baseline_plus_memory(Model(lambda m, t: "??", 3), "Apple", "AAPL", DATES[0], base)
    assert p == base and "baseline used" in status


# --- one stock and date ---

def test_evaluate_one_gives_every_variant_on_the_same_world():
    rows = E.evaluate_one("AAPL", DATES[0], chat(), "scripted", world_factory=fake_world,
                          price_source=fake_prices, answer_key=fake_key)
    assert [r["variant"] for r in rows] == E.VARIANTS
    assert all(r["error"] == "" for r in rows)
    assert {r["outcome"] for r in rows} <= {"fall", "flat", "gain"} and len({r["outcome"] for r in rows}) == 1
    for r in rows:
        assert sum(r[k] for k in ("fall", "flat", "gain")) == pytest.approx(100, abs=0.1)
        assert r["implied_flat"] == 42.0
    by = {r["variant"]: r for r in rows}
    assert by["single pass"]["model_calls"] > 0 and by["ToT"]["model_calls"] > 0
    assert by["prompt only"]["gain"] == pytest.approx(50, abs=0.1)


def test_memory_drop_and_flag():
    dropped = E.evaluate_one("AAPL", DATES[0], chat(memory="fell a lot"), "scripted", world_factory=fake_world,
                             price_source=fake_prices, memory="drop")
    assert all(r["status"].startswith("dropped") for r in dropped)
    flagged = E.evaluate_one("AAPL", DATES[0], chat(memory="fell a lot"), "scripted", world_factory=fake_world,
                             price_source=fake_prices, memory="flag")
    assert all(r["memory_knows"] is True and "fall" in r for r in flagged)


def test_a_failing_world_gives_error_rows_not_a_crash():
    def broken(t, d):
        raise RuntimeError("SEC down")
    rows = E.evaluate_one("AAPL", DATES[0], chat(), "scripted", world_factory=broken, price_source=fake_prices)
    assert all(r["status"] == "error" and "SEC down" in r["error"] for r in rows)


# --- the whole run ---

def run(tmp_path, **kw):
    logs = []
    t = E.run(["AAPL", "NKE"], DATES, chat(), "scripted", tmp_path, world_factory=fake_world,
              price_source=fake_prices, answer_key=fake_key, log=logs.append, **kw)
    return t, logs


def test_run_writes_all_outputs_and_resumes(tmp_path):
    t, _ = run(tmp_path, limit=3)
    assert len(t) == 3 * len(E.VARIANTS)
    for f in ("forecasts.csv", "summary.csv", "calibration.csv", "summary.md"):
        assert (tmp_path / f).exists()
    t, logs = run(tmp_path, resume=True)
    assert "3 stock-dates already done, 1 to run." in logs[0]
    assert len(t) == 4 * len(E.VARIANTS)
    assert not t.duplicated(["ticker", "as_of", "variant"]).any()
    t, logs = run(tmp_path, resume=False, limit=1)
    assert len(t) == len(E.VARIANTS)


def test_scores_match_hand_values(tmp_path):
    table = pd.DataFrame([
        {"ticker": "AAPL", "as_of": "2026-07-01", "variant": "baseline", "fall": 20, "flat": 30, "gain": 50,
         "outcome": "gain", "memory_knows": False, "model_calls": 0, "implied_fall": 25, "implied_flat": 50, "implied_gain": 25},
    ])
    s = E.scored(table).iloc[0]
    assert s["brier_outcome"] == pytest.approx(0.2**2 + 0.3**2 + 0.5**2)
    assert s["log_loss_outcome"] == pytest.approx(-math.log(0.5))
    assert s["log_loss_implied"] == pytest.approx(-(0.25 * math.log(0.2) + 0.5 * math.log(0.3) + 0.25 * math.log(0.5)))


def test_summary_and_calibration(tmp_path):
    t, _ = run(tmp_path)
    summ = pd.read_csv(tmp_path / "summary.csv")
    assert list(summ["variant"]) == E.VARIANTS
    assert (summ["forecasts"] == 4).all()
    assert "vs baseline (log_loss_implied)" in summ.columns
    cal = pd.read_csv(tmp_path / "calibration.csv")
    for (v, k), g in cal.groupby(["variant", "outcome"]):
        assert g["forecasts"].sum() == 4
    md = (tmp_path / "summary.md").read_text()
    assert "# Evaluation on real stocks" in md and "| prompt only" in md


def test_flagged_dates_get_a_separate_table(tmp_path):
    E.run(["AAPL"], DATES, chat(memory="rose a lot"), "scripted", tmp_path, world_factory=fake_world,
          price_source=fake_prices, log=lambda *_: None)
    md = (tmp_path / "summary.md").read_text()
    assert "claimed to remember the month after 2 of them" in md
    assert "Without the dates the model claims to remember" in md


# --- command line ---

def test_cli_checks_its_arguments(capsys):
    assert E.main(["--backend", "scripted", "--tickers", "ZZZZ"]) == 2
    assert E.main(["--backend", "scripted", "--variants", "magic"]) == 2
    assert E.main(["--backend", "scripted", "--start", "2030-01-01", "--end", "2030-02-01"]) == 2   # no date has passed
    args = E.parse_args(["--tickers", "all"])
    assert args.tickers == "all"


def test_options_dir_adds_the_answer_key(monkeypatch, tmp_path):
    seen = {}

    def fake_run(*args, answer_key=None, **kw):
        seen["key"] = answer_key
        return pd.DataFrame({"outcome": []})

    monkeypatch.setattr(E, "run", fake_run)
    args = ["--backend", "scripted", "--tickers", "AAPL", "--start", "2026-07-01", "--end", "2026-07-10",
            "--out", str(tmp_path)]
    assert E.main(args) == 0
    assert seen["key"] is None
    assert E.main(args + ["--options-dir", str(tmp_path)]) == 0
    assert callable(seen["key"])


def test_the_results_say_whether_claude_was_thinking():
    from mosaic.evaluate import model_label, parse_args
    assert model_label("claude-haiku-5-5", True) == "claude-haiku-5-5 (thinking on)"
    assert model_label("claude-haiku-5-5", False) == "claude-haiku-5-5 (thinking off)"
    assert model_label("Qwen/Qwen2.5-3B-Instruct", True) == "Qwen/Qwen2.5-3B-Instruct"
    assert parse_args(["--thinking", "off"]).thinking == "off" and parse_args([]).thinking is None
    assert parse_args(["--news-offline"]).news_offline and not parse_args([]).news_offline
