import json

import numpy as np
import pytest

from tests.notebook.core import load_baseline_core
from tests.notebook.test_bl_versions import make_stock

ns = load_baseline_core(N_BOOT=100)


@pytest.mark.parametrize("reply,expected", [
    ('{"fall": 20, "flat": 50, "gain": 30}', {"fall": 20, "flat": 50, "gain": 30}),
    ('```json\n{"fall": 1, "flat": 2, "gain": 1}\n```', {"fall": 25, "flat": 50, "gain": 25}),
    ("I can't predict that.", None),
    ('{"fall": 0, "flat": 0, "gain": 0}', None),
    ('{"fall": "a lot"}', None),
])
def test_parse_probs(reply, expected):
    got = ns["parse_probs"](reply)
    assert got == (pytest.approx(expected) if expected else None)


@pytest.fixture(scope="module")
def po_run():
    stocks = []
    for seed in range(2):
        s, idx, _ = make_stock(n=1400, seed=seed)
        s.ticker = f"S{seed}"
        stocks.append(s)
    calm = ns["CalmProfile"](stocks)
    dates = ns["weekly_dates"](str(idx[400].date()), str(idx[-30].date()))
    bt = ns["run_backtest"](stocks, calm, dates)
    prompts = []

    def chat(messages, max_tokens):
        prompts.append(messages[1]["content"])
        return "no idea" if len(prompts) % 10 == 0 else json.dumps({"fall": 25, "flat": 45, "gain": 30})
    po, unreadable = ns["prompt_only_rows"](stocks, dates[::2], chat, {"S0": "Stock Zero Inc"})
    return bt, po, unreadable, prompts


def test_prompt_only_rows(po_run):
    bt, po, unreadable, prompts = po_run
    assert len(po) == len(prompts) and unreadable == len(prompts) // 10
    assert "Stock Zero Inc (S0)" in prompts[0] and "Forecast date:" in prompts[0]
    assert set(po["version"]) == {"Prompt only"}
    assert np.allclose(po["fall"] + po["flat"] + po["gain"], 100)


def test_comparison_uses_the_same_forecasts(po_run):
    bt, po, _, _ = po_run
    both, avg = ns["with_prompt_only"](bt, po)
    per = both.groupby(["ticker", "as_of"])["version"].nunique()
    assert (per == len(ns["VERSIONS"]) + 2).all()
    assert len(per) == len(po)
    assert avg["flat"] == pytest.approx(po["flat"].mean())
    t = ns["compare_versions"](both)
    assert {"Prompt only", "Prompt-only average"} <= set(t.index)


def test_a_constant_model_scores_like_its_own_average(po_run):
    bt, po, _, _ = po_run
    # without unreadable replies, a model that always answers the same is exactly its average
    po2 = po.copy()
    for k, v in {"fall": 25, "flat": 45, "gain": 30}.items():
        po2[k] = float(v)
    po2["brier"] = [ns["brier"]({"fall": 25, "flat": 45, "gain": 30}, o) for o in po2["outcome"]]
    both, _ = ns["with_prompt_only"](bt, po2)
    m = both.groupby("version")["brier"].mean()
    assert m["Prompt only"] == pytest.approx(m["Prompt-only average"])
