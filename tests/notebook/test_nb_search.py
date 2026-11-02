import json

import numpy as np
import pytest

from tests.notebook.core import load_core
from tests.notebook.helpers import chunk, dt, unit

ns = load_core()
BASE = {"fall": 30.0, "flat": 40.0, "gain": 30.0}
AS_OF = dt("2026-08-01")


def reading(effect, size="medium", score=0.5, ok=True, why="", src="s"):
    return {"effect": effect, "size": size, "reason": "r", "source": src, "ok": ok, "why": why,
            "score": score, "base_rate": score, "novelty": score}


def entry(*readings, vec=None, i=0):
    item = {"id": f"i{i}", "vec": vec if vec is not None else unit(*([0] * i + [1] + [0] * (6 - i)))}
    return {"item": item, "readings": list(readings)}


def tree(*per_item):
    """per_item: list of (effect, score) for each item; 'none' is added with score 0.5 if missing."""
    out = []
    for i, rs in enumerate(per_item):
        readings = [reading(e, score=s, src=f"i{i}") for e, s in rs]
        if not any(e == "none" for e, _ in rs):
            readings.append(reading("none", "small", 0.5, src=f"i{i}"))
        out.append(entry(*readings, i=i))
    return out


# ---------- scoring a node ----------

def test_node_score_is_weighted_average_with_consistency():
    p = tree([("fall", 0.9)])
    s = ns["node_score"](p, (0,), BASE)
    assert s == pytest.approx(0.8 * 0.9 + 0.2 * 1.0)


def test_same_story_counted_twice_is_penalised():
    v = unit(1, 0, 0)
    p = [entry(reading("fall", score=0.8), reading("none", "small"), vec=v, i=0),
         entry(reading("fall", score=0.8), reading("none", "small"), vec=v, i=1)]
    assert ns["consistency"](p, (0, 0), BASE) == pytest.approx(0.5)
    p[1]["item"]["vec"] = unit(0, 1, 0)
    assert ns["consistency"](p, (0, 0), BASE) == pytest.approx(1.0)


def test_cap_lowers_consistency():
    p = tree([("gain", 0.6)], [("gain", 0.6)], [("gain", 0.6)])
    for e in p:
        e["readings"][0]["size"] = "large"            # 27 points on gain, capped at 15
    assert ns["consistency"](p, (0, 0, 0), BASE) == pytest.approx(15 / 27)


# ---------- beam search ----------

def test_picks_the_best_supported_ledger():
    p = tree([("fall", 0.9), ("gain", 0.2)], [("spread", 0.7), ("fall", 0.3)])
    res = ns["beam_search"](p, BASE)
    rs = ns["readings_of"](p, res["choice"])
    assert [r["effect"] for r in rs] == ["fall", "spread"]
    assert res["final"]["fall"] > BASE["fall"]


def test_no_effect_wins_when_evidence_is_weak():
    p = tree([("fall", 0.3), ("spread", 0.35)])
    res = ns["beam_search"](p, BASE)
    assert res["ledger"] == [] and res["final"] == pytest.approx(BASE)


def test_beam_keeps_at_most_width_nodes_per_level():
    p = tree([("fall", 0.9), ("gain", 0.8)], [("fall", 0.9), ("gain", 0.8)], [("fall", 0.9), ("gain", 0.8)])
    res = ns["beam_search"](p, BASE, prune_margin=None)
    for d in (1, 2, 3):
        kept = [n for n in res["nodes"] if n["depth"] == d and n["status"] in ("kept", "chosen")]
        assert len(kept) <= 3


def test_all_no_effect_node_is_always_kept():
    p = tree([("fall", 0.95), ("gain", 0.9)], [("fall", 0.95), ("gain", 0.9)], [("fall", 0.95), ("gain", 0.9)])
    res = ns["beam_search"](p, BASE, prune_margin=None)
    none_idx = [2, 2, 2]
    for d in (1, 2, 3):
        node = next(n for n in res["nodes"] if n["choice"] == tuple(none_idx[:d]))
        assert node["status"] in ("kept", "chosen"), node


def test_beam_1_is_a_real_greedy_search_even_with_the_rule_on():
    p = tree([("fall", 0.95)], [("fall", 0.95)])
    res = ns["beam_search"](p, BASE, beam_width=1, keep_no_effect=True)
    assert [r["effect"] for r in res["ledger"]] == ["fall", "fall"]


def test_without_the_rule_greedy_beam_1_can_drop_it():
    p = tree([("fall", 0.95)], [("fall", 0.95)])
    res = ns["beam_search"](p, BASE, beam_width=1, keep_no_effect=False)
    assert all(n["choice"] != (1,) or n["status"] == "pruned" for n in res["nodes"])


def test_margin_drops_clearly_worse_nodes():
    p = tree([("fall", 1.0), ("gain", 0.1)])
    res = ns["beam_search"](p, BASE, keep_no_effect=False)
    gain_node = next(n for n in res["nodes"] if n["choice"] == (1,))
    assert gain_node["status"] == "pruned" and "below the best" in gain_node["why"]


def test_failed_readings_are_pruned_with_reason():
    p = tree([("fall", 0.99)])
    p[0]["readings"][0].update(ok=False, why="failed grounding")
    res = ns["beam_search"](p, BASE)
    node = next(n for n in res["nodes"] if n["choice"] == (0,))
    assert node["status"] == "pruned" and node["why"] == "failed grounding"
    assert res["ledger"] == []


def test_ties_go_to_the_smaller_shift():
    p = [entry(reading("fall", "large", 0.6), reading("fall", "small", 0.6), reading("none", "small", 0.2), i=0)]
    p[0]["readings"][1]["effect"] = "gain"            # same score, smaller size
    res = ns["beam_search"](p, BASE, keep_no_effect=False)
    assert ns["readings_of"](p, res["choice"])[0]["size"] == "small"


def test_every_node_is_logged_including_pruned(tmp_path):
    p = tree([("fall", 0.9), ("gain", 0.1)], [("spread", 0.6)])
    lines = []
    res = ns["beam_search"](p, BASE, log=lines.append)
    assert len(lines) == len(res["nodes"]) == 1 + 3 + 2 * 2       # root, 3 children, then 2 kept parents x 2
    rec = json.loads(json.dumps(lines[-1], default=str))
    assert {"id", "parent", "depth", "score", "status", "readings"} <= set(rec)
    assert any(l["status"] == "pruned" for l in lines)


def test_no_evidence_returns_the_baseline():
    res = ns["beam_search"]([], BASE)
    assert res["final"] == pytest.approx(BASE) and res["ledger"] == []


# ---------- comparisons ----------

def test_single_pass_takes_the_first_reading_that_passes():
    p = tree([("fall", 0.1), ("gain", 0.9)], [("spread", 0.2)])
    sp = ns["single_pass"](p, BASE)
    assert [r["effect"] for r in sp["ledger"]] == ["fall", "spread"]
    p[0]["readings"][0].update(ok=False)
    sp = ns["single_pass"](p, BASE)
    assert [r["effect"] for r in sp["ledger"]] == ["spread"]


def test_exhaustive_agrees_with_a_wide_beam():
    rng = np.random.default_rng(3)
    p = tree(*[[("fall", float(rng.uniform())), ("gain", float(rng.uniform()))] for _ in range(4)])
    best = ns["exhaustive"](p, BASE)[0]
    wide = ns["beam_search"](p, BASE, beam_width=100, prune_margin=None)
    assert wide["choice"] == best["choice"]
    assert len(ns["exhaustive"](p, BASE)) == 3 ** 4


def test_prompt_only_parses_or_falls_back():
    m = ns["Model"](lambda msgs, n: 'Sure: {"fall": 20, "flat": 50, "gain": 30}')
    probs, ok = ns["prompt_only"](m, "Apple", "AAPL", AS_OF)
    assert ok and probs == {"fall": 20, "flat": 50, "gain": 30}
    m = ns["Model"](lambda msgs, n: "I cannot predict markets.")
    probs, ok = ns["prompt_only"](m, "Apple", "AAPL", AS_OF)
    assert not ok and sum(probs.values()) == pytest.approx(100)


# ---------- preparing items with a stand-in model ----------

class FlatCase:
    def score(self, effect, size, event, lag, as_of, p0=None):
        return (0.5, {"n_events": None}) if effect == "none" else (0.7, {"n_events": 30})


def scripted_model(gen_reply, ground_reply="yes"):
    def chat(messages, n):
        return gen_reply if "TWO most plausible" in messages[0]["content"] else ground_reply
    return ns["Model"](chat)


def one_item_index(accepted="2026-07-31"):
    c = dict(chunk("AAPL-10-Q-1", accepted=accepted, text="Apple faces a new antitrust lawsuit."), row=0)
    c["section_key"] = "legal proceedings"
    return ns["Index"]([c], np.array([unit(1, 0)], dtype=np.float32)), c


def test_prepare_scores_readings_and_counts_calls():
    idx, item = one_item_index()
    gen = json.dumps([{"effect": "spread", "size": "small", "reason": "new antitrust lawsuit"},
                      {"effect": "fall", "size": "medium", "reason": "lawsuit"}])
    m = scripted_model(gen)
    prep = ns["prepare_items"](m, [item], AS_OF, FlatCase(), idx)
    rs = prep[0]["readings"]
    assert [r["effect"] for r in rs] == ["spread", "fall", "none"]
    assert all(r["ok"] for r in rs)
    assert m.calls == 3                                    # 1 generation + 2 grounding
    assert rs[0]["score"] == pytest.approx(0.5 * 0.7 + 0.5 * 1.0)   # novelty 1: no previous filing
    assert rs[2]["score"] == pytest.approx(0.5 * 0.5 + 0.5 * 0.0)


def test_prepare_applies_grounding_and_date_rules():
    idx, item = one_item_index()
    gen = json.dumps([{"effect": "fall", "size": "medium", "reason": "x"}])
    prep = ns["prepare_items"](scripted_model(gen, "no"), [item], AS_OF, FlatCase(), idx)
    assert prep[0]["readings"][0]["why"] == "failed grounding"
    idx, old = one_item_index(accepted="2026-05-01")
    prep = ns["prepare_items"](scripted_model(gen), [old], AS_OF, FlatCase(), idx)
    assert prep[0]["readings"][0]["why"] == "date rule"


def test_budget_never_exceeded():
    ns2 = load_core(MAX_MODEL_CALLS=4)
    idx = ns2["Index"]([], np.zeros((0, 2), dtype=np.float32))
    items = []
    for i in range(5):
        c = dict(chunk(f"c{i}", filing=f"F{i}", accepted="2026-07-30"), row=i)
        c["section_key"] = "x"
        items.append(c)
    idx = ns2["Index"](items, np.array([unit(1, i) for i in range(5)], dtype=np.float32))
    gen = json.dumps([{"effect": "fall", "size": "small", "reason": "x"}, {"effect": "gain", "size": "small", "reason": "y"}])
    m = ns2["Model"](lambda msgs, n: gen if "TWO most plausible" in msgs[0]["content"] else "yes")
    prep = ns2["prepare_items"](m, items, AS_OF, FlatCase(), idx)
    assert m.calls <= 4
    assert prep[-1]["raw"] == "budget used up"
    assert len(prep) == 5
