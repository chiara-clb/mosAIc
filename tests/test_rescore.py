"""Rescoring saved Tree of Thought readings: the replay must match the run, and each scorer must do what it says."""

import pytest

from mosaic import rescore as R
from tests.test_evaluate import run as evaluate_run


def saved(tmp_path):
    evaluate_run(tmp_path)
    return R.load(tmp_path / "readings.jsonl")


def test_the_current_scorer_reproduces_the_run(tmp_path):
    recs = saved(tmp_path)
    assert recs
    for rec in recs:
        again = R.rerun(rec, R.current)
        assert all(again[k] == pytest.approx(rec["probabilities"][k], abs=0.01) for k in again)


def item(bucket, readings, novelty=0.8):
    return {"id": "F1", "event": "8-K 2.02", "lag_days": 3, "novelty": novelty, "reaction_bucket": bucket,
            "readings": readings}


def reading(effect, size="medium", score=0.5, conf=None, ok=True):
    return {"effect": effect, "size": size, "ok": ok, "score": score, "confidence": conf, "base_rate": 0.5,
            "novelty": 0.5, "n": 40}


def record(items, as_of="2026-07-08", status="accepted"):
    base = {"fall": 30.0, "flat": 40.0, "gain": 30.0}
    return {"ticker": "ACME", "as_of": as_of, "outcome": "gain", "implied": {"fall": 25.0, "flat": 40.0, "gain": 35.0},
            "baseline": base, "probabilities": base, "status": status, "choice": [], "items": items}


def test_the_reaction_scorer_follows_the_price_reaction():
    rec = record([item("big rise", [reading("fall"), reading("gain"), reading("none", size=None)])])
    out = R.rerun(rec, R.reaction)
    assert out["gain"] > rec["baseline"]["gain"] and out["fall"] < rec["baseline"]["fall"]
    rec = record([item("small", [reading("fall"), reading("gain"), reading("none", size=None)])])
    assert R.rerun(rec, R.reaction) == pytest.approx(rec["baseline"])


def test_the_confidence_scorer_picks_the_reading_the_model_believes():
    rec = record([item(None, [reading("fall", conf=0.9), reading("gain", conf=0.2), reading("none", size=None)])])
    assert R.rerun(rec, R.confidence)["fall"] > rec["baseline"]["fall"]


def test_a_reading_that_failed_its_check_is_never_chosen():
    rec = record([item("big rise", [reading("gain", ok=False), reading("fall"), reading("none", size=None)])])
    assert R.rerun(rec, R.reaction)["gain"] <= rec["baseline"]["gain"]


def test_a_run_that_ended_in_the_baseline_stays_there():
    rec = record([item("big rise", [reading("gain"), reading("none", size=None)])], status="baseline")
    assert R.rerun(rec, R.reaction) == rec["baseline"]


def test_pooled_scorers_use_the_given_base_rate_and_keep_novelty():
    scorer_set = R.scorers(pooled_base_rate=lambda rec, it, r: 1.0 if r["effect"] == "gain" else 0.0)
    assert {"pooled", "pooled + reaction", "pooled + reaction + confidence"} <= set(scorer_set)
    it = item(None, [])
    assert scorer_set["pooled"]({}, it, reading("gain")) == pytest.approx(0.5 * 1.0 + 0.5 * 0.8)
    assert scorer_set["pooled"]({}, it, reading("none", size=None)) == pytest.approx(0.5 * 0.0 + 0.5 * 0.2)


def test_the_summary_splits_july_from_august_and_lists_every_method():
    recs = [record([item("big rise", [reading("gain"), reading("none", size=None)])], as_of=d)
            for d in ("2026-07-08", "2026-08-12")]
    table = R.rescore(recs, R.scorers())
    s = R.summary(table)
    assert set(s["half"]) == {"all", "choose (July)", "check (August)"}
    assert set(s["method"]) == {"baseline", "ToT as run", "ToT, current", "ToT, confidence", "ToT, reaction",
                                "ToT, current + confidence"}
    assert (s[s["method"] == "baseline"]["vs baseline"] == 0).all()
