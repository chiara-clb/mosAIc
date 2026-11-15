import pytest

from mosaic.handoffs import handoff
from mosaic.ledger import apply_ledger, direction, floor_and_normalise, shift_for, total_shift

BASE = {"fall": 32.0, "flat": 36.0, "gain": 32.0}


@pytest.mark.parametrize("effect", ["fall", "gain", "spread", "none"])
@pytest.mark.parametrize("size", ["small", "medium", "large"])
def test_each_line_sums_to_zero(effect, size):
    assert abs(sum(shift_for(effect, size).values())) < 1e-9


def test_spread_moves_flat_into_both_tails():
    s = shift_for("spread", "medium")
    assert s["flat"] == -6 and s["fall"] == s["gain"] == 3


def test_cap_scales_all_lines_together():
    out, factor = apply_ledger(BASE, [{"effect": "gain", "size": "large"}, {"effect": "gain", "size": "large"}])
    assert factor == pytest.approx(15 / 18) and out["gain"] == pytest.approx(47.0, abs=0.01)


def test_floor_holds_after_rescaling():
    out = floor_and_normalise({"fall": 0.0, "flat": 50.0, "gain": 0.2})
    assert min(out.values()) >= 1.0 and abs(sum(out.values()) - 100) < 0.05


def test_direction_and_total_shift():
    assert direction([{"effect": "gain", "size": "small"}]) == 1
    assert direction([{"effect": "fall", "size": "small"}]) == -1
    assert direction([{"effect": "spread", "size": "large"}]) == 0
    assert total_shift([{"effect": "spread", "size": "small"}]) == pytest.approx(6.0)


def test_good_handoffs_pass():
    handoff("baseline", {"fall": 30.85, "flat": 38.3, "gain": 30.85})
    handoff("reading", {"effect": "none", "size": None, "quote": "", "reason": ""})
    handoff("line", {"source": "F1", "effect": "spread", "size": "small", "quote": "x", "reason": "", "shift": {}})
    handoff("verdict", {"ok": True, "route": "finalise", "reasons": []})


@pytest.mark.parametrize("kind,obj", [
    ("baseline", {"fall": 50, "flat": 50, "gain": 10}),
    ("baseline", {"fall": 0.5, "flat": 49.5, "gain": 50}),
    ("reading", {"effect": "crash", "size": "small", "quote": "x"}),
    ("reading", {"effect": "fall", "size": "huge", "quote": "x"}),
    ("reading", {"effect": "fall", "size": "small", "quote": " "}),
    ("evidence", {"id": "F1", "kind": "tweet", "header": "h", "published": "d", "extract": "e", "full_text": "t"}),
    ("verdict", {"ok": "yes", "route": "somewhere"}),
])
def test_bad_handoffs_stop_at_the_boundary(kind, obj):
    with pytest.raises(ValueError):
        handoff(kind, obj)
