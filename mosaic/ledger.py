"""Ledger arithmetic: how a line shifts the probabilities, the cap, and the 1% floor."""

from __future__ import annotations

from mosaic.config import OUTCOMES, SETTINGS

EFFECTS = ("fall", "gain", "spread", "none")


def shift_for(effect: str, size: str | None) -> dict[str, float]:
    """Points moved by one line. Every line sums to zero; "spread" moves probability from flat to both tails."""
    if effect == "none":
        return {"fall": 0.0, "flat": 0.0, "gain": 0.0}
    s = float(SETTINGS.strength_points[size])
    return {"fall": {"fall": s, "flat": -s / 3, "gain": -2 * s / 3},
            "gain": {"fall": -2 * s / 3, "flat": -s / 3, "gain": s},
            "spread": {"fall": s / 2, "flat": -s, "gain": s / 2}}[effect]


def floor_and_normalise(p: dict[str, float]) -> dict[str, float]:
    """Scale to 100 while keeping every outcome at the floor or above (the floor holds after rescaling)."""
    floor = SETTINGS.min_probability
    floored: set[str] = set()
    while True:
        free = [k for k in OUTCOMES if k not in floored]
        room = 100.0 - floor * len(floored)
        s = sum(p[k] for k in free)
        out = {k: (floor if k in floored else (room * p[k] / s if s > 0 else room / len(free))) for k in OUTCOMES}
        low = [k for k in free if out[k] < floor]
        if not low:
            return {k: round(v, 2) for k, v in out.items()}
        floored.update(low)


def apply_ledger(base: dict[str, float], readings: list[dict]) -> tuple[dict[str, float], float]:
    """Baseline plus the readings' shifts, scaled down together if they pass the cap. Returns (forecast, scale)."""
    total = {k: sum(shift_for(r["effect"], r["size"])[k] for r in readings) for k in OUTCOMES}
    biggest = max([abs(v) for v in total.values()] + [0.0])
    factor = SETTINGS.max_total_shift / biggest if biggest > SETTINGS.max_total_shift else 1.0
    return floor_and_normalise({k: max(base[k] + factor * total[k], 0.0) for k in OUTCOMES}), factor


def total_shift(readings: list[dict]) -> float:
    return sum(abs(v) for v in {k: sum(shift_for(r["effect"], r["size"])[k] for r in readings) for k in OUTCOMES}.values())


def direction(readings: list[dict]) -> int:
    """+1 if the readings lean toward a gain, -1 toward a fall, 0 if neither."""
    net = sum(shift_for(r["effect"], r["size"])["gain"] - shift_for(r["effect"], r["size"])["fall"] for r in readings)
    return (net > 1e-9) - (net < -1e-9)
