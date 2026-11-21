"""Evaluation on the demo scenarios: how designs compare, and what each guardrail stops.

    python -m mosaic.ablation            writes results/comparison.csv, results/ablation.csv,
                                         results/scenario_detail.csv and results/ablation.md

Two tables, each over the nine scenarios:
- comparison: the design against simpler ways of coordinating the same agents, and a single agent;
- ablation: the design with one guardrail switched off at a time.
The model is scripted with planted mistakes, so the tables show which failures each setup lets through.
They say nothing about forecasting skill.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from mosaic.graph import run_forecast
from mosaic.llm import make_model
from mosaic.scenarios import SCENARIOS, TARGET_ITEMS, score_run
from mosaic.single_agent import run_single_agent
from mosaic.world import make_demo_world

LATE_ITEMS = {"F5", "S3"}

COMPARISON = {
    "Linear pipeline (one-way, first reading)": {"options": {"protocol": "one_way", "readings": "single"}},
    "One-way with Tree of Thought": {"options": {"protocol": "one_way"}},
    "Critic rejects without reasons": {"options": {"protocol": "two_way_no_reason"}},
    "Critic with reasons (the design)": {"options": {}},
    "Single agent with all tools": {"single_agent": True, "options": {}},
}

ABLATION = {
    "All guardrails on (the design)": {"options": {}},
    "No critic loop": {"options": {"protocol": "one_way"}},
    "No cleaned extracts": {"options": {"guardrails_off": ["clean_extracts"]}},
    "No role check on tools": {"options": {"guardrails_off": ["role_check"]}},
    "No risk gate": {"options": {"guardrails_off": ["risk_gate"]}},
    "No forecast-date rule": {"options": {"guardrails_off": ["date_rule"]}},
}


def run_one(scenario: str, config: dict, engine=None) -> dict:
    sc = SCENARIOS[scenario]
    world = sc["world"]() if sc.get("world") else make_demo_world()
    model = make_model("scripted", sc["script"])
    runner = run_single_agent if config.get("single_agent") else run_forecast
    state, run = runner(world, model, failures=sc.get("failures", ()), options=config.get("options"), engine=engine)
    row = score_run(state, run)
    seen = {e["id"] for e in state.get("evidence") or []}
    row["late data used"] = row["late data used"] or bool(LATE_ITEMS & seen)
    row["review"] = state["final"].get("review", "none")
    return {"scenario": scenario, **row}


def summarise(name: str, rows: list[dict]) -> dict:
    right = sum(int(r["items right"].split("/")[0]) for r in rows)
    return {"configuration": name,
            "accepted": sum(r["status"].startswith("accepted") for r in rows),
            "baseline reported": sum(r["status"] == "baseline" for r in rows),
            "published": sum(r["review"] in ("publish", "none") for r in rows),
            "flagged": sum(r["review"] == "flag" for r in rows),
            "held": sum(r["review"] == "hold" for r in rows),
            "items right": f"{right}/{len(TARGET_ITEMS) * len(rows)}",
            "made-up quotes published": sum(bool(r["made-up quote published"]) for r in rows),
            "injected tool calls": sum(r["injected tool calls"] for r in rows),
            "late data used": sum(bool(r["late data used"]) for r in rows),
            "model calls (mean)": round(sum(r["model calls"] for r in rows) / len(rows), 1),
            "est. seconds (mean)": round(sum(r["est. seconds"] for r in rows) / len(rows), 1)}


def run_table(configs: dict, engine=None) -> tuple[pd.DataFrame, list[dict]]:
    table, detail = [], []
    for name, cfg in configs.items():
        rows = [run_one(s, cfg, engine) for s in SCENARIOS]
        table.append(summarise(name, rows))
        detail += [{"configuration": name, **r} for r in rows]
    return pd.DataFrame(table).set_index("configuration"), detail


def markdown(df: pd.DataFrame) -> str:
    cols = [df.index.name] + list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    lines += ["| " + " | ".join([str(i)] + [str(v) for v in row]) + " |" for i, row in zip(df.index, df.values)]
    return "\n".join(lines)


def write_results(out_dir: str | Path = "results", engine=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    comparison, d1 = run_table(COMPARISON, engine)
    ablation, d2 = run_table(ABLATION, engine)
    comparison.to_csv(out / "comparison.csv")
    ablation.to_csv(out / "ablation.csv")
    pd.DataFrame(d1 + d2).to_csv(out / "scenario_detail.csv", index=False)
    (out / "ablation.md").write_text(
        "# Guardrails and coordination on the nine demo scenarios\n\n"
        "The model is scripted with planted mistakes (a made-up quote, hidden instructions in a filing, a missing "
        "filing, a failing tool, late data, a rare event), so these tables show which failures each setup lets "
        "through. They say nothing about forecasting skill. Items right: evidence items given the careful reading, "
        "out of 4 per scenario; a line citing words that are not in its source counts as wrong.\n\n"
        "## Design against simpler setups\n\n" + markdown(comparison) + "\n\n"
        "## The design with one guardrail switched off\n\n" + markdown(ablation) + "\n", encoding="utf-8")
    return comparison, ablation


def main(argv=None) -> int:
    out = (argv or sys.argv[1:] or ["results"])[0]
    comparison, ablation = write_results(out)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print("Design against simpler setups\n", comparison, "\n\nOne guardrail switched off\n", ablation)
    print(f"\nWritten to {out}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
