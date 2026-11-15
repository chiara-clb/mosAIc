"""Command-line tool.

    python -m mosaic.cli --demo                       one forecast on the demo company
    python -m mosaic.cli --demo --scenario all        all nine guardrail scenarios, as a table
    python -m mosaic.cli --demo --backend anthropic   the demo company with Claude (needs ANTHROPIC_API_KEY)
    python -m mosaic.cli AAPL --as-of 2026-07-15      a real stock (live data)

Each forecast writes report.md, state.json and trace.jsonl under runs/<ticker>-<date>/.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from mosaic.graph import graph_engine, run_forecast
from mosaic.llm import make_model
from mosaic.report import render_markdown
from mosaic.request import check_request
from mosaic.scenarios import SCENARIOS, run_scenario


def save(state: dict, run, out_dir: Path) -> Path:
    folder = out_dir / run.tools.trace.run_id
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "report.md").write_text(render_markdown(state, run), encoding="utf-8")
    keep = {k: v for k, v in state.items() if k != "tot"}
    keep["tot"] = {k: v for k, v in (state.get("tot") or {}).items() if k != "prepared"}
    (folder / "state.json").write_text(json.dumps(keep, indent=2, default=str), encoding="utf-8")
    run.tools.trace.write_jsonl(folder / "trace.jsonl")
    return folder


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="mosaic", description="One-month fall / flat / gain forecast with a cited ledger.")
    p.add_argument("ticker", nargs="?", help="stock ticker, e.g. AAPL (live data)")
    p.add_argument("--as-of", help="forecast date, YYYY-MM-DD (start of that day)")
    p.add_argument("--demo", action="store_true", help="use the synthetic demo company")
    p.add_argument("--scenario", default="1 normal", help='demo scenario name, or "all"')
    p.add_argument("--backend", default="scripted", choices=["scripted", "anthropic", "hf"])
    p.add_argument("--model", default=None, help="model name for the anthropic or hf backend")
    p.add_argument("--out", default="runs", help="folder for the outputs")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    out = Path(args.out)
    print(f"Graph engine: {graph_engine()[3]}. Model backend: {args.backend}.")
    if args.demo:
        names = list(SCENARIOS) if args.scenario == "all" else [args.scenario]
        unknown = [n for n in names if n not in SCENARIOS]
        if unknown:
            print(f"Unknown scenario {unknown[0]!r}. Choose from: {', '.join(SCENARIOS)} or all.")
            return 2
        factory = None if args.backend == "scripted" else (lambda script: make_model(args.backend, model_name=args.model))
        for name in names:
            state, run, row = run_scenario(name, model_factory=factory)
            folder = save(state, run, out / name.replace(" ", "_").replace(",", ""))
            print(f"{name:<24} {row['status']:<26} review: {row['review']:<8} retries: {row['retries']}  "
                  f"forecast {row['forecast']}  -> {folder}")
        return 0
    if not args.ticker or not args.as_of:
        print("Give a ticker and --as-of, or use --demo.")
        return 2
    as_of = datetime.strptime(args.as_of, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    reasons = check_request(args.ticker.upper(), as_of)
    if reasons:
        print("Request refused: " + "; ".join(reasons))
        return 2
    try:
        from mosaic.live import live_world                # added with the live data sources
    except ImportError:
        print("Live data is not available in this version yet. Use --demo.")
        return 3
    world = live_world(args.ticker.upper(), as_of)
    state, run = run_forecast(world, make_model(args.backend, model_name=args.model))
    print(f"{state['final']['status']}, review: {state['final'].get('review')} -> {save(state, run, out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
