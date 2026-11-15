# MosAIc build plan

Deliver the final capstone (Assignment 7.1) by **23 November, 17:29**: a 1,000–1,500 word report in the course template, an 8–10 minute video (8–10 slides) and a public GitHub repository. The video is recorded on 22 November, so the system is frozen on 20 November.

The system is the design of Checkpoints 4 to 6 (any forecast date, 21 trading days ahead, scored against option-implied probabilities). The earnings-focused variation stays out of the build and is presented as the next step.

Working rules: a plan before code, one small commit per feature with its tests, nothing pushed without asking first.

## Where we are

- **Package (`mosaic/`):** settings and the 30-stock universe, buckets, answer key from option chains, scoring. Tested.
- **Notebooks, tested through their core cells:**
  - `tot_validation`: filings, search, prices, results dates, case memory with the price reaction, GDELT news, Tree of Thought.
  - `baseline_comparison`: baseline versions A to D.
  - `multi_agent`: the five agents, tools behind `mcp_call`, LangGraph flow, critic, retries and fallback, risk gate, analyst review, nine scenarios.
- **Write-ups:** Checkpoints 4, 5 and 6, design review, presentation notes.

Most of what is left is moving tested notebook code into the package, plus the option-data loader, the evaluation run, the README and the deliverables.

## Two ways to run it

- **Demo mode (no API key):** the synthetic Acme world and the scripted model. Runs in seconds and shows every guardrail. This is what a reviewer runs first.
- **Live mode:** real SEC filings, GDELT news and prices, with Claude through the Anthropic API.

## Schedule

### A. One working system end to end (15–17 Nov)

| # | Commit | Tests |
| --- | --- | --- |
| 1 | `trace.py`: run trace and leakage audit (nothing dated after the forecast date) | A planted late source is caught |
| 2 | `world.py` + `tools.py`: the synthetic demo world and the tools behind `mcp_call` (role check, forecast-date rule, cleaned extracts, failure switch) | Ported from the notebook tests |
| 3 | `llm.py`: `Model` with call budget, JSON parsing, scripted model, Anthropic and Hugging Face backends | Budget, parsing, scripted replies; the API backend with a fake client |
| 4 | `agents.py`: coordinator, quant, research (with guards), synthesis (ToT), critic, finalise, fallback | Ported agent tests |
| 5 | `risk.py`: risk gate, analyst review, review log; a refused tool call (an attempt to act on something the model read) is a trigger, so it never goes out unflagged (Lab 6.2) | Ported review tests; scenario 7 is flagged |
| 6 | `graph.py`: the LangGraph flow; `run_forecast` | Ported graph and scenario tests (offline stand-in) |
| 6b | `request.py`: request check before any run (ticker in the universe, date not in the future, a year of prices, sources reachable); a failed check refuses the run with the reason (Lab 6.1, pre-generation) | Each refusal reason; a valid request passes |
| 7 | `cli.py`: `python -m mosaic.cli --demo` and `python -m mosaic.cli AAPL --as-of 2026-07-15`, writing the forecast, ledger and trace as Markdown and JSON | CLI on the demo world |

### B. Live data (17–18 Nov)

| # | Commit | Tests |
| --- | --- | --- |
| 8 | `edgar.py` + `index.py`: filings, sections, chunks, embeddings, search, filing candidates (from the ToT notebook) | Fixture filings; no late chunks; boilerplate filter |
| 9 | `prices.py` + `baseline.py`: prices, results dates, baseline C | Ported baseline tests |
| 10 | `case_memory.py` + `news.py`: base rates with the price reaction, GDELT stories, same-story rule | Ported tests |
| 11 | `live.py`: the live world behind the same tools, with caching | Fakes for each source |

### C. Evaluation (18 Nov)

| # | Commit | Tests |
| --- | --- | --- |
| 12 | `firstrate.py`: loader for the FirstRate files into the standard chain format | Fixture file in the real layout |
| 13 | `evaluate.py`: run on many dates; log loss against the answer key; calibration; variants (baseline, prompt only, baseline + memory, single pass, ToT); memory check; guardrail ablation (each guardrail switched off in turn on the scenarios: critic, cleaned extracts, role check, risk gate) (Lab 6.1) | Fakes end to end; each ablation lets its planted failure through |
| 13b | Optional, if time allows: the risk triggers as a 0–100 score, with the hold threshold set from the evaluation run to a target review rate (Lab 6.2) | Threshold gives the target rate on a fixed sample |
| 14 | Evaluation run with Claude Haiku on dates after June 2026 (the model's knowledge cutoff); results saved to `results/` | — |

### D. Show and document (19 Nov)

| # | Commit | Tests |
| --- | --- | --- |
| 15 | `app.py`: Streamlit screen over `run_forecast` (forecast, ledger, tree, trace, risk gate, analyst review). First to drop if time is short | AppTest smoke test when installed |
| 16 | README: problem, architecture, setup, demo and live runs, tests, results, repository map | — |
| 17 | Repository tidy-up: sample outputs, evaluation artefacts, figures | — |

**20 Nov:** freeze, fix what broke, report draft. **21 Nov:** slides. **22 Nov:** record and upload. **23 Nov:** final checks and submit.

## Model and the memory problem

- **Live and evaluation runs:** Claude through the API. `claude-haiku-5-5` for large runs, `claude-sonnet-5-5` for the demo.
- **Both have a reliable knowledge cutoff of June 2026.** For 2025 and early 2026 dates they may remember what happened next. The main evaluation therefore uses forecast dates from July 2026 onward, and a memory check asks the model what each stock did after each test date and drops dates it knows.
- **Qwen 2.5 (training data before 2025)** can still run the earlier dates as a comparison free of this problem.

## Report (course template) and where each section comes from

| Template section | Source |
| --- | --- |
| 1. Project title | MosAIc and a one-line description |
| 2. Problem and user | Checkpoint 1; the analyst as user |
| 3. Goal and scope | The forecast definition, the answer key, the 30 stocks, public data only, options for evaluation only |
| 4. Final architecture | Checkpoint 5 (five agents, LangGraph, MCP-style tools), Checkpoint 4 (ToT), Checkpoint 6 (guardrails, risk gate) |
| 5. Design evolution | Checkpoints 1–6 and the design review: baseline C, filing-first evidence, news and the same-story rule, the critic's reasons, the research guards |
| 6. Implementation | The package, LangGraph, CrewAI role cards, Claude API, sentence-transformers, SEC EDGAR, GDELT, FirstRate |
| 7. Evaluation and results | The evaluation run (C14), the nine scenarios, the protocol comparison |
| 8. Safety and reliability | Checkpoint 6 and the notebook scenarios |
| 9. Limitations and next steps | The hostile review; the earnings-focused calibration experiment |
| 10. Repository | The public link |

## Video (8–10 slides, 8–10 minutes)

1. Problem and user
2. Goal and how it is scored (the answer key)
3. Architecture (Figure 1)
4. Live demo: one forecast with its ledger, trace and risk gate
5. Tree of Thought: why and how
6. Guardrails and human review (scenarios 3, 7, 9)
7. Evaluation results
8. What the hostile review found, and the earnings-focused next step
9. Repository and how to run it

## Needed from Chiara

1. The presentation outline template.
2. The FirstRate zip listing (file names and a few lines of one file), by 18 November.
3. An Anthropic API key in Colab Secrets or a `.env` file (never committed).
4. Permission to push, and to make the repository public before the 23rd.

## Decisions

1. **Answer key:** option-implied probabilities from a standard chain format; FirstRate loader in C12.
2. **Baseline:** version C. Drift benchmarks and "baseline + memory" are comparison variants only.
3. **Prompt injection guard:** research sees headers and cleaned extracts; full passages go only to synthesis, which has no search tools.
4. **Metrics:** log loss against the implied probabilities first; calibration plots and Brier score against outcomes alongside.
5. **Market information:** price reaction in case memory, GDELT news with the same-story rule.
6. **Frameworks:** LangGraph runs the flow, CrewAI holds the role cards, tools sit behind one MCP-style interface.
7. **Guardrails (Checkpoint 6, Labs 6.1 and 6.2):** request check before the run, role checks, forecast-date rule, critic with reasons and two retries, baseline fallback, risk gate (publish, flag, hold) including refused tool calls as a trigger, analyst review with a feedback log, and a guardrail ablation in the evaluation.

## Later (not in this build)

The earnings-focused version: forecasts right after after-close results releases for 1 day, 1 week and 1 month, compared by calibration with the price-history baseline, the options at the day-X close, and the options with a statistical correction. Also: analyst estimates from Alpha Vantage, the results press release (Exhibit 99.1), a measured base rate for "no effect", a reranking model for evidence selection.
