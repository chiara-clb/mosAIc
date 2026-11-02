# MosAIc build plan

Build the application from the full design, checked part by part against the initial goal in `docs/design_review.md`, including the lessons from the validation notebooks (its section 5). Each step is one small commit containing the feature and its tests. Nothing is pushed without asking first.

## Choices

- **Order:** the full design's milestones. First the forecast definition, answer key and scoring, then the data and tools with as_of enforced, then the agent, then evaluation and the interface.
- **LLM:** pluggable through LangChain chat models. An API model is the default; a local Hugging Face model (e.g. Qwen2.5-3B on Colab) is an option. Temperature 0 and the model version are recorded in every run.
- **Frameworks (Checkpoint 4):** LangChain for the research tools and the ToT generator, evaluator and controller; MCP servers (official `mcp` SDK) for case memory, the document index and the branch log; CrewAI for the agent roles.
- **Testable core:** logic lives in plain modules with small interfaces (fetcher, embedder, price source, option-chain source, chat model), so every feature is tested offline with fakes. Framework wrappers sit on top.

## Steps (one commit each)

### A. Forecast definition and evaluation target

| # | Commit | Tests |
| --- | --- | --- |
| 0 | Scaffold: `pyproject.toml` with extras, `config.py` with all design values and the 30-stock universe, pytest, `.gitignore`, plan and design review | Config matches the design values; 30 unique tickers |
| 1 | `buckets.py`: σ from one year of prices, thresholds, realised bucket of a 21-day return | Synthetic prices; only prices up to the as-of date are used |
| 2 | `answer_key.py`: implied bucket probabilities from an option chain, chain quality checks, unscorable dates logged | Chains built from a known distribution give back its probabilities; thin chains rejected |
| 3 | `scoring.py`: log loss and total variation distance to implied probabilities, Brier score vs outcomes, confidence intervals resampled by week | Hand-computed values; perfect forecast scores best |

### B. Data and tools, with as_of enforced

| # | Commit | Tests |
| --- | --- | --- |
| 4 | `trace.py`: JSONL trace and leakage audit | Planted late source is caught |
| 5 | `edgar.py`: download, clean, sections, chunks, answer-key leakage filter | Fixture filings; sentences quoting implied moves dropped |
| 6 | `index.py`: embeddings, date-filtered search, MMR, "nothing found" at 0.20 (a setting); boilerplate filter; every filing in the 30-day window as a candidate (passage closest to its item title); `select_evidence`: pool searches and filing candidates, best passage per filing, MMR to 5 (design review 3.8, 5.2) | Hand-made vectors: no late chunks, MMR drops near-copies, threshold; short 8-Ks missed by the searches are still found; one passage per filing, at most 5 |
| 7 | `prices.py` and `coordinator.py`: price loader (stocks and the S&P 500 fund), results dates from 8-K item 2.02 kept one per quarter (last before each 10-Q / 10-K), results-day rule (after 20:00 UTC = next day), earnings-in-window estimate, warning for stocks with too few results dates (XOM) | Window edges; no later data; interim updates skipped; after-close filings move the next day |
| 8 | `baseline.py`: version C (recent moves without results days, blended half and half with the past year), results-day jump when results are due, version E option (root mean square of the last 8 jumps), optional frozen ratio, fallback (design review 5.1) | Steady volatility gives about 31/38/31; a fresh results-day jump does not inflate C; earnings widens both tails; E vs C on synthetic jumps |
| 9 | `ledger.py`: fall, gain, wider spread, no effect; cap; a 1% floor that holds after rescaling | Each line sums to zero; cap scales all lines together; floor holds |
| 10 | `novelty.py`: compare with the same section (matched by title) of the previous filing; similarity rescaled between 0.5 and 0.95; neutral for news | Repeated wording low, new wording high; 10-K counts as previous report for a 10-Q |
| 11 | `case_memory.py`: event base rates from 8-K items and prices, at the same delay; price reaction against the S&P 500 and base rates conditioned on it; "any event" rates for news; past-ledger table | Synthetic events give the expected rates; only windows ended before the forecast date; drift after big falls found; market-wide moves are not company reactions |
| 11b | `news.py`: GDELT client (cached, 5-second spacing), headline parsing with the answer-key leakage filter, stories, tone shift, same-story rule (design review 5.3) | Fake GDELT replies: later and option-related headlines dropped; stories need 2 articles; coverage linked to a filing within 3 days; at most 2 news items |

### C. The agent

| # | Commit | Tests |
| --- | --- | --- |
| 12 | `llm.py`: chat model factory and structured output with a JSON fallback | Stand-in model; messy JSON parsed |
| 13 | `research.py`: ReAct loop with LangChain tools over filings and news; agent sees headers and cleaned extracts only; reflection notes in the prompt | Scripted model: step limit, repeated queries, empty results; planted instruction text never reaches the agent |
| 14 | `mcp_servers/`: case memory, document index, branch log | In-memory MCP client calls each tool |
| 15 | `tot.py`: beam search synthesis; generator sees each passage with its price reaction, news coverage and tone; every node serialised as a JSON record (id, parent, readings, scores) in the branch log | Node records round-trip through the branch log; beam of 3; all-"no effect" ledger kept (only with 2+ slots); grounding and date failures pruned; 0.3 rule; ties to smaller shift; 25-call budget |
| 16 | `critic.py` and `reflection.py`: final checks, entailment check, notes | Each rejection rule; notes written and read back |
| 17 | `pipeline.py` and `crew.py`: run_forecast with retries and baseline fallback; run record with model version and seed; CrewAI roles | Fakes end to end; leakage audit clean; CrewAI test skips if not installed |

### D. Evaluation and interface

| # | Commit | Tests |
| --- | --- | --- |
| 18 | `evaluate.py`: baselines (fixed guess, price-only, prompt-only with its average-answer check, no retrieval, no news, no ToT) and ablations, a sweep over beam width, pruning margin and depth, search vs scoring errors; five test scenarios (after results, mid-quarter, results coming, event days, random weeks) reported separately; `forward_log.py` with hashes | Each baseline runs on fakes; scenario dates as specified; edited record fails the hash |
| 19 | Streamlit app: forecast, ledger, tree, trace, audit; interactive mode (strike, resize, add lines, plain-language assumptions); implied probabilities shown only after the forecast | UI helpers; AppTest smoke test when installed |
| 20 | README: setup, API keys, Colab, building the index, running tests | — |

## Decisions (from the design review)

1. **Answer key** (3.1): decided. The calculation is built now against a standard option-chain format and tested on synthetic chains. The data provider is chosen later, then a loader is added.
2. **Baseline** (3.2, 5.1): decided, version C; version E to test. The option-based correction ratio stays open until the FirstRate data is loaded.
3. **Prompt injection guard** (3.3): decided, option A. The research agent sees each result's header, score and a cleaned extract of about 40 words; full passages go only to the synthesis step, which has no tools.
4. **Metrics** (3.4): decided. Main metric is log loss against the implied probabilities, with total variation distance alongside; Brier score against outcomes is secondary. ToT weights are tuned on the main metric, on dates kept apart from the test dates. Checkpoint 4 needs one sentence changed to match.
5. **The 30 stocks:** proposed list in `mosaic/config.py` (about 3 per sector; AAPL, NKE and INTC kept from the checkpoint runs).
6. **Market information** (5.3): decided. Price reaction in case memory, then GDELT news (free) with the same-story rule.
7. **Option data:** FirstRate Data end-of-day files; a loader for them is added to the answer key once the file layout is known.

## Testing here vs on your machine

This workspace can't reach PyPI, SEC EDGAR, Yahoo Finance or Hugging Face, and LangChain, CrewAI, Streamlit and sentence-transformers aren't installed. Steps 0–11, 14–16 and the logic in 17–18 are tested here with fakes. Tests needing LangChain, CrewAI or Streamlit skip when missing and must be run on your machine or Colab, as must the first live run.

## Later (not in this build)

Topic-based superseding, λ and search cut-off tuning on labelled queries, real earnings calendar, past news in case memory. Experiments on choosing the five evidence items, compared with MMR on the same past dates: a reranking model (seen in class) and one passage per topic.
