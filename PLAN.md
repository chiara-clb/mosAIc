# MosAIc build plan

Build the application from the full design, checked part by part against the initial goal in `docs/design_review.md`. Each step is one small commit containing the feature and its tests. Nothing is pushed without asking first.

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
| 6 | `index.py`: embeddings, date-filtered search, MMR, "nothing found" | Hand-made vectors: no late chunks, MMR drops near-copies, threshold |
| 7 | `prices.py` and `coordinator.py`: price loader, results dates from 8-K item 2.02, earnings-in-window estimate | Window edges; no later data |
| 8 | `baseline.py`: EWMA (GARCH optional), earnings-move mix, optional frozen ratio, fallback | Steady volatility gives about 31/38/31; earnings widens both tails |
| 9 | `ledger.py`: fall, gain, wider spread, no effect; cap; floors | Each line sums to zero; cap scales all lines together |
| 10 | `novelty.py`: compare with the same section of the previous filing | Repeated wording low, new wording high |
| 11 | `case_memory.py`: event base rates from 8-K items and prices; past-ledger table | Synthetic events give the expected rates; only earlier events used |

### C. The agent

| # | Commit | Tests |
| --- | --- | --- |
| 12 | `llm.py`: chat model factory and structured output with a JSON fallback | Stand-in model; messy JSON parsed |
| 13 | `research.py`: ReAct loop with LangChain tools; agent sees headers and cleaned extracts only; reflection notes in the prompt | Scripted model: step limit, repeated queries, empty results; planted instruction text never reaches the agent |
| 14 | `mcp_servers/`: case memory, document index, branch log | In-memory MCP client calls each tool |
| 15 | `tot.py`: beam search synthesis; every node serialised as a JSON record (id, parent, readings, scores) in the branch log | Node records round-trip through the branch log; beam of 3; all-"no effect" ledger kept; grounding and date failures pruned; 0.3 rule; ties to smaller shift; 25-call budget; baseline if nothing survives |
| 16 | `critic.py` and `reflection.py`: final checks, entailment check, notes | Each rejection rule; notes written and read back |
| 17 | `pipeline.py` and `crew.py`: run_forecast with retries and baseline fallback; run record with model version and seed; CrewAI roles | Fakes end to end; leakage audit clean; CrewAI test skips if not installed |

### D. Evaluation and interface

| # | Commit | Tests |
| --- | --- | --- |
| 18 | `evaluate.py`: baselines (normal prior, price-only, prompt-only, no retrieval, no ToT) and ablations, including a sweep over the ToT settings the class lesson names (beam width, pruning margin, depth limit); `forward_log.py` with hashes | Each baseline runs on fakes; edited record fails the hash |
| 19 | Streamlit app: forecast, ledger, tree, trace, audit; interactive mode (strike, resize, add lines, plain-language assumptions); implied probabilities shown only after the forecast | UI helpers; AppTest smoke test when installed |
| 20 | README: setup, API keys, Colab, building the index, running tests | — |

## Decisions (from the design review)

1. **Answer key** (3.1): decided. The calculation is built now against a standard option-chain format and tested on synthetic chains. The data provider is chosen later, then a loader is added.
2. **Baseline** (3.2): open. Frozen implied-to-realised ratio on or off; EWMA or GARCH. Blocks step 8 only.
3. **Prompt injection guard** (3.3): decided, option A. The research agent sees each result's header, score and a cleaned extract of about 40 words; full passages go only to the synthesis step, which has no tools.
4. **Metrics** (3.4): decided. Main metric is log loss against the implied probabilities, with total variation distance alongside; Brier score against outcomes is secondary. ToT weights are tuned on the main metric, on dates kept apart from the test dates. Checkpoint 4 needs one sentence changed to match.
5. **The 30 stocks:** proposed list in `mosaic/config.py` (about 3 per sector; AAPL, NKE and INTC kept from the checkpoint runs).

## Testing here vs on your machine

This workspace can't reach PyPI, SEC EDGAR, Yahoo Finance or Hugging Face, and LangChain, CrewAI, Streamlit and sentence-transformers aren't installed. Steps 0–11, 14–16 and the logic in 17–18 are tested here with fakes. Tests needing LangChain, CrewAI or Streamlit skip when missing and must be run on your machine or Colab, as must the first live run.

## Later (not in this build)

News API and GDELT, topic-based superseding, λ tuning on labelled queries, real earnings calendar.
