# MosAIc design review

A check that every part of the design serves the initial goal, made before writing the application. Sources: the full design (revised Checkpoint 1.1), Checkpoints 2–4, the retrieval notebook, the presentation plan and the feedback.

## 1. The goal

For one of 30 large-cap US stocks and an as-of date, give the probability that the 21-trading-day return is a large fall (below −0.5σ), roughly flat or a large gain (above +0.5σ), where σ is the trailing one-year volatility scaled to 21 days. The forecast is a baseline from prices and the earnings calendar plus a ledger of cited adjustments that add up exactly. The main test is how close the forecast gets to the probabilities implied by option prices on the as-of date, which the agent never sees. Actual outcomes are a secondary check. The agent must beat simple baselines, and the gain must come from evidence the agent found.

Everything below is judged against three questions:

1. Does it help recover the market's view from public information?
2. Does it keep the forecast explainable and adjustable?
3. Does it keep the test fair (no look-ahead, no answer-key leakage, results reproducible)?

## 2. Coverage: each part of the design, what exists, where it goes

Status: **built** = working in the notebook; **partly** = a simpler version exists; **missing** = designed but not built.

| Part | Defined in | Status | Module | Test |
| --- | --- | --- | --- | --- |
| **Forecast definition** (buckets, 21 days, σ from one year of prices) | Full design §1 | built | `config.py`, `buckets.py` | thresholds from synthetic prices |
| **Universe of 30 stocks**, about 3 per sector, liquid options | Full design §1 | missing (only AAPL, NKE, INTC used) | `config.py` | list has 30 tickers, each with a CIK |
| **Answer key**: implied bucket probabilities from option chains | Full design §3, §7, milestone 1 | missing | `answer_key.py` | synthetic chain from a known distribution gives back its probabilities |
| **Scoring**: distance to implied probabilities, Brier vs outcomes, confidence intervals resampled by week | Full design §7; CP2 | missing | `scoring.py` | hand-computed values |
| **Point-in-time corpus** with acceptance times | Full design §3; CP3 | built (EDGAR) | `edgar.py` | fixture filings |
| **as_of enforced in the tool layer** | Full design §3 | built in search | every tool in `tools.py` | each tool refuses later data |
| **Answer-key leakage filter** (drop text quoting implied volatility or implied moves) | Full design §3, §6 | missing | `edgar.py`, `news.py` | planted sentences are dropped |
| **Retrieval**: 150/30 chunks, MiniLM, 30-day filter, 20 → MMR 5, "nothing found" < 0.30 | CP3 | built | `index.py` | hand-made vectors |
| **News agent** | Full design §3 (MVP); CP3 (second step) | missing | `news.py` | interface only in this build |
| **Coordinator**: earnings in the window? | Full design §4; CP1–2 | partly (guesses next results 91 days after the last) | `coordinator.py` | dates around the window edge |
| **Quant baseline** | Full design §4 | partly (EWMA, no implied-to-realised ratio) | `baseline.py` | synthetic prices |
| **ReAct research loop** with step limit | CP2 | built | `research.py` | scripted model |
| **ToT synthesis** (beam 3, depth ≤ 5, wider spread, scoring, pruning, ties) | CP4 | missing | `tot.py` | see plan step |
| **Ledger arithmetic** (3/6/9 points, cap 15, sums to zero) | CP1, notebook | built (no wider spread) | `ledger.py` | sums, cap, floors |
| **Critic** (sources, dates, direction, double counting, arithmetic), retries at most twice, then baseline | Full design §4; CP2 | partly (code checks only, no retries) | `critic.py`, `pipeline.py` | each rejection rule |
| **Citation entailment check** by a model | Full design §6; CP4 grounding | missing | `critic.py` (shared with ToT) | stand-in model |
| **Working record** for one forecast | CP2 | built (trace) | `trace.py` | — |
| **Case memory** | CP2, CP4 | missing | `case_memory.py` | synthetic events |
| **Reflection notes**, read at the start of later forecasts | CP2 | partly (written, never read) | `reflection.py` | notes appear in the next prompt |
| **Novelty check** against the previous filing | CP4; plan | missing | `novelty.py` | repeated vs new wording |
| **Trace and leakage audit** | Plan; CP2 | built | `trace.py` | planted violation is caught |
| **Interactive mode**: strike, resize, add lines, plain-language assumptions, implied probabilities shown afterwards | Full design §5 | missing | `app/` | recompute without new research |
| **Autonomous mode**, reproducible from (ticker, as_of, model version, seed) | Full design §5 | partly (greedy decoding) | `pipeline.py` | same inputs give same record |
| **Forward test**: forecasts logged and hashed before option data is pulled | Full design §7 | missing | `forward_log.py` | hash changes if a record is edited |
| **Baselines**: normal prior, price-only, prompt-only, no retrieval, no ToT | Full design §7; CP2; CP3 feedback | partly (no retrieval only) | `evaluate.py` | each runs on fakes |
| **Safety**: prompt injection, fabricated figures, unsupported citations, runaway cost | Full design §6 | partly | across modules | planted documents, caps |

## 3. Where the documents disagree or leave a gap

Each item has a recommendation. Decisions so far: 3.1, 3.3 and 3.4 accepted as recommended; 3.2 still open; 3.5 to 3.8 follow the recommendations unless revisited.

### 3.1 The answer key has no data source yet

**Issue:** the full design makes the answer-key pipeline the first milestone, because every result depends on it. It needs option chains, which come from a paid source (ThetaData or ORATS, about $40–100 a month). Nothing has been built or bought.

**Recommendation:** build the answer key now against a common chain format, tested on synthetic chains with a known answer. Plug in the real source once you've picked one. Without it the app can still forecast, but cannot be scored except against actual outcomes.

### 3.2 The baseline uses option data, which the checkpoints say never happens

**Issue:** the full design scales the price-based volatility by a frozen implied-to-realised ratio per stock, estimated from 2020–2025 options. Checkpoints 1.1 and 2 and the working rules say option prices are never an input to the agent. The notebook leaves the ratio out.

**Recommendation:** keep the ratio as an optional input, off by default, and report the price-only baseline both with and without it. The ratio is fixed before the test window, so it does not reveal any test date's answer, but the submissions should state it openly. Also choose between GARCH (full design) and EWMA (notebook). I suggest EWMA first, since it has no fitting step that can fail, with GARCH as an option.

### 3.3 Raw passages reach the research agent, which can call tools

**Issue:** the full design's guard against prompt injection says raw retrieved text never reaches an agent that can call tools. In the notebook's ReAct loop, the research model reads full passages and then chooses the next search.

**Recommendation:** show the research agent only each result's header, date and score plus a short extract with instruction-like text removed. Full passages go only to the synthesis step, which has no tools. This keeps the stated guard true, and the tests plant adversarial passages to check it.

### 3.4 Two different metrics, and ToT tuned on the secondary one

**Issue:** the full design's primary metric is KL divergence and total variation distance to the implied probabilities. The checkpoints say log loss against the implied probabilities and Brier score against outcomes. Checkpoint 4 tunes the ToT weights with the Brier score, which uses the secondary target.

**Recommendation:** primary metric is log loss against the implied probabilities. It equals KL divergence plus a constant, so the two agree. Report total variation distance alongside. Secondary metric is Brier score against outcomes. Tune the ToT weights on the primary metric, on dates kept separate from the test dates.

### 3.5 "Case memory" means two things

**Issue:** Checkpoint 2 describes past forecasts with their lines and sizes. Checkpoint 4 uses base rates of similar past events over the next 21 days.

**Recommendation:** one store with two tables. Event base rates are built from filings and prices: event type from the 8-K item code, outcome bucket 21 days later. Past ledgers are added after each forecast with their scores. ToT reads the base rates now; past ledgers become useful once there is a backtest.

### 3.6 The 30-day rule versus "what the price does not yet reflect"

**Issue:** the 30-day window fixed stale results, but the plan notes that results already announced are mostly priced in, while open matters older than 30 days (guidance, pending deals, lawsuits) can still matter.

**Recommendation:** keep the 30-day window for this build, since it is tested and in the submissions. Topic-based superseding stays on the later list. The novelty check and the "no effect" reading in ToT already lower the weight of priced-in items.

### 3.7 The earnings date is guessed

**Issue:** the coordinator assumes the next results come about 91 days after the last results 8-K. A wrong guess changes the baseline, and the design names earnings as the main source of gains.

**Recommendation:** keep the estimate, log it, and report forecasts near the window edges separately. A real earnings calendar can replace it later behind the same function.

### 3.8 Smaller gaps

- **Reflection notes** are written but never read; the pipeline will add the latest notes to the research prompt.
- **News agent** is MVP in the full design but a second step in Checkpoint 3; this build adds the interface and the leakage filter, and the API comes later.
- **Code sandbox**: the full design has the quant agent writing code, while the notebook runs fixed code. Fixed, tested code is safer and meets "all numbers computed in code", so the sandbox is dropped unless the course requires it.
- **Model training cutoff**: with an API model, test dates before its cutoff may be remembered. The pinned model's cutoff goes in the run record, and later dates are reported separately.
- **The ledger example** in the full design (buyback −2 / 0 / +2) is not on the 3/6/9 scale; the code uses the scale.

## 4. What this changes in the plan

The build follows the full design's milestones instead of the notebook's order: forecast definition, answer key and scoring first, then the data and tools with as_of enforced, then the agent, then evaluation and the interface. The updated steps are in `PLAN.md`.
