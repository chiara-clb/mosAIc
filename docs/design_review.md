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

Each item has a recommendation. Decisions so far: 3.1, 3.3, 3.4 and 3.8 decided; 3.2 decided for the volatility part (version C) with the option-based ratio still open; 3.5 to 3.7 and 3.9 follow the recommendations unless revisited. Section 5 adds what the validation notebooks taught.

### 3.1 The answer key has no data source yet

**Issue:** the full design makes the answer-key pipeline the first milestone, because every result depends on it. It needs option chains, which come from a paid source (ThetaData or ORATS, about $40–100 a month). Nothing has been built or bought.

**Recommendation:** build the answer key now against a common chain format, tested on synthetic chains with a known answer. Plug in the real source once you've picked one. Without it the app can still forecast, but cannot be scored except against actual outcomes.

### 3.2 The baseline uses option data, which the checkpoints say never happens

**Issue:** the full design scales the price-based volatility by a frozen implied-to-realised ratio per stock, estimated from 2020–2025 options. Checkpoints 1.1 and 2 and the working rules say option prices are never an input to the agent. The notebook leaves the ratio out.

**Recommendation:** keep the ratio as an optional input, off by default, and report the price-only baseline both with and without it. The ratio is fixed before the test window, so it does not reveal any test date's answer, but the submissions should state it openly.

**Decision on the volatility part (November 2026):** version C from the baseline comparison (section 5.1). GARCH is not needed. The ratio stays open until the FirstRate option data is loaded; it can then be measured on 2020–2024 and tested both ways.

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

### 3.8 How the five evidence items for ToT are chosen

**Issue:** the research loop runs up to 4 searches of 5 passages each, but ToT takes at most five items, and no document said how they are picked. The notebook took the 10 newest.

**Decision:** pool the passages from all searches, keep the best passage per filing, then run MMR to 5, using each passage's best score across the searches as its relevance. MMR decides which evidence goes in; ToT decides what it means. A reranking model and one passage per topic are later experiments, compared with MMR on the same past dates.

**Updated (section 5.2):** every filing in the 30-day window also enters the pool directly, and news stories take up to 2 of the 5 places.

### 3.9 Smaller gaps

- **Reflection notes** are written but never read; the pipeline will add the latest notes to the research prompt.
- **News agent** is MVP in the full design but a second step in Checkpoint 3. Now decided (section 5.3): GDELT headlines and tone, free, in this build.
- **Code sandbox**: the full design has the quant agent writing code, while the notebook runs fixed code. Fixed, tested code is safer and meets "all numbers computed in code", so the sandbox is dropped unless the course requires it.
- **Model training cutoff**: with an API model, test dates before its cutoff may be remembered. The pinned model's cutoff goes in the run record, and later dates are reported separately.
- **The ledger example** in the full design (buyback −2 / 0 / +2) is not on the 3/6/9 scale; the code uses the scale.

## 4. What this changes in the plan

The build follows the full design's milestones instead of the notebook's order: forecast definition, answer key and scoring first, then the data and tools with as_of enforced, then the agent, then evaluation and the interface. The updated steps are in `PLAN.md`.

## 5. Lessons from the validation notebooks (November 2026)

The Tree of Thought validation notebook (21, then 114 forecasts) and the baseline comparison notebook (16,800 weekly forecasts, all 30 stocks) showed where the design fell short. Each lesson below is now a design decision for the final application; the numbers are in `docs/presentation_notes.md`.

### 5.1 The baseline

| Lesson | Decision for the application |
| --- | --- |
| A results-day jump made the old baseline (recent moves, last days weigh most) expect a jumpy month for weeks after every announcement, while the month after results is usually calm (44% flat, against 38% expected) | **Leave results days out of "recent moves"**; results count only through the separate results-day jump term |
| Mixing recent moves half and half with the past year (also without results days) was the best version overall, and the old version was worse than a fixed 31 / 38 / 31 guess right after results | **Baseline = version C**: 0.5 × recent (weights 0.94 per day) + 0.5 × past year, results days left out |
| A learned "calm-down after results" correction added nothing on top of C | **Not used** |
| Every version expects too many flat months when the next results fall inside the month (34% actual, 37–40% predicted) | **Version E to test:** the results-day jump term uses the root mean square of the last 8 results-day moves instead of the average size; keep it only if it fixes this |
| Some companies file item 2.02 for mid-quarter updates (Chevron, Nike, Tesla, Prologis); Exxon's results are missing from the SEC item list | **One results announcement per quarter** (the last 2.02 filing before each 10-Q or 10-K); a check for stocks with too few results dates; XOM to be fixed by hand if the SEC list stays incomplete |
| Results filed after the close move the stock the next day | **Results-day rule:** filed at or after 20:00 UTC means next trading day |
| Against actual outcomes, price history beats a fixed guess by only about 1%; the option-implied answer key should separate versions better | **Choose the final baseline again on the main metric** once the FirstRate data is loaded |

### 5.2 Evidence from filings

| Lesson | Decision |
| --- | --- |
| In 81 of 114 forecasts the filings were in the 30-day window but all searches scored them just under 0.30, so nothing was kept; event 8-Ks are short and formally worded | **Every filing in the window is a candidate**, with its passage closest to the filing's own item title (or the main report topics); searches stay as an extra source for long reports |
| The 0.30 cut-off from Checkpoint 3 was never tested | **Search cut-off 0.20, as a setting**, to be tuned on labelled examples (presentation plan task) |
| Signature pages and form text compete with real content | **Boilerplate filter** before candidates are chosen |
| 10-Qs and 10-Ks number their items differently | **Match sections by title** for the novelty check |
| Raw "1 − similarity" made almost every passage look repeated | **Novelty rescaled** between similarity 0.5 (new) and 0.95 (repeated); check the cut-offs by reading the side-by-side examples |
| Between results, the stock often has no filing at all in 30 days | **News fills the gap** (5.3); with no evidence, ToT keeps the baseline |

### 5.3 Market information and news

| Lesson | Decision |
| --- | --- |
| Reading only the company's own documents gives a partial view of the market; the options market reacts to price moves, news and coverage too | **Market information is part of the evidence step**, with the order: price reaction first, news second |
| The market's own verdict on a filing is how the stock moved since then | **Price reaction in case memory:** the move since the filing against the S&P 500, in typical daily moves, bucketed from big fall to big rise; base rates count past events of the same type *with the same reaction* over the same delay; the reaction is shown to the generator |
| Free news with dates and tone exists; paid APIs are out of budget | **GDELT**: headlines from the 14 days before the forecast, grouped into stories (at least 2 articles), tone over 14 days against the 60 before; option-related headlines dropped (answer-key leakage); calls cached and spaced 5 seconds |
| A filing and its press coverage are often the same story | **Same-story rule:** a story starting within 3 days of a filing and similar to it is coverage of that filing (shown next to it, no own line); other stories become news items, at most 2 per forecast |
| Case memory has no history of past news | News items use base rates for **any event with the same price reaction**, and neutral novelty (0.5) |
| GDELT's history may be limited (officially 3 months, older notes say 2017) | **Coverage check by year**; backtests report forecasts with and without news separately |

### 5.4 Testing

| Lesson | Decision |
| --- | --- |
| Forecasts only two days after results test one calm phase with one kind of evidence | **Five test scenarios** in every evaluation: after results, mid-quarter, results coming, event days, random weeks; results reported per scenario |
| The beam-of-1 comparison was broken by the "keep the no-effect node" rule | **Keep that node only when the beam has at least 2 slots** |
| The 1% floor could end below 1% after rescaling (Checkpoint 3 code) | **Floor that holds after rescaling** |
| ToT beat the first reading but not the baseline; the search breadth never mattered with 1–2 items per forecast | Keep beam 3; **report search errors and scoring errors separately**; revisit the beam once forecasts have more evidence |
| The prompt-only model scored well where the agent had no evidence | **Prompt-only stays a baseline**, tested on the large sample with a "model's average answer" check |
