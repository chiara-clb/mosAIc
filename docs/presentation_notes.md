# Observations for the final presentation

Findings from the validation notebooks that are not part of a checkpoint submission. Each entry says where it came from, so it can be rerun.

## 1. Tree of Thought test (Colab, 1 November 2026)

**Setup.** `notebooks/tot_validation`, Qwen2.5-3B, 21 forecasts (AAPL, NKE, INTC, two days after each results filing, 2025–26), scored against actual outcomes with the Brier score (lower is better). No option data yet.

| Variant | Brier | vs baseline | Forecasts changed |
| --- | --- | --- | --- |
| Baseline | 0.716 | — | — |
| Prompt only | 0.653 | −0.063 | 21 |
| Single pass (first reading) | 0.759 | +0.043 | 12 |
| ToT beam 1 | 0.732 | +0.015 | 6 |
| ToT beam 3 | 0.732 | +0.015 | 6 |

1. **ToT beat committing to the first reading.** Base rates and novelty stopped most first impressions; ToT kept the baseline in 16 of 21 forecasts.
2. **Search breadth did not matter here.** Beams of 1, 3 and 5 and every pruning margin chose the same ledgers. The beam found the top-scored ledger in 21 of 21. Forecasts had 1.7 evidence items on average, so there was little to search over. The gain came from the scoring.
3. **Scoring vs outcomes was weak.** Across forecasts with at least 3 possible ledgers, higher-scored ledgers were only slightly more often right (average rank correlation +0.17, range −0.54 to +1.00). A pruned ledger was closer to the outcome in 8 of 21, mostly when the stock moved and ToT stayed at the baseline.
4. **Novelty drives "no effect".** Median novelty 0.09; a quarter of passages score 0 (similarity to the previous filing at or above 0.95). 10-Q legal sections and results press releases repeat almost word for word. Novelty cannot see changed numbers, since the embeddings ignore them.
5. **Cost.** 3.6 model calls and 15 seconds per forecast on average; at most 10 calls and 42 seconds.
6. **Leakage audit.** 36 evidence passages checked, 0 dated on or after the forecast date.
7. **Gaps found.** NKE had no evidence on two dates (to diagnose). A bug made the beam-1 rows of the settings sweep equal the baseline; fix: keep the "no effect" node only when the beam has at least 2 slots.

## 2. Baseline comparison (Colab, 1 November 2026)

**Setup.** `notebooks/baseline_comparison`, all 30 stocks, every Monday 2016 to September 2026, 16,800 forecasts, scored against actual outcomes. Version D only learns from windows that ended before each forecast date.

| Version | Brier | vs fixed guess (95% interval) | Log loss |
| --- | --- | --- | --- |
| A: current (recent moves, last days weigh most) | 0.6567 | −0.0025 (−0.0060, +0.0008) | 1.0848 |
| B: A without results-day moves | 0.6551 | −0.0041 (−0.0075, −0.0006) | 1.0822 |
| C: B mixed half and half with the past year | **0.6528** | **−0.0064 (−0.0084, −0.0047)** | **1.0787** |
| D: C with the learned post-results calm-down factor | 0.6541 | −0.0051 (−0.0071, −0.0033) | 1.0806 |
| Fixed guess 30.85 / 38.3 / 30.85 | 0.6592 | — | 1.0877 |

Right after results (1 to 5 trading days, 1,362 forecasts): A 0.6628 (worse than the fixed guess, +0.0086), B 0.6522, C 0.6485, D 0.6484, fixed guess 0.6542.

How often months were actually flat, against what each version predicted (percent):

| Days since results | Forecasts | Actually flat | A | B | C | D | Fixed |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1–5 | 1,362 | 44.2 | 38.6 | 44.5 | 41.6 | 43.2 | 38.3 |
| 6–15 | 2,666 | 44.4 | 39.8 | 44.1 | 41.3 | 41.3 | 38.3 |
| 16–30 | 3,970 | 42.5 | 42.2 | 44.9 | 41.6 | 39.3 | 38.3 |
| 31–45 | 3,842 | 42.7 | 42.2 | 42.9 | 40.1 | 40.4 | 38.3 |
| 46–63 | 3,987 | 34.3 | 40.4 | 38.7 | 36.7 | 37.3 | 38.3 |
| 64+ | 246 | 34.6 | 43.4 | 43.5 | 40.6 | 39.7 | 38.3 |

Learned calm-down factors (variance actually seen / variance version C expected): 1–5 days 0.91, 6–15 0.99, 16–30 1.17, 31–45 0.99, 46–63 0.92, 64+ 1.09.

1. **Months after results are calmer:** 44% flat against the 38% a fixed guess expects. Version A misses this because a results-day jump inflates its estimate for weeks (tests showed a 12% jump still matters 40 trading days later).
2. **Leaving results days out (B) fixes most of it**; mixing with the past year (C) is the best overall and the answer to question 2b. The learned calm-down factor (D) adds nothing on top, so it is dropped.
3. **Remaining error: the month before results.** When results fall inside the window only 34% of months are flat, but every version predicts 37–40%. Likely causes: the results-jump term uses the average jump size squared instead of the average squared jump (undercounts big jumps), XOM's missing results dates, and mid-quarter item-2.02 filings (fixed since: one results filing per quarter). Next: version E with the root-mean-square of the last 8 results-day moves.
4. **Price history adds little against actual outcomes:** the best version beats the fixed guess by about 1% in Brier score. Differences should be larger against the option-implied probabilities, the main metric, once the FirstRate data is loaded.
5. **Prompt only** beat the price baseline in the ToT test (0.653 vs 0.716 on 21 forecasts). Its larger test (about 1,200 forecasts from 2025, with a "model's average answer" check) is ready in section 9 of the baseline notebook and not yet run.

## 3. Data issues found

- **XOM:** only 2 item-2.02 filings in the SEC list (both from mid-2026), although Exxon's April 2019 results 8-K uses item 2.02. To check with the diagnostic cell.
- **Extra item-2.02 filings** for NKE (145 since 2004), TSLA (117), CVX (123), PLD (117): mid-quarter updates. Fixed by keeping the last one before each 10-Q or 10-K.
- **Section matching** between 10-Q and 10-K uses titles, because item numbers differ between the two forms.
