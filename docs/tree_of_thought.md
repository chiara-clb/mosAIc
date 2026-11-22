# Testing Tree of Thought: what it really adds

Tree of Thought is the reasoning method at the centre of MosAIc's synthesis step. This page follows how it was built and tested, one question at a time. The numbers come from `docs/results.md`.

## How MosAIc uses it

For each evidence item, the synthesis agent proposes two readings, and code adds "no effect". Each reading is a fall, a gain, a wider spread or no effect, with a size and an exact quote. A reading's score is half its base rate and half its novelty:
- the **base rate** is how often similar past events at the same delay were followed by that kind of move for this stock (case memory);
- the **novelty** is how different the passage is from the same section of the previous filing.

A beam search (width 3) builds the ledger one item at a time and keeps the best combinations. Readings whose quote fails the check are pruned.

Against the four design questions of the Tree of Thoughts paper (Yao et al., 2023), as rebuilt in Lab 4 on the Game of 24:

| Design question | Game of 24 (Lab 4) | MosAIc |
| --- | --- | --- |
| What is one thought? | One equation. Each step depends on the previous one, so a wrong first step dooms the rest | One reading of one evidence item. Items are almost independent (only a 20% consistency term links them) |
| How are thoughts generated? | The model proposes several next steps in one prompt | The model proposes two readings per item, and code adds "no effect" |
| How are states scored? | The model judges sure, likely or impossible, sampled three times, backed by an exact Python check | Code: half base rate, half novelty |
| How does the search run? | BFS keeping the best 5 states, 3 steps | Beam search keeping the best 3, one level per item, with pruning |

## The investigation

**1. Built as the lab teaches (Checkpoint 4, Qwen2.5-3B, 114 dates).** Tree of Thought avoided the first reading's mistakes and left the baseline unchanged in 83 of 114 forecasts. The search worked: beams of 1, 3 and 5 chose the same ledgers. The warning sign was the scorer. Scores and accuracy were almost unrelated (rank correlation 0.06).

**2. It seemed to work (Claude Haiku 5.5, thinking off, 23 forecasts).** Against what actually happened, Tree of Thought clearly beat a single pass: Brier 0.650 against 0.680. Against the option market, both stayed close to the price baseline.

**3. The gap closed (thinking on).** With the model thinking before each answer, the single pass improved most (Brier 0.680 to 0.653, log loss against the options 1.097 to 1.090) and caught up with the tree (1.092). The model now does some of the comparing inside one call.

**4. The lab's own control: five single passes, averaged.** The paper compares Tree of Thought with CoT-SC, many sampled answers combined. MosAIc's version runs research once, then the single pass five times on the same evidence, and averages the five forecasts. It scored almost exactly like the tree: 1.091 against 1.092 against the options, and Brier 0.665 against 0.666. So the tree's gain so far is what smoothing over several readings gives, rather than better reasoning. In the paper the tree beats CoT-SC by a wide margin (74% against 9% on the Game of 24) because its evaluator is strong. Here they tie, which is what the method predicts with a weak evaluator.

**5. How consistent the agent is.** Comparing the forecasts one by one:
- The same evidence often gives different answers. In 11 of the 23 averaged forecasts, the five passes split on whether to hold the forecast for the analyst.
- Variants often lean opposite ways. Where a single pass and the tree both moved, they disagreed on direction 3 times out of 5.
- On the forecasts it moved, the tree leaned the same way as the option market only 5 times in 13 (38%). The single passes did so 8 times in 12 (67%), and the five-pass average 9 times in 14 (64%). One explanation fits the rest: noisy base rates can overrule a first reading that was in line with the market.
- The tree is more cautious. It never moved more than 4.5 points, against up to 6 for a single pass.

These counts rest on 12–14 forecasts each, so they are hints, not findings.

**6. The first diagnosis.** The lab shows when Tree of Thought pays off. Each step must depend on the previous one, the evaluator must recognise good partial solutions, and the cases must be hard enough that a single chain fails. Monthly forecasting from news that is mostly already priced meets none of the three. The items are nearly independent, the scorer rests on few past events per stock, and many dates have little evidence (about half the Checkpoint 4 forecasts had at most one item). So the method is sound, but this problem does not give it what it needs.

**7. Can a better scorer help?** `python -m mosaic.rescore` replays the saved readings of a Tree of Thought run with other scorers, without new model calls. The scorers were fixed before any result was seen:
- the model's own confidence;
- agreement with the stock's price reaction since the filing (past prices only, never options);
- base rates pooled across all 30 stocks;
- their averages.

July forecasts are for choosing, and August ones for checking, and every scorer is reported. The run covers AAPL, MSFT, JPM, AMZN, GOOGL and NVDA from 1 July to 19 August 2026: 47 forecasts (AAPL on 19 August fails on its option file), with thinking on and the same saved news. The replay with the current scorer reproduced all 47 forecasts exactly.

On this larger sample the tree as run is slightly but clearly further from the option market than the baseline: +0.0017 in log loss, 95% interval +0.0006 to +0.0029.

| Scorer | Forecasts moved | vs baseline, all 47 | July (choose, 30) | August (check, 17) |
| --- | --- | --- | --- | --- |
| Model confidence | 2 | −0.0006 | −0.0010 | 0.0000 |
| Current + confidence | 13 | +0.0006 | +0.0008 | +0.0001 |
| Pooled + reaction + confidence | 11 | +0.0013 | +0.0004 | +0.0030 |
| Reaction | 18 | +0.0014 | −0.0003 | +0.0044 |
| Current (the run) | 29 | +0.0017 | +0.0016 | +0.0018 |
| Pooled base rates | 29 | +0.0027 | +0.0022 | +0.0037 |
| Pooled + reaction | 32 | +0.0033 | +0.0028 | +0.0040 |

What this shows:

- **No scorer passes.** Only model confidence comes out ahead, and only because it almost never moves the forecast (2 of 47, none in August). The price reaction helped in July and failed in August, which is what noise looks like.
- **Pooling base rates across all 30 stocks made it worse.** Thin per-stock base rates were not the main problem.
- **The more a scorer moves the forecast, the worse it scores.** The order is almost perfect, from 2 moves to 32. So the readings themselves carry almost nothing that the option market has not already priced. No way of choosing among them can help. In the lab's terms, neither the generator nor the evaluator has a signal to work with.

One data note: the pooled memory found only 8 filings for XOM, which looks like a lookup problem. It affects only the pooled scorers, and only slightly.

**A check on the hard cases.** The paper tested Tree of Thought on deliberately hard puzzles, where a single chain fails. The closest thing here is the weeks before a results release, when there is more news and a big move is coming. The 47 forecasts were split into three groups (log loss against the options, compared with the baseline, so positive is worse):

| Group | Forecasts | Tree moved | Tree vs baseline (95% interval) |
| --- | --- | --- | --- |
| Amazon, Alphabet and Nvidia before their results, with saved Wayback news | 13 | 9 | +0.0017 (+0.0002 to +0.0037) |
| The same three stocks, other weeks, filings only | 11 | 4 | +0.0031 (+0.0008 to +0.0060) |
| AAPL, MSFT and JPM, saved GDELT news | 23 | 16 | +0.0010 (−0.0015 to +0.0036) |

No alternative scorer did better in any group. In the weeks before results, the tree leaned towards a fall 8 times out of 9, while the stocks mostly rose: Amazon was leaned down in all five July weeks and gained over the month in four of them. Only one lean (Alphabet, 1 July) matched what happened. The news before results seems to carry worries, such as AI spending, that the market had already priced. The tree turned them into bearish adjustments, as the results-day reader did with Amazon's free cash flow when thinking was off. So more news before results did not help: the agent moved more and was wrong more often.

This changes the diagnosis in step 6. A better scorer is not enough for the monthly forecast. The evidence needs to be new, which is what results day is for.

**8. Where it should fit.** The same fragility appears on results day. With thinking off, the reader leaned slightly towards a fall on Amazon's Q2 2026 release. With thinking on, the same release was read towards the gain that happened. Voting over several readings, or a tree with a good scorer, addresses exactly this. A natural home for Tree of Thought is tracing one company's results to its suppliers, customers and competitors. There each link is a branch that can point either way. Past releases give hundreds of cases (how company B moved after company A reported) to score each branch.
