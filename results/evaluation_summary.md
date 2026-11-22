# Evaluation summaries

Averages only, from `python -m mosaic.evaluate` and `python -m mosaic.rescore`. The per-forecast files (`forecasts.csv`, `readings.jsonl`, `rescored.csv`) stay out of the repository because they hold probabilities worked out from licensed FirstRate option data. Log loss is against the option-implied probabilities unless marked "outcome"; lower is better. Intervals are 95%, resampled by week.

## 1. Main run: AAPL, MSFT, JPM, 1 July to 19 August 2026, thinking off

23 forecasts (AAPL on 19 August fails on a duplicated row in its option file).

| Variant | Log loss | vs baseline (95%) | Log loss, outcome | Brier, outcome | Model calls |
| --- | --- | --- | --- | --- | --- |
| Baseline | 1.0897 | | 1.0708 | 0.6473 | 0 |
| Prompt only | 1.1182 | +0.0285 (+0.0105 to +0.0464) | 1.0795 | 0.6635 | 1 |
| Baseline + memory | 1.0894 | −0.0002 (−0.0005 to −0.0000) | 1.0675 | 0.6451 | 1 |
| Single pass | 1.0966 | +0.0070 (−0.0032 to +0.0218) | 1.1171 | 0.6798 | 10.8 |
| Tree of Thought | 1.0937 | +0.0040 (−0.0004 to +0.0093) | 1.0737 | 0.6497 | 10.9 |

## 2. Same forecasts, thinking on (`--news-offline`, same news)

| Variant | Log loss | vs baseline (95%) | Log loss, outcome | Brier, outcome | Model calls |
| --- | --- | --- | --- | --- | --- |
| Baseline | 1.0897 | | 1.0708 | 0.6473 | 0 |
| Prompt only | 1.1081 | +0.0185 (+0.0036 to +0.0322) | 1.0557 | 0.6436 | 1 |
| Baseline + memory | 1.0892 | −0.0004 (−0.0016 to +0.0005) | 1.0670 | 0.6447 | 1 |
| Single pass | 1.0902 | +0.0005 (−0.0026 to +0.0036) | 1.0782 | 0.6534 | 10.5 |
| Tree of Thought | 1.0916 | +0.0020 (−0.0002 to +0.0046) | 1.0988 | 0.6663 | 10.1 |
| 5 single passes, averaged | 1.0911 | +0.0014 (−0.0013 to +0.0059) | 1.0962 | 0.6650 | 37.0 |

## 3. Larger Tree of Thought run: six stocks, thinking on

AAPL, MSFT, JPM, AMZN, GOOGL, NVDA, 1 July to 19 August 2026: 47 forecasts.

| Variant | Log loss | vs baseline (95%) | Log loss, outcome | Brier, outcome | Model calls |
| --- | --- | --- | --- | --- | --- |
| Baseline | 1.0917 | | 1.0752 | 0.6504 | 0 |
| Tree of Thought | 1.0934 | +0.0017 (+0.0006 to +0.0029) | 1.1045 | 0.6704 | 8.6 |

## 4. The same readings, rescored (`python -m mosaic.rescore --pooled`)

The current scorer reproduced all 47 forecasts. July forecasts were for choosing (30), August ones for checking (17).

| Scorer | Forecasts moved | vs baseline, all | July | August | Brier, outcome |
| --- | --- | --- | --- | --- | --- |
| Current (the run) | 29 | +0.0017 | +0.0016 | +0.0018 | 0.6704 |
| Model confidence | 2 | −0.0006 | −0.0010 | 0.0000 | 0.6454 |
| Price reaction | 18 | +0.0014 | −0.0003 | +0.0044 | 0.6673 |
| Current + confidence | 13 | +0.0006 | +0.0008 | +0.0001 | 0.6607 |
| Pooled base rates | 29 | +0.0027 | +0.0022 | +0.0037 | 0.6726 |
| Pooled + reaction | 32 | +0.0033 | +0.0028 | +0.0040 | 0.6803 |
| Pooled + reaction + confidence | 12 | +0.0015 | +0.0007 | +0.0030 | 0.6642 |
| Current, boilerplate pruned (check added after the run) | 18 | −0.0000 | −0.0001 | +0.0001 | 0.6571 |

19 of the 42 lines the tree chose quoted boilerplate. The boilerplate patterns were written from these lines, so this row is in-sample. The pooled rows use the XOM filings after the lookup fix.

Every scorer replayed with the boilerplate check (log loss against the options, compared with the baseline):

| Scorer | With boilerplate | Without: moved | Without: all 47 | Without: July | Without: August | Brier, outcome |
| --- | --- | --- | --- | --- | --- | --- |
| Current (the design) | +0.0017 | 18 | −0.0000 | −0.0001 | +0.0001 | 0.657 |
| Model confidence | −0.0006 | 1 | −0.0001 | −0.0001 | 0.0000 | 0.648 |
| Current + confidence | +0.0006 | 8 | +0.0001 | +0.0001 | +0.0001 | 0.657 |
| Pooled + reaction + confidence | +0.0015 | 8 | +0.0005 | +0.0001 | +0.0013 | 0.658 |
| Pooled base rates | +0.0027 | 19 | +0.0008 | +0.0001 | +0.0019 | 0.660 |
| Price reaction | +0.0014 | 13 | +0.0009 | +0.0001 | +0.0024 | 0.662 |
| Pooled + reaction | +0.0033 | 22 | +0.0014 | +0.0009 | +0.0023 | 0.667 |
| Baseline | 0 | 0 | 0 | 0 | 0 | 0.650 |

Removing boilerplate cuts the harm for every scorer, but no scorer beats the baseline in the August check, and the order holds: the more a scorer moves the forecast, the worse it scores. So boilerplate explains why the tree was worse than the baseline, not why it was not better. The real news lines that remain carry nothing the option market had not already priced.

## 5. Results day, three releases (thinking on)

| Release | Next day | Chance options (B3) gave it | Chance MosAIc gave it |
| --- | --- | --- | --- |
| Alphabet, 22 July 2026 | fall (−7.4%) | 37.5% | 35.0% |
| Nvidia, 26 August 2026 | gain (+8.4%) | 34.5% | 49.5% |
| Amazon, 30 July 2026 | gain (+14.3%) | 37.0% | 42.8% |

Average log loss over the three: options 1.013, MosAIc 0.867.
