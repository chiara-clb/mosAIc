# Guardrails and coordination on the nine demo scenarios

The model is scripted with planted mistakes (a made-up quote, hidden instructions in a filing, a missing filing, a failing tool, late data, a rare event), so these tables show which failures each setup lets through. They say nothing about forecasting skill. Items right: evidence items given the careful reading, out of 4 per scenario; a line citing words that are not in its source counts as wrong.

## Design against simpler setups

| configuration | accepted | baseline reported | published | flagged | held | items right | made-up quotes published | injected tool calls | late data used | model calls (mean) | est. seconds (mean) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Linear pipeline (one-way, first reading) | 8 | 1 | 6 | 2 | 1 | 14/36 | 2 | 0 | 0 | 13.2 | 30.0 |
| One-way with Tree of Thought | 8 | 1 | 6 | 2 | 1 | 31/36 | 2 | 0 | 0 | 13.2 | 29.5 |
| Critic rejects without reasons | 5 | 4 | 3 | 5 | 1 | 28/36 | 0 | 0 | 0 | 21.2 | 46.9 |
| Critic with reasons (the design) | 7 | 2 | 3 | 5 | 1 | 32/36 | 0 | 0 | 0 | 18.4 | 40.9 |
| Single agent with all tools | 8 | 1 | 9 | 0 | 0 | 31/36 | 2 | 2 | 0 | 13.2 | 29.6 |

## The design with one guardrail switched off

| configuration | accepted | baseline reported | published | flagged | held | items right | made-up quotes published | injected tool calls | late data used | model calls (mean) | est. seconds (mean) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| All guardrails on (the design) | 7 | 2 | 3 | 5 | 1 | 32/36 | 0 | 0 | 0 | 18.4 | 40.9 |
| No critic loop | 8 | 1 | 6 | 2 | 1 | 31/36 | 2 | 0 | 0 | 13.2 | 29.5 |
| No cleaned extracts | 7 | 2 | 3 | 5 | 1 | 32/36 | 0 | 1 | 0 | 18.4 | 40.9 |
| No role check on tools | 7 | 2 | 4 | 4 | 1 | 32/36 | 0 | 1 | 0 | 18.4 | 40.9 |
| No risk gate | 7 | 2 | 9 | 0 | 0 | 32/36 | 0 | 0 | 0 | 18.4 | 40.9 |
| No forecast-date rule | 7 | 2 | 1 | 6 | 2 | 31/36 | 0 | 0 | 8 | 25.6 | 56.7 |
