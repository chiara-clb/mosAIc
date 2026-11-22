# Design

## The forecast

For a stock and a date, MosAIc gives the probability that the return over the next 21 trading days is:

- a **large fall**: below −0.5σ,
- **flat**: between −0.5σ and +0.5σ,
- a **large gain**: above +0.5σ,

where σ is the stock's volatility over the past year, scaled to 21 days. The forecast is a baseline plus a ledger. Each ledger line moves the probabilities by 3, 6 or 9 points, the total movement is capped at 15 points, and every outcome keeps at least 1%.

## The agents

| Agent | Job | Tools | What it sees |
| --- | --- | --- | --- |
| Coordinator | Checks whether results fall within the month, starts the others, reports the baseline if the run fails (code only) | Market data | The request and the earnings calendar |
| Quant | Calculates the baseline from prices (code only) | Market data | Prices before the forecast date |
| Research | Searches filings and news in a loop of up to 4 steps | Document search, news search | A header and a 40-word cleaned extract per result |
| Synthesis | Turns evidence into ledger lines with Tree of Thought | Case memory, branch log | The full passages |
| Critic | Checks quotes, dates and arithmetic, accepts or sends back with a reason | Fetch a passage by its id | The ledger and the passages |

LangGraph runs the flow. Quant and research run at the same time, synthesis waits for both, and the critic's verdict decides the next step. The roles are written as CrewAI role cards, and with `--backend crewai` the model-backed agents run as CrewAI agents. All tools sit behind one interface that checks the caller's role and the forecast date. The same tools can be served by a real MCP server, and a forecast run through it gives the same result as calling them directly.

## Main decisions and why

1. **Start from a baseline and adjust in small steps.** A model asked for probabilities directly gives numbers no one can check. A baseline from prices plus cited shifts can be checked line by line, and the analyst can change any line.
2. **Tree of Thought only where readings compete.** Most steps are linear. The one place where a first reading is often wrong is interpreting evidence: standard legal wording read as bad news, old results read as new. There, the agent compares several readings against base rates and novelty before choosing.
3. **Score readings with data.** How convincing a reading sounds does not count. Its score comes from how often similar past events led to that kind of move (base rate) and whether the passage is new compared with the previous report (novelty).
4. **Separate who reads from who acts.** The research agent can search but sees only cleaned extracts. Synthesis reads full passages but cannot call a search. Instructions hidden in a filing therefore cannot trigger a tool call.
5. **Option prices only for scoring.** They are the market's view, so they are the answer key. Showing them to the agent would make the test meaningless.
6. **Every check is code where possible.** Dates, quotes, arithmetic and tool permissions are checked in code, which is testable and cannot be talked out of a decision.
7. **A human sees the risky cases only.** Reviewing every forecast is slow and reviewers stop reading carefully. The risk gate holds a forecast only when at least two of six warning signs appear (large shift, thin evidence, opposite readings scoring alike, low model confidence, a troubled run, a refused tool call). One sign publishes it with a flag.

## Results day

A second LangGraph flow forecasts the next-day move minutes after an after-close results release (fall, flat or gain, with a large move set at half the typical results-day move). It reuses the tool interface, the risk gate and the analyst review.

| Step | Code or agent | Job |
| --- | --- | --- |
| Coordinator | Code | Checks the request. The world is fixed 15 minutes after the release, with prices up to that day's close |
| Quant | Code | Three baselines: one year of daily moves (B1), past results-day moves (B2), option prices at the close (B3) |
| Fetch release | Code | Gets the press release through the tool interface, which applies the date rule |
| Release reader | Agent (Claude, no tools) | Reads the release against the baselines and the analysts' consensus, and writes ledger lines with exact quotes |
| Critic | Code | Checks every quote against the release, sends failures back with reasons, at most twice, then strikes the lines that still fail |
| Finalise | Code | The ledger tilts B3 in small steps; strong lines count more than weak ones |
| Risk gate | Code | Same warning signs, plus missing consensus. Publishes, flags or holds |

Consensus values (metric, provider, value) are given to the reader. The providers' text and links are not. What happened the next day is added to the log only after the run, marked as never shown to the agents.

## How the design changed during the program

| Module | Change | Why |
| --- | --- | --- |
| 1 | Problem defined: one-month fall, flat or gain, scored against option prices | Option prices give a daily, market-wide answer key |
| 2 | ReAct research loop, critic, case memory, reflection notes | A single model call could not search, check and explain at once |
| 3 | Retrieval over SEC filings: 150-word chunks, MiniLM embeddings, MMR, 30-day window | Evidence must be recent and varied |
| 4 | Tree of Thought for synthesis, "wider spread" effect, scoring by base rate and novelty | The first reading was often wrong |
| 4 (tests) | Baseline version C, every recent filing becomes a candidate, price reaction and GDELT news added, same-story rule | The old baseline was fooled by results-day jumps, searches missed short filings, filings alone gave a partial view |
| 5 | Five agents in LangGraph, checked JSON handoffs, critic sends work back with reasons, at most twice | Feedback without reasons did not converge |
| 6 | Request check, risk gate with publish, flag or hold, analyst review with a feedback log | Make the guardrails and human oversight explicit |
| 7 | Results-day graph: three baselines, a release reader agent, the same critic, risk gate and review | The evaluation showed news matters only when it is new |
| Final | Claude thinks before every answer, and the evaluation was rerun with the same news | Thinking helped the single pass most, so Tree of Thought's value depends on its scorer |

## Known limits

- Public filings may carry little information about the next month that the market has not already priced.
- With small models, the agent rarely moves away from the baseline.
- Tree of Thought scores readings with per-stock base rates, which rest on few past events. With thinking on, a single pass does as well.
- The language models used by the live system may remember 2025 and early 2026 events, so their evaluation uses dates after their knowledge cutoff.
- Results day has been run on three releases as a demonstration only, with shift sizes fixed by hand. It needs an evaluation on many releases, which would also let the shift sizes be learned.
