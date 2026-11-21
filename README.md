# MosAIc

MosAIc is a multi-agent system that estimates how a large US stock is likely to move over the next month, and explains its estimate line by line.

For one of 30 large-cap stocks and a date, it gives three probabilities: a large fall, a roughly flat month, or a large gain over the next 21 trading days. It starts from a baseline calculated from price history. It then reads the company's recent SEC filings and news, and adds a short ledger: each line shifts the probabilities by a small amount and quotes the passage it relies on.

The name comes from the "mosaic theory" of equity research, where analysts build a view from many small pieces of public information.

Capstone project for the CMU Agentic AI program, 2026.

## Who it is for

An equity analyst who wants a quick, checkable view of a stock before results or after news. Every number can be traced to a source, and the analyst can keep, resize or strike any line of the ledger.

## How a forecast is made

![MosAIc architecture](docs/figures/design_proposed.png)

1. **Coordinator** checks whether results are due within the month.
2. **Quant agent** calculates the baseline from one year of prices. This is plain code.
3. **Research agent** searches the company's filings and news headlines, rephrasing searches that find nothing. It sees only short, cleaned extracts.
4. **Synthesis agent** turns the evidence into ledger lines. It uses Tree of Thought: for each piece of evidence it compares several readings (fall, gain, wider spread, no effect) against what happened after similar past events, and keeps the best combination.
5. **Critic** checks that every quote appears in its source, that every source was published before the forecast date, and that the numbers add up. A failed check goes back with its reason, at most twice. After that the system reports the baseline and says why.
6. **Risk gate** publishes the forecast, flags it, or holds it for the analyst, depending on how large the shift is, how much evidence supports it and whether the run had problems.

Every tool returns only data published before the forecast date, and every step is logged.

## How it is evaluated

The main test is how close the forecast gets to the probabilities implied by option prices on the forecast date. Option prices are used only for scoring and are never shown to the agent. Forecasts are also compared with what actually happened, and with simpler baselines (price history only, a prompt-only model, the agent without Tree of Thought).

Results so far are in [docs/results.md](docs/results.md).

## What is in this repository

| Path | What it is |
| --- | --- |
| `mosaic/` | The Python package: the agents, tools, graph, guardrails, risk gate, demo world and scenarios, the option-based answer key and scoring, and the command-line tool |
| `notebooks/` | Three tested notebooks: the Tree of Thought step on real filings, the baseline comparison, and the full multi-agent system with its guardrails. See [notebooks/README.md](notebooks/README.md) |
| `tests/` | Tests for the package and for the notebooks' code |
| `docs/design.md` | The design, the main decisions and how they changed during the program |
| `docs/results.md` | What the notebooks showed |
| `docs/figures/` | Architecture diagrams |
| `results/` | The guardrail ablation and design comparison on the demo scenarios, and the evaluation on real stocks once it has run |

The notebooks are where each part was first tested. The package holds the final version of the same code.

## Running it

**From the command line** (Python 3.10 or later):

```bash
pip install -e .
python -m mosaic.cli --demo                         # one forecast on the demo company
python -m mosaic.cli --demo --scenario all          # the nine guardrail scenarios
python -m mosaic.cli --demo --backend anthropic     # the demo company with Claude (needs ANTHROPIC_API_KEY)
```

Each forecast writes `report.md` (probabilities, ledger, review decision), `state.json` and `trace.jsonl` (every step and tool call, with the leakage audit) under `runs/`.

**A real stock** needs internet access, a few extra packages, a contact for the SEC (its rule for automated access) and, for Claude, an Anthropic API key. Put the last two in a `.env` file once (copy `.env.example`; git ignores `.env`, so it never reaches GitHub):

```bash
pip install -e ".[retrieval,prices,llm]"
cp .env.example .env                               # then fill in ANTHROPIC_API_KEY and MOSAIC_SEC_USER_AGENT
python -m mosaic.cli AAPL --as-of 2026-07-15 --backend anthropic --model claude-haiku-5-5
```

Filings, prices and news are cached under `data/`, so a second run on the same stock is fast.

**Other ways to run the agents.** `--backend crewai` runs the model-backed agents as CrewAI agents (`pip install -e ".[crew]"`). The tools can also be served as a real MCP server, which any MCP client can call; the forecast-date rule runs inside each tool:

```bash
pip install -e ".[mcp]"
python -m mosaic.mcp_server --demo                  # or --ticker AAPL --as-of 2026-07-15
```

**The guardrail ablation** reruns the nine scenarios with each guardrail switched off in turn, and compares the design with simpler setups. It needs no internet and writes the tables in `results/`:

```bash
python -m mosaic.ablation
```

**The evaluation on real stocks** forecasts every Wednesday from July 2026 (after the Claude models' knowledge cutoff) with five variants: baseline, prompt only, baseline with the model's memory, single pass and Tree of Thought. It first asks the model whether it remembers what the stock did, and reports scores with and without those dates. With `--options-dir` pointing at a folder of FirstRate option files, the main score is log loss against the option-implied probabilities:

```bash
python -m mosaic.evaluate --backend anthropic --model claude-haiku-5-5 \
  --tickers AAPL,NKE,INTC,MSFT,JPM,XOM --start 2026-07-01 --end 2026-10-15 \
  --options-dir data/firstrate --resume
```

Add `--limit 2` for a cheap first try. Results go to `results/evaluation/`, and `--resume` continues an interrupted run.

**Results day (a first step towards forecasting right after results).** For a stock that reports after the close, three baselines for the move to the next day's close, all known before the release: one year of daily moves (B1), the stock's last eight results-day moves (B2), and the option prices at the close (B3, from the FirstRate files). No model is called, and the actual move is shown next to them:

```bash
python -m mosaic.results_day AMZN 2026-07-30 --options-dir data/firstrate
```

It is also the third choice in the app's sidebar. The option code is ported from an earnings-release prototype, in which an agent that reads the release itself scored better than all three baselines on 40 releases (see the report).

**In the browser** (a simple Streamlit screen):

```bash
pip install -e ".[ui]"
streamlit run app.py
```

Choose the demo company and a scenario (or a real stock), press **Run forecast**, and see the forecast against the baseline, the ledger with its sources and dates, the risk gate's decision, and how the agents got there: evidence, every Tree of Thought reading with its scores, each step and each tool call. A held forecast can be reviewed line by line (keep, resize or strike).

**In a notebook, no setup needed:** open `notebooks/multi_agent/checkpoint5_multi_agent.ipynb` in Google Colab and choose Run all. It runs in seconds on a small synthetic company and shows nine scenarios, including the guardrails at work. To use a real model, set `MODE = "real"` in section 1 (needs a GPU runtime for Qwen, or an Anthropic API key).

**The tests, locally:**

```bash
pip install -e ".[dev]"
python -m pytest
```

The notebooks are built from their `notebook.src` files, which are easier to review and diff:

```bash
python notebooks/build.py notebooks/multi_agent/notebook.src notebooks/multi_agent/checkpoint5_multi_agent.ipynb
```

## Data

All data is public: SEC EDGAR filings, GDELT news headlines and daily prices. Option prices for scoring come from FirstRate Data and are licensed, so they are not included in this repository. `mosaic/firstrate.py` reads their files as they come (one file per stock and quarter, plain or zipped). The files have no stock price column, so the price is taken from the daily closes or, failing that, worked out from the option prices themselves (put-call parity).
