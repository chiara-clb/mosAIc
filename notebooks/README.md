# Notebooks

Each notebook is written as a `notebook.src` file and built into a `.ipynb`. The `.src` file is easier to read and review. Cells marked `core` hold the logic, and the tests in `tests/notebook/` run them offline.

| Notebook | What it shows | Needs |
| --- | --- | --- |
| `multi_agent/checkpoint5_multi_agent.ipynb` | The whole system on a synthetic company: five agents in LangGraph, tools with role checks, the critic, the risk gate and the analyst review, nine scenarios and a comparison of designs | Nothing for the scripted run. A GPU or an Anthropic API key for the real-model section |
| `tot_validation/checkpoint4_tot_validation.ipynb` | The Tree of Thought step on real SEC filings and GDELT news for AAPL, NKE and INTC | A GPU runtime (Qwen2.5-3B), internet access |
| `baseline_comparison/baseline_comparison.ipynb` | Four versions of the price baseline on 30 stocks over ten years | Internet access for prices and SEC data |

To rebuild a notebook after editing its source:

```bash
python notebooks/build.py notebooks/<name>/notebook.src notebooks/<name>/<file>.ipynb
```
