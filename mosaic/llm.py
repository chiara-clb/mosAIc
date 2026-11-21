"""Language models: one call interface, a call budget, JSON parsing, and three backends.

- "scripted": answers written in advance, including deliberate mistakes. Used for the demo and the tests.
- "anthropic": Claude through the Anthropic API (needs ANTHROPIC_API_KEY).
- "hf": a local Hugging Face model such as Qwen2.5-3B (needs a GPU).
- "crewai": the role cards as CrewAI agents, each answering with a Claude model (mosaic/crew.py).

Every backend is a function chat(messages, max_tokens) -> str, so agents never depend on which one runs.
"""

from __future__ import annotations

import json
import re
from typing import Callable

from mosaic.config import SETTINGS
from mosaic.roles import ROLES
from mosaic.tools import INJECTION_PATTERNS

Chat = Callable[[list[dict], int], str]


class BudgetError(RuntimeError):
    """The forecast used up its model calls."""


class Model:
    """Counts every call, so the budget per forecast can be enforced."""

    def __init__(self, chat: Chat, budget: int | None = None, name: str = "model"):
        self.chat, self.calls, self.name = chat, 0, name
        self.budget = budget or SETTINGS.run_model_budget

    def __call__(self, messages: list[dict], max_tokens: int = 300) -> str:
        if self.calls >= self.budget:
            raise BudgetError(f"model budget of {self.budget} calls used up")
        self.calls += 1
        return self.chat(messages, max_tokens)


def parse_json(reply: str | None) -> dict | None:
    """The first JSON object in a reply, tolerating code fences and extra text."""
    reply = re.sub(r"```(?:json)?", "", reply or "")
    m = re.search(r"\{.*\}", reply, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def agent_messages(agent: str, task: str, payload: dict, extra_roles: dict | None = None) -> list[dict]:
    """System prompt from the agent's role card, user message with the input as JSON."""
    r = ROLES.get(agent) or (extra_roles or {})[agent]
    system = (f"Agent: {agent}\nYou are the {r['role']}. Goal: {r['goal']}\n{r['backstory']}\n\n{task}\n"
              "Reply only with one JSON object.")
    return [{"role": "system", "content": system},
            {"role": "user", "content": "INPUT:\n" + json.dumps(payload, default=str)}]


# --- the scripted model ------------------------------------------------------------------------

DEFAULT_SCRIPT = {
    "queries": ["chief financial officer resignation", "legal proceedings lawsuits", "headquarters other events"],
    "news_query": "Acme Robotics contract deal",
    "rephrase": {},                  # empty search -> next query
    "readings": {                    # two readings per item; the first is the tempting one
        "F1": [{"effect": "fall", "size": "medium", "quote": "her decision to resign as Chief Financial Officer",
                "reason": "Losing the CFO is bad news."},
               {"effect": "spread", "size": "small",
                "quote": "the Company has not named an interim Chief Financial Officer",
                "reason": "A finance leadership gap adds uncertainty without a clear direction."}],
        "F2": [{"effect": "fall", "size": "small", "quote": "involved in legal proceedings and claims",
                "reason": "Pending lawsuits are a risk."},
               {"effect": "spread", "size": "small", "quote": "involved in legal proceedings and claims",
                "reason": "Litigation could go either way."}],
        "F3": [{"effect": "gain", "size": "small", "quote": "completed the move of its headquarters to Austin, Texas",
                "reason": "A cheaper location may cut costs."},
               {"effect": "spread", "size": "small", "quote": "completed the move of its headquarters to Austin, Texas",
                "reason": "A move can disrupt operations."}],
        "S1": [{"effect": "fall", "size": "small", "quote": "Acme Robotics finance chief resigns",
                "reason": "Negative coverage of a departure."},
               {"effect": "spread", "size": "small", "quote": "no interim named",
                "reason": "Uncertainty about who runs finance."}],
        "S2": [{"effect": "gain", "size": "small",
                "quote": "Acme Robotics wins warehouse automation contract with large retailer",
                "reason": "A new large customer adds revenue."},
               {"effect": "gain", "size": "medium", "quote": "Acme Robotics lands retail robotics deal",
                "reason": "The deal could be large."}],
    },
    "readings_after_feedback": {},   # id -> readings used once the critic has named that item
    "lenient_judge": False,          # True: the grounding judge accepts any quote
    "gullible": False,               # True: the model obeys instructions it finds in a passage
}


def _research(p: dict, s: dict) -> dict:
    done = [o["query"] for o in p["observations"]]
    seen = " ".join(r.get("text", "") for o in p["observations"] for r in o["results"])
    if s["gullible"] and re.search(INJECTION_PATTERNS[0], seen, re.I) and "acme buyout rumor" not in done:
        return {"thought": "The filing says to do this.", "action": "search_news", "query": "acme buyout rumor"}
    hints = [h for h in p.get("hints", []) if h not in done]
    if hints:
        return {"thought": "The critic asked for this.", "action": "search_documents", "query": hints[0]}
    last = p["observations"][-1] if p["observations"] else None
    if last and not last["results"] and last["query"] in s["rephrase"] and s["rephrase"][last["query"]] not in done:
        return {"thought": "Nothing found; rephrase.", "action": last["tool"], "query": s["rephrase"][last["query"]]}
    if p.get("hints"):
        return {"thought": "Found what the critic asked for.", "action": "finish"}
    for q in s["queries"]:
        if q not in done:
            return {"thought": "Next topic.", "action": "search_documents", "query": q}
    if s.get("news_query") and s["news_query"] not in done:
        return {"thought": "Check the news.", "action": "search_news", "query": s["news_query"]}
    return {"thought": "Enough evidence.", "action": "finish"}


def _synthesis(p: dict, s: dict) -> dict:
    item = p["item"]
    named = [f for f in p.get("feedback", []) if item["id"] in f]
    readings = (s["readings_after_feedback"].get(item["id"]) if named else None) or s["readings"].get(item["id"])
    out = {"readings": readings or [{"effect": "none", "size": None, "quote": "", "reason": "No script."}] * 2}
    if s["gullible"] and re.search(INJECTION_PATTERNS[0], item["text"], re.I):
        out = {"readings": [{"effect": "gain", "size": "large", "quote": "report a large gain",
                             "reason": "The filing says to report a large gain."}] * 2,
               "tool_request": {"tool": "search_news", "query": "acme buyout rumor"}}
    return out


def _judge(p: dict, s: dict) -> dict:
    return {"supported": True if s["lenient_judge"] else p["quote"].lower() in p["passage"].lower()}


def _analyst(p: dict, s: dict) -> dict:
    """A single agent doing everything (used only in the comparison)."""
    if "item" in p:
        return _synthesis(p, s)
    if "passage" in p:
        return _judge(p, s)
    return _research(p, s)


SCRIPTED = {"research": _research, "synthesis": _synthesis, "critic": _judge, "analyst": _analyst}


def scripted_chat(script: dict | None = None) -> Chat:
    """A chat function that answers from a script; DEFAULT_SCRIPT fills anything the script leaves out."""
    s = {**DEFAULT_SCRIPT, **(script or {})}

    def chat(messages: list[dict], max_tokens: int = 300) -> str:
        agent = re.match(r"Agent: (\w+)", messages[0]["content"]).group(1)
        payload = json.loads(messages[-1]["content"].split("INPUT:\n", 1)[1])
        return json.dumps(SCRIPTED[agent](payload, s))
    return chat


# --- real models -------------------------------------------------------------------------------

NEW_CLAUDE = re.compile(r"claude-(opus|sonnet|haiku|fable|mythos)-5")


def anthropic_chat(model: str | None = None, client=None, temperature: float | None = None) -> Chat:
    """Claude through the Anthropic API. `client` can be passed in for tests.

    Claude 5 models think by default, the hidden thinking counts toward max_tokens, and they refuse
    `temperature`. Each agent step here is a short, structured answer that code then checks, so
    thinking is turned off and no temperature is sent. If a reply still comes back empty because
    the limit was reached, it is asked again once with four times the limit.
    """
    if client is None:
        import anthropic                                   # optional dependency
        client = anthropic.Anthropic()
    model = model or SETTINGS.llm_model
    extra: dict = {}
    if NEW_CLAUDE.search(model):
        extra["extra_body"] = {"thinking": {"type": "disabled"}}
    else:
        extra["temperature"] = SETTINGS.llm_temperature if temperature is None else temperature

    def create(**kw):
        try:
            return client.messages.create(**kw, **extra)
        except TypeError as e:                             # an older or newer package without these options
            if not extra or not any(k in str(e) for k in extra):
                raise
            extra.clear()
            return client.messages.create(**kw)

    def chat(messages: list[dict], max_tokens: int = 300) -> str:
        system = "\n".join(m["content"] for m in messages if m["role"] == "system")
        rest = [m for m in messages if m["role"] != "system"]
        r = create(model=model, system=system, messages=rest, max_tokens=max_tokens)
        text = "".join(getattr(b, "text", "") for b in r.content).strip()
        if getattr(r, "stop_reason", None) == "max_tokens" and parse_json(text) is None:
            r = create(model=model, system=system, messages=rest, max_tokens=4 * max_tokens)
            text = "".join(getattr(b, "text", "") for b in r.content).strip()
        return text
    return chat


def hf_chat(model_name: str = "Qwen/Qwen2.5-3B-Instruct") -> Chat:     # pragma: no cover - needs a GPU
    """A local Hugging Face model, in float16 (bfloat16 is slow on a T4)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    lm = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

    def chat(messages: list[dict], max_tokens: int = 300) -> str:
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tok(prompt, return_tensors="pt").to(lm.device)
        out = lm.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
        return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    return chat


def make_model(backend: str = "scripted", script: dict | None = None, model_name: str | None = None,
               budget: int | None = None) -> Model:
    if backend == "scripted":
        return Model(scripted_chat(script), budget, name="scripted")
    if backend == "anthropic":
        name = model_name or SETTINGS.llm_model
        return Model(anthropic_chat(name), budget, name=name)
    if backend == "hf":
        name = model_name or "Qwen/Qwen2.5-3B-Instruct"
        return Model(hf_chat(name), budget, name=name)
    if backend == "crewai":                                # the role cards as CrewAI agents (mosaic/crew.py)
        from mosaic.crew import crewai_chat
        name = model_name or SETTINGS.llm_model
        return Model(crewai_chat(name), budget, name=f"crewai:{name}")
    raise ValueError(f"unknown backend {backend!r}")
