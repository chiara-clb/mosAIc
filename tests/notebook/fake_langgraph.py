"""A small stand-in for LangGraph's StateGraph, so the notebook's graph can be tested offline.

It follows LangGraph's execution model: nodes active in the same step run on the same state;
fields annotated with a reducer merge, others are overwritten (two writes in one step are an error);
an edge from a list of nodes waits for all of them; conditional edges are evaluated after the step.
"""

from typing import get_type_hints

START, END = "__start__", "__end__"


class InvalidUpdateError(RuntimeError):
    pass


class StateGraph:
    def __init__(self, schema):
        hints = get_type_hints(schema, include_extras=True)
        self.reducers = {k: t.__metadata__[0] for k, t in hints.items() if hasattr(t, "__metadata__")}
        self.nodes, self.edges, self.joins, self.cond = {}, [], [], {}

    def add_node(self, name, fn):
        self.nodes[name] = fn

    def add_edge(self, src, dst):
        if isinstance(src, (list, tuple)):
            self.joins.append((tuple(src), dst))
        else:
            self.edges.append((src, dst))

    def add_conditional_edges(self, src, fn, mapping):
        self.cond[src] = (fn, mapping)

    def compile(self):
        return Compiled(self)


class Compiled:
    def __init__(self, g, recursion_limit=25):
        self.g, self.limit, self.steps = g, recursion_limit, []

    def invoke(self, state, config=None):
        g, state = self.g, dict(state)
        limit = (config or {}).get("recursion_limit", self.limit)
        active = [d for s, d in g.edges if s == START]
        seen = [set() for _ in g.joins]
        self.steps = []
        while active:
            if len(self.steps) >= limit:
                raise RecursionError("recursion limit reached")
            self.steps.append(list(active))
            snapshot = dict(state)
            updates = [(n, g.nodes[n](dict(snapshot)) or {}) for n in active]
            written = {}
            for n, u in updates:
                for k, v in u.items():
                    if k in g.reducers:
                        state[k] = g.reducers[k](state.get(k, []), v)
                    else:
                        if k in written:
                            raise InvalidUpdateError(f"{k} written by {written[k]} and {n} in one step")
                        written[k] = n
                        state[k] = v
            nxt = []
            for n, _ in updates:
                nxt += [d for s, d in g.edges if s == n]
                if n in g.cond:
                    fn, mapping = g.cond[n]
                    nxt.append(mapping[fn(state)])
                for j, (srcs, d) in enumerate(g.joins):
                    if n in srcs:
                        seen[j].add(n)
                        if seen[j] == set(srcs):
                            nxt.append(d)
                            seen[j] = set()
            active = [d for d in dict.fromkeys(nxt) if d != END]
        return state

    def get_graph(self):
        return _Drawing(self.g)


class _Drawing:
    """Text drawing only; the real LangGraph also renders a PNG."""

    def __init__(self, g):
        self.g = g

    def draw_mermaid(self):
        lines = ["graph TD"] + [f"  {s} --> {d}" for s, d in self.g.edges]
        lines += [f"  {s} --> {d}" for srcs, d in self.g.joins for s in srcs]
        lines += [f"  {s} -.->|{k}| {d}" for s, (_, m) in self.g.cond.items() for k, d in m.items()]
        return "\n".join(lines)

    def draw_mermaid_png(self):
        raise RuntimeError("no renderer offline")
