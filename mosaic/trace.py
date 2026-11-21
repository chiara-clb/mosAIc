"""Run trace and leakage audit.

Every forecast writes a trace: one event per step (which agent ran, what it called, what it used).
The leakage audit reads the trace and lists every source used that was published on or after the
start of the forecast day. A clean run has none.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _iso(value: Any) -> Any:
    return value.isoformat() if isinstance(value, datetime) else value


@dataclass
class Trace:
    """An append-only record of one forecast run."""

    run_id: str
    as_of: datetime
    events: list[dict] = field(default_factory=list)
    listeners: list = field(default_factory=list, repr=False, compare=False)   # called with each event, live

    def add(self, node: str, kind: str, **fields: Any) -> dict:
        """Record one event, e.g. add("research", "tool_call", tool="search_documents", status="ok")."""
        event = {"step": len(self.events), "node": node, "kind": kind,
                 "at": datetime.now(timezone.utc).isoformat(), **fields}
        self.events.append(event)
        self._tell(event)
        return event

    def announce(self, node: str, message: str) -> None:
        """Tell the listeners what is happening now, without adding an event to the record."""
        self._tell({"node": node, "kind": "progress", "message": message})

    def _tell(self, event: dict) -> None:
        for listen in self.listeners:
            try:
                listen(event)
            except Exception:              # a broken display must never stop a forecast
                pass

    def used(self, node: str, source_id: str, published: datetime, **fields: Any) -> dict:
        """Record that a step relied on a source, with the time the source became public."""
        return self.add(node, "source_used", source=source_id, published=_iso(published), **fields)

    def of_kind(self, kind: str) -> list[dict]:
        return [e for e in self.events if e["kind"] == kind]

    def to_dict(self) -> dict:
        return {"run_id": self.run_id, "as_of": _iso(self.as_of),
                "events": [{k: _iso(v) for k, v in e.items()} for e in self.events]}

    def write_jsonl(self, path: str | Path) -> Path:
        """One header line, then one line per event."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        d = self.to_dict()
        with path.open("w", encoding="utf-8") as f:
            f.write(json.dumps({"run_id": d["run_id"], "as_of": d["as_of"]}) + "\n")
            for e in d["events"]:
                f.write(json.dumps(e, default=str) + "\n")
        return path

    @classmethod
    def read_jsonl(cls, path: str | Path) -> "Trace":
        lines = Path(path).read_text(encoding="utf-8").splitlines()
        head = json.loads(lines[0])
        return cls(run_id=head["run_id"], as_of=datetime.fromisoformat(head["as_of"]),
                   events=[json.loads(line) for line in lines[1:] if line.strip()])


def leakage_audit(trace: Trace) -> list[dict]:
    """Every source used that was published at or after the forecast moment (start of the forecast day)."""
    found = []
    for e in trace.of_kind("source_used"):
        published = e["published"]
        when = datetime.fromisoformat(published) if isinstance(published, str) else published
        if when.tzinfo is None:
            when = when.replace(tzinfo=timezone.utc)
        if when >= trace.as_of:
            found.append({"step": e["step"], "node": e["node"], "source": e["source"], "published": _iso(when)})
    return found
