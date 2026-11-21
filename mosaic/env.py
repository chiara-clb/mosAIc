"""Read settings and keys from a .env file, so they do not have to be typed in every terminal.

The file sits in the folder you run MosAIc from (or the project folder) and holds lines like
    ANTHROPIC_API_KEY=sk-ant-...
    MOSAIC_SEC_USER_AGENT=Your Name your.email@example.com
Variables already set in the environment win. The file is in .gitignore, so it never reaches GitHub.
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_env(text: str) -> dict[str, str]:
    out = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.removeprefix("export ").strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key:
            out[key] = value
    return out


def load_env(paths=None) -> list[str]:
    """Load the first .env found; never overwrite variables already set. Returns the names it set."""
    for p in paths or [Path.cwd() / ".env", PROJECT_ROOT / ".env"]:
        p = Path(p)
        if p.is_file():
            loaded = []
            for k, v in parse_env(p.read_text(encoding="utf-8")).items():
                if k not in os.environ:
                    os.environ[k] = v
                    loaded.append(k)
            return loaded
    return []
