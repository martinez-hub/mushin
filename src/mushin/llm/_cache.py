"""On-disk output cache keyed by (system, seed, input)."""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any


def _key(value: Any) -> str:
    blob = json.dumps(value, sort_keys=True, default=repr).encode()
    return hashlib.sha256(blob).hexdigest()


def _safe_dir(name: str) -> str:
    """A filesystem-safe directory name for a system, so an odd name (with `/`,
    `..`, etc.) can't escape the cache root. A short hash keeps it unambiguous."""
    slug = re.sub(r"[^A-Za-z0-9_-]", "_", name) or "_"
    return f"{slug}-{hashlib.sha256(name.encode()).hexdigest()[:8]}"


class OutputCache:
    """JSONL-per-(system, seed) cache of system outputs."""

    def __init__(self, root: str | os.PathLike[str]):
        self.root = Path(root)

    def _path(self, system: str, seed: int) -> Path:
        return self.root / _safe_dir(system) / f"seed{seed}.jsonl"

    def _load(self, system: str, seed: int) -> dict[str, Any]:
        path = self._path(system, seed)
        if not path.exists():
            return {}
        out: dict[str, Any] = {}
        for line in path.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                out[rec["key"]] = rec["output"]
        return out

    def partition(self, system: str, seed: int, inputs: list[Any]):
        """Return (cached: dict[index -> output], missing: list[(index, input)])."""
        store = self._load(system, seed)
        cached, missing = {}, []
        for i, inp in enumerate(inputs):
            k = _key(inp)
            if k in store:
                cached[i] = store[k]
            else:
                missing.append((i, inp))
        return cached, missing

    def put_many(self, system: str, seed: int, pairs: list[tuple[Any, Any]]):
        """Append (input, output) pairs to the cache."""
        path = self._path(system, seed)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as f:
            for inp, output in pairs:
                try:
                    record = json.dumps({"key": _key(inp), "output": output})
                except TypeError as e:
                    raise TypeError(
                        f"system {system!r} produced an output that is not "
                        f"JSON-serializable and cannot be cached: {output!r}. "
                        "Return JSON-serializable outputs (e.g. strings) when "
                        "using `cache=`, or call without a cache."
                    ) from e
                f.write(record + "\n")
