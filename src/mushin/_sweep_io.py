# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
"""On-disk primitives for resilient/resumable sweeps: canonical combo keys and
the per-job metrics sidecar + sweep manifest (see the sweep-resilience design)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

METRICS_FILE = "mushin_metrics.json"
MANIFEST_FILE = "mushin_sweep_manifest.json"


def combo_key(combo: dict[str, Any]) -> str:
    """Canonical, order-stable key for a swept-parameter combination."""
    return ",".join(f"{k}={_scalar(combo[k])}" for k in sorted(combo))


def _scalar(v: Any) -> Any:
    """Best-effort convert numpy/torch scalars & arrays to JSON-native values."""
    if hasattr(v, "tolist"):  # numpy scalar/array, torch tensor
        return v.tolist()
    if isinstance(v, (list, tuple)):
        return [_scalar(x) for x in v]
    return v


def write_metrics_sidecar(job_dir, metrics: dict[str, Any]) -> None:
    payload = {k: _scalar(v) for k, v in metrics.items()}
    _atomic_write_json(Path(job_dir) / METRICS_FILE, payload)


def read_metrics_sidecar(job_dir) -> dict | None:
    p = Path(job_dir) / METRICS_FILE
    if not p.exists():
        return None
    # A corrupt/unreadable sidecar is treated like a missing one (return None ->
    # the cell re-runs) rather than aborting an otherwise-resumable sweep.
    try:
        return json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(path)  # atomic on POSIX/Windows


class Manifest:
    """Tracks each requested grid cell's status in <working_dir>/mushin_sweep_manifest.json."""

    SCHEMA = 1

    def __init__(self, root: Path, params: list[str], cells: dict | None = None):
        self.root = Path(root)
        self.params = list(params)
        self.cells: dict[str, dict] = cells or {}

    @classmethod
    def load_or_new(cls, root, params: list[str]) -> Manifest:
        p = Path(root) / MANIFEST_FILE
        if p.exists():
            d = json.loads(p.read_text())
            return cls(root, d.get("params", params), d.get("cells", {}))
        return cls(root, params)

    @classmethod
    def from_cell_status(cls, root, params: list[str]) -> Manifest:
        """Reconstruct a manifest, kill-durably, by scanning per-cell status
        sidecars under ``root/*/`` (each written from inside its own job, so a
        mid-sweep process kill cannot lose completed cells).

        Backward compatible: seeds from the legacy end-of-run manifest first, so a
        sweep dir created before per-cell sidecars existed still resumes; per-cell
        sidecars (when present) are authoritative and overlay the seed."""
        from ._resume import read_cell_status

        root = Path(root)
        # Seed from the legacy manifest (empty if none) -> pre-upgrade sweeps still
        # resume their completed cells.
        m = cls.load_or_new(root, params)
        if not root.exists():
            return m
        for d in root.iterdir():
            if not d.is_dir():
                continue
            s = read_cell_status(d)
            if s is None or "combo" not in s:
                continue
            m.cells[combo_key(s["combo"])] = {
                "dir": d.name,
                "status": s.get("status", "pending"),
            }
        return m

    def status(self, combo: dict) -> str:
        return self.cells.get(combo_key(combo), {}).get("status", "pending")

    def dir(self, combo: dict) -> str | None:
        return self.cells.get(combo_key(combo), {}).get("dir")

    def mark(
        self, combo: dict, *, dir: str, status: str, error: str | None = None
    ) -> None:
        entry = {"dir": str(dir), "status": status}
        if error is not None:
            entry["error"] = error
        self.cells[combo_key(combo)] = entry  # replace in place

    def failed_cells(self) -> list[dict]:
        return [
            {"key": k, **v}
            for k, v in self.cells.items()
            if v.get("status") == "failed"
        ]

    def is_complete(self) -> bool:
        return all(v.get("status") == "completed" for v in self.cells.values())

    def save(self) -> None:
        _atomic_write_json(
            self.root / MANIFEST_FILE,
            {"schema": self.SCHEMA, "params": self.params, "cells": self.cells},
        )
