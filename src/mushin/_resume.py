# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
"""Primitives for kill-durable, resumable sweeps: the per-cell status sidecar,
best-effort checkpoint discovery, and the ResumeContext handed to a task that
opts in via a ``mushin_resume`` parameter (see the resume-hardening design)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ._sweep_io import _atomic_write_json

STATUS_FILE = "mushin_cell_status.json"


@dataclass(frozen=True)
class ResumeContext:
    """Handed to a task that declares a ``mushin_resume`` parameter.

    ``dir`` is the cell's working directory (already the cwd when the task runs).
    ``is_resume`` is True when a prior attempt of the SAME combo left artifacts
    here. ``last_ckpt`` is the newest checkpoint in ``dir`` (or None). ``attempt``
    is 1 on the first run and increments on each re-execution of this combo."""

    dir: Path | None
    is_resume: bool
    last_ckpt: Path | None
    attempt: int


def write_cell_status(
    cell_dir, *, status: str, combo: dict[str, Any], attempt: int
) -> None:
    """Atomically write this cell's status sidecar into its own dir."""
    _atomic_write_json(
        Path(cell_dir) / STATUS_FILE,
        {"status": status, "combo": combo, "attempt": int(attempt)},
    )


def read_cell_status(cell_dir) -> dict | None:
    """Read a cell status sidecar; a missing/corrupt one reads as None."""
    p = Path(cell_dir) / STATUS_FILE
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def discover_last_ckpt(cell_dir) -> Path | None:
    """Best-effort: the checkpoint a resuming task should load. Prefers an exact
    ``last.ckpt``; otherwise the most-recently-modified ``*.ckpt`` in ``cell_dir``.
    Returns None if there is none."""
    d = Path(cell_dir)
    exact = d / "last.ckpt"
    if exact.exists():
        return exact
    ckpts = [p for p in d.glob("*.ckpt") if p.is_file()]
    if not ckpts:
        return None
    return max(ckpts, key=lambda p: p.stat().st_mtime)
