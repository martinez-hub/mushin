# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
"""Per-run provenance capture (git, versions, config); graceful without git."""

from __future__ import annotations

import json
import platform
import subprocess
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

_PKGS = ("mushin-py", "torch", "numpy", "pytorch-lightning", "hydra-core", "hydra-zen")


def _git() -> dict:
    def run(*a):
        try:
            return (
                subprocess.run(
                    a, capture_output=True, text=True, timeout=5
                ).stdout.strip()
                or None
            )
        except Exception:
            return None

    sha = run("git", "rev-parse", "HEAD")
    if sha is None:
        return {"sha": None, "dirty": None, "branch": None}
    return {
        "sha": sha,
        "dirty": bool(run("git", "status", "--porcelain")),
        "branch": run("git", "rev-parse", "--abbrev-ref", "HEAD"),
    }


def _versions() -> dict:
    out = {}
    for p in _PKGS:
        try:
            out[p] = version(p)
        except PackageNotFoundError:
            out[p] = None
    return out


def capture_base() -> dict:
    """The sweep-constant part of a provenance record: python/platform plus the
    git state and package versions. This is identical for every cell in a sweep
    but expensive to compute (``_git()`` spawns three git subprocesses), so a
    sweep captures it ONCE and reuses it for all cells rather than paying it per
    cell (3N git subprocesses for an N-cell sweep)."""
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "git": _git(),
        "packages": _versions(),
    }


def capture(config: Any = None, base: dict | None = None) -> dict:
    """A full provenance record. ``base`` is the sweep-constant part from
    ``capture_base()``; when omitted it is computed fresh (so a standalone call
    still works). Only ``timestamp`` and ``config`` vary per cell."""
    if base is None:
        base = capture_base()
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **base,
        "config": _to_plain(config) if config is not None else None,
    }


def write_provenance(job_dir, config: Any = None, base: dict | None = None) -> None:
    (Path(job_dir) / "mushin_provenance.json").write_text(
        json.dumps(capture(config, base), indent=2)
    )


def _to_plain(cfg):
    try:
        from omegaconf import OmegaConf

        return OmegaConf.to_container(cfg, resolve=True)
    except Exception:
        return None
