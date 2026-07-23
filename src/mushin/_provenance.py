# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
"""Per-run provenance capture (git, versions, config); graceful without git."""

from __future__ import annotations

import json
import platform
import re
import subprocess
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

_PKGS = ("mushin-py", "torch", "numpy", "pytorch-lightning", "hydra-core", "hydra-zen")

_REDACTED = "***REDACTED***"
# Config keys whose values are secrets (redacted regardless of their content).
_REDACT_KEY_RE = re.compile(
    r"(?i)(api[_-]?key|secret|token|password|passwd|credential|auth)"
)
# Secret-shaped values that may hide under an innocent key (provider tokens).
_REDACT_VALUE_RE = re.compile(
    r"(?:sk-|hf_|ghp_|gho_|xox[baprs]-|AKIA)[A-Za-z0-9_-]{8,}"
)


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
    # `dirty` must distinguish "verified clean" (False) from "could not verify"
    # (None): the shared `run` helper maps a failure AND clean-empty output to
    # None, which would record a confident dirty=False for a status call that
    # timed out (plausible on a large repo / slow cluster filesystem).
    try:
        st = subprocess.run(
            ("git", "status", "--porcelain"), capture_output=True, text=True, timeout=5
        )
        dirty = bool(st.stdout.strip()) if st.returncode == 0 else None
    except Exception:  # noqa: BLE001 - unknown, never "clean"
        dirty = None
    return {
        "sha": sha,
        "dirty": dirty,
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


def _apple_chip() -> str | None:
    """The Apple Silicon chip name (e.g. 'Apple M5'), or None off-macOS /
    on failure. Best-effort — provenance must never break a run."""
    try:
        r = subprocess.run(
            ("sysctl", "-n", "machdep.cpu.brand_string"),
            capture_output=True,
            text=True,
            timeout=5,
        )
        return r.stdout.strip() or None
    except Exception:  # noqa: BLE001 - best-effort
        return None


def _accelerator() -> dict:
    """Accelerator identity — the part of GPU numerics the torch wheel version
    alone cannot reconstruct: CUDA/cuDNN + device name on NVIDIA, the MPS
    device on Apple Silicon. All-None on CPU-only builds (and if torch itself
    fails to import)."""
    out: dict = {"cuda": None, "cudnn": None, "device": None}
    try:
        import torch

        out["cuda"] = torch.version.cuda
        try:
            if torch.backends.cudnn.is_available():
                out["cudnn"] = torch.backends.cudnn.version()
        except Exception:  # noqa: BLE001 - best-effort
            pass
        try:
            if torch.cuda.is_available():
                out["device"] = torch.cuda.get_device_name(0)
            elif (
                getattr(torch.backends, "mps", None) is not None
                and torch.backends.mps.is_available()
            ):
                # Apple Silicon: record the MPS backend + chip so an M-series
                # run's provenance is not silently hardware-blind.
                out["device"] = f"mps ({_apple_chip() or 'Apple Silicon'})"
        except Exception:  # noqa: BLE001 - best-effort
            pass
    except Exception:  # noqa: BLE001 - best-effort
        pass
    return out


def capture_base() -> dict:
    """The sweep-constant part of a provenance record: python/platform plus the
    git state, package versions, and accelerator identity. This is identical
    for every cell in a sweep but expensive to compute (``_git()`` spawns three
    git subprocesses), so a sweep captures it ONCE and reuses it for all cells
    rather than paying it per cell (3N git subprocesses for an N-cell sweep)."""
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "git": _git(),
        "packages": _versions(),
        "accelerator": _accelerator(),
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


def _redact_config(obj: Any) -> Any:
    """Redact secret-shaped config values before they are written to disk.

    A value under a secret-named key (``api_key``, ``token``, ...) is redacted
    wholesale; a provider-token-shaped string (``sk-...``, ``hf_...``) is
    redacted even under an innocent key. Structure and non-secret values are
    preserved so the record stays useful.
    """
    if isinstance(obj, dict):
        return {
            k: (_REDACTED if _REDACT_KEY_RE.search(str(k)) else _redact_config(v))
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [_redact_config(v) for v in obj]
    if isinstance(obj, str) and _REDACT_VALUE_RE.search(obj):
        return _REDACTED
    return obj


def _to_plain(cfg):
    try:
        from omegaconf import OmegaConf

        # resolve=False: never evaluate interpolations (e.g. `${oc.env:SECRET}`),
        # which would bake a resolved secret into the on-disk record; the config
        # is recorded as authored. Literal secret values are then redacted.
        plain = OmegaConf.to_container(cfg, resolve=False)
    except Exception:
        return None
    return _redact_config(plain)
