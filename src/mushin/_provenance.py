# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
"""Per-run provenance capture (git, versions, config); graceful without git."""

from __future__ import annotations

import json
import platform
import re
import subprocess
import sys
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from ._sweep_io import _atomic_write_json

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
    """The Apple Silicon chip name (e.g. 'Apple M5'), or None off-macOS / on
    failure / on a non-Apple CPU (Intel Macs can expose MPS via a Metal GPU;
    their x86 brand string must not be recorded as the chip). The lookup is
    best-effort: provenance must never break a run."""
    try:
        r = subprocess.run(
            ("sysctl", "-n", "machdep.cpu.brand_string"),
            capture_output=True,
            text=True,
            timeout=5,
        )
        s = r.stdout.strip()
        return s if r.returncode == 0 and s.startswith("Apple") else None
    except Exception:  # noqa: BLE001 - best-effort
        return None


_NO_ACCELERATOR: dict = {"cuda": None, "cudnn": None, "device": None}


def _accelerator() -> dict | None:
    """Accelerator identity: the part of GPU numerics the torch wheel version
    alone cannot reconstruct: CUDA/cuDNN build + device name on NVIDIA, the MPS
    device on Apple Silicon. ``None`` (not recorded) in a process that has not
    imported torch; all-None values when torch is loaded but found no device.
    Never initializes CUDA itself: the device is named once the task has."""
    torch = sys.modules.get("torch")
    if torch is None:
        # Never import torch here. Importing it is the dominant cost of a
        # provenance capture, and a process that has not loaded torch has no
        # accelerator numerics to record: a sweep that never uses torch (a
        # scikit-learn comparison, say) would pay the import to record None.
        # Each cell's record is taken from the process that ran it, after its
        # task returned (`record_accelerator`), when a task that uses torch has
        # imported it, wherever the import statement sits.
        return None
    out = dict(_NO_ACCELERATOR)
    try:
        out["cuda"] = torch.version.cuda
        try:
            if torch.backends.cudnn.is_available():
                out["cudnn"] = torch.backends.cudnn.version()
        except Exception:  # noqa: BLE001 - best-effort
            pass
        try:
            if torch.cuda.is_available():
                # get_device_name initializes CUDA. In a worker whose task never
                # touched the GPU that would create a context (hundreds of MB)
                # and make the next cell's pin_gpu_round_robin raise, so the
                # name is read only once the task has initialized CUDA itself.
                if torch.cuda.is_initialized():
                    out["device"] = torch.cuda.get_device_name(
                        torch.cuda.current_device()
                    )
            elif (
                getattr(torch.backends, "mps", None) is not None
                and torch.backends.mps.is_available()
            ):
                # MPS: record the backend + the real Apple chip when known, or
                # the CPU architecture otherwise — never a hardcoded "Apple
                # Silicon" claim (Intel Macs can expose MPS via a Metal GPU).
                out["device"] = f"mps ({_apple_chip() or platform.machine()})"
        except Exception:  # noqa: BLE001 - best-effort
            pass
    except Exception:  # noqa: BLE001 - best-effort: torch may be mid-import
        pass
    return out


def _accelerator_settled() -> bool:
    """Whether this process's accelerator answer can no longer change. A CUDA
    device is named only once CUDA is initialized, which a later cell's task
    may be the first to do."""
    torch = sys.modules.get("torch")
    try:
        return not torch.cuda.is_available() or torch.cuda.is_initialized()
    except Exception:  # noqa: BLE001 - best-effort
        return False


# The accelerator of this process once the answer is settled. The query costs
# about 10 ms, which a 1000-cell sweep would otherwise pay per cell.
_ACCELERATOR_SEEN: dict | None = None


def record_accelerator(job_dir) -> None:
    """Rewrite the ``accelerator`` of a cell's provenance record from the
    process that ran the cell, once its task has returned or raised.

    ``capture_base`` runs in the process that called ``run()``, before the
    first cell. That process is a login node under submitit, and has not
    imported torch when the task does so inside its body (as the GPU-packing
    example does), so its record can miss the device the cell ran on, or name
    one the cell never used. The cell's own process, after its task, knows: a
    process that has not imported torch by then did not use it, and leaves the
    record as written. Best-effort, like every provenance write."""
    global _ACCELERATOR_SEEN
    if sys.modules.get("torch") is None:
        return
    path = Path(job_dir) / "mushin_provenance.json"
    try:
        seen = _ACCELERATOR_SEEN
        if seen is None:
            seen = _accelerator()
            if _accelerator_settled():
                _ACCELERATOR_SEEN = seen
        record = json.loads(path.read_text())
        if record.get("accelerator") == seen:
            return
        record["accelerator"] = seen
        _atomic_write_json(path, record)
    except Exception:  # noqa: BLE001 - provenance is best-effort
        pass


def capture_base() -> dict:
    """The sweep-constant part of a provenance record: python/platform plus the
    git state and package versions, which are identical for every cell in a
    sweep but expensive to compute (``_git()`` spawns three git subprocesses),
    so a sweep captures it ONCE and reuses it for all cells rather than paying
    it per cell (3N git subprocesses for an N-cell sweep). The accelerator in
    it is this process's view, a placeholder that ``record_accelerator``
    replaces in each cell once its task has run."""
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
    still works, with this process's view of the accelerator). Only
    ``timestamp`` and ``config`` vary per cell."""
    if base is None:
        base = capture_base()
    if _ACCELERATOR_SEEN is not None:
        # Known from an earlier cell of this process: the record is right at
        # its first write, and record_accelerator has nothing to rewrite.
        base = {**base, "accelerator": _ACCELERATOR_SEEN}
    return {
        "timestamp": datetime.now(UTC).isoformat(),
        **base,
        "config": _to_plain(config) if config is not None else None,
    }


def write_provenance(job_dir, config: Any = None, base: dict | None = None) -> None:
    # Atomic: under an external multi-rank launch every rank writes the same
    # cell's record, and a reader (show, the MCP server, record_accelerator)
    # must never see a torn file.
    _atomic_write_json(Path(job_dir) / "mushin_provenance.json", capture(config, base))


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
