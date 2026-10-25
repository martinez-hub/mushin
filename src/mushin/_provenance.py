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

from ._sweep_io import _atomic_write_text

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
    # git writes ref names and paths in UTF-8 whatever the locale: decoding
    # them in the locale's encoding recorded a non-ASCII branch as mojibake on
    # Windows, or as None where the decode raised. "replace" keeps a stray
    # byte from losing the field.
    utf8 = {"encoding": "utf-8", "errors": "replace"}

    def run(*a):
        try:
            return (
                subprocess.run(a, capture_output=True, timeout=5, **utf8).stdout.strip()
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
            ("git", "status", "--porcelain"), capture_output=True, timeout=5, **utf8
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
    imported torch. With torch loaded, ``cuda`` and ``cudnn`` are the wheel's
    build (None on a CPU-only build) and ``device`` is None until the task
    initializes CUDA, unless MPS is available. Makes no ``torch.cuda`` call
    unless the task already initialized CUDA: the device is named once the
    task has (the cuDNN version is a library constant, read without one)."""
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
            # No torch.cuda call before the task has initialized CUDA itself:
            # is_available() alone initializes the driver and fixes the visible
            # devices, so the next cell's pin_gpu_round_robin in a reused worker
            # would pass its is_initialized() guard and silently pin nothing,
            # and get_device_name would create a context (hundreds of MB).
            # Device 0: the pinned GPU under pin_gpu_round_robin, and the same
            # answer on every rank of a node (current_device differs per rank,
            # and per cell in a reused worker).
            if _cuda_initialized(torch):
                out["device"] = torch.cuda.get_device_name(0)
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


def _cuda_initialized(torch) -> bool:
    try:
        return bool(torch.cuda.is_initialized())
    except Exception:  # noqa: BLE001 - best-effort
        return False


# This process's accelerator, per CUDA state (False: before any task
# initialized CUDA, when no device is named; True: after). Each answer is
# queried once: the MPS branch spawns a sysctl subprocess (about 10 ms, see
# _apple_chip), which a 1000-cell sweep would otherwise pay per cell.
_ACCELERATOR_SEEN: dict[bool, dict] = {}


def _seen_accelerator() -> dict | None:
    """This process's accelerator, from the cache for its current CUDA state;
    ``None`` when torch is not loaded."""
    torch = sys.modules.get("torch")
    if torch is None:
        return None
    state = _cuda_initialized(torch)
    seen = _ACCELERATOR_SEEN.get(state)
    if seen is None:
        seen = _accelerator()
        # With CUDA initialized the device is always named, so None there is
        # a query that failed (a sticky CUDA error, say): not an answer to
        # keep for every later cell of this worker.
        if not (state and seen.get("device") is None):
            _ACCELERATOR_SEEN[state] = seen
    # A copy: the record is the caller's to change, the cache is not.
    return dict(seen)


NOT_WRITTEN = object()  # record_accelerator: the record's first write failed


def record_accelerator(job_dir, written: Any) -> None:
    """Rewrite the ``accelerator`` of a cell's provenance record from the
    process that ran the cell, once its task has returned or raised.

    ``capture_base`` runs in the process that called ``run()``, before the
    first cell. That process is a login node under submitit, and has not
    imported torch when the task does so inside its body (as the GPU-packing
    example does), so its record can miss the device the cell ran on, or name
    one the cell never used. The cell's own process, after its task, knows: a
    process that has not imported torch by then did not use it, and leaves the
    record as written. ``written`` is the accelerator ``write_provenance``
    put in the record before the task: when it is already the answer there is
    nothing to read back, and when that write failed (``NOT_WRITTEN``) there
    is nothing of this cell's to patch, only perhaps a prior sweep's record in
    a reused directory. Best-effort, like every provenance write."""
    seen = _seen_accelerator()
    if seen is None or written is NOT_WRITTEN or written == seen:
        return
    path = Path(job_dir) / "mushin_provenance.json"
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("accelerator") == seen:
            return
        record["accelerator"] = seen
        _atomic_write_text(path, json.dumps(record, indent=2), mkdir=False)
    except Exception:  # noqa: BLE001 - provenance is best-effort
        pass


def representative_record(paths) -> dict | None:
    """The record that stands for a sweep (``wf.provenance``, ``show``,
    ``diff``): the first readable one that has an accelerator, since a first
    cell that failed before its task imported torch has none, else the first
    readable one. The search for a readable record is not bounded (a sampled
    or budget-skipped sweep has no record in the cells it did not run); the
    look for one with an accelerator reads a few more records at most, so a
    sweep without torch is not read whole for a record it does not have."""
    first = None
    readable = 0
    for p in paths:
        try:
            data = json.loads(Path(p).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(data, dict):
            continue
        if isinstance(data.get("accelerator"), dict):
            return data
        if first is None:
            first = data
        readable += 1
        if readable >= 8:
            break
    return first


def capture_base() -> dict:
    """The sweep-constant part of a provenance record: python/platform plus the
    git state and package versions, which are identical for every cell in a
    sweep but expensive to compute (``_git()`` spawns three git subprocesses),
    so a sweep captures it ONCE and reuses it for all cells rather than paying
    it per cell (3N git subprocesses for an N-cell sweep). The accelerator is
    not part of it: it belongs to the process that runs the cell (``capture``,
    then ``record_accelerator`` once the task has run)."""
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "git": _git(),
        "packages": _versions(),
    }


def capture(config: Any = None, base: dict | None = None) -> dict:
    """A full provenance record. ``base`` is the sweep-constant part from
    ``capture_base()``; when omitted it is computed fresh (so a standalone call
    still works). ``timestamp``, ``config`` and the accelerator vary per cell:
    the accelerator is this process's view, never another process's (a base
    from before 0.17 may carry one; it is replaced)."""
    if base is None:
        base = capture_base()
    return {
        "timestamp": datetime.now(UTC).isoformat(),
        **base,
        "accelerator": _seen_accelerator(),
        "config": _to_plain(config) if config is not None else None,
    }


def write_provenance(job_dir, config: Any = None, base: dict | None = None) -> Any:
    """Write the cell's record and return the accelerator it holds, for
    ``record_accelerator`` to compare against after the task."""
    record = capture(config, base)
    # Atomic: under an external multi-rank launch every rank writes the same
    # cell's record, and a reader (show, the MCP server, record_accelerator)
    # must never see a torn file. Plain json.dumps, not the metrics
    # normalizer: a config value of inf stays `Infinity`, as it always was. No
    # mkdir: the cell directory exists, and a write to one that does not
    # raises into the runner's best-effort guard rather than inventing it.
    _atomic_write_text(
        Path(job_dir) / "mushin_provenance.json",
        json.dumps(record, indent=2),
        mkdir=False,
    )
    return record["accelerator"]


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
