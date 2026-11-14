# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
"""On-disk primitives for resilient/resumable sweeps: canonical combo keys and
the per-job metrics sidecar + sweep manifest (see the sweep-resilience design)."""

from __future__ import annotations

import json
import os
import sys
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

METRICS_FILE = "mushin_metrics.json"
MANIFEST_FILE = "mushin_sweep_manifest.json"


def decode_legacy(
    data: bytes, *, encoding: str = "utf-8", errors: str = "strict"
) -> str:
    """Text from a file that mushin, a user or an earlier mushin release
    wrote: ``encoding`` first (UTF-8, which mushin now writes on every
    platform), then the machine's locale encoding, which earlier releases
    wrote and which Windows tools write by default (cp1252 on Western
    systems). ``errors`` applies to that fallback only.

    The locale encoding is ``locale.getencoding()``, not
    ``getpreferredencoding()``: in UTF-8 mode (``PYTHONUTF8=1``, which the
    agent guide recommends on Windows, and the default from Python 3.15) the
    latter says "utf-8", so a cp1252 file would fail UTF-8 twice."""
    import locale

    try:
        return data.decode(encoding)
    except UnicodeDecodeError:
        return data.decode(locale.getencoding(), errors)


def _esc(s: Any) -> str:
    """Escape the ``combo_key`` delimiters in a key or value so the join is
    injective: a value literally containing ``,`` or ``=`` cannot be confused
    with a separator. Clean values are unchanged (backward compatible)."""
    return str(s).replace("%", "%25").replace("=", "%3D").replace(",", "%2C")


def combo_key(combo: dict[str, Any]) -> str:
    """Canonical, order-stable, injective key for a swept-parameter combination.

    Delimiters inside keys/values are percent-escaped so that, e.g.,
    ``{'a': '1,b=2'}`` and ``{'a': 1, 'b': 2}`` do not collide.
    """
    return ",".join(f"{_esc(k)}={_esc(_scalar(combo[k]))}" for k in sorted(combo))


def _scalar(v: Any) -> Any:
    """Best-effort convert numpy/torch scalars & arrays to JSON-native values."""
    if hasattr(v, "tolist"):  # numpy scalar/array, torch tensor
        return v.tolist()
    if isinstance(v, (list, tuple)):
        return [_scalar(x) for x in v]
    return v


# Strict-JSON encoding of ±Inf (json's own Infinity literal is non-standard).
# NaN keeps its legacy ``null`` encoding so pre-existing sidecars read the same.
_NONFINITE_TAG = "__mushin_nonfinite__"
_NONFINITE_VALUES = {"inf": float("inf"), "-inf": float("-inf")}


def _json_normalize(
    v: Any, as_text: list | None = None, *, tag_nan: bool = False
) -> Any:
    """Recursively normalize a metric value to strict-JSON-native types.

    Handles the full metric shape (nested dicts and lists), converts
    numpy/torch scalars and arrays, maps NaN to ``None`` and ±Inf to a tagged
    marker so the file is valid JSON (``read_metrics_sidecar`` restores them
    to NaN / signed Inf), and stringifies any other type rather than letting an
    otherwise-successful task's sidecar write crash the sweep. A genuine
    ``None`` metric value also encodes as ``null``, so it reads back as NaN,
    the one remaining lossy case. ``as_text``, when given, collects the type
    of each value (and dict key) written as its text instead of as itself.
    ``tag_nan`` tags a NaN too, as a cell's combo is stored, where a ``null``
    is a swept None (#355).
    """
    import math

    if hasattr(v, "item") and getattr(v, "ndim", None) == 0:  # 0-d np/torch scalar
        v = v.item()
    elif hasattr(v, "tolist"):  # numpy/torch array
        return _json_normalize(v.tolist(), as_text, tag_nan=tag_nan)
    if isinstance(v, dict):
        if as_text is not None:
            as_text.extend(type(k).__name__ for k in v if not isinstance(k, str))
        return {
            str(k): _json_normalize(x, as_text, tag_nan=tag_nan) for k, x in v.items()
        }
    if isinstance(v, (list, tuple)):
        return [_json_normalize(x, as_text, tag_nan=tag_nan) for x in v]
    if isinstance(v, bool) or v is None:
        return v
    if isinstance(v, float):
        if math.isfinite(v):
            return v
        if math.isnan(v):
            return {_NONFINITE_TAG: "nan"} if tag_nan else None
        return {_NONFINITE_TAG: "inf" if v > 0 else "-inf"}
    if isinstance(v, (int, str)):
        return v
    if as_text is not None:
        as_text.append(type(v).__name__)
    return str(v)  # datetime, Path, arbitrary objects -> stringify, never crash


def _restore_nonfinite(v: Any) -> Any:
    """Inverse of the non-finite mapping in ``_json_normalize``: a ``None``
    in a metrics payload marks a value that was NaN on write (restored to NaN so
    metric columns stay float rather than object dtype on resume), and a tagged
    ``{"__mushin_nonfinite__": "inf"|"-inf"}`` marker restores to signed Inf."""
    if isinstance(v, dict):
        # Decode ONLY the exact marker we write; a user dict that merely shares
        # the tag key (any other value) round-trips unchanged.
        if set(v) == {_NONFINITE_TAG} and v[_NONFINITE_TAG] in _NONFINITE_VALUES:
            return _NONFINITE_VALUES[v[_NONFINITE_TAG]]
        return {k: _restore_nonfinite(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_restore_nonfinite(x) for x in v]
    return float("nan") if v is None else v


# A cell's combo tags a NaN too (`_json_normalize(tag_nan=True)`); metrics
# files keep NaN as null, so this table is the combo's alone.
_COMBO_NONFINITE = {**_NONFINITE_VALUES, "nan": float("nan")}


def restore_combo_value(v: Any) -> Any:
    """One value of a cell's stored combo as it was swept: a tagged NaN or
    infinity restored, a ``null`` left None (a swept None; a record from
    before NaN was tagged stored a NaN so too)."""
    if isinstance(v, dict):
        tag = v.get(_NONFINITE_TAG)
        # Only a tag mushin writes (a string): a list or dict there is a
        # value of the user's, and not hashable (Codex, #382).
        if (
            set(v) == {_NONFINITE_TAG}
            and isinstance(tag, str)
            and tag in _COMBO_NONFINITE
        ):
            return _COMBO_NONFINITE[tag]
        return {k: restore_combo_value(x) for k, x in v.items()}
    if isinstance(v, list):
        return [restore_combo_value(x) for x in v]
    return v


def combo_text(combo: Any) -> str:
    """A combo as a record stores it, as canonical text: two combos are the
    same cell's when their texts are equal. A NaN equals a NaN, and differs
    from None; 1, 1.0 and True differ, as their `combo_key` does."""
    return json.dumps(
        _json_normalize(combo, tag_nan=True), sort_keys=True, allow_nan=False
    )


def write_metrics_sidecar(job_dir, metrics: dict[str, Any]) -> str | None:
    """Write a cell's metrics file, and return a digest of what it wrote when
    no value had to be written as its text (an object with no JSON form,
    say), else None. A launcher that loses a job's result takes the file for
    it only when it still holds what the cell's record says it wrote (#345)."""
    as_text: list[str] = []
    text = _atomic_write_json(Path(job_dir) / METRICS_FILE, metrics, as_text)
    return None if as_text else _digest(text)


def read_metrics_sidecar_written(job_dir, digest: str) -> dict | None:
    """The metrics file of ``job_dir`` when it is the one whose digest
    `write_metrics_sidecar` returned, else None; read once, so what is
    checked is what is parsed."""
    try:
        text = (Path(job_dir) / METRICS_FILE).read_text(encoding="utf-8")
    except (OSError, ValueError):
        return None
    if _digest(text) != digest:
        return None
    try:
        return _restore_nonfinite(json.loads(text))
    except (ValueError, RecursionError):
        return None


def _digest(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def read_metrics_sidecar(job_dir) -> dict | None:
    p = Path(job_dir) / METRICS_FILE
    if not p.exists():
        return None
    # A corrupt/unreadable sidecar is treated like a missing one (return None ->
    # the cell re-runs) rather than aborting an otherwise-resumable sweep.
    # ValueError, not just JSONDecodeError: an int past the parser's digit limit
    # raises a plain ValueError, and RecursionError guards a deeply nested file.
    try:
        return _restore_nonfinite(json.loads(p.read_text(encoding="utf-8")))
    except (ValueError, OSError, RecursionError):
        return None


def _atomic_write_text(path: Path, text: str, *, mkdir: bool = True) -> None:
    if mkdir:
        path.parent.mkdir(parents=True, exist_ok=True)
    # Unique temp name: a fixed `<path>.tmp` would let two concurrent writers
    # (e.g. two resume runs on the same working_dir) truncate each other
    # mid-write and rename half-written content over the target.
    tmp = path.with_suffix(f"{path.suffix}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
    try:
        tmp.write_text(text, encoding="utf-8")
        _while_held(lambda: tmp.replace(path))  # atomic on POSIX/Windows
    finally:
        # No residue if the replace raced or failed; a delete that fails here
        # must not mask the replace's own error.
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass


#: How long, in seconds, each retry waits on Windows while another process
#: holds a sidecar (#357): about two seconds in all.
_HELD_WAITS = (0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.5, 0.7)

#: On Windows a file another process has open cannot be replaced or deleted:
#: Python opens files without sharing them for deletion.
_WINDOWS = sys.platform == "win32"


def _while_held(operation: Callable[[], object]) -> None:
    """Run ``operation``, a replace or a delete of a sidecar, retrying it on
    Windows while another process has the file open (#357): it raises
    ``PermissionError`` there, so `mushin.show` on a running sweep, or a test
    polling the status files, would fail the cell whose record it read.
    mushin's own readers hold a file only while they read it. A process that
    holds one longer (a sync client, an antivirus scan) still fails the write
    after about two seconds. A Windows share mounted from another system
    (WSL's /mnt/c, SMB) is not seen as Windows: there it fails at once, as
    before."""
    if not _WINDOWS:
        operation()
        return
    for wait in _HELD_WAITS:
        try:
            operation()
            return
        except PermissionError:
            time.sleep(wait)
    operation()


def remove_sidecar(path: Path) -> bool:
    """Delete a sidecar if it is there, waiting on Windows while another
    process has it open (#357); best-effort, as every caller wants it:
    returns whether it is gone, and never raises an ``OSError``."""
    try:
        _while_held(lambda: Path(path).unlink(missing_ok=True))
    except OSError:
        return False
    return True


def _atomic_write_json(path: Path, payload: Any, as_text: list | None = None) -> str:
    # Normalize every sidecar to strict-JSON-native values (nested numpy,
    # non-finite floats -> None, non-native -> str) so a write never emits
    # the invalid `NaN`/`Infinity` literals nor crashes on an exotic value.
    # `as_text` collects what had to be written as its text. Returns the text.
    text = json.dumps(_json_normalize(payload, as_text), indent=2, allow_nan=False)
    _atomic_write_text(path, text)
    return text


class Manifest:
    """Tracks each requested grid cell's status in <working_dir>/mushin_sweep_manifest.json."""

    SCHEMA = 1

    def __init__(
        self,
        root: Path,
        params: list[str],
        cells: dict | None = None,
        notes: str | None = None,
        tags: list[str] | None = None,
    ):
        self.root = Path(root)
        self.params = list(params)
        self.cells: dict[str, dict] = cells or {}
        # Free-form sweep-level lineage annotations (see run(notes=, tags=)).
        self.notes = notes
        self.tags = list(tags) if tags else []

    @classmethod
    def load_or_new(cls, root, params: list[str]) -> Manifest:
        p = Path(root) / MANIFEST_FILE
        if p.exists():
            # A manifest truncated by a mid-write kill is treated like a
            # missing one (same policy as the metrics sidecar): the per-cell
            # status sidecars remain the durable source of truth, so resume
            # reconstructs from them instead of aborting the whole sweep.
            try:
                d = json.loads(p.read_text(encoding="utf-8"))
            # ValueError, not just JSONDecodeError: bytes that are not
            # UTF-8 raise UnicodeDecodeError, a corrupt file all the same;
            # RecursionError a file nested too deep, as for the sidecars.
            except (ValueError, OSError, RecursionError):
                return cls(root, params)
            # A valid-JSON manifest of the wrong SHAPE (list payload, non-dict
            # cells) degrades like a corrupt one instead of crashing callers.
            if not isinstance(d, dict):
                return cls(root, params)
            cells = d.get("cells", {})
            if not isinstance(cells, dict) or not all(
                isinstance(v, dict) for v in cells.values()
            ):
                cells = {}
            return cls(
                root,
                d.get("params", params),
                cells,
                notes=d.get("notes"),
                tags=d.get("tags"),
            )
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
            # Skip missing/wrong-shape sidecars (read_cell_status already drops
            # non-dict payloads); combo must be a mapping for combo_key.
            if s is None or not isinstance(s.get("combo"), dict):
                continue
            m.cells[combo_key(s["combo"])] = {
                "dir": d.name,
                "status": s.get("status", "pending"),
            }
        return m

    def mark(
        self, combo: dict, *, dir: str, status: str, error: str | None = None
    ) -> None:
        entry = {"dir": str(dir), "status": status}
        if error is not None:
            entry["error"] = error
        self.cells[combo_key(combo)] = entry  # replace in place

    def is_complete(self) -> bool:
        return all(v.get("status") == "completed" for v in self.cells.values())

    def save(self) -> None:
        payload = {"schema": self.SCHEMA, "params": self.params, "cells": self.cells}
        # Only emit lineage keys when set, so a manifest without notes/tags is
        # unchanged from prior behavior.
        if self.notes is not None:
            payload["notes"] = self.notes
        if self.tags:
            payload["tags"] = self.tags
        _atomic_write_json(self.root / MANIFEST_FILE, payload)
