# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
"""On-disk primitives for resilient/resumable sweeps: canonical combo keys,
the per-job metrics sidecar + sweep manifest (see the sweep-resilience design),
and a fast read of each job's Hydra record."""

from __future__ import annotations

import functools
import json
import os
import uuid
from pathlib import Path
from typing import Any

import yaml
from yaml.composer import Composer
from yaml.constructor import ConstructorError, SafeConstructor
from yaml.resolver import BaseResolver, Resolver

try:  # libyaml, which PyYAML's wheels include
    from yaml.cyaml import CDumper, CParser
except ImportError:  # a PyYAML built without it: reloads take OmegaConf's path
    CDumper = CParser = None

METRICS_FILE = "mushin_metrics.json"
MANIFEST_FILE = "mushin_sweep_manifest.json"


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


def _json_normalize(v: Any) -> Any:
    """Recursively normalize a metric value to strict-JSON-native types.

    Handles the full metric shape (nested dicts and lists), converts
    numpy/torch scalars and arrays, maps NaN to ``None`` and ±Inf to a tagged
    marker so the file is valid JSON (``read_metrics_sidecar`` restores them
    to NaN / signed Inf), and stringifies any other type rather than letting an
    otherwise-successful task's sidecar write crash the sweep. A genuine
    ``None`` metric value also encodes as ``null``, so it reads back as NaN,
    the one remaining lossy case.
    """
    import math

    if hasattr(v, "item") and getattr(v, "ndim", None) == 0:  # 0-d np/torch scalar
        v = v.item()
    elif hasattr(v, "tolist"):  # numpy/torch array
        return _json_normalize(v.tolist())
    if isinstance(v, dict):
        return {str(k): _json_normalize(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_json_normalize(x) for x in v]
    if isinstance(v, bool) or v is None:
        return v
    if isinstance(v, float):
        if math.isfinite(v):
            return v
        if math.isnan(v):
            return None
        return {_NONFINITE_TAG: "inf" if v > 0 else "-inf"}
    if isinstance(v, (int, str)):
        return v
    return str(v)  # datetime, Path, arbitrary objects -> stringify, never crash


def _restore_nonfinite(v: Any) -> Any:
    """Inverse of the non-finite mapping in :func:`_json_normalize`: a ``None``
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


def write_metrics_sidecar(job_dir, metrics: dict[str, Any]) -> None:
    _atomic_write_json(Path(job_dir) / METRICS_FILE, metrics)


def read_metrics_sidecar(job_dir) -> dict | None:
    p = Path(job_dir) / METRICS_FILE
    if not p.exists():
        return None
    # A corrupt/unreadable sidecar is treated like a missing one (return None ->
    # the cell re-runs) rather than aborting an otherwise-resumable sweep.
    try:
        return _restore_nonfinite(json.loads(p.read_text()))
    except (json.JSONDecodeError, OSError):
        return None


if CParser is not None:

    class _LibyamlLoader(Composer, CParser, SafeConstructor, Resolver):
        """libyaml's parser under PyYAML's Python composer.

        CSafeLoader also composes in C, recursing on the C stack with no depth
        limit, so a deeply nested file (about 20,000 levels, 60 KB) kills the
        process past any ``except``. The Python composer raises RecursionError
        instead, as the pure-Python loader of ``OmegaConf.load`` does. On
        Hydra's 3 KB hydra.yaml it takes 0.4 ms, against 0.3 ms all in C and
        3 ms all in Python.
        """

        def __init__(self, stream: str) -> None:
            CParser.__init__(self, stream)
            Composer.__init__(self)
            SafeConstructor.__init__(self)
            Resolver.__init__(self)

        def construct_mapping(self, node: Any, deep: bool = False) -> Any:
            # OmegaConf's loader refuses a duplicate key; so does this one.
            keys = set()
            for key_node, _ in node.value:
                if key_node.tag != BaseResolver.DEFAULT_SCALAR_TAG:
                    continue
                if key_node.value in keys:
                    raise ConstructorError(
                        "while constructing a mapping",
                        node.start_mark,
                        f"found duplicate key {key_node.value}",
                        key_node.start_mark,
                    )
                keys.add(key_node.value)
            return super().construct_mapping(node, deep=deep)


# Names Python itself puts in a class __dict__ (3.13 adds the last two).
_CLASS_METADATA = {
    "__module__",
    "__doc__",
    "__qualname__",
    "__dict__",
    "__weakref__",
    "__firstlineno__",
    "__static_attributes__",
}
# What OmegaConf's loader and dumper define over yaml.SafeLoader and
# yaml.Dumper, in omegaconf 2.2.3 to 2.3.1. _LibyamlLoader copies the loader's
# tables and reimplements its construct_mapping; the dumper's representers
# are copied whole.
_OMEGACONF_LOADER_ATTRS = {
    "construct_mapping",
    "yaml_constructors",
    "yaml_implicit_resolvers",
}
_OMEGACONF_DUMPER_ATTRS = {
    "str_representer",
    "str_representer_added",
    "yaml_representers",
}

# Hydra's hydra.yaml nests about 6 deep. OmegaConf.create recurses per level
# and overflows somewhere past 100, at a depth that depends on the recursion
# limit, so a deeper file is left to it.
_MAX_PLAIN_DEPTH = 16


def _plain_codecs() -> tuple[type, type] | None:
    """A libyaml loader and dumper that read and write YAML as OmegaConf's do,
    or None (no libyaml, or OmegaConf's private helpers moved or changed
    shape), which sends every file to OmegaConf.

    OmegaConf builds its loader afresh for each load, from PyYAML's tables;
    these are rebuilt only when OmegaConf's helpers or those tables change,
    which a library adding a resolver, constructor or representer does."""
    if CParser is None:
        return None
    try:
        from omegaconf import _utils

        key = (_utils.get_yaml_loader, _utils.get_omega_conf_dumper, _yaml_tables())
    except Exception:  # noqa: BLE001 - only the fast path is lost
        return None
    return _build_plain_codecs(key)


def _yaml_tables() -> tuple:
    """The identity and size of each PyYAML table OmegaConf's loader and
    dumper start from. Entries are added in place, so the size tells."""
    from omegaconf._utils import get_omega_conf_dumper

    tables = (
        yaml.SafeLoader.yaml_implicit_resolvers,
        yaml.SafeLoader.yaml_constructors,
        yaml.SafeLoader.yaml_multi_constructors,
        yaml.Dumper.yaml_implicit_resolvers,
        yaml.Dumper.yaml_multi_representers,
        get_omega_conf_dumper().yaml_representers,
    )
    return tuple(
        (id(t), sum(len(v) if isinstance(v, list) else 1 for v in t.values()))
        for t in tables
    )


@functools.lru_cache(maxsize=1)
def _build_plain_codecs(key: tuple) -> tuple[type, type] | None:
    get_yaml_loader, get_omega_conf_dumper, _ = key
    try:
        omegaconf_loader = get_yaml_loader()
        omegaconf_dumper = get_omega_conf_dumper()
    except Exception:  # noqa: BLE001 - only the fast path is lost
        return None
    if (
        omegaconf_loader.__bases__ != (yaml.SafeLoader,)
        or set(vars(omegaconf_loader)) - _CLASS_METADATA != _OMEGACONF_LOADER_ATTRS
        or omegaconf_dumper.__bases__ != (yaml.Dumper,)
        or set(vars(omegaconf_dumper)) - _CLASS_METADATA != _OMEGACONF_DUMPER_ATTRS
    ):
        return None
    loader = type(
        "_HydraYamlLoader",
        (_LibyamlLoader,),
        {
            # OmegaConf reads `1e3` as a float, leaves dates as strings and
            # builds pathlib tags; its tables carry all of that.
            "yaml_implicit_resolvers": omegaconf_loader.yaml_implicit_resolvers,
            "yaml_constructors": omegaconf_loader.yaml_constructors,
        },
    )
    dumper = type(
        "_HydraYamlDumper",
        (CDumper,),
        {"yaml_representers": omegaconf_dumper.yaml_representers},
    )
    return loader, dumper


def read_hydra_job(hydra_yaml: Path | str) -> tuple[Any, dict | None]:
    """The job number and the config-group choices in a job's ``hydra.yaml``.

    The file holds Hydra's whole runtime config (about 150 lines), and a reload
    needs two of its fields. Building an OmegaConf config from it was about 90%
    of ``load_from_dir``'s time on a 1000-cell sweep, so it is read as plain
    YAML, typed as OmegaConf types it, and the two fields are taken from that.
    The result is always what ``OmegaConf.load`` gives:

    - A file that ``OmegaConf.load`` would parse or check differently, anywhere
      in it, goes to ``OmegaConf.load`` (see ``_parse_plain``).
    - Fields Hydra does not write that way (a ``???`` or ``${...}`` job number,
      choices that are not names) are taken from ``OmegaConf.create`` on the
      parsed tree, which is what ``OmegaConf.load`` builds its config from.
    """
    hydra_yaml = Path(hydra_yaml)
    doc = _parse_plain(hydra_yaml)
    if doc is None:
        from hydra_zen import load_from_yaml

        hydra = load_from_yaml(hydra_yaml).hydra
    else:
        fields = _plain_hydra_fields(doc)
        if fields is not None:
            return fields
        from omegaconf import OmegaConf

        hydra = OmegaConf.create(doc).hydra
    return hydra.job.num, hydra_choices(hydra)


def _parse_plain(hydra_yaml: Path) -> dict | None:
    """The file parsed as ``OmegaConf.load`` would parse it, without building
    a config, or None to leave the file to ``OmegaConf.load``: on any error, so
    it raises its own, and on anything the two could treat differently.

    libyaml is more lenient than PyYAML's pure-Python scanner, which
    ``OmegaConf.load`` uses: it accepts tabs as separators, a byte-order mark
    at any line start, an empty ``!`` tag, a comment right after a block-scalar
    header and a ``?`` inside a flow scalar, among others. Rather than chase
    each one, the parsed tree is written back the way Hydra writes hydra.yaml
    (``OmegaConf.to_yaml``), on libyaml's emitter. Only a file that comes out
    byte for byte the same is read here: text that a YAML emitter produced,
    which both parsers read alike. Anything hand-edited goes to OmegaConf.
    """
    codecs = _plain_codecs()
    if codecs is None:
        return None
    loader, dumper = codecs
    try:
        text = hydra_yaml.read_text(encoding="utf-8")
        doc = yaml.load(text, Loader=loader)
        if not isinstance(doc, dict) or not _omegaconf_would_create(doc):
            return None
        canonical = yaml.dump(
            doc,
            Dumper=dumper,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
        )
    except Exception:  # noqa: BLE001 - OmegaConf decides, and raises its own error
        return None
    return doc if canonical == text else None


def _plain_hydra_fields(doc: dict) -> tuple[int, dict | None] | None:
    """The job number and choices from the parsed tree, when they are what
    Hydra writes: an int, and a mapping of group names to option names (or
    None). None otherwise (``???``, ``${...}``, an interpolated node, a missing
    field), for OmegaConf to resolve or reject."""
    try:
        hydra = doc["hydra"]
        num = hydra["job"]["num"]
        runtime = hydra.get("runtime")
        choices = None if runtime is None else runtime.get("choices")
    except Exception:  # noqa: BLE001 - OmegaConf decides, and raises its own error
        return None
    if type(num) is not int:
        return None
    if choices is None:
        return num, None
    if not isinstance(choices, dict) or not all(
        isinstance(k, str) and (v is None or _is_option_name(v))
        for k, v in choices.items()
    ):
        return None
    return num, choices


def _omegaconf_would_create(tree: Any) -> bool:
    """Whether ``OmegaConf.create`` accepts this parsed YAML, by its own rules:
    every key of a type a DictConfig takes, every leaf of a type its nodes take,
    and valid syntax in every string holding ``${`` (it parses each one).

    False, which sends the file to OmegaConf to decide, also for a container
    reached twice (a YAML alias: a cyclic one makes OmegaConf.create recurse
    until RecursionError), for nesting deeper than ``_MAX_PLAIN_DEPTH``, and if
    those private helpers move. Walks with a stack, since a recursion here
    would undo the composer's RecursionError guard."""
    try:
        from omegaconf import DictKeyType
        from omegaconf._utils import is_primitive_type_annotation

        key_types = DictKeyType.__args__  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001 - only the fast path is lost
        return False
    seen: set[int] = set()
    stack = [(tree, 0)]
    while stack:
        node, depth = stack.pop()
        if isinstance(node, dict) or type(node) in (list, tuple):
            if id(node) in seen or depth > _MAX_PLAIN_DEPTH:
                return False
            seen.add(id(node))
            if isinstance(node, dict):
                if not all(isinstance(k, key_types) for k in node):
                    return False
                stack.extend((v, depth + 1) for v in node.values())
            else:
                stack.extend((v, depth + 1) for v in node)
        elif isinstance(node, str):
            if "${" in node and not _valid_interpolation(node):
                return False
        elif not is_primitive_type_annotation(node):
            return False
    return True


@functools.lru_cache(maxsize=256)
def _valid_interpolation(text: str) -> bool:
    """OmegaConf's syntax check on a string holding ``${``. Cached: the cells
    of a sweep repeat the same few, and a full grammar parse is slow. Small,
    since the MCP server lives long and reads many sweeps."""
    try:
        from omegaconf._utils import get_value_kind

        get_value_kind(text, strict_interpolation_validation=True)
    except Exception:  # noqa: BLE001 - a syntax error, or the helper moved
        return False
    return True


def _is_option_name(v: Any) -> bool:
    return isinstance(v, str) and v != "???" and "${" not in v


def hydra_choices(hydra: Any) -> dict | None:
    """``runtime.choices`` of a ``hydra`` node, or None where it has none."""
    try:
        ch = hydra["runtime"]["choices"]
        return None if ch is None else dict(ch)
    except Exception:  # noqa: BLE001 - older dirs without choices
        return None


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Unique temp name: a fixed `<path>.tmp` would let two concurrent writers
    # (e.g. two resume runs on the same working_dir) truncate each other
    # mid-write and rename half-written content over the target.
    tmp = path.with_suffix(f"{path.suffix}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
    try:
        # Normalize every sidecar to strict-JSON-native values (nested numpy,
        # non-finite floats -> None, non-native -> str) so a write never emits
        # the invalid `NaN`/`Infinity` literals nor crashes on an exotic value.
        tmp.write_text(json.dumps(_json_normalize(payload), indent=2, allow_nan=False))
        tmp.replace(path)  # atomic on POSIX/Windows
    finally:
        tmp.unlink(missing_ok=True)  # no residue if the replace raced/failed


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
                d = json.loads(p.read_text())
            except (json.JSONDecodeError, OSError):
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
        payload = {"schema": self.SCHEMA, "params": self.params, "cells": self.cells}
        # Only emit lineage keys when set, so a manifest without notes/tags is
        # unchanged from prior behavior.
        if self.notes is not None:
            payload["notes"] = self.notes
        if self.tags:
            payload["tags"] = self.tags
        _atomic_write_json(self.root / MANIFEST_FILE, payload)
