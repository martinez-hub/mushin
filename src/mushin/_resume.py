# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
"""Primitives for kill-durable, resumable sweeps: the per-cell status sidecar,
best-effort checkpoint discovery, and the ResumeContext handed to a task that
opts in via a ``mushin_resume`` parameter (see the resume-hardening design)."""

from __future__ import annotations

import contextvars
import functools
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ._sweep_io import _atomic_write_json

STATUS_FILE = "mushin_cell_status.json"


@dataclass(frozen=True)
class ResumeContext:
    """Handed to a task that declares a ``mushin_resume`` parameter.

    Attributes
    ----------
    dir : Path | None
        The cell's working directory (already the cwd when the task runs).
    is_resume : bool
        True when a prior attempt of the SAME combo left artifacts here.
    last_ckpt : Path | None
        The newest checkpoint in ``dir``, or None.
    attempt : int
        1 on the first run; increments on each re-execution of this combo.
    """

    dir: Path | None
    is_resume: bool
    last_ckpt: Path | None
    attempt: int


def write_cell_status(
    cell_dir,
    *,
    status: str,
    combo: dict[str, Any],
    attempt: int,
    config_hash: str | None = None,
    code_hash: str | None = None,
    code_deps: dict[str, str] | None = None,
    job_num: int | None = None,
    choices: dict[str, str | None] | None = None,
) -> None:
    """Atomically write this cell's status sidecar into its own dir.

    ``job_num`` and ``choices`` are the Hydra job number and config-group
    choices the job's hydra.yaml also records. Written when both are known
    (inside a Hydra multirun job) and are what Hydra writes (the rule the
    reader applies, so nothing is recorded that a reload would then reject),
    never one without the other, they let a reload sort the cells and key
    config-group axes without parsing hydra.yaml, which is most of a reload's
    time; see ``sidecar_job_fields``."""

    payload: dict[str, Any] = {
        "status": status,
        "combo": combo,
        "attempt": int(attempt),
    }
    if config_hash is not None:
        payload["config_hash"] = config_hash
    if code_hash is not None:
        payload["code_hash"] = code_hash
    if code_deps is not None:
        payload["code_deps"] = code_deps
    if _records_job_fields(job_num, choices):
        payload["job_num"] = job_num
        payload["choices"] = choices
    _atomic_write_json(Path(cell_dir) / STATUS_FILE, payload)


def _records_job_fields(num: Any, choices: Any) -> bool:
    """The one rule for what the sidecar records and what a reload accepts:
    a job number and choices as Hydra writes them, the choices a mapping (a
    number is never recorded without them)."""
    from ._hydra_yaml import job_fields_as_written

    return isinstance(choices, dict) and job_fields_as_written(num, choices)


def sidecar_job_fields(cell_dir, hydra_yaml=None) -> tuple[int, dict] | None:
    """The job number and config-group choices this cell's status sidecar
    records, as ``read_hydra_job`` reads them from its hydra.yaml, or None
    when the sidecar has none to use: a sweep from before they were recorded,
    a field of the wrong shape (choices must be a mapping; a sidecar never
    records a number without them), a sidecar older than ``hydra_yaml``, or a
    ``hydra_yaml`` that is there but cannot be examined.

    Hydra writes a job's hydra.yaml before the task runs, and the task writes
    the sidecar, so a sidecar older than the hydra.yaml beside it is a previous
    sweep's record of a reused directory (the new job was killed before its
    first status write), and that hydra.yaml is the record to read."""
    if hydra_yaml is not None:
        # The sidecar's age is taken before its content: a sidecar replaced in
        # between is then read in its newer state, and only its age is
        # understated, so the comparison errs toward hydra.yaml.
        try:
            written = (Path(cell_dir) / STATUS_FILE).stat().st_mtime_ns
        except OSError:
            return None
        try:
            newer = Path(hydra_yaml).stat().st_mtime_ns > written
        except FileNotFoundError:  # no hydra.yaml: nothing newer to defer to
            newer = False
        except OSError:  # unreadable: let the hydra.yaml path report it
            return None
        if newer:
            return None
    status = read_cell_status(cell_dir)
    if status is None:
        return None
    num = status.get("job_num")
    choices = status.get("choices")
    return (num, choices) if _records_job_fields(num, choices) else None


def config_fingerprint(cfg) -> str | None:
    """Stable short hash of the fully-resolved job config, or None if the
    config cannot be resolved/serialized.

    Guards resume reuse: a completed cell's cached metrics are only returned
    when the config that would run now matches the one that produced them;
    otherwise a changed NON-swept value (same combo key) would silently mix
    results from two configurations into one dataset. Task *source* changes are
    guarded separately by ``code_fingerprint``."""
    try:
        from omegaconf import OmegaConf

        data = (
            OmegaConf.to_container(cfg, resolve=True)
            if OmegaConf.is_config(cfg)
            else cfg
        )
        payload = json.dumps(data, sort_keys=True, default=str)
    except Exception:  # noqa: BLE001 - unresolvable config -> no guard
        return None
    import hashlib

    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def code_fingerprint(fn) -> str | None:
    """Stable short hash of a task function's source, or None if the source
    can't be read.

    Complements ``config_fingerprint`` in the resume guard: a task whose
    *body* was edited between runs (same config, same combo key) would
    otherwise have its stale cached metrics silently returned. Only the task
    function's own source is hashed; the project's other code the task reaches
    (helpers, module-level constants) is covered by ``project_fingerprints``,
    installed packages and data files by neither, so a fresh ``working_dir`` is
    still the safe choice after upgrading a dependency or changing the data."""
    import inspect

    try:
        # unwrap functools.wraps / decorators to the underlying function
        target = inspect.unwrap(fn) if callable(fn) else fn
        src = inspect.getsource(target)
    except (OSError, TypeError):
        return None
    import hashlib

    return hashlib.sha256(src.encode()).hexdigest()[:16]


#: The interpreter's own directories, by their sysconfig names: the standard
#: library and the installed packages. Not the bare prefixes: a project checked
#: out under the interpreter's prefix (/usr/src/app on a Debian system Python,
#: code kept inside a conda env) is still the project's own.
_FOREIGN_PATH_KEYS = ("stdlib", "platstdlib", "purelib", "platlib")


@functools.cache
def _foreign_roots() -> tuple[Path, ...]:
    """Directories whose Python files are not the project's own: the running
    interpreter's standard library and site-packages (its own, a venv's, the
    user's) and mushin itself. Resolved once; a file under any of them is left
    out of ``project_fingerprints``."""
    import site
    import sysconfig

    paths = sysconfig.get_paths()
    roots: set[str] = {paths[k] for k in _FOREIGN_PATH_KEYS if paths.get(k)}
    try:
        roots.update(site.getsitepackages())
        roots.add(site.getusersitepackages())
    except Exception:  # noqa: BLE001 - a restricted site module; nothing to add
        pass
    roots.add(str(Path(__file__).parent))  # mushin's own package
    out = []
    for r in roots:
        if not r:
            continue
        try:
            out.append(Path(r).resolve())
        except OSError:
            continue
    return tuple(out)


def _project_file(module) -> Path | None:
    """The file ``module`` was loaded from when it is the project's own (not
    under the interpreter or mushin), else None. Any file counts, a compiled
    extension included: rebuilding one changes its bytes."""
    f = getattr(module, "__file__", None)
    if not isinstance(f, str) or not f:
        return None
    try:
        path = Path(f).resolve()
        if not path.is_file():
            return None
    except OSError:
        return None
    if any(path.is_relative_to(root) for root in _foreign_roots()):
        return None
    return path


def _loaded_from_older_source(module, path: Path) -> bool:
    """True when ``module``'s bytecode cache says it was compiled from a version
    of ``path`` with a different size or mtime than the file has now: the file
    was edited after this process imported it, so the code that would run is
    not the code on disk. Best effort: without a timestamp-based ``.pyc`` (no
    bytecode written, a hash-based pyc) there is nothing to compare."""
    cached = getattr(module, "__cached__", None)
    if not isinstance(cached, str) or not cached:
        return False
    try:
        with open(cached, "rb") as f:
            header = f.read(16)
        st = path.stat()
    except OSError:
        return False
    if len(header) < 16 or int.from_bytes(header[4:8], "little") & 0b11:
        return False  # truncated, or hash-based: no source mtime recorded
    mtime = int.from_bytes(header[8:12], "little")
    size = int.from_bytes(header[12:16], "little")
    return (int(st.st_mtime) & 0xFFFFFFFF) != mtime or (st.st_size & 0xFFFFFFFF) != size


def _is_main_guard(node) -> bool:
    """``if __name__ == "__main__":`` (either order): a script's driver block,
    which an import never runs and a resume edit most often touches."""
    import ast

    if not isinstance(node, ast.If) or not isinstance(node.test, ast.Compare):
        return False
    parts = [node.test.left, *node.test.comparators]
    names = {part.id for part in parts if isinstance(part, ast.Name)}
    consts = {part.value for part in parts if isinstance(part, ast.Constant)}
    return "__name__" in names and "__main__" in consts


def _source_digest(path: Path) -> str:
    """The hash of the code in ``path``: for a Python file its syntax tree
    without the ``if __name__ == "__main__":`` blocks, so comments, formatting,
    line endings and a driver script's own run block do not count as edits
    (``ast.dump`` is stable within a Python version; the first resume after an
    interpreter upgrade may re-run once); for any other file, or one that does
    not parse, its bytes."""
    import ast

    data = path.read_bytes()
    if path.suffix == ".py":
        try:
            tree = ast.parse(data, filename=str(path))
        except (SyntaxError, ValueError):
            pass
        else:
            tree.body = [node for node in tree.body if not _is_main_guard(node)]
            dumped = ast.dump(tree, include_attributes=False)
            return hashlib.sha256(dumped.encode()).hexdigest()[:16]
    return hashlib.sha256(data).hexdigest()[:16]


def _module_owner(value):
    """The module name a global's code comes from: a function's or class's
    own, otherwise its type's; None when that cannot be read (a proxy whose
    attribute access raises, a descriptor that is not a string)."""
    import types

    try:
        owner = (
            value.__module__
            if isinstance(value, (types.FunctionType, type))
            else type(value).__module__
        )
    except Exception:  # noqa: BLE001 - best effort over arbitrary globals
        return None
    return owner if isinstance(owner, str) else None


def project_fingerprints(fn) -> dict[str, str] | None:
    """Hashes of the code in the project's own source files the task reaches,
    keyed by path relative to the task's module (so a script run as
    ``__main__`` and the same file imported by name agree, and a relocated
    checkout still matches): the module that defines ``fn`` and, transitively,
    every project module referenced from a reached module's globals (an
    imported module, or the module of a function, class or object held there).
    A Python file is hashed as its syntax tree without the ``if __name__ ==
    "__main__":`` blocks (see ``_source_digest``), so an edit to a driver
    script's run block, a comment or the formatting is not a code change, while
    a constant or a helper in the same file is. None when ``fn``'s
    module cannot be found (a function built at runtime), or when a reached
    file was edited after this process imported it (with a warning: the code
    that would run is not the code on disk, so nothing is recorded under the
    new hash; restart the process). An empty dict when the task reaches no
    project file (defined inside an installed package, or in a notebook that
    imports none).

    Complements ``code_fingerprint``, which covers the task's own body: an
    edited helper, a changed module constant or a modified imported module of
    the project changes this fingerprint, so a resume re-runs the cell and a
    ``cache_dir`` lookup misses, instead of returning a result computed by code
    that no longer exists. Installed packages are left out: a dependency
    upgrade is provenance's job, and hashing site-packages would miss the cache
    on every unrelated install."""
    import inspect
    import os
    import sys
    import types
    import warnings

    try:
        target = fn
        while isinstance(target, functools.partial):
            target = target.func
        if callable(target):
            try:
                target = inspect.unwrap(target)
            except ValueError:  # a __wrapped__ cycle: hash what we have
                pass
        module = inspect.getmodule(target)
        if module is None and not isinstance(target, (types.FunctionType, type)):
            module = inspect.getmodule(type(target))
        if module is None:
            return None
        base_file = _project_file(module)
        base = base_file.parent if base_file is not None else Path.cwd()
        hashes: dict[str, str] = {}
        visited: set[str] = set()
        pending = [module]
        while pending:
            mod = pending.pop()
            name = getattr(mod, "__name__", None)
            if not isinstance(name, str) or name in visited:
                continue
            visited.add(name)
            path = _project_file(mod)
            if path is not None:
                if _loaded_from_older_source(mod, path):
                    warnings.warn(
                        f"{path} was edited after this process imported it, so the "
                        "code that would run is not the code on disk; the project "
                        "code the task reaches is not fingerprinted for this run "
                        "(a resume reuses completed cells as before). Restart the "
                        "process to run, and record, the edited code.",
                        UserWarning,
                        stacklevel=3,
                    )
                    return None
                try:
                    rel = Path(os.path.relpath(path, base)).as_posix()
                except ValueError:  # another drive on Windows
                    rel = path.as_posix()
                try:
                    hashes[rel] = _source_digest(path)
                except OSError:
                    continue
            elif mod is not module:
                continue  # a foreign module: do not walk into site-packages
            # The task's own module is walked whether or not it has a file (a
            # notebook's __main__ reaches the project modules it imports).
            try:
                values = list(vars(mod).values())
            except TypeError:
                continue
            for value in values:
                if isinstance(value, types.ModuleType):
                    pending.append(value)
                    continue
                owner = _module_owner(value)
                if owner is not None and owner in sys.modules:
                    pending.append(sys.modules[owner])
        return hashes
    except Exception:  # noqa: BLE001 - a best-effort guard never fails the sweep
        return None


def read_cell_status(cell_dir) -> dict | None:
    """Read a cell status sidecar; a missing/corrupt/wrong-shape one reads as
    None (so callers degrade rather than crash on `.get`)."""
    p = Path(cell_dir) / STATUS_FILE
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text())
    except (ValueError, OSError, RecursionError):
        # ValueError covers JSONDecodeError, a non-UTF-8 file and an integer
        # past Python's digit limit; RecursionError a file nested too deep. All
        # read as a corrupt sidecar, as the MCP server lists a directory of runs
        # it did not write.
        return None
    return data if isinstance(data, dict) else None


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


_CURRENT_RESUME: contextvars.ContextVar[ResumeContext | None] = contextvars.ContextVar(
    "mushin_current_resume", default=None
)


def current_resume() -> ResumeContext | None:
    """The ResumeContext for the cell currently executing, or None."""
    return _CURRENT_RESUME.get()


def build_resume_context(cell_dir, combo: dict[str, Any]) -> ResumeContext:
    """Compute the ResumeContext for a cell about to (re-)execute in ``cell_dir``.

    Combo-match guard: a prior status sidecar is honored ONLY if its recorded
    combo equals ``combo``. This makes numeric-dir reuse safe; if a grid change
    reused this dir for a different cell, we neither resume nor surface that
    cell's checkpoint. (Resume is only meaningful for a workflow that records a
    non-degenerate per-cell combo; an empty combo cannot distinguish cells.)"""
    cell_dir = Path(cell_dir)
    prior = read_cell_status(cell_dir)
    matches = prior is not None and prior.get("combo") == combo
    last = discover_last_ckpt(cell_dir) if matches else None
    # Coerce a malformed/older-schema `attempt` (missing, null, "3") to an int
    # so it degrades to a fresh attempt instead of raising inside the wrapper.
    try:
        prior_attempt = int(prior["attempt"]) if matches else 0
    except (KeyError, TypeError, ValueError):
        prior_attempt = 0
    attempt = prior_attempt + 1 if matches else 1
    return ResumeContext(
        dir=cell_dir, is_resume=matches, last_ckpt=last, attempt=attempt
    )
