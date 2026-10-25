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


def _short_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


def _canonical_dump(node) -> str:
    """A syntax tree as text, leaving out empty and None fields. ``ast.dump``
    prints every field, and the fields grow across Python minors (3.12 added
    ``type_params=[]`` to every def and class, 3.13 changed which empty fields
    are shown), so the same code would hash differently under another
    interpreter and a resume would blame every project file. Code that uses
    no syntax newer than 3.11 hashes alike on 3.11 to 3.14."""
    import ast

    if isinstance(node, ast.AST):
        fields = [
            f"{name}={_canonical_dump(value)}"
            for name, value in ast.iter_fields(node)
            if value is not None and value != []
        ]
        return f"{type(node).__name__}({','.join(fields)})"
    if isinstance(node, list):
        return "[" + ",".join(_canonical_dump(item) for item in node) + "]"
    return repr(node)


def _unwrap_task(fn):
    """The function behind ``fn``: through functools.partial and decorators."""
    import inspect

    target = fn
    while isinstance(target, functools.partial):
        target = target.func
    if callable(target):
        try:
            target = inspect.unwrap(target)
        except ValueError:  # a __wrapped__ cycle: hash what we have
            pass
    return target


def _task_source(fn) -> str | None:
    """The task's source through decorators; a functools.partial has none
    (callers key it on its bound arguments instead)."""
    import inspect

    try:
        target = inspect.unwrap(fn) if callable(fn) else fn
        return inspect.getsource(target)
    except (OSError, TypeError, ValueError):
        return None


def code_fingerprint(fn) -> str | None:
    """Stable short hash of a task function's code, or None if the source
    can't be read.

    Complements ``config_fingerprint`` in the resume guard: a task whose
    *body* was edited between runs (same config, same combo key) would
    otherwise have its stale cached metrics silently returned. The source is
    hashed as its syntax tree, so a comment or a reformat inside the body is
    not an edit, the same rule ``project_fingerprints`` applies to the files
    the task reaches (a source that does not parse on its own, a lambda in a
    larger statement, is hashed as text). Only the task function's own source
    is hashed; the project's other code the task reaches (helpers,
    module-level constants) is covered by ``project_fingerprints``, installed
    packages and data files by neither, so a fresh ``working_dir`` is still
    the safe choice after upgrading a dependency or changing the data."""
    import ast
    import textwrap

    src = _task_source(fn)
    if src is None:
        return None
    try:
        tree = ast.parse(textwrap.dedent(src))
        return _short_hash(_canonical_dump(tree).encode())
    except (SyntaxError, ValueError, RecursionError):
        return _short_hash(src.encode())


def legacy_code_fingerprint(fn) -> str | None:
    """The hash of the task's source as text, which sidecars recorded before
    ``code_fingerprint`` hashed the syntax tree. A resume accepts either, so a
    sweep recorded by an earlier version still reuses its cells."""
    src = _task_source(fn)
    return None if src is None else _short_hash(src.encode())


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


def _project_path(file) -> Path | None:
    """``file`` resolved, when it is an existing file of the project's own (not
    under the interpreter or mushin), else None. Any file counts, a compiled
    extension included: rebuilding one changes its bytes."""
    if not isinstance(file, str) or not file:
        return None
    try:
        path = Path(file).resolve()
        if not path.is_file():
            return None
    except OSError:
        return None
    if any(path.is_relative_to(root) for root in _foreign_roots()):
        return None
    return path


def _project_file(module) -> Path | None:
    """The file ``module`` was loaded from when it is the project's own."""
    return _project_path(getattr(module, "__file__", None))


def _loaded_from_older_source(module, path: Path) -> bool:
    """True when ``module``'s bytecode cache says it was compiled from a version
    of ``path`` with a different size or mtime than the file has now: the file
    was edited after this process imported it, so the code that would run is
    not the code on disk. Best effort, and silent when the cache cannot be
    trusted: no timestamp-based ``.pyc`` (none written, or hash-based), or a
    process that does not rewrite a stale one (``PYTHONDONTWRITEBYTECODE``, a
    read-only ``__pycache__``), where a stale header means the opposite, an
    import that compiled the current source fresh."""
    import os
    import sys

    cached = getattr(module, "__cached__", None)
    if not isinstance(cached, str) or not cached or sys.dont_write_bytecode:
        return False
    try:
        if Path(cached).resolve() == path:
            return False  # a sourceless module: the .pyc is the code itself
        if not os.access(os.path.dirname(cached) or ".", os.W_OK):
            return False
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
    which an import never runs and a resume edit most often touches. Not
    ``!=``, whose block is import-time code."""
    import ast

    if not isinstance(node, ast.If) or not isinstance(node.test, ast.Compare):
        return False
    if len(node.test.ops) != 1 or not isinstance(node.test.ops[0], ast.Eq):
        return False
    parts = [node.test.left, *node.test.comparators]
    names = {part.id for part in parts if isinstance(part, ast.Name)}
    consts = {part.value for part in parts if isinstance(part, ast.Constant)}
    return "__name__" in names and "__main__" in consts


def _parse_source(path: Path):
    """The syntax tree of a Python file, or None for any other file or one
    that does not parse."""
    import ast

    if path.suffix != ".py":
        return None
    try:
        return ast.parse(path.read_bytes(), filename=str(path))
    except (SyntaxError, ValueError):
        return None


def _source_digest(path: Path, tree=None) -> str:
    """The hash of the code in ``path``: for a Python file its syntax tree
    without the ``if __name__ == "__main__":`` blocks, so comments, formatting,
    line endings and a driver script's own run block do not count as edits
    (``_canonical_dump`` keeps it stable across Python minors); for any other
    file, or one that does not parse, its bytes. A value assigned inside the
    main block is not code the task reaches, even if the task reads it as a
    global: pass it as a config value instead, which the resume guards too."""
    import ast

    if tree is None:
        return _short_hash(path.read_bytes())
    body = []
    for node in tree.body:
        if _is_main_guard(node):
            body.extend(node.orelse)  # the else branch runs on import: it is code
        else:
            body.append(node)
    stripped = ast.Module(body=body, type_ignores=[])
    return _short_hash(_canonical_dump(stripped).encode())


def _imported_modules(tree, package: str | None) -> list[str]:
    """Every module name a file imports, at any depth (a function-local import
    included), each with its parent packages, since a package's ``__init__``
    is code that runs too; relative imports resolved against ``package``."""
    import ast
    import importlib.util

    names: list[str] = []

    def add(name: str) -> None:
        parts = name.split(".")
        for i in range(1, len(parts) + 1):
            dotted = ".".join(parts[:i])
            if dotted not in names:
                names.append(dotted)

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                if not package:
                    continue
                try:
                    base = importlib.util.resolve_name("." * node.level + base, package)
                except (ImportError, ValueError):
                    continue
            if base:
                add(base)
            for alias in node.names:
                if alias.name != "*" and base:
                    add(
                        f"{base}.{alias.name}"
                    )  # a submodule, or a name: resolved below
    return names


def _module_file(name: str) -> str | None:
    """The file a module name denotes: the loaded module's, or the one the
    import system would load for a top-level name, or a submodule of a loaded
    package (finding it imports nothing; ``from pkg import name`` where
    ``name`` is a function resolves to no file and is skipped)."""
    import importlib.util
    import sys

    mod = sys.modules.get(name)
    if mod is not None:
        return getattr(mod, "__file__", None)
    parent = name.rpartition(".")[0]
    if parent and parent not in sys.modules:
        return None
    try:
        spec = importlib.util.find_spec(name)
    except (ImportError, AttributeError, ValueError):
        return None
    return spec.origin if spec is not None and spec.has_location else None


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


def _referenced_globals(fn, module) -> list:
    """The values in ``module``'s globals that ``fn`` uses, following the
    module's own functions it calls (their code objects, nested ones too). For
    a task defined where there is no file (a notebook), so that only what the
    task uses counts, not everything the notebook has imported."""
    import types

    try:
        namespace = vars(module)
    except TypeError:
        return []
    seen_code: set[int] = set()
    seen_names: set[str] = set()
    values = []
    pending = [getattr(fn, "__code__", None)]
    while pending:
        code = pending.pop()
        if not isinstance(code, types.CodeType) or id(code) in seen_code:
            continue
        seen_code.add(id(code))
        pending.extend(c for c in code.co_consts if isinstance(c, types.CodeType))
        for name in code.co_names:
            if name in seen_names or name not in namespace:
                continue
            seen_names.add(name)
            value = namespace[name]
            values.append(value)
            inner = getattr(value, "__code__", None)
            if (
                inner is not None
                and getattr(value, "__module__", None) == module.__name__
            ):
                pending.append(inner)
            if isinstance(value, type) and value.__module__ == module.__name__:
                for attr in vars(value).values():
                    func = getattr(attr, "__func__", attr)
                    if isinstance(getattr(func, "__code__", None), types.CodeType):
                        pending.append(func.__code__)
    return values


_STALE_WARNED: set[str] = set()


class _Stale(Exception):
    """A reached file was edited after this process imported it."""


def project_fingerprints(fn, *more) -> dict[str, str] | None:
    """Hashes of the code in the project's own source files the task reaches,
    keyed by path relative to the task's module (so a script run as
    ``__main__`` and the same file imported by name agree, and a relocated
    checkout still matches; absolute when the task's module has no file, as in
    a notebook): the module that defines ``fn`` and, transitively, every
    project module a reached file imports, at any depth (``from consts import
    THRESH``, an import inside a function). For a task defined where there is
    no file (a notebook), the walk starts from the globals the task uses,
    following the notebook functions it calls, not from everything the notebook
    has imported. ``more`` adds further roots walked the same way (``Study``'s
    training functions, which its task calls but does not import); keys are
    then relative to the first root that has a file.

    A Python file is hashed as its syntax tree without the ``if __name__ ==
    "__main__":`` blocks (see ``_source_digest``), so an edit to a driver
    script's run block, a comment or the formatting is not a code change, while
    a constant or a helper in the same file is. None when a root's module
    cannot be found (a function built at runtime), or when a reached file was
    edited after this process imported it (with a warning, once per file: the
    code that would run is not the code on disk, so nothing is recorded under
    the new hash and a resume does not compare it; restart the process). An
    empty dict when the task reaches no project file (defined inside an
    installed package, or in a notebook that uses none).

    Complements ``code_fingerprint``, which covers the task's own body: an
    edited helper, a changed module constant or a modified imported module of
    the project changes this fingerprint, so a resume re-runs the cell and a
    ``cache_dir`` lookup misses, instead of returning a result computed by code
    that no longer exists. Not reached: installed packages (a dependency
    upgrade is provenance's job, and hashing site-packages would miss the cache
    on every unrelated install), data files, and a module the task only meets
    through the config (a class named by ``_target_`` whose module nothing the
    task's module imports)."""
    import warnings

    try:
        return _walk_project(fn, *more)
    except _Stale as stale:
        path = str(stale)
        if path not in _STALE_WARNED:
            _STALE_WARNED.add(path)
            # Outside the best-effort guard below, so a caller that turns
            # warnings into errors sees it rather than a silent None.
            warnings.warn(
                f"{path} was edited after this process imported it, so the code "
                "that would run is not the code on disk; the project code the "
                "task reaches is not fingerprinted for this run, and a resume "
                "does not compare it. Restart the process to run, and record, "
                "the edited code.",
                UserWarning,
                stacklevel=2,
            )
        return None
    except Exception:  # noqa: BLE001 - a best-effort guard never fails the sweep
        return None


def _walk_project(*roots) -> dict[str, str] | None:
    """``project_fingerprints`` without its guards: raises ``_Stale``."""
    import inspect
    import os
    import sys
    import types

    pending: list[Path] = []
    base: Path | None = None
    for root in roots:
        target = _unwrap_task(root)
        module = inspect.getmodule(target)
        if module is None and not isinstance(target, (types.FunctionType, type)):
            module = inspect.getmodule(type(target))
        if module is None:
            return None
        module_file = _project_file(module)
        if module_file is not None:
            base = base or module_file.parent
            pending.append(module_file)
            continue
        # No file (a notebook, or a task defined inside mushin): start from the
        # globals the task uses; the import walk below takes it from there.
        for value in _referenced_globals(target, module):
            if isinstance(value, types.ModuleType):
                path = _project_file(value)
            else:
                owner = _module_owner(value)
                path = _project_path(_module_file(owner)) if owner else None
            if path is not None:
                pending.append(path)
    # Loaded modules by file, for the staleness check and a module's package.
    loaded: dict[Path, object] = {}
    for mod in list(sys.modules.values()):
        f = getattr(mod, "__file__", None)
        if isinstance(f, str) and f:
            try:
                loaded.setdefault(Path(f).resolve(), mod)
            except OSError:
                continue
    hashes: dict[str, str] = {}
    visited: set[Path] = set()
    while pending:
        path = pending.pop()
        if path in visited:
            continue
        visited.add(path)
        mod = loaded.get(path)
        if mod is not None and _loaded_from_older_source(mod, path):
            raise _Stale(str(path))
        try:
            tree = _parse_source(path)
            digest = _source_digest(path, tree)
        except (OSError, RecursionError):
            continue
        key = path.as_posix()
        if base is not None:
            try:
                key = Path(os.path.relpath(path, base)).as_posix()
            except ValueError:  # another drive on Windows
                pass
        hashes[key] = digest
        if tree is None:
            continue
        package = getattr(mod, "__package__", None) if mod is not None else None
        if not isinstance(package, str):
            package = None
        for name in _imported_modules(tree, package):
            dep = _project_path(_module_file(name))
            if dep is not None and dep not in visited:
                pending.append(dep)
    return hashes


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
