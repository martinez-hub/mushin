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
    code_deps: dict | None = None,
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
    function's own source is hashed; the project code it reaches (helpers,
    module-level constants) is covered by ``project_fingerprints``, installed
    packages and data files by neither, so a fresh ``working_dir`` is still
    the safe choice after upgrading a dependency or changing the data."""
    import inspect

    try:
        # unwrap functools.wraps / decorators to the underlying function
        target = inspect.unwrap(fn) if callable(fn) else fn
        src = inspect.getsource(target)
    except (OSError, TypeError):
        return None
    import hashlib

    return hashlib.sha256(src.encode()).hexdigest()[:16]


def _short_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


#: The interpreter's own directories, by their sysconfig names: the standard
#: library and the installed packages. Not the bare prefixes: a project checked
#: out under the interpreter's prefix (/usr/src/app on a Debian system Python,
#: code kept inside a conda env) is still the project's own.
_FOREIGN_PATH_KEYS = ("stdlib", "platstdlib", "purelib", "platlib")


@functools.cache
def _foreign_roots() -> tuple[Path, ...]:
    """Directories whose Python files are not the project's own: the running
    interpreter's standard library and site-packages (its own, a venv's, the
    user's). The prefixes themselves are dropped, since on Windows
    ``site.getsitepackages()`` lists ``sys.prefix`` next to its
    ``Lib\\site-packages``."""
    import site
    import sys
    import sysconfig

    paths = sysconfig.get_paths()
    roots: set[str] = {paths[k] for k in _FOREIGN_PATH_KEYS if paths.get(k)}
    try:
        roots.update(site.getsitepackages())
        roots.add(site.getusersitepackages())
    except Exception:  # noqa: BLE001 - a restricted site module; nothing to add
        pass
    prefixes = set()
    for prefix in (sys.prefix, sys.base_prefix, sys.exec_prefix, sys.base_exec_prefix):
        try:
            prefixes.add(Path(prefix).resolve())
        except OSError:
            continue
    out = []
    for r in roots:
        if not r:
            continue
        try:
            path = Path(r).resolve()
        except OSError:
            continue
        if path not in prefixes:
            out.append(path)
    return tuple(out)


@functools.cache
def _installed_top_levels() -> frozenset[str]:
    """Top-level module names that belong to an installed distribution other
    than an editable one: code on ``PYTHONPATH`` from a cluster's ``module
    load``, a ``pip install --target`` directory. An editable install is the
    project itself (``pip install -e .``, a uv project), so its modules stay
    the project's. Read once per process."""
    import importlib.metadata as md
    import json

    editable: set[str] = set()
    for dist in md.distributions():
        try:
            direct = dist.read_text("direct_url.json")
            if direct and json.loads(direct).get("dir_info", {}).get("editable"):
                name = dist.metadata["Name"]
                if name:
                    editable.add(name.lower().replace("_", "-"))
        except Exception:  # noqa: BLE001 - a broken dist-info: not editable
            continue
    out = set()
    for top, dists in md.packages_distributions().items():
        if not all(d.lower().replace("_", "-") in editable for d in dists):
            out.add(top)
    return frozenset(out)


@functools.cache
def _is_foreign_module(name: str) -> bool:
    """True for a module that is not the project's own: the standard library,
    mushin, an installed (non-editable) package, or a file under the
    interpreter's library directories. Decided by name, once per process."""
    import sys

    top = name.partition(".")[0]
    if top in sys.stdlib_module_names or top in sys.builtin_module_names:
        return True
    if top == "mushin":
        return True
    module = sys.modules.get(name)
    file = getattr(module, "__file__", None) if module is not None else None
    if isinstance(file, str) and file:
        try:
            path = Path(file).resolve()
        except OSError:
            return True
        if any(path.is_relative_to(root) for root in _foreign_roots()):
            return True
    try:
        return top in _installed_top_levels()
    except Exception:  # noqa: BLE001 - unreadable metadata: by path alone
        return False


#: Values hashed by their repr when the task reads them as globals. Mutable
#: containers are left out: a module-level cache filled during a run would
#: otherwise look like an edit.
_CONSTANT_TYPES = (int, float, complex, str, bytes, bool, type(None))


def _constant_repr(value, depth: int = 0) -> str | None:
    """``repr`` of an immutable constant (tuples and frozensets of them too),
    or None for anything else."""
    if isinstance(value, _CONSTANT_TYPES):
        return repr(value)
    if isinstance(value, (tuple, frozenset)) and depth < 4 and len(value) <= 1000:
        items = [_constant_repr(v, depth + 1) for v in value]
        if any(i is None for i in items):
            return None
        if isinstance(value, frozenset):
            items.sort()
        return f"{type(value).__name__}({','.join(items)})"
    return None


def _code_hash(code, nested: dict | None = None) -> str:
    """The hash of what a code object does: its bytecode, names, constants and
    nested code, not its file name or line numbers, so a comment, a blank line
    or a moved function is not an edit. Bytecode is stable within a Python
    minor version; the sidecar records the version, and a resume under another
    one does not compare."""
    import types

    parts = [
        code.co_code,
        repr(code.co_names).encode(),
        repr(code.co_varnames).encode(),
        repr(code.co_freevars).encode(),
        repr(code.co_cellvars).encode(),
        repr((code.co_argcount, code.co_kwonlyargcount, code.co_flags)).encode(),
    ]
    for const in code.co_consts:
        if isinstance(const, types.CodeType):
            parts.append(_code_hash(const).encode())
        else:
            parts.append(repr(const).encode())
    return _short_hash(b"\x00".join(parts))


def _code_names(code) -> set[str]:
    """Every global or attribute name a code object and its nested code use."""
    import types

    names: set[str] = set()
    pending = [code]
    while pending:
        c = pending.pop()
        names.update(c.co_names)
        pending.extend(k for k in c.co_consts if isinstance(k, types.CodeType))
    return names


def _function_imports(code, package: str | None) -> list[str]:
    """The modules a function imports in its body (``import x.y``, ``from .a
    import b``), resolved against ``package``; each parent package included."""
    import dis
    import importlib.util
    import types

    found: list[str] = []
    pending = [code]
    while pending:
        c = pending.pop()
        pending.extend(k for k in c.co_consts if isinstance(k, types.CodeType))
        level = 0
        for ins in dis.get_instructions(c):
            if ins.opname == "LOAD_CONST" and isinstance(ins.argval, int):
                level = ins.argval
            elif ins.opname == "IMPORT_NAME" and isinstance(ins.argval, str):
                name = ins.argval
                if level:
                    if not package:
                        continue
                    try:
                        name = importlib.util.resolve_name(
                            "." * level + name, package
                        ).rstrip(".")
                    except (ImportError, ValueError):
                        continue
                parts = name.split(".")
                for i in range(1, len(parts) + 1):
                    dotted = ".".join(parts[:i])
                    if dotted and dotted not in found:
                        found.append(dotted)
    return found


def _unloaded_module_hash(name: str) -> tuple[str, str] | None:
    """A project module the task imports in its body but this process has not
    imported yet: its file's hash, found without importing it. None when it is
    foreign, already loaded, or cannot be located."""
    import importlib.util
    import sys

    if name in sys.modules or _is_foreign_module(name):
        return None
    parent = name.rpartition(".")[0]
    if parent and parent not in sys.modules:
        return None  # finding a submodule would import its parent
    try:
        spec = importlib.util.find_spec(name)
    except (ImportError, AttributeError, ValueError):
        return None
    if spec is None or not spec.has_location or not spec.origin:
        return None
    try:
        path = Path(spec.origin).resolve()
        if any(path.is_relative_to(root) for root in _foreign_roots()):
            return None
        return f"{name} (file)", _short_hash(path.read_bytes())
    except OSError:
        return None


class _CodeWalk:
    """Walks from a task through the names its code uses, recording a hash for
    every project function, class and constant it reaches, keyed by where the
    name was looked up (``helpers:scale``), so a module attribute rebound to
    another function is seen as a change."""

    def __init__(self, roots):
        import sys

        self.labels: dict[str, str] = {}
        for root in roots:
            module = getattr(_unwrap_task(root), "__module__", None)
            if isinstance(module, str):
                mod = sys.modules.get(module)
                file = getattr(mod, "__file__", None) if mod is not None else None
                # The task's module is named by its file, so a script run as
                # __main__ and the same file imported by name agree; a module
                # with no file (a notebook) is __main__ either way.
                self.labels[module] = (
                    Path(file).stem if isinstance(file, str) and file else module
                )
        self.hashes: dict[str, str] = {}
        self._object_hash: dict[int, str] = {}
        self._walked: set[int] = set()

    def label(self, module: str) -> str:
        return self.labels.get(module, module)

    def walk_root(self, root) -> None:
        target = _unwrap_task(root)
        code = getattr(target, "__code__", None)
        if code is not None:
            # The root's own body is covered by code_fingerprint (the task) or
            # by Study's methods fingerprint; walk only what it uses.
            self._walk_function(target, record_as=None)
        elif isinstance(target, type):
            self._walk_class(target, record_as=None)
        else:
            self._walk_class(type(target), record_as=None)

    def _project(self, value) -> bool:
        module = getattr(value, "__module__", None)
        return isinstance(module, str) and not _is_foreign_module(module)

    def _record(self, key: str | None, digest: str) -> None:
        if key is not None:
            self.hashes[key] = digest

    def _walk_function(self, fn, record_as: str | None) -> None:
        import sys

        code = fn.__code__
        digest = self._object_hash.get(id(fn))
        if digest is None:
            extras = [_code_hash(code)]
            for value in (fn.__defaults__ or ()) + tuple(
                (fn.__kwdefaults__ or {}).values()
            ):
                extras.append(_constant_repr(value) or type(value).__qualname__)
            digest = _short_hash("|".join(extras).encode())
            self._object_hash[id(fn)] = digest
        self._record(record_as, digest)
        if id(fn) in self._walked:
            return
        self._walked.add(id(fn))
        names = _code_names(code)
        module = fn.__module__ if isinstance(fn.__module__, str) else "?"
        here = self.label(module)
        namespace = getattr(fn, "__globals__", {}) or {}
        for name in sorted(names):
            if name in namespace:
                self._walk_value(namespace[name], f"{here}:{name}", names)
        for cell_name, cell in zip(
            code.co_freevars, fn.__closure__ or (), strict=False
        ):
            try:
                value = cell.cell_contents
            except ValueError:  # an empty cell
                continue
            self._walk_value(value, f"{here}:{fn.__qualname__}.<{cell_name}>", names)
        package = namespace.get("__package__")
        package = package if isinstance(package, str) else None
        for name in _function_imports(code, package):
            loaded = sys.modules.get(name)
            if loaded is not None:
                self._walk_value(loaded, f"{name}:", names)
            else:
                found = _unloaded_module_hash(name)
                if found is not None:
                    self._record(*found)

    def _walk_class(self, cls, record_as: str | None) -> None:
        import types

        if not self._project(cls):
            return
        digest = self._object_hash.get(id(cls))
        if digest is None:
            parts = [cls.__qualname__]
            members = []
            for attr, value in sorted(vars(cls).items()):
                func = getattr(value, "__func__", value)
                code = getattr(func, "__code__", None)
                if isinstance(code, types.CodeType):
                    members.append((attr, func))
                    parts.append(f"{attr}={_code_hash(code)}")
                else:
                    const = _constant_repr(value)
                    if const is not None:
                        parts.append(f"{attr}={const}")
            parts.extend(f"base={b.__module__}.{b.__qualname__}" for b in cls.__bases__)
            digest = _short_hash("|".join(parts).encode())
            self._object_hash[id(cls)] = digest
        else:
            members = []
        self._record(record_as, digest)
        if id(cls) in self._walked:
            return
        self._walked.add(id(cls))
        if not members:
            for attr, value in vars(cls).items():
                func = getattr(value, "__func__", value)
                if isinstance(getattr(func, "__code__", None), types.CodeType):
                    members.append((attr, func))
        for _, func in members:
            if self._project(func):
                self._walk_function(func, record_as=None)
        for base in cls.__bases__:
            if self._project(base):
                key = f"{self.label(base.__module__)}:{base.__qualname__}"
                self._walk_class(base, record_as=key)

    def _walk_value(self, value, key: str, names: set[str]) -> None:
        import functools as _ft
        import types

        try:
            if isinstance(value, types.ModuleType):
                if id(value) in self._walked or _is_foreign_module(value.__name__):
                    return
                self._walked.add(id(value))
                namespace = vars(value)
                here = self.label(value.__name__)
                for name in sorted(names):
                    if name in namespace:
                        self._walk_value(namespace[name], f"{here}:{name}", names)
                return
            if isinstance(value, _ft.partial):
                self._walk_value(value.func, key + ".func", names)
                for i, arg in enumerate(value.args):
                    const = _constant_repr(arg)
                    if const is not None:
                        self._record(f"{key}.args[{i}]", _short_hash(const.encode()))
                return
            if isinstance(value, types.MethodType):
                value = value.__func__
            if isinstance(value, types.FunctionType):
                if self._project(value):
                    self._walk_function(value, record_as=key)
                return
            if isinstance(value, type):
                self._walk_class(value, record_as=key)
                return
            const = _constant_repr(value)
            if const is not None:
                self._record(key, _short_hash(const.encode()))
                return
            # An instance of a project class (a model, a config object): its
            # class is code the task runs; its state is not hashed.
            if self._project(type(value)):
                self._walk_class(type(value), record_as=f"{key}.__class__")
        except Exception:  # noqa: BLE001 - a global that resists inspection
            return


def project_fingerprints(fn, *more) -> dict | None:
    """What the project code a task reaches looks like now, as recorded in
    each cell's status sidecar: ``{"python": "3.11", "code": {key: hash}}``.

    The walk starts from the names the task's code uses and follows them
    through the project's own functions, classes, modules and constants, as
    loaded in this process, which is the code that will run. Each reached
    project function or class is hashed by its bytecode (not its text: a
    comment, a blank line or a moved definition is not an edit), and each
    immutable constant it reads (``SCALE = 10``, or ``THRESH`` imported with
    ``from consts import THRESH``) by its value. Keys say where a name was
    looked up, ``helpers:scale`` or, for the task's own module, its file's
    name, so a script run as ``__main__`` and the same file imported by name
    agree. A project module the task imports inside its body that this process
    has not imported yet is hashed by its file. ``more`` adds roots walked the
    same way (``Study``'s training functions, which its task calls but does not
    import). The roots' own bodies are not entries: ``code_fingerprint`` and
    ``Study``'s methods fingerprint cover them.

    Not reached: the standard library, mushin, installed packages other than
    editable ones (a dependency upgrade is provenance's job), data files, a
    mutable module-level container, code called only through a string (a
    class named by ``_target_`` whose module nothing the task uses), and
    attribute chains on objects rather than modules. None when the walk fails;
    a resume then does not compare, and ``cache_dir`` is not used."""
    import sys

    try:
        roots = (fn, *more)
        walk = _CodeWalk(roots)
        for root in roots:
            walk.walk_root(root)
        return {
            "python": f"{sys.version_info[0]}.{sys.version_info[1]}",
            "code": walk.hashes,
        }
    except Exception:  # noqa: BLE001 - a best-effort guard never fails the sweep
        return None


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


def changed_project_code(prior, current) -> list[str] | None:
    """The keys whose hashes differ between two ``project_fingerprints``
    records, compared on the keys both have (a name one run reached and the
    other did not is not an edit: reaching it changed some code both reached,
    or it was imported lazily in one process only). None when they cannot be
    compared: either is missing or malformed (a sidecar from before the field),
    or they were taken under different Python versions, whose bytecode
    differs (an interpreter upgrade is, like a package upgrade, provenance's
    job)."""
    if not isinstance(prior, dict) or not isinstance(current, dict):
        return None
    if prior.get("python") != current.get("python"):
        return None
    a, b = prior.get("code"), current.get("code")
    if not isinstance(a, dict) or not isinstance(b, dict):
        return None
    return sorted(k for k in a.keys() & b.keys() if a[k] != b[k])


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
