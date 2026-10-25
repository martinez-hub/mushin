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
def _installed_roots() -> dict[str, tuple[Path, ...]]:
    """For each top-level module name an installed distribution provides
    (other than an editable one), the directories that distribution is
    installed into: code on ``PYTHONPATH`` from a cluster's ``module load``, a
    ``pip install --target`` directory. An editable install is the project
    itself (``pip install -e .``, a uv project), so its modules stay the
    project's. Read once per process."""
    import importlib.metadata as md
    import json

    by_name: dict[str, set[Path]] = {}
    for dist in md.distributions():
        try:
            direct = dist.read_text("direct_url.json")
            if direct and json.loads(direct).get("dir_info", {}).get("editable"):
                continue
            where = Path(str(dist.locate_file(""))).resolve()
            tops = (dist.read_text("top_level.txt") or "").split()
            if not tops:
                tops = {
                    Path(str(f)).parts[0].removesuffix(".py")
                    for f in (dist.files or [])
                    if Path(str(f)).parts and not str(f).startswith("..")
                }
        except Exception:  # noqa: BLE001 - a broken dist-info: skip it
            continue
        for top in tops:
            by_name.setdefault(top, set()).add(where)
    return {name: tuple(paths) for name, paths in by_name.items()}


@functools.cache
def _is_foreign_module(name: str) -> bool:
    """True for a module that is not the project's own: the standard library,
    mushin, a file under the interpreter's library directories, or a file
    inside the install directory of a non-editable distribution that provides
    the module's top-level name. A module with a file elsewhere is the
    project's, even when an installed distribution has the same top-level name
    (a checkout's ``src`` or ``utils``, or a project installed non-editable and
    run from its checkout). Decided once per process."""
    import sys

    top = name.partition(".")[0]
    if top in sys.stdlib_module_names or top in sys.builtin_module_names:
        return True
    if top == "mushin":
        return True
    module = sys.modules.get(name)
    file = getattr(module, "__file__", None) if module is not None else None
    try:
        installed = _installed_roots().get(top, ())
    except Exception:  # noqa: BLE001 - unreadable metadata: by path alone
        installed = ()
    if isinstance(file, str) and file:
        try:
            path = Path(file).resolve()
        except OSError:
            return True
        roots = _foreign_roots() + installed
        return any(path.is_relative_to(root) for root in roots)
    return bool(installed)


#: Values hashed by their repr when the task reads them. Mutable containers
#: are hashed by their type only (a module-level cache filled during a run
#: would otherwise look like an edit), and the project code they hold is
#: walked.
_CONSTANT_TYPES = (int, float, complex, str, bytes, bool, type(None))


def _constant_repr(value, depth: int = 0) -> str | None:
    """``repr`` of an immutable constant (tuples and frozensets of them too, in
    a sorted order for frozensets, whose iteration order follows string
    hashing; and an enum member with its value), or None for anything else."""
    import enum

    if isinstance(value, enum.Enum):
        inner = _constant_repr(value.value, depth + 1)
        return f"{type(value).__qualname__}.{value.name}={inner}" if inner else None
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


def _code_hash(code) -> str:
    """The hash of what a code object does: its bytecode, names, constants and
    nested code, not its file name or line numbers, so a comment, a blank line
    or a moved function is not an edit. Bytecode is stable within a Python
    minor version; the record carries the version, and a resume under another
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
            # A frozenset constant (``x in {"a", "b"}``) prints in string-hash
            # order, which differs per process: sorted, through _constant_repr.
            parts.append((_constant_repr(const) or repr(const)).encode())
    return _short_hash(b"\x00".join(parts))


#: Opcodes that read a name as a global (or from a class or module namespace),
#: as opposed to an attribute of some object.
_GLOBAL_OPS = frozenset(
    {
        "LOAD_GLOBAL",
        "LOAD_NAME",
        "STORE_GLOBAL",
        "DELETE_GLOBAL",
        "LOAD_FROM_DICT_OR_GLOBALS",
    }
)
_ATTR_OPS = frozenset(
    {
        "LOAD_ATTR",
        "LOAD_METHOD",
        "STORE_ATTR",
        "DELETE_ATTR",
        "IMPORT_FROM",
        "LOAD_SUPER_ATTR",
    }
)
_INT_LOADS = frozenset({"LOAD_CONST", "LOAD_SMALL_INT"})


def _code_names(code) -> tuple[set[str], set[str], list[tuple[str, int]]]:
    """The global names a code object (and its nested code) reads, the
    attribute names it uses, and the modules it imports as ``(name, level)``
    pairs, read from its instructions so an attribute named like an unrelated
    global (``random.seed`` next to a global ``seed``) is not mistaken for it."""
    import dis
    import types

    globals_: set[str] = set()
    attrs: set[str] = set()
    imports: list[tuple[str, int]] = []
    pending = [code]
    while pending:
        c = pending.pop()
        pending.extend(k for k in c.co_consts if isinstance(k, types.CodeType))
        instructions = list(dis.get_instructions(c))
        for i, ins in enumerate(instructions):
            name = ins.argval
            if not isinstance(name, str):
                continue
            if ins.opname in _GLOBAL_OPS:
                globals_.add(name)
            elif ins.opname in _ATTR_OPS:
                attrs.add(name)
            elif ins.opname == "IMPORT_NAME":
                # The level is the integer loaded two instructions before:
                # LOAD_CONST (LOAD_SMALL_INT on 3.14) level, the fromlist,
                # then IMPORT_NAME.
                level = 0
                if i >= 2 and instructions[i - 2].opname in _INT_LOADS:
                    arg = instructions[i - 2].argval
                    level = arg if isinstance(arg, int) else 0
                imports.append((name, level))
    return globals_, attrs, imports


def _resolve_imports(imports, package: str | None) -> list[str]:
    """Absolute module names for ``(name, level)`` imports, each with its
    parent packages."""
    import importlib.util

    found: list[str] = []
    for name, level in imports:
        if level:
            if not package:
                continue
            try:
                name = importlib.util.resolve_name("." * level + name, package).rstrip(
                    "."
                )
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


def _module_label(module_name: str) -> str:
    """How a module is named in a key: by its import name, except a script run
    as ``__main__``, which is named as ``python -m`` names it (its spec) or by
    its file, so the run and a later ``import`` of the same file agree. A
    notebook's ``__main__`` has neither and stays ``__main__``."""
    import sys

    if module_name != "__main__":
        return module_name
    main = sys.modules.get("__main__")
    spec = getattr(main, "__spec__", None)
    if spec is not None and getattr(spec, "name", None):
        return spec.name
    file = getattr(main, "__file__", None)
    return Path(file).stem if isinstance(file, str) and file else "__main__"


class _CodeWalk:
    """Walks from the roots through the names their code uses, recording a hash
    for every project function, class and constant reached, keyed by where the
    name was looked up (``helpers:scale``), so a module attribute rebound to
    another function is seen as a change. A name that resolves to something not
    hashed (a mutable container, a foreign function) gets an entry naming what
    it is, so an edit that turns a recorded value into one of those still
    shows."""

    def __init__(self):
        self.hashes: dict[str, str] = {}
        self._object_hash: dict[int, str] = {}
        self._walked: set[int] = set()
        self._module_names: dict[int, set[str]] = {}

    # -- helpers ------------------------------------------------------------
    def _project(self, value) -> bool:
        try:
            module = getattr(value, "__module__", None)
        except Exception:  # noqa: BLE001 - a descriptor that raises
            return False
        return isinstance(module, str) and not _is_foreign_module(module)

    def _record(self, key: str | None, digest: str) -> None:
        if key is not None:
            self.hashes[key] = digest

    def _kind(self, value) -> str:
        """A stand-in for a value that is not hashed: its type, and for a
        callable its qualified name."""
        t = type(value)
        name = getattr(value, "__qualname__", None)
        module = getattr(value, "__module__", None)
        if isinstance(name, str) and isinstance(module, str):
            return _short_hash(f"<{t.__qualname__} {module}.{name}>".encode())
        return _short_hash(f"<{t.__module__}.{t.__qualname__}>".encode())

    # -- roots --------------------------------------------------------------
    def walk_root(self, root, *, with_body: bool = False) -> None:
        target = _unwrap_task(root)
        if isinstance(target, functools.partial):
            target = target.func
        code = getattr(target, "__code__", None)
        if code is not None:
            # The task's own body is covered by code_fingerprint, and Study's
            # training functions' by its methods fingerprint, so only what they
            # use is walked; a root nothing else covers (pre_task) is recorded
            # with its body.
            key = None
            if with_body and self._project(target):
                key = f"{_module_label(target.__module__)}:{target.__qualname__}"
            self._walk_function(target, record_as=key)
        elif isinstance(target, type):
            self._walk_class(target, record_as=None)
        elif target is not None:
            self._walk_class(type(target), record_as=None)

    def walk_target(self, path: str) -> None:
        """A class or function the config names as ``_target_``: resolved from
        the modules this process has loaded (never importing one)."""
        import sys

        parts = path.split(".")
        for i in range(len(parts) - 1, 0, -1):
            module = sys.modules.get(".".join(parts[:i]))
            if module is None:
                continue
            value = module
            try:
                for attr in parts[i:]:
                    value = getattr(value, attr)
            except Exception:  # noqa: BLE001 - not there, or raises
                return
            self._walk_value(value, f"target:{path}", set(), set())
            return

    # -- values -------------------------------------------------------------
    def _function_digest(self, fn) -> str:
        """A function's code and its default arguments (a default is part of
        what the function does, and lives outside its code object)."""
        digest = self._object_hash.get(id(fn))
        if digest is None:
            extras = [_code_hash(fn.__code__)]
            for value in (fn.__defaults__ or ()) + tuple(
                (fn.__kwdefaults__ or {}).values()
            ):
                extras.append(_constant_repr(value) or self._kind(value))
            digest = _short_hash("|".join(extras).encode())
            self._object_hash[id(fn)] = digest
        return digest

    def _walk_function(self, fn, record_as: str | None) -> None:
        import sys

        code = fn.__code__
        self._record(record_as, self._function_digest(fn))
        globals_, attrs, imports = _code_names(code)
        module = fn.__module__ if isinstance(fn.__module__, str) else "?"
        here = _module_label(module)
        # Closure cells belong to this function object, so they are keyed
        # under where it was reached: two closures from one factory differ.
        owner = record_as or f"{here}:{fn.__qualname__}"
        for cell_name, cell in zip(
            code.co_freevars, fn.__closure__ or (), strict=False
        ):
            try:
                value = cell.cell_contents
            except ValueError:  # an empty cell
                continue
            self._walk_value(value, f"{owner}.<{cell_name}>", globals_, attrs)
        if id(fn) in self._walked:
            return
        self._walked.add(id(fn))
        namespace = getattr(fn, "__globals__", {}) or {}
        for name in sorted(globals_):
            if name in namespace:
                self._walk_value(namespace[name], f"{here}:{name}", globals_, attrs)
        package = namespace.get("__package__")
        package = package if isinstance(package, str) else None
        for name in _resolve_imports(imports, package):
            loaded = sys.modules.get(name)
            if loaded is not None:
                self._walk_module(loaded, attrs)
            else:
                found = _unloaded_module_hash(name)
                if found is not None:
                    self._record(*found)

    def _walk_module(self, module, attrs: set[str]) -> None:
        """The attributes of a project module that the reaching code uses. A
        module reached again by other code is walked again for the names that
        code uses."""
        if _is_foreign_module(module.__name__):
            return
        seen = self._module_names.setdefault(id(module), set())
        new = attrs - seen
        if not new:
            return
        seen.update(new)
        try:
            namespace = vars(module)
        except TypeError:
            return
        here = _module_label(module.__name__)
        for name in sorted(new):
            if name in namespace:
                self._walk_value(namespace[name], f"{here}:{name}", set(), attrs)

    def _class_members(self, cls):
        """``(name, kind, value)`` for each member that is code: functions,
        static and class methods, properties (getter, setter, deleter),
        cached properties, and nested project classes."""
        import types

        out = []
        for attr, value in sorted(vars(cls).items()):
            if attr.startswith("__") and attr.endswith("__"):
                func = getattr(value, "__func__", value)
                if not isinstance(getattr(func, "__code__", None), types.CodeType):
                    continue  # __module__, __doc__, __firstlineno__ (3.13+)
            if isinstance(value, property):
                for part, f in (
                    ("fget", value.fget),
                    ("fset", value.fset),
                    ("fdel", value.fdel),
                ):
                    if f is not None:
                        out.append((f"{attr}.{part}", "function", f))
                continue
            func = (
                getattr(value, "func", None)
                if type(value).__name__ == "cached_property"
                else None
            )
            func = func or getattr(value, "__func__", value)
            if isinstance(getattr(func, "__code__", None), types.CodeType):
                out.append((attr, "function", func))
            elif isinstance(value, type):
                out.append((attr, "class", value))
            else:
                out.append((attr, "value", value))
        return out

    def _walk_class(self, cls, record_as: str | None) -> None:
        if not self._project(cls):
            if record_as is not None:
                self._record(record_as, self._kind(cls))
            return
        members = self._class_members(cls)
        digest = self._object_hash.get(id(cls))
        if digest is None:
            parts = [cls.__qualname__]
            for attr, kind, value in members:
                if kind == "function":
                    parts.append(f"{attr}={self._function_digest(value)}")
                elif kind == "class":
                    parts.append(f"{attr}=class {value.__qualname__}")
                else:
                    const = _constant_repr(value)
                    if const is not None:
                        parts.append(f"{attr}={const}")
            parts.extend(
                f"base={_module_label(b.__module__)}.{b.__qualname__}"
                for b in cls.__bases__
            )
            digest = _short_hash("|".join(parts).encode())
            self._object_hash[id(cls)] = digest
        self._record(record_as, digest)
        if id(cls) in self._walked:
            return
        self._walked.add(id(cls))
        for _attr, kind, value in members:
            if kind == "function" and self._project(value):
                self._walk_function(value, record_as=None)
            elif kind == "class" and self._project(value):
                key = f"{_module_label(value.__module__)}:{value.__qualname__}"
                self._walk_class(value, record_as=key)
        for base in cls.__bases__:
            if self._project(base):
                key = f"{_module_label(base.__module__)}:{base.__qualname__}"
                self._walk_class(base, record_as=key)

    def _walk_value(
        self, value, key: str, globals_: set[str], attrs: set[str], depth: int = 0
    ) -> None:
        import types

        try:
            if isinstance(value, types.ModuleType):
                self._walk_module(value, attrs)
                return
            if isinstance(value, functools.partial):
                parts = [_constant_repr(a) or self._kind(a) for a in value.args]
                parts += [
                    f"{k}={_constant_repr(v) or self._kind(v)}"
                    for k, v in sorted((value.keywords or {}).items())
                ]
                self._record(key, _short_hash("|".join(parts).encode()))
                self._walk_value(value.func, key + ".func", globals_, attrs, depth)
                return
            if isinstance(value, types.MethodType):
                value = value.__func__
            wrapped = getattr(value, "__wrapped__", None)
            if wrapped is not None and wrapped is not value:
                # A decorated function (functools.lru_cache, functools.wraps):
                # the wrapper may be foreign or a C object; the code that runs
                # is what it wraps.
                self._walk_value(wrapped, key + ".__wrapped__", globals_, attrs, depth)
            if isinstance(value, types.FunctionType):
                if self._project(value):
                    self._walk_function(value, record_as=key)
                else:
                    self._record(key, self._kind(value))
                return
            if isinstance(value, type):
                self._walk_class(value, record_as=key)
                return
            const = _constant_repr(value)
            if const is not None:
                self._record(key, _short_hash(const.encode()))
                return
            if isinstance(value, (dict, list, tuple, set, frozenset)):
                # A registry such as {"cnn": CNN, "mlp": mlp}: its type is
                # recorded (not its contents, which may change at runtime), and
                # the project code it holds is walked.
                self._record(key, self._kind(value))
                if depth < 2 and len(value) <= 1000:
                    items = (
                        sorted(value.items(), key=lambda kv: repr(kv[0]))
                        if isinstance(value, dict)
                        else enumerate(value)
                    )
                    for k, v in items:
                        if callable(v) or isinstance(v, (dict, list, tuple)):
                            self._walk_value(
                                v, f"{key}[{k!r}]", globals_, attrs, depth + 1
                            )
                return
            # An instance of a project class (a model, a config object): its
            # class is code the task runs; its state is not hashed.
            if self._project(type(value)):
                self._walk_class(type(value), record_as=f"{key}.__class__")
            else:
                self._record(key, self._kind(value))
        except Exception:  # noqa: BLE001 - a global that resists inspection
            return


def _config_targets(cfg) -> list[str]:
    """Every ``_target_`` string in a cell's config, at any depth."""
    try:
        from omegaconf import OmegaConf

        data = (
            OmegaConf.to_container(cfg, resolve=True)
            if OmegaConf.is_config(cfg)
            else cfg
        )
    except Exception:  # noqa: BLE001 - an unresolvable config: no targets
        return []
    found: list[str] = []
    pending = [data]
    while pending:
        item = pending.pop()
        if isinstance(item, dict):
            target = item.get("_target_")
            if isinstance(target, str) and target not in found:
                found.append(target)
            pending.extend(item.values())
        elif isinstance(item, (list, tuple)):
            pending.extend(item)
    return sorted(found)


def project_fingerprints(fn, *more, cfg=None, with_body=()) -> dict | None:
    """What the project code a task reaches looks like now, as recorded in
    each cell's status sidecar: ``{"python": "3.11", "code": {key: hash}}``.

    The walk starts from the names the task's code uses and follows them
    through the project's own functions, classes, modules and constants, as
    loaded in this process, which is the code that will run (a cell computes
    it in its own process, so a parallel worker records what it imported).
    Each reached project function or class is hashed by its bytecode (not its
    text: a comment, a blank line or a moved definition is not an edit), each
    immutable constant it reads (``SCALE = 10``, or ``THRESH`` imported with
    ``from consts import THRESH``) by its value, and the project functions and
    classes held in a container it reads (a registry dict) are walked. Keys say
    where a name was looked up, ``helpers:scale``. A project module the task
    imports inside its body that this process has not imported yet is hashed
    by its file. ``more`` adds roots walked the same way (``pre_task``, and
    ``Study``'s training functions, which its task calls but does not import),
    and ``cfg`` adds the classes and functions the config names as
    ``_target_``, resolved from the modules already loaded. The roots' own
    bodies are not entries (``code_fingerprint`` and ``Study``'s methods
    fingerprint cover them), except those in ``with_body`` (``pre_task``).

    Not reached: the standard library, mushin, installed packages other than
    editable ones (a dependency upgrade is provenance's job), data files, the
    contents of a mutable container other than the code it holds, code called
    only through a string that is not a ``_target_`` of the config, a
    ``_target_`` whose module this process has not imported, and attribute
    chains on objects rather than modules. None when the walk fails; a resume
    then does not compare, and ``cache_dir`` is not used."""
    import sys

    try:
        walk = _CodeWalk()
        for root in (fn, *more):
            if root is not None:
                walk.walk_root(root)
        for root in with_body:
            if root is not None:
                walk.walk_root(root, with_body=True)
        if cfg is not None:
            for target in _config_targets(cfg):
                walk.walk_target(target)
        return {
            "python": f"{sys.version_info[0]}.{sys.version_info[1]}",
            "code": walk.hashes,
        }
    except Exception:  # noqa: BLE001 - a best-effort guard never fails the sweep
        return None


#: The roots' record per run token, in this process: a parallel worker walks
#: once per run however many cells it runs. Bounded, since a reused worker
#: process (loky keeps them across run() calls) sees many runs.
_ROOT_RECORDS: dict[str, dict | None] = {}
_ROOT_RECORDS_MAX = 64


def cell_project_record(token: str, roots, with_body, cfg) -> dict | None:
    """The project-code record of one cell, taken in the process that runs it:
    the roots' walk (cached per process for the run ``token``) plus the
    ``_target_`` classes and functions of this cell's config, which can differ
    from cell to cell (a swept model class)."""
    if token in _ROOT_RECORDS:
        base = _ROOT_RECORDS[token]
    else:
        base = project_fingerprints(*roots, with_body=with_body) if roots else None
        while len(_ROOT_RECORDS) >= _ROOT_RECORDS_MAX:
            _ROOT_RECORDS.pop(next(iter(_ROOT_RECORDS)))
        _ROOT_RECORDS[token] = base
    if base is None:
        return None
    targets = _config_targets(cfg) if cfg is not None else []
    if not targets:
        return base
    try:
        walk = _CodeWalk()
        for target in targets:
            walk.walk_target(target)
    except Exception:  # noqa: BLE001 - best effort: the roots' record stands
        return base
    return {"python": base["python"], "code": {**base["code"], **walk.hashes}}


def merge_project_records(prior, current):
    """The record to keep for a cell reused under ``current``: its prior keys
    as well as the current ones, so a name only one kind of process reaches
    (a module imported lazily, hashed by file in a fresh process) stays
    comparable for a later resume. Under another Python version, or when
    either record is missing, the current one (or else the prior) is kept."""
    if current is None:
        return prior if isinstance(prior, dict) else None
    if (
        not isinstance(prior, dict)
        or not isinstance(prior.get("code"), dict)
        or prior.get("python") != current.get("python")
    ):
        return current
    return {"python": current["python"], "code": {**prior["code"], **current["code"]}}


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
