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
import re
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
            # A legacy develop install (setup.py develop, an .egg-link) keeps its
            # .egg-info in the project's own tree, not in a site directory:
            # editable too.
            meta = getattr(dist, "_path", None)
            if meta is not None and str(meta).endswith(".egg-info"):
                if not {"site-packages", "dist-packages"} & set(Path(str(meta)).parts):
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
def _foreign_by_path(file: str, top: str) -> bool:
    try:
        path = Path(file).resolve()
    except OSError:
        return True
    try:
        installed = _installed_roots().get(top, ())
    except Exception:  # noqa: BLE001 - unreadable metadata: by path alone
        installed = ()
    return any(path.is_relative_to(root) for root in _foreign_roots() + installed)


def _is_foreign_module(name: str, file: str | None = None) -> bool:
    """True for a module that is not the project's own: the standard library,
    mushin, a file under the interpreter's library directories, or a file
    inside the install directory of a non-editable distribution that provides
    the module's top-level name. A module with a file elsewhere is the
    project's, even when an installed distribution has the same top-level name
    (a checkout's ``src`` or ``utils``, or a project installed non-editable and
    run from its checkout). ``__main__`` is the user's program, never foreign:
    in a parallel worker ``sys.modules["__main__"]`` is the launcher's own
    module, while the task's functions keep ``__main__`` as their module.
    ``file`` overrides the file looked up in ``sys.modules`` (a function's
    ``__globals__`` knows its own). Answers by file are cached; a module not
    imported yet is decided by name each time, so a later import is seen."""
    import sys

    if name == "__main__":
        return False
    top = name.partition(".")[0]
    if top in sys.stdlib_module_names or top in sys.builtin_module_names:
        return True
    if top == "mushin":
        return True
    if file is None:
        module = sys.modules.get(name)
        file = getattr(module, "__file__", None) if module is not None else None
    if isinstance(file, str) and file:
        return _foreign_by_path(file, top)
    try:
        return bool(_installed_roots().get(top, ()))
    except Exception:  # noqa: BLE001 - unreadable metadata
        return False


# Tests reset the by-file answers through the old name.
_is_foreign_module.cache_clear = _foreign_by_path.cache_clear  # type: ignore[attr-defined]


#: Values hashed by their repr when the task reads them. Mutable containers
#: are hashed by their type only (a module-level cache filled during a run
#: would otherwise look like an edit), and the project code they hold is
#: walked.
_CONSTANT_TYPES = (int, float, complex, str, bytes, bool, type(None))


def _constant_repr(value, depth: int = 0) -> str | None:
    """``repr`` of an immutable constant (tuples and frozensets of them too, in
    a sorted order for frozensets, whose iteration order follows string
    hashing; an enum member with its value; a path; a dataclass instance by its
    fields, a settings object), or None for anything else. A secret-shaped
    string becomes a marker, so its hash is never written anywhere."""
    import dataclasses
    import enum
    import pathlib

    if isinstance(value, enum.Enum):
        inner = _constant_repr(value.value, depth + 1)
        return f"{type(value).__qualname__}.{value.name}={inner}" if inner else None
    if isinstance(value, str) and _secret_value(value):
        return "<secret>"
    if isinstance(value, _CONSTANT_TYPES):
        return repr(value)
    if isinstance(value, pathlib.PurePath):
        return f"path({value.as_posix()!r})"
    if (
        dataclasses.is_dataclass(value)
        and not isinstance(value, type)
        and type(value).__dataclass_params__.frozen
        and depth < 4
    ):
        fields = []
        for f in dataclasses.fields(value):
            v = getattr(value, f.name)
            c = "<secret>" if _secret_name(f.name, v) else _constant_repr(v, depth + 1)
            fields.append(f"{f.name}={c if c is not None else type(v).__qualname__}")
        return f"{type(value).__qualname__}({','.join(fields)})"
    if isinstance(value, (tuple, frozenset)) and depth < 4 and len(value) <= 1000:
        items = [_constant_repr(v, depth + 1) for v in value]
        if any(i is None for i in items):
            return None
        if isinstance(value, frozenset):
            items.sort()
        return f"{type(value).__name__}({','.join(items)})"
    return None


def _code_hash(code, doc: str | None = None) -> str:
    """The hash of what a code object does: its bytecode, names, constants and
    nested code, not its file name, line numbers or docstring (``doc``, the
    function's ``__doc__``), so a comment, a docstring, a blank line or a moved
    function is not an edit. Bytecode is stable within a Python minor version;
    the record carries the version, and a resume under another one does not
    compare."""
    import types

    parts = [
        code.co_code,
        repr(code.co_names).encode(),
        repr(code.co_varnames).encode(),
        repr(code.co_freevars).encode(),
        repr(code.co_cellvars).encode(),
        repr((code.co_argcount, code.co_kwonlyargcount, code.co_flags)).encode(),
    ]
    consts = list(code.co_consts)
    if doc is not None and consts and consts[0] == doc:
        consts = consts[1:]
    for const in consts:
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
        instructions = [
            ins
            for ins in dis.get_instructions(c)
            if ins.opname not in ("EXTENDED_ARG", "CACHE", "NOP")
        ]
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
                # `from pkg import sub`: sub may be a submodule, named only in
                # the fromlist loaded just before.
                fromlist = instructions[i - 1].argval if i >= 1 else None
                if isinstance(fromlist, tuple):
                    for item in fromlist:
                        if isinstance(item, str) and item != "*":
                            imports.append((f"{name}.{item}" if name else item, level))
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


def _module_label(module_name: str, namespace: dict | None = None) -> str:
    """How a module is named in a key: by its import name, except a script run
    as ``__main__``, which is named as ``python -m`` names it (its spec) or by
    its file, so the run and a later ``import`` of the same file agree. A
    notebook's ``__main__`` has neither and stays ``__main__``. ``namespace``
    (a function's ``__globals__``) says which ``__main__``: in a parallel worker
    ``sys.modules["__main__"]`` is the launcher's module, while the task's
    functions carry their script's globals."""
    import sys

    if module_name != "__main__":
        return module_name
    if namespace is None:
        main = sys.modules.get("__main__")
        namespace = getattr(main, "__dict__", {}) or {}
        file = namespace.get("__file__")
        if isinstance(file, str) and file and _foreign_by_path(file, "__main__"):
            return "__main__"  # the launcher's module, not the user's script
    spec = namespace.get("__spec__")
    if spec is not None and getattr(spec, "name", None):
        return spec.name
    file = namespace.get("__file__")
    return Path(file).stem if isinstance(file, str) and file else "__main__"


def _canonical_dump(node) -> str:
    """A syntax tree as text, leaving out empty and None fields, so the same
    source dumps alike on Python 3.11 to 3.14 (``ast.dump`` prints every field,
    and the fields grow across minors)."""
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


def _find_file(name: str) -> str | None:
    """The file a module name denotes, without importing anything: the loaded
    module's, or the one the path finders would load, walking a dotted name's
    packages by their search locations."""
    import importlib.machinery
    import sys

    loaded = sys.modules.get(name)
    if loaded is not None:
        file = getattr(loaded, "__file__", None)
        return file if isinstance(file, str) and file else None
    import importlib.util

    parts = name.split(".")
    path = None
    spec = None
    for i in range(len(parts)):
        dotted = ".".join(parts[: i + 1])
        loaded = sys.modules.get(dotted)
        if loaded is not None:
            path = getattr(loaded, "__path__", None)
            spec = getattr(loaded, "__spec__", None)
            continue
        try:
            if i == 0:
                # Every finder, for the top level only (importing nothing):
                # an editable install may be found by a meta-path finder.
                spec = importlib.util.find_spec(dotted)
            else:
                spec = importlib.machinery.PathFinder.find_spec(dotted, path)
        except (ImportError, AttributeError, ValueError):
            return None
        if spec is None:
            return None
        path = spec.submodule_search_locations
    if spec is None or not spec.has_location or not spec.origin:
        return None
    return spec.origin


def _file_entry(name: str):
    """``(key, hash, tree)`` for a project module imported inside a function's
    body, by its file: the same entry whether or not this process has
    imported it yet, so the key does not change between processes. A Python
    file is hashed as its syntax tree (a comment is not an edit), anything
    else by its bytes (``tree`` is then None)."""
    import ast

    if _is_foreign_module(name):
        return None
    file = _find_file(name)
    if file is None:
        return None
    try:
        path = Path(file).resolve()
        if _foreign_by_path(str(path), name.partition(".")[0]):
            return None
        data = path.read_bytes()
    except (OSError, ValueError):
        return None
    if path.suffix == ".py":
        try:
            tree = ast.parse(data)
            return f"{name} (file)", _short_hash(_canonical_dump(tree).encode()), tree
        except (SyntaxError, ValueError, RecursionError):
            pass
    return f"{name} (file)", _short_hash(data), None


def _file_imports(tree, name: str, is_package: bool) -> list[str]:
    """The modules a file imports, at any depth, with ``from pkg import sub``
    naming the submodule too, relative imports resolved against the file's own
    package."""
    import ast

    package = name if is_package else name.rpartition(".")[0]
    pairs: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            pairs.extend((alias.name, 0) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            pairs.append((base, node.level))
            for alias in node.names:
                if alias.name != "*":
                    pairs.append(
                        (f"{base}.{alias.name}" if base else alias.name, node.level)
                    )
    return _resolve_imports([(n, lv) for n, lv in pairs if n], package or None)


def _file_entries(name: str) -> dict[str, str]:
    """Entries for a module imported inside a function's body and, through
    their files, every project module it imports in turn: a helper imported
    lazily that itself imports the constants it uses."""
    out: dict[str, str] = {}
    pending = [name]
    seen: set[str] = set()
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        found = _file_entry(current)
        if found is None:
            continue
        key, digest, tree = found
        out[key] = digest
        if tree is not None:
            file = _find_file(current) or ""
            is_package = Path(file).name == "__init__.py"
            pending.extend(_file_imports(tree, current, is_package))
    return out


#: A stand-in for a module-level name the project's own code rebinds (a
#: lazily created object, a warn-once flag): state, not code, so its value is
#: not hashed; and for a secret-shaped name or value, which is never written
#: to a sidecar even as a hash.
_STATE = _short_hash(b"<state>")
_SECRET = _short_hash(b"<secret>")


#: Secret-shaped names and values, narrower than provenance's redaction rules
#: (where a false positive costs nothing): here one would hide an edit. A name
#: matches as a whole word between underscores (``HF_TOKEN``, ``api_key``, not
#: ``AUTHOR`` or ``tokenizer``); a value matches a provider token that starts
#: the string or follows a separator (not ``mask-rcnn_r50`` or a path holding
#: ``hf_cache``).
_SECRET_NAME_RE = re.compile(
    r"(?i)(?:^|_)(?:api_?key|secret|token|password|passwd|credentials?|auth)(?:_|$)"
)
_SECRET_VALUE_RE = re.compile(
    r"(?:^|[\s=:'\"])(?:sk-|hf_|ghp_|gho_|xox[baprs]-|AKIA)[A-Za-z0-9_-]{16,}"
)


def _secret_value(value: str) -> bool:
    return bool(_SECRET_VALUE_RE.search(value))


def _secret_name(name: str, value) -> bool:
    """A string under a secret-shaped name (``api_key``): a number under one
    (``MAX_TOKENS = 512``) is a setting, and is hashed."""
    return isinstance(value, (str, bytes)) and bool(_SECRET_NAME_RE.search(name))


def _secret(key: str, value) -> bool:
    name = key.rpartition(":")[2].rpartition(".")[2]
    if _secret_name(name, value):
        return True
    return isinstance(value, str) and _secret_value(value)


#: Bounds on one walk: steps, and the length of a key (a chain of names that
#: long is an object inventing attributes, not code).
_WALK_MAX_OPS = 50_000
_WALK_MAX_KEY = 400


class _CodeWalk:
    """Walks from the roots through the names their code uses, recording a hash
    for every project function, class and constant reached, keyed by where the
    name was looked up (``helpers:scale``), so a module attribute rebound to
    another function is seen as a change. A name that resolves to something not
    hashed (a mutable container, a foreign function) gets an entry naming what
    it is, so an edit that turns a recorded value into one of those still
    shows. The walk is iterative: a large project's call graph would exceed the
    recursion limit."""

    def __init__(self):
        self.hashes: dict[str, str] = {}
        self._object_hash: dict[int, str] = {}
        self._walked: set[int] = set()
        self._module_names: dict[int, set[str]] = {}
        self._state_names: dict[int, frozenset[str]] = {}
        self._stack: list[tuple] = []
        self._ops = 0
        self.truncated = False

    def run(self) -> None:
        # The walk runs for every sweep, so it must end: at most _WALK_MAX_OPS
        # steps. The order is fixed (sorted names, a stack), so a cut-off walk
        # records the same entries in every process.
        while self._stack:
            if self._ops >= _WALK_MAX_OPS:
                self.truncated = True
                self._stack.clear()
                return
            self._ops += 1
            op, *args = self._stack.pop()
            try:
                op(*args)
            except Exception:  # noqa: BLE001 - a value that resists inspection
                continue

    def _push(self, op, *args) -> None:
        self._stack.append((op, *args))

    # -- helpers ------------------------------------------------------------
    def _project(self, value) -> bool:
        import types

        try:
            if isinstance(value, types.FunctionType):
                # A function knows its own module from its globals; a
                # functools.wraps wrapper copies __module__ from what it wraps.
                g = value.__globals__
                name, file = g.get("__name__"), g.get("__file__")
                return isinstance(name, str) and not _is_foreign_module(
                    name, file if isinstance(file, str) else None
                )
            module = getattr(value, "__module__", None)
        except Exception:  # noqa: BLE001 - a descriptor that raises
            return False
        return isinstance(module, str) and not _is_foreign_module(module)

    def _here(self, fn) -> str:
        """The label of the module a function's globals belong to."""
        g = getattr(fn, "__globals__", None) or {}
        name = g.get("__name__")
        if not isinstance(name, str):
            name = fn.__module__ if isinstance(fn.__module__, str) else "?"
        return _module_label(name, g)

    def _record(self, key: str | None, digest: str) -> None:
        if key is not None and len(key) <= _WALK_MAX_KEY:
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

    def _state(self, namespace: dict) -> frozenset[str]:
        """The module-level names the module's own functions and methods rebind
        (``global x; x = ...``), through decorators: state, whose value changes
        as the code runs. ``namespace`` is the module's globals."""
        import dis
        import types

        cached = self._state_names.get(id(namespace))
        if cached is not None:
            return cached
        module_name = namespace.get("__name__")
        codes = []
        for value in list(namespace.values()):
            try:
                if not isinstance(value, (types.FunctionType, type)):
                    # A decorated function may be a C wrapper: what it wraps.
                    value = getattr(value, "__wrapped__", None)
                    if not isinstance(value, types.FunctionType):
                        continue
                if isinstance(value, types.FunctionType):
                    layer, seen = value, set()
                    while (
                        isinstance(layer, types.FunctionType) and id(layer) not in seen
                    ):
                        seen.add(id(layer))
                        if layer.__globals__ is namespace:
                            codes.append(layer.__code__)
                        layer = getattr(layer, "__wrapped__", None)
                elif value.__module__ == module_name:
                    for member in vars(value).values():
                        func = getattr(member, "__func__", member)
                        if isinstance(getattr(func, "__code__", None), types.CodeType):
                            codes.append(func.__code__)
            except Exception:  # noqa: BLE001 - a class whose attributes raise
                continue
        names: set[str] = set()
        while codes:
            code = codes.pop()
            codes.extend(k for k in code.co_consts if isinstance(k, types.CodeType))
            for ins in dis.get_instructions(code):
                if ins.opname in ("STORE_GLOBAL", "DELETE_GLOBAL"):
                    names.add(ins.argval)
        out = frozenset(names)
        self._state_names[id(namespace)] = out
        return out

    # -- roots --------------------------------------------------------------
    def walk_root(self, root, *, with_body: bool = False) -> None:
        import types

        layer = root
        while isinstance(layer, functools.partial):
            # A partial task's bound arguments are part of what runs.
            inner = layer.func
            while isinstance(inner, functools.partial):
                inner = inner.func
            name = getattr(inner, "__qualname__", type(inner).__qualname__)
            label = self._here(inner) if hasattr(inner, "__globals__") else "?"
            self._record(f"{label}:{name}.<partial>", self._partial_digest(layer))
            layer = layer.func
        # A project decorator around the task is code it runs: each wrapper
        # layer is recorded with its body; the innermost function is the
        # task itself, covered by code_fingerprint (or by Study's methods
        # fingerprint), unless with_body (pre_task).
        layers = [layer]
        seen = {id(layer)}
        while True:
            inner = getattr(layers[-1], "__wrapped__", None)
            if inner is None or id(inner) in seen:
                break
            seen.add(id(inner))
            layers.append(inner)
        for i, target in enumerate(layers):
            innermost = i == len(layers) - 1
            if isinstance(getattr(target, "__code__", None), types.CodeType):
                key = None
                if (not innermost or with_body) and self._project(target):
                    key = f"{_module_label(target.__module__)}:{target.__qualname__}"
                self._push(self._walk_function, target, key)
            elif innermost and isinstance(target, type):
                self._push(self._walk_class, target, None)
            elif innermost and target is not None:
                self._push(self._walk_class, type(target), None)
        self.run()

    def walk_target(self, path: str) -> None:
        """A class or function the config names as ``_target_``: recorded by
        the file of its module (and the project modules that file imports), so
        the entry is the same whether or not this process has imported it yet.
        Hydra imports it from that file when the cell instantiates it."""
        parts = path.split(".")
        if _is_foreign_module(parts[0]):
            return
        for i in range(len(parts) - 1, 0, -1):
            module = ".".join(parts[:i])
            if _find_file(module) is None:
                continue
            for key, digest in _file_entries(module).items():
                self._record(key, digest)
            return

    # -- values -------------------------------------------------------------
    def _function_digest(self, fn) -> str:
        """A function's code, where it is defined, and its default arguments
        (a default is part of what the function does, and lives outside its
        code object)."""
        digest = self._object_hash.get(id(fn))
        if digest is None:
            extras: list[str] = [
                f"{self._here(fn)}.{fn.__qualname__}",
                _code_hash(
                    fn.__code__, fn.__doc__ if isinstance(fn.__doc__, str) else None
                ),
            ]
            code = fn.__code__
            positional = code.co_varnames[: code.co_argcount]
            defaults = fn.__defaults__ or ()
            named = list(
                zip(
                    positional[len(positional) - len(defaults) :],
                    defaults,
                    strict=False,
                )
            )
            named += sorted((fn.__kwdefaults__ or {}).items())
            for name, value in named:
                if _secret_name(name, value):
                    extras.append(f"{name}=<secret>")
                else:
                    extras.append(
                        f"{name}={_constant_repr(value) or self._kind(value)}"
                    )
            digest = _short_hash("|".join(extras).encode())
            self._object_hash[id(fn)] = digest
        return digest

    def _partial_digest(self, value) -> str:
        parts = []
        for a in value.args:
            parts.append(_constant_repr(a) or self._kind(a))
        for k, v in sorted((value.keywords or {}).items()):
            c = "<secret>" if _secret_name(k, v) else _constant_repr(v)
            parts.append(f"{k}={c or self._kind(v)}")
        return _short_hash("|".join(parts).encode())

    def _item_key(self, value) -> str:
        """A key for an item in a set or dict whose repr would show a memory
        address (a function, a class): its module and name, the same in every
        process."""
        const = _constant_repr(value)
        if const is not None:
            return const
        name = getattr(value, "__qualname__", None)
        if isinstance(name, str):
            module = getattr(value, "__module__", "?")
            return f"{module}.{name}"
        return f"<{type(value).__qualname__}>"

    def _walk_function(self, fn, record_as: str | None) -> None:
        code = fn.__code__
        self._record(record_as, self._function_digest(fn))
        globals_, attrs, imports = _code_names(code)
        here = self._here(fn)
        if id(fn) in self._walked:
            return
        self._walked.add(id(fn))
        # Closure cells and callable defaults belong to this function object,
        # so they are keyed under where it was first reached (two closures
        # from one factory are two objects, and differ), and walked once: a
        # closure that holds the function itself (a call counter, a recursive
        # inner function) would otherwise be walked forever.
        owner = record_as or f"{here}:{fn.__qualname__}"
        for cell_name, cell in zip(
            code.co_freevars, fn.__closure__ or (), strict=False
        ):
            try:
                value = cell.cell_contents
            except ValueError:  # an empty cell
                continue
            self._push(
                self._walk_value, value, f"{owner}.<{cell_name}>", globals_, attrs, 0
            )
        defaults = list(fn.__defaults__ or ()) + list(
            (fn.__kwdefaults__ or {}).values()
        )
        for i, value in enumerate(defaults):
            if callable(value):
                self._push(
                    self._walk_value, value, f"{owner}.<default {i}>", set(), attrs, 0
                )
        namespace = getattr(fn, "__globals__", {}) or {}
        state = self._state(namespace)
        for name in sorted(globals_):
            if name not in namespace:
                continue
            key = f"{here}:{name}"
            if name in state:
                self._record(key, _STATE)
                continue
            self._push(self._walk_value, namespace[name], key, globals_, attrs, 0)
        package = namespace.get("__package__")
        package = package if isinstance(package, str) else None
        # Imports inside the body are recorded by their files, the same in a
        # process that has imported them and one that has not.
        for name in _resolve_imports(imports, package):
            for key, digest in _file_entries(name).items():
                self._record(key, digest)

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
        here = _module_label(module.__name__, namespace)
        state = self._state(namespace)
        for name in sorted(new):
            if name not in namespace:
                continue
            key = f"{here}:{name}"
            if name in state:
                self._record(key, _STATE)
                continue
            self._push(self._walk_value, namespace[name], key, set(), attrs, 0)

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
            return  # installed code: provenance's job, and no entry
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
                    if const is not None and not _secret(attr, value):
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
                self._push(self._walk_function, value, None)
            elif kind == "class" and self._project(value):
                key = f"{_module_label(value.__module__)}:{value.__qualname__}"
                self._push(self._walk_class, value, key)
        for base in cls.__bases__:
            if self._project(base):
                key = f"{_module_label(base.__module__)}:{base.__qualname__}"
                self._push(self._walk_class, base, key)

    def _walk_value(
        self, value, key: str, globals_: set[str], attrs: set[str], depth: int
    ) -> None:
        """What a name refers to. An entry is recorded only for the project's
        own code and for constants read from its modules: an object, a
        container's contents, an installed function or module get none. That
        prefers missing an edit over re-running unchanged work, since such
        values may change as the code runs."""
        import types

        if len(key) > _WALK_MAX_KEY:
            return
        if isinstance(value, types.ModuleType):
            if not _is_foreign_module(value.__name__):
                # Which module it is (`import impl_a as impl`): an alias
                # pointed at another project module is a change.
                self._record(key, _short_hash(f"<module {value.__name__}>".encode()))
                self._walk_module(value, attrs)
            return
        if isinstance(value, functools.partial):
            if depth == 0:
                self._record(key, self._partial_digest(value))
            self._push(
                self._walk_value, value.func, key + ".func", globals_, attrs, depth
            )
            return
        if isinstance(value, types.MethodType):
            value = value.__func__
        if isinstance(value, types.FunctionType):
            if self._project(value):
                self._walk_function(value, key)
            else:
                self._unwrap(value, key, globals_, attrs, depth)
            return
        if isinstance(value, type):
            self._walk_class(value, key)
            return
        if isinstance(value, (dict, list, tuple, set, frozenset)):
            const = _constant_repr(value) if depth == 0 else None
            if const is not None:
                self._record(
                    key, _SECRET if _secret(key, value) else _short_hash(const.encode())
                )
                return
            # A registry such as {"cnn": CNN, "mlp": mlp}: the project code it
            # holds is walked; its other contents, and the container itself,
            # are not entries (it may be filled as the code runs).
            if depth < 2 and len(value) <= 1000:
                if isinstance(value, dict):
                    items = sorted(
                        ((self._item_key(k), v) for k, v in value.items()),
                        key=lambda kv: kv[0],
                    )
                elif isinstance(value, (set, frozenset)):
                    # A set iterates in hash order, which for functions and
                    # classes follows their addresses: key items by name.
                    items = sorted(
                        ((self._item_key(v), v) for v in value), key=lambda kv: kv[0]
                    )
                else:
                    items = [(str(i), v) for i, v in enumerate(value)]
                code_like = (
                    types.FunctionType,
                    type,
                    functools.partial,
                    dict,
                    list,
                    tuple,
                )
                for k, v in items:
                    if isinstance(v, code_like):
                        self._push(
                            self._walk_value,
                            v,
                            f"{key}[{k}]",
                            globals_,
                            attrs,
                            depth + 1,
                        )
            return
        if depth > 0:
            return  # inside a registry, only code
        const = _constant_repr(value)
        if const is not None:
            self._record(
                key, _SECRET if _secret(key, value) else _short_hash(const.encode())
            )
            if self._project(type(value)):
                # A frozen settings dataclass: its fields above, its class too.
                self._push(self._walk_class, type(value), f"{key}.__class__")
            return
        if callable(value):
            # A decorated function behind a C wrapper (functools.lru_cache).
            self._unwrap(value, key, globals_, attrs, depth)
        if self._project(type(value)):
            # An instance of a project class (a model): its class is code the
            # task runs; its state is not hashed.
            self._push(self._walk_class, type(value), f"{key}.__class__")

    def _unwrap(self, value, key: str, globals_, attrs, depth: int) -> None:
        """Follow a decorator's ``__wrapped__`` chain to project code, as
        inspect.unwrap does: only through functions, classes and partials, at
        most a few layers, each object once (an object that invents attributes,
        such as an addict.Dict, never leads anywhere)."""
        import types

        code_like = (types.FunctionType, type, functools.partial)
        seen = {id(value)}
        try:
            layer = getattr(value, "__wrapped__", None)
        except Exception:  # noqa: BLE001 - an attribute that raises
            return
        for _ in range(16):
            if not isinstance(layer, code_like) or id(layer) in seen:
                return
            seen.add(id(layer))
            if isinstance(layer, functools.partial) or self._project(layer):
                self._push(
                    self._walk_value,
                    layer,
                    key + ".__wrapped__",
                    globals_,
                    attrs,
                    depth,
                )
                return
            layer = getattr(layer, "__wrapped__", None)


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


#: Each ``_target_``'s walk per (run token, target), in this process: every
#: cell of a sweep that names the same classes shares it. Bounded like the
#: roots' records.
_TARGET_RECORDS: dict[tuple[str, str], dict[str, str]] = {}
_TARGET_RECORDS_MAX = 4096


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
    code = dict(base["code"])
    for target in targets:
        entries = _TARGET_RECORDS.get((token, target))
        if entries is None:
            try:
                walk = _CodeWalk()
                walk.walk_target(target)
                entries = walk.hashes
            except Exception:  # noqa: BLE001 - best effort: the roots' record stands
                entries = {}
            while len(_TARGET_RECORDS) >= _TARGET_RECORDS_MAX:
                _TARGET_RECORDS.pop(next(iter(_TARGET_RECORDS)))
            _TARGET_RECORDS[(token, target)] = entries
        code.update(entries)
    return {"python": base["python"], "code": code}


def merge_project_records(prior, current):
    """The record to keep for a cell reused under ``current``. A cell keeps its
    prior record when it has one: only the keys both records share were
    compared, so writing keys only ``current`` has would declare the cell
    computed by code it was never checked against (a name reached lazily, an
    alias pointed elsewhere, another Python version). A cell with no record
    (written before the field) takes the current one: reusing it already
    trusted it."""
    if isinstance(prior, dict) and isinstance(prior.get("code"), dict):
        return prior
    return current


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
