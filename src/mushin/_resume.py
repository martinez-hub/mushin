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
    budget_id: str | None = None,
) -> None:
    """Atomically write this cell's status sidecar into its own dir.

    ``job_num`` and ``choices`` are the Hydra job number and config-group
    choices the job's hydra.yaml also records. Written when both are known
    (inside a Hydra multirun job) and are what Hydra writes (the rule the
    reader applies, so nothing is recorded that a reload would then reject),
    never one without the other, they let a reload sort the cells and key
    config-group axes without parsing hydra.yaml, which is most of a reload's
    time; see ``sidecar_job_fields``.

    ``budget_id`` names the run whose ``max_total_seconds`` budget the cell
    started under; the runner records it with 'running', so a cell that a
    scheduler requeues after the deadline is known to have started before
    it."""

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
    if budget_id is not None:
        payload["budget_id"] = budget_id
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


#: 3.14 marks a code object that has a docstring with this flag; it is not
#: part of what the code does.
_CO_HAS_DOCSTRING = 0x4000000
_CONST_OPS = frozenset({"LOAD_CONST", "RETURN_CONST"})
_SKIPPED_OPS = frozenset({"CACHE", "EXTENDED_ARG", "NOP", "RESUME"})
#: 3.13+ fuse two adjacent local-variable instructions into one only when both
#: are on the same line, so wrapping a call over several lines changes which
#: opcodes appear; each is hashed as the instructions it stands for. 3.14's
#: borrowed loads are the compiler's choice, not the code's.
_FUSED_OPS = {
    "LOAD_FAST_LOAD_FAST": ("LOAD_FAST", "LOAD_FAST"),
    "STORE_FAST_LOAD_FAST": ("STORE_FAST", "LOAD_FAST"),
    "STORE_FAST_STORE_FAST": ("STORE_FAST", "STORE_FAST"),
    "LOAD_FAST_BORROW_LOAD_FAST_BORROW": ("LOAD_FAST", "LOAD_FAST"),
}
_RENAMED_OPS = {"LOAD_FAST_BORROW": "LOAD_FAST"}


def _fused(ins) -> bool:
    return ins.opname in _FUSED_OPS and isinstance(ins.argval, tuple)


def _code_hash(code, function: bool = False) -> str:
    """The hash of what a code object does: its instructions, with each
    constant they load by value (not by its index in ``co_consts``, which a
    docstring shifts on 3.14) and nested code by its own hash, plus its names
    and signature; not its file name, line numbers or docstring (no
    instruction loads a docstring). So a comment, a docstring, a blank line or
    a moved function is not an edit. Bytecode is stable within a Python minor
    version; the record carries the version, and a resume under another one
    does not compare. ``function`` is kept for callers; every code object is
    hashed the same way. Jumps are hashed by the index of the instruction
    they reach, since a line number's NOP (dropped here) shifts every later
    offset, and fused instructions as the ones they stand for, so most
    reformatting is not an edit either."""
    import bisect
    import dis
    import types

    del function
    jumps = (
        set(getattr(dis, "hasjump", ()))
        | set(getattr(dis, "hasjrel", ()))
        | set(getattr(dis, "hasjabs", ()))
    )
    kept = [i for i in dis.get_instructions(code) if i.opname not in _SKIPPED_OPS]
    # Where each kept instruction starts in the stream as hashed, in which a
    # fused instruction counts as the two it stands for.
    offsets, positions, at = [], [], 0
    for ins in kept:
        offsets.append(ins.offset)
        positions.append(at)
        at += 2 if _fused(ins) else 1
    parts = [
        repr(code.co_names).encode(),
        repr(code.co_varnames).encode(),
        repr(code.co_freevars).encode(),
        repr(code.co_cellvars).encode(),
        repr(
            (
                code.co_argcount,
                code.co_kwonlyargcount,
                code.co_flags & ~_CO_HAS_DOCSTRING,
            )
        ).encode(),
    ]
    for ins in kept:
        opname = _RENAMED_OPS.get(ins.opname, ins.opname)
        if _fused(ins):
            for op, name in zip(_FUSED_OPS[opname], ins.argval, strict=False):
                parts.append(f"{op} {name!r}".encode())
            continue
        if opname in _CONST_OPS:
            const = ins.argval
            if isinstance(const, types.CodeType):
                arg = _code_hash(const)
            else:
                # A frozenset constant (``x in {"a", "b"}``) prints in
                # string-hash order, which differs per process: sorted.
                arg = _constant_repr(const) or repr(const)
        elif ins.opcode in jumps and isinstance(ins.argval, int):
            # The target's index among the kept instructions (a dropped NOP
            # target resolves to the next kept one).
            at = bisect.bisect_left(offsets, ins.argval)
            arg = f"->{positions[at] if at < len(positions) else 'end'}"
        else:
            arg = repr(ins.argval) if ins.argval is not None else ""
        parts.append(f"{opname} {arg}".encode())
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


def _stored_attrs(codes) -> set[str]:
    """The attribute names some code, or code nested in it, assigns or deletes
    (``Net.made += 1``, ``cls._cache = ...``): a class attribute its own
    methods store is state."""
    import dis
    import types

    names: set[str] = set()
    stack = list(codes)
    while stack:
        current = stack.pop()
        stack.extend(k for k in current.co_consts if isinstance(k, types.CodeType))
        for ins in dis.get_instructions(current):
            if ins.opname in ("STORE_ATTR", "DELETE_ATTR"):
                names.add(ins.argval)
    return names


def _rebound_cells(code) -> set[str]:
    """The free variables a function, or code nested in it, rebinds with
    ``nonlocal``."""
    import dis
    import types

    names: set[str] = set()
    stack = [code]
    while stack:
        current = stack.pop()
        stack.extend(k for k in current.co_consts if isinstance(k, types.CodeType))
        for ins in dis.get_instructions(current):
            if ins.opname in ("STORE_DEREF", "DELETE_DEREF"):
                names.add(ins.argval)
    return names & set(code.co_freevars)


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
        if i > 0 and path is None:
            # The parent is a plain module, not a package: `from model import
            # train` names an attribute, and searching sys.path for `train`
            # would find an unrelated top-level file.
            return None
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


def _strip_tree(tree):
    """A module's syntax tree without what does not run on import or is not
    code: ``if __name__ == "__main__":`` blocks (an imported module's driver
    block) and docstrings, at any depth."""
    import ast

    def is_main_guard(node) -> bool:
        if not isinstance(node, ast.If) or not isinstance(node.test, ast.Compare):
            return False
        if len(node.test.ops) != 1 or not isinstance(node.test.ops[0], ast.Eq):
            return False
        parts = [node.test.left, *node.test.comparators]
        names = {p.id for p in parts if isinstance(p, ast.Name)}
        consts = {p.value for p in parts if isinstance(p, ast.Constant)}
        return "__name__" in names and "__main__" in consts

    def is_doc(node) -> bool:
        return (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        )

    def clean(body: list) -> list:
        out = []
        for i, node in enumerate(body):
            if is_main_guard(node):
                out.extend(clean(node.orelse))  # the else branch runs on import
                continue
            if i == 0 and is_doc(node):
                continue
            for field in ("body", "orelse", "finalbody", "handlers"):
                inner = getattr(node, field, None)
                if isinstance(inner, list) and inner and isinstance(inner[0], ast.AST):
                    setattr(node, field, clean(inner) if field != "handlers" else inner)
            out.append(node)
        return out

    tree.body = clean(tree.body)
    return tree


#: Parsed project files, per process: (resolved path, mtime_ns, size) ->
#: (digest, tree). A walk that reaches the same file through many lazy imports
#: reads and parses it once.
_FILE_CACHE: dict[tuple[str, int, int], tuple[str, object]] = {}
_FILE_CACHE_MAX = 2048


def _parsed(path: Path):
    """``(digest, tree)`` for a project file, cached: a Python file is hashed as
    its syntax tree without its main block and docstrings (a comment, a
    docstring or the driver block is not an edit), anything else by its bytes
    (``tree`` is then None)."""
    import ast

    st = path.stat()
    cache_key = (str(path), st.st_mtime_ns, st.st_size)
    hit = _FILE_CACHE.get(cache_key)
    if hit is not None:
        return hit
    data = path.read_bytes()
    result: tuple[str, object] = (_short_hash(data), None)
    if path.suffix == ".py":
        try:
            tree = ast.parse(data)
            digest = _short_hash(_canonical_dump(_strip_tree(ast.parse(data))).encode())
            result = (digest, tree)
        except (SyntaxError, ValueError, RecursionError):
            pass
    while len(_FILE_CACHE) >= _FILE_CACHE_MAX:
        _FILE_CACHE.pop(next(iter(_FILE_CACHE)))
    _FILE_CACHE[cache_key] = result
    return result


def _project_file_of(name: str) -> Path | None:
    """The project's own file for a module name, found without importing it,
    judged by where the file is (not by name: a checkout's ``utils`` stays the
    project's when an installed distribution also provides ``utils``)."""
    if name.partition(".")[0] == "mushin":
        return None
    file = _find_file(name)
    if file is None:
        return None
    try:
        path = Path(file).resolve()
    except OSError:
        return None
    if _foreign_by_path(str(path), name.partition(".")[0]):
        return None
    return path


def _file_entry(name: str):
    """``(key, hash, tree)`` for a project module imported inside a function's
    body, by its file: the same entry whether or not this process has
    imported it yet, so the key does not change between processes."""
    path = _project_file_of(name)
    if path is None:
        return None
    try:
        digest, tree = _parsed(path)
    except (OSError, ValueError):
        return None
    return f"{name} (file)", digest, tree


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


#: At most this many files per module imported in a body: a lazily imported
#: project module that pulls in a large package is covered that far.
_FILES_MAX = 400


def _file_entries(name: str) -> tuple[dict[str, str], bool]:
    """Entries for a module imported inside a function's body and, through
    their files, every project module it imports in turn (a helper imported
    lazily that itself imports the constants it uses), and whether
    ``_FILES_MAX`` cut the list short."""
    out: dict[str, str] = {}
    pending = [name]
    seen: set[str] = set()
    while pending and len(seen) < _FILES_MAX:
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
    return out, any(p not in seen for p in pending)


#: Each project file's module-level bindings and class bodies, per process:
#: (path, mtime_ns, size) -> index.
_SOURCE_CACHE: dict[tuple[str, int, int], tuple[dict, dict]] = {}


def _source_index(file: str, module_name: str):
    """``(bindings, class_bodies)`` for a project file. ``bindings`` maps each
    module-level name assigned in the file to the source of its assignments
    (or, for ``from m import x``, to ``("from", m, x)``); ``class_bodies`` maps
    each class's qualified name to the source of its non-method statements.
    Assignments inside functions and classes are not module-level bindings."""
    import ast
    import importlib.util

    path = Path(file).resolve()
    st = path.stat()
    cache_key = (str(path), st.st_mtime_ns, st.st_size)
    hit = _SOURCE_CACHE.get(cache_key)
    if hit is not None:
        return hit
    tree = ast.parse(path.read_bytes())
    is_package = path.name == "__init__.py"
    package = module_name if is_package else module_name.rpartition(".")[0]
    bindings: dict[str, object] = {}
    class_bodies: dict[str, str] = {}

    def names(target) -> list[str]:
        if isinstance(target, ast.Name):
            return [target.id]
        if isinstance(target, (ast.Tuple, ast.List)):
            return [n for elt in target.elts for n in names(elt)]
        if isinstance(target, ast.Starred):
            return names(target.value)
        return []

    def literal(node) -> bool:
        value = getattr(node, "value", None)
        if value is None or isinstance(node, ast.AugAssign):
            return False
        try:
            ast.literal_eval(value)
        except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
            return False
        return True

    def bind(name: str, node) -> None:
        dumped = _canonical_dump(_sanitized(node))
        prior = bindings.get(name)
        entries = prior if isinstance(prior, list) else []
        value = _NO_LITERAL
        if literal(node) and isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if all(isinstance(t, ast.Name) for t in targets):
                value = ast.literal_eval(node.value)
        bindings[name] = entries + [(value, dumped)]

    def visit(body, prefix: str) -> None:
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if isinstance(node, ast.ClassDef):
                qual = f"{prefix}{node.name}"
                stmts = [
                    b
                    for b in node.body
                    if not isinstance(
                        b, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
                    )
                    and not (
                        isinstance(b, ast.Expr)
                        and isinstance(b.value, ast.Constant)
                        and isinstance(b.value.value, str)
                    )
                ]
                class_bodies[qual] = _short_hash(
                    "|".join(_canonical_dump(_sanitized(b)) for b in stmts).encode()
                )
                visit_classes(node.body, qual + ".")
                continue
            for sub_node in ast.walk(node):
                if isinstance(
                    sub_node,
                    (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef),
                ):
                    continue
                if isinstance(sub_node, ast.Assign):
                    for t in sub_node.targets:
                        for n in names(t):
                            bind(n, sub_node)
                elif isinstance(sub_node, (ast.AnnAssign, ast.AugAssign)):
                    for n in names(sub_node.target):
                        bind(n, sub_node)
                elif isinstance(sub_node, ast.NamedExpr):
                    for n in names(sub_node.target):
                        bind(n, sub_node)
                elif isinstance(sub_node, ast.ImportFrom):
                    base = sub_node.module or ""
                    if sub_node.level:
                        try:
                            base = importlib.util.resolve_name(
                                "." * sub_node.level + base, package or None
                            ).rstrip(".")
                        except (ImportError, ValueError):
                            continue
                    for alias in sub_node.names:
                        if alias.name != "*" and not isinstance(
                            bindings.get(alias.asname or alias.name), list
                        ):
                            bindings[alias.asname or alias.name] = (
                                "from",
                                base,
                                alias.name,
                            )

    def visit_classes(body, prefix: str) -> None:
        for node in body:
            if isinstance(node, ast.ClassDef):
                visit([node], prefix)

    visit(tree.body, "")
    out = (bindings, class_bodies)
    while len(_SOURCE_CACHE) >= _FILE_CACHE_MAX:
        _SOURCE_CACHE.pop(next(iter(_SOURCE_CACHE)))
    _SOURCE_CACHE[cache_key] = out
    return out


#: Each project file's default arguments and ``global`` declarations, per
#: process: (path, mtime_ns, size) -> (defaults, rebound).
_DEF_CACHE: dict[tuple[str, int, int], tuple[dict, frozenset]] = {}


def _default_entry(node) -> tuple[str, object]:
    """How a default argument is written: ``("literal", value)`` for a literal,
    else ``("source", digest)`` of its expression with secret-shaped strings
    blanked (a default computed on import, ``stamp=time.time()``, then reads
    alike in every process and on every node)."""
    import ast

    try:
        return ("literal", ast.literal_eval(node))
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
        return ("source", _short_hash(_canonical_dump(_sanitized(node)).encode()))


def _def_index(file: str) -> tuple[dict, frozenset]:
    """``(defaults, rebound)`` for a project file. ``defaults`` maps each
    function's ``(name, first line)`` (its first decorator's line, as
    ``co_firstlineno`` gives it) to ``{argument: _default_entry}``, or to None
    when two functions share that key; ``rebound`` holds every name a
    ``global`` statement in the file declares: state the module's own code
    rebinds, the same set in every process (a parallel worker's copy of a
    script's globals holds only the names its pickled functions use)."""
    import ast

    path = Path(file).resolve()
    st = path.stat()
    cache_key = (str(path), st.st_mtime_ns, st.st_size)
    hit = _DEF_CACHE.get(cache_key)
    if hit is not None:
        return hit
    _, tree = _parsed(path)
    defaults: dict[tuple[str, int], dict | None] = {}
    rebound: set[str] = set()
    for node in ast.walk(tree) if tree is not None else ():
        if isinstance(node, ast.Global):
            rebound.update(node.names)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            decorators = getattr(node, "decorator_list", None) or []
            first = decorators[0].lineno if decorators else node.lineno
            a = node.args
            positional = [*a.posonlyargs, *a.args]
            entries = {
                arg.arg: _default_entry(d)
                for arg, d in zip(
                    positional[len(positional) - len(a.defaults) :],
                    a.defaults,
                    strict=False,
                )
            }
            for arg, d in zip(a.kwonlyargs, a.kw_defaults, strict=False):
                if d is not None:
                    entries[arg.arg] = _default_entry(d)
            key = (getattr(node, "name", "<lambda>"), first)
            defaults[key] = None if key in defaults else entries
    out = (defaults, frozenset(rebound))
    while len(_DEF_CACHE) >= _FILE_CACHE_MAX:
        _DEF_CACHE.pop(next(iter(_DEF_CACHE)))
    _DEF_CACHE[cache_key] = out
    return out


#: _default_sources' answer for a function generated at runtime (a
#: dataclass's ``__init__``, a NamedTuple's ``__new__``): its defaults are the
#: class body's, which the class's entry covers.
_GENERATED = object()
#: _default_sources' answer for a function in a module with a file that does
#: not show its definition where its code says (two lambdas on one line, the
#: file edited and not reloaded, or unreadable here): its defaults by type
#: only, rather than by a value that may differ between processes.
_UNKNOWN_DEFS = object()


def _default_sources(fn):
    """How ``fn``'s defaults are written in its file: ``{argument: entry}``;
    ``_GENERATED`` for generated code; ``_UNKNOWN_DEFS`` when its module has
    a file that does not show the definition; None when its module has no
    file (a notebook), whose defaults are hashed by value."""
    code = fn.__code__
    filename = code.co_filename
    if not isinstance(filename, str):
        return None
    if (
        filename.startswith("<")
        and filename.endswith(">")
        and not filename.startswith("<ipython-input-")
    ):
        return _GENERATED
    # Whether the module has a file is the namespace's to say, as for a
    # class: a notebook cell's code has a file name that exists nowhere.
    if not isinstance((getattr(fn, "__globals__", None) or {}).get("__file__"), str):
        return None
    try:
        defaults, _ = _def_index(filename)
    except (OSError, SyntaxError, ValueError, RecursionError):
        return _UNKNOWN_DEFS
    found = defaults.get((code.co_name, code.co_firstlineno))
    return _UNKNOWN_DEFS if found is None else found


#: _source_constant's answer for a name assigned one literal (``SCALE =
#: 10``), with the literal's value: the value that runs is hashed while it
#: equals the literal. When it does not, the process changed it at runtime
#: (another module's ``state.STEP += 1``) or the file was edited and not
#: reloaded: either way no entry, rather than a re-run of unchanged cells.
_LITERAL = "<literal>"
_NO_LITERAL = object()
#: _class_source's answer for a class in a module with no file (a notebook).
_NO_FILE = object()


def _sanitized(node):
    """A copy of an assignment's syntax tree with secret-shaped strings
    blanked: a string that looks like a provider token, and a string bound to
    a secret-shaped name (a target, a keyword argument, a dict key)."""
    import ast
    import copy

    node = copy.deepcopy(node)
    secret_values: set[int] = set()
    for sub_node in ast.walk(node):
        if (
            isinstance(sub_node, ast.keyword)
            and sub_node.arg
            and _SECRET_NAME_RE.search(sub_node.arg)
        ):
            secret_values.add(id(sub_node.value))
        if isinstance(sub_node, (ast.Assign, ast.AnnAssign)):
            targets = (
                sub_node.targets
                if isinstance(sub_node, ast.Assign)
                else [sub_node.target]
            )
            if any(
                isinstance(t, ast.Name) and _SECRET_NAME_RE.search(t.id)
                for t in targets
            ):
                secret_values.add(id(sub_node.value))
        if isinstance(sub_node, ast.Dict):
            for k, v in zip(sub_node.keys, sub_node.values, strict=False):
                if (
                    isinstance(k, ast.Constant)
                    and isinstance(k.value, str)
                    and _SECRET_NAME_RE.search(k.value)
                ):
                    secret_values.add(id(v))
    for sub_node in ast.walk(node):
        if isinstance(sub_node, ast.Constant) and isinstance(
            sub_node.value, (str, bytes)
        ):
            text = sub_node.value if isinstance(sub_node.value, str) else ""
            if id(sub_node) in secret_values or _secret_value(text):
                sub_node.value = "<secret>"
    return node


def _source_constant(file: str, module_name: str, name: str, depth: int = 0):
    """How a module-level name is assigned in a project file. A name assigned
    only literals (``SCALE = 10``) gives ``_LITERAL``: its value, which is
    what runs, is hashed by the caller. A name computed on import (from the
    host or the environment: ``os.cpu_count()``, ``"cuda" if ... else "cpu"``)
    gives the digest of its assignments' source, so it reads alike in every
    process and on every node, with secret-shaped strings blanked. A name
    imported with ``from m import x`` is followed to ``m``. None when the file
    does not bind the name at module level (it is then not compared)."""
    try:
        bindings, _ = _source_index(file, module_name)
    except (OSError, SyntaxError, ValueError, RecursionError):
        return None
    binding = bindings.get(name)
    if isinstance(binding, list):
        if len(binding) == 1 and binding[0][0] is not _NO_LITERAL:
            # The caller hashes the value, which is what runs, while it still
            # equals this literal.
            return (_LITERAL, binding[0][0])
        return _short_hash("|".join(dumped for _, dumped in binding).encode())
    if isinstance(binding, tuple) and depth < 5:
        _, module, attr = binding
        path = _project_file_of(module)
        if path is None:
            return None
        return _source_constant(str(path), module, attr, depth + 1)
    return None


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
    # The last name in the key: `mod:f.<api_key>` for a closure cell.
    name = key.rpartition(":")[2].rpartition(".")[2].strip("<>")
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
    another function is seen as a change. Only project code and constants get
    entries: an object, a container's contents and installed code do not, so
    an edit that turns a constant into one of those is not seen (where in
    doubt the guard misses an edit rather than re-running unchanged cells). A
    module-level constant in a project file is recorded by the source of its
    assignment. The walk is iterative and bounded."""

    def __init__(self):
        self.hashes: dict[str, str] = {}
        self._object_hash: dict[int, str] = {}
        self._walked: set[int] = set()
        self._module_names: dict[int, set[str]] = {}
        self._state_names: dict[int, frozenset[str]] = {}
        self._stack: list[tuple] = []
        self._ops = 0
        self.truncated = False
        self._file_memo: dict[str, dict[str, str]] = {}
        # The module namespace each class was reached through by name: a
        # class with no functions of its own (an Enum, a constants class) in
        # the driver script says nothing else about where it is, since in a
        # parallel worker ``__main__`` is the launcher's module.
        self._class_home: dict[int, dict] = {}

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

    def _class_namespace(self, cls) -> dict | None:
        """The globals of the module that defines a class: one of its own
        functions' globals (in a parallel worker ``__main__`` is the launcher's
        module, while a script's class keeps its functions' script globals);
        else the namespace the walk reached it through by name; else its
        module's, except for a class in ``__main__``, which then has none in
        every process."""
        import sys
        import types

        try:
            for value in vars(cls).values():
                func = getattr(value, "__func__", value)
                # Only a function defined in this class says where the class
                # is (enum puts its own functions in an Enum subclass).
                if isinstance(
                    func, types.FunctionType
                ) and func.__qualname__.startswith(cls.__qualname__ + "."):
                    return func.__globals__
        except Exception:  # noqa: BLE001 - a class whose namespace resists
            return None
        home = self._class_home.get(id(cls))
        if home is not None:
            return home
        if cls.__module__ == "__main__":
            return None
        try:
            return vars(sys.modules[cls.__module__])
        except (KeyError, TypeError):
            return None

    def _class_label(self, cls) -> str:
        """The label of a class's module."""
        namespace = self._class_namespace(cls)
        if namespace is None:
            return cls.__module__
        return _module_label(cls.__module__, namespace)

    def _class_source(self, cls):
        """The digest of a class body's non-method statements, from its file;
        ``_NO_FILE`` for a class in a module with no file (a notebook); None
        when it cannot be told."""
        namespace = self._class_namespace(cls)
        if namespace is None:
            return None
        file, name = namespace.get("__file__"), namespace.get("__name__")
        if not isinstance(name, str):
            return None
        if file is None:
            return _NO_FILE
        if not (isinstance(file, str) and file.endswith(".py")):
            return None
        try:
            _, bodies = _source_index(file, name)
        except (OSError, SyntaxError, ValueError, RecursionError):
            return None
        return bodies.get(cls.__qualname__)

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
        (``global x; x = ...``): state, whose value changes as the code runs.
        ``namespace`` is the module's globals. Read from the module's file when
        it has one; otherwise (a notebook) from its functions' code, through
        decorators."""
        import dis
        import types

        cached = self._state_names.get(id(namespace))
        if cached is not None:
            return cached
        file = namespace.get("__file__")
        if isinstance(file, str) and file.endswith(".py"):
            # From the file's own ``global`` statements: the same in every
            # process, where a parallel worker's copy of a script's globals
            # holds only the names its pickled functions use.
            try:
                _, out = _def_index(file)
            except (OSError, SyntaxError, ValueError, RecursionError):
                pass
            else:
                self._state_names[id(namespace)] = out
                return out
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
                    key = f"{self._here(target)}:{target.__qualname__}"
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
        if parts[0] == "__main__" or _is_foreign_module(parts[0]):
            # A class in the driver script (builds(Net) there names
            # __main__.Net): hashing the script's file would make an edit to
            # its run block an edit; such a class is covered when the task's
            # code reaches it.
            return
        for i in range(len(parts) - 1, 0, -1):
            module = ".".join(parts[:i])
            if _find_file(module) is None:
                continue
            self._record_files(module)
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
                _code_hash(fn.__code__, function=True),
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
            sources = _default_sources(fn) if named else None
            for name, value in named:
                extras.append(f"{name}={self._default_repr(name, value, sources)}")
            digest = _short_hash("|".join(extras).encode())
            self._object_hash[id(fn)] = digest
        return digest

    def _default_repr(self, name: str, value, sources) -> str:
        """A default argument as hashed. In a project file, by how it is
        written, as module-level names are: a literal by its value while the
        value still equals it, an expression computed on import by its source
        (it then reads alike on every node). A generated function's defaults
        by type only (its class's body covers them); with no file (a
        notebook), by value."""
        if _secret(name, value):
            return "<secret>"
        if sources is _GENERATED or sources is _UNKNOWN_DEFS:
            return self._kind(value)
        if sources is None:
            return _constant_repr(value) or self._kind(value)
        how, written = sources.get(name, (None, None))
        if how == "source":
            return f"<source {written}>"
        if how == "literal":
            try:
                same = type(value) is type(written) and value == written
            except Exception:  # noqa: BLE001 - a comparison that raises
                same = False
            if same:
                return _constant_repr(value) or self._kind(value)
        # Changed since import (``__defaults__`` set, or the file edited and
        # not reloaded), or not found: the type only.
        return self._kind(value)

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
        if id(fn) in self._walked:
            return
        self._walked.add(id(fn))
        globals_, attrs, imports = _code_names(code)
        here = self._here(fn)
        # Closure cells and callable defaults belong to this function object,
        # so they are keyed under where it was first reached (two closures
        # from one factory are two objects, and differ), and walked once: a
        # closure that holds the function itself (a call counter, a recursive
        # inner function) would otherwise be walked forever.
        owner = record_as or f"{here}:{fn.__qualname__}"
        rebound = _rebound_cells(code) if code.co_freevars else set()
        for cell_name, cell in zip(
            code.co_freevars, fn.__closure__ or (), strict=False
        ):
            if cell_name in rebound:
                # ``nonlocal x; x = ...`` (a warn-once flag, a call counter):
                # state, whose value changes as the code runs.
                self._record(f"{owner}.<{cell_name}>", _STATE)
                continue
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
        for name in sorted(globals_):
            if name in namespace:
                self._walk_name(namespace, name, f"{here}:{name}", globals_, attrs)
        package = namespace.get("__package__")
        package = package if isinstance(package, str) else None
        # Imports inside the body are recorded by their files, the same in a
        # process that has imported them and one that has not.
        for name in _resolve_imports(imports, package):
            self._record_files(name)

    def _record_files(self, name: str) -> None:
        """The file entries for module ``name`` and the project modules its
        file imports, once per walk."""
        entries = self._file_memo.get(name)
        if entries is None:
            entries, cut = _file_entries(name)
            self._file_memo[name] = entries
            self._ops += len(entries)  # file work counts against the bound
            self.truncated = self.truncated or cut
        for key, digest in entries.items():
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
        for name in sorted(new):
            if name in namespace:
                self._walk_name(namespace, name, f"{here}:{name}", set(), attrs)

    def _walk_name(self, namespace: dict, name: str, key: str, globals_, attrs) -> None:
        """A module-level name the reaching code uses. Code (a function, a
        class, a module, code behind a C wrapper) is walked. A value in a
        project file assigned only literals is recorded by its value, which is
        what runs; one computed on import, by the source of its assignment (it
        then reads alike on every node), with the code it holds walked. In a
        module with no file (a notebook) values are recorded as they are,
        unless the module's own code rebinds them."""
        import types

        value = namespace[name]
        if isinstance(value, type) and value.__module__ == namespace.get("__name__"):
            self._class_home.setdefault(id(value), namespace)
        code_like = (types.FunctionType, types.MethodType, type, types.ModuleType)
        wrapped = None
        if callable(value) and not isinstance(value, (functools.partial, type)):
            try:
                wrapped = getattr(value, "__wrapped__", None)
            except Exception:  # noqa: BLE001 - an attribute that raises
                wrapped = None
        if isinstance(value, code_like) or isinstance(wrapped, types.FunctionType):
            self._push(self._walk_value, value, key, globals_, attrs, 0)
            return
        if name in self._state(namespace):
            self._record(key, _STATE)
            return
        file = namespace.get("__file__")
        module_name = namespace.get("__name__")
        if (
            isinstance(file, str)
            and file.endswith(".py")
            and isinstance(module_name, str)
        ):
            digest = _source_constant(file, module_name, name)
            if isinstance(digest, tuple):
                _, written = digest
                try:
                    same = type(value) is type(written) and value == written
                except Exception:  # noqa: BLE001 - a comparison that raises
                    same = False
                if not same:
                    return  # changed at runtime, or edited and not reloaded
            else:
                if digest is not None:
                    self._record(key, _SECRET if _secret(key, value) else digest)
                # The code the value holds, without a value entry.
                self._push(self._walk_value, value, key, globals_, attrs, 1)
                return
        self._push(self._walk_value, value, key, globals_, attrs, 0)

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
        home = self._class_namespace(cls)
        if home is not None:
            # Its bases and nested classes from the same module live there too.
            related = [v for _, k, v in members if k == "class"] + list(cls.__bases__)
            for other in related:
                if getattr(other, "__module__", None) == cls.__module__:
                    self._class_home.setdefault(id(other), home)
        digest = self._object_hash.get(id(cls))
        if digest is None:
            parts = [cls.__qualname__]
            for attr, kind, value in members:
                if kind == "function":
                    parts.append(f"{attr}={self._function_digest(value)}")
                elif kind == "class":
                    parts.append(f"{attr}=class {value.__qualname__}")
            # Class-level values by the class body's source (a value a method
            # fills in later, such as a lazy cache, is state, not code). With
            # no file (a notebook), the immutable ones by value.
            body = self._class_source(cls)
            if body is _NO_FILE:
                stored = _stored_attrs(
                    value.__code__ for _, kind, value in members if kind == "function"
                )
                for attr, kind, value in members:
                    const = _constant_repr(value) if kind == "value" else None
                    if const is not None and attr not in stored:
                        secret = _secret(attr, value)
                        parts.append(f"{attr}={'<secret>' if secret else const}")
            elif body is not None:
                parts.append(f"body={body}")
            parts.extend(
                f"base={self._class_label(b) if self._project(b) else b.__module__}.{b.__qualname__}"
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
                key = f"{self._class_label(value)}:{value.__qualname__}"
                self._push(self._walk_class, value, key)
        for base in cls.__bases__:
            if self._project(base):
                key = f"{self._class_label(base)}:{base.__qualname__}"
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
            # hydra-zen routes a project object through its own functions:
            # just(Cls) is get_obj with `path`, builds(..., zen_meta=...) is
            # zen_processing with `_zen_target`.
            for extra in (
                item.get("_zen_target"),
                item.get("path") if target == "hydra_zen.funcs.get_obj" else None,
            ):
                if isinstance(extra, str) and extra not in found:
                    found.append(extra)
            pending.extend(item.values())
        elif isinstance(item, (list, tuple)):
            pending.extend(item)
    return sorted(found)


_TRUNCATION_WARNED: list = []


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
    ``_target_`` (or through hydra-zen's ``just()`` and ``zen_meta``), by the
    file of their module. The roots' own
    bodies are not entries (``code_fingerprint`` and ``Study``'s methods
    fingerprint cover them), except those in ``with_body`` (``pre_task``).

    Not reached: the standard library, mushin, installed packages other than
    editable ones (a dependency upgrade is provenance's job), data files, the
    contents of a mutable container other than the code it holds, code called
    only through a string that is not a ``_target_`` of the config, a
    ``_target_`` in the driver script (``__main__``), and attribute
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
        _warn_if_truncated(walk)
        return {
            "python": f"{sys.version_info[0]}.{sys.version_info[1]}",
            "code": walk.hashes,
        }
    except Exception:  # noqa: BLE001 - a best-effort guard never fails the sweep
        return None


def _warn_if_truncated(walk: _CodeWalk) -> None:
    """Warn, once per process, that a walk reached one of its bounds."""
    if not walk.truncated or _TRUNCATION_WARNED:
        return
    _TRUNCATION_WARNED.append(True)
    import warnings

    try:
        warnings.warn(
            "resume/cache_dir: the project code this task reaches is larger than "
            "the fingerprint walks, so edits past that point are not compared; "
            "use a fresh working_dir after a large refactor.",
            UserWarning,
            stacklevel=4,
        )
    except Warning:  # warnings as errors: the walk is best effort
        pass


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
                _warn_if_truncated(walk)
            except Exception:  # noqa: BLE001 - best effort: the roots' record stands
                entries = {}
            while len(_TARGET_RECORDS) >= _TARGET_RECORDS_MAX:
                _TARGET_RECORDS.pop(next(iter(_TARGET_RECORDS)))
            _TARGET_RECORDS[(token, target)] = entries
        code.update(entries)
    return {"python": base["python"], "code": code}


def merge_project_records(prior, current):
    """The record to keep for a cell reused under ``current``. A cell keeps its
    prior record when it has one taken under the same Python version: only the
    keys both records share were compared, so writing keys only ``current``
    has would declare the cell computed by code it was never checked against
    (a name reached lazily, an alias pointed elsewhere). A cell with no record
    (written before the field), or one taken under another Python version
    (which can never be compared again), takes the current one: reusing it
    already trusted it, and the cell is guarded from now on."""
    if (
        isinstance(prior, dict)
        and isinstance(prior.get("code"), dict)
        and (current is None or prior.get("python") == current.get("python"))
    ):
        return prior
    # No record, or one taken under another Python version, which can never
    # be compared again: the current one, so the cell is guarded from now on.
    return current if current is not None else prior


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
        data = json.loads(p.read_text(encoding="utf-8"))
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
