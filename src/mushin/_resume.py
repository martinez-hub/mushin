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


#: Values whose repr is the same in every process, by type: what a partial
#: task's bound arguments are compared by.
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


def _bound_repr(value, depth: int = 0) -> str | None:
    """How a partial task's bound argument is compared: a constant (as
    ``_constant_repr``), a list, dict or set of them by value, a function or
    class by its qualified name (its code is covered by its module's file);
    None for anything else (an array, an object), which a cache key cannot
    tell apart from another of its type."""
    const = _constant_repr(value, depth)
    if const is not None:
        return const
    name = getattr(value, "__qualname__", None)
    if isinstance(value, type) or (callable(value) and isinstance(name, str)):
        return f"<{getattr(value, '__module__', '?')}.{name}>"
    if isinstance(value, functools.partial) and depth < 4:
        inner = [_bound_repr(value.func, depth + 1)]
        inner += [_bound_repr(a, depth + 1) for a in value.args]
        inner += [
            "<secret>" if _secret_name(k, v) else _bound_repr(v, depth + 1)
            for k, v in sorted((value.keywords or {}).items())
        ]
        return None if None in inner else f"partial({','.join(inner)})"
    if depth >= 4:
        return None
    if isinstance(value, (list, set)) and len(value) <= 1000:
        items = [_bound_repr(v, depth + 1) for v in value]
        if None in items:
            return None
        if isinstance(value, set):
            items.sort()
        return f"{type(value).__name__}({','.join(items)})"
    if isinstance(value, dict) and len(value) <= 1000:
        items = []
        for k, v in value.items():
            key = _constant_repr(k, depth + 1)
            inner = "<secret>" if _secret_name(str(k), v) else _bound_repr(v, depth + 1)
            if key is None or inner is None:
                return None
            items.append(f"{key}:{inner}")
        return f"dict({','.join(sorted(items))})"
    return None


def _partial_digest(value) -> str:
    """A partial task's bound arguments, as ``_bound_repr`` compares them; an
    argument it cannot represent by its type."""
    parts = []
    for a in value.args:
        parts.append(_bound_repr(a) or f"<{type(a).__module__}.{type(a).__qualname__}>")
    for k, v in sorted((value.keywords or {}).items()):
        rep = "<secret>" if _secret_name(k, v) else _bound_repr(v)
        parts.append(f"{k}={rep or f'<{type(v).__module__}.{type(v).__qualname__}>'}")
    return _short_hash("|".join(parts).encode())


def partial_task_cacheable(task) -> bool:
    """Whether every argument a partial task binds can be told apart from
    another of its type (``_bound_repr``), so a cache key can include it.
    Main never cached a partial task; one that binds an array or an object
    still is not, since two such partials over different data would share an
    entry."""
    while isinstance(task, functools.partial):
        values = list(task.args) + list((task.keywords or {}).values())
        if any(_bound_repr(v) is None for v in values):
            return False
        task = task.func
    return True


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


def _is_main_guard(node) -> bool:
    """``if __name__ == "__main__":`` (either way round)."""
    import ast

    if not isinstance(node, ast.If) or not isinstance(node.test, ast.Compare):
        return False
    if len(node.test.ops) != 1 or not isinstance(node.test.ops[0], ast.Eq):
        return False
    parts = [node.test.left, *node.test.comparators]
    names = {p.id for p in parts if isinstance(p, ast.Name)}
    consts = {p.value for p in parts if isinstance(p, ast.Constant)}
    return "__name__" in names and "__main__" in consts


def _strip_tree(tree):
    """A module's syntax tree without what does not run on import or is not
    code: ``if __name__ == "__main__":`` blocks (an imported module's driver
    block) and docstrings, at any depth."""
    import ast

    is_main_guard = _is_main_guard

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
#: (digest, tree). A record that reaches the same file through many imports
#: reads and parses it once.
_FILE_CACHE: dict[tuple[str, int, int], tuple[str, object]] = {}
_FILE_CACHE_MAX = 2048


def _cache_put(cache: dict, key, value) -> None:
    while len(cache) >= _FILE_CACHE_MAX:
        cache.pop(next(iter(cache)))
    cache[key] = value


def _stat_key(path: Path) -> tuple[str, int, int]:
    st = path.stat()
    return (str(path), st.st_mtime_ns, st.st_size)


def _parsed(path: Path):
    """``(digest, tree)`` for a project file, cached: a Python file is hashed as
    its syntax tree without its main block and docstrings, with secret-shaped
    strings blanked (a comment, formatting, a docstring or the driver block is
    not an edit), anything else by its bytes (``tree`` is then None). ``tree``
    is the stripped tree: its imports are the ones the module can run."""
    import ast

    cache_key = _stat_key(path)
    hit = _FILE_CACHE.get(cache_key)
    if hit is not None:
        return hit
    data = path.read_bytes()
    result: tuple[str, object] = (_short_hash(data), None)
    if path.suffix == ".py":
        try:
            tree = _strip_tree(ast.parse(data))
            digest = _short_hash(_canonical_dump(_sanitized(tree)).encode())
            result = (digest, tree)
        except (SyntaxError, ValueError, RecursionError):
            pass
    _cache_put(_FILE_CACHE, cache_key, result)
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


def _file_imports(
    tree, name: str, is_package: bool, package: str | None = None
) -> list[str]:
    """The modules a file imports, at any depth, with ``from pkg import sub``
    naming the submodule too, relative imports resolved against the file's own
    package (or ``package``, for a script run with ``python -m``)."""
    import ast

    if package is None:
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


#: Bounds on one record: project files, and module names looked at (most are
#: the standard library's or installed packages', which cost a lookup each).
_FILES_MAX = 400
_NAMES_MAX = 20_000


def _file_entries(names) -> tuple[dict[str, str], bool]:
    """Entries for project modules ``names`` and, through their files, every
    project module they import in turn, at any depth (a helper imported inside
    a function, and the constants it imports); and whether ``_FILES_MAX`` cut
    the list short. A loaded module that no longer matches its file is
    ``_STALE``."""
    out: dict[str, str] = {}
    pending = list(reversed(list(names)))
    seen: set[str] = set()
    files = 0
    while pending and files < _FILES_MAX and len(seen) < _NAMES_MAX:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        found = _file_entry(current)
        if found is None:
            continue
        files += 1
        key, digest, tree = found
        out[key] = _STALE if _loaded_stale(current, digest) else digest
        if tree is not None:
            file = _find_file(current) or ""
            is_package = Path(file).name == "__init__.py"
            pending.extend(reversed(_file_imports(tree, current, is_package)))
    return out, any(p not in seen for p in pending)


#: Each script's digest, per process: (path, mtime_ns, size) -> (digest, tree).
_SCRIPT_CACHE: dict[tuple[str, int, int], tuple[str, object]] = {}


def _script_digest(path: Path):
    """``(digest, kept)`` for a driver script: its code without what drives the
    sweep, so the resilience loop (extend the grid, add a print, resume) is
    not an edit; ``kept`` is a module of the statements that count, whose
    imports are followed. Outside the main block, every statement counts but a
    call to a ``run`` method or to ``print``, a bare name or constant, and an
    assignment to names nothing that counts reads (``GRID = dict(...)``, read
    only by the run call). In the main block, only a function, class or
    assignment that something that counts reads (a helper the task calls);
    not its imports, calls or prints. Docstrings never count."""
    import ast
    import copy

    cache_key = _stat_key(path)
    hit = _SCRIPT_CACHE.get(cache_key)
    if hit is not None:
        return hit
    tree = ast.parse(path.read_bytes())

    def drives(node) -> bool:
        if not isinstance(node, ast.Expr):
            return False
        call = node.value
        if not isinstance(call, ast.Call):
            return True  # a bare name or constant (a docstring) does nothing
        func = call.func
        return (isinstance(func, ast.Attribute) and func.attr == "run") or (
            isinstance(func, ast.Name) and func.id == "print"
        )

    def bound_names(node) -> set[str] | None:
        """The names a statement binds when binding names is all it does;
        None when it does more (imports, sets an attribute or an item, calls
        something at the top level)."""
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            return {node.name}
        if isinstance(node, ast.Expr):
            return None
        names: set[str] = set()
        for sub in ast.walk(node):
            if isinstance(
                sub,
                (
                    ast.Import,
                    ast.ImportFrom,
                    ast.Global,
                    ast.Nonlocal,
                    # A definition inside `try:` or `if ...:` counts as one
                    # at the top level does.
                    ast.FunctionDef,
                    ast.AsyncFunctionDef,
                    ast.ClassDef,
                ),
            ):
                return None
            if isinstance(sub, (ast.Attribute, ast.Subscript)) and isinstance(
                sub.ctx, (ast.Store, ast.Del)
            ):
                return None
            if isinstance(sub, ast.Name) and isinstance(sub.ctx, (ast.Store, ast.Del)):
                names.add(sub.id)
        return names

    kept, pending = [], []
    for node in tree.body:
        if _is_main_guard(node):
            for inner in node.body:
                if isinstance(inner, (ast.Import, ast.ImportFrom)) or drives(inner):
                    continue
                names = bound_names(inner)
                if names:  # a call or assertion in the main block: not counted
                    pending.append((inner, names))
            nodes = node.orelse  # runs on import
        else:
            nodes = [node]
        for inner in nodes:
            if drives(inner):
                continue
            names = bound_names(inner)
            is_def = isinstance(
                inner, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
            )
            if names is None or is_def:
                kept.append((inner, names))
            else:
                pending.append((inner, names))
    while pending:
        read = {
            sub.id
            for node, _ in kept
            for sub in ast.walk(node)
            if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Load)
        }
        add = [(n, names) for n, names in pending if names & read]
        if not add:
            break
        kept += add
        pending = [(n, names) for n, names in pending if not names & read]
    kept.sort(key=lambda pair: pair[0].lineno)
    module = ast.Module(body=[copy.deepcopy(n) for n, _ in kept], type_ignores=[])
    module = _strip_tree(module)  # docstrings inside what counts
    digest = _short_hash(_canonical_dump(_sanitized(module)).encode())
    _cache_put(_SCRIPT_CACHE, cache_key, (digest, module))
    return digest, module


#: The code objects compiled from each file as it is now, per process:
#: (path, mtime_ns, size) -> {(qualname, first line): [code, ...]}, or None
#: when the file does not compile.
_COMPILED_CACHE: dict[tuple[str, int, int], dict | None] = {}


def _compiled_index(file: str):
    import sys
    import types

    path = Path(file)
    try:
        cache_key = _stat_key(path)
    except OSError:
        return None
    if cache_key in _COMPILED_CACHE:
        return _COMPILED_CACHE[cache_key]
    try:
        # As the import system compiles a source file, so an unedited file
        # gives code objects equal to the loaded ones.
        code = compile(
            path.read_bytes(),
            file,
            "exec",
            dont_inherit=True,
            optimize=sys.flags.optimize,
        )
    except (SyntaxError, ValueError, OSError, RecursionError, MemoryError):
        index = None
    else:
        index: dict | None = {}
        stack = [code]
        while stack:
            current = stack.pop()
            index.setdefault((current.co_qualname, current.co_firstlineno), []).append(
                current
            )
            stack.extend(k for k in current.co_consts if isinstance(k, types.CodeType))
    _cache_put(_COMPILED_CACHE, cache_key, index)
    return index


def _defined_codes(namespace: dict, file: str) -> list:
    """The code objects of the functions and methods a module's namespace
    defines from ``file``: its functions, its classes' methods (static, class,
    property parts, nested classes), through decorators' ``__wrapped__``."""
    import types

    module_name = namespace.get("__name__")
    out, seen = [], set()
    stack = list(namespace.values())
    steps = 0
    while stack and steps < _NAMES_MAX:
        steps += 1
        value = stack.pop()
        if id(value) in seen:
            continue
        seen.add(id(value))
        try:
            if isinstance(value, (staticmethod, classmethod)):
                stack.append(value.__func__)
            elif isinstance(value, property):
                stack.extend(f for f in (value.fget, value.fset, value.fdel) if f)
            elif isinstance(value, types.FunctionType):
                if value.__code__.co_filename == file:
                    out.append(value.__code__)
                wrapped = getattr(value, "__wrapped__", None)
                if wrapped is not None:
                    stack.append(wrapped)
            elif isinstance(value, type):
                if value.__module__ == module_name:
                    stack.extend(vars(value).values())
            else:
                # A C wrapper (functools.cache) around a function.
                wrapped = getattr(value, "__wrapped__", None)
                if isinstance(wrapped, types.FunctionType):
                    stack.append(wrapped)
        except Exception:  # noqa: BLE001 - a value that resists inspection
            continue
    return out


def _same_code(a, b) -> bool:
    """Whether two code objects are the same code: instructions, constants
    (nested code alike), names and first line. Not their compile flags, which
    a runner that inherits ``__future__`` flags (IPython's ``%run``) sets
    differently for the same source."""
    import types

    stack = [(a, b)]
    while stack:
        x, y = stack.pop()
        if (
            x.co_code != y.co_code
            or x.co_names != y.co_names
            or x.co_varnames != y.co_varnames
            or x.co_firstlineno != y.co_firstlineno
            or len(x.co_consts) != len(y.co_consts)
        ):
            return False
        for cx, cy in zip(x.co_consts, y.co_consts, strict=True):
            if isinstance(cx, types.CodeType) and isinstance(cy, types.CodeType):
                stack.append((cx, cy))
                continue
            try:
                if type(cx) is not type(cy) or cx != cy:
                    return False
            except Exception:  # noqa: BLE001 - a constant that cannot compare
                return False
    return True


def _namespace_stale(namespace: dict, file) -> bool:
    """Whether the code a module's namespace runs is no longer what ``file``
    compiles to: the file was edited after the process loaded it, and not
    reloaded. The record then must not describe the file, or a cell this
    process computes with the old code would be filed under the new."""
    if not isinstance(file, str):
        return False
    index = _compiled_index(file)
    if index is None:
        return True
    for code in _defined_codes(namespace, file):
        compiled = index.get((code.co_qualname, code.co_firstlineno), ())
        if not any(_same_code(code, c) for c in compiled):
            _warn_stale(file)
            return True
    return False


#: The file digest each loaded module had when this process first recorded
#: that load of it: name -> (its ``__spec__``, digest). A reload replaces the
#: spec, so a new load starts afresh.
_LOAD_BASELINE: dict[str, tuple[object, str]] = {}


def _loaded_stale(name: str, digest: str) -> bool:
    """Whether module ``name``, loaded from source by the standard loader, no
    longer matches its file, whose digest is now ``digest``: a function or
    method differs from what the file compiles to, or the file changed since
    this process first recorded this load of the module (a constant edited
    and not reloaded). A module an import hook rewrites (pytest's assertion
    rewriting, a type-checking hook) never matches a plain compile, so it is
    compared by its file only."""
    import importlib.machinery
    import sys

    module = sys.modules.get(name)
    if module is None:
        return False
    if type(getattr(module, "__loader__", None)) is not (
        importlib.machinery.SourceFileLoader
    ):
        return False
    file = getattr(module, "__file__", None)
    spec = getattr(module, "__spec__", None)
    seen = _LOAD_BASELINE.get(name)
    if seen is None or seen[0] is not spec:
        _LOAD_BASELINE[name] = (spec, digest)
    elif seen[1] != digest:
        _warn_stale(str(file))
        return True
    try:
        namespace = vars(module)
    except TypeError:
        return False
    return _namespace_stale(namespace, file)


#: Each script's literal top-level names, per process: (path, mtime_ns,
#: size) -> {name: value}.
_LITERALS_CACHE: dict[tuple[str, int, int], dict] = {}


def _script_literals(path: Path) -> dict:
    """The top-level names a script assigns exactly once, to a literal, and
    that no ``global`` statement in it declares: what its loaded values must
    still equal if the script was not edited since they were set."""
    import ast

    cache_key = _stat_key(path)
    hit = _LITERALS_CACHE.get(cache_key)
    if hit is not None:
        return hit
    tree = ast.parse(path.read_bytes())
    counts: dict[str, int] = {}
    literals: dict[str, object] = {}
    declared = {
        n for sub in ast.walk(tree) if isinstance(sub, ast.Global) for n in sub.names
    }
    stack = list(tree.body)
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            counts[node.name] = counts.get(node.name, 0) + 1
            continue
        for sub in ast.walk(node):
            if isinstance(sub, ast.Name) and isinstance(sub.ctx, (ast.Store, ast.Del)):
                counts[sub.id] = counts.get(sub.id, 0) + 1
        if isinstance(node, (ast.Assign, ast.AnnAssign)) and node.value is not None:
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if len(targets) == 1 and isinstance(targets[0], ast.Name):
                try:
                    literals[targets[0].id] = ast.literal_eval(node.value)
                except (
                    ValueError,
                    TypeError,
                    SyntaxError,
                    MemoryError,
                    RecursionError,
                ):
                    pass
        for field in ("body", "orelse", "finalbody", "handlers"):
            inner = getattr(node, field, None)
            if isinstance(inner, list):
                stack.extend(n for n in inner if isinstance(n, ast.stmt))
    out = {
        k: v for k, v in literals.items() if counts.get(k) == 1 and k not in declared
    }
    _cache_put(_LITERALS_CACHE, cache_key, out)
    return out


def _script_stale(namespace: dict, path: Path) -> bool:
    """Whether a script's loaded code (the driver's ``__main__``, or the copy a
    parallel worker unpickled) no longer matches its file: a function differs
    from what it compiles to, or a literal top-level name's value differs from
    the literal (a constant edited while the sweep ran)."""
    file = str(path)
    if _namespace_stale(namespace, file):
        return True
    try:
        literals = _script_literals(path)
    except (OSError, SyntaxError, ValueError, RecursionError):
        return True
    for name, written in literals.items():
        if name in namespace:
            value = namespace[name]
            try:
                same = type(value) is type(written) and value == written
            except Exception:  # noqa: BLE001 - a comparison that raises
                same = False
            if not same:
                _warn_stale(file)
                return True
    return False


#: The digest of a module this process loaded before its file was edited: it
#: never equals a file's digest, so a cell computed now with the old code
#: re-runs once a process runs the new code, and is reused meanwhile.
_STALE = _short_hash(b"<stale>")
_STALE_WARNED: set[str] = set()


def _warn_stale(file: str) -> None:
    if file in _STALE_WARNED:
        return
    _STALE_WARNED.add(file)
    _best_effort_warning(
        f"resume/cache_dir: {file} changed on disk after this process imported "
        "it, so cells computed now run the old code and are recorded as such; "
        "they re-run once the module is reloaded or the process restarts."
    )


def _best_effort_warning(message: str) -> None:
    import warnings

    try:
        warnings.warn(message, UserWarning, stacklevel=4)
    except Warning:  # warnings as errors: the record is best effort
        pass


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


def _namespace_modules(namespace: dict) -> list[str]:
    """For a module with no file (a notebook): the project modules its names
    refer to, by module values and by the modules their functions and classes
    come from. Its own code is not compared."""
    import types

    names = set()
    for value in list(namespace.values()):
        try:
            if isinstance(value, types.ModuleType):
                names.add(value.__name__)
            elif isinstance(value, (types.FunctionType, type)):
                module = value.__module__
                if isinstance(module, str) and module != namespace.get("__name__"):
                    names.add(module)
        except Exception:  # noqa: BLE001 - a value that resists inspection
            continue
    return sorted(names)


def _root_layers(root) -> list:
    """A root and what it wraps: a partial's function, a decorator's
    ``__wrapped__``, at most a few layers, each object once."""
    layers, seen = [], set()
    layer = root
    for _ in range(16):
        if layer is None or id(layer) in seen:
            break
        seen.add(id(layer))
        layers.append(layer)
        if isinstance(layer, functools.partial):
            layer = layer.func
        else:
            try:
                layer = getattr(layer, "__wrapped__", None)
            except Exception:  # noqa: BLE001 - an attribute that raises
                break
    return layers


def _root_entries(root, code: dict[str, str], seeds: list[str]) -> None:
    """Add the entries a root gives directly (a driver script's, a partial's
    bound arguments) to ``code``, and the project modules to follow to
    ``seeds``."""
    import sys
    import types

    for layer in _root_layers(root):
        if isinstance(layer, functools.partial):
            inner = layer.func
            while isinstance(inner, functools.partial):
                inner = inner.func
            name = getattr(inner, "__qualname__", type(inner).__qualname__)
            module = getattr(inner, "__module__", "?")
            code[f"{module}:{name}.<partial>"] = _partial_digest(layer)
            continue
        if isinstance(layer, types.FunctionType):
            namespace = layer.__globals__
        else:
            cls = layer if isinstance(layer, type) else type(layer)
            module = sys.modules.get(getattr(cls, "__module__", ""), None)
            namespace = getattr(module, "__dict__", None) or {}
        name, file = namespace.get("__name__"), namespace.get("__file__")
        if not isinstance(name, str):
            continue
        if name == "__main__" and isinstance(file, str) and file.endswith(".py"):
            path = Path(file)
            digest, tree = _script_digest(path)
            stale = _script_stale(namespace, path)
            code[f"{path.stem} (script)"] = _STALE if stale else digest
            package = namespace.get("__package__")
            seeds.extend(
                _file_imports(
                    tree,
                    "__main__",
                    False,
                    package=package if isinstance(package, str) else None,
                )
            )
        elif not isinstance(file, str):
            seeds.extend(_namespace_modules(namespace))  # a notebook
        elif not _is_foreign_module(name, file):
            seeds.append(name)


def _target_module(path: str) -> str | None:
    """The module a ``_target_`` string's class or function lives in, found
    without importing it; None for one in the driver script (``__main__``,
    covered when the script imports or defines it) or in foreign code."""
    parts = path.split(".")
    if parts[0] == "__main__" or _is_foreign_module(parts[0]):
        return None
    for i in range(len(parts) - 1, 0, -1):
        module = ".".join(parts[:i])
        if _find_file(module) is not None:
            return module
    return None


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


#: Warned once per process each: a record cut short by its bounds, and one
#: that could not be taken.
_TRUNCATION_WARNED: list = []
_FAILURE_WARNED: list = []


def project_fingerprints(fn, *more, cfg=None, with_body=()) -> dict | None:
    """What the project code a task can run looks like now, as recorded in
    each cell's status sidecar: ``{"python": "3.11", "code": {key: hash}}``.

    Every project file the task's module imports, directly or through other
    project files, at any depth (imports inside functions included), is an
    entry ``"<module> (file)"``, hashed as its syntax tree without its main
    block and docstrings: an edit anywhere in such a file is a change, and a
    comment, a docstring or formatting is not. A task defined in the driver
    script gives ``"<script> (script)"``, its top-level code without what
    drives the sweep (see ``_script_digest``), and the project files the
    script imports. A module that this process loaded before its file was
    edited is recorded as ``_STALE``. ``more`` and ``with_body`` add roots
    walked the same way (``pre_task``; ``Study``'s training functions), and
    ``cfg`` adds the modules of the classes and functions it names as
    ``_target_`` (or through hydra-zen's ``just()`` and ``zen_meta``). A partial
    task's bound arguments are an entry too.

    Not covered: the standard library, mushin, installed packages other than
    editable ones (a dependency upgrade is provenance's job), data files, the
    environment, code defined in a notebook (the project modules it imports
    are covered), and code reached only through a string that is not a
    ``_target_`` of the config. None when the record cannot be taken; a
    resume then does not compare, and ``cache_dir`` is not used."""
    import sys

    try:
        code: dict[str, str] = {}
        seeds: list[str] = []
        for root in (fn, *more, *with_body):
            if root is not None:
                _root_entries(root, code, seeds)
        if cfg is not None:
            seeds += [m for m in map(_target_module, _config_targets(cfg)) if m]
        entries, cut = _file_entries(seeds)
        code.update(entries)
        if cut and not _TRUNCATION_WARNED:
            _TRUNCATION_WARNED.append(True)
            _best_effort_warning(
                "resume/cache_dir: the task's code imports more project files than "
                "the fingerprint covers, so edits past that point are not compared; "
                "use a fresh working_dir after a large refactor."
            )
        return {
            "python": f"{sys.version_info[0]}.{sys.version_info[1]}",
            "code": code,
        }
    except Exception:  # noqa: BLE001 - a best-effort guard never fails the sweep
        if not _FAILURE_WARNED:
            _FAILURE_WARNED.append(True)
            _best_effort_warning(
                "resume/cache_dir: the project code this task uses could not be "
                "fingerprinted, so a resume does not compare it and cache_dir is not "
                "used; use a fresh working_dir after editing it."
            )
        return None


#: The roots' record per run token, in this process: a parallel worker takes
#: it once per run however many cells it runs. Bounded, since a reused worker
#: process (loky keeps them across run() calls) sees many runs.
_ROOT_RECORDS: dict[str, dict | None] = {}
_ROOT_RECORDS_MAX = 64


#: Each ``_target_``'s entries per (run token, target), in this process: every
#: cell of a sweep that names the same classes shares them. Bounded like the
#: roots' records.
_TARGET_RECORDS: dict[tuple[str, str], dict[str, str]] = {}
_TARGET_RECORDS_MAX = 4096


def cell_project_record(token: str, roots, with_body, cfg) -> dict | None:
    """The project-code record of one cell, taken in the process that runs it:
    the roots' record (cached per process for the run ``token``) plus the
    modules of this cell's config ``_target_``\\ s, which can differ from cell
    to cell (a swept model class)."""
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
                module = _target_module(target)
                entries = _file_entries([module])[0] if module else {}
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
