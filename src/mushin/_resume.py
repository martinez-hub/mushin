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


#: What ``resolved_config`` returns for a config that cannot be resolved.
_UNRESOLVED = object()


def resolved_config(cfg):
    """A job config as plain containers with its interpolations resolved, or
    ``_UNRESOLVED`` when it cannot be. A cell resolves its config once, here,
    and its fingerprint and its ``_target_`` scan share the result: resolving
    a large structured config is not free, and a resolver that reads a clock
    or a counter then gives both the same values. A config that is already
    plain containers is returned as it is."""
    try:
        from omegaconf import OmegaConf

        return (
            OmegaConf.to_container(cfg, resolve=True)
            if OmegaConf.is_config(cfg)
            else cfg
        )
    except Exception:  # noqa: BLE001 - unresolvable config -> no guard
        return _UNRESOLVED


def config_fingerprint(cfg) -> str | None:
    """Stable short hash of the fully-resolved job config, or None if the
    config cannot be resolved/serialized.

    Guards resume reuse: a completed cell's cached metrics are only returned
    when the config that would run now matches the one that produced them;
    otherwise a changed NON-swept value (same combo key) would silently mix
    results from two configurations into one dataset. Task *source* changes are
    guarded separately by ``code_fingerprint``."""
    return resolved_config_hash(resolved_config(cfg))


def resolved_config_hash(data) -> str | None:
    """``config_fingerprint`` of a config ``resolved_config`` returned."""
    if data is _UNRESOLVED:
        return None
    try:
        payload = json.dumps(data, sort_keys=True, default=str)
    except Exception:  # noqa: BLE001 - unserializable config -> no guard
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
    project's. A distribution installed under the interpreter's own
    directories is left out: its files are foreign by path already, and
    reading the metadata of every one cost a fifth of a second in every
    process. Read once per process."""
    import importlib.metadata as md
    import json

    interpreter = _foreign_roots()
    by_name: dict[str, set[Path]] = {}
    for dist in md.distributions():
        try:
            where = Path(str(dist.locate_file(""))).resolve()
            if any(where.is_relative_to(root) for root in interpreter):
                continue
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
    if top == "mushin":
        return True
    if file is None:
        module = sys.modules.get(name)
        file = getattr(module, "__file__", None) if module is not None else None
    if file is None:
        file = _find_file(name)  # not imported yet: where it would load from
    if isinstance(file, str) and file:
        # The file decides: a checkout's `code/` or `test.py` is the
        # project's, though the standard library has that name.
        return _foreign_by_path(file, top)
    if top in sys.stdlib_module_names or top in sys.builtin_module_names:
        return True
    try:
        return bool(_installed_roots().get(top, ()))
    except Exception:  # noqa: BLE001 - unreadable metadata
        return False


def partial_task_cacheable(task) -> bool:
    """Whether a task may use ``cache_dir``: any task but a partial, as on
    main. A partial's bound arguments are data the key cannot see into (an
    array, a closure, a credential), so two partials over different data
    would share an entry; its bound functions' modules are still recorded."""
    return not isinstance(task, functools.partial)


def _resolve_imports(imports, package: str | None) -> list[str]:
    """Absolute module names for ``(name, level)`` imports (their parent
    packages are added where the names are followed, ``_file_entries``)."""
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
        if name and name not in found:
            found.append(name)
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


def _find_in(dotted: str, entries) -> tuple[object, list]:
    """``(spec, [])`` for module ``dotted`` in the directories ``entries``,
    or ``(None, portions)``: a namespace package's directories there (none if
    it is not found). Each directory's own finder, so nothing builds a
    _NamespacePath, which would need the parent imported."""
    import importlib.machinery as m
    import sys

    portions: list[str] = []
    for entry in entries:
        if not isinstance(entry, str):
            continue
        finder = sys.path_importer_cache.get(entry)
        try:
            if finder is None:
                finder = m.FileFinder(
                    entry,
                    (m.ExtensionFileLoader, m.EXTENSION_SUFFIXES),
                    (m.SourceFileLoader, m.SOURCE_SUFFIXES),
                    (m.SourcelessFileLoader, m.BYTECODE_SUFFIXES),
                )
            spec = finder.find_spec(dotted)
        except Exception:  # noqa: BLE001 - a finder that cannot look here
            continue
        if spec is None:
            continue
        if spec.loader is not None:
            return spec, []
        portions.extend(spec.submodule_search_locations or [])
    return None, portions


def _find_file(name: str) -> str | None:
    """The file a module name denotes, without importing anything: the loaded
    module's, or the one the import system would load, walking a dotted name's
    packages (namespace packages included) by their directories."""
    import importlib.util
    import os
    import sys

    loaded = sys.modules.get(name)
    if loaded is not None:
        file = getattr(loaded, "__file__", None)
        return file if isinstance(file, str) and file else None
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
        if i == 0:
            # Every finder, for the top level only (importing nothing): an
            # editable install may be found by a meta-path finder.
            try:
                spec = importlib.util.find_spec(dotted)
            except Exception:  # noqa: BLE001 - a finder that raises
                return None
            if spec is None:
                return None
            if spec.origin is None and spec.submodule_search_locations is not None:
                # A top-level namespace package: its directories on sys.path.
                path = [
                    os.path.join(e or os.curdir, dotted)
                    for e in sys.path
                    if isinstance(e, str)
                    and os.path.isdir(os.path.join(e or os.curdir, dotted))
                ]
            else:
                path = spec.submodule_search_locations
            continue
        spec, portions = _find_in(dotted, list(path))
        if spec is None:
            if not portions:
                return None
            path = portions  # a namespace portion
            continue
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


def _strip_tree(tree, main_guards: bool = True):
    """A module's syntax tree without what does not run on import or is not
    code: ``if __name__ == "__main__":`` blocks (an imported module's driver
    block) and docstrings, at any depth. ``main_guards=False`` keeps the
    guards: in a driver script they run, in its helpers too."""
    import ast

    def is_main_guard(node) -> bool:
        return main_guards and _is_main_guard(node)

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


def _import_pairs(tree) -> list[tuple[str, int]]:
    """``(name, level)`` for each module a syntax tree imports, at any depth,
    with ``from pkg import sub`` naming the submodule too."""
    import ast

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
    return [(n, lv) for n, lv in pairs if n]


def _resolved(pairs, name: str, is_package: bool, package: str | None = None):
    """``pairs`` as absolute module names, relative ones against the file's own
    package (or ``package``, for a script run with ``python -m``)."""
    if package is None:
        package = name if is_package else name.rpartition(".")[0]
    return _resolve_imports(pairs, package or None)


#: Per-process caches keyed by (path, hash of the bytes), bounded alike. They
#: hold digests and import lists, not syntax trees, so a large project costs
#: little memory in every worker.
_FILE_CACHE: dict = {}
_SCRIPT_CACHE: dict = {}
_FILE_CACHE_MAX = 2048


def _cache_put(cache: dict, key, value) -> None:
    while len(cache) >= _FILE_CACHE_MAX:
        cache.pop(next(iter(cache)))
    cache[key] = value


def _content_key(path: Path) -> tuple[bytes, tuple[str, str]]:
    """A file's bytes and the key the caches use for them: its path and a
    hash of its bytes, not its modification time, which a same-size edit
    within one tick of a coarse file clock (Windows) leaves as it was."""
    data = path.read_bytes()
    return data, (str(path), _short_hash(data))


def _parsed(path: Path):
    """``(digest, imports)`` for a project file, cached: a Python file is
    hashed as its syntax tree without its main block and docstrings (a
    comment, formatting, a docstring or the driver block is not an edit),
    anything else by its bytes. ``imports`` are the ``(name, level)`` pairs
    the stripped tree imports (none for a file that does not parse). Strings
    are hashed as they are: a whole file's digest cannot be guessed back to a
    secret in it without every other byte of the file, and blanking by shape
    hid ordinary settings (``"hf_tiny_random_gpt2"``)."""
    import ast

    data, cache_key = _content_key(path)
    hit = _FILE_CACHE.get(cache_key)
    if hit is not None:
        return hit
    result: tuple[str, list] = (_short_hash(data), [])
    if path.suffix == ".py":
        try:
            tree = _strip_tree(ast.parse(data))
            digest = _short_hash(_canonical_dump(tree).encode())
            result = (digest, _import_pairs(tree))
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
    """``(key, digest, imports, path)`` for a project module, by its file: the
    same entry whether or not this process has imported it."""
    path = _project_file_of(name)
    if path is None:
        return None
    try:
        digest, imports = _parsed(path)
    except (OSError, ValueError):
        return None
    return f"{name} (file)", digest, imports, path


#: Bounds on one record: project files, and module names looked at (most are
#: the standard library's or installed packages', which cost a lookup each).
_FILES_MAX = 400
_NAMES_MAX = 20_000


def _file_entries(names) -> tuple[dict[str, str], bool]:
    """Entries for project modules ``names`` and, through their files, every
    project module they import in turn, at any depth (a helper imported inside
    a function, and the constants it imports), each with the packages it is
    in (their ``__init__.py`` runs first); and whether ``_FILES_MAX`` cut
    the list short. Each file as it is now; one that changed since this
    process first recorded that load of it warns. A name that cannot be
    resolved costs its own entry only."""
    import sys

    out: dict[str, str] = {}
    pending = list(reversed(list(names)))
    seen: set[str] = set()
    foreign_tops: dict[str, bool] = {}
    files = 0
    while pending and files < _FILES_MAX and len(seen) < _NAMES_MAX:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        top = current.partition(".")[0]
        if top not in foreign_tops:
            # A loaded installed package (torch) is foreign as a whole: its
            # submodules need no lookup each.
            module = sys.modules.get(top)
            file = getattr(module, "__file__", None) if module is not None else None
            foreign_tops[top] = top == "mushin" or (
                isinstance(file, str) and bool(file) and _foreign_by_path(file, top)
            )
        if foreign_tops[top]:
            continue
        # Importing `pkg.models` runs `pkg/__init__.py` first, so a module's
        # packages are code it runs, whether or not it imports from them (a
        # `_target_`, or a task module that imports only the standard library).
        parent = current.rpartition(".")[0]
        if parent and parent not in seen:
            pending.append(parent)
        try:
            found = _file_entry(current)
        except Exception:  # noqa: BLE001 - one unresolvable name, not the record
            continue
        if found is None:
            continue
        files += 1
        key, digest, imports, path = found
        out[key] = _loaded_digest(current, digest)
        is_package = path.name == "__init__.py"
        pending.extend(reversed(_resolved(imports, current, is_package)))
    return out, any(p not in seen for p in pending)


def _script_digest(path: Path, roots: frozenset[str]):
    """``(digest, imports)`` for a driver script whose top-level names
    ``roots`` define the task (the class or function holding it,
    ``pre_task``, a ``_target_``): what those definitions can reach by name in
    the script, so the resilience loop (extend the grid, add a print, resume)
    is not an edit. Counted: the statements defining ``roots``, every
    statement, in the main block too, that binds a name counted code reads as
    a global (function parameters and locals do not count: the script's symbol
    table says which names are global), a star import (it may bind any name),
    and a decorated definition whose decorator registers it somewhere counted
    code reads (``@register`` defined in the script, or an alias of it,
    writing the ``MODELS`` the task reads)
    or comes from a project module; not one whose decorators are foreign
    (``@click.command()``, ``@hydra.main``, ``@dataclass``). A ``__future__``
    import always counts. A main block's body and its ``else`` (which runs
    where the script is imported, as in a spawn child) both count, the
    ``else`` marked as such. A name is bound by the statements
    that bind it in the module's scope (a ``match`` capture does; a
    comprehension's variable, a nested function's local and an annotation
    without a value do not). Not counted: statements that
    only do something (a call, the run call, an attribute or item set) and
    what only they read (``GRID``, ``ds = exp.run(...)``). ``imports`` are
    those of what counts."""
    import ast
    import copy
    import symtable

    source, content = _content_key(path)
    cache_key = (*content, tuple(sorted(roots)))
    hit = _SCRIPT_CACHE.get(cache_key)
    if hit is not None:
        return hit
    tree = ast.parse(source)
    defs = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)

    # The globals each top-level function or class body reads, by its name.
    # The bytes, so a byte order mark or a coding cookie is honored.
    table = symtable.symtable(source, str(path), "exec")
    scope_reads: dict[str, set[str]] = {}
    for child in table.get_children():
        # Every scope by its name: on 3.12+ a generic definition (`def f[T]`)
        # is a type-parameter scope named like it, holding the function's.
        names = scope_reads.setdefault(child.get_name(), set())
        stack = [child]
        while stack:
            t = stack.pop()
            stack.extend(t.get_children())
            names.update(
                sym.get_name()
                for sym in t.get_symbols()
                if sym.is_referenced() and sym.is_global()
            )

    def loads(nodes) -> set[str]:
        return {
            sub.id
            for node in nodes
            for sub in ast.walk(node)
            if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Load)
        }

    read_cache: dict[int, set[str]] = {}

    def reads(node) -> set[str]:
        hit = read_cache.get(id(node))
        if hit is None:
            hit = read_cache[id(node)] = _reads(node)
        return hit

    def _reads(node) -> set[str]:
        if isinstance(node, defs):
            # Evaluated where it is defined: decorators, defaults, bases.
            outer = list(node.decorator_list)
            if isinstance(node, ast.ClassDef):
                outer += node.bases + [k.value for k in node.keywords]
            else:
                outer += node.args.defaults + [
                    d for d in node.args.kw_defaults if d is not None
                ]
                a = node.args
                params = [*a.posonlyargs, *a.args, *a.kwonlyargs, a.vararg, a.kwarg]
                outer += [arg.annotation for arg in params if arg and arg.annotation]
                if node.returns is not None:
                    outer.append(node.returns)
            return scope_reads.get(node.name, set()) | loads(outer)
        return loads([node])

    def run_here(node) -> list:
        """What a definition evaluates where it is defined, but annotations:
        decorators and defaults, or decorators, bases and keywords."""
        if isinstance(node, ast.ClassDef):
            return [
                *node.decorator_list,
                *node.bases,
                *(k.value for k in node.keywords),
            ]
        args = node.args
        return [
            *getattr(node, "decorator_list", ()),
            *args.defaults,
            *(d for d in args.kw_defaults if d is not None),
        ]

    def declared_global(node) -> set[str]:
        """The names a definition, or a definition nested in it (a method, an
        inner function), declares ``global`` and binds in that same scope (by
        ``scope_binds``, so a walrus, an import or a capture binds too):
        running it binds the module's name. One it only reads binds nothing.
        Each nested scope is its own: its globals are found as it is walked."""
        found: set[str] = set()
        scopes = [node]
        while scopes:
            scope = scopes.pop()
            declared: set[str] = set()
            stack = list(scope.body)
            while stack:
                n = stack.pop()
                if isinstance(n, defs):
                    scopes.append(n)
                    continue
                if isinstance(n, ast.Lambda):
                    continue
                if isinstance(n, ast.Global):
                    declared.update(n.names)
                stack.extend(ast.iter_child_nodes(n))
            if declared:  # most scopes declare none: no second walk
                stored: set[str] = set()
                for stmt in scope.body:
                    stored |= scope_binds(stmt)
                found |= declared & stored
        return found

    def binds(node) -> set[str]:
        if isinstance(node, defs):
            # Its name, as on main. A global its body sets (a function's
            # when called, a class's when defined) and a walrus in its
            # header are not counted: a `def main()` that sets a global and
            # holds the run call would otherwise count its grid.
            return {node.name}
        return scope_binds(node, nested_globals=True)

    def scope_binds(node, nested_globals: bool = False) -> set[str]:
        """The names a statement binds in the scope it is in: its targets, a
        definition's name, an import's names, a ``match`` capture, and a
        walrus in what a nested definition evaluates there (its defaults,
        decorators, annotations); not a comprehension's variables or a nested
        scope's locals. With ``nested_globals``, also the globals a
        definition nested in the statement assigns (``declared_global``)."""
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            return {
                (alias.asname or alias.name).partition(".")[0] for alias in node.names
            }
        names: set[str] = set()
        comps = (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)
        # Iterative: a long generated expression must not exhaust the stack.
        stack = [(node, False)]
        while stack:
            sub, in_comp = stack.pop()
            if isinstance(sub, defs):
                names.add(sub.name)  # a definition inside `try:` or `if:`
                if nested_globals:
                    # A global that a definition in a block (or one nested
                    # in it) assigns binds the module's name, as main's walk
                    # over the whole block saw.
                    names.update(declared_global(sub))
                # Its body binds its own names; what it evaluates here (a
                # walrus in a default or an annotation) binds the module's.
                stack.extend((e, in_comp) for e in run_here(sub))
                if not isinstance(sub, ast.ClassDef):
                    a = sub.args
                    params = [*a.posonlyargs, *a.args, *a.kwonlyargs, a.vararg, a.kwarg]
                    stack.extend(
                        (p.annotation, in_comp) for p in params if p and p.annotation
                    )
                    if sub.returns is not None:
                        stack.append((sub.returns, in_comp))
                continue
            if isinstance(sub, ast.Lambda):
                args = sub.args
                stack.extend(
                    (e, in_comp)
                    for e in (*args.defaults, *(d for d in args.kw_defaults if d))
                )
                continue
            if isinstance(sub, ast.NamedExpr):
                if isinstance(sub.target, ast.Name):
                    names.add(sub.target.id)  # the module's, from a comprehension too
                stack.append((sub.value, in_comp))
                continue
            if isinstance(sub, comps):
                in_comp = True  # its variables are its own
            elif isinstance(sub, ast.Name) and isinstance(
                sub.ctx, (ast.Store, ast.Del)
            ):
                if not in_comp:
                    names.add(sub.id)
            elif isinstance(sub, ast.AnnAssign) and sub.value is None:
                # An annotation alone binds nothing, but what it evaluates
                # can: a walrus in the annotation, or in a subscript or
                # attribute target, which runs though nothing is assigned.
                stack.append((sub.annotation, in_comp))
                if not isinstance(sub.target, ast.Name):
                    stack.append((sub.target, in_comp))
                continue
            elif isinstance(sub, (ast.Import, ast.ImportFrom)):
                names.update(scope_binds(sub))
                continue
            elif isinstance(sub, (ast.MatchAs, ast.MatchStar)) and sub.name:
                names.add(sub.name)
            elif isinstance(sub, ast.MatchMapping) and sub.rest:
                names.add(sub.rest)
            stack.extend((child, in_comp) for child in ast.iter_child_nodes(sub))
        return names

    statements = []
    in_else: set[int] = set()
    for node in tree.body:
        if _is_main_guard(node):
            # The body runs as __main__; the else runs where the script is
            # imported (a spawn child re-imports it as __mp_main__). Both
            # count, the else marked as such, so moving a statement between
            # them is an edit.
            statements += node.body + node.orelse
            in_else.update(id(stmt) for stmt in node.orelse)
        else:
            statements.append(node)
    candidates = [(n, binds(n)) for n in statements]
    binders: dict[str, list] = {}
    for n, names in candidates:
        for name in names:
            binders.setdefault(name, []).append(n)

    def base(node):
        while isinstance(node, (ast.Call, ast.Attribute, ast.Subscript)):
            node = node.func if isinstance(node, ast.Call) else node.value
        return node.id if isinstance(node, ast.Name) else None

    def project_import(node, name: str) -> bool:
        for alias in node.names:
            bound = (alias.asname or alias.name).partition(".")[0]
            if bound != name and alias.name != "*":
                continue
            if isinstance(node, ast.ImportFrom):
                if node.level:
                    # Relative: the script's own package (`python -m
                    # pkg.train`), so the project's.
                    return True
                if not node.module:
                    return False
                module = node.module
            else:
                module = alias.name
            return _project_file_of(module) is not None
        return False

    def decorator_registers(name: str, node, read: set[str], seen: set) -> bool:
        """Main's rule for decorator ``name``: one of its binders (or, for a
        name no statement binds, a star import) is a project import, or is
        script code reading what counted code reads. Followed through an alias
        of it too (``register_model = register``), bounded."""
        if name in seen or len(seen) > 50:
            return False
        seen.add(name)
        # A name no statement binds may come from a star import.
        star = [b for b in binders.get("*", ()) if not binders.get(name)]
        for binder in [*binders.get(name, ()), *star]:
            if isinstance(binder, (ast.Import, ast.ImportFrom)):
                if project_import(binder, name):
                    return True
            elif binder is not node and reads(binder) & read:
                return True
        for binder in binders.get(name, ()):
            if isinstance(binder, ast.Assign) and isinstance(
                binder.value, (ast.Name, ast.Attribute)
            ):
                alias = base(binder.value)
                if alias and decorator_registers(alias, node, read, seen):
                    return True
        return False

    def registers(node, read: set[str]) -> bool:
        """A decorated definition whose decorator comes from a project module,
        or is defined in the script reading what counted code reads. Not one
        whose decorator is merely read too (``@mushin.sweep`` on the task and
        on another sweep)."""
        for deco in getattr(node, "decorator_list", ()):
            name = base(deco)
            if name is not None and decorator_registers(name, node, read, set()):
                return True
        return False

    kept = [
        n
        for n, names in candidates
        if names & roots
        or "*" in names
        or (isinstance(n, ast.ImportFrom) and n.module == "__future__")
    ]
    kept_ids = {id(n) for n in kept}
    pending = [(n, names) for n, names in candidates if names and id(n) not in kept_ids]
    read: set[str] = set()
    for n in kept:
        read |= reads(n)
    while True:
        add = [
            (n, names)
            for n, names in pending
            if names & read
            or (isinstance(n, defs) and n.decorator_list and registers(n, read))
        ]
        if not add:
            break
        added = {id(n) for n, _ in add}
        for n, _ in add:
            kept.append(n)
            read |= reads(n)
        pending = [(n, names) for n, names in pending if id(n) not in added]
    kept.sort(key=lambda node: node.lineno)

    body = []
    for n in kept:
        c = copy.deepcopy(n)
        if id(n) in in_else:
            # A fixed marker, not the guard's own test: both spellings of the
            # guard are one guard.
            # A name no source can spell.
            marker = ast.Name(id="<main else>", ctx=ast.Load())
            c = ast.If(test=marker, body=[ast.Pass()], orelse=[c])
        body.append(c)
    module = _strip_tree(
        ast.Module(body=body, type_ignores=[]),
        main_guards=False,  # the script runs as __main__: its guards run too
    )
    result = (_short_hash(_canonical_dump(module).encode()), _import_pairs(module))
    _cache_put(_SCRIPT_CACHE, cache_key, result)
    return result


#: The digest each loaded module (or driver script) had when this process
#: first recorded that load of it, to warn when its file moves: key -> (the
#: load's identity, digest). A
#: module's identity is its ``__spec__``, which a reload replaces; a script's,
#: the root functions and classes it defined, which running it again (IPython's
#: ``%run``) makes anew.
_LOAD_BASELINE: dict = {}


def _loaded_digest(name: str, digest: str) -> str:
    """``digest``, the file as it is now, after warning (once) when module
    ``name`` is loaded and its file changed since this process first recorded
    that load of it: if it was not reloaded, the old code still runs. The file
    is recorded either way, since an in-place reload (IPython's %autoreload
    patches functions without replacing ``__spec__``) does run the new code,
    and this process cannot tell the two apart."""
    import sys

    module = sys.modules.get(name)
    if module is not None:
        identity = (getattr(module, "__spec__", None) or module,)
        _warn_if_moved(name, name, identity, digest)
    return digest


def _refs(objects) -> tuple:
    """Weak references to a load's identity objects where they allow one, so
    the baseline never keeps a script's namespace (its datasets, its models)
    alive after the user dropped it."""
    import weakref

    out = []
    for obj in objects:
        try:
            out.append(weakref.ref(obj))
        except TypeError:
            out.append(lambda obj=obj: obj)
    return tuple(out)


def _warn_if_moved(key, what: str, identity: tuple, digest: str) -> None:
    """Record ``digest`` for this load (``identity``) of ``key`` the first
    time; afterwards, warn (once per load) when the digest moved within it."""
    seen = _LOAD_BASELINE.get(key)
    same = (
        seen is not None
        and len(seen[0]) == len(identity)
        and all(ref() is obj for ref, obj in zip(seen[0], identity, strict=True))
    )
    if not same:
        _LOAD_BASELINE[key] = (_refs(identity), digest, [False])
    elif seen[1] != digest and not seen[2][0]:
        seen[2][0] = True  # warned for this load; a reload warns again
        _warn_stale(what)


def _warn_stale(what: str) -> None:
    _best_effort_warning(
        f"resume/cache_dir: {what} changed on disk after this process loaded it; "
        "unless it was reloaded, the old code still runs while the cells computed "
        "now are recorded as the new code's: reload it (or restart) before "
        "resuming."
    )


def _best_effort_warning(message: str) -> None:
    import warnings

    try:
        warnings.warn(message, UserWarning, stacklevel=4)
    except Warning:  # warnings as errors: the record is best effort
        pass


def _namespace_modules(namespace: dict) -> list[str]:
    """For a module with no file (a notebook): the project modules its names
    refer to, by module values, by the modules their functions and classes
    come from, and by the classes of other values, inspected by type only.
    Its own code is not compared."""
    import types

    names = set()
    for value in list(namespace.values()):
        kind = type(value)
        try:
            if issubclass(kind, types.ModuleType):
                names.add(value.__name__)
            elif kind is types.FunctionType or issubclass(kind, type):
                module = value.__module__
                if isinstance(module, str):
                    names.add(module)
            elif isinstance(kind.__module__, str):
                names.add(kind.__module__)  # an instance: its class's module
        except Exception:  # noqa: BLE001 - a value that resists inspection
            continue
    names.discard(namespace.get("__name__"))
    return sorted(names)


# --- code defined in a notebook ----------------------------------------------
#
# A notebook has no file to compare, and its cell history is not the code in
# force (cells run out of order, fail partway, and are replayed after a
# restart). The kernel's namespace is: a function's code names the cell text
# of the version now bound, a class is found through its methods, and a plain
# constant's value is its only record. The record follows the names the
# roots' code looks up, through those objects, one entry per name. Main
# compared none of it, so each choice below compares less where a value can
# move without an edit, and a walk cut short by a bound leaves out what it
# feeds rather than recording a part chosen by the walk's order (see
# docs/superpowers/specs/2026-10-27-notebook-code-record-design.md).

#: The key the driver's notebook entries ship under in ``script_entries``: no
#: script path can be it (those end in ``.py``).
NOTEBOOK_ENTRIES = "<notebook>"

#: IPython's own names in a kernel's namespace (its input and output history
#: and helpers): they change with every cell and are never the task's code.
_IPYTHON_NAME = re.compile(
    r"_+|_i+|_i?\d+|_oh|_ih|_dh|In|Out|get_ipython|exit|quit|_exit_code"
)

#: Names whose value is a secret whatever it holds, and names that hold one
#: only as a string (``MAX_TOKENS = 512`` is a setting). Provenance redacts
#: both kinds of name from configs.
_SECRET_NAME = re.compile(r"(?i)(api[_-]?key|secret|password|passwd|credential)")
_SECRET_STRING_NAME = re.compile(r"(?i)(token|auth)")

_PLAIN_TYPES = (int, float, complex, bool, str, bytes, type(None))
_PLAIN_MAX_ITEMS = 10_000
_PLAIN_MAX_CHARS = 1_000_000
#: The functions and classes one record takes, and the container items one
#: walk of a value visits.
_NOTEBOOK_MAX_OBJECTS = 2000
_CONTAINER_BUDGET = 10_000
_CELL_TREES_MAX = 128

#: Bytecode that looks a name up in the globals, that rebinds one, that
#: stores an attribute, and that rebinds a closure's variable (``nonlocal``).
_GLOBAL_READS = frozenset({"LOAD_GLOBAL", "LOAD_NAME", "LOAD_FROM_DICT_OR_GLOBALS"})
_GLOBAL_WRITES = frozenset({"STORE_GLOBAL", "DELETE_GLOBAL"})
_ATTR_WRITES = frozenset({"STORE_ATTR", "DELETE_ATTR"})
_CELL_WRITES = frozenset({"STORE_DEREF", "DELETE_DEREF"})

_CELL_TREES: dict[str, object] = {}
_CODE_NAMES: dict = {}
_NOTEBOOK_TRUNCATION_WARNED: list = []


class _NotPlain(Exception):
    pass


def _plain_repr(value) -> str | None:
    """A plain value's canonical text, or None: exactly ``int``, ``float``,
    ``complex``, ``bool``, ``str``, ``bytes``, ``None``, and tuples and
    frozensets of them, within the bounds. Mutable values are never plain: a
    task that appends to a global list or fills a dict during one run would
    make the resume re-run every cell. The type is part of the text, so
    ``1``, ``1.0`` and ``True`` differ; a frozenset's members are sorted,
    since string hashing is randomized per process."""
    budget = [_PLAIN_MAX_ITEMS, _PLAIN_MAX_CHARS]

    def text(v, depth: int) -> str:
        kind = type(v)
        if kind in _PLAIN_TYPES:
            if kind in (str, bytes) and len(v) > budget[1]:
                raise _NotPlain
            out = f"{kind.__name__}:{v!r}"  # a huge int raises ValueError
        elif (kind is tuple or kind is frozenset) and depth < 16:
            budget[0] -= len(v)
            if budget[0] < 0:
                raise _NotPlain
            parts = [text(item, depth + 1) for item in v]
            if kind is frozenset:
                parts.sort()
            out = f"{kind.__name__}({','.join(parts)})"
        else:
            raise _NotPlain
        budget[1] -= len(out)
        if budget[1] < 0:
            raise _NotPlain
        return out

    try:
        return text(value, 0)
    except (_NotPlain, ValueError, RecursionError):
        return None


def _looks_secret(names, value) -> bool:
    """Whether a value, known by ``names`` (its own, and the name its source
    read: ``pw=DB_PASSWORD``), may be a secret: its value is then left out,
    since a digest of a short string alone, or of a known definition with it,
    can confirm a guess of it. A provider-token-shaped string counts under any
    name."""
    from ._provenance import _REDACT_VALUE_RE

    strings: list[str] = []
    stack = [value]
    while stack:
        v = stack.pop()
        kind = type(v)
        if kind is str:
            strings.append(v)
        elif kind is bytes:
            strings.append(v.decode("latin-1"))
        elif kind in (tuple, frozenset):
            stack.extend(v)
    for name in names:
        if type(name) is not str:
            continue
        if _SECRET_NAME.search(name) or (strings and _SECRET_STRING_NAME.search(name)):
            return True
    return any(_REDACT_VALUE_RE.search(s) for s in strings)


def _cell_entry(entry) -> bool:
    return type(entry) is tuple and len(entry) == 4 and type(entry[2]) is list


def _cell_lines(filename) -> list | None:
    """The text of a notebook cell, as IPython keeps it in ``linecache``: an
    entry with no modification time (a file on disk has one, and can have
    changed since the code was compiled), or, before IPython 8.6, which wrote
    one, an entry in its own ``linecache._ipython_cache``."""
    import linecache

    entry = linecache.cache.get(filename)
    if _cell_entry(entry) and entry[1] is None:
        return entry[2]
    kept = getattr(linecache, "_ipython_cache", None)
    if type(kept) is dict:
        entry = kept.get(filename)
        if _cell_entry(entry):
            return entry[2]
    return None


def _cell_tree(lines):
    """A cell's syntax tree without docstrings, cached by its text (a few,
    since a long session runs many cells); None when it does not parse."""
    import ast

    text = "".join(lines)
    key = _short_hash(text.encode())
    if key in _CELL_TREES:
        return _CELL_TREES[key]
    try:
        # A cell may await at the top level (ipykernel compiles it so).
        flags = ast.PyCF_ONLY_AST | ast.PyCF_ALLOW_TOP_LEVEL_AWAIT
        tree = _strip_tree(compile(text, "<cell>", "exec", flags), main_guards=False)
    except (SyntaxError, ValueError, RecursionError):
        tree = None
    while len(_CELL_TREES) >= _CELL_TREES_MAX:
        _CELL_TREES.pop(next(iter(_CELL_TREES)))
    _CELL_TREES[key] = tree
    return tree


def _code_names(code) -> tuple[frozenset, frozenset, frozenset, frozenset]:
    """``(reads, writes, attributes, cells)`` of a code object and the code
    nested in it: the global names it looks up, those it rebinds, the
    attribute names it stores, and the closure variables it rebinds. From
    bytecode, not ``co_names``, which also lists attribute names (``cfg.lr``
    would follow a global ``lr``). Bytecode only chooses what to follow; it is
    never hashed, since records are compared across Python versions."""
    import dis
    import types

    hit = _CODE_NAMES.get(code)
    if hit is not None:
        return hit
    found: tuple[set, set, set, set] = (set(), set(), set(), set())
    kinds = (_GLOBAL_READS, _GLOBAL_WRITES, _ATTR_WRITES, _CELL_WRITES)
    stack = [code]
    while stack:
        current = stack.pop()
        for ins in dis.get_instructions(current):
            if type(ins.argval) is not str:
                continue
            for names, ops in zip(found, kinds, strict=True):
                if ins.opname in ops:
                    names.add(ins.argval)
        stack.extend(k for k in current.co_consts if type(k) is types.CodeType)
    result = (
        frozenset(found[0]),
        frozenset(found[1]),
        frozenset(found[2]),
        frozenset(found[3]),
    )
    _cache_put(_CODE_NAMES, code, result)
    return result


def _definition_nodes(code) -> list | None:
    """The syntax nodes that define a function in its cell: the ``def`` named
    like it whose first line (or a decorator's, which is a decorated
    function's first line) is the code's, or for a lambda the lambdas on its
    line, so a lambda passed to ``run(...)`` does not bring the grid. None
    when the cell is not kept or names something else."""
    import ast

    lines = _cell_lines(code.co_filename)
    tree = _cell_tree(lines) if lines is not None else None
    if tree is None:
        return None
    line = code.co_firstlineno
    if code.co_name == "<lambda>":
        found = [
            n for n in ast.walk(tree) if type(n) is ast.Lambda and n.lineno == line
        ]
        return sorted(found, key=lambda n: n.col_offset) or None
    for node in ast.walk(tree):
        if (
            type(node) in (ast.FunctionDef, ast.AsyncFunctionDef)
            and node.name == code.co_name
            and (
                node.lineno == line
                or any(d.lineno == line for d in node.decorator_list)
            )
        ):
            return [node]
    return None


def _source_name(node) -> str | None:
    """The name an expression reads, when it is just a name or an attribute
    (``DB_PASSWORD``, ``cfg.password``): what a captured value came from."""
    import ast

    if type(node) is ast.Name:
        return node.id
    if type(node) is ast.Attribute:
        return node.attr
    return None


def _default_sources(nodes) -> dict:
    """``{parameter: name its default reads}`` from a definition's node."""
    if not nodes or len(nodes) != 1:
        return {}
    args = nodes[0].args
    positional = [*args.posonlyargs, *args.args]
    pairs = list(
        zip(
            positional[len(positional) - len(args.defaults) :],
            args.defaults,
            strict=False,
        )
    )
    pairs += [
        (a, d)
        for a, d in zip(args.kwonlyargs, args.kw_defaults, strict=False)
        if d is not None
    ]
    return {a.arg: _source_name(d) for a, d in pairs}


def _attribute_sources(node) -> dict:
    """``{attribute: name its value reads}`` for a class statement's body."""
    import ast

    out: dict = {}
    for stmt in getattr(node, "body", ()):
        if type(stmt) is ast.Assign:
            for target in stmt.targets:
                if type(target) is ast.Name:
                    out[target.id] = _source_name(stmt.value)
        elif type(stmt) is ast.AnnAssign and type(stmt.target) is ast.Name:
            out[stmt.target.id] = _source_name(stmt.value)
    return out


def _method_functions(value) -> list:
    """The functions a class attribute holds: a function, a static or class
    method's, a property's accessors. Not 3.14's ``__annotate__``, which
    holds a class body's annotations: the compiler's, not a method, and
    absent before 3.14."""
    import types

    kind = type(value)
    found: list = []
    if kind is types.FunctionType:
        found = [value]
    elif kind in (staticmethod, classmethod):
        found = [value.__func__]
    elif kind is property:
        found = [value.fget, value.fset, value.fdel]
    return [
        f
        for f in found
        if type(f) is types.FunctionType and f.__code__.co_name != "__annotate__"
    ]


def _class_node(cls, methods):
    """The ``class`` statement that defined ``cls``, found through a method
    defined in its body (``inspect.getsource`` cannot find a class in a
    notebook): the innermost one with its name spanning the method's line."""
    import ast

    for fn in methods:
        code = fn.__code__
        lines = _cell_lines(code.co_filename)
        tree = _cell_tree(lines) if lines is not None else None
        if tree is None:
            continue
        best = None
        for node in ast.walk(tree):
            if (
                type(node) is ast.ClassDef
                and node.name == cls.__name__
                and node.lineno <= code.co_firstlineno <= (node.end_lineno or 0)
                and (best is None or node.lineno >= best.lineno)
            ):
                best = node
        if best is not None:
            return best
    return None


def _cell_function(value, namespace):
    """A function defined in a cell of ``namespace``, through decorators'
    ``__wrapped__``, a library's too (``@contextmanager``,
    ``@torch.no_grad()``): None for anything else. Only a function's or an
    ``lru_cache`` wrapper's own ``__dict__`` is read, so an arbitrary object
    (a lazy proxy) is never asked for an attribute."""
    import types

    for _ in range(8):
        kind = type(value)
        if kind is types.FunctionType:
            if (
                value.__globals__ is namespace
                and _cell_lines(value.__code__.co_filename) is not None
            ):
                return value
            value = value.__dict__.get("__wrapped__")
        elif kind is getattr(functools, "_lru_cache_wrapper", None):
            value = vars(value).get("__wrapped__")
        else:
            return None
    return None


def _notebook_class(value, namespace):
    """``value`` when it is a class defined in ``namespace``, else None."""
    if not issubclass(type(value), type):
        return None
    try:
        module = vars(value).get("__module__")
    except TypeError:
        return None
    return value if module == namespace.get("__name__") else None


def _class_methods(cls, namespace) -> list:
    """The functions defined in a notebook class's body, through their
    decorators."""
    out = []
    for value in vars(cls).values():
        for fn in _method_functions(value):
            fn = _cell_function(fn, namespace)
            if fn is not None and fn.__qualname__.startswith(cls.__qualname__ + "."):
                out.append(fn)
    return out


def _container_items(value) -> list:
    return list(value.values()) if type(value) is dict else list(value)


def _notebook_entries(namespace: dict, roots) -> dict[str, str]:
    """The entries of a notebook's code the roots reach: ``{"<name>
    (notebook)": digest}`` for each name their code looks up (and each root
    by its name), through the objects bound now. A function defined in a
    cell is its definition's syntax tree plus the values it captured (its
    defaults and closure: what runs, not what their names hold now); a class
    defined there is its ``class`` statement plus its plain attributes; an
    instance of one is its class; a plain value is its text; a container is
    the notebook functions and classes it holds (a registry). A digest
    covers what its object reaches through captured values, methods and
    bases, a cycle (a method calling ``super()``) as one component.

    Left out, since each can move without an edit: a mutable value's
    contents, an object's state, data; a name some reached code rebinds, an
    attribute name it stores, a closure variable it rebinds, and every
    default when it stores ``__defaults__``; IPython's own names; what may be
    a secret. A container walk past its bound leaves out the entries it
    feeds, and a record past its object bound has no notebook entries (with
    a warning): what a cut walk finds depends on its order. Objects are
    inspected by type, and only a function's own ``__dict__`` is read, so a
    lazy proxy is never resolved."""
    import types

    # uid -> [kind, object, captured or methods, children, complete]
    units: dict[int, list] = {}
    codes: list = []
    walked: dict[int, tuple[list, bool]] = {}
    over = [False]

    def callables_in(value) -> tuple[list, bool]:
        """``(callables, complete)``: the notebook functions and classes a
        value holds, breadth first (each container at its least depth, once),
        within one item budget; ``complete`` is False past it."""
        if id(value) in walked:
            return walked[id(value)]
        found: list = []
        seen: set[int] = set()
        level, depth, budget, complete = [value], 0, _CONTAINER_BUDGET, True
        while level and complete:
            nxt = []
            for v in level:
                if type(v) in (dict, list, tuple, set, frozenset):
                    if id(v) in seen or depth > 4:
                        continue
                    seen.add(id(v))
                    budget -= len(v)
                    if budget < 0:
                        complete = False
                        break
                    nxt.extend(_container_items(v))
                else:
                    f = _cell_function(v, namespace) or _notebook_class(v, namespace)
                    if f is not None:
                        found.append(f)
            level, depth = nxt, depth + 1
        walked[id(value)] = (found, complete)
        return found, complete

    def visit(obj) -> int | None:
        """Take a function or class into ``units``; None past the bound."""
        if id(obj) in units:
            return id(obj)
        if len(units) >= _NOTEBOOK_MAX_OBJECTS:
            over[0] = True
            return None
        if type(obj) is types.FunctionType:
            code = obj.__code__
            defaults = obj.__defaults__ or ()
            positional = code.co_varnames[: code.co_argcount]
            captured = [
                ("default", name, value)
                # `__defaults__` can be assigned any length.
                for name, value in zip(
                    positional[len(positional) - len(defaults) :],
                    defaults,
                    strict=False,
                )
            ]
            captured += [
                ("kwdefault", name, value)
                for name, value in sorted((obj.__kwdefaults__ or {}).items())
            ]
            for name, cell in zip(
                code.co_freevars, obj.__closure__ or (), strict=False
            ):
                try:
                    captured.append(("cell", name, cell.cell_contents))
                except ValueError:  # an empty cell
                    captured.append(("cell", name, None))
            unit = units[id(obj)] = ["function", obj, captured, [], True]
            codes.append(code)
            values = [value for _, _, value in captured]
        else:
            methods = _class_methods(obj, namespace)
            unit = units[id(obj)] = ["class", obj, methods, [], True]
            codes.extend(fn.__code__ for fn in methods)
            values = [*methods, *obj.__bases__]
        for value in values:
            found, complete = callables_in(value)
            if not complete:
                unit[4] = False
                continue
            for f in found:
                uid = visit(f)
                if uid is not None:
                    unit[3].append(uid)
        return id(obj)

    def unit_of(value) -> int | None:
        found = _cell_function(value, namespace) or _notebook_class(value, namespace)
        if found is None and _notebook_class(type(value), namespace) is not None:
            found = type(value)  # an instance: its class runs
        return visit(found) if found is not None else None

    root_units: dict[str, list[int]] = {}
    for key, root in roots:
        uid = unit_of(root)
        if uid is not None:
            root_units.setdefault(key, []).append(uid)

    # Follow the names the reached code looks up, breadth first.
    followed: set[str] = set()
    named: dict[str, object] = {}
    done = 0
    while done < len(codes):
        code = codes[done]
        done += 1
        for name in sorted(_code_names(code)[0]):
            if name in followed or name not in namespace:
                continue
            followed.add(name)
            if (
                name.startswith("__") and name.endswith("__")
            ) or _IPYTHON_NAME.fullmatch(name):
                continue
            value = namespace[name]
            named[name] = value
            if unit_of(value) is None:
                found, complete = callables_in(value)
                for f in found if complete else ():
                    visit(f)
    if over[0]:
        if not _NOTEBOOK_TRUNCATION_WARNED:
            _NOTEBOOK_TRUNCATION_WARNED.append(True)
            _best_effort_warning(
                "resume/cache_dir: the notebook code this task reaches is larger "
                "than the record covers, so it is not compared; use a fresh "
                "working_dir after editing it."
            )
        return {}

    rebound: set[str] = set()
    stored: set[str] = set()
    cells: set[str] = set()
    for code in codes:
        _, writes, attrs, derefs = _code_names(code)
        rebound |= writes
        stored |= attrs
        cells |= derefs
    moved = {
        "default": "__defaults__" in stored,
        "kwdefault": "__kwdefaults__" in stored,
        "cell": "cell_contents" in stored,
    }

    def own(uid: int) -> str:
        """A unit's own parts, without what it reaches."""
        kind, obj, extra, _, _ = units[uid]
        parts = [kind]
        if kind == "function":
            nodes = _definition_nodes(obj.__code__)
            parts.append(
                "?" if nodes is None else "|".join(_canonical_dump(n) for n in nodes)
            )
            sources = _default_sources(nodes)
            for where, name, value in extra:
                text = None
                if not (
                    moved[where]
                    or (where == "cell" and name in cells)
                    or _looks_secret((name, sources.get(name)), value)
                ):
                    text = _plain_repr(value)
                if text is None:
                    text = f"{type(value).__module__}.{type(value).__qualname__}"
                parts.append(f"{where}:{name}={text}")
        else:
            node = _class_node(obj, extra)
            parts.append("?" if node is None else _canonical_dump(node))
            sources = _attribute_sources(node)
            for attr, value in sorted(vars(obj).items(), key=lambda kv: str(kv[0])):
                if (
                    type(attr) is not str
                    or (attr.startswith("__") and attr.endswith("__"))
                    or attr in stored
                    or _looks_secret((attr, sources.get(attr)), value)
                ):
                    continue
                text = _plain_repr(value)
                if text is not None:
                    parts.append(f"{attr}={text}")
        return _short_hash("\n".join(parts).encode())

    # Digests by strongly connected component (iterative Tarjan, which emits
    # a component after every component it reaches): a component hashes its
    # members' own parts and the digests of the components it reaches, so a
    # digest does not depend on the order the walk took, and a cycle (a
    # method calling super(), closures sharing a table) costs one pass.
    index: dict[int, int] = {}
    low: dict[int, int] = {}
    on_stack: set[int] = set()
    stack: list[int] = []
    comp_of: dict[int, int] = {}
    comp_digest: list[str] = []
    comp_complete: list[bool] = []
    owns = {uid: own(uid) for uid in units}

    def close(members: list[int]) -> None:
        ci = len(comp_digest)
        for m in members:
            comp_of[m] = ci
        reached = {comp_of[c] for m in members for c in units[m][3] if comp_of[c] != ci}
        comp_complete.append(
            all(units[m][4] for m in members) and all(comp_complete[r] for r in reached)
        )
        text = "\n".join(sorted(owns[m] for m in members))
        text += "\n--\n" + "\n".join(sorted({comp_digest[r] for r in reached}))
        comp_digest.append(_short_hash(text.encode()))

    for start in units:
        if start in index:
            continue
        index[start] = low[start] = len(index)
        stack.append(start)
        on_stack.add(start)
        work = [(start, iter(units[start][3]))]
        while work:
            node, children = work[-1]
            for child in children:
                if child not in index:
                    index[child] = low[child] = len(index)
                    stack.append(child)
                    on_stack.add(child)
                    work.append((child, iter(units[child][3])))
                    break
                if child in on_stack:
                    low[node] = min(low[node], index[child])
            else:
                work.pop()
                if work:
                    parent = work[-1][0]
                    low[parent] = min(low[parent], low[node])
                if low[node] == index[node]:
                    members = []
                    while True:
                        member = stack.pop()
                        on_stack.discard(member)
                        members.append(member)
                        if member == node:
                            break
                    close(members)

    def digest(uid: int) -> str | None:
        ci = comp_of.get(uid)  # None: an object no walk took
        if ci is None or not comp_complete[ci]:
            return None
        return _short_hash(f"{owns[uid]}\n{comp_digest[ci]}".encode())

    entries: dict[str, str] = {}
    for key, uids in root_units.items():
        found = [digest(u) for u in uids]
        if None in found:
            continue
        joined = "\n".join(sorted(set(found)))  # roots sharing a name
        entries[f"{key} (notebook)"] = (
            found[0] if len(set(found)) == 1 else _short_hash(joined.encode())
        )
    for name, value in named.items():
        key = f"{name} (notebook)"
        if key in entries or name in rebound:
            continue
        uid = unit_of(value)
        if uid is not None:
            d = digest(uid)
            if d is not None:
                entries[key] = d
            continue
        text = _plain_repr(value)
        if text is not None:
            if not _looks_secret((name,), value):
                entries[key] = _short_hash(f"value\n{text}".encode())
            continue
        found, complete = callables_in(value)
        if not complete or not found:
            continue
        # A set: a registration cell run again holds the same code twice.
        held = {digest(id(f)) for f in found}
        if None not in held:
            entries[key] = _short_hash(
                ("registry\n" + "\n".join(sorted(held))).encode()
            )
    return entries


def _notebook_record(notebooks: dict) -> dict[str, str]:
    """The notebook entries for ``notebooks`` (``{id: (namespace, [(key,
    root)])}``), best effort: a notebook whose entries cannot be taken gives
    none (as on main)."""
    out: dict[str, str] = {}
    for namespace, roots in notebooks.values():
        try:
            out.update(_notebook_entries(namespace, roots))
        except Exception:  # noqa: BLE001 - never fail the record for this
            continue
    return out


def _root_layers(root) -> list:
    """A root and what it wraps: a partial's function, a decorator's
    ``__wrapped__``, at most a few layers, each object once."""
    import inspect

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
                layer = inspect.getattr_static(layer, "__wrapped__", None)
            except Exception:  # noqa: BLE001 - an attribute that raises
                break
    return layers


def _bound_callables(value, depth: int = 0):
    """The functions, methods and classes a partial binds, in lists, tuples
    and dicts too: code the task runs, whose modules the record follows."""
    import types

    if depth > 4:
        return
    if isinstance(value, (types.FunctionType, types.MethodType, type)):
        yield value
    elif isinstance(value, functools.partial):
        yield from _bound_callables(value.func, depth + 1)
        for v in (*value.args, *(value.keywords or {}).values()):
            yield from _bound_callables(v, depth + 1)
    elif isinstance(value, (list, tuple, set, frozenset)):
        for v in value:
            yield from _bound_callables(v, depth + 1)
    elif isinstance(value, dict):
        for v in value.values():
            yield from _bound_callables(v, depth + 1)


def _code_global_names(code) -> set[str]:
    """The names a code object and the code nested in it look up."""
    import types

    names: set[str] = set()
    stack = [code]
    while stack:
        current = stack.pop()
        names.update(current.co_names)
        stack.extend(k for k in current.co_consts if isinstance(k, types.CodeType))
    return names


def _root_entries(
    root,
    code: dict[str, str],
    seeds: list[str],
    scripts: dict,
    depth: int = 0,
    notebooks: dict | None = None,
):
    """Add what a root gives: the project modules to follow to ``seeds``, and, for a root defined in a
    driver script, its top-level name and itself to ``scripts[file] =
    (namespace, names, objects)``; for one defined where there is no file (a
    notebook), itself to ``notebooks[id(namespace)] = (namespace, [(name,
    root)])``. A partial's bound functions and classes, and a bound method's
    function and class, are roots too."""
    import sys
    import types

    for layer in _root_layers(root):
        if isinstance(layer, functools.partial):
            # Its bound arguments are not compared (as on main); the code it
            # binds is followed.
            if depth < 2:
                bound = [*layer.args, *(layer.keywords or {}).values()]
                for value in _bound_callables(bound):
                    _root_entries(value, code, seeds, scripts, depth + 1, notebooks)
            continue
        if type(layer) is types.MethodType:
            if depth < 2:
                _root_entries(
                    layer.__func__, code, seeds, scripts, depth + 1, notebooks
                )
            layer = type(layer.__self__)
        if type(layer) is types.FunctionType:
            namespace = layer.__globals__
            top = layer.__qualname__.partition(".")[0]
            # A lambda at the top of a script has no name to start from: the
            # globals its code reads instead.
            tops = {top} if top.isidentifier() else _code_global_names(layer.__code__)
            identity = layer
        else:
            cls = layer if isinstance(layer, type) else type(layer)
            module = sys.modules.get(getattr(cls, "__module__", ""), None)
            namespace = getattr(module, "__dict__", None) or {}
            tops = {cls.__qualname__.partition(".")[0]}
            identity = cls
        name, file = namespace.get("__name__"), namespace.get("__file__")
        if not isinstance(name, str):
            continue
        if name == "__main__" and isinstance(file, str) and file.endswith(".py"):
            if not _foreign_by_path(file, "__main__"):  # not a launcher's module
                _, names, objects = scripts.setdefault(file, (namespace, set(), []))
                names |= tops
                objects.append(identity)
        elif not isinstance(file, str):
            seeds.extend(_namespace_modules(namespace))  # a notebook
            if notebooks is not None:
                # Its name: `task.__mushin_task__` is @mushin.sweep's copy.
                key = identity.__qualname__.removesuffix(".__mushin_task__")
                _, held = notebooks.setdefault(id(namespace), (namespace, []))
                held.append((key, identity))
        elif not _is_foreign_module(name, file):
            seeds.append(name)


def _script_entry(file: str, namespace: dict, names, objects) -> tuple[str, list]:
    """``(digest, imports)`` for a driver script defining ``names``, as the
    file is now, after warning (once) when it changed since this process first
    recorded the same load of it (the same root ``objects``: running the
    script again makes new ones)."""
    digest, imports = _script_digest(Path(file), frozenset(names))
    key = ("script", file, frozenset(names))
    _warn_if_moved(key, file, tuple(objects), digest)
    return digest, imports


def driver_script_entries(roots) -> dict:
    """The entries of the driver scripts the roots are defined in, taken in
    the driver when a run starts: ``{file: (digest, imports, package)}``. A
    parallel worker runs the driver's pickled script functions, so the
    driver's view of the script is the one that runs; the runner ships these
    to every cell, so an edit to the script while the sweep runs is not
    mistaken for the code its cells ran. Empty when they cannot be taken."""
    try:
        code: dict[str, str] = {}
        seeds: list[str] = []
        scripts: dict = {}
        notebooks: dict = {}
        for root in roots:
            if root is not None:
                _root_entries(root, code, seeds, scripts, notebooks=notebooks)
        out: dict = {}
        for file, (namespace, names, objects) in scripts.items():
            digest, imports = _script_entry(file, namespace, names, objects)
            package = namespace.get("__package__")
            out[file] = (digest, imports, package if isinstance(package, str) else None)
        if notebooks:
            # A worker gets notebook code by value, without its cells' text:
            # the driver's entries are what its cells record.
            out[NOTEBOOK_ENTRIES] = _notebook_record(notebooks)
        return out
    except Exception:  # noqa: BLE001 - best effort: each cell takes its own
        return {}


def _script_entries(
    scripts: dict, code: dict[str, str], seeds: list[str], given: dict
) -> None:
    """The entry for each driver script a root or a target is defined in (the
    driver's, from ``given``, when it took one), and the project modules its
    counted code imports."""
    for file, (namespace, names, objects) in scripts.items():
        if file in given:
            digest, imports, package = given[file]
        else:
            digest, imports = _script_entry(file, namespace, names, objects)
            package = namespace.get("__package__")
            package = package if isinstance(package, str) else None
        code[f"{Path(file).stem} (script)"] = digest
        if package:
            # `python -m pkg.run` runs `pkg/__init__.py` before the script.
            seeds.append(package)
        seeds.extend(_resolved(imports, "__main__", False, package=package))


def _target_entries(
    path: str, seeds: list[str], scripts: dict, notebooks: dict | None = None
) -> None:
    """A ``_target_`` string's class or function: its module, found without
    importing it, to follow; one in the driver script (``__main__``), the
    script, read from this process's ``__main__`` (only this process can
    instantiate it); one in a module with no file (a notebook's
    ``__main__``), the object itself, from ``notebooks``. Foreign code gives
    no entry."""
    import sys

    parts = path.split(".")
    if notebooks is not None:
        for i in range(len(parts) - 1, 0, -1):
            module = sys.modules.get(".".join(parts[:i]))
            if module is None:
                continue
            namespace = getattr(module, "__dict__", None)
            # A module with no file can be an extension's submodule, not a
            # notebook, so its object must be defined in a cell. A kernel's
            # `__main__` is the notebook: a class with no methods (a config
            # dataclass) counts there too.
            if type(namespace) is dict and not isinstance(
                namespace.get("__file__"), str
            ):
                value = namespace.get(parts[i])
                cls = _notebook_class(value, namespace)
                if _cell_function(value, namespace) is not None or (
                    cls is not None
                    and (parts[0] == "__main__" or _class_methods(cls, namespace))
                ):
                    _, held = notebooks.setdefault(id(namespace), (namespace, []))
                    held.append((parts[i], value))
                    return
            break
    if parts[0] == "__main__":
        main = sys.modules.get("__main__")
        namespace = getattr(main, "__dict__", None) or {}
        file = namespace.get("__file__")
        if (
            len(parts) > 1
            and isinstance(file, str)
            and file.endswith(".py")
            and not _foreign_by_path(file, "__main__")
        ):
            _, names, objects = scripts.setdefault(file, (namespace, set(), []))
            names.add(parts[1])
            objects.append(namespace.get(parts[1]))
        return
    # Not judged by name first: a module not imported yet would be, and a
    # checkout's `utils` would read as an installed one. _file_entries
    # decides by the file it finds.
    for i in range(len(parts) - 1, 0, -1):
        module = ".".join(parts[:i])
        if _find_file(module) is not None:
            seeds.append(module)
            return


def _config_targets(cfg) -> list[str]:
    """Every ``_target_`` string in a cell's config, at any depth: a config,
    or one already resolved by ``resolved_config``. An unresolvable config
    has none."""
    data = resolved_config(cfg)
    if data is _UNRESOLVED:
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


def _record(roots, targets=(), scripts_given=None) -> dict:
    """The record for ``roots`` (functions, partials, classes) and ``_target_``
    strings ``targets``, with the driver's script entries ``scripts_given``
    (see ``driver_script_entries``); raises when it cannot be taken."""
    import sys

    code: dict[str, str] = {}
    seeds: list[str] = []
    scripts: dict = {}
    notebooks: dict = {}
    for root in roots:
        if root is not None:
            _root_entries(root, code, seeds, scripts, notebooks=notebooks)
    for target in targets:
        _target_entries(target, seeds, scripts, notebooks)
    _script_entries(scripts, code, seeds, scripts_given or {})
    if notebooks:
        shipped = (scripts_given or {}).get(NOTEBOOK_ENTRIES)
        code.update(shipped if type(shipped) is dict else _notebook_record(notebooks))
    entries, cut = _file_entries(seeds)
    code.update(entries)
    if cut and not _TRUNCATION_WARNED:
        _TRUNCATION_WARNED.append(True)
        _best_effort_warning(
            "resume/cache_dir: the task's code imports more project files than "
            "the fingerprint covers, so edits past that point are not compared; "
            "use a fresh working_dir after a large refactor."
        )
    return {"python": f"{sys.version_info[0]}.{sys.version_info[1]}", "code": code}


def project_fingerprints(fn, *more, scripts=None) -> dict | None:
    """What the project code a task can run looks like now, as recorded in
    each cell's status sidecar: ``{"python": "3.11", "code": {key: hash}}``.

    Every project file the task's module imports, directly or through other
    project files, at any depth (imports inside functions included), and the
    ``__init__.py`` of each package those modules are in (it runs first), is
    an entry ``"<module> (file)"``, hashed as its syntax tree without its main
    block and docstrings: an edit anywhere in such a file is a change, and a
    comment, a docstring or formatting is not. A task defined in the driver
    script gives ``"<script> (script)"``, what the task's definition reads in
    the script (see ``_script_digest``), and the project files that imports.
    Files are read as they are now; one that changed since this process
    first recorded that load of it warns, since unless it was reloaded the
    old code still runs. ``more`` adds roots taken
    the same way (``pre_task``; ``Study``'s training functions), and
    ``scripts`` the driver's script entries; the config's ``_target_``\\ s are
    added per cell (``cell_project_record``). A partial task's bound
    arguments are not compared (as on main); the modules of the functions and
    classes it binds are followed.

    A root defined where there is no file (a notebook) gives ``"<name>
    (notebook)"`` for each name its code looks up there, through the objects
    bound now (see ``_notebook_entries``), and the project modules of the
    objects its namespace holds; the driver takes these when the run starts
    and ships them in ``scripts``.

    Not covered: the standard library, mushin, installed packages other than
    editable ones (a dependency upgrade is provenance's job), data files, the
    environment, a notebook's mutable values and objects' state, and code
    reached only through a string that is not a ``_target_`` of the config. None when the record cannot be taken (with a
    warning, once); a resume then does not compare, and ``cache_dir`` is not
    used."""
    try:
        return _record((fn, *more), (), scripts)
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


def cell_project_record(token: str, roots, with_body, cfg, scripts=None):
    """The project-code record of one cell, taken in the process that runs it:
    the roots' record (``roots`` and ``with_body`` alike, cached per process
    for the run ``token``) plus the entries of this cell's config
    ``_target_``\\ s, which can differ from cell to cell (a swept model class).
    ``cfg`` may be the config already resolved by ``resolved_config``.
    ``scripts`` are the driver's script entries (``driver_script_entries``).
    A target in the driver script gets its own key, ``"<script>:<name>
    (script)"``, so it never replaces the roots' entry for that script."""
    if token in _ROOT_RECORDS:
        base = _ROOT_RECORDS[token]
    else:
        all_roots = (*roots, *with_body)
        base = project_fingerprints(*all_roots, scripts=scripts) if roots else None
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
                record = _record((), [target])["code"]
                parts = target.split(".")
                entries = {}
                for k, v in record.items():
                    if k.endswith(" (script)") and len(parts) > 1:
                        k = k.replace(" (script)", f":{parts[1]} (script)")
                    # Only what the roots' record lacks, by the key as it is
                    # written: a file both reach keeps the digest the run
                    # took (a target first named after an edit would read
                    # the edited file while the old code runs), and the entry
                    # holds no second copy of the roots'.
                    if k not in base["code"]:
                        entries[k] = v
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
    computed by code it was never checked against (a module imported lazily
    in one process only). A cell with no record (written before the field)
    takes the current one: reusing it already trusted it, and the cell is
    guarded from now on."""
    if isinstance(prior, dict) and isinstance(prior.get("code"), dict):
        return prior
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
    records, compared on the keys both have (a key one run has and the other
    lacks is not an edit: a module imported lazily in one process only). File
    digests read alike on every supported Python (syntax trees, not
    bytecode), so records from two versions compare too. None when either is
    missing or malformed (a sidecar from before the field)."""
    if not isinstance(prior, dict) or not isinstance(current, dict):
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
