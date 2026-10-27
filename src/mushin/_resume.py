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
        inner function), declares ``global`` and assigns in that same scope:
        running it binds the module's name. One it only reads binds nothing."""
        found: set[str] = set()
        scopes = [node]
        while scopes:
            scope = scopes.pop()
            declared: set[str] = set()
            stored: set[str] = set()
            stack = list(scope.body)
            while stack:
                n = stack.pop()
                if isinstance(n, defs):
                    stored.add(n.name)
                    scopes.append(n)
                    continue
                if isinstance(n, ast.Lambda):
                    continue
                if isinstance(n, ast.Global):
                    declared.update(n.names)
                elif isinstance(n, ast.Name) and isinstance(
                    n.ctx, (ast.Store, ast.Del)
                ):
                    stored.add(n.id)
                elif isinstance(n, (ast.Import, ast.ImportFrom)):
                    stored.update(
                        (a.asname or a.name).partition(".")[0] for a in n.names
                    )
                stack.extend(ast.iter_child_nodes(n))
            found |= declared & stored
        return found

    def binds(node) -> set[str]:
        if isinstance(node, defs):
            return {node.name}
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
                # A `global` in its body binds the module's name, as the
                # walk over the whole block did before.
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
                # An annotation alone binds nothing (a walrus in it does).
                stack.append((sub.annotation, in_comp))
                continue
            elif isinstance(sub, (ast.Import, ast.ImportFrom)):
                names.update(binds(sub))
                continue
            elif isinstance(sub, (ast.MatchAs, ast.MatchStar)) and sub.name:
                names.add(sub.name)
            elif isinstance(sub, ast.MatchMapping) and sub.rest:
                names.add(sub.rest)
            stack.extend((child, in_comp) for child in ast.iter_child_nodes(sub))
        return names

    statements = []
    in_else: dict[int, object] = {}
    for node in tree.body:
        if _is_main_guard(node):
            # The body runs as __main__; the else runs where the script is
            # imported (a spawn child re-imports it as __mp_main__). Both
            # count, the else marked as such, so moving a statement between
            # them is an edit.
            statements += node.body + node.orelse
            in_else.update((id(stmt), node.test) for stmt in node.orelse)
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
            marker = ast.Name(id="__main_else__", ctx=ast.Load())
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
    root, code: dict[str, str], seeds: list[str], scripts: dict, depth: int = 0
):
    """Add what a root gives: the project modules to follow to ``seeds``, and, for a root defined in a
    driver script, its top-level name and itself to ``scripts[file] =
    (namespace, names, objects)``. A partial's bound functions and classes, and a bound method's
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
                    _root_entries(value, code, seeds, scripts, depth + 1)
            continue
        if type(layer) is types.MethodType:
            if depth < 2:
                _root_entries(layer.__func__, code, seeds, scripts, depth + 1)
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
        for root in roots:
            if root is not None:
                _root_entries(root, code, seeds, scripts)
        out = {}
        for file, (namespace, names, objects) in scripts.items():
            digest, imports = _script_entry(file, namespace, names, objects)
            package = namespace.get("__package__")
            out[file] = (digest, imports, package if isinstance(package, str) else None)
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


def _target_entries(path: str, seeds: list[str], scripts: dict) -> None:
    """A ``_target_`` string's class or function: its module, found without
    importing it, to follow; one in the driver script (``__main__``), the
    script, read from this process's ``__main__`` (only this process can
    instantiate it). Foreign code gives no entry."""
    import sys

    parts = path.split(".")
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
    for root in roots:
        if root is not None:
            _root_entries(root, code, seeds, scripts)
    for target in targets:
        _target_entries(target, seeds, scripts)
    _script_entries(scripts, code, seeds, scripts_given or {})
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

    Not covered: the standard library, mushin, installed packages other than
    editable ones (a dependency upgrade is provenance's job), data files, the
    environment, code defined in a notebook (the project modules of the
    objects its namespace holds are covered), and code reached only through a
    string that is not a ``_target_`` of the config. None when the record cannot be taken (with a
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
