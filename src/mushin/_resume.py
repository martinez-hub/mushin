# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
"""Primitives for kill-durable, resumable sweeps: the per-cell status sidecar,
best-effort checkpoint discovery, and the ResumeContext handed to a task that
opts in via a ``mushin_resume`` parameter (see the resume-hardening design)."""

from __future__ import annotations

import contextvars
import functools
import hashlib
import itertools
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


def _foreign_top(top: str, loaded_only: bool = False) -> bool:
    """True when the top-level module ``top`` is foreign as a whole, so none
    of its submodules needs a lookup: mushin; one whose file an installed
    package or the standard library holds (``torch``, loaded or not: the file
    decides, so a checkout's ``code/`` stays the project's); or one with no
    file that the standard library has by name (``sys``). Any other top-level
    name with no file (a namespace package, one not found) is not: each of
    its modules is judged by its own file. ``loaded_only`` judges only a
    loaded one, by its file alone (main's rule for ``_file_entries``), so a
    record of many files does no ``find_spec`` per name it has not seen
    loaded."""
    import sys

    if top == "mushin":
        return True
    if loaded_only:  # main's rule: a loaded module, by its file alone
        module = sys.modules.get(top)
        file = getattr(module, "__file__", None) if module is not None else None
        return isinstance(file, str) and bool(file) and _foreign_by_path(file, top)
    try:
        file = _find_file(top)
        if isinstance(file, str) and file:
            return _foreign_by_path(file, top)
    except Exception:  # noqa: BLE001 - undecided here: judged per module
        return False
    return top in sys.stdlib_module_names or top in sys.builtin_module_names


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
    """A syntax tree as text, leaving out the fields a source does not use, so
    the same source dumps alike on Python 3.11 to 3.15 (``ast.dump`` prints
    every field, and the fields grow across minors). Unused is ``None`` or
    ``[]`` for most added fields (3.12's ``type_params``, 3.13's
    ``default_value``), and 0 for 3.15's ``is_lazy`` on an import that is not
    ``lazy`` (PEP 810). A ``lazy import`` keeps ``is_lazy=1``, so making an
    import lazy is an edit. Leaving more out (every field at its default)
    would change the dump on every version, and with it every recorded
    digest.

    A string is written with ``ascii``, which is ``repr`` for an ASCII string.
    ``repr`` escapes what the running Python's Unicode database calls
    unprintable, and each minor ships a newer database, so a literal holding a
    character new in Unicode 17 would dump differently on 3.15 than on
    3.14."""
    import ast

    if isinstance(node, ast.AST):
        fields = [
            f"{name}={_canonical_dump(value)}"
            for name, value in ast.iter_fields(node)
            if value is not None
            and value != []
            and not (name == "is_lazy" and value == 0)
        ]
        return f"{type(node).__name__}({','.join(fields)})"
    if isinstance(node, list):
        return "[" + ",".join(_canonical_dump(item) for item in node) + "]"
    if isinstance(node, str):
        return ascii(node)
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
            # Every statement list, an except handler's and a match case's
            # too (each holds a body of its own), so a function defined in an
            # optional dependency's fallback loses its docstring like any.
            for field in ("body", "orelse", "finalbody"):
                inner = getattr(node, field, None)
                if isinstance(inner, list) and inner and isinstance(inner[0], ast.AST):
                    setattr(node, field, clean(inner))
            for part in (*getattr(node, "handlers", ()), *getattr(node, "cases", ())):
                part.body = clean(part.body)
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


def _file_entries(
    names, seen: set | None = None, files_max: int | None = None
) -> tuple[dict[str, str], bool]:
    """Entries for project modules ``names`` and, through their files, every
    project module they import in turn, at any depth (a helper imported inside
    a function, and the constants it imports), each with the packages it is
    in (their ``__init__.py`` runs first); and whether ``files_max`` files
    (``_FILES_MAX``) cut the list short. ``seen``, shared with an earlier
    call, skips the names that call followed. Each file as it is now; one
    that changed since this process first recorded that load of it warns. A
    name that cannot be resolved costs its own entry only."""

    out: dict[str, str] = {}
    pending = list(reversed(list(names)))
    seen = set() if seen is None else seen
    files_max = _FILES_MAX if files_max is None else files_max
    foreign_tops: dict[str, bool] = {}
    files = 0
    while pending and files < files_max and len(seen) < _NAMES_MAX:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        top = current.partition(".")[0]
        if top not in foreign_tops:
            # An installed package (torch) is foreign as a whole: its
            # submodules need no lookup each.
            foreign_tops[top] = _foreign_top(top, loaded_only=True)
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
    Its own code is not compared.

    On 3.15 a lazy import (PEP 810: ``lazy import m``, or any import under
    ``-X lazy_imports=all`` or a ``__lazy_modules__``) binds a proxy, and it
    stays in the namespace until a lookup of its name imports the module,
    usually when the task first runs, after this record is taken. It counts
    as the name it will import (``_file_entries`` adds the packages)."""
    import types

    names = set()
    for value in list(namespace.values()):
        kind = type(value)
        try:
            module_name = _module_name_of(value)
            if module_name is not None:
                if module_name[0] is not None:
                    names.add(module_name[0])
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


def _lazy_import_name(proxy) -> str | None:
    """The dotted name a PEP 810 lazy import will import: ``m``, or ``m.f``
    for ``lazy from m import f``, which names a module or an attribute (an
    attribute resolves to no file, and ``m``'s is recorded through it). An
    unresolved proxy refuses every attribute (AttributeError), and resolving
    it would import the module, so the name comes from the type's repr,
    ``<lazy_import 'm.f'>``. That format is
    CPython's, pinned by a test on every supported version that has lazy
    imports."""
    import re

    # Any text up to the quote: an identifier cannot hold one, and `\w` misses
    # identifier characters such as U+00B7 MIDDLE DOT and Indic vowel signs.
    match = re.fullmatch(r"<lazy_import '([^']+)'>", type(proxy).__repr__(proxy))
    return match.group(1) if match else None


# --- code defined in a notebook ----------------------------------------------
#
# A notebook has no file to compare, and its cell history is not the code in
# force (cells run out of order, fail partway, and are replayed after a
# restart). The kernel's namespace is: a function's code names the cell text
# of the version now bound, and a class is found through its methods. The
# record follows the names the roots' code looks up, through those objects,
# and compares their definitions, one entry per name. It compares no value:
# a value is only safe to compare when every writer of it is found, and in a
# notebook a writer can sit wherever a value can (four review rounds found
# them in containers, instances, bound methods and decorator closures).
# Definitions are static. See
# docs/superpowers/specs/2026-10-27-notebook-code-record-design.md.

#: The keys the driver's notebook entries, the modules the notebook code
#: names as main read them, and the modules followed after every other seed
#: (those its kept cells import, and the later readings) ship under in
#: ``script_entries``: no script path can be any of them (those end in
#: ``.py``).
NOTEBOOK_ENTRIES = "<notebook>"
NOTEBOOK_MODULES = "<notebook modules>"
NOTEBOOK_LATER_MODULES = "<notebook later modules>"

#: IPython's own names in a kernel's namespace (its input and output history
#: and helpers): they change with every cell and are never the task's code.
_IPYTHON_NAME = re.compile(
    r"_+|_i+|_i?\d+|_oh|_ih|_dh|In|Out|get_ipython|exit|quit|_exit_code"
)

#: The functions and classes one record takes.
_NOTEBOOK_MAX_OBJECTS = 2000
_CELL_TREES_MAX = 128

#: Bytecode that looks a name up in the globals, and that rebinds one.
_GLOBAL_READS = frozenset({"LOAD_GLOBAL", "LOAD_NAME", "LOAD_FROM_DICT_OR_GLOBALS"})
_GLOBAL_WRITES = frozenset({"STORE_GLOBAL", "DELETE_GLOBAL"})

_CELL_TREES: dict[str, object] = {}
_CODE_NAMES: dict = {}
#: A cell tree's scope index and its units' readings, one entry per tree
#: (the tree kept with it, so its nodes' ids stay its own), bounded like the
#: trees.
_SCOPE_INDEXES: dict = {}


def _kernel_name(name: str) -> bool:
    """A name a kernel's namespace holds that is never the task's code: a
    dunder, or one of IPython's (``_IPYTHON_NAME``)."""
    return bool(
        (name.startswith("__") and name.endswith("__")) or _IPYTHON_NAME.fullmatch(name)
    )


_NOTEBOOK_TRUNCATION_WARNED: list = []
_NOTEBOOK_FAILURE_WARNED: list = []
_CELL_SCAN_WARNED: list = []
#: The notebook's later modules cut by the file bound: per run in the driver
#: (``driver_script_entries`` clears it), once per process in a parallel
#: worker, as ``_TRUNCATION_WARNED`` is.
_LATER_TRUNCATION_WARNED: list = []


def _cell_entry(entry) -> bool:
    return type(entry) is tuple and len(entry) == 4 and type(entry[2]) is list


def _kept_cell(entry, legacy: bool) -> bool:
    """Whether a ``linecache`` entry is a notebook cell's text: one with no
    mtime (IPython 8.6 and later, ipykernel), or any entry of IPython 7's own
    ``linecache._ipython_cache`` (``legacy``), which wrote one."""
    return _cell_entry(entry) and (legacy or entry[1] is None)


def _cell_lines(filename) -> list | None:
    """The text of a notebook cell, as IPython keeps it in ``linecache``: an
    entry with no modification time (a file on disk has one, and can have
    changed since the code was compiled), or, before IPython 8.6, which wrote
    one, an entry in its own ``linecache._ipython_cache``."""
    import linecache

    entry = linecache.cache.get(filename)
    if _kept_cell(entry, legacy=False):
        return entry[2]
    kept = getattr(linecache, "_ipython_cache", None)
    if type(kept) is dict:
        entry = kept.get(filename)
        if _kept_cell(entry, legacy=True):
            return entry[2]
    return None


def _parse_cell(text: str):
    """A cell's syntax tree, or None when it does not parse. Its warnings (an
    invalid escape sequence's ``SyntaxWarning``) are not shown again: the
    kernel showed them when the cell ran. (Changing the filters resets
    Python's once-per-place memory; this module's own warnings keep their
    own.)"""
    import ast
    import warnings

    # A cell may await at the top level (ipykernel compiles it so).
    flags = ast.PyCF_ONLY_AST | ast.PyCF_ALLOW_TOP_LEVEL_AWAIT
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return compile(text, "<cell>", "exec", flags)
    except (SyntaxError, ValueError, RecursionError, MemoryError, OverflowError):
        # (3.11's parser raises MemoryError past its depth guard.) The kernel
        # could not compile such a cell either, so it never ran.
        return None


def _cell_tree(lines):
    """A cell's syntax tree without docstrings, cached by its text (a few,
    since a long session runs many cells); None when it does not parse."""
    text = "".join(lines)
    key = _short_hash(text.encode("utf-8", "surrogatepass"))
    if key in _CELL_TREES:
        return _CELL_TREES[key]
    tree = _parse_cell(text)
    if tree is not None:
        try:
            tree = _strip_tree(tree, main_guards=False)
        except (ValueError, RecursionError):  # a tree too deep to walk
            tree = None
    while len(_CELL_TREES) >= _CELL_TREES_MAX:
        evicted = _CELL_TREES.pop(next(iter(_CELL_TREES)))
        _SCOPE_INDEXES.pop(id(evicted), None)  # its scope index goes with it
    _CELL_TREES[key] = tree
    return tree


#: What each kept cell imports, by its text: names only, no trees, so a
#: session keeping many cells parses each once. It holds the cells the last
#: scan saw, so it is bounded by what ``linecache`` keeps.
_CELL_IMPORTS: dict[str, tuple] = {}


def _first_line(node) -> int:
    """A ``class`` statement's first line: its first decorator's, or its own
    (as the compiler's ``__firstlineno__`` and a body function's first line
    count it)."""
    return min([node.lineno, *(d.lineno for d in node.decorator_list)])


def _class_nodes(tree) -> list:
    """``(node, qualname, path, top)`` for each ``class`` statement in a cell's
    tree: its qualname as the compiler makes it (``A.C`` in a class,
    ``f.<locals>.C`` in a function, the bare name when the enclosing function
    or class declares it ``global``); its path from the cell's top, one
    ``(parent type, field, index)`` per step (``_exclusive``); and
    whether it is one of the cell's own statements, outside any function's
    or class's body (``top``)."""
    import ast

    defs = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)

    def declared_global(node) -> frozenset:
        names: set[str] = set()
        stack = list(node.body)
        while stack:
            sub = stack.pop()
            if type(sub) is ast.Global:
                names.update(sub.names)
            elif type(sub) not in defs and type(sub) is not ast.Lambda:
                stack.extend(ast.iter_child_nodes(sub))
        return frozenset(names)

    def children(node, prefix, globs, path, top) -> list:
        out = []
        kind = type(node).__name__
        for field, value in ast.iter_fields(node):
            items = value if type(value) is list else [value]
            for index, child in enumerate(items):
                if isinstance(child, ast.AST):
                    step = (kind, field, index)
                    out.append((child, prefix, globs, (*path, step), top))
        return out

    found: list = []
    stack = children(tree, "", frozenset(), (), True)
    while stack:
        node, prefix, globs, path, top = stack.pop()
        kind = type(node)
        if kind in defs:
            qualname = node.name if node.name in globs else prefix + node.name
            if kind is ast.ClassDef:
                found.append((node, qualname, path, top))
                inner = qualname + "."
            else:
                inner = qualname + ".<locals>."
            inside = declared_global(node)
            for i, child in enumerate(node.body):
                step = (kind.__name__, "body", i)
                stack.append((child, inner, inside, (*path, step), False))
        elif kind is not ast.Lambda:
            stack.extend(children(node, prefix, globs, path, top))
    return found


def _exclusive(one: tuple, other: tuple) -> bool:
    """Whether two statements' paths (``_class_nodes``) part where only one of
    them can run: an ``if``'s body against its ``orelse`` (an ``elif`` chain
    is nested ``orelse``s), or two cases of a ``match``. Any other parting
    (one block, a ``try``'s body and a handler, a statement after an ``if``)
    can run both. Both paths start at one root, so at their first different
    step they leave the same parent."""
    for a, b in zip(one, other, strict=False):
        if a == b:
            continue
        if a[0] == "If":
            return a[1] != b[1]
        if a[0] == "Match":
            return a[1] == b[1] == "cases" and a[2] != b[2]
        return False
    return False


def _cell_imports(
    text: str,
) -> tuple[tuple[str, ...], tuple[tuple[str, str, bool], ...], tuple[str, ...]]:
    """``(modules, bindings, classes)`` for a cell's text. ``classes``: the
    qualnames of its ``class`` statements (``_class_nodes``). ``modules``:
    what its absolute imports name, at any depth (``_import_pairs``).
    ``bindings``:
    ``(name, dotted, lazy)`` for each import among the cell's own statements
    (not a function's or a class's, whose names are local) that binds
    ``name`` to a dotted module starting with it (``import pkg.sub`` binds
    ``pkg``, and so do ``import pkg.sub as pkg`` and ``from pkg import pkg``),
    ``lazy`` when it is written ``lazy import`` (3.15): such a statement's
    proxy reads two ways (``_chain_bases``). A cell with neither the word
    ``import`` nor a line starting with ``class`` (``_CLASS_LINE``) is not
    parsed, and one that does not parse imports and defines nothing (it never
    ran)."""
    import ast

    modules: tuple[str, ...] = ()
    bindings: tuple[tuple[str, str, bool], ...] = ()
    wanted = "import" in text or _CLASS_LINE.search(text) is not None
    tree = _parse_cell(text) if wanted else None
    if tree is None:
        return modules, bindings, ()
    classes = tuple(sorted({qualname for _, qualname, _, _ in _class_nodes(tree)}))
    modules = tuple(sorted({n for n, level in _import_pairs(tree) if level == 0}))
    scopes = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)
    found: set[tuple[str, str, bool]] = set()
    stack = list(ast.iter_child_nodes(tree))
    while stack:
        node = stack.pop()
        kind = type(node)
        if kind in scopes:
            continue
        stack.extend(ast.iter_child_nodes(node))
        if kind is ast.Import:
            pairs = [(a.asname or a.name.partition(".")[0], a.name) for a in node.names]
        elif kind is ast.ImportFrom and node.level == 0 and node.module:
            pairs = [
                (a.asname or a.name, f"{node.module}.{a.name}")
                for a in node.names
                if a.name != "*"
            ]
        else:
            continue
        lazy = bool(getattr(node, "is_lazy", 0))
        found.update(
            (name, dotted, lazy)
            for name, dotted in pairs
            if dotted != name and dotted.partition(".")[0] == name
        )
    return modules, tuple(sorted(found)), classes


def _scan_kept_cells() -> list[tuple[str, tuple]]:
    """``(text, (modules, bindings, classes))`` for every cell the kernel
    keeps: a ``linecache`` entry ``_cell_lines`` would serve (``_kept_cell``),
    with what ``_cell_imports`` reads in it, cached by the text. The cache keeps what
    this scan saw: a fixed bound would evict cells the next scan needs again,
    and re-parse every one once a session passes it."""
    import linecache

    sources = [linecache.cache]
    legacy = getattr(linecache, "_ipython_cache", None)
    if type(legacy) is dict:
        sources.append(legacy)
    texts: set[str] = set()
    for source in sources:
        for entry in list(source.values()):
            if _kept_cell(entry, legacy=source is legacy):
                try:
                    texts.add("".join(entry[2]))
                except TypeError:  # a line that is not text
                    continue
    out: list[tuple[str, tuple]] = []
    seen: dict[str, tuple] = {}
    for text in texts:
        key = _short_hash(text.encode("utf-8", "surrogatepass"))
        found = _CELL_IMPORTS.get(key)
        if found is None:
            found = _cell_imports(text)
        seen[key] = found
        out.append((text, found))
    _CELL_IMPORTS.clear()
    _CELL_IMPORTS.update(seen)
    return out


def _kept_cell_reading(
    scan=None,
) -> tuple[set[str], dict[str, set[tuple[str, bool]]]]:
    """What every cell the kernel keeps imports (``scan``, or
    ``_scan_kept_cells``):
    ``(modules, bindings)``, ``bindings`` mapping a name to ``(dotted,
    lazy)`` for each import binding it to a dotted module starting with it.
    Every kept cell counts, an older version of a re-run cell too: a
    submodule's code runs only if something imported it, and a cell's import
    is there however the task then reaches the module (an argument, a call's
    result, a local name the syntax cannot follow)."""
    modules: set[str] = set()
    bindings: dict[str, set[tuple[str, bool]]] = {}
    for _, found in _scan_kept_cells() if scan is None else scan:
        modules.update(found[0])
        for name, dotted, lazy in found[1]:
            bindings.setdefault(name, set()).add((dotted, lazy))
    return modules, bindings


def _kept_class_index(scan) -> dict[str, list[str]]:
    """``{qualname: [text, ...]}`` for the ``class`` statements in the cells
    the kernel keeps (a ``_scan_kept_cells`` result), each text once per
    qualname."""
    index: dict[str, list[str]] = {}
    for text, found in scan:
        for qualname in found[2]:
            index.setdefault(qualname, []).append(text)
    return index


def _lazy_bindings(bindings: dict, namespace: dict) -> dict[str, set[str]]:
    """Of the kept cells' ``bindings`` (``_kept_cell_reading``), the dotted
    modules a statement binding each name may have bound a lazy proxy for,
    in ``namespace``: one written ``lazy import``, or any once lazy imports are
    on for every import (``-X lazy_imports=all``) or for some of the
    namespace's (``__lazy_modules__``). Empty before 3.15, and when lazy
    imports are off."""
    import sys

    get_mode = getattr(sys, "get_lazy_imports", None)
    if _LAZY_IMPORT_TYPE is None or get_mode is None:
        return {}
    mode = get_mode()
    if mode == "none":
        return {}
    every = mode == "all" or "__lazy_modules__" in namespace
    out: dict[str, set[str]] = {}
    for name, pairs in bindings.items():
        lazy = {dotted for dotted, written in pairs if written or every}
        if lazy:
            out[name] = lazy
    return out


def _code_names(code) -> tuple[frozenset, frozenset]:
    """``(reads, writes)`` of a code object and the code nested in it: the
    global names it looks up and those it rebinds. From bytecode, not
    ``co_names``, which also lists attribute names (``cfg.lr`` would follow a
    global ``lr``). Bytecode only chooses what to follow; it is never hashed,
    since records are compared across Python versions."""
    import dis
    import types

    hit = _CODE_NAMES.get(code)
    if hit is not None:
        return hit
    reads: set[str] = set()
    writes: set[str] = set()
    stack = [code]
    while stack:
        current = stack.pop()
        for ins in dis.get_instructions(current):
            if type(ins.argval) is not str:
                continue
            if ins.opname in _GLOBAL_READS:
                reads.add(ins.argval)
            elif ins.opname in _GLOBAL_WRITES:
                writes.add(ins.argval)
        stack.extend(k for k in current.co_consts if type(k) is types.CodeType)
    result = (frozenset(reads), frozenset(writes))
    _cache_put(_CODE_NAMES, code, result)
    return result


def _outer_parts(node) -> list:
    """What a definition evaluates where it stands, in the enclosing scope:
    a function's decorators, defaults and annotations, a lambda's defaults,
    a class's decorators, bases and keywords, and the bounds and defaults of
    either's type parameters (3.12+, ``def f[T: pkg.sub.Base]``)."""
    import ast

    typed = [
        part
        for param in getattr(node, "type_params", ())
        for part in (
            getattr(param, "bound", None),
            getattr(param, "default_value", None),
        )
        if part is not None
    ]
    if type(node) is ast.ClassDef:
        keywords = [k.value for k in node.keywords]
        return [*node.decorator_list, *node.bases, *keywords, *typed]
    args = node.args
    parts = [*args.defaults, *(d for d in args.kw_defaults if d is not None)]
    if type(node) is ast.Lambda:
        return parts
    params = [*args.posonlyargs, *args.args, *args.kwonlyargs, args.vararg, args.kwarg]
    parts += [p.annotation for p in params if p is not None and p.annotation]
    returns = [node.returns] if node.returns else []
    return [*node.decorator_list, *parts, *returns, *typed]


class _Scope:
    """A scope of a cell's syntax tree: the module's, a function's or a
    lambda's, a class body, or a comprehension's."""

    __slots__ = (
        "kind", "node", "own", "bound", "globals", "nonlocals", "declared",
        "rebinds", "visible",
    )  # fmt: skip

    def __init__(self, kind: str, node, visible: tuple):
        self.kind = kind  # "module", "function", "class" or "comp"
        self.node = node  # what opens it (the tree, for the module)
        self.own: list = []  # its nodes; a nested scope's body is not
        self.bound: dict[str, set] = {}  # name -> its targets
        self.globals: set[str] = set()
        self.nonlocals: set[str] = set()
        # The modules an import binds to a name declared `global` or
        # `nonlocal` here: they count where the declaration sends the name.
        self.declared: dict[str, set] = {}
        # ``(name, "global" or "nonlocal", targets)`` for each import or alias
        # that rebinds such a name here (``_index_entry`` registers them on
        # the binding they rebind).
        self.rebinds: list = []
        # The function scopes around it whose names it sees, outermost first
        # (a class body's are seen by neither its methods nor its
        # comprehensions).
        self.visible = visible


def _name_chain(node) -> tuple[str, ...] | None:
    """``("pkg", "sub")`` for the expression ``pkg.sub``, ``("pkg",)`` for
    ``pkg``; None for anything else."""
    import ast

    parts = []
    while type(node) is ast.Attribute:
        parts.append(node.attr)
        node = node.value
    if type(node) is not ast.Name:
        return None
    return (node.id, *reversed(parts))


def _resolve_chain(scope: _Scope, chain: tuple[str, ...]) -> set:
    """The targets of ``chain`` read in ``scope``, as Python resolves its first
    name: the innermost scope that binds it decides (``nonlocal`` skips the
    scope itself; ``global`` makes it the namespace's). A target is
    ``("module", path)``, ``("global", chain)`` (the namespace's, resolved
    when the record is taken) or ``("cell", scope id, chain)`` (a name bound to
    something unknown, in that scope)."""
    name, rest = chain[0], chain[1:]

    def extended(targets) -> set:
        out = set()
        for target in targets:
            if target[0] == "cell":
                out.add(("cell", target[1], target[2] + rest))
            else:
                out.add((target[0], target[1] + rest))
        return out

    # `global pkg; import pkg` binds the namespace's `pkg`: its module counts.
    extra: set = set()
    for current in (scope, *reversed(scope.visible)):
        if name in current.globals:
            extra |= extended(current.declared.get(name, ()))
            break
        if name in current.nonlocals:
            extra |= extended(current.declared.get(name, ()))
            continue
        hit = current.bound.get(name)
        if hit is not None:
            out = extended(hit) | extra
            if current.kind == "class":
                # A class body reads a name it binds with LOAD_NAME, which
                # falls back to the globals before the binding has run.
                out.add(("global", chain))
            return out
    return {("global", chain)} | extra


#: The most targets one name keeps; past it the name is unknown (as on main,
#: which read every local as unknown), so a long rebind chain stays cheap.
_TARGETS_MAX = 64


def _scope_bindings(scope: _Scope, standing: _Scope | None) -> None:
    """Fill ``scope.bound``: every name Python makes local to it, minus its
    ``global`` and ``nonlocal`` names.

    - An import binds a module.
    - An alias (``m = pkg``, ``s = pkg.sub``, a parameter defaulting to
      ``pkg``, read where the definition stands) binds what its value
      resolves to, and outside a class body its own cell too (a function
      made inside reads it through its closure).
    - Anything the syntax cannot tell (a parameter a caller passes, a plain
      assignment, a loop, ``with``, ``except`` or ``match`` name) binds its
      cell and the namespace's module of that name. A missed edit is the
      cost of guessing wrong the other way; a recompute is this one's.
    - A nested ``def`` or ``class`` name, and a relative import, bind only
      their cell: whatever it holds, it is not the namespace's module.
    - An import of a name declared ``global`` or ``nonlocal`` here adds its
      module to ``scope.declared`` (read in this scope); it and an alias of
      such a name are also ``scope.rebinds``, which ``_index_entry`` puts on
      the binding the declaration sends the name to.

    Aliases resolve once each, in dependency order. One in a cycle (``x =
    x.a``, ``a = x.l; x = a.r``) binds as unknown, and a name past
    ``_TARGETS_MAX`` targets becomes unknown, so the cost stays linear. The
    module's names are the namespace's, so it binds none."""
    import ast

    if scope.kind == "module":
        return
    for node in scope.own:
        if type(node) is ast.Global:
            scope.globals.update(node.names)
        elif type(node) is ast.Nonlocal:
            scope.nonlocals.update(node.names)
    skip = scope.globals | scope.nonlocals
    sid = id(scope)

    def bind(name: str, *targets) -> None:
        if name not in skip:
            scope.bound.setdefault(name, set()).update(targets)

    def cell(name: str) -> tuple:
        return ("cell", sid, (name,))

    def unknown(name: str) -> None:
        bind(name, cell(name), ("global", (name,)))

    def declaration(name: str) -> str:
        return "global" if name in scope.globals else "nonlocal"

    def imported(name: str, target) -> None:
        # Its cell too: a function made inside asks its closure, as on main.
        if name in skip:
            scope.declared.setdefault(name, set()).add(target)
            scope.rebinds.append((name, declaration(name), {target}))
        else:
            bind(name, target, cell(name))

    aliases: list = []  # (name, chain), read in this scope
    if scope.kind == "function":
        args = scope.node.args
        positional = [*args.posonlyargs, *args.args]
        defaults = dict(
            zip(
                [a.arg for a in positional[len(positional) - len(args.defaults) :]],
                args.defaults,
                strict=True,
            )
        )
        for arg, default in zip(args.kwonlyargs, args.kw_defaults, strict=True):
            if default is not None:
                defaults[arg.arg] = default
        for arg in (*positional, *args.kwonlyargs, args.vararg, args.kwarg):
            if arg is None:
                continue
            unknown(arg.arg)  # a caller can pass anything
            chain = _name_chain(defaults.get(arg.arg))
            if chain is not None:
                # Read where the definition stands, whose names are final.
                bind(arg.arg, *_resolve_chain(standing, chain))
    alias_names: set[int] = set()
    for node in scope.own:
        kind = type(node)
        if kind is ast.Assign or kind is ast.AnnAssign or kind is ast.NamedExpr:
            chain = _name_chain(node.value) if node.value is not None else None
            targets = node.targets if kind is ast.Assign else [node.target]
            for target in targets:
                if chain is not None and type(target) is ast.Name:
                    aliases.append((target.id, chain))
                    alias_names.add(id(target))
    for node in scope.own:
        kind = type(node)
        if kind is ast.Name:
            if type(node.ctx) is not ast.Load and id(node) not in alias_names:
                unknown(node.id)
        elif kind is ast.Import:
            for alias in node.names:
                # `import a.b` binds `a`; `import a.b as c` binds `a.b`.
                top = alias.name.partition(".")[0]
                path = tuple(alias.name.split(".")) if alias.asname else (top,)
                imported(alias.asname or top, ("module", path))
        elif kind is ast.ImportFrom:
            for alias in node.names:
                name = alias.asname or alias.name
                if node.level == 0 and node.module:
                    path = (*node.module.split("."), alias.name)
                    imported(name, ("module", path))
                else:
                    bind(name, cell(name))  # relative: the closure can tell
        elif kind in (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef):
            bind(node.name, cell(node.name))
        elif kind is ast.ExceptHandler and node.name:
            unknown(node.name)
        elif kind.__name__ in ("MatchAs", "MatchStar") and node.name:
            unknown(node.name)
        elif kind.__name__ == "MatchMapping" and node.rest:
            unknown(node.rest)
    named = {name for name, _ in aliases if name not in skip}
    for name in named:
        scope.bound.setdefault(name, set())
        if scope.kind != "class":
            bind(name, cell(name))
    # What each name holds before aliases add to it (an import, a default):
    # the cap keeps these, so it never drops a module main recorded.
    seeds = {name: set(scope.bound[name]) for name in named}
    # Each alias depends on the alias its value starts from; resolve them
    # dependencies first, and a cycle as unknown.
    depends = {name: set() for name in named}
    for name, chain in aliases:
        if name in named and chain[0] in named:
            depends[name].add(chain[0])
    comp_of = _components({name: sorted(deps) for name, deps in depends.items()})
    sizes: dict[int, int] = {}
    for comp in comp_of.values():
        sizes[comp] = sizes.get(comp, 0) + 1
    for name, chain in sorted(
        (a for a in aliases if a[0] in named), key=lambda a: comp_of[a[0]]
    ):
        if sizes[comp_of[name]] > 1 or name in depends[name]:
            unknown(name)
        else:
            resolved = _resolve_chain(scope, chain)
            bind(name, *resolved)
            if any(target[0] != "module" for target in resolved):
                # From something unknown, or from a global another function
                # may set before this runs: also as main read the name.
                unknown(name)
        if len(scope.bound[name]) > _TARGETS_MAX:
            scope.bound[name] = seeds[name] | {cell(name), ("global", (name,))}
    # An alias of a name declared `global` or `nonlocal` here (`nonlocal s;
    # s = pkg.other`) rebinds it where the declaration sends it. Resolved
    # here, in the scope it runs in; `_index_entry` registers it on that
    # binding, apart from `bound` and `declared`, so it never feeds another
    # resolution (`s = s.a` repeated cannot compound).
    for name, chain in aliases:
        if name in skip:
            scope.rebinds.append(
                (name, declaration(name), _resolve_chain(scope, chain))
            )


def _index_entry(tree) -> tuple:
    """``(tree, (scopes, opens, standing, rebound), readings)`` for a cell
    tree, built once. ``scopes`` is every scope, the module's first and each
    before the scopes in it; ``opens`` maps each definition or comprehension
    node (by id) to the scope it opens; ``standing`` maps each scope (by id)
    to the scope it stands in; ``rebound`` maps ``(scope id, name)`` to what
    the functions in the tree rebind that binding to (``nonlocal`` or
    ``global``); ``readings`` holds the units' readings
    (``_unit_reading``). Syntax only: nothing from a unit, a closure or the
    namespace is in it (a name bound to the namespace's value is a
    ``("global", ...)`` target, resolved later), so it is cached per tree,
    with the tree kept so its nodes' ids stay its own, and evicted with it."""
    import ast

    hit = _SCOPE_INDEXES.get(id(tree))
    if hit is not None and hit[0] is tree:
        return hit
    defs = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)
    comps = (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)
    module = _Scope("module", tree, ())
    scopes = [module]
    opens: dict[int, _Scope] = {}
    standing: dict[int, _Scope] = {id(module): None}
    roots = list(tree.body) if type(tree) is ast.Module else [tree]
    pending = [(module, roots)]
    while pending:
        scope, nodes = pending.pop()
        stack = list(nodes)
        while stack:
            node = stack.pop()
            kind = type(node)
            if kind in defs:
                # Its decorators, defaults, annotations and bases are
                # evaluated here; its body is a scope of its own.
                stack.extend(_outer_parts(node))
                if kind is not ast.Lambda:
                    scope.own.append(node)  # it binds its name here
                body = [node.body] if kind is ast.Lambda else list(node.body)
                child_kind = "class" if kind is ast.ClassDef else "function"
            elif kind in comps:
                # The first iterable is evaluated here; the rest in its own
                # scope.
                first, *later = node.generators
                stack.append(first.iter)
                body = [first.target, *first.ifs]
                for gen in later:
                    body += [gen.target, gen.iter, *gen.ifs]
                body += [node.key, node.value] if kind is ast.DictComp else [node.elt]
                child_kind = "comp"
            else:
                scope.own.append(node)
                stack.extend(ast.iter_child_nodes(node))
                continue
            sees = scope.visible
            if scope.kind in ("function", "comp"):
                sees = (*sees, scope)
            child = _Scope(child_kind, node, sees)
            opens[id(node)] = child
            standing[id(child)] = scope
            scopes.append(child)
            pending.append((child, body))
    for scope in scopes:  # each after the scope it stands in
        _scope_bindings(scope, standing[id(scope)])
    # A rebinding is the binding's too: `nonlocal s; s = pkg.pkg` in one
    # function changes what every function reading that `s` sees, a sibling's
    # included, so its targets go on the nearest enclosing function that binds
    # the name (the module's, for `global`). Capped per binding.
    rebound: dict[tuple[int, str], set] = {}
    for scope in scopes:
        for name, declared_as, targets in scope.rebinds:
            owner = module if declared_as == "global" else None
            for outer in () if owner is not None else reversed(scope.visible):
                if outer.kind != "function" or name in outer.nonlocals:
                    continue  # a comprehension is no target; this one passes it on
                if name in outer.bound:
                    owner = outer
                    break
            if owner is None:
                continue
            held = rebound.setdefault((id(owner), name), set())
            if len(held | targets) <= _TARGETS_MAX:
                held |= targets
    entry = (tree, (scopes, opens, standing, rebound), {})
    while len(_SCOPE_INDEXES) >= _CELL_TREES_MAX:
        _SCOPE_INDEXES.pop(next(iter(_SCOPE_INDEXES)))
    _SCOPE_INDEXES[id(tree)] = entry
    return entry


def _floor_reading(nodes) -> tuple[set[str], set[tuple[str, ...]]]:
    """The reading main took before #293, kept as the floor: whatever it names
    counts too, so the scope-aware reading (``_unit_reading``) only adds to
    what main recorded and never misses what it caught.

    ``(modules, chains)`` that definitions' syntax trees name: the modules
    their absolute imports name (as ``_import_pairs`` reads them) and each
    maximal attribute chain on a name, scope by scope with imports as the
    only bindings; a chain on a name an import binds is a module through each
    binding, a relative import's name names none, and the rest are for the
    caller to resolve (the closure first, then the namespace)."""
    import ast

    defs = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
    modules: set[str] = set()
    chains: set[tuple[str, ...]] = set()
    # (the nodes evaluated in a scope, the bindings of the functions that
    # enclose it, whether it is a class body)
    pending: list = [(list(nodes), (), False)]
    while pending:
        roots, enclosing, is_class = pending.pop()
        here: list = []
        inner_scopes: list = []
        stack = list(roots)
        while stack:
            node = stack.pop()
            kind = type(node)
            if kind in defs or kind is ast.Lambda:
                stack.extend(_outer_parts(node))
                body = [node.body] if kind is ast.Lambda else node.body
                inner_scopes.append((body, kind is ast.ClassDef))
                continue
            here.append(node)
            stack.extend(ast.iter_child_nodes(node))
        bound: dict[str, set[str]] = {}  # "" for a relative import's name
        inner = {id(n.value) for n in here if type(n) is ast.Attribute}
        found: list[tuple[str, ...]] = []
        for node in here:
            kind = type(node)
            if kind is ast.Import or kind is ast.ImportFrom:
                modules.update(n for n, level in _import_pairs(node) if level == 0)
                for alias in node.names:
                    if kind is ast.Import:
                        # `import a.b` binds `a`; `import a.b as c` binds `a.b`.
                        name = alias.asname or alias.name.partition(".")[0]
                        target = alias.name if alias.asname else name
                    else:
                        name = alias.asname or alias.name
                        absolute = node.level == 0 and node.module
                        target = f"{node.module}.{alias.name}" if absolute else ""
                    bound.setdefault(name, set()).add(target)
            elif kind is ast.Attribute and id(node) not in inner:
                parts = []
                current = node
                while type(current) is ast.Attribute:
                    parts.append(current.attr)
                    current = current.value
                if type(current) is ast.Name:
                    found.append((current.id, *reversed(parts)))
        for chain in found:
            targets = bound.get(chain[0])
            for scope in reversed(enclosing):
                if targets is not None:
                    break
                targets = scope.get(chain[0])
            if targets is None:
                chains.add(chain)
            else:
                modules.update(".".join((t, *chain[1:])) for t in targets if t)
        # A function's bindings reach the functions in it; a class body's do
        # not.
        visible = enclosing if is_class else (*enclosing, bound)
        for body, body_is_class in inner_scopes:
            pending.append((body, visible, body_is_class))
    return modules, chains


def _standing_nodes(root) -> list:
    """The nodes of a definition's outer parts, which run where it stands,
    as ``_index_entry`` puts them in that scope: a definition or
    comprehension among them adds its own outer parts (or first iterable),
    not its body."""
    import ast

    defs = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)
    comps = (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)
    out: list = []
    stack = list(_outer_parts(root))
    while stack:
        node = stack.pop()
        kind = type(node)
        if kind in defs:
            stack.extend(_outer_parts(node))
        elif kind in comps:
            stack.append(node.generators[0].iter)
        else:
            out.append(node)
            stack.extend(ast.iter_child_nodes(node))
    return out


def _unit_reading(tree, roots) -> tuple[frozenset, frozenset, frozenset, frozenset]:
    """``(modules, global chains, cell chains, rebound)`` that a unit's definitions
    (``roots``, nodes of ``tree``) name: the modules their absolute imports
    name (as ``_import_pairs`` reads them) and the attribute chains they read
    (``pkg.sub.f``) through a binding to a module; the chains on a name the
    namespace binds (``("pkg", "sub", "f")``), for the caller to resolve; and
    the chains on a name a function around the unit binds to something
    unknown, which the unit's closure holds. A chain on a name the unit binds
    to something unknown (a parameter, a loop variable) reads the
    namespace's module of that name (``_scope_bindings``). The definitions'
    outer parts are read where they stand. ``rebound`` holds what a chain
    reaches through a binding some function in the tree rebinds
    (``nonlocal``/``global``, see ``_index_entry``), as ``("module", path)``
    or ``("global", chain)`` targets, apart: main's reading did not have
    them. Each unit is read alone, so what one names does not depend on which
    were read before it; the reading is cached per (tree, definitions)."""
    import ast

    _, (scopes, opens, standing, rebinds), readings = _index_entry(tree)
    key = tuple(id(root) for root in roots)
    if key in readings:
        return readings[key]
    within = [id(n) for root in roots for n in ast.walk(root)]
    inside = [opens[i] for i in within if i in opens]
    if scopes[0].node in roots:  # the whole module (a test reads one so)
        inside.append(scopes[0])
    inside_ids = {id(scope) for scope in inside}
    reads = [(scope, scope.own) for scope in inside]
    for root in roots:
        if id(root) in opens:  # its outer parts, where it stands
            reads.append((standing[id(opens[id(root)])], _standing_nodes(root)))
    modules: set[str] = set()
    global_chains: set[tuple[str, ...]] = set()
    cell_chains: set[tuple[str, ...]] = set()
    rebound: set[tuple] = set()
    module_id = id(scopes[0])
    for scope, nodes in reads:
        inner = {id(n.value) for n in nodes if type(n) is ast.Attribute}
        for node in nodes:
            kind = type(node)
            if kind is ast.Import or kind is ast.ImportFrom:
                modules.update(n for n, level in _import_pairs(node) if level == 0)
                continue
            if kind is not ast.Attribute or id(node) in inner:
                continue
            chain = _name_chain(node)
            if chain is None:
                continue
            for target in _resolve_chain(scope, chain):
                if target[0] == "module":
                    modules.add(".".join(target[1]))
                    continue
                if target[0] == "global":
                    global_chains.add(target[1])
                    where, path = module_id, target[1]
                else:
                    if target[1] not in inside_ids:
                        cell_chains.add(target[2])
                    where, path = target[1], target[2]
                # Once per chain: a rebinding's own targets are not looked up
                # again (`x = x.left` under `global x` starts with `x` too).
                for extra in rebinds.get((where, path[0]), ()):
                    if extra[0] == "module":
                        rebound.add(("module", extra[1] + path[1:]))
                    elif extra[0] == "global":
                        rebound.add(("global", extra[1] + path[1:]))
    result = (
        frozenset(modules),
        frozenset(global_chains),
        frozenset(cell_chains),
        frozenset(rebound),
    )
    readings[key] = result
    return result


#: The types a namespace value is checked against, looked up once (``types``
#: is loaded with the interpreter). LazyImportType is new in 3.15.
_MODULE_TYPE = type(functools)
_LAZY_IMPORT_TYPE = getattr(__import__("types"), "LazyImportType", None)


def _module_name_of(value) -> tuple[str | None, bool] | None:
    """For a module, or on 3.15 a lazy import (PEP 810) of one, ``(its name,
    whether it is lazy)``; the name is None when none can be read. None for
    any other value. A module's ``__name__`` is read; a lazy proxy is not
    asked for anything (``_lazy_import_name``). Raises what a module's
    attribute hook raises."""
    kind = type(value)
    if kind is _LAZY_IMPORT_TYPE:
        return _lazy_import_name(value), True
    if issubclass(kind, _MODULE_TYPE):
        name = value.__name__
        return (name if isinstance(name, str) else None), False
    return None


def _chain_bases(name: str, value) -> tuple[str, ...]:
    """The modules an attribute chain on the global ``name`` can start from,
    when the namespace binds it to ``value``: a module's name. On 3.15 a lazy
    proxy names ``pkg.sub`` for ``lazy import pkg.sub``, which binds ``pkg``
    to the package, and alike for ``lazy import pkg.sub as s`` and ``lazy
    from pkg import sub``, which bind the submodule. Bound as its first
    component, it stands for either the package or the full name (``lazy
    from pkg import pkg``), so both are bases: a reading that names no file
    costs a lookup, and one that is missed is a missed edit. Otherwise the
    full name. None of it for any other value."""
    try:
        found = _module_name_of(value)
    except Exception:  # noqa: BLE001 - a value that resists inspection
        return ()
    if found is None or found[0] is None:
        return ()
    full, lazy = found
    if not lazy:
        return (full,)
    top = full.partition(".")[0]
    if name != top:
        return (full,)
    return (top,) if full == top else (top, full)


def _definition_site(code) -> tuple | None:
    """``(tree, nodes)``: the cell's syntax tree and the nodes in it that
    define a function: the ``def`` named like it whose first line (or a
    decorator's, which is a decorated function's first line) is the code's,
    or for a lambda the lambdas on its line, so a lambda passed to
    ``run(...)`` does not bring the grid. None when the cell is not kept or
    names something else. One lookup, so the nodes are the tree's own."""
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
        return (tree, sorted(found, key=lambda n: n.col_offset)) if found else None
    for node in ast.walk(tree):
        if (
            type(node) in (ast.FunctionDef, ast.AsyncFunctionDef)
            and node.name == code.co_name
            and (
                node.lineno == line
                or any(d.lineno == line for d in node.decorator_list)
            )
        ):
            return tree, [node]
    return None


def _method_functions(value) -> list:
    """The functions a class attribute holds: a function, a static or class
    method's, a property's accessors, a ``functools.cached_property``'s, and
    through an ``lru_cache`` (``functools.cache``) wrapper. Not 3.14's
    ``__annotate__``, which holds a class body's annotations: the
    compiler's, not a method, and absent before 3.14."""
    import types

    lru = getattr(functools, "_lru_cache_wrapper", None)
    kind = type(value)
    found: list = []
    if kind is types.FunctionType or kind is lru:
        found = [value]
    elif kind is staticmethod or kind is classmethod:  # identity: see _notebook_writes
        found = [value.__func__]
    elif kind is property:
        found = [value.fget, value.fset, value.fdel]
    elif kind is functools.cached_property:
        found = [value.func]
    out = []
    for f in found:
        if type(f) is lru:
            f = vars(f).get("__wrapped__")
        if type(f) is types.FunctionType and f.__code__.co_name != "__annotate__":
            out.append(f)
    return out


#: The attribute of an IPython shell holding mushin's cell watch (a dict,
#: so a reloaded ``_resume`` finds the same one): ``pins`` maps ``id(cls)``
#: to ``(weakref, lines)``, the transformed text of the cell that bound it;
#: ``running`` is the cells running now (a magic's nested cell too), each
#: ``(info, (raw text, {id(class): class}))`` with every class the record
#: could reach when the cell started (``_cell_classes``), or
#: ``(info, None)`` when the cell's text is unknown.
_CELL_WATCH = "_mushin_cell_watch"
#: A ``class`` keyword at a line's start, after its indentation: where every
#: ``class`` statement of a kept cell's text starts (it is Python, magics
#: already transformed, and ``compile`` takes ``\r`` as a line end too).
#: ``_cell_imports`` parses only a text with one, or with the word ``import``.
_CLASS_LINE = re.compile(r"(?:^|[\r\n])[ \t\f]*class\b")


def _cell_classes(namespace: dict) -> dict:
    """``{id(cls): cls}`` for every notebook class the record can reach from
    the namespace's names, following its walk (``_notebook_entries``): a
    value as ``_notebook_object`` resolves it (a class or its metaclass, an
    instance's class, a bound method's owner or its class, a cell function
    through its wrappers, ``_cell_function``); a cell function's
    ``__wrapped__``, again;
    a class's notebook bases and its methods (``_method_functions``), with
    their ``__wrapped__``; and the classes a class's body made
    (``Outer.Inner``). Names are read as ``str``s, values by type, functions
    through their own ``__dict__`` and classes through the type's own slots,
    so no notebook hook runs; the items are copied once, as a callback
    thread can add names."""
    import types

    notebook = namespace.get("__name__")
    found: dict = {}
    checked: set = set()
    functions: set = set()
    pending: list = []
    calls: list = []

    def take(value) -> None:
        if id(value) in checked or not issubclass(type(value), type):
            return
        checked.add(id(value))
        module = type.__dict__["__dict__"].__get__(value).get("__module__")
        # Compared through `str`'s own slot, as a `str` subclass can be one.
        if issubclass(type(module), str) and str.__eq__(module, notebook) is True:
            found[id(value)] = value
            pending.append(value)

    def take_function(value) -> bool:
        """Queue the cell function ``value`` stands for; False if none."""
        function = _cell_function(value, namespace)
        if function is None:
            return False
        if id(function) not in functions:  # a wrapper can wrap itself
            functions.add(id(function))
            calls.append(function)
        return True

    def follow(value) -> None:
        if type(value) is types.MethodType:
            owner = value.__self__
            take(owner)
            take(type(owner))
        elif issubclass(type(value), type):
            take(value)
            take(type(value))  # a foreign class's notebook metaclass
        elif not take_function(value):
            take(type(value))

    for name, value in list(namespace.items()):
        # A `str` subclass as a key too, read through `str`'s own slot; not
        # a kernel name (`_`, `_N`, a dunder), which the record never follows.
        if issubclass(type(name), str) and not _kernel_name(str.__str__(name)):
            follow(value)
    while pending or calls:
        if calls:
            # A notebook decorator's wrapper: the record follows what it wraps.
            follow(calls.pop().__dict__.get("__wrapped__"))
            continue
        cls = pending.pop()
        for base in type.__dict__["__bases__"].__get__(cls):
            take(base)
        prefix = type.__dict__["__qualname__"].__get__(cls) + "."
        for value in list(type.__dict__["__dict__"].__get__(cls).values()):
            if issubclass(type(value), type):
                inner = type.__dict__["__qualname__"].__get__(value)
                if inner.startswith(prefix):
                    take(value)
            else:
                for method in _method_functions(value):
                    take_function(method)
    return found


def _live_pin(state: dict, cls):
    """The lines ``cls`` is pinned to in the watch ``state``, or None."""
    pin = state["pins"].get(id(cls))
    return pin[1] if pin is not None and pin[0]() is cls else None


def _shell():
    """The running IPython shell, or None; IPython is never imported here."""
    import sys

    ipython = sys.modules.get("IPython")
    get = getattr(ipython, "get_ipython", None) if ipython is not None else None
    return get() if callable(get) else None


def _cell_watch(register: bool = True) -> dict | None:
    """The shell's cell watch (``_CELL_WATCH``), registering its two event
    callbacks the first time (``register``); None outside a shell. Best
    effort: an IPython that refuses the callbacks leaves a watch without
    them, which pins nothing, and no later call registers again."""
    try:
        shell = _shell()
        if shell is None:
            return None
        state = getattr(shell, _CELL_WATCH, None)
        if type(state) is dict:
            return state
        if not register:
            return None
        # IPython's static transforms only (magics, `!`, `?`): its full
        # `transform_cell` runs the prefilter on a one-line cell, which reads
        # the namespace (`isinstance`), and after a cell that can run a
        # notebook object's hook in the callback.
        manager = getattr(shell, "input_transformer_manager", None)
        transform = getattr(manager, "transform_cell", None) or shell.transform_cell
        state = {"pins": {}, "running": [], "transform": transform}

        def pre(info=None) -> None:
            try:
                running = state["running"]
                if len(running) > 8:  # cells whose end never fired
                    running.clear()
                entry = None
                try:
                    raw = getattr(info, "raw_cell", None)
                    if type(raw) is str:
                        # Every class the record could reach, whatever the
                        # text says: a cell's text cannot tell a statement
                        # from a mention, and `exec`, `%run` or a factory
                        # makes one with no `class` in it. Held by identity,
                        # so no id is reused while the cell runs.
                        entry = (raw, _cell_classes(shell.user_ns))
                except Exception:  # noqa: BLE001 - the cell still ends at post
                    entry = None
                running.append((info, entry))
            except Exception:  # noqa: BLE001 - a callback must never raise
                pass

        def post(result=None) -> None:
            # IPython fires post without pre for an empty cell (a magic's
            # empty body): end the cell whose info this result carries. An
            # interrupted cell's result can carry a new info around the same
            # raw text, and ipykernel can pass no result at all: then the
            # raw text, or the newest cell.
            try:
                running = state["running"]
                if not running:
                    return
                info = getattr(result, "info", None)
                raw = getattr(info, "raw_cell", None)
                found = None
                for i in range(len(running) - 1, -1, -1):
                    if running[i][0] is info:
                        found = i
                        break
                if found is None and raw is not None:
                    for i in range(len(running) - 1, -1, -1):
                        if getattr(running[i][0], "raw_cell", None) is raw:
                            found = i
                            break
                if found is None and result is None:
                    found = len(running) - 1
                if found is not None:
                    entry = running.pop(found)[1]
                    if entry is not None:
                        _pin_classes(state, shell.user_ns, entry)
            except Exception:  # noqa: BLE001 - a callback must never raise
                pass

        # The state first, so a failure below cannot leave callbacks that
        # the next call would register again.
        setattr(shell, _CELL_WATCH, state)
        added = []
        try:
            for event, callback in (("pre_run_cell", pre), ("post_run_cell", post)):
                shell.events.register(event, callback)
                added.append((event, callback))
        except Exception:
            for event, callback in added:
                try:
                    shell.events.unregister(event, callback)
                except Exception:  # noqa: BLE001 - best effort
                    pass
            raise
        return state
    except Exception:  # noqa: BLE001 - no watch: classes go to the kept cells
        return None


def _pin_classes(state: dict, namespace: dict, entry, reach=None) -> None:
    """Pin each class the cell ``entry`` (``(raw, classes)``) made to the
    cell's transformed text: every class the record can reach now
    (``_cell_classes``) that it could not when the cell started, unless its
    qualname is a function's local one (``make.<locals>.C``, read from the
    function's cell). By the object, whatever holds it: a name, an alias
    (``saved = Config; del Config``), an instance (``Config = Config()``), a
    derived class. The first pin wins: a magic's nested cell (``%%capture``)
    ends before the magic's own cell, and only its text holds the statement.
    Nothing is parsed here; a record reads the text (``_class_sites``), and a
    class whose text holds no statement of it (``%%time``, ``exec``, a
    factory's call, a copy) is ambiguous. ``reach`` is ``_cell_classes``'s,
    when the caller has it."""
    import weakref

    raw, before = entry
    pins = state["pins"]
    if reach is None:
        reach = _cell_classes(namespace)
    made = [
        cls
        for key, cls in reach.items()
        if key not in before
        and "<locals>" not in type.__dict__["__qualname__"].__get__(cls)
        and _live_pin(state, cls) is None
    ]
    if not made:
        return
    lines = state["transform"](raw).splitlines(True)
    if len(pins) > 4096:  # drop the pins of classes that are gone
        for gone in [k for k, (ref, _) in pins.items() if ref() is None]:
            del pins[gone]
    for cls in made:
        pins[id(cls)] = (weakref.ref(cls), lines)


def _pinned_cell(cls, namespace: dict) -> tuple[bool, list | None]:
    """``(pinned, lines)``: whether the shell's cell watch saw the cell that
    bound ``cls`` run, and that cell's transformed text (one may hold no
    statement of the class: ``%%time``, ``exec``, a factory's call; see
    ``_class_sites``). Pinned when the cell ended, or, for a record taken
    inside running cells, now: the newest running cell whose start could not
    reach the class has what it made so far pinned (``_pin_classes``). A cell
    started after the class was made (a nested ``run_cell``) could, and is
    passed over; once a cell that could not is found, the cells passed over
    have their own classes pinned first, newest first. One whose text is
    unknown may have made it, so the walk stops there."""
    state = _cell_watch(register=False)
    if state is None:
        return False, None
    lines = _live_pin(state, cls)
    if lines is None:
        passed: list = []
        for _, entry in reversed(state["running"]):
            if entry is None:
                break  # its text is unknown: no pin to the cell around it
            if id(cls) in entry[1]:
                passed.append(entry)  # started after the class was made
                continue
            # The cells passed over claim their own classes first, newest
            # first, so this one's pin takes only what it made itself.
            reach = _cell_classes(namespace)
            for inner in passed:
                _pin_classes(state, namespace, inner, reach)
            _pin_classes(state, namespace, entry, reach)
            break
        lines = _live_pin(state, cls)
    return lines is not None, lines


#: A per-process token and a counter per class object, for a class whose
#: statement cannot be told from several kept texts, or (3.11/3.12) from
#: two of its cell's own statements that both run (``_ambiguous_own``).
_PROCESS_TOKEN: list = []
_CLASS_GENERATIONS: dict = {}
_GENERATION_COUNTER = itertools.count()


def _ambiguous_own(cls) -> str:
    """A class's own digest when several kept texts could be its statement:
    unlike any digest an earlier process or object wrote, and the same while
    this object is the one bound (a recompute, never a wrong reuse)."""
    import os
    import uuid
    import weakref

    if not _PROCESS_TOKEN or _PROCESS_TOKEN[0][0] != os.getpid():
        # A forked child inherits the list: a new one per process id.
        _PROCESS_TOKEN[:] = [(os.getpid(), uuid.uuid4().hex)]
    hit = _CLASS_GENERATIONS.get(id(cls))
    if hit is None or hit[0]() is not cls:
        if len(_CLASS_GENERATIONS) > 4096:
            for key in [k for k, (r, _) in _CLASS_GENERATIONS.items() if r() is None]:
                del _CLASS_GENERATIONS[key]
        hit = (weakref.ref(cls), next(_GENERATION_COUNTER))
        _CLASS_GENERATIONS[id(cls)] = hit
    return _short_hash(f"class\n?\n{_PROCESS_TOKEN[0][1]}\n{hit[1]}".encode())


def _class_sites(cls, methods, namespace: dict, class_texts) -> tuple[list, bool, bool]:
    """``(sites, ambiguous, later)``: the ``class`` statement that defined
    ``cls``, as ``[(tree, nodes)]`` (``inspect.getsource`` cannot find a class
    in a notebook), from the first of these that answers:
    - a method, whose code names the cell: the innermost statement of its
      name spanning the method's first line;
    - 3.14's ``__annotate_func__`` (a class with annotations), the same way;
    - the cell the shell's watch saw bind it (``_pinned_cell``). When that
      cell states no such class, code it ran made it, from text that is not
      kept (``%%time``, an ``exec`` string, ``%run``) or from a kept
      statement (a ``global`` factory, a copy such as
      ``dataclass(slots=True)(C)``, ``exec(In[n])``), which cannot be told:
      ambiguous;
    - the kept cells' texts with a statement of its qualname
      (``class_texts(qualname)``, None when the cells could not be scanned).
    In a text, every statement of the qualname counts (an ``if``'s two
    branches: which one runs is a value), but on 3.13+ only those at the
    class's ``__firstlineno__`` (the first decorator's line), when it has
    one: assigning a class's ``__module__`` deletes it. Two different
    ones among the cell's own statements that can both run (``_exclusive``),
    which 3.11 and 3.12 cannot tell apart, are ambiguous: a record can stand
    between them. Inside a function's or a class's body, they run as one step,
    and which object comes out is a value. Several kept texts with different
    statements are ambiguous (which is in force cannot be told) and all of
    them are read.
    ``later``: found otherwise than through a method, so main's record had
    no reading of it (``_notebook_entries`` follows it after main's seeds).
    Kept statements of the qualname, none at the first line, are ambiguous.
    No site when none is kept, or (3.13+) for a class no ``class`` statement
    made (``type()``, ``make_dataclass``: no ``__static_attributes__``): its
    digest never changes, so it is not compared."""
    import ast
    import sys
    import types

    name = type.__dict__["__name__"].__get__(cls)
    qualname = type.__dict__["__qualname__"].__get__(cls)
    own_dict = type.__dict__["__dict__"].__get__(cls)
    annotate = own_dict.get("__annotate_func__")
    body = [(fn, False) for fn in methods]
    if type(annotate) is types.FunctionType:
        body.append((annotate, True))
    for fn, later in body:
        code = fn.__code__
        lines = _cell_lines(code.co_filename)
        tree = _cell_tree(lines) if lines is not None else None
        if tree is None:
            continue
        best = None
        for node in ast.walk(tree):
            if (
                type(node) is ast.ClassDef
                and node.name == name
                and _first_line(node) <= code.co_firstlineno <= (node.end_lineno or 0)
                and (best is None or node.lineno >= best.lineno)
            ):
                best = node
        if best is not None:
            return [(tree, [best])], False, later
    # A class body's compile sets both names on 3.13+. Assigning `__module__`
    # deletes `__firstlineno__` (a line in another module means nothing) but
    # keeps `__static_attributes__`, so only the latter tells whether a
    # `class` statement made the class; without a first line, the statements
    # are read as on 3.11/3.12.
    if sys.version_info >= (3, 13):
        if type(own_dict.get("__static_attributes__")) is not tuple:
            return [], False, True  # no `class` statement made it
    first = own_dict.get("__firstlineno__")
    if type(first) is not int:
        first = None
    dumps: dict[int, str] = {}

    def dump(node) -> str:
        if id(node) not in dumps:
            dumps[id(node)] = _canonical_dump(node)
        return dumps[id(node)]

    def statements(tree) -> tuple[list, bool]:
        """The text's statements of the qualname (on 3.13+, at its first
        line), sorted; and whether two different ones of the cell's own can
        both run, so either could be in force at a record taken between."""
        found = [
            (node, path, top)
            for node, q, path, top in _class_nodes(tree)
            if q == qualname and (first is None or _first_line(node) == first)
        ]
        both = any(
            dump(a) != dump(b) and not _exclusive(pa, pb)
            for i, (a, pa, ta) in enumerate(found)
            for b, pb, tb in found[i + 1 :]
            if ta and tb
        )
        return sorted((node for node, _, _ in found), key=dump), both

    pinned, lines = _pinned_cell(cls, namespace)
    if pinned:
        tree = _cell_tree(lines)
        nodes, siblings = statements(tree) if tree is not None else ([], False)
        if nodes:
            return [(tree, nodes)], siblings, True
        # The cell states no such class, so code it ran made it, and whether
        # from a kept statement cannot be told. Reading one would miss a
        # `%%time` cell's edits; reading none, a copy's. Ambiguous re-runs.
        return [], True, True
    texts = class_texts(qualname)
    if texts is None:
        return [], True, True
    groups: dict[str, list] = {}
    ambiguous = False
    for text in texts:
        tree = _cell_tree(text.splitlines(True))
        nodes, siblings = statements(tree) if tree is not None else ([], False)
        if nodes:
            ambiguous |= siblings
            groups.setdefault("|".join(map(dump, nodes)), []).append((tree, nodes))
    if texts and not groups:
        # Kept statements of the qualname, none at the class's first line
        # (3.13+): IPython keeps a cell as `splitlines()` lines, so a form
        # feed adds a line `compile` never counted. Which made it cannot be
        # told: ambiguous, not a constant digest that misses its edits.
        return [], True, True
    sites = [site for key in sorted(groups) for site in groups[key]]
    return sites, ambiguous or len(groups) > 1, True


def _cell_function(value, namespace):
    """A function defined in a cell of ``namespace``, through decorators'
    ``__wrapped__``, a library's too (``@contextmanager``,
    ``@torch.no_grad()``), and through a ``functools.partial``'s function
    (its bound arguments are not compared, as for a partial task): None for
    anything else. Only a function's or an ``lru_cache`` wrapper's own
    ``__dict__`` is read, so an arbitrary object (a lazy proxy) is never
    asked for an attribute."""
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
        elif kind is functools.partial:
            value = value.func
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
    decorators: direct members only, so a function a method defines and
    stores on the class as it runs (``Model.f.<locals>.g``) is not one."""
    out = []
    for value in vars(cls).values():
        for fn in _method_functions(value):
            fn = _cell_function(fn, namespace)
            if (
                fn is not None
                and fn.__qualname__.rpartition(".")[0] == cls.__qualname__
            ):
                out.append(fn)
    return out


#: How far the write scan looks through containers, closures and defaults,
#: how large a container it walks, and how many objects it takes in all.
_WRITE_SCAN_DEPTH = 3
_WRITE_SCAN_CONTAINER = 256
_WRITE_SCAN_MAX = 20_000


def _notebook_writes(namespace: dict, roots) -> frozenset | None:
    """The global names any of a notebook's functions rebinds (``global
    helper; helper = fast``), or None past the scan's bound. Every function
    the namespace holds counts, and every one it can reach without running
    anything: through decorators, an ``lru_cache`` wrapper, a ``partial``
    (its arguments too), a bound method, a class's members (its methods,
    nested classes and class attributes), an instance's class, a closure's
    cells and a function's defaults, a class's bases, and lists, tuples,
    sets, deques and dicts (``defaultdict`` and ``OrderedDict`` too) of at most
    ``_WRITE_SCAN_CONTAINER`` items (``@HOOKS.append`` keeps a writer nowhere
    else). Only those objects count toward the bound, not the data beside
    them. A name rebound through ``globals()`` or ``exec`` is not seen. From
    bytecode, so a function whose text the kernel did not keep counts too.
    Taken from the whole namespace before the walk, not from what the walk
    reaches, since a name the run can rebind must not shape the record (see
    ``_notebook_entries``). Breadth first, each object at its least depth,
    and a container walked by its size alone, so the result does not depend
    on iteration order; it depends only on the notebook's definitions and
    the shape of what it holds."""
    import collections
    import types

    lru = getattr(functools, "_lru_cache_wrapper", None)
    # Exact types only, so no subclass's __iter__ or __len__ runs, and a
    # mapping read through dict's own iterator: an OrderedDict's looks each
    # key up again, which runs the key's __hash__ and raises KeyError once a
    # key's hash has changed since it went in.
    mappings = (dict, collections.defaultdict, collections.OrderedDict)
    containers = (list, tuple, set, frozenset, collections.deque, *mappings)
    callable_kinds = (types.FunctionType, lru, functools.partial, types.MethodType)
    # Types are matched by identity: `kind in containers` compares with ==,
    # which runs a notebook metaclass's __eq__, and one that returns
    # something true would have the scan iterate the notebook's own object.
    mapping_ids = frozenset(map(id, mappings))
    container_ids = frozenset(map(id, containers))
    callable_ids = frozenset(id(k) for k in callable_kinds if k is not None)
    functions: list = []
    seen: set[int] = set()
    # Whether a type's values can lead to a function, by the type's id: a
    # notebook holding thousands of numbers asks once per type, not per
    # number. The type is kept alongside, so its id is not reused.
    leads: dict[int, tuple[type, bool]] = {}
    level = [
        value
        for name, value in list(namespace.items())
        if type(name) is str and not _kernel_name(name)
    ] + [root for _, root in roots]
    depth = 0
    while level and depth <= _WRITE_SCAN_DEPTH:
        nxt: list = []
        for value in level:
            kind = type(value)
            # Only what can lead to a function counts toward the bound: a
            # notebook's data (lists of numbers) must not turn the record off.
            if id(kind) not in leads:
                leads[id(kind)] = (
                    kind,
                    id(kind) in callable_ids
                    or id(kind) in container_ids
                    or issubclass(kind, type)
                    or _notebook_class(kind, namespace) is not None,
                )
            if not leads[id(kind)][1]:
                continue
            if id(value) in seen:
                continue
            seen.add(id(value))
            if len(seen) > _WRITE_SCAN_MAX:
                return None
            if kind is types.FunctionType:
                if value.__globals__ is namespace:
                    functions.append(value)
                nxt.append(value.__dict__.get("__wrapped__"))
                nxt.extend(value.__defaults__ or ())
                nxt.extend((value.__kwdefaults__ or {}).values())
                for cell in value.__closure__ or ():
                    try:
                        nxt.append(cell.cell_contents)
                    except ValueError:  # an empty cell
                        pass
            elif kind is lru:
                nxt.append(vars(value).get("__wrapped__"))
            elif kind is functools.partial:
                nxt.extend([value.func, *value.args, *(value.keywords or {}).values()])
            elif kind is types.MethodType:
                nxt.extend([value.__func__, value.__self__])
            elif id(kind) in container_ids:
                if len(value) <= _WRITE_SCAN_CONTAINER:
                    nxt.extend(dict.values(value) if id(kind) in mapping_ids else value)
            elif _notebook_class(value, namespace) is not None:
                try:
                    members = list(vars(value).values())
                except TypeError:
                    continue
                for member in members:
                    nxt.extend(_method_functions(member))
                    nxt.append(member)
                # Its bases too, as the walk follows them: a writer can sit in a
                # base no name holds (`class Model(Model)`, a decorator's subclass).
                nxt.extend(value.__mro__[1:])
            elif _notebook_class(kind, namespace) is not None:
                nxt.append(kind)  # an instance: its class
        level, depth = nxt, depth + 1
    writes: set[str] = set()
    for fn in functions:
        writes |= _code_names(fn.__code__)[1]
    return frozenset(writes)


def _notebook_object(value, namespace):
    """What a value stands for in the notebook's code: a function defined in
    a cell (see ``_cell_function``), a class defined there, an instance's
    class (its methods run), or for a bound method its instance's (or its
    class method's) class, since the method can call the others through
    ``self``; None for anything else. One resolver for the roots and the
    names the code reads."""
    import types

    if type(value) is types.MethodType:
        owner = value.__self__
        cls = _notebook_class(owner, namespace) or _notebook_class(
            type(owner), namespace
        )
        if cls is not None:
            return cls
    found = _cell_function(value, namespace) or _notebook_class(value, namespace)
    if found is None and _notebook_class(type(value), namespace) is not None:
        found = type(value)
    return found


def _components(children: dict) -> dict:
    """``{node: component index}`` for a graph ``{node: [child, ...]}``, by
    strongly connected component (iterative Tarjan), indexed in the order
    Tarjan closes them: a component is closed after every component it
    reaches, so an edge between two components points to a lower index."""
    index: dict = {}
    low: dict = {}
    on_stack: set = set()
    stack: list = []
    comp_of: dict = {}
    closed = 0
    for start in children:
        if start in index:
            continue
        index[start] = low[start] = len(index)
        stack.append(start)
        on_stack.add(start)
        work = [(start, iter(children[start]))]
        while work:
            node, pending = work[-1]
            for child in pending:
                if child not in index:
                    index[child] = low[child] = len(index)
                    stack.append(child)
                    on_stack.add(child)
                    work.append((child, iter(children[child])))
                    break
                if child in on_stack:
                    low[node] = min(low[node], index[child])
            else:
                work.pop()
                if work:
                    parent = work[-1][0]
                    low[parent] = min(low[parent], low[node])
                if low[node] == index[node]:
                    while True:
                        member = stack.pop()
                        on_stack.discard(member)
                        comp_of[member] = closed
                        if member == node:
                            break
                    closed += 1
    return comp_of


def _maybe_project(name: str, foreign_tops: dict) -> bool:
    """False when ``name``'s top-level module is foreign as a whole
    (``_foreign_top``), decided once per top-level name in
    ``foreign_tops``."""
    top = name.partition(".")[0]
    if top not in foreign_tops:
        foreign_tops[top] = _foreign_top(top)
    return not foreign_tops[top]


def _parameter_defaults(fn) -> dict:
    """A function's parameters that have a default, by name, with the value
    its ``__defaults__`` or ``__kwdefaults__`` holds."""
    code = fn.__code__
    positional = code.co_varnames[: code.co_argcount]
    out: dict = {}
    values = fn.__defaults__
    if type(values) is tuple and len(values) <= len(positional):
        names = positional[len(positional) - len(values) :]
        out.update(zip(names, values, strict=True))
    keyword = fn.__kwdefaults__
    if type(keyword) is dict:
        end = code.co_argcount + code.co_kwonlyargcount
        kwonly = set(code.co_varnames[code.co_argcount : end])
        out.update((k, v) for k, v in keyword.items() if k in kwonly)
    return out


def _notebook_entries(
    namespace: dict, roots, cell_bindings: dict, class_texts
) -> tuple[dict[str, str], list[str], list[str]]:
    """``(entries, modules, added)``: the entries of a notebook's code the
    roots reach; the modules their definitions name (see ``_unit_reading``),
    through the closure or the namespace, as main read them; and what only
    the later readings add (a parameter's default value, a rebinding
    elsewhere in the cell, and on 3.15 the other reading of a lazy proxy of a
    kept cell's import, ``cell_bindings`` from ``_lazy_bindings``; a class
    with no method of its own, ``class_texts``: see ``_class_sites``). Both are
    to follow as files; ``_notebook_record`` drops the foreign ones. Entries
    are ``{"<name> (notebook)": digest}`` for each name their code looks up
    (and each root by its name) that stands for a function or class defined
    in a cell. A
    function is its definition's syntax tree; a class its ``class``
    statement; an instance, a bound method's instance, its class. A digest
    covers what its object reaches through edges the run cannot move: a
    class's methods and notebook bases, a wrapper's ``__wrapped__``.

    No value is compared, a closure's or a default's contents neither: a
    value is only safe to compare when every writer of it is found, and in a
    notebook a writer can sit wherever a value can. A name any of the
    notebook's functions rebinds (``global helper; helper = fast``, see
    ``_notebook_writes``) is not followed: what it holds when the run starts
    can be gone by the next. A record past its
    object bound has no notebook entries and no modules (with a warning):
    which objects a cut walk took depends on its order. Objects are
    inspected by type, and only a function's own ``__dict__`` and a module's
    ``__name__`` (which ``_namespace_modules`` reads too) are read, so a
    lazy proxy is never resolved."""
    import types

    def resolve(value):
        return _notebook_object(value, namespace)

    over = [False]
    rebound = _notebook_writes(namespace, roots)
    if rebound is None:  # past the scan's bound: which writes it saw is unknown
        over[0] = True
        rebound = frozenset()

    # uid -> [kind, object, children, methods]
    units: dict[int, list] = {}
    base_modules: set[str] = set()
    codes: list = []

    def visit(obj) -> int | None:
        """Take a function or class into ``units``; None past the bound."""
        if id(obj) in units:
            return id(obj)
        if len(units) >= _NOTEBOOK_MAX_OBJECTS:
            over[0] = True
            return None
        unit = units[id(obj)] = [None, obj, [], []]
        if type(obj) is types.FunctionType:
            unit[0] = "function"
            codes.append(obj.__code__)
            # A notebook decorator's wrapper is a cell function itself.
            targets = [resolve(obj.__dict__.get("__wrapped__"))]
        else:
            methods = _class_methods(obj, namespace)
            unit[0], unit[3] = "class", methods  # each method adds its code
            bases = [(b, _notebook_class(b, namespace)) for b in obj.__bases__]
            targets = [*methods, *(nb for _, nb in bases)]
            for base, notebook_base in bases:
                if notebook_base is None:
                    # A base from a module: its module, by type, as
                    # _notebook_class reads it, whether or not the class's
                    # statement is found (_class_sites).
                    try:
                        module = vars(base).get("__module__")
                    except TypeError:
                        module = None
                    if isinstance(module, str):
                        base_modules.add(module)
        for target in targets:
            uid = visit(target) if target is not None else None
            if uid is not None:
                unit[2].append(uid)
        return id(obj)

    root_units: dict[str, list[int]] = {}
    for key, root in roots:
        found = resolve(root)
        uid = visit(found) if found is not None else None
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
            if _kernel_name(name) or name in rebound:
                continue
            found = resolve(namespace[name])
            if found is not None:
                named[name] = found
                visit(found)
    if over[0]:
        if not _NOTEBOOK_TRUNCATION_WARNED:
            _NOTEBOOK_TRUNCATION_WARNED.append(True)
            _best_effort_warning(
                "resume/cache_dir: the notebook code this task reaches is larger "
                "than the record covers, so it is not compared; use a fresh "
                "working_dir after editing it."
            )
        return {}, [], []

    def sites_of(uid: int) -> tuple[list, bool, bool]:
        """``([(tree, nodes)], ambiguous, later)``: a unit's cell trees and its
        syntax nodes in them (a function's definition, a class's ``class``
        statement, see ``_class_sites``); none when its cell is not kept."""
        kind, obj, _, methods = units[uid]
        if kind == "function":
            site = _definition_site(obj.__code__)
            return ([] if site is None else [site]), False, False
        return _class_sites(obj, methods, namespace, class_texts)

    def own(kind: str, nodes) -> str:
        """A unit's own definition, without what it reaches."""
        text = "?" if nodes is None else "|".join(_canonical_dump(n) for n in nodes)
        return _short_hash(f"{kind}\n{text}".encode())

    # A component hashes its members' own definitions and the digests of the
    # components it reaches, which close first, so a digest does not depend
    # on the order the walk took. Without captures the edges form no cycle
    # (methods, bases and __wrapped__ only point down); components keep that
    # true if one ever appears.
    unit_sites = {uid: sites_of(uid) for uid in units}
    owns = {
        uid: (
            _ambiguous_own(units[uid][1])
            if ambiguous
            else own(units[uid][0], sites[0][1] if sites else None)
        )
        for uid, (sites, ambiguous, _) in unit_sites.items()
    }

    # The project modules the reached definitions name: a cell's `import
    # pkg.sub` binds only `pkg`, so `pkg.sub.f` in a definition (its body, a
    # decorator, a base class) is what reaches the submodule, and an import
    # inside a function binds a local. By name, from the text the digests
    # hash, so what happens to be loaded does not count.
    # A chain skips only the kernel's own names: a name a notebook function
    # rebinds (a reload helper) still holds a module, and skipping it would
    # drop the submodule.
    def lazy_extra(name: str, found) -> tuple[str, ...]:
        """On 3.15, the readings a lazy proxy of a kept cell's import binding
        the namespace's ``name`` had before it resolved (``cell_bindings``),
        while ``name`` holds that module or a submodule of it (``found``, its
        ``_chain_bases``): once the proxy resolves, the record reads as it
        did before. Not while ``name`` holds another module (``import other
        as pkg`` after ``lazy import pkg.sub``)."""
        extra = cell_bindings.get(name)
        if not extra or not found or any(b.partition(".")[0] != name for b in found):
            return ()
        return (name, *sorted(extra))

    def follow(into: set, bases, rest) -> None:
        for base in bases:
            into.add(".".join((base, *rest)))

    modules: set[str] = set(base_modules)
    # What only this record's later readings give (a default, a rebinding,
    # a lazy proxy's other reading): main's had none of it, so it is followed
    # after every seed main had (``_record``).
    added: set[str] = set()
    for uid, (sites, _, later) in unit_sites.items():
        # A class found otherwise than through a method: main read nothing of
        # it, so what it names is followed after main's seeds.
        main_modules = set(modules) if later else None
        # A name a function around the unit binds to something unknown is a
        # free variable of the unit: its closure holds the value.
        kind, obj = units[uid][0], units[uid][1]
        free = {}
        defaults: dict = {}
        if kind == "function":
            # A function's closure has one cell per free variable.
            cells = obj.__closure__ or ()
            free = dict(zip(obj.__code__.co_freevars, cells, strict=False))
            defaults = _parameter_defaults(obj)
        for site in sites:
            # Each unit alone (a method in its class's statement and on its
            # own): the union does not depend on the order the walk took.
            unit_modules, global_chains, cell_chains, rebound = _unit_reading(*site)
            modules |= unit_modules
            # The compiler's free variables decide (a walrus in a comprehension
            # binds in the enclosing function, which the scopes read as the
            # comprehension's): a chain on one asks the closure too.
            asked = cell_chains | {c for c in global_chains if c[0] in free}
            for chain in asked:
                if chain[0] not in free:
                    continue
                try:
                    value = free[chain[0]].cell_contents
                except ValueError:  # an empty cell: nothing bound yet
                    continue
                follow(modules, _chain_bases(chain[0], value), chain[1:])
            # The namespace too, for a free variable as well: its closure can be
            # empty now and filled during the run with what the syntax cannot
            # tell (`nonlocal pkg; pkg = importlib.import_module("pkg")`), and
            # main's record read the namespace's module of that name.
            for chain in global_chains:
                if chain[0] in namespace and not _kernel_name(chain[0]):
                    found = _chain_bases(chain[0], namespace[chain[0]])
                    follow(modules, found, chain[1:])
                    follow(added, lazy_extra(chain[0], found), chain[1:])
            # The floor: what main's reading names counts too, resolved as main
            # resolved it (the closure first, then the namespace), so this record
            # never holds less than main's did.
            floor_modules, floor_chains = _floor_reading(site[1])
            modules |= floor_modules
            for chain in floor_chains:
                extra: tuple[str, ...] = ()
                if chain[0] in free:
                    try:
                        found = _chain_bases(chain[0], free[chain[0]].cell_contents)
                    except ValueError:  # an empty cell: nothing bound yet
                        continue
                elif chain[0] in namespace and not _kernel_name(chain[0]):
                    found = _chain_bases(chain[0], namespace[chain[0]])
                    extra = lazy_extra(chain[0], found)
                else:
                    continue
                follow(modules, found, chain[1:])
                follow(added, extra, chain[1:])
            # A chain on one of the unit's own parameters that has a default: the
            # default is a value it can hold, which the syntax may not tell (`m=mod`
            # in a factory), read by type as a closure cell is.
            for chain in global_chains | floor_chains:
                if chain[0] in defaults:
                    follow(added, _chain_bases(chain[0], defaults[chain[0]]), chain[1:])
            # What a rebinding elsewhere in the tree can make a chain reach.
            for target in rebound:
                if target[0] == "module":
                    added.add(".".join(target[1]))
                    continue
                name, rest = target[1][0], target[1][1:]
                if name in namespace and not _kernel_name(name):
                    found = _chain_bases(name, namespace[name])
                    follow(added, (*found, *lazy_extra(name, found)), rest)
        if main_modules is not None:
            added |= modules - main_modules
            modules = main_modules
    comp_of = _components({uid: unit[2] for uid, unit in units.items()})
    members: dict[int, list[int]] = {}
    for uid, ci in comp_of.items():
        members.setdefault(ci, []).append(uid)
    comp_digest: dict[int, str] = {}
    for ci in sorted(members):
        reached = {
            comp_of[c] for m in members[ci] for c in units[m][2] if comp_of[c] != ci
        }
        text = "\n".join(sorted(owns[m] for m in members[ci]))
        text += "\n--\n" + "\n".join(sorted({comp_digest[r] for r in reached}))
        comp_digest[ci] = _short_hash(text.encode())

    def digest(uid: int) -> str:
        return _short_hash(f"{owns[uid]}\n{comp_digest[comp_of[uid]]}".encode())

    entries: dict[str, str] = {}
    for key, uids in root_units.items():
        found = sorted({digest(u) for u in uids})  # roots sharing a name
        entries[f"{key} (notebook)"] = (
            found[0] if len(found) == 1 else _short_hash("\n".join(found).encode())
        )
    for name, obj in named.items():
        key = f"{name} (notebook)"
        if key not in entries:
            entries[key] = digest(id(obj))
    return entries, sorted(modules), sorted(added - modules)


def _error_kind(error: BaseException) -> str:
    """An exception's type name, read from its slot: its message is the
    notebook's own text, and `type(error).__name__` could run a notebook
    metaclass's property."""
    try:
        return type.__dict__["__name__"].__get__(type(error))
    except Exception:  # noqa: BLE001 - a name is not worth failing for
        return "an error"


def _warn_notebook_unrecorded(error: BaseException) -> None:
    """Warn, once per run (``driver_script_entries`` clears the flag, so a
    resume warns again), that a notebook's record could not be taken."""
    if _NOTEBOOK_FAILURE_WARNED:
        return
    _NOTEBOOK_FAILURE_WARNED.append(True)
    _best_effort_warning(
        "resume/cache_dir: the notebook code this task reaches could "
        f"not be recorded ({_error_kind(error)}), so it is not compared; use a "
        "fresh working_dir after editing it."
    )


def _warn_cells_unread(error: BaseException) -> None:
    """Warn, once per run like ``_warn_notebook_unrecorded``, that the
    modules the kept cells import could not be read (the notebook's own
    entries still count)."""
    if _CELL_SCAN_WARNED:
        return
    _CELL_SCAN_WARNED.append(True)
    _best_effort_warning(
        "resume/cache_dir: the project modules the notebook's cells import "
        f"could not be read ({_error_kind(error)}), so an edit to one may go "
        "unnoticed; use a fresh working_dir after editing one."
    )


def _notebook_record(notebooks: dict) -> tuple[dict[str, str], list[str], list[str]]:
    """``(entries, reached, later)`` for ``notebooks`` (``{id: (namespace,
    [(key, root)])}``): the notebook entries; the project modules their
    reached code names as main read them; and the rest, followed after every
    seed main had (``_record``), so a record cut by its file bound keeps what
    main's kept: what the later readings add (``_notebook_entries``) and what
    the kernel's kept cells import (``_kept_cell_reading``). The cells'
    modules do not depend on the walk, so they count when it passes its bound
    or fails. Best effort: a notebook whose entries cannot be taken gives
    neither its entries nor its modules (as on main), and a scan of the cells
    that fails gives no cell modules, each with its own warning, since an
    edit then goes unnoticed (a notebook metaclass's attribute hook that
    raises, say)."""
    _cell_watch()  # a shell made after mushin was imported
    out: dict[str, str] = {}
    reached: set[str] = set()
    added: set[str] = set()
    cell_modules: set[str] = set()
    cell_bindings: dict = {}
    scan = None
    try:
        scan = _scan_kept_cells()
        cell_modules, cell_bindings = _kept_cell_reading(scan)
    except Exception as error:  # noqa: BLE001 - never fail the record
        cell_modules, cell_bindings = set(), {}
        _warn_cells_unread(error)
    # The kept cells' `class` statements, from the same scan, indexed only if
    # a class with no method of its own needs them; None when the scan
    # failed (its warning is given above, and such a class is ambiguous).
    index: list = []

    def class_texts(qualname: str) -> list[str] | None:
        if not index:
            index.append(None if scan is None else _kept_class_index(scan))
        return None if index[0] is None else index[0].get(qualname, [])

    for namespace, roots in notebooks.values():
        try:
            lazy = _lazy_bindings(cell_bindings, namespace)
            entries, names, more = _notebook_entries(
                namespace, roots, lazy, class_texts
            )
            out.update(entries)
            reached.update(names)
            added.update(more)
        except Exception as error:  # noqa: BLE001 - never fail the record
            _warn_notebook_unrecorded(error)
    # One judgment per top-level name for both lists.
    foreign_tops: dict[str, bool] = {}
    reached = {m for m in reached if _maybe_project(m, foreign_tops)}
    later = (added | cell_modules) - reached
    later = {m for m in later if _maybe_project(m, foreign_tops)}
    return out, sorted(reached), sorted(later)


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
            # A class method is bound to its class (whose type is a metaclass).
            owner = layer.__self__
            layer = owner if issubclass(type(owner), type) else type(owner)
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
    the driver when a run starts: ``{file: (digest, imports, package)}``, and
    for roots defined in a notebook ``{NOTEBOOK_ENTRIES: {key: digest},
    NOTEBOOK_MODULES: [module, ...], NOTEBOOK_LATER_MODULES: [module, ...]}``
    (``_notebook_record``; no script path can be those keys). A
    parallel worker runs the driver's pickled script functions, so the
    driver's view of the script is the one that runs; the runner ships these
    to every cell, so an edit to the script while the sweep runs is not
    mistaken for the code its cells ran. Empty when they cannot be taken."""
    # A kernel is one long process, and a resume is where a missed edit
    # costs: warn about the notebook record on every run. (Python's own
    # once-per-place memory does not hide the repeat: it is cleared whenever
    # the warning filters change, which a sweep's run does.)
    _NOTEBOOK_TRUNCATION_WARNED.clear()
    _NOTEBOOK_FAILURE_WARNED.clear()
    _CELL_SCAN_WARNED.clear()
    _LATER_TRUNCATION_WARNED.clear()
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
            (
                out[NOTEBOOK_ENTRIES],
                out[NOTEBOOK_MODULES],
                out[NOTEBOOK_LATER_MODULES],
            ) = _notebook_record(notebooks)
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
    notebooks: dict = {}
    for root in roots:
        if root is not None:
            _root_entries(root, code, seeds, scripts, notebooks=notebooks)
    for target in targets:
        _target_entries(target, seeds, scripts)
    _script_entries(scripts, code, seeds, scripts_given or {})
    if notebooks:
        given = scripts_given or {}
        notebook = given.get(NOTEBOOK_ENTRIES)
        modules, later = given.get(NOTEBOOK_MODULES), given.get(NOTEBOOK_LATER_MODULES)
        if type(notebook) is not dict:
            notebook, modules, later = _notebook_record(notebooks)
        # A driver from before a key was shipped (a runner pickled by an
        # earlier version) took none; its entries still count.
        modules = modules if type(modules) is list else []
        later = later if type(later) is list else []
        code.update(notebook)
        # First, so a record cut by _FILES_MAX starts from what the task's own
        # code names (depth first: the first one's imports come before the
        # second one).
        seeds[:0] = modules
    else:
        later = []
    # Two passes: the first is main's seeds in main's order, so it keeps what
    # main's record kept; the second, with what is left of the bound, what
    # only this record reads (the notebook's later modules).
    seen: set[str] = set()
    entries, cut = _file_entries(seeds, seen)
    code.update(entries)
    later_entries, later_cut = _file_entries(later, seen, _FILES_MAX - len(entries))
    code.update(later_entries)
    if cut and not _TRUNCATION_WARNED:
        _TRUNCATION_WARNED.append(True)
        _best_effort_warning(
            "resume/cache_dir: the task's code imports more project files than "
            "the fingerprint covers, so edits past that point are not compared; "
            "use a fresh working_dir after a large refactor."
        )
    elif later_cut and not cut and not _LATER_TRUNCATION_WARNED:
        _LATER_TRUNCATION_WARNED.append(True)
        _best_effort_warning(
            "resume/cache_dir: the project modules the notebook's cells import, "
            "and those its code reaches through a default, a rebinding or a lazy "
            "import, reach more files than the fingerprint covers, so an edit past "
            "that point may go unnoticed; use a fresh working_dir after editing one."
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
    (notebook)"`` for each function or class defined in its cells that its
    code reaches by name, by its definition (see ``_notebook_entries``), and
    the project modules of the objects its namespace holds, those its
    reached definitions name (``pkg.sub.f`` after ``import pkg.sub``, an
    import inside a function; see ``_unit_reading``), and every one its
    kept cells import (``_kept_cell_reading``); the driver takes the
    notebook entries and those modules when the run starts and ships them in
    ``scripts``.

    Not covered: the standard library, mushin, installed packages other than
    editable ones (a dependency upgrade is provenance's job), data files, the
    environment, a notebook's values and the functions reached only through a
    container, a closure or a default, a ``_target_`` defined in a notebook,
    and code reached only through a
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


# In a running IPython shell, watch its cells from this import on, so a class
# statement run after it is pinned to its cell (``_class_sites``).
_cell_watch()
