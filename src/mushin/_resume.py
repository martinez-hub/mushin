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
    a function, and the constants it imports); and whether ``_FILES_MAX`` cut
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


def _outer_exprs(node) -> list:
    """What a ``def``, ``lambda`` or ``class`` evaluates where it is defined
    (decorators, defaults, annotations, bases, type parameters), not its
    body."""
    import ast

    out = list(getattr(node, "decorator_list", ()))
    out += list(getattr(node, "type_params", ()) or ())
    if isinstance(node, ast.ClassDef):
        return out + node.bases + [k.value for k in node.keywords]
    a = node.args
    out += a.defaults + [d for d in a.kw_defaults if d is not None]
    if isinstance(node, ast.Lambda):
        return out
    params = [*a.posonlyargs, *a.args, *a.kwonlyargs, a.vararg, a.kwarg]
    out += [p.annotation for p in params if p is not None and p.annotation is not None]
    if node.returns is not None:
        out.append(node.returns)
    return out


def _bound_names(node) -> set[str]:
    """The module-level names a top-level statement binds, by Python's
    scopes: a comprehension's variables, a lambda's parameters and the names
    a nested ``def`` or ``class`` body binds are their own, while a walrus in
    a comprehension binds the module's. An annotation without a value binds
    nothing. A star import binds ``"*"``."""
    import ast

    out: set[str] = set()
    defs = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
    comps = (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)

    def visit(n, in_comp: bool) -> None:
        if isinstance(n, defs) or isinstance(n, ast.Lambda):
            if isinstance(n, defs) and not in_comp:
                out.add(n.name)
            for sub in _outer_exprs(n):
                visit(sub, in_comp)
            return
        if isinstance(n, ast.NamedExpr):
            if isinstance(n.target, ast.Name):
                out.add(n.target.id)
            visit(n.value, in_comp)
            return
        if isinstance(n, comps):
            in_comp = True
        elif in_comp:
            pass  # a comprehension's own targets
        elif isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
            out.add(n.id)
        elif isinstance(n, ast.AnnAssign) and n.value is None:
            visit(n.annotation, in_comp)
            return
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            for alias in n.names:
                out.add(
                    "*"
                    if alias.name == "*"
                    else (alias.asname or alias.name).partition(".")[0]
                )
            return
        elif isinstance(n, ast.ExceptHandler) and n.name:
            out.add(n.name)
        elif isinstance(n, (ast.MatchAs, ast.MatchStar)) and n.name:
            out.add(n.name)
        elif isinstance(n, ast.MatchMapping) and n.rest:
            out.add(n.rest)
        for sub in ast.iter_child_nodes(n):
            visit(sub, in_comp)

    visit(node, False)
    return out


def _eval_loads(node) -> set[str]:
    """The names a top-level statement reads when it runs: everything but the
    bodies of the functions and lambdas it defines, which run when called. A
    class body runs where it is defined, and a name it reads before binding
    it there is the module's (``SCALE = SCALE``). ``X += 1`` reads ``X``."""
    import ast

    out: set[str] = set()

    def visit(n) -> None:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            for sub in _outer_exprs(n):
                visit(sub)
            return
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
            out.add(n.id)
        elif isinstance(n, ast.AugAssign) and isinstance(n.target, ast.Name):
            out.add(n.target.id)
        for sub in ast.iter_child_nodes(n):
            visit(sub)

    visit(node)
    return out


#: Symbol-table scopes that only evaluate where they are, so their reads are
#: the statement's own (read from its syntax tree): comprehensions (which
#: Python 3.12+ inlines and earlier versions do not), and the annotation and
#: type-parameter scopes some versions add. Their nested functions still run
#: when called.
_EAGER_SCOPES = ("listcomp", "setcomp", "dictcomp", "genexpr")


def _call_scope_facts(scopes) -> tuple[set[str], set[str]]:
    """``(reads, writes)`` of the functions a statement defines, from its
    symbol-table scopes: the globals their bodies read when called, and the
    globals they assign (``global DEVICE``; a call can rebind the module's
    name). A function's comprehensions and nested classes are part of its
    body. The same on every Python version: the scopes only some versions
    have (an inlined comprehension, an annotation scope) are read alike."""
    reads: set[str] = set()
    writes: set[str] = set()
    stack = [(scope, False) for scope in scopes]
    while stack:
        table, in_function = stack.pop()
        kind = table.get_type()
        is_function = kind == "function"
        if is_function and table.get_name() in _EAGER_SCOPES and not in_function:
            is_function = False  # at module level: the statement's syntax tree
        counts = in_function or is_function
        if counts:
            for sym in table.get_symbols():
                if not sym.is_global():
                    continue
                if sym.is_referenced():
                    reads.add(sym.get_name())
                if sym.is_assigned() and sym.is_declared_global():
                    writes.add(sym.get_name())
        stack.extend((child, counts) for child in table.get_children())
    return reads, writes


def _decorator_bases(node) -> list[str]:
    """The base names of the decorators on every ``def`` and ``class`` in a
    statement outside function bodies (in a block, or on a class's methods)."""
    import ast

    out: list[str] = []

    def base(expr):
        while isinstance(expr, (ast.Call, ast.Attribute, ast.Subscript)):
            expr = expr.func if isinstance(expr, ast.Call) else expr.value
        return expr.id if isinstance(expr, ast.Name) else None

    def visit(n) -> None:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.extend(b for b in map(base, n.decorator_list) if b)
            if isinstance(n, ast.ClassDef):
                for sub in n.body:
                    visit(sub)
            return
        for sub in ast.iter_child_nodes(n):
            if isinstance(sub, (ast.stmt, ast.excepthandler, ast.match_case)):
                visit(sub)

    visit(node)
    return out


class _Stmt:
    """What the reach rule needs of one top-level statement."""

    __slots__ = (
        "node",
        "binds",
        "must",
        "evals",
        "calls",
        "writes",
        "decorators",
        "_dh",
        "_imports",
    )

    def __init__(self, node, scopes):
        import ast

        self.node = node
        self.binds = _bound_names(node)
        must = (
            ast.Assign,
            ast.AugAssign,
            ast.AnnAssign,
            ast.Import,
            ast.ImportFrom,
            ast.FunctionDef,
            ast.AsyncFunctionDef,
            ast.ClassDef,
            ast.Delete,
            *((ast.TypeAlias,) if hasattr(ast, "TypeAlias") else ()),
        )
        # A block may not run what binds a name (`if FAST: SCALE = 5`): the
        # binding before it stays live too.
        self.must = self.binds if isinstance(node, must) else set()
        self.evals = _eval_loads(node)
        self.calls, self.writes = _call_scope_facts(scopes)
        self.decorators = _decorator_bases(node)
        self._dh = None
        self._imports = None

    @property
    def star(self) -> bool:
        return "*" in self.binds

    @property
    def future(self) -> bool:
        import ast

        return (
            isinstance(self.node, ast.ImportFrom) and self.node.module == "__future__"
        )

    def stripped(self):
        import ast
        import copy

        # The script runs as __main__: its helpers' guards run too.
        return _strip_tree(
            ast.Module(body=[copy.deepcopy(self.node)], type_ignores=[]),
            main_guards=False,
        )

    def dh(self) -> str:
        if self._dh is None:
            tree = self.stripped()
            self._dh = _short_hash(_canonical_dump(tree).encode())
            self._imports = _import_pairs(tree)
        return self._dh

    def imports(self) -> list:
        self.dh()
        return self._imports


def _statements(nodes, table) -> list[_Stmt]:
    """A ``_Stmt`` for each top-level statement ``nodes`` of one source, with
    the symbol-table scopes ``table`` holds mapped to the statement they are
    in by line (two definitions of one name are two statements)."""
    spans = []
    for node in nodes:
        start = min(
            [node.lineno, *(d.lineno for d in getattr(node, "decorator_list", ()))]
        )
        spans.append((start, getattr(node, "end_lineno", None) or node.lineno))
    scopes: list[list] = [[] for _ in nodes]
    for child in table.get_children():
        line = child.get_lineno()
        for i, (start, end) in enumerate(spans):
            if start <= line <= end:
                scopes[i].append(child)
                break
    return [_Stmt(node, s) for node, s in zip(nodes, scopes, strict=True)]


def _run_statements(tree) -> list:
    """A source's top-level statements in the order they run as ``__main__``:
    a main guard's body runs and its ``else`` does not."""
    out = []
    for node in tree.body:
        if _is_main_guard(node):
            out.extend(node.body)
        else:
            out.append(node)
    return out


#: Bounds on one reach: decorator names followed through their binders.
_REACH_NAMES_MAX = 2000


def _reach(
    stmts: list[_Stmt], roots, is_project_import, names=()
) -> tuple[str, list, set]:
    """What the statements at positions ``roots`` can reach in ``stmts``, a
    source's top-level statements in the order they ran, as ``(digest,
    import pairs, unbound names)``.

    A name a statement reads when it runs is the binding before it (``Y = X
    * 2`` keeps the ``X`` it was computed from); a name a function reads when
    called is the latest binding. A binding a block may skip keeps the one
    before it live too, and a function that assigns a global is a binding of
    it wherever it is. Also counted: star imports, ``__future__`` imports, and
    a decorated definition whose decorator reaches, through its bindings (an
    alias, a helper it calls), a project import or module state that counted
    code reads (``@register`` filling the ``MODELS`` the task reads).

    The digest is of the set of reached statements, each with the statements
    its names resolve to, so it depends on what runs, not on the order of
    independent statements or on how often a cell was run unchanged.
    ``names`` are roots too, by their latest binding."""
    import ast
    import bisect
    from collections import deque

    binders: dict[str, list[int]] = {}
    writers: dict[str, list[int]] = {}
    for pos, s in enumerate(stmts):
        for name in s.binds:
            binders.setdefault(name, []).append(pos)
        for name in s.writes:
            writers.setdefault(name, []).append(pos)
    stars = [p for p in binders.get("*", ())]

    def live(name: str, pos: int) -> list[int]:
        found = []
        positions = binders.get(name, ())
        i = bisect.bisect_left(positions, pos) - 1
        while i >= 0:
            found.append(positions[i])
            if name in stmts[positions[i]].must:
                break
            i -= 1
        # A function that assigns the global can have run before `pos` only
        # if it was defined before it.
        return found + [p for p in writers.get(name, ()) if p < pos and p not in found]

    included: dict[int, list] = {}
    counted: set[str] = set()
    unbound: set[str] = set()
    queue = deque(
        sorted({*roots, *stars, *(p for p, s in enumerate(stmts) if s.future)})
    )
    end = len(stmts)
    for name in names:
        queue.extend(live(name, end))

    def visit(pos: int) -> None:
        s = stmts[pos]
        edges = []
        for names, at in ((s.evals, pos), (s.calls, end)):
            for name in names:
                targets = live(name, at)
                if not targets:
                    unbound.add(name)
                for t in targets:
                    edges.append((name, t))
                    if t not in included:
                        queue.append(t)
        included[pos] = edges
        counted.update(s.evals, s.calls)

    def is_state(name: str) -> bool:
        """Module state a decorator may write: a name the source binds to a
        value or a class (a registry), or imports from the project; not a
        function both call, nor a foreign module both use."""
        for p in binders.get(name, ()):
            node = stmts[p].node
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                if is_project_import(node, name):
                    return True
            elif not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                return True
        return False

    def registers(name: str, pos: int, seen: set) -> bool:
        if name in seen or len(seen) > _REACH_NAMES_MAX:
            return False
        seen.add(name)
        targets = live(name, pos)
        if not targets:
            # A name nothing binds may come from a star import.
            return any(is_project_import(stmts[p].node, name) for p in stars)
        for t in targets:
            node = stmts[t].node
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                if is_project_import(node, name):
                    return True
                continue
            reads = stmts[t].evals | stmts[t].calls
            if any(is_state(n) for n in reads & counted if n != name):
                return True
            if any(registers(n, end, seen) for n in reads):
                return True
        return False

    decorated = [p for p, s in enumerate(stmts) if s.decorators]
    while True:
        while queue:
            pos = queue.popleft()
            if pos not in included:
                visit(pos)
        added = False
        for pos in decorated:
            if pos in included:
                continue
            if any(registers(d, pos, set()) for d in stmts[pos].decorators):
                queue.append(pos)
                added = True
        if not added:
            break

    sigs = set()
    imports: list = []
    for pos, edges in included.items():
        s = stmts[pos]
        resolved = sorted({f"{name}>{stmts[t].dh()}" for name, t in edges})
        sigs.add(s.dh() + "|" + ",".join(resolved))
        imports.extend(s.imports())
    digest = _short_hash("\n".join(sorted(sigs)).encode())
    return digest, imports, unbound


def _is_project_import(node, name: str) -> bool:
    """Whether import statement ``node`` binds ``name`` (or ``*``) from a
    project module. A relative import is the script's own package's (``python
    -m pkg.train``), so the project's."""
    import ast

    for alias in node.names:
        bound = (alias.asname or alias.name).partition(".")[0]
        if bound != name and alias.name != "*":
            continue
        if isinstance(node, ast.ImportFrom):
            if node.level:
                return True
            if not node.module:
                return False
            module = node.module
        else:
            module = alias.name
        return _project_file_of(module) is not None
    return False


def _script_digest(path: Path, roots: frozenset[str]):
    """``(digest, imports)`` for a driver script whose top-level names
    ``roots`` define the task (the class or function holding it,
    ``pre_task``, a ``_target_``): what those definitions can reach by name in
    the script as it runs (see ``_reach``), so the resilience loop (extend the
    grid, add a print, resume) is not an edit. The script runs as
    ``__main__``: its main block's body runs, its ``else`` does not. Not
    counted: statements that only do something (a call, the run call, an
    attribute or item set) and what only they read (``GRID``, ``ds =
    exp.run(...)``). ``imports`` are those of what counts."""
    import ast
    import symtable

    source, content = _content_key(path)
    cache_key = (*content, tuple(sorted(roots)))
    hit = _SCRIPT_CACHE.get(cache_key)
    if hit is not None:
        return hit
    tree = ast.parse(source)
    # The bytes, so a byte order mark or a coding cookie is honored.
    table = symtable.symtable(source, str(path), "exec")
    stmts = _statements(_run_statements(tree), table)
    starts = [pos for pos, s in enumerate(stmts) if s.binds & roots]
    digest, imports, _ = _reach(stmts, starts, _is_project_import)
    result = (digest, imports)
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
        elif name == "__main__" and file is None:
            # No file: a notebook's cells, read by `_notebook_entry` (in a
            # parallel worker, which has the task by value, the driver's).
            _, names, objects = scripts.setdefault(_NOTEBOOK, (namespace, set(), []))
            names |= tops
            objects.append(identity)
        elif not isinstance(file, str):
            seeds.extend(_namespace_modules(namespace))  # a notebook
        elif not _is_foreign_module(name, file):
            seeds.append(name)


#: The ``scripts`` key of code defined where there is no file: a notebook's
#: cells (or a REPL's, which cannot be read).
_NOTEBOOK = "<notebook>"

#: Cell statements by the cell's text, bounded like the file caches.
_CELL_CACHE: dict = {}

#: The most recent cells of a notebook that are read.
_CELLS_MAX = 5000


def _ipython_shell_of(namespace):
    """The IPython shell whose namespace ``namespace`` is, or None. IPython is
    never imported here: a shell exists only if it already is."""
    import sys

    module = sys.modules.get("IPython")
    if module is None:
        return None
    try:
        shell = module.get_ipython()
    except Exception:  # noqa: BLE001 - an IPython that cannot answer
        return None
    if shell is None or getattr(shell, "user_ns", None) is not namespace:
        return None
    return shell


#: The magics that run their code in the notebook's namespace, by how IPython
#: rewrites a cell that uses them: ``%time x = 1`` is
#: ``get_ipython().run_line_magic('time', 'x = 1')``.
_LINE_MAGICS_RUN = ("time",)
_CELL_MAGICS_RUN = ("time", "capture")


def _magic_code(node) -> str | None:
    """The code a statement runs through one of those magics, or None."""
    import ast

    if not (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)):
        return None
    func, args = node.value.func, node.value.args
    if not (
        isinstance(func, ast.Attribute)
        and isinstance(func.value, ast.Call)
        and isinstance(func.value.func, ast.Name)
        and func.value.func.id == "get_ipython"
        and all(isinstance(a, ast.Constant) and isinstance(a.value, str) for a in args)
    ):
        return None
    if func.attr == "run_line_magic" and len(args) == 2:
        return args[1].value if args[0].value in _LINE_MAGICS_RUN else None
    if func.attr == "run_cell_magic" and len(args) == 3:
        return args[2].value if args[0].value in _CELL_MAGICS_RUN else None
    return None


def _cell_statements(text: str, depth: int = 0):
    """The statements of one cell, cached by its text, or None for a cell
    that does not parse (it never ran). The code a ``%time`` or ``%%time``
    runs stands in for the magic's call."""
    import ast
    import symtable

    key = _short_hash(text.encode("utf-8", "surrogatepass"))
    if key in _CELL_CACHE:
        return _CELL_CACHE[key]
    try:
        tree = ast.parse(text)
        stmts = _statements(
            _run_statements(tree), symtable.symtable(text, "<cell>", "exec")
        )
    except (SyntaxError, ValueError, RecursionError):
        stmts = None
    if stmts is not None and depth < 4:
        expanded = []
        for stmt in stmts:
            code = _magic_code(stmt.node)
            if code is None:
                expanded.append(stmt)
            else:
                expanded.extend(_cell_statements(code, depth + 1) or [])
        stmts = expanded
    _cache_put(_CELL_CACHE, key, stmts)
    return stmts


def _code_of(obj):
    """The code object that places ``obj`` in a cell: a function's, or a
    class's own method's."""
    import types

    if isinstance(obj, types.FunctionType):
        return obj.__code__
    if isinstance(obj, type):
        for value in vars(obj).values():
            fn = (
                getattr(value, "__func__", None)
                or getattr(value, "fget", None)
                or value
            )
            if isinstance(fn, types.FunctionType) and fn.__qualname__.startswith(
                obj.__qualname__ + "."
            ):
                return fn.__code__
    return None


def _cell_position(obj, cells: list[str], starts: list[tuple[int, list]]):
    """The position of the statement that made ``obj``, in the latest run of
    the cell its code comes from (IPython registers each cell's text in
    linecache under the cell's file name), or None."""
    import linecache

    code = _code_of(obj)
    if code is None:
        return None
    lines = linecache.getlines(code.co_filename)
    if not lines:
        return None
    text = "".join(lines).rstrip("\n")
    for i in range(len(cells) - 1, -1, -1):
        if cells[i].rstrip("\n") != text:
            continue
        start, part = starts[i]
        line = code.co_firstlineno  # a decorated def's first decorator
        for j, stmt in enumerate(part):
            node = stmt.node
            first = min(
                [node.lineno, *(d.lineno for d in getattr(node, "decorator_list", ()))]
            )
            if first <= line <= (node.end_lineno or node.lineno):
                return start + j
        return None
    return None


def _notebook_entry(namespace: dict, names, objects):
    """``(digest, imports)`` for code defined in an IPython notebook (or
    console): what the roots can reach by name through the cells the kernel
    ran, in the order it ran them (see ``_reach``), so re-running a cell
    unchanged, or the cell that extends the grid, is not an edit. Each root is
    the statement that made it, in the latest run of its cell (a sweep built
    before its task's cell changed runs the old task), else its name's latest
    binding. A name no cell binds (``%run``, an extension) brings the project
    module of its value's type. None outside IPython, or when no root is
    found in the cells."""
    shell = _ipython_shell_of(namespace)
    if shell is None:
        return None
    cells = [c for c in shell.history_manager.input_hist_parsed if isinstance(c, str)]
    cells = cells[-_CELLS_MAX:]
    stmts: list[_Stmt] = []
    starts: list[tuple[int, list]] = []
    for text in cells:
        part = _cell_statements(text) or []
        starts.append((len(stmts), part))
        stmts.extend(part)
    roots, unplaced = [], set()
    for obj in objects:
        pos = _cell_position(obj, cells, starts)
        if pos is None:
            unplaced = set(names)
        else:
            roots.append(pos)
    if not roots and not any(n in s.binds for s in stmts for n in unplaced):
        return None
    digest, imports, unbound = _reach(stmts, roots, _is_project_import, names=unplaced)
    values = {n: namespace[n] for n in unbound if n in namespace}
    modules = [m for m in _namespace_modules(values) if m != "__main__"]
    return digest, [*imports, *((m, 0) for m in modules)]


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
            if file == _NOTEBOOK:
                found = _notebook_entry(namespace, names, objects)
                if found is not None:
                    out[file] = (*found, None)
                continue
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
    counted code imports. Code defined where there is no file is a notebook's
    cells when they can be read, else the project modules its namespace holds
    (a REPL, or a worker the driver gave no entry)."""
    for file, (namespace, names, objects) in scripts.items():
        if file == _NOTEBOOK:
            entry = given.get(file)
            if entry is None:
                try:
                    found = _notebook_entry(namespace, names, objects)
                except Exception:  # noqa: BLE001 - best effort: as before
                    found = None
                entry = None if found is None else (*found, None)
            if entry is None:
                seeds.extend(_namespace_modules(namespace))
                continue
            digest, imports, package = entry
            code["__main__ (notebook)"] = digest
            seeds.extend(_resolved(imports, "__main__", False, package=package))
            continue
        if file in given:
            digest, imports, package = given[file]
        else:
            digest, imports = _script_entry(file, namespace, names, objects)
            package = namespace.get("__package__")
            package = package if isinstance(package, str) else None
        code[f"{Path(file).stem} (script)"] = digest
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
        if len(parts) < 2:
            return
        if file is None:
            # A class defined in a notebook cell.
            key = _NOTEBOOK
        elif isinstance(file, str) and file.endswith(".py"):
            if _foreign_by_path(file, "__main__"):
                return
            key = file
        else:
            return
        _, names, objects = scripts.setdefault(key, (namespace, set(), []))
        names.add(parts[1])
        target = namespace.get(parts[1])
        if target is not None:
            objects.append(target)
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
    project files, at any depth (imports inside functions included), is an
    entry ``"<module> (file)"``, hashed as its syntax tree without its main
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
    ``scripts`` are the driver's script entries (``driver_script_entries``).
    A target in the driver script gets its own key, ``"<script>:<name>
    (script)"`` (``"__main__:<name> (notebook)"`` in a notebook), so it never
    replaces the roots' entry for that script."""
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
                    head, sep, kind = k.rpartition(" (")
                    if sep and kind in ("script)", "notebook)") and len(parts) > 1:
                        k = f"{head}:{parts[1]} ({kind}"
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
