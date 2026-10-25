# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
"""resume=True and cache_dir= notice edits to the project code a task can run.

The task's own source was fingerprinted before; a helper it calls, or a
constant in its module, was not, so a resume after editing one returned the
stale result. Each cell now records the project files its task's module
imports, directly or through other project files, by their syntax trees (a
driver script by the code that is not its run block), and a resume re-runs a
cell whose files changed.
"""

from __future__ import annotations

import functools
import hashlib
import importlib
import json
import os
import pathlib
import subprocess
import sys
import textwrap
import types
import warnings

import pytest

from mushin import multirun
from mushin._resume import (
    STATUS_FILE,
    changed_project_code,
    code_fingerprint,
    project_fingerprints,
)

HELPERS = "mushin_t_helpers"
EXP = "mushin_t_exp"
PY = f"{sys.version_info[0]}.{sys.version_info[1]}"


@pytest.fixture(autouse=True)
def _new_modules_are_found(monkeypatch):
    """Tests here write a module and then import it or look it up. The import
    system caches each directory's listing and rescans it only when the
    directory's mtime moves, which on Windows can lag a file written a moment
    ago, so the module is not found. ``importlib.invalidate_caches`` is the
    documented remedy; every ``.py`` written here calls it."""
    write_text = pathlib.Path.write_text

    def write_and_invalidate(self, *args, **kwargs):
        n = write_text(self, *args, **kwargs)
        if self.suffix == ".py":
            importlib.invalidate_caches()
        return n

    monkeypatch.setattr(pathlib.Path, "write_text", write_and_invalidate)


def _write(root, scale: int = 10, offset: int = 0, comment: str = ""):
    (root / f"{HELPERS}.py").write_text(
        textwrap.dedent(f"""
            SCALE = {scale}
            {comment}

            def scale(x):
                return float(x) * SCALE
            """)
    )
    (root / f"{EXP}.py").write_text(
        textwrap.dedent(f"""
            import {HELPERS} as helpers
            from mushin.workflows import MultiRunMetricsWorkflow

            OFFSET = {offset}


            class W(MultiRunMetricsWorkflow):
                @staticmethod
                def task(a):
                    return dict(v=helpers.scale(a) + OFFSET)
            """)
    )


@pytest.fixture
def project(tmp_path, monkeypatch):
    """A two-module project on sys.path: a task module and the helper it calls.
    Yields the project root and the imported task module; ``_reload()`` picks
    up an edit the way a fresh process would."""
    root = tmp_path / "proj"
    root.mkdir()
    _write(root)
    # No .pyc: an edit that keeps a file's size within the same second would
    # otherwise pass Python's staleness check and reload the old bytecode.
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    monkeypatch.syspath_prepend(str(root))
    for name in (EXP, HELPERS):
        sys.modules.pop(name, None)
    exp = importlib.import_module(EXP)
    yield root, exp
    for name in (EXP, HELPERS):
        sys.modules.pop(name, None)


def _reload(exp):
    importlib.reload(sys.modules[HELPERS])
    return importlib.reload(exp)


def _code(fn, *more):
    record = project_fingerprints(fn, *more)
    assert record is not None and record["python"] == PY
    return record["code"]


def _resume(exp, wd, **kwargs):
    wf = exp.W()
    wf.run(a=multirun([1, 2]), working_dir=wd, resume=True, **kwargs)
    return wf.to_xarray()


def _module(root, name, src):
    (root / f"{name}.py").write_text(textwrap.dedent(src))
    sys.modules.pop(name, None)
    return importlib.import_module(name)


def _changed(before, fn, **kwargs):
    return changed_project_code(before, project_fingerprints(fn, **kwargs))


# --- what the record covers -----------------------------------------------


def test_the_record_has_the_task_module_and_the_project_files_it_imports(project):
    _, exp = project
    code = _code(exp.W.task)
    assert set(code) == {f"{EXP} (file)", f"{HELPERS} (file)"}


#: Edits the reviews found an object walk missing, each a common pattern: the
#: record must change. ``{v}`` is replaced by the before and after values.
EDITS = {
    "a module-level instance of a project class": (
        "import dataclasses\n\n\n@dataclasses.dataclass(frozen=True)\nclass S:\n    k: int = {v}\n\n\nSETTINGS = S()\n\n\ndef task(a):\n    return a * SETTINGS.k\n",
        2,
        20,
    ),
    "a constant computed from another": (
        "BASE = {v}\nSCALE = BASE * 10\n\n\ndef task(a):\n    return a * SCALE\n",
        2,
        20,
    ),
    "a call moved across a try": (
        "{v}\n\n\ndef task(a):\n    return f()\n",
        "def f():\n    try:\n        a()\n        b()\n    except ValueError:\n        return 2\n    return 3",
        "def f():\n    a()\n    try:\n        b()\n    except ValueError:\n        return 2\n    return 3",
    ),
    "a positional-only signature": (
        "{v}\n    return x\n\n\ndef task(a):\n    return helper(a)\n",
        "def helper(x):",
        "def helper(x, /):",
    ),
    "a cached method": (
        "import functools\n\n\nclass Sc:\n    @functools.cache\n    def scale(self, x):\n        return x * {v}\n\n\ndef task(a):\n    return Sc().scale(a)\n",
        2,
        20,
    ),
    "a method behind an installed decorator": (
        "import contextlib\n\n\n@contextlib.contextmanager\ndef quiet():\n    yield\n\n\nclass Sc:\n    @quiet()\n    def score(self, x):\n        return x * {v}\n\n\ndef task(a):\n    return Sc().score(a)\n",
        2,
        3,
    ),
    "a metaclass": (
        "class Meta(type):\n    def __call__(cls, x):\n        return x * {v}\n\n\nclass Model(metaclass=Meta):\n    pass\n\n\ndef task(a):\n    return Model(a)\n",
        2,
        20,
    ),
    "a callback bound in a partial": (
        "import functools\n\n\ndef transform(x):\n    return x * {v}\n\n\ndef apply(x, transform):\n    return transform(x)\n\n\nscale = functools.partial(apply, transform=transform)\n\n\ndef task(a):\n    return scale(a)\n",
        2,
        20,
    ),
    "a cached function in a registry": (
        "import functools\n\n\ndef transform(x):\n    return x * {v}\n\n\nCALLBACKS = [functools.cache(transform)]\n\n\ndef task(a):\n    return CALLBACKS[0](a)\n",
        2,
        20,
    ),
    "a registry filled by a decorator": (
        "REGISTRY = {{}}\n\n\ndef register(name):\n    def deco(cls):\n        REGISTRY[name] = cls\n        return cls\n\n    return deco\n\n\n@register('cnn')\nclass CNN:\n    def forward(self, x):\n        return x * {v}\n\n\ndef task(a):\n    return REGISTRY['cnn']().forward(a)\n",
        2,
        3,
    ),
    "a singleton's method": (
        "class Scorer:\n    def score(self, x):\n        return x * {v}\n\n\nSCORER = Scorer()\n\n\ndef task(a):\n    return SCORER.score(a)\n",
        2,
        3,
    ),
    "a class defined inside a try": (
        "try:\n\n    class Settings:\n        LR = {v}\nexcept Exception:\n    pass\n\n\ndef task(a):\n    return a * Settings.LR\n",
        0.1,
        0.9,
    ),
    "a list literal": (
        "LAYERS = [{v}, 128]\n\n\ndef task(a):\n    return a * LAYERS[0]\n",
        64,
        256,
    ),
    "a default argument": (
        "def helper(x, k={v}):\n    return x * k\n\n\ndef task(a):\n    return helper(a)\n",
        2,
        3,
    ),
}

#: Not edits: the record must stay as it was.
NOT_EDITS = {
    "a comment": (
        "def helper(x):\n    return x{v}\n\n\ndef task(a):\n    return helper(a)\n",
        "",
        "  # a note",
    ),
    "a docstring": (
        "def helper(x):\n    {v}\n    return x\n\n\ndef task(a):\n    return helper(a)\n",
        '"""Old."""',
        '"""New."""',
    ),
    "formatting": (
        "def helper(x, y):\n    return {v}\n\n\ndef task(a):\n    return helper(a, a)\n",
        "max(x, y)",
        "max(\n        x,\n        y,\n    )",
    ),
    "a value computed on import": (
        "import os\nimport time\n\nSTAMP = time.time_ns()\nDEVICE = 'cuda' if os.environ.get('CUDA_VISIBLE_DEVICES') else 'cpu'\n\n\ndef helper(x, stamp=time.time_ns()):\n    return x\n\n\ndef task(a):\n    return helper(a)\n# {v}\n",
        "",
        "",
    ),
    "the main block": (
        "def task(a):\n    return a\n\n\nif __name__ == '__main__':\n    print({v})\n",
        1,
        2,
    ),
}


def _edited(root, name, src, before, after):
    """The record of ``src`` with ``{v}`` as ``before``, then as ``after``,
    each loaded as a fresh process would."""
    records = []
    for v in (before, after):
        mod = _module(root, name, src.format(v=v))
        records.append(project_fingerprints(mod.task))
    sys.modules.pop(name, None)
    return changed_project_code(*records)


@pytest.mark.parametrize("case", sorted(EDITS))
def test_an_edit_to_project_code_is_seen(project, case):
    root, _ = project
    assert _edited(root, "mushin_t_edit", *EDITS[case]) == ["mushin_t_edit (file)"]


@pytest.mark.parametrize("case", sorted(NOT_EDITS))
def test_what_is_not_code_is_not_an_edit(project, case):
    root, _ = project
    assert _edited(root, "mushin_t_noedit", *NOT_EDITS[case]) == []


def test_imports_inside_functions_are_followed_at_any_depth(project):
    """A helper imported inside the task, and the constant it imports from a
    third file: both files are in the record, whether or not they are loaded."""
    root, _ = project
    (root / "mushin_t_deep2.py").write_text("SCALE = 2\n")
    (root / "mushin_t_deep1.py").write_text(
        "from mushin_t_deep2 import SCALE\n\n\ndef scale(x):\n    return x * SCALE\n"
    )
    mod = _module(
        root,
        "mushin_t_deep0",
        "def task(a):\n    from mushin_t_deep1 import scale\n\n    return scale(a)\n",
    )
    try:
        code = project_fingerprints(mod.task)["code"]
        assert {"mushin_t_deep1 (file)", "mushin_t_deep2 (file)"} <= set(code)
        assert "mushin_t_deep1" not in sys.modules  # nothing was imported
    finally:
        for n in ("mushin_t_deep0", "mushin_t_deep1", "mushin_t_deep2"):
            sys.modules.pop(n, None)


def test_a_module_imported_only_by_a_main_block_is_not_followed(project):
    root, _ = project
    (root / "mushin_t_plot.py").write_text("def plot():\n    pass\n")
    mod = _module(
        root,
        "mushin_t_mainonly",
        "def task(a):\n    return a\n\n\nif __name__ == '__main__':\n    import mushin_t_plot\n",
    )
    try:
        assert "mushin_t_plot (file)" not in project_fingerprints(mod.task)["code"]
    finally:
        sys.modules.pop("mushin_t_mainonly", None)


def test_a_secret_in_a_project_file_does_not_reach_its_hash(project):
    root, _ = project
    src = "API_KEY = {v!r}\n\n\ndef task(a):\n    return a\n"
    assert _edited(root, "mushin_t_secretfile", src, "hunter2", "correct-horse") == []


def test_the_record_is_the_same_in_every_process(tmp_path):
    """Nothing in the record depends on the process: string hashing, the
    clock, the environment."""
    (tmp_path / "mushin_t_seeded.py").write_text(
        textwrap.dedent("""
            import time

            NAMES = frozenset({"alpha", "beta", "gamma", "delta"})
            STAMP = time.time_ns()


            def task(a):
                return dict(v=float(a in {"x", "y", "z", "w", "q"} or a in NAMES))
            """)
    )
    probe = (
        "import json, sys; sys.path.insert(0, sys.argv[1]); "
        "import mushin_t_seeded as m; from mushin._resume import project_fingerprints; "
        "print(json.dumps(project_fingerprints(m.task)))"
    )
    outs = {
        subprocess.run(
            [sys.executable, "-c", probe, str(tmp_path)],
            capture_output=True,
            text=True,
            check=True,
            env={
                **os.environ,
                "PYTHONHASHSEED": seed,
                "PYTHONDONTWRITEBYTECODE": "1",
                "CUDA_VISIBLE_DEVICES": seed,
            },
        ).stdout
        for seed in ("0", "1", "2", "3")
    }
    assert len(outs) == 1, outs


def test_a_notebook_task_records_the_project_modules_it_uses(project):
    """A task defined where there is no file: the project modules its
    namespace uses are in the record; its own cells are not compared."""
    nb = types.ModuleType("mushin_t_notebook")  # a kernel's __main__ has no file
    sys.modules[nb.__name__] = nb
    try:
        exec(
            f"import {HELPERS} as helpers\n\n\ndef task(a):\n    return helpers.scale(a)\n",
            nb.__dict__,
        )
        assert f"{HELPERS} (file)" in project_fingerprints(nb.task)["code"]
    finally:
        sys.modules.pop(nb.__name__, None)


# --- what is foreign ----------------------------------------------------------


def test_the_standard_library_mushin_and_installed_packages_are_foreign():
    from mushin._resume import _foreign_roots, _is_foreign_module

    assert _is_foreign_module("json")
    assert _is_foreign_module("numpy")
    assert _is_foreign_module("mushin.workflows")
    # The bare prefix is never a root: on Windows site.getsitepackages() lists
    # it, and a project kept under it would otherwise go unguarded.
    assert pathlib.Path(sys.prefix).resolve() not in _foreign_roots()


def test_an_installed_but_editable_distribution_is_the_project(monkeypatch):
    """A non-editable distribution on PYTHONPATH (a cluster's module load) is
    an installed package; an editable install is the project itself."""
    import mushin._resume as r

    monkeypatch.setattr(
        r, "_installed_roots", lambda: {"clusterlib": (pathlib.Path("/x"),)}
    )
    r._foreign_by_path.cache_clear()
    try:
        assert r._is_foreign_module("clusterlib.ops")
        assert not r._is_foreign_module("mylab_project.models")
    finally:
        r._foreign_by_path.cache_clear()


def test_a_legacy_develop_install_is_the_project(tmp_path, monkeypatch):
    """`setup.py develop` leaves an .egg-info in the project's own tree; its
    package must not be taken for an installed one."""
    import importlib.metadata as md

    import mushin._resume as r

    tree = tmp_path / "checkout"
    (tree / "devpkg").mkdir(parents=True)
    (tree / "devpkg" / "__init__.py").write_text("")
    egg = tree / "devpkg.egg-info"
    egg.mkdir()
    (egg / "PKG-INFO").write_text(
        "Metadata-Version: 2.1\nName: devpkg\nVersion: 0.0.1\n"
    )
    (egg / "top_level.txt").write_text("devpkg\n")
    site = tmp_path / "site-packages"
    (site / "wheelpkg").mkdir(parents=True)
    dist_info = site / "wheelpkg-1.0.dist-info"
    dist_info.mkdir()
    (dist_info / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: wheelpkg\nVersion: 1.0\n"
    )
    (dist_info / "top_level.txt").write_text("wheelpkg\n")
    dists = [md.PathDistribution(egg), md.PathDistribution(dist_info)]
    monkeypatch.setattr(md, "distributions", lambda: iter(dists))
    r._installed_roots.cache_clear()
    try:
        roots = r._installed_roots()
        assert "devpkg" not in roots
        assert roots["wheelpkg"] == (site.resolve(),)
    finally:
        r._installed_roots.cache_clear()
        r._foreign_by_path.cache_clear()


def test_a_project_file_is_the_project_even_if_an_installed_name_matches(
    project, monkeypatch
):
    """A checkout's `utils` next to an installed wheel that ships a top-level
    `utils`: the loaded module's file decides."""
    import mushin._resume as r

    root, exp = project
    elsewhere = root.parent / "site"
    elsewhere.mkdir()
    monkeypatch.setattr(
        r, "_installed_roots", lambda: {EXP: (elsewhere.resolve(),), "numpy": ()}
    )
    r._foreign_by_path.cache_clear()
    try:
        assert not r._is_foreign_module(EXP)  # its file is in the project
        monkeypatch.setattr(r, "_installed_roots", lambda: {EXP: (root.resolve(),)})
        r._foreign_by_path.cache_clear()
        assert r._is_foreign_module(EXP)  # its file is inside the install dir
    finally:
        r._foreign_by_path.cache_clear()


def test_a_project_module_seen_before_import_is_judged_again_after(
    project, monkeypatch
):
    """A module not yet imported is decided by name each time; once imported,
    its file decides, so a checkout's `utils` next to an installed `utils` is
    seen as the project's after it loads."""
    import mushin._resume as r

    root, exp = project
    monkeypatch.setattr(
        r,
        "_installed_roots",
        lambda: {"mushin_t_late": ((root.parent / "elsewhere").resolve(),)},
    )
    r._foreign_by_path.cache_clear()
    try:
        assert r._is_foreign_module("mushin_t_late")  # not imported: by name
        (root / "mushin_t_late.py").write_text("X = 1\n")
        importlib.import_module("mushin_t_late")
        assert not r._is_foreign_module("mushin_t_late")  # now by its file
    finally:
        sys.modules.pop("mushin_t_late", None)
        r._foreign_by_path.cache_clear()


def test_a_lazily_imported_module_named_like_an_installed_one_is_found(
    project, monkeypatch
):
    """A body import of the checkout's module while an installed distribution
    also provides that name: the file decides, before and after the import."""
    import mushin._resume as r

    root, _ = project
    (root / "mushin_t_utils2.py").write_text("def helper(x):\n    return x\n")
    monkeypatch.setattr(
        r,
        "_installed_roots",
        lambda: {"mushin_t_utils2": ((root.parent / "elsewhere").resolve(),)},
    )
    r._foreign_by_path.cache_clear()
    mod = _module(
        root,
        "mushin_t_lazyutil",
        "def task(a):\n    from mushin_t_utils2 import helper\n\n    return dict(v=helper(a))\n",
    )
    try:
        assert "mushin_t_utils2 (file)" in project_fingerprints(mod.task)["code"]
    finally:
        for n in ("mushin_t_utils2", "mushin_t_lazyutil"):
            sys.modules.pop(n, None)
        r._foreign_by_path.cache_clear()


def test_changed_project_code_does_not_compare_across_python_versions():
    a = {"python": "3.11", "code": {"m:f": "1"}}
    assert changed_project_code(a, {"python": "3.11", "code": {"m:f": "2"}}) == ["m:f"]
    assert changed_project_code(a, {"python": "3.12", "code": {"m:f": "2"}}) is None
    assert changed_project_code(None, a) is None  # a sidecar from before the field
    assert changed_project_code(a, None) is None  # not fingerprinted this run


# --- resume -----------------------------------------------------------------


def test_resume_reruns_when_a_helper_changes_and_names_it(project):
    root, exp = project
    wd = str(root / "sweep")
    exp.W().run(a=multirun([1, 2]), working_dir=wd)
    _write(root, scale=100)
    exp = _reload(exp)
    with pytest.warns(
        UserWarning, match=rf"code the task reaches \({HELPERS} \(file\)\)"
    ) as rec:
        ds = _resume(exp, wd)
    assert float(ds["v"].sel(a=2)) == 200.0  # the edited helper ran
    msgs = [str(w.message) for w in rec if "fingerprint mismatch" in str(w.message)]
    assert msgs and all("task code" not in m for m in msgs)  # only the helper moved


def test_resume_reruns_when_a_constant_in_the_task_module_changes(project):
    root, exp = project
    wd = str(root / "sweep")
    exp.W().run(a=multirun([1, 2]), working_dir=wd)
    _write(root, offset=1)  # the task body is unchanged, its module is not
    exp = _reload(exp)
    with pytest.warns(UserWarning, match=rf"code the task reaches \({EXP} \(file\)\)"):
        ds = _resume(exp, wd)
    assert float(ds["v"].sel(a=2)) == 21.0


def test_resume_reuses_when_project_code_is_unchanged(project):
    root, exp = project
    wd = str(root / "sweep")
    exp.W().run(a=multirun([1, 2]), working_dir=wd)
    _write(root, comment="# only a comment")
    exp = _reload(exp)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any resume warning fails
        ds = _resume(exp, wd)
    assert float(ds["v"].sel(a=2)) == 20.0


def test_an_edit_this_process_has_not_loaded_is_recorded_as_stale(project):
    """The file was edited but the process still runs the old code: the cells
    it computes are recorded as stale, never as the new code's, and re-run
    once the module is reloaded."""
    from mushin._resume import _STALE

    root, exp = project
    wd = root / "sweep"
    exp.W().run(a=multirun([1, 2]), working_dir=str(wd))
    _write(root, scale=100)  # on disk only
    with pytest.warns(UserWarning, match="changed on disk after this process"):
        ds = _resume(exp, str(wd))
    assert float(ds["v"].sel(a=2)) == 20.0  # the old code ran
    for status in wd.rglob(STATUS_FILE):
        code = json.loads(status.read_text())["code_deps"]["code"]
        assert code[f"{HELPERS} (file)"] == _STALE
    exp = _reload(exp)  # now the process runs the edit
    with pytest.warns(UserWarning, match="fingerprint mismatch"):
        ds = _resume(exp, str(wd))
    assert float(ds["v"].sel(a=2)) == 200.0


def test_a_script_constant_edited_while_it_runs_is_stale(tmp_path):
    """A parallel worker unpickles the driver's old values while the file
    already says otherwise: the record must not describe the edited file."""
    from mushin._resume import _STALE

    script = tmp_path / "mushin_t_live.py"
    src = "OFFSET = {v}\n\n\ndef task(a):\n    return a + OFFSET\n"
    script.write_text(src.format(v=0))
    ns = {"__name__": "__main__", "__file__": str(script)}
    # Compiled with this module's __future__ flags, as IPython's %run would
    # with its own: compile flags are not an edit.
    exec(compile(script.read_bytes(), str(script), "exec"), ns)
    key = "mushin_t_live (script)"
    assert project_fingerprints(ns["task"])["code"][key] != _STALE
    script.write_text(src.format(v=5))  # edited while the sweep runs
    with pytest.warns(UserWarning, match="changed on disk after this process"):
        assert project_fingerprints(ns["task"])["code"][key] == _STALE


def test_a_function_edited_before_the_first_record_is_stale(project):
    """Imported, then edited on disk, then recorded for the first time: the
    loaded function no longer compiles from the file."""
    from mushin._resume import _STALE

    root, exp = project
    helpers = root / f"{HELPERS}.py"
    text = helpers.read_text()  # the same lines, a different body
    helpers.write_text(text.replace("* SCALE", "* SCALE * 2"))
    with pytest.warns(UserWarning, match="changed on disk after this process"):
        assert _code(exp.W.task)[f"{HELPERS} (file)"] == _STALE


def test_a_module_an_import_hook_rewrote_is_compared_by_its_file(project):
    """pytest's assertion rewriting (or a type-checking hook) loads code that
    never equals a plain compile: such a module is never taken for stale."""
    from mushin._resume import _file_entry, _loaded_stale

    root, _ = project
    assert not _loaded_stale(HELPERS, _file_entry(HELPERS)[1])
    _write(root, scale=3)  # edited, not reloaded: stale for the standard loader
    assert _loaded_stale(HELPERS, _file_entry(HELPERS)[1])
    sys.modules[HELPERS].__loader__ = object()  # as an import hook's loader
    assert not _loaded_stale(HELPERS, _file_entry(HELPERS)[1])


def test_sidecar_records_the_project_code(project):
    root, exp = project
    wd = root / "sweep"
    exp.W().run(a=multirun([1]), working_dir=str(wd))
    (status,) = wd.rglob(STATUS_FILE)
    assert json.loads(status.read_text())["code_deps"] == project_fingerprints(
        exp.W.task
    )


def test_resume_reuses_legacy_sidecars_without_project_code(project):
    """A sweep recorded before the field cannot be checked, which is not a
    reason to re-run: it resumes like one whose code did not change."""
    root, exp = project
    wd = root / "sweep"
    exp.W().run(a=multirun([1, 2]), working_dir=str(wd))
    for status in wd.rglob(STATUS_FILE):
        d = json.loads(status.read_text())
        d.pop("code_deps")
        status.write_text(json.dumps(d))
    _write(root, scale=100)
    exp = _reload(exp)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ds = _resume(exp, str(wd))
    assert float(ds["v"].sel(a=2)) == 20.0  # reused, as before the field


def test_a_reused_cell_keeps_its_record_when_this_run_has_none(project, monkeypatch):
    """When the project code cannot be fingerprinted this run, a reused cell's
    sidecar keeps the record it had, so the next resume still compares."""
    import mushin._resume as r

    root, exp = project
    wd = root / "sweep"
    exp.W().run(a=multirun([1, 2]), working_dir=str(wd))
    recorded = project_fingerprints(exp.W.task)
    monkeypatch.setattr(r, "project_fingerprints", lambda *a, **k: None)
    _resume(exp, str(wd))
    for status in wd.rglob(STATUS_FILE):
        assert json.loads(status.read_text())["code_deps"] == recorded


def test_the_task_body_hash_is_the_text_hash_main_recorded(project):
    """code_fingerprint is unchanged, so sweeps and cache entries recorded
    before this change keep their keys."""
    import inspect

    _, exp = project
    src = inspect.getsource(exp.W.task)
    assert code_fingerprint(exp.W.task) == hashlib.sha256(src.encode()).hexdigest()[:16]


def test_a_record_that_cannot_be_taken_warns_once(project, monkeypatch):
    import mushin._resume as r

    _, exp = project

    def boom(*a, **k):
        raise RuntimeError("an exotic object")

    monkeypatch.setattr(r, "_root_entries", boom)
    monkeypatch.setattr(r, "_FAILURE_WARNED", [])
    with pytest.warns(UserWarning, match="could not be fingerprinted"):
        assert project_fingerprints(exp.W.task) is None
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert project_fingerprints(exp.W.task) is None  # once per process


def test_too_many_project_files_warn_once(project, monkeypatch):
    import mushin._resume as r

    _, exp = project
    monkeypatch.setattr(r, "_FILES_MAX", 1)
    monkeypatch.setattr(r, "_TRUNCATION_WARNED", [])
    with pytest.warns(UserWarning, match="more project files than"):
        project_fingerprints(exp.W.task)


# --- cache_dir --------------------------------------------------------------


def _counting_project(root):
    """A task that appends to a file each time it computes, so a cache hit is
    observable without changing any code."""
    calls = root / "calls.txt"
    (root / f"{EXP}.py").write_text(
        textwrap.dedent(f"""
            import {HELPERS} as helpers
            from mushin.workflows import MultiRunMetricsWorkflow


            class W(MultiRunMetricsWorkflow):
                @staticmethod
                def task(a):
                    with open({str(calls)!r}, "a") as f:
                        f.write("x")
                    return dict(v=helpers.scale(a))
            """)
    )
    return calls


def test_cache_dir_misses_after_a_helper_edit_and_hits_when_it_is_undone(project):
    root, exp = project
    calls = _counting_project(root)
    exp = importlib.reload(exp)
    cache = str(root / "cache")

    def run(wd):
        wf = exp.W()
        wf.run(a=multirun([1, 2]), working_dir=str(root / wd), cache_dir=cache)
        return float(wf.to_xarray()["v"].sel(a=2))

    assert run("s1") == 20.0 and calls.read_text() == "xx"
    (root / f"{HELPERS}.py").write_text(
        "SCALE = 100\n\n\ndef scale(x):\n    return float(x) * SCALE\n"
    )
    importlib.reload(sys.modules[HELPERS])
    assert run("s2") == 200.0 and calls.read_text() == "xxxx"  # a miss
    _write(root)  # back to SCALE = 10 (rewrites the task module too)
    calls = _counting_project(root)
    importlib.reload(sys.modules[HELPERS])
    exp = importlib.reload(exp)
    assert run("s3") == 20.0 and calls.read_text() == "xxxx"  # the s1 entry: no compute


def test_cache_key_keeps_main_two_part_form_without_project_code():
    """A cell that reaches no project code keeps main's key, so its entries from
    before this change are still found; one that does gets another; one whose
    project code could not be fingerprinted does not use the cache."""
    from mushin.workflows import _TaskRunner

    runner = _TaskRunner(
        task=None,
        pre_task=None,
        swept_names=(),
        base_provenance=None,
        on_error="raise",
        inject_resume=False,
        prior_manifest=None,
        code_hash="c0de",
        cache_dir="cache",
        code_roots=(len,),  # a run always walks from its task
    )
    two_part = hashlib.sha256(b"h:c0de").hexdigest()[:32]
    assert runner._cache_key("h", {"python": PY, "code": {}}) == two_part
    assert runner._cache_key("h", {"python": PY, "code": {"m:f": "1"}}) != two_part
    assert runner._cache_key("h", None) is None


def test_a_runner_with_no_code_roots_keys_the_cache_as_before(tmp_path):
    """A runner pickled by an earlier mushin walks nothing (no roots): it
    keeps the two-part cache key that version used, rather than no cache."""
    import hashlib

    from hydra_zen import zen

    from mushin.workflows import _TaskRunner

    runner = _TaskRunner(
        task=zen(lambda a: dict(v=a)),
        pre_task=zen(lambda: None),
        swept_names=(),
        base_provenance={},
        on_error="raise",
        inject_resume=False,
        prior_manifest=None,
        cache_dir=str(tmp_path),
        code_hash="c0de",
    )
    assert (
        runner._cache_key("cfg", None) == (hashlib.sha256(b"cfg:c0de").hexdigest()[:32])
    )
    runner.code_roots = (object(),)
    assert runner._cache_key("cfg", None) is None  # a walk that failed


def test_a_partial_task_over_data_is_never_cached():
    """Main never cached a partial task. One binding only values a key can
    tell apart (numbers, a list of them, a function) is cached; one binding an
    object of a type the key cannot see inside (an array) is not."""
    from mushin._resume import partial_task_cacheable

    def train(a, layers, fn=None):
        return a

    assert partial_task_cacheable(train)
    assert partial_task_cacheable(functools.partial(train, layers=[64, 128]))
    assert partial_task_cacheable(functools.partial(train, layers=(1,), fn=len))
    assert not partial_task_cacheable(functools.partial(train, layers=object()))


def test_partials_over_different_lists_record_different_bound_arguments():
    from mushin._resume import _partial_digest

    def train(a, layers):
        return a

    a = _partial_digest(functools.partial(train, layers=[64, 128]))
    b = _partial_digest(functools.partial(train, layers=[64, 256]))
    assert a != b


# --- a task defined in the script that runs it --------------------------------

SCRIPT = """
import sys

import mushin_t_helpers as helpers
from mushin import multirun
from mushin.workflows import MultiRunMetricsWorkflow

OFFSET = {offset}


class W(MultiRunMetricsWorkflow):
    @staticmethod
    def task(a):
        with open({calls!r}, "a") as f:
            f.write("x")
        return dict(v=helpers.scale(a) + OFFSET + bonus(a))


if __name__ == "__main__":

    def bonus(a):
        return {bonus}

    W().run(a=multirun([1, 2]), working_dir={wd!r}, resume="resume" in sys.argv)
    {tail}
"""


def _script_runner(root, script):
    def run(*args):
        env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
        return subprocess.run(
            [sys.executable, str(script), *args],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
            env=env,
        ).stderr

    return run


def test_a_task_in_the_driver_script_survives_edits_to_the_run_block(tmp_path):
    """The resilience loop edits the script that defines the task (a flag, a
    print, a comment) and resumes: that reuses the cells. A constant the task
    reads, or a helper defined in the main block, is an edit."""
    root = tmp_path / "script"
    root.mkdir()
    _write(root)
    calls = root / "calls.txt"
    script = root / "exp_script.py"

    def write_script(offset=0, bonus=0, tail="pass"):
        script.write_text(
            SCRIPT.format(
                offset=offset,
                bonus=bonus,
                calls=str(calls),
                wd=str(root / "wd"),
                tail=tail,
            )
        )

    run = _script_runner(root, script)
    write_script()
    run()
    assert calls.read_text() == "xx"
    write_script(tail='print("done")  # an edit to the run block')
    script.write_text(script.read_text() + "\n# and a comment\n")
    err = run("resume")
    assert "fingerprint mismatch" not in err, err
    assert calls.read_text() == "xx"  # both cells reused
    write_script(offset=1)
    err = run("resume")
    assert "code the task reaches (exp_script (script))" in err, err
    assert calls.read_text() == "xxxx"
    write_script(offset=1, bonus=5)
    err = run("resume")
    assert "code the task reaches (exp_script (script))" in err, err
    assert calls.read_text() == "xxxxxx"


UNGUARDED = """
import mushin_t_helpers as helpers
from mushin import multirun
from mushin.workflows import MultiRunMetricsWorkflow

SCALE = {scale}
GRID = multirun({grid})


class W(MultiRunMetricsWorkflow):
    @staticmethod
    def task(a):
        with open({calls!r}, "a") as f:
            f.write("x")
        return dict(v=helpers.scale(a) * SCALE)


wf = W()
wf.run(a=GRID, working_dir={wd!r}, resume={resume})
print("done", {tail!r})
"""


def test_an_unguarded_script_can_extend_its_grid(tmp_path):
    """A script with no main block: its run call, its grid and its prints are
    not code the task runs, so extending the grid reuses the completed
    cells; an edit to a constant the task reads is an edit."""
    root = tmp_path / "ug"
    root.mkdir()
    _write(root)
    calls = root / "calls.txt"
    script = root / "ug_script.py"

    def write(scale=1, grid="[1, 2]", resume=False, tail=""):
        script.write_text(
            UNGUARDED.format(
                scale=scale,
                grid=grid,
                calls=str(calls),
                wd=str(root / "wd"),
                resume=resume,
                tail=tail,
            )
        )

    run = _script_runner(root, script)
    write()
    run()
    assert calls.read_text() == "xx"
    write(grid="[1, 2, 3]", resume=True, tail="more")
    err = run()
    assert "fingerprint mismatch" not in err, err
    assert calls.read_text() == "xxx"  # only the new cell ran
    write(scale=2, grid="[1, 2, 3]", resume=True)
    err = run()
    assert "code the task reaches (ug_script (script))" in err, err
    assert calls.read_text() == "xxxxxx"


# --- Study ------------------------------------------------------------------


def _train_module(root, name):
    (root / f"{name}.py").write_text(
        textwrap.dedent(f"""
            from pathlib import Path

            import {HELPERS} as helpers


            def train(seed):
                p = Path(f"ckpt_{{seed}}.bin")
                p.write_text(str(helpers.scale(seed + 1)))
                return str(p.resolve())
            """)
    )
    sys.modules.pop(name, None)
    return importlib.import_module(name)


@pytest.fixture
def in_cwd(tmp_path):
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    old = os.getcwd()
    os.chdir(cwd)
    yield cwd
    os.chdir(old)


def test_study_reruns_when_a_method_helper_changes(project, in_cwd):
    from mushin.study._sweep import run_training_sweep

    root, _ = project
    train_mod = _train_module(root, "mushin_t_train")
    try:
        wd = str(root / "study")
        run_training_sweep(
            {"m": train_mod.train}, seeds=[0], ckpt_dir=root / "ck", working_dir=wd
        )
        _write(root, scale=100)
        importlib.reload(sys.modules[HELPERS])
        with pytest.warns(
            UserWarning, match=rf"code the task reaches \({HELPERS} \(file\)\)"
        ):
            ckpts = run_training_sweep(
                {"m": train_mod.train},
                seeds=[0],
                ckpt_dir=root / "ck",
                working_dir=wd,
                resume=True,
            )
        assert pathlib.Path(ckpts["m"][0]).read_text() == "100.0"
    finally:
        sys.modules.pop("mushin_t_train", None)


def test_study_methods_fingerprint_is_the_one_main_recorded(project, in_cwd):
    """The methods fingerprint is a config value: if it moved, every Study sweep
    recorded before this change would re-train. Pin main's formula. Study's
    own code (mushin) is not in the record, so a checkpoint path is not."""
    import inspect

    import yaml

    from mushin.study._sweep import run_training_sweep

    root, _ = project
    train_mod = _train_module(root, "mushin_t_train2")
    try:
        wd = root / "study2"
        run_training_sweep(
            {"m": train_mod.train},
            seeds=[0],
            ckpt_dir=root / "ck2",
            working_dir=str(wd),
        )
        text_hash = hashlib.sha256(
            inspect.getsource(train_mod.train).encode()
        ).hexdigest()[:16]
        expected = hashlib.sha256(f"m:{text_hash}".encode()).hexdigest()[:16]
        (config,) = wd.rglob(".hydra/config.yaml")
        assert yaml.safe_load(config.read_text())["methods_fingerprint"] == expected
        (status,) = wd.rglob(STATUS_FILE)
        code = json.loads(status.read_text())["code_deps"]["code"]
        assert f"{HELPERS} (file)" in code  # the training function's helper
        assert not any(k.startswith("mushin") and "study" in k for k in code)
    finally:
        sys.modules.pop("mushin_t_train2", None)


# --- pre_task and the config's targets -----------------------------------------


def test_pre_task_and_config_targets_are_roots(project):
    """pre_task's module, and the modules of the classes the config names as
    `_target_` (hydra-zen's builds), are code a cell runs."""
    from mushin._resume import cell_project_record

    root, exp = project
    (root / "mushin_t_models.py").write_text(
        "class Model:\n    def forward(self, a):\n        return a * 2\n"
    )
    pre = _module(
        root, "mushin_t_pre", "OFFSET = 0\n\n\ndef pre_task(seed):\n    return seed\n"
    )
    cfg = {"model": {"_target_": "mushin_t_models.Model"}, "a": 1}
    try:
        record = cell_project_record("tok-a", (exp.W.task,), (pre.pre_task,), cfg)
        assert {"mushin_t_pre (file)", "mushin_t_models (file)"} <= set(record["code"])
        assert "mushin_t_models" not in sys.modules  # found without importing
    finally:
        sys.modules.pop("mushin_t_pre", None)


def test_hydra_zen_routes_classes_through_its_own_functions():
    """just(Cls) names Cls under `path` of hydra_zen.funcs.get_obj, and
    builds(..., zen_meta=...) under `_zen_target` of zen_processing."""
    from mushin._resume import _config_targets

    cfg = {
        "m": {"_target_": "hydra_zen.funcs.get_obj", "path": "pkg.models.Net"},
        "n": {
            "_target_": "hydra_zen.funcs.zen_processing",
            "_zen_target": "pkg.models.Head",
        },
    }
    assert {"pkg.models.Net", "pkg.models.Head"} <= set(_config_targets(cfg))


def test_a_driver_script_target_is_not_a_module_to_follow(project, monkeypatch):
    """builds(Net) in a script names __main__.Net; the script is covered as a
    script, not by its whole file. ``__main__`` here is a project script with
    a spec, so without the guard its file would be followed."""
    import importlib.util

    from mushin._resume import _target_module

    root, _ = project
    script = root / "mushin_t_driver.py"
    script.write_text("class Net:\n    pass\n")
    spec = importlib.util.spec_from_file_location("__main__", script)
    main = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(main)
    monkeypatch.setitem(sys.modules, "__main__", main)
    assert _target_module("__main__.Net") is None


# --- parallel launchers -------------------------------------------------------

JOBLIB_SCRIPT = """
import sys

from mushin import multirun
from mushin.workflows import MultiRunMetricsWorkflow

OFFSET = {offset}


def _mark(calls):
    # One file per call: two worker processes appending to one file can
    # lose a write on Windows.
    import os, uuid

    os.makedirs(calls, exist_ok=True)
    open(os.path.join(calls, uuid.uuid4().hex), "w").close()


def helper(a):
    return a + OFFSET


class W(MultiRunMetricsWorkflow):
    @staticmethod
    def task(a):
        _mark({calls!r})
        return dict(v=float(helper(a)))


if __name__ == "__main__":
    launcher = sys.argv[1]
    W().run(
        a=multirun([1, 2]),
        working_dir={wd!r},
        resume="resume" in sys.argv,
        **({{"launcher": "joblib", "launcher_config": {{"n_jobs": 2}}}} if launcher == "joblib" else {{}}),
    )
"""


def _count(calls):
    """How many task calls left a marker in ``calls``."""
    return len(list(calls.iterdir())) if calls.exists() else 0


@pytest.mark.parametrize("second", ["joblib", "basic"])
def test_a_script_task_under_joblib_records_the_code_its_workers_run(tmp_path, second):
    """In a loky worker, sys.modules['__main__'] is loky's own module, while the
    task's functions carry their script's globals: the record names the
    script the same way there, so a joblib resume (or a sequential one after a
    joblib run) sees an edit."""
    pytest.importorskip("hydra_plugins.hydra_joblib_launcher")
    root = tmp_path / "jl"
    root.mkdir()
    calls = root / "calls"
    script = root / "jl_script.py"

    def write(offset):
        script.write_text(
            JOBLIB_SCRIPT.format(offset=offset, calls=str(calls), wd=str(root / "wd"))
        )

    run = _script_runner(root, script)
    write(0)
    run("joblib")
    assert _count(calls) == 2
    err = run(second, "resume")
    assert "fingerprint mismatch" not in err, err
    assert _count(calls) == 2  # unchanged: reused
    write(100)
    err = run(second, "resume")
    assert "code the task reaches (jl_script (script))" in err, err
    assert _count(calls) == 4


# --- a stability matrix over the record's invariants ---------------------------
#
# The same code must give the same record in the driver and in a parallel
# worker, in two fresh processes with a different environment and clock, and
# in one process before and after its cells ran; and an edit to each covered
# name must still be caught. One project holds the cases reviews found
# breaking one of those: a class with no methods and an Enum in the driver
# script, a module global only an uncalled function rebinds, a dataclass field
# default and a function default computed on import, and a warn-once decorator
# that rebinds its closure with `nonlocal`.

MX_LIB = """
import dataclasses
import os
import time


@dataclasses.dataclass
class TrainConfig:
    lr: float = 0.5
    stamp: int = time.time_ns()
    device: str = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"


def log_name(name, stamp=time.time_ns(), device=os.environ.get("CUDA_VISIBLE_DEVICES", "")):
    return name


def once(fn):
    warned = False

    def wrapper(*a):
        nonlocal warned
        if not warned:
            warned = True
        return fn(*a)

    return wrapper


SCALE = {scale}


@once
def scale(x):
    return x * SCALE
"""


MX_SCRIPT = """
import enum
import sys

import mx_lib

from mushin import multirun
from mushin.workflows import MultiRunMetricsWorkflow

DEVICE = "cpu"


def _mark(calls):
    # One file per call: two worker processes appending to one file can
    # lose a write on Windows.
    import os, uuid

    os.makedirs(calls, exist_ok=True)
    open(os.path.join(calls, uuid.uuid4().hex), "w").close()


def configure(dev):
    global DEVICE
    DEVICE = dev


class HP:
    LR = {lr}


class Mode(enum.Enum):
    TRAIN = {mode}


class W(MultiRunMetricsWorkflow):
    @staticmethod
    def task(a):
        _mark({calls!r})
        cfg = mx_lib.TrainConfig()
        mx_lib.log_name("run")
        v = mx_lib.scale(a) * HP.LR + Mode.TRAIN.value + len(DEVICE) + cfg.lr
        return dict(v=float(v))


def run(launcher, resume):
    extra = {{}}
    if launcher == "joblib":
        extra = {{"launcher": "joblib", "launcher_config": {{"n_jobs": 2}}}}
    W().run(a=multirun([1, 2]), working_dir={wd!r}, resume=resume, **extra)


if __name__ == "__main__":
    launcher = sys.argv[1]
    if launcher == "twice":  # one process: a run, then a resume
        run("basic", False)
        run("basic", True)
    else:
        run(launcher, "resume" in sys.argv)
"""


class _Matrix:
    """The matrix project in ``root``: write it, run it in a fresh process."""

    def __init__(self, root):
        self.root = root
        self.calls = root / "calls"
        self.write()

    def write(self, lr=0.1, mode=1, scale=2):
        (self.root / "mx_lib.py").write_text(MX_LIB.format(scale=scale))
        (self.root / "mx_script.py").write_text(
            MX_SCRIPT.format(
                lr=lr, mode=mode, calls=str(self.calls), wd=str(self.root / "wd")
            )
        )

    def run(self, *args, cuda=None):
        env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
        env.pop("CUDA_VISIBLE_DEVICES", None)
        if cuda is not None:
            env["CUDA_VISIBLE_DEVICES"] = cuda
        return subprocess.run(
            [sys.executable, str(self.root / "mx_script.py"), *args],
            cwd=self.root,
            capture_output=True,
            text=True,
            check=True,
            env=env,
        ).stderr


def _needs_joblib(*launchers):
    if "joblib" in launchers:
        pytest.importorskip("hydra_plugins.hydra_joblib_launcher")


@pytest.mark.parametrize(
    "first, second",
    [("joblib", "basic"), ("basic", "joblib"), ("joblib", "joblib")],
)
def test_matrix_unchanged_code_is_reused_across_launchers_and_processes(
    tmp_path, first, second
):
    """Each run is a fresh process with another clock, and the resume has
    another CUDA_VISIBLE_DEVICES: unchanged code reuses every cell."""
    _needs_joblib(first, second)
    mx = _Matrix(tmp_path)
    mx.run(first)
    assert _count(mx.calls) == 2
    err = mx.run(second, "resume", cuda="0")
    assert "fingerprint mismatch" not in err, err
    assert _count(mx.calls) == 2


def test_matrix_unchanged_code_is_reused_in_the_same_process(tmp_path):
    """A run then a resume in one process (a notebook extending its grid)."""
    mx = _Matrix(tmp_path)
    err = mx.run("twice")
    assert "fingerprint mismatch" not in err, err
    assert _count(mx.calls) == 2


@pytest.mark.parametrize(
    "edit, key",
    [
        ({"lr": 0.2}, "mx_script (script)"),
        ({"mode": 2}, "mx_script (script)"),
        ({"scale": 3}, "mx_lib (file)"),
    ],
)
@pytest.mark.parametrize("launcher", ["joblib", "basic"])
def test_matrix_an_edit_is_caught_under_every_launcher(tmp_path, launcher, edit, key):
    _needs_joblib(launcher)
    mx = _Matrix(tmp_path)
    mx.run(launcher)
    mx.write(**edit)
    err = mx.run(launcher, "resume")
    assert f"code the task reaches ({key})" in err, err
    assert _count(mx.calls) == 4
