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


def _train_for_partials(a, layers=None, fn=None):
    """A module-level task for the partial-task tests: its name identifies
    it, as a real task's does."""
    return a


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


def test_changed_project_code_compares_the_keys_both_records_have():
    a = {"python": "3.11", "code": {"m:f": "1", "only_a": "1"}}
    assert changed_project_code(a, {"python": "3.11", "code": {"m:f": "2"}}) == ["m:f"]
    assert changed_project_code(a, {"python": "3.11", "code": {"other": "2"}}) == []
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


def test_an_edit_this_process_has_not_loaded_warns(project):
    """The file was edited but the process still runs the old code. The record
    is the file as it is (under %autoreload, the new code does run), and a
    warning says to reload or restart: this process cannot tell which code
    runs."""
    from mushin._resume import _file_entry

    root, exp = project
    wd = root / "sweep"
    exp.W().run(a=multirun([1, 2]), working_dir=str(wd))
    _write(root, scale=100)  # on disk only
    on_disk = _file_entry(HELPERS)[1]
    with pytest.warns(UserWarning, match="changed on disk after this process"):
        _resume(exp, str(wd))
    for status in wd.rglob(STATUS_FILE):
        code = json.loads(status.read_text())["code_deps"]["code"]
        assert code[f"{HELPERS} (file)"] == on_disk


def test_a_script_edited_while_it_runs_warns(tmp_path):
    """The driver keeps running the script it loaded: a later record after the
    file changed warns. A script run again (new functions, as IPython's %run
    makes) is a new load and does not."""
    from mushin._resume import _script_entry

    path = tmp_path / "mushin_t_live.py"
    src = "OFFSET = {v}\n\n\ndef task(a):\n    return a + OFFSET\n"
    ns = _script_ns(path, src.format(v=0))
    _script_entry(str(path), ns, {"task"}, (ns["task"],))
    path.write_text(src.format(v=5))  # edited while the sweep runs
    with pytest.warns(UserWarning, match="changed on disk after this process"):
        _script_entry(str(path), ns, {"task"}, (ns["task"],))
    exec(compile(path.read_bytes(), str(path), "exec", dont_inherit=True), ns)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _script_entry(str(path), ns, {"task"}, (ns["task"],))


def test_a_comment_or_docstring_edited_after_import_is_not_stale(project):
    """Only an edit to code makes a loaded module stale: a comment line added
    above its functions, or a docstring reworded, leaves its digest as it was."""
    root, exp = project
    before = _code(exp.W.task)[f"{HELPERS} (file)"]
    helpers = root / f"{HELPERS}.py"
    helpers.write_text("# a new first line\n" + helpers.read_text())
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        after = _code(exp.W.task)[f"{HELPERS} (file)"]
    assert after == before


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


def test_a_partial_task_is_never_cached():
    """As on main: a cache key cannot tell one partial's bound data from
    another's, so a partial task never uses the cache. A plain function does."""
    from mushin._resume import partial_task_cacheable

    train = _train_for_partials
    assert partial_task_cacheable(train)
    assert not partial_task_cacheable(functools.partial(train, layers=[64, 128]))
    assert not partial_task_cacheable(functools.partial(train, fn=len))


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


def test_a_driver_script_target_is_recorded_as_the_script(project, monkeypatch):
    """builds(Net) in a script names __main__.Net: the script is recorded the
    way a task defined in it is, not followed as a whole file."""
    import importlib.util

    from mushin._resume import _target_entries

    root, _ = project
    script = root / "mushin_t_driver.py"
    script.write_text("class Net:\n    pass\n")
    spec = importlib.util.spec_from_file_location("__main__", script)
    main = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(main)
    monkeypatch.setitem(sys.modules, "__main__", main)
    seeds, scripts = [], {}
    _target_entries("__main__.Net", seeds, scripts)
    assert seeds == []
    assert scripts[str(script)][1] == {"Net"}


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


# --- the redesign's review: each case in the context it breaks in ---------------

REGISTRY_SCRIPT = """
import sys

from mushin import multirun
from mushin.workflows import MultiRunMetricsWorkflow

MODELS = {{}}


def register(cls):
    MODELS[cls.__name__] = cls
    return cls


@register
class Lin:
    SCALE = {scale}


class W(MultiRunMetricsWorkflow):
    @staticmethod
    def task(a):
        with open({calls!r}, "a") as f:
            f.write("x")
        return dict(v=a * MODELS["Lin"].SCALE)


if __name__ == "__main__":
    W().run(
        a=multirun([1, 2]),
        working_dir=sys.argv[1],
        resume="resume" in sys.argv,
        cache_dir={cache!r} if "cache" in sys.argv else None,
    )
"""


def _registry_project(tmp_path):
    calls = tmp_path / "calls.txt"
    script = tmp_path / "reg_script.py"

    def write(scale):
        script.write_text(
            REGISTRY_SCRIPT.format(
                scale=scale, calls=str(calls), cache=str(tmp_path / "cache")
            )
        )

    return calls, write, _script_runner(tmp_path, script)


def test_a_registry_a_decorator_fills_does_not_make_the_script_stale(tmp_path):
    """`MODELS = {}` filled at import is not a constant edited on disk: edits
    to the script after it are still caught, resume after resume."""
    calls, write, run = _registry_project(tmp_path)
    wd = str(tmp_path / "wd")
    write(1.0)
    err = run(wd)
    assert "changed on disk" not in err, err
    write(5.0)
    err = run(wd, "resume")
    assert "code the task reaches (reg_script (script))" in err, err
    write(7.0)
    err = run(wd, "resume")
    assert "code the task reaches (reg_script (script))" in err, err
    assert calls.read_text() == "xxxxxx"


def test_a_fresh_working_dir_with_a_shared_cache_misses_after_an_edit(tmp_path):
    calls, write, run = _registry_project(tmp_path)
    write(1.0)
    run(str(tmp_path / "wd1"), "cache")
    write(5.0)
    run(str(tmp_path / "wd2"), "cache")
    assert calls.read_text() == "xxxx"  # the edited code was computed


MAIN_BLOCK_TASK = """
import sys

from mushin import multirun
from mushin.workflows import MultiRunMetricsWorkflow

if __name__ == "__main__":
    SCALE = {scale}

    class W(MultiRunMetricsWorkflow):
        @staticmethod
        def task(a):
            with open({calls!r}, "a") as f:
                f.write("x")
            return dict(v=a * SCALE)

    W().run(a=multirun([1, 2]), working_dir={wd!r}, resume="resume" in sys.argv)
"""

MAIN_FUNCTION = """
import sys

from mushin import multirun
from mushin.workflows import MultiRunMetricsWorkflow

SCALE = {scale}


class W(MultiRunMetricsWorkflow):
    @staticmethod
    def task(a):
        with open({calls!r}, "a") as f:
            f.write("x")
        return dict(v=a * SCALE)


def main():
    W().run(a=multirun({grid}), working_dir={wd!r}, resume="resume" in sys.argv)


if __name__ == "__main__":
    main()
"""


@pytest.mark.parametrize("layout", ["main block task", "main function"])
def test_the_script_counts_what_the_task_reads_wherever_it_is(tmp_path, layout):
    """A task defined in the main block reads a main-block constant; a task
    run from `def main()` has its grid there: the constant is code the task
    runs, the grid is not."""
    calls = tmp_path / "calls.txt"
    script = tmp_path / "lay_script.py"
    template = MAIN_BLOCK_TASK if layout == "main block task" else MAIN_FUNCTION

    def write(scale=1.0, grid="[1, 2]"):
        script.write_text(
            template.format(
                scale=scale, grid=grid, calls=str(calls), wd=str(tmp_path / "wd")
            )
        )

    run = _script_runner(tmp_path, script)
    write()
    run()
    if layout == "main function":
        write(grid="[1, 2, 3]")
        err = run("resume")
        assert "fingerprint mismatch" not in err, err
        assert calls.read_text() == "xxx"  # only the new cell ran
    before = len(calls.read_text())
    write(scale=5.0, grid="[1, 2, 3]" if layout == "main function" else "[1, 2]")
    err = run("resume")
    assert "code the task reaches (lay_script (script))" in err, err
    assert len(calls.read_text()) > before


TARGET_SCRIPT = """
import sys

from hydra_zen import builds, make_config

import tgt_mod
from mushin import multirun


class Net:
    def f(self, x):
        return x * {scale}


if __name__ == "__main__":
    tgt_mod.W(eval_task_cfg=make_config(net=builds(Net))).run(
        a=multirun([1, 2]), working_dir={wd!r}, resume="resume" in sys.argv
    )
"""

TARGET_MODULE = """
from mushin.workflows import MultiRunMetricsWorkflow


class W(MultiRunMetricsWorkflow):
    @staticmethod
    def task(net, a):
        with open({calls!r}, "a") as f:
            f.write("x")
        return dict(v=net.f(a))
"""


def test_a_target_class_in_the_driver_script_is_covered(tmp_path):
    calls = tmp_path / "calls.txt"
    (tmp_path / "tgt_mod.py").write_text(TARGET_MODULE.format(calls=str(calls)))
    script = tmp_path / "tgt_script.py"

    def write(scale):
        script.write_text(TARGET_SCRIPT.format(scale=scale, wd=str(tmp_path / "wd")))

    run = _script_runner(tmp_path, script)
    write(1.0)
    run()
    write(5.0)
    err = run("resume")
    assert "code the task reaches (tgt_script:Net (script))" in err, err
    assert calls.read_text() == "xxxx"


def test_settings_named_like_secrets_are_compared(project):
    """`EOS_TOKEN = "</s>"` is a tokenizer setting: a project file is hashed
    as it is, strings included."""
    root, _ = project
    src = "EOS_TOKEN = {v!r}\n\n\ndef task(a):\n    return a\n"
    assert _edited(root, "mushin_t_eos", src, "</s>", "<|endoftext|>") != []


def test_a_record_from_another_python_version_is_compared():
    """File digests read alike on every supported Python, so an edit made
    around an interpreter upgrade is still seen."""
    a = {"python": "3.11", "code": {"m (file)": "1"}}
    assert changed_project_code(a, {"python": "3.12", "code": {"m (file)": "2"}}) == [
        "m (file)"
    ]


LAZY_PROXY = """
RESOLVED = []


class Proxy:
    \"\"\"Resolves its target (a large dataset) the first time any attribute
    is read: it notes each read here.\"\"\"

    def __getattr__(self, name):
        RESOLVED.append(name)
        raise AttributeError(name)

    @property
    def __class__(self):
        RESOLVED.append("__class__")
        return Proxy

    @property
    def __wrapped__(self):
        RESOLVED.append("__wrapped__")
        return None


DATA = Proxy()


def task(a):
    return a
"""


def test_the_record_never_resolves_a_lazy_object():
    """Taking the record runs in every cell's process, resume or not: it must
    not read attributes of the objects a notebook's namespace holds."""
    nb = types.ModuleType("mushin_t_lazynb")  # a notebook: no file
    exec(LAZY_PROXY, nb.__dict__)
    assert project_fingerprints(nb.task) is not None
    assert nb.RESOLVED == []


def test_a_project_module_named_like_the_standard_library_is_the_project(project):
    """A checkout's `code/` package: its file, not its name, decides."""
    root, _ = project
    pkg = root / "code"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "train.py").write_text("def task(a):\n    return a\n")
    saved = {
        n: sys.modules.pop(n) for n in list(sys.modules) if n.split(".")[0] == "code"
    }
    try:
        mod = importlib.import_module("code.train")
        assert mod.__file__.startswith(str(root))  # the checkout's, not the stdlib's
        assert "code.train (file)" in project_fingerprints(mod.task)["code"]
    finally:
        for n in [n for n in sys.modules if n.split(".")[0] == "code"]:
            sys.modules.pop(n, None)
        sys.modules.update(saved)


def _digest_of_script(tmp_path, name, text, roots):
    from mushin._resume import _script_digest

    path = tmp_path / f"{name}.py"
    path.write_text(text)
    return _script_digest(path, frozenset(roots))[0]


def test_a_script_counts_what_the_task_reads_not_its_parameters(tmp_path):
    """`a = [1, 2]` feeds the run call; the task's parameter `a` is a local
    name, not a read of it. A run result and what uses it do not count."""
    text = (
        "SCALE = 2\na = {grid}\n\n\ndef task(a):\n    return a * SCALE\n\n\n"
        "ds = run(a=a)\nsummary = ds.mean()\nprint(summary)\n"
    )
    one = _digest_of_script(tmp_path, "par1", text.format(grid="[1, 2]"), {"task"})
    two = _digest_of_script(tmp_path, "par2", text.format(grid="[1, 2, 3]"), {"task"})
    assert one == two
    three = _digest_of_script(
        tmp_path,
        "par3",
        text.replace("SCALE = 2", "SCALE = 3").format(grid="[1]"),
        {"task"},
    )
    assert three != one


def test_a_name_both_assigned_and_imported_is_never_stale(tmp_path):
    """`np = None` then `try: import numpy as np`: no check compares a loaded
    value with the file, so a name the script rebinds never reads as stale."""
    from mushin._resume import _script_entry

    path = tmp_path / "mushin_t_np.py"
    ns = _script_ns(
        path,
        "np = None\ntry:\n    import json as np\nexcept ImportError:\n    pass\n\n\n"
        "def task(a):\n    return a if np is None else a + 1\n",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        first = _script_entry(str(path), ns, {"task"}, (ns["task"],))[0]
        assert _script_entry(str(path), ns, {"task"}, (ns["task"],))[0] == first


def test_a_cell_uses_the_drivers_script_entry(project, tmp_path):
    """A worker runs the driver's pickled script functions: the entry the
    driver took when the run started is the one recorded, even if the file
    changed since."""
    from mushin._resume import cell_project_record, driver_script_entries

    path = tmp_path / "mushin_t_drv.py"
    path.write_text("OFFSET = 0\n\n\ndef task(a):\n    return a + OFFSET\n")
    ns = {"__name__": "__main__", "__file__": str(path)}
    exec(compile(path.read_bytes(), str(path), "exec"), ns)
    given = driver_script_entries([ns["task"]])
    path.write_text("OFFSET = 5\n\n\ndef task(a):\n    return a + OFFSET\n")
    record = cell_project_record("tok-drv", (ns["task"],), (), None, scripts=given)
    assert record["code"]["mushin_t_drv (script)"] == given[str(path)][0]


def test_a_partials_bound_function_records_its_module(project):
    """`partial(train, fn=helpers.scale)`: the module of the bound function is
    code the task runs, though the task's module does not import it."""
    root, _ = project
    mod = _module(root, "mushin_t_ptrain", "def train(a, fn):\n    return fn(a)\n")
    helpers = importlib.import_module(HELPERS)
    try:
        record = project_fingerprints(functools.partial(mod.train, fn=helpers.scale))
        assert f"{HELPERS} (file)" in record["code"]
    finally:
        sys.modules.pop("mushin_t_ptrain", None)


def test_a_bound_method_root_records_its_module(project):
    """Study can be given a bound method (`trainer.fit`)."""
    root, _ = project
    mod = _module(
        root,
        "mushin_t_trainer",
        f"import {HELPERS} as helpers\n\n\nclass Trainer:\n    def fit(self, seed):\n        return helpers.scale(seed)\n",
    )
    try:
        code = project_fingerprints(mod.Trainer().fit)["code"]
        assert {"mushin_t_trainer (file)", f"{HELPERS} (file)"} <= set(code)
    finally:
        sys.modules.pop("mushin_t_trainer", None)


def test_a_notebook_records_the_modules_of_the_objects_it_holds(project):
    """A notebook holding an instance of a project class (a model made from
    an imported module, the module itself not kept under a name)."""
    root, _ = project
    mod = _module(
        root,
        "mushin_t_nbmodel",
        "class Model:\n    def f(self, x):\n        return x\n",
    )
    nb = types.ModuleType("mushin_t_nbinst")
    nb.__dict__["MODEL"] = mod.Model()
    exec("def task(a):\n    return MODEL.f(a)\n", nb.__dict__)
    try:
        assert "mushin_t_nbmodel (file)" in project_fingerprints(nb.task)["code"]
    finally:
        sys.modules.pop("mushin_t_nbmodel", None)


# --- the final review: each case where it breaks --------------------------------


def _script_ns(path, text):
    path.write_text(text)
    ns = {"__name__": "__main__", "__file__": str(path)}
    exec(compile(path.read_bytes(), str(path), "exec", dont_inherit=True), ns)
    return ns


def test_two_records_of_one_script_with_other_roots_are_not_stale(tmp_path):
    """A task and a `_target_` class in one script are two records of it, with
    two sets of roots: neither is taken for an edit."""
    from mushin._resume import _script_digest, _script_entry

    path = tmp_path / "mushin_t_two.py"
    ns = _script_ns(path, "class Net:\n    K = 1\n\n\ndef task(a):\n    return a\n")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for _ in range(2):
            a = _script_entry(str(path), ns, {"task"}, (ns["task"],))[0]
            b = _script_entry(str(path), ns, {"Net"}, (ns["Net"],))[0]
    assert not [w for w in caught if "changed on disk" in str(w.message)]
    assert a == _script_digest(path, frozenset({"task"}))[0]
    assert b == _script_digest(path, frozenset({"Net"}))[0]


def test_a_star_import_in_a_script_is_followed(tmp_path):
    from mushin._resume import _resolved, _script_digest

    path = tmp_path / "mushin_t_star.py"
    path.write_text("from starhelp import *\n\n\ndef task(a):\n    return scale(a)\n")
    _, imports = _script_digest(path, frozenset({"task"}))
    assert "starhelp" in _resolved(imports, "__main__", False)


def test_a_script_saved_with_a_byte_order_mark_is_read(tmp_path):
    from mushin._resume import _script_digest

    path = tmp_path / "mushin_t_bom.py"
    path.write_bytes(b"\xef\xbb\xbfSCALE = 2\n\n\ndef task(a):\n    return a * SCALE\n")
    assert _script_digest(path, frozenset({"task"}))[0]


def test_a_foreign_decorator_on_main_does_not_count_a_project_one_does(tmp_path):
    """`@click.command()` on a `main()` that runs the sweep is not code the
    task runs; a registry decorator defined in the script is."""
    text = (
        "import functools\n\nREGISTRY = {{}}\nGRID = {grid}\n\n\n"
        "def register(cls):\n    REGISTRY[cls.__name__] = cls\n    return cls\n\n\n"
        "@register\nclass Lin:\n    K = {k}\n\n\n"
        "def task(a):\n    return a * REGISTRY['Lin'].K\n\n\n"
        "@functools.lru_cache\ndef main():\n    run(a=GRID)\n"
    )
    base = _digest_of_script(
        tmp_path, "dec1", text.format(grid="[1, 2]", k=1), {"task"}
    )
    grid = _digest_of_script(
        tmp_path, "dec2", text.format(grid="[1, 2, 3]", k=1), {"task"}
    )
    lin = _digest_of_script(tmp_path, "dec3", text.format(grid="[1, 2]", k=2), {"task"})
    assert base == grid
    assert lin != base


def test_a_partial_over_a_closure_or_a_bound_method_is_not_cached():
    from mushin._resume import partial_task_cacheable

    def make(w):
        return lambda a: a * w

    class T:
        def fit(self, a):
            return a

    assert not partial_task_cacheable(functools.partial(make(2.0)))
    assert not partial_task_cacheable(functools.partial(T().fit))
    assert not partial_task_cacheable(functools.partial(lambda a: a))


def test_a_target_module_not_imported_yet_is_judged_by_its_file(project, monkeypatch):
    """`_target_: mushin_t_tgtpkg.models.Net` while an installed distribution
    also ships that top-level name: before Hydra imports it, its file decides,
    so a worker that has not imported it records it as the driver does."""
    import mushin._resume as r

    root, _ = project
    pkg = root / "mushin_t_tgtpkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "models.py").write_text("class Net:\n    pass\n")
    monkeypatch.setattr(
        r,
        "_installed_roots",
        lambda: {"mushin_t_tgtpkg": ((root.parent / "x").resolve(),)},
    )
    r._foreign_by_path.cache_clear()
    try:
        seeds, scripts = [], {}
        r._target_entries("mushin_t_tgtpkg.models.Net", seeds, scripts)
        assert seeds == ["mushin_t_tgtpkg.models"]
        assert "mushin_t_tgtpkg" not in sys.modules
    finally:
        r._foreign_by_path.cache_clear()


def test_a_partials_bound_arguments_are_not_hashed():
    """A partial's bound arguments are not compared (as on main), so nothing
    of them, a credential included, reaches a sidecar."""

    def task(a, db_password):
        return a

    one = project_fingerprints(functools.partial(task, db_password="hunter2"))
    two = project_fingerprints(functools.partial(task, db_password="correct-horse"))
    assert one == two


def test_a_setting_shaped_like_a_token_prefix_is_compared(project):
    root, _ = project
    src = "MODEL = {v!r}\n\n\ndef task(a):\n    return a\n"
    a, b = "hf_tiny_random_gpt2_model", "hf_tiny_random_llama_model"
    assert _edited(root, "mushin_t_hfname", src, a, b) != []


def test_the_cache_key_does_not_depend_on_the_python_version():
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
        code_roots=(len,),
    )
    code = {"m (file)": "1"}
    assert runner._cache_key(
        "h", {"python": "3.11", "code": code}
    ) == runner._cache_key("h", {"python": "3.13", "code": code})


def test_a_nested_namespace_package_does_not_drop_the_record(project):
    """`import nspkg.sub.mod` inside the task, no `__init__.py` anywhere, the
    parent not imported: one name that cannot be resolved costs its entry,
    never the whole record."""
    root, _ = project
    (root / "mushin_t_nspkg" / "sub").mkdir(parents=True)
    (root / "mushin_t_nspkg" / "sub" / "mod.py").write_text("X = 1\n")
    mod = _module(
        root,
        "mushin_t_nsuser",
        "def task(a):\n    import mushin_t_nspkg.sub.mod as m\n\n    return a + m.X\n",
    )
    try:
        record = project_fingerprints(mod.task)
        assert record is not None and "mushin_t_nsuser (file)" in record["code"]
        assert "mushin_t_nspkg.sub.mod (file)" in record["code"]
        assert "mushin_t_nspkg" not in sys.modules  # nothing was imported
    finally:
        sys.modules.pop("mushin_t_nsuser", None)


def test_a_lambda_root_in_a_script_counts_what_it_reads(tmp_path):
    """Study methods given as lambdas in the script: `{"a": lambda s: train(s,
    LR)}` reads `train` and `LR`."""
    from mushin._resume import driver_script_entries

    path = tmp_path / "mushin_t_lam.py"
    src = "LR = {v}\n\n\ndef train(s, lr):\n    return s * lr\n\n\nFN = lambda s: train(s, LR)\n"
    ns = _script_ns(path, src.format(v=0.1))
    first = driver_script_entries([ns["FN"]])[str(path)][0]
    path2 = tmp_path / "mushin_t_lam2.py"
    ns2 = _script_ns(path2, src.format(v=0.2))
    second = driver_script_entries([ns2["FN"]])[str(path2)][0]
    assert first != second


def test_an_old_pickled_runner_keys_the_cache(tmp_path):
    """A runner pickled before `cache_ok` and `script_entries` existed still
    computes a cache key and a record."""
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
        code_roots=(len,),
    )
    state = {
        k: v for k, v in vars(runner).items() if k not in {"cache_ok", "script_entries"}
    }
    old = _TaskRunner.__new__(_TaskRunner)
    old.__setstate__(state)
    assert old.script_entries == {} and old.cache_ok is True
    assert old._cache_key("cfg", {"python": PY, "code": {"m (file)": "1"}})


def test_a_relative_project_decorator_registers(tmp_path):
    """`python -m pkg.train` with `from .registry import MODELS, register`: a
    relative import is the script's own package's, so `@register` registers a
    definition the task reaches through `MODELS`."""
    text = (
        "from .registry import MODELS, register\n\n\n"
        "@register\nclass Lin:\n    K = {k}\n\n\n"
        "def task(a):\n    return a * MODELS['Lin'].K\n"
    )
    one = _digest_of_script(tmp_path, "rel1", text.format(k=1), {"task"})
    two = _digest_of_script(tmp_path, "rel2", text.format(k=2), {"task"})
    assert one != two


def test_a_star_imported_project_decorator_registers(project, tmp_path):
    root, _ = project
    (root / "mushin_t_starreg.py").write_text(
        "MODELS = {}\n\n\ndef register(cls):\n    MODELS[cls.__name__] = cls\n    return cls\n"
    )
    text = (
        "from mushin_t_starreg import *\n\n\n"
        "@register\nclass Lin:\n    K = {k}\n\n\n"
        "def task(a):\n    return a * MODELS['Lin'].K\n"
    )
    one = _digest_of_script(tmp_path, "sr1", text.format(k=1), {"task"})
    two = _digest_of_script(tmp_path, "sr2", text.format(k=2), {"task"})
    assert one != two


def test_a_generic_helper_in_a_script_counts_what_it_reads(tmp_path):
    """`def helper[T](x)` (3.12+): the symbol table wraps it in a type
    parameter scope of the same name."""
    if sys.version_info < (3, 12):
        pytest.skip("PEP 695 syntax")
    text = (
        "SCALE = {v}\n\n\ndef helper[T](x: T) -> T:\n    return x * SCALE\n\n\n"
        "def task(a):\n    return helper(a)\n"
    )
    one = _digest_of_script(tmp_path, "gen1", text.format(v=2), {"task"})
    two = _digest_of_script(tmp_path, "gen2", text.format(v=3), {"task"})
    assert one != two


def test_a_second_sweep_in_the_script_is_not_the_firsts_code(tmp_path):
    """Two `@mushin.sweep` functions in one script: the task reads `mushin`
    through its own decorator, which does not make the other sweep its code."""
    text = (
        "import mushin\n\n\n@mushin.sweep\ndef exp_a(a):\n    return {{'v': a}}\n\n\n"
        "@mushin.sweep\ndef exp_b(a):\n    return {{'v': a * {k}}}\n"
    )
    one = _digest_of_script(tmp_path, "two1", text.format(k=1), {"exp_a"})
    two = _digest_of_script(tmp_path, "two2", text.format(k=2), {"exp_a"})
    assert one == two


def test_a_main_guard_inside_a_script_helper_counts(tmp_path):
    """In a driver script `__name__ == "__main__"` holds everywhere, so a
    helper's guarded branch runs: it is code the task runs."""
    text = (
        "def helper(x):\n    if __name__ == '__main__':\n        return x * {k}\n"
        "    return x\n\n\ndef task(a):\n    return helper(a)\n"
    )
    one = _digest_of_script(tmp_path, "mg1", text.format(k=2), {"task"})
    two = _digest_of_script(tmp_path, "mg2", text.format(k=20), {"task"})
    assert one != two


def test_a_script_helpers_annotations_count(tmp_path):
    """An annotation is evaluated where the definition is: what it names is
    code the definition reads (a decorator may act on it)."""
    text = (
        "SCALE = {v}\n\n\ndef helper(x, factor: SCALE = 1):\n    return x\n\n\n"
        "def task(a):\n    return helper(a)\n"
    )
    one = _digest_of_script(tmp_path, "ann1", text.format(v=2), {"task"})
    two = _digest_of_script(tmp_path, "ann2", text.format(v=20), {"task"})
    assert one != two


def test_a_second_stale_episode_after_a_reload_warns_again(project):
    root, exp = project
    _code(exp.W.task)
    _write(root, scale=100)  # edited, not reloaded
    with pytest.warns(UserWarning, match="changed on disk after this process"):
        _code(exp.W.task)
    exp = _reload(exp)
    _code(exp.W.task)  # the new load's baseline
    _write(root, scale=1000)  # edited again, not reloaded
    with pytest.warns(UserWarning, match="changed on disk after this process"):
        _code(exp.W.task)


def test_the_baseline_does_not_keep_a_script_alive(tmp_path):
    """A notebook re-running a script keeps replacing its functions: the
    baseline must not pin the old ones (and their namespace)."""
    import gc
    import weakref

    from mushin._resume import _script_entry

    path = tmp_path / "mushin_t_pin.py"
    ns = _script_ns(path, "BIG = list(range(10))\n\n\ndef task(a):\n    return a\n")
    task = ns.pop("task")
    probe = weakref.ref(task)
    _script_entry(str(path), ns, {"task"}, (task,))
    del task
    gc.collect()
    assert probe() is None


def test_an_edit_that_keeps_size_and_mtime_is_seen(project):
    """A same-size edit within one tick of a coarse file clock (Windows) has
    the same size and modification time: the file's bytes decide."""
    from mushin._resume import _parsed, _script_digest

    root, _ = project
    path = root / "mushin_t_tick.py"
    path.write_text("K = 2\n")
    st = path.stat()
    first = _parsed(path)[0]
    script_first = _script_digest(path, frozenset({"K"}))[0]
    path.write_text("K = 3\n")
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns))
    assert path.stat().st_mtime_ns == st.st_mtime_ns
    assert _parsed(path)[0] != first
    assert _script_digest(path, frozenset({"K"}))[0] != script_first


# --- the script rule follows what runs (Codex's cases on #270) ------------------
#
# Each text is a driver script whose task is `task`; `{k}` is the edit. An
# EDIT changes what the task computes, so the script's digest must change; a
# NOT_EDIT does not, so it must not (extending a grid, renaming nothing).

SCRIPT_EDITS = {
    "a decorator alias": (
        "MODELS = {{}}\n\n\ndef register(cls):\n    MODELS[cls.__name__] = cls\n"
        "    return cls\n\n\nregister_model = register\n\n\n@register_model\n"
        "class Lin:\n    K = {k}\n\n\ndef task(a):\n    return a * MODELS['Lin'].K\n"
    ),
    "a decorator that registers through a helper": (
        "MODELS = {{}}\n\n\ndef _add(cls):\n    MODELS[cls.__name__] = cls\n\n\n"
        "def register(cls):\n    _add(cls)\n    return cls\n\n\n@register\n"
        "class Lin:\n    K = {k}\n\n\ndef task(a):\n    return a * MODELS['Lin'].K\n"
    ),
    "a registration inside a block": (
        "MODELS = {{}}\nENABLED = True\n\n\ndef register(cls):\n"
        "    MODELS[cls.__name__] = cls\n    return cls\n\n\nif ENABLED:\n\n"
        "    @register\n    class Lin:\n        K = {k}\n\n\n"
        "def task(a):\n    return a * MODELS['Lin'].K\n"
    ),
    "a match capture": (
        "CFG = {{'scale': {k}}}\nmatch CFG:\n    case {{'scale': SCALE}}:\n"
        "        pass\n\n\ndef task(a):\n    return a * SCALE\n"
    ),
    "a class body reading the global it shadows": (
        "SCALE = {k}\n\n\nclass Config:\n    SCALE = SCALE\n\n\n"
        "def task(a):\n    return a * Config.SCALE\n"
    ),
    "a function that sets a global": (
        "DEVICE = 'cpu'\n\n\ndef configure():\n    global DEVICE\n"
        "    DEVICE = 'cuda:{k}'\n\n\ndef task(a):\n    return len(DEVICE) * a\n"
    ),
    "a value a conditional rebinding may keep": (
        "FAST = False\nSCALE = {k}\nif FAST:\n    SCALE = 5\n\n\n"
        "def task(a):\n    return a * SCALE\n"
    ),
    "a walrus inside a comprehension": (
        "VALS = [(LAST := v) for v in range({k})]\n\n\ndef task(a):\n    return a * LAST\n"
    ),
    "a constant computed from another": (
        "BASE = {k}\nSCALE = BASE * 10\n\n\ndef task(a):\n    return a * SCALE\n"
    ),
    "an augmented assignment": (
        "SCALE = {k}\nSCALE += 1\n\n\ndef task(a):\n    return a * SCALE\n"
    ),
    "a value read when it was computed": (
        "SCALE = {k}\nFACTOR = SCALE * 2\nSCALE = 100\n\n\n"
        "def task(a):\n    return a * FACTOR\n"
    ),
}

SCRIPT_NOT_EDITS = {
    "a comprehension variable named like a global": (
        "SCALE = 2\nGRID = [SCALE for SCALE in range({k})]\n\n\n"
        "def task(a):\n    return a * SCALE\n"
    ),
    "a helper's local inside a block": (
        "SCALE = 2\nif True:\n\n    def make():\n        SCALE = 3\n        return SCALE\n\n"
        "    GRID = list(range({k}))\n\n\ndef task(a):\n    return a * SCALE\n"
    ),
    "a value rebound before the task reads it": (
        "SCALE = {k}\nSCALE = 5\n\n\ndef task(a):\n    return a * SCALE\n"
    ),
    "a declaration without a value after the binding": (
        "SCALE = 2\nSCALE: 'v{k}'\n\n\ndef task(a):\n    return a * SCALE\n"
    ),
}


@pytest.mark.parametrize("case", sorted(SCRIPT_EDITS))
def test_a_script_edit_is_seen(tmp_path, case):
    text = SCRIPT_EDITS[case]
    one = _digest_of_script(tmp_path, "se1", text.format(k=1), {"task"})
    two = _digest_of_script(tmp_path, "se2", text.format(k=2), {"task"})
    assert one != two


@pytest.mark.parametrize("case", sorted(SCRIPT_NOT_EDITS))
def test_a_script_change_the_task_cannot_see_is_not_an_edit(tmp_path, case):
    text = SCRIPT_NOT_EDITS[case]
    one = _digest_of_script(tmp_path, "sn1", text.format(k=1), {"task"})
    two = _digest_of_script(tmp_path, "sn2", text.format(k=2), {"task"})
    assert one == two


def test_a_future_import_in_a_script_counts(tmp_path):
    """`from __future__ import annotations` changes how every definition in
    the script runs, whatever names the task reads."""
    body = (
        "SCALE = 2\n\n\ndef helper(x: SCALE):\n    return x\n\n\n"
        "def task(a):\n    return helper.__annotations__['x']\n"
    )
    plain = _digest_of_script(tmp_path, "fut1", body, {"task"})
    future = _digest_of_script(
        tmp_path, "fut2", "from __future__ import annotations\n\n" + body, {"task"}
    )
    assert plain != future


def test_a_main_guard_else_branch_does_not_run_in_the_script(tmp_path):
    """The driver script runs as `__main__`: its main block's body runs and
    its `else` does not, so moving an assignment between them is an edit."""
    text = (
        "def task(a):\n    return a * SCALE\n\n\n"
        "if __name__ == '__main__':\n    SCALE = 2\n{moved}else:\n{other}"
    )
    body = _digest_of_script(
        tmp_path,
        "mge1",
        text.format(moved="    SCALE = 20\n", other="    pass\n"),
        {"task"},
    )
    other = _digest_of_script(
        tmp_path, "mge2", text.format(moved="", other="    SCALE = 20\n"), {"task"}
    )
    assert body != other


def test_reordering_independent_definitions_is_not_an_edit(tmp_path):
    first = "SCALE = 2\n\n\ndef helper(x):\n    return x * SCALE\n\n\n"
    second = "def task(a):\n    return helper(a)\n"
    one = _digest_of_script(tmp_path, "ord1", first + "\n\n" + second, {"task"})
    two = _digest_of_script(tmp_path, "ord2", second + "\n\n" + first, {"task"})
    assert one == two


#: One script holding most constructs the rule reads. Its digest must be the
#: same on every supported Python (records are compared across versions, and
#: a digest that moved with the interpreter would re-run every cell).
CROSS_VERSION_SCRIPT = '''
"""A driver script."""
import functools
import math as m
from collections import OrderedDict

SCALE = 2
NAMES = [n for n in ("a", "b")]
TABLE = {k: v for k, v in zip(NAMES, range(2))}
GEN = sum(x * SCALE for x in range(3))
BASE = 2
SCALED = [BASE * v for v in range(2)]
BASE = 5
FN = lambda s: s * SCALE


class Config:
    LR: float = 0.1
    SCALE = SCALE

    def scaled(self, x: int = 1) -> float:
        return x * self.LR * m.pi


@functools.lru_cache
def helper(x, *, factor: int = SCALE):
    try:
        import json as _json
    except ImportError as err:
        _json = err
    with open(__file__) as f:
        _ = f.read(1)
    return [Config().scaled(x) for _ in range(factor)], OrderedDict(), _json


def task(a):
    global LAST
    LAST = a
    return helper(a), FN(a), TABLE, GEN, SCALED


if __name__ == "__main__":
    GRID = list(range(3))
'''

CROSS_VERSION_DIGEST = "e0128ffdf7cad4fe"


def test_a_script_digest_is_the_same_on_every_python(tmp_path):
    digest = _digest_of_script(tmp_path, "xver", CROSS_VERSION_SCRIPT, {"task"})
    assert digest == CROSS_VERSION_DIGEST
