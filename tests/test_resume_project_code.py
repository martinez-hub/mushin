# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
"""resume=True and cache_dir= notice edits to the project code a task reaches.

The task's own source was fingerprinted before; a helper it calls, or a
constant in its module, was not, so a resume after editing one returned the
stale result. The fingerprint now follows the names the task's code uses
through the project's own functions, classes, modules and constants, as the
process has loaded them, and hashes functions by their bytecode.
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


# --- what the walk reaches -------------------------------------------------


def test_the_walk_reaches_the_helper_and_the_constants_the_task_uses(project):
    _, exp = project
    code = _code(exp.W.task)
    # Keyed by where the name was looked up; the task's module by its file name.
    assert {f"{EXP}:OFFSET", f"{HELPERS}:scale"} <= set(code)
    # A module global names which module it is (an alias pointed at another
    # module is a change); the names used on it are entries of their own.
    assert f"{EXP}:helpers" in code
    assert f"{HELPERS}:SCALE" in code  # read by scale, reached through it
    assert not [k for k in code if k.startswith(("mushin.", "numpy", "json"))]
    # The root's own body is code_fingerprint's: not an entry here.
    assert f"{EXP}:W.task" not in code
    # A partial adds its bound arguments to what its function reaches.
    partial_code = _code(functools.partial(exp.W.task, a=1))
    assert set(partial_code) - set(code) == {f"{EXP}:W.task.<partial>"}


def test_a_constant_imported_by_name_is_reached(project):
    """`from consts import THRESH` leaves an int in the globals; its value is
    what the task reads, so its value is hashed."""
    root, exp = project
    (root / "mushin_t_consts.py").write_text("THRESH = 10\n")
    (root / f"{EXP}.py").write_text(
        textwrap.dedent("""
            from mushin_t_consts import THRESH


            def task(a):
                return dict(v=a * THRESH)
            """)
    )
    sys.modules.pop("mushin_t_consts", None)
    exp = importlib.reload(exp)
    try:
        before = _code(exp.task)
        assert f"{EXP}:THRESH" in before
        (root / "mushin_t_consts.py").write_text("THRESH = 100\n")
        importlib.reload(sys.modules["mushin_t_consts"])
        exp = importlib.reload(exp)
        assert changed_project_code(
            {"python": PY, "code": before}, project_fingerprints(exp.task)
        ) == [f"{EXP}:THRESH"]
    finally:
        sys.modules.pop("mushin_t_consts", None)


def test_comments_and_moved_lines_are_not_edits_but_code_is(project):
    root, exp = project
    before = _code(exp.W.task)
    _write(root, comment="# a comment that shifts every line below")
    exp = _reload(exp)
    assert _code(exp.W.task) == before
    _write(root, scale=100)
    exp = _reload(exp)
    assert changed_project_code(
        {"python": PY, "code": before}, project_fingerprints(exp.W.task)
    ) == [f"{HELPERS}:SCALE"]


def test_a_rebound_module_attribute_is_a_change(project):
    """Keys say where a name was looked up, so `scale = other` in the helper
    module changes the entry under `helpers:scale`."""
    root, exp = project
    before = _code(exp.W.task)
    helpers = root / f"{HELPERS}.py"
    helpers.write_text(
        helpers.read_text()
        + "\n\ndef other(x):\n    return float(x) * 2\n\n\nscale = other\n"
    )
    exp = _reload(exp)
    changed = changed_project_code(
        {"python": PY, "code": before}, project_fingerprints(exp.W.task)
    )
    assert f"{HELPERS}:scale" in changed


def test_a_notebook_task_reaches_its_own_helpers_and_only_what_it_uses(project):
    """A task defined where there is no file: its notebook helpers and
    constants are code it runs, and an unused import is not a change."""
    root, exp = project
    (root / "mushin_t_unused.py").write_text("X = 1\n")
    nb = types.ModuleType("mushin_t_notebook")  # a kernel's __main__ has no file
    sys.modules[nb.__name__] = nb
    try:
        src = textwrap.dedent(f"""
            import {HELPERS} as helpers

            FACTOR = {{factor}}


            def inner(a):
                return helpers.scale(a) * FACTOR


            def task(a):
                return dict(v=inner(a))
            """)
        exec(src.format(factor=2), nb.__dict__)
        before = _code(nb.task)
        assert {"mushin_t_notebook:inner", "mushin_t_notebook:FACTOR"} <= set(before)
        assert f"{HELPERS}:scale" in before
        exec("import mushin_t_unused", nb.__dict__)
        assert _code(nb.task) == before
        exec(src.format(factor=3), nb.__dict__)  # re-running the cell
        assert "mushin_t_notebook:FACTOR" in changed_project_code(
            {"python": PY, "code": before}, project_fingerprints(nb.task)
        )
    finally:
        sys.modules.pop("mushin_t_notebook", None)
        sys.modules.pop("mushin_t_unused", None)


def test_an_import_inside_the_task_is_followed(project):
    """A module imported in the task's body is recorded by its file, found
    without importing it, and keyed the same whether or not the process has
    imported it: a fresh process and a notebook that already ran the task
    compare. A relative import resolves against the task's package, and a
    comment in that file is not an edit."""
    root, exp = project
    pkg = root / "mushin_t_pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "lazy.py").write_text("K = 3\n")
    (pkg / "exp.py").write_text(
        textwrap.dedent("""
            def task(a):
                from .lazy import K

                return dict(v=a * K)
            """)
    )
    for name in ("mushin_t_pkg", "mushin_t_pkg.lazy", "mushin_t_pkg.exp"):
        sys.modules.pop(name, None)
    mod = importlib.import_module("mushin_t_pkg.exp")
    try:
        unloaded = project_fingerprints(mod.task)
        assert "mushin_t_pkg.lazy (file)" in unloaded["code"]
        assert "mushin_t_pkg.lazy" not in sys.modules  # found without importing
        importlib.import_module("mushin_t_pkg.lazy")
        assert project_fingerprints(mod.task) == unloaded
        (pkg / "lazy.py").write_text("# a comment\nK = 3\n")
        assert project_fingerprints(mod.task) == unloaded
        (pkg / "lazy.py").write_text("K = 4\n")
        assert _changed(unloaded, mod.task) == ["mushin_t_pkg.lazy (file)"]
    finally:
        for name in ("mushin_t_pkg", "mushin_t_pkg.lazy", "mushin_t_pkg.exp"):
            sys.modules.pop(name, None)


def test_a_submodule_of_an_unloaded_package_is_found_without_importing(project):
    """`from otherpkg.sub import g` inside the task, with nothing importing
    otherpkg: the submodule's file is located through the package's search
    path, not just the package's __init__."""
    root, _ = project
    pkg = root / "mushin_t_other"
    (pkg / "inner").mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    (pkg / "inner" / "__init__.py").write_text("")
    (pkg / "inner" / "sub.py").write_text("G = 1\n")
    mod = _module(
        root,
        "mushin_t_lazyuser",
        "def task(a):\n    from mushin_t_other.inner.sub import G\n\n    return dict(v=a * G)\n",
    )
    try:
        code = project_fingerprints(mod.task)["code"]
        assert "mushin_t_other.inner.sub (file)" in code
        assert "mushin_t_other" not in sys.modules
    finally:
        for n in list(sys.modules):
            if n.startswith(("mushin_t_other", "mushin_t_lazyuser")):
                sys.modules.pop(n, None)


def test_the_walk_survives_cycles_and_globals_that_resist_inspection(project):
    root, exp = project
    (root / f"{EXP}.py").write_text(
        textwrap.dedent("""
            import sys


            class Proxy:
                def __getattribute__(self, name):
                    raise RuntimeError("not configured")


            SETTINGS = Proxy()


            def again(n):
                return again(n - 1) if n else SETTINGS


            def task(a):
                again(0)
                return dict(v=a)
            """)
    )
    exp = importlib.reload(exp)
    code = _code(exp.task)
    assert f"{EXP}:again" in code


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
        UserWarning, match=rf"code the task reaches \({HELPERS}:SCALE\)"
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
    with pytest.warns(UserWarning, match=rf"code the task reaches \({EXP}:OFFSET\)"):
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


def test_an_edit_the_process_has_not_loaded_reuses_until_it_is_loaded(project):
    """The fingerprint is of the loaded code, which is what runs: an edit on
    disk that this process has not imported does not change what the resume
    would compute, so the cells are reused; a fresh process sees the edit."""
    root, exp = project
    wd = str(root / "sweep")
    exp.W().run(a=multirun([1, 2]), working_dir=wd)
    _write(root, scale=100)  # on disk only
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ds = _resume(exp, wd)
    assert float(ds["v"].sel(a=2)) == 20.0
    exp = _reload(exp)  # now the process runs the edit
    with pytest.warns(UserWarning, match="fingerprint mismatch"):
        ds = _resume(exp, wd)
    assert float(ds["v"].sel(a=2)) == 200.0


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
    )
    two_part = hashlib.sha256(b"h:c0de").hexdigest()[:32]
    assert runner._cache_key("h", {"python": PY, "code": {}}) == two_part
    assert runner._cache_key("h", {"python": PY, "code": {"m:f": "1"}}) != two_part
    assert runner._cache_key("h", None) is None


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

    write_script()
    run()
    assert calls.read_text() == "xx"
    write_script(tail='print("done")  # an edit to the run block')
    script.write_text(script.read_text() + "\n# and a comment\n")
    err = run("resume")
    assert "fingerprint mismatch" not in err
    assert calls.read_text() == "xx"  # both cells reused
    write_script(offset=1)
    err = run("resume")
    assert "code the task reaches (exp_script:OFFSET)" in err
    assert calls.read_text() == "xxxx"
    write_script(offset=1, bonus=5)
    err = run("resume")
    assert "code the task reaches (exp_script:bonus)" in err
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
            UserWarning, match=rf"code the task reaches \({HELPERS}:SCALE\)"
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
    recorded before this change would re-train. Pin main's formula."""
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
        assert f"{HELPERS}:scale" in code  # the training function's helper
    finally:
        sys.modules.pop("mushin_t_train2", None)


def test_the_hash_is_the_same_in_every_process(tmp_path):
    """A frozenset constant (`x in {"a", "b"}`) prints in string-hash order,
    which PYTHONHASHSEED changes per process; the hash must not."""
    (tmp_path / "mushin_t_seeded.py").write_text(
        textwrap.dedent("""
            NAMES = frozenset({"alpha", "beta", "gamma", "delta"})


            def task(a):
                return dict(v=float(a in {"x", "y", "z", "w", "q"} or a in NAMES))
            """)
    )
    probe = (
        "import json, sys; sys.path.insert(0, sys.argv[1]); "
        "import mushin_t_seeded as m; from mushin._resume import project_fingerprints; "
        "from mushin._resume import code_fingerprint, _code_hash; "
        "print(json.dumps([project_fingerprints(m.task), _code_hash(m.task.__code__)]))"
    )
    outs = {
        subprocess.run(
            [sys.executable, "-c", probe, str(tmp_path)],
            capture_output=True,
            text=True,
            check=True,
            env={**os.environ, "PYTHONHASHSEED": seed, "PYTHONDONTWRITEBYTECODE": "1"},
        ).stdout
        for seed in ("0", "1", "2", "3")
    }
    assert len(outs) == 1, outs


def test_enum_values_and_decorated_functions_are_reached(project):
    """An enum member's value is a constant the task reads; a function behind a
    foreign decorator (functools.lru_cache) is code it runs."""
    root, exp = project
    src = textwrap.dedent("""
        import enum
        import functools


        class Mode(enum.Enum):
            FAST = {fast}


        @functools.lru_cache
        def cached(a):
            return a * {factor}


        def task(a):
            return dict(v=float(cached(a) + Mode.FAST.value))
        """)
    (root / f"{EXP}.py").write_text(src.format(fast=1, factor=2))
    exp = importlib.reload(exp)
    before = project_fingerprints(exp.task)
    assert f"{EXP}:cached.__wrapped__" in before["code"]
    (root / f"{EXP}.py").write_text(src.format(fast=5, factor=2))
    exp = importlib.reload(exp)
    assert f"{EXP}:Mode" in changed_project_code(before, project_fingerprints(exp.task))
    (root / f"{EXP}.py").write_text(src.format(fast=1, factor=3))
    exp = importlib.reload(exp)
    assert f"{EXP}:cached.__wrapped__" in changed_project_code(
        before, project_fingerprints(exp.task)
    )


# --- round 4: the walk's precision ----------------------------------------------


def _module(root, name, src):
    (root / f"{name}.py").write_text(textwrap.dedent(src))
    sys.modules.pop(name, None)
    return importlib.import_module(name)


def _changed(before, fn, **kwargs):
    return changed_project_code(before, project_fingerprints(fn, **kwargs))


def test_a_module_reached_by_two_functions_is_walked_for_both(project):
    """The names a second function reads in a module the first already reached
    are fingerprinted too."""
    root, _ = project
    _module(root, "mushin_t_h2", "OFFSET = 1\n\n\ndef scale(x):\n    return x * 2\n")
    src = """
        import mushin_t_h2 as h


        def inner(a):
            return h.OFFSET * a


        def task(a):
            return dict(v=inner(a) + h.scale(a))
        """
    mod = _module(root, "mushin_t_two", src)
    try:
        code = project_fingerprints(mod.task)["code"]
        assert {"mushin_t_h2:OFFSET", "mushin_t_h2:scale"} <= set(code)
    finally:
        for n in ("mushin_t_h2", "mushin_t_two"):
            sys.modules.pop(n, None)


def test_objects_containers_and_installed_code_get_no_entry(project):
    """An installed module the task reaches gets no entry of its own, and a
    container the task fills as it runs does not move the record: a project
    file's literal names are hashed by value and the rest by the source of
    their assignment, so `CACHE = {}` reads the same after a run fills it, and
    `act = math.fabs` swapped for another installed function shows."""
    root, _ = project
    src = """
        import math

        import numpy as np

        LAYERS = [64, 64]
        CACHE = {{}}
        act = math.{fn}


        def task(a):
            CACHE[a] = 1
            return dict(v=act(a) * len(LAYERS) + float(np.mean([a])))
        """
    mod = _module(root, "mushin_t_kinds", textwrap.dedent(src).format(fn="fabs"))
    try:
        before = project_fingerprints(mod.task)
        assert "mushin_t_kinds:np" not in before["code"]
        mod.task(3)
        assert project_fingerprints(mod.task) == before
        mod = _module(root, "mushin_t_kinds", textwrap.dedent(src).format(fn="sqrt"))
        assert _changed(before, mod.task) == ["mushin_t_kinds:act"]
    finally:
        sys.modules.pop("mushin_t_kinds", None)


def test_a_registry_of_project_classes_is_walked(project):
    """`MODELS = {"cnn": CNN}` with the task picking one by name: the classes it
    holds are code the task runs."""
    root, _ = project
    src = """
        class CNN:
            def forward(self, a):
                return a * {k}


        MODELS = {{"cnn": CNN}}


        def task(method, a):
            return dict(v=MODELS[method]().forward(a))
        """
    mod = _module(root, "mushin_t_registry", src.format(k=2))
    try:
        before = project_fingerprints(mod.task)
        mod = _module(root, "mushin_t_registry", src.format(k=3))
        assert "mushin_t_registry:MODELS['cnn']" in _changed(before, mod.task)
    finally:
        sys.modules.pop("mushin_t_registry", None)


def test_partial_keywords_closures_and_properties_are_code(project):
    root, _ = project
    src = """
        import functools


        def mul(x, factor):
            return x * factor


        def make(k):
            def f(x):
                return x + k

            return f


        class Model:
            @property
            def lr(self):
                return {lr}


        pscale = functools.partial(mul, factor={factor})
        s10 = make({k10})
        s100 = make(100)


        def task(a):
            return dict(v=pscale(a) + s10(a) + s100(a) + Model().lr)
        """
    mod = _module(root, "mushin_t_wrapped", src.format(lr=0.01, factor=10, k10=10))
    try:
        before = project_fingerprints(mod.task)
        for edit, key in (
            (dict(lr=0.5, factor=10, k10=10), "mushin_t_wrapped:Model"),
            (dict(lr=0.01, factor=99, k10=10), "mushin_t_wrapped:pscale"),
            (dict(lr=0.01, factor=10, k10=20), "mushin_t_wrapped:s10.<k>"),
        ):
            mod = _module(root, "mushin_t_wrapped", src.format(**edit))
            assert key in _changed(before, mod.task), edit
    finally:
        sys.modules.pop("mushin_t_wrapped", None)


def test_class_bookkeeping_is_not_code(project):
    """3.13+ stores __firstlineno__ on every class: a comment above it would
    otherwise be an edit. Dunder values (module, doc, line) are not code."""
    root, _ = project
    mod = _module(
        root,
        "mushin_t_cls",
        "class M:\n    def f(self):\n        return 1\n\n\ndef task(a):\n    return dict(v=M().f())\n",
    )
    try:
        before = project_fingerprints(mod.task)
        mod.M.__firstlineno__ = 999
        mod.M.__doc__ = "now documented"
        assert _changed(before, mod.task) == []
    finally:
        sys.modules.pop("mushin_t_cls", None)


def test_an_attribute_named_like_a_global_is_not_that_global(project):
    """`random.seed(a)` next to a module-level `seed = 0`: the attribute is not
    the global, so changing the global is not an edit to the task."""
    root, _ = project
    src = "import random\n\nseed = {seed}\n\n\ndef task(a):\n    random.seed(a)\n    return dict(v=a)\n"
    mod = _module(root, "mushin_t_attr", src.format(seed=0))
    try:
        before = project_fingerprints(mod.task)
        assert "mushin_t_attr:seed" not in before["code"]
        mod = _module(root, "mushin_t_attr", src.format(seed=1))
        assert _changed(before, mod.task) == []
    finally:
        sys.modules.pop("mushin_t_attr", None)


def test_relative_import_levels_are_read_on_every_python():
    """3.14 loads the level with LOAD_SMALL_INT; a large int constant earlier in
    the body must not be taken for it."""
    from mushin._resume import _code_names

    ns = {}
    exec(
        "def f():\n    n = 100000\n    import helpers\n    from .lazy import K\n    from ..up import U\n",
        ns,
    )
    _, _, imports = _code_names(ns["f"].__code__)
    assert ("helpers", 0) in imports and ("lazy", 1) in imports and ("up", 2) in imports


def test_roots_from_two_packages_with_the_same_file_name_do_not_collide(project):
    """Study methods in pa/train.py and pb/train.py: keys use import names."""
    root, _ = project
    for pkg, lr in (("mushin_t_pa", 0.1), ("mushin_t_pb", 0.5)):
        (root / pkg).mkdir()
        (root / pkg / "__init__.py").write_text("")
        (root / pkg / "train.py").write_text(
            f"LR = {lr}\n\n\ndef train(seed):\n    return LR\n"
        )
    try:
        a = importlib.import_module("mushin_t_pa.train")
        b = importlib.import_module("mushin_t_pb.train")
        code = project_fingerprints(a.train, b.train)["code"]
        assert {"mushin_t_pa.train:LR", "mushin_t_pb.train:LR"} <= set(code)
    finally:
        for n in (
            "mushin_t_pa",
            "mushin_t_pa.train",
            "mushin_t_pb",
            "mushin_t_pb.train",
        ):
            sys.modules.pop(n, None)


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


def test_pre_task_and_config_targets_are_roots(project):
    """pre_task's body and what it calls, and the classes the config names as
    `_target_` (hydra-zen's builds), are code a cell runs."""
    from mushin._resume import cell_project_record

    root, _ = project
    src = """
        SEED_OFFSET = {off}


        def pre_task(seed):
            return seed + SEED_OFFSET


        class Model:
            def forward(self, a):
                return a * {k}


        def task(model, a):
            return dict(v=model.forward(a))
        """
    mod = _module(root, "mushin_t_roots", src.format(off=0, k=2))
    cfg = {"model": {"_target_": "mushin_t_roots.Model"}, "a": 1}
    try:
        before = cell_project_record("tok-a", (mod.task,), (mod.pre_task,), cfg)
        assert {"mushin_t_roots:pre_task", "mushin_t_roots (file)"} <= set(
            before["code"]
        )
        mod = _module(root, "mushin_t_roots", src.format(off=1, k=2))
        after = cell_project_record("tok-b", (mod.task,), (mod.pre_task,), cfg)
        # The target's module is this file, so its file entry moves too.
        assert "mushin_t_roots:SEED_OFFSET" in changed_project_code(before, after)
        mod = _module(root, "mushin_t_roots", src.format(off=0, k=3))
        after = cell_project_record("tok-c", (mod.task,), (mod.pre_task,), cfg)
        assert "mushin_t_roots (file)" in changed_project_code(before, after)
    finally:
        sys.modules.pop("mushin_t_roots", None)


def test_a_reused_cell_keeps_its_prior_record():
    """Only the keys both records share were compared, so a reused cell keeps
    its prior record: writing keys only the current run has would declare it
    computed by code it was never checked against. A cell with no record takes
    the current one."""
    from mushin._resume import merge_project_records

    prior = {"python": PY, "code": {"pkg.lazy (file)": "1", "m:f": "2"}}
    current = {"python": PY, "code": {"m:f": "2", "m:new": "3"}}
    assert merge_project_records(prior, current) == prior
    assert merge_project_records(prior, None) == prior
    assert merge_project_records(None, current) == current
    # A record from another Python version can never be compared again: the
    # current one replaces it, so the cell is guarded from now on.
    other = {"python": "0.0", "code": {"m:f": "9"}}
    assert merge_project_records(other, current) == current


def test_a_method_default_is_part_of_its_class(project):
    """`def scale(self, x, factor=2)` to `factor=20` leaves the code object as it
    was; the default is what changed (Codex, #270)."""
    root, _ = project
    src = "class Scaler:\n    def scale(self, x, factor={f}):\n        return x * factor\n\n\ndef task(a):\n    return dict(v=Scaler().scale(a))\n"
    mod = _module(root, "mushin_t_defaults", src.format(f=2))
    try:
        before = project_fingerprints(mod.task)
        mod = _module(root, "mushin_t_defaults", src.format(f=20))
        assert "mushin_t_defaults:Scaler" in _changed(before, mod.task)
    finally:
        sys.modules.pop("mushin_t_defaults", None)


# --- round 5 --------------------------------------------------------------------


def test_names_the_project_rebinds_are_state_not_code(project):
    """A lazily created singleton (`_model = None`, set on first use) and a
    warn-once flag change as the code runs: hashing them would re-run every
    cell on an in-process resume."""
    root, _ = project
    src = """
        _model = None
        _warned = False


        def get_model():
            global _model
            if _model is None:
                _model = object()
            return _model


        def warn_once():
            global _warned
            _warned = True


        def task(a):
            get_model()
            warn_once()
            return dict(v=float(a) + (_model is not None) + _warned)
        """
    mod = _module(root, "mushin_t_state", src)
    try:
        before = project_fingerprints(mod.task)
        mod.task(1)  # runs: _model and _warned change
        assert _changed(before, mod.task) == []
        assert {"mushin_t_state:_model", "mushin_t_state:_warned"} <= set(
            before["code"]
        )
    finally:
        sys.modules.pop("mushin_t_state", None)


def test_secret_shaped_names_and_values_are_never_hashed(project):
    """A sidecar never holds a hash of a secret, which a dictionary attack on
    a short one could undo."""
    from mushin._resume import _SECRET

    root, _ = project
    src = 'API_KEY = "hunter2"\nURL = "https://x"\nTOKENISH = "sk-abcdefghijklmnop"\n\n\ndef task(a):\n    return dict(v=len(API_KEY + URL + TOKENISH) * a)\n'
    mod = _module(root, "mushin_t_secret", src)
    try:
        code = project_fingerprints(mod.task)["code"]
        assert code["mushin_t_secret:API_KEY"] == _SECRET
        assert code["mushin_t_secret:TOKENISH"] == _SECRET  # secret-shaped value
        assert code["mushin_t_secret:URL"] != _SECRET
    finally:
        sys.modules.pop("mushin_t_secret", None)


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


def test_a_project_decorator_on_the_task_and_callable_defaults_are_code(project):
    root, _ = project
    src = """
        import functools


        def timed(fn):
            @functools.wraps(fn)
            def wrapper(*a, **k):
                return fn(*a, **k) * {mult}

            return wrapper


        def double(x):
            return x * {d}


        @timed
        def task(a, post=double):
            return dict(v=post(a))
        """
    mod = _module(root, "mushin_t_deco", textwrap.dedent(src).format(mult=1, d=2))
    try:
        before = project_fingerprints(mod.task)
        mod = _module(root, "mushin_t_deco", textwrap.dedent(src).format(mult=3, d=2))
        # functools.wraps gives the wrapper the task's name: the layer is keyed
        # as the task, and its body (the decorator's code) is what changed.
        assert "mushin_t_deco:task" in _changed(before, mod.task)
        mod = _module(root, "mushin_t_deco", textwrap.dedent(src).format(mult=1, d=5))
        assert any("<default" in k for k in _changed(before, mod.task))
    finally:
        sys.modules.pop("mushin_t_deco", None)


def test_a_frozen_settings_dataclass_is_hashed_by_its_fields(project):
    root, _ = project
    src = "import dataclasses\n\n\n@dataclasses.dataclass(frozen=True)\nclass Settings:\n    lr: float = 0.1\n\n\nSETTINGS = Settings(lr={lr})\n\n\ndef task(a):\n    return dict(v=a * SETTINGS.lr)\n"
    mod = _module(root, "mushin_t_settings", src.format(lr=0.1))
    try:
        before = project_fingerprints(mod.task)
        mod = _module(root, "mushin_t_settings", src.format(lr=0.5))
        assert "mushin_t_settings:SETTINGS" in _changed(before, mod.task)
    finally:
        sys.modules.pop("mushin_t_settings", None)


def test_the_same_body_from_another_module_is_a_change(project):
    """`from a import f` switched to `from b import f`, both `return x * 2`:
    where the function is defined is part of what is called."""
    root, _ = project
    _module(root, "mushin_t_fa", "def f(x):\n    return x * 2\n")
    _module(root, "mushin_t_fb", "def f(x):\n    return x * 2\n")
    src = "from {m} import f\n\n\ndef task(a):\n    return dict(v=f(a))\n"
    mod = _module(root, "mushin_t_swap", src.format(m="mushin_t_fa"))
    try:
        before = project_fingerprints(mod.task)
        mod = _module(root, "mushin_t_swap", src.format(m="mushin_t_fb"))
        assert _changed(before, mod.task) == ["mushin_t_swap:f"]
    finally:
        for n in ("mushin_t_fa", "mushin_t_fb", "mushin_t_swap"):
            sys.modules.pop(n, None)


def test_a_call_chain_deeper_than_the_recursion_limit_is_walked(project):
    """The walk is iterative: a project with a call chain of thousands of
    functions is fingerprinted in full, not cut off where the stack ends."""
    root, _ = project
    n = sys.getrecursionlimit() + 200
    lines = [f"def f{i}(x):\n    return f{i + 1}(x)\n" for i in range(n)]
    lines.append(f"def f{n}(x):\n    return x * {{k}}\n")
    lines.append("def task(a):\n    return dict(v=f0(a))\n")
    src = "\n\n".join(lines)
    mod = _module(root, "mushin_t_deep", src.format(k=2))
    try:
        before = project_fingerprints(mod.task)
        assert f"mushin_t_deep:f{n}" in before["code"]
        mod = _module(root, "mushin_t_deep", src.format(k=3))
        assert _changed(before, mod.task) == [f"mushin_t_deep:f{n}"]
    finally:
        sys.modules.pop("mushin_t_deep", None)


# --- round 6 --------------------------------------------------------------------


def test_numeric_settings_under_secret_shaped_names_are_hashed(project):
    """`MAX_TOKENS = 512` is a setting, not a secret: only strings under such a
    name get the stand-in."""
    from mushin._resume import _SECRET

    root, _ = project
    src = "MAX_TOKENS = {n}\nAPI_KEY = 'x'\n\n\ndef task(a):\n    return dict(v=a * MAX_TOKENS + len(API_KEY))\n"
    mod = _module(root, "mushin_t_tokens", src.format(n=512))
    try:
        before = project_fingerprints(mod.task)
        assert before["code"]["mushin_t_tokens:MAX_TOKENS"] != _SECRET
        assert before["code"]["mushin_t_tokens:API_KEY"] == _SECRET
        mod = _module(root, "mushin_t_tokens", src.format(n=4096))
        assert _changed(before, mod.task) == ["mushin_t_tokens:MAX_TOKENS"]
    finally:
        sys.modules.pop("mushin_t_tokens", None)


def test_a_lazily_imported_helper_is_followed_to_its_own_imports(project):
    """`from lazyh import scale` inside the task, where lazyh imports the
    constant it uses from consts; and `from pkg import sub` names the
    submodule, not only the package's empty __init__."""
    root, _ = project
    (root / "mushin_t_consts2.py").write_text("THRESH = 10\n")
    (root / "mushin_t_lazyh.py").write_text(
        "from mushin_t_consts2 import THRESH\n\n\ndef scale(x):\n    return x * THRESH\n"
    )
    (root / "mushin_t_pk").mkdir()
    (root / "mushin_t_pk" / "__init__.py").write_text("")
    (root / "mushin_t_pk" / "sub.py").write_text("def f(x):\n    return x + 1\n")
    src = "def task(a):\n    from mushin_t_lazyh import scale\n    from mushin_t_pk import sub\n\n    return dict(v=scale(a) + sub.f(a))\n"
    mod = _module(root, "mushin_t_lazyuser2", src)
    try:
        code = project_fingerprints(mod.task)["code"]
        assert {
            "mushin_t_lazyh (file)",
            "mushin_t_consts2 (file)",
            "mushin_t_pk.sub (file)",
        } <= set(code)
        before = project_fingerprints(mod.task)
        (root / "mushin_t_consts2.py").write_text("THRESH = 100\n")
        assert _changed(before, mod.task) == ["mushin_t_consts2 (file)"]
    finally:
        for n in list(sys.modules):
            if n.startswith(
                (
                    "mushin_t_consts2",
                    "mushin_t_lazyh",
                    "mushin_t_pk",
                    "mushin_t_lazyuser2",
                )
            ):
                sys.modules.pop(n, None)


def test_an_alias_pointed_at_another_module_is_a_change(project):
    root, _ = project
    _module(root, "mushin_t_impl_a", "def scale(x):\n    return x * 2\n")
    _module(root, "mushin_t_impl_b", "def scale(x):\n    return x * 3\n")
    src = "import {m} as impl\n\n\ndef task(a):\n    return dict(v=impl.scale(a))\n"
    mod = _module(root, "mushin_t_alias", src.format(m="mushin_t_impl_a"))
    try:
        before = project_fingerprints(mod.task)
        mod = _module(root, "mushin_t_alias", src.format(m="mushin_t_impl_b"))
        assert "mushin_t_alias:impl" in _changed(before, mod.task)
    finally:
        for n in ("mushin_t_impl_a", "mushin_t_impl_b", "mushin_t_alias"):
            sys.modules.pop(n, None)


def test_a_decorators_globals_belong_to_the_decorator_module(project):
    """A functools.wraps wrapper copies __module__ from what it wraps; its
    globals are its own module's, so deco.CONFIG and the task module's CONFIG
    are two keys, and an edit to the decorator's shows."""
    root, _ = project
    deco_src = "import functools\n\nCONFIG = {c}\n\n\ndef scaled(fn):\n    @functools.wraps(fn)\n    def wrapper(x):\n        return fn(x) * CONFIG\n\n    return wrapper\n"
    _module(root, "mushin_t_deco2", deco_src.format(c=1))
    src = "from mushin_t_deco2 import scaled\n\nCONFIG = 2\n\n\n@scaled\ndef helper(x):\n    return x + CONFIG\n\n\ndef task(a):\n    return dict(v=helper(a))\n"
    mod = _module(root, "mushin_t_decouser", src)
    try:
        before = project_fingerprints(mod.task)
        assert {"mushin_t_deco2:CONFIG", "mushin_t_decouser:CONFIG"} <= set(
            before["code"]
        )
        _module(root, "mushin_t_deco2", deco_src.format(c=100))
        mod = _module(root, "mushin_t_decouser", src)
        assert "mushin_t_deco2:CONFIG" in _changed(before, mod.task)
    finally:
        for n in ("mushin_t_deco2", "mushin_t_decouser"):
            sys.modules.pop(n, None)


def test_settings_objects_are_hashed_by_value(project):
    """A module-level frozen dataclass instance, with a path and a nested
    settings object among its fields, is hashed by value; a mutable one is
    state (it may change as the code runs) and is not."""
    root, _ = project
    src = """
        import dataclasses
        from pathlib import Path


        @dataclasses.dataclass(frozen=True)
        class Opt:
            lr: float = 0.01


        @dataclasses.dataclass(frozen=True)
        class Settings:
            lr: float
            data: Path
            opt: Opt


        SETTINGS = Settings(lr={lr}, data=Path("d"), opt=Opt({olr}))


        def task(a):
            return dict(v=a * SETTINGS.lr * SETTINGS.opt.lr)
        """
    mod = _module(
        root, "mushin_t_settings2", textwrap.dedent(src).format(lr=0.1, olr=0.01)
    )
    try:
        before = project_fingerprints(mod.task)
        mod = _module(
            root, "mushin_t_settings2", textwrap.dedent(src).format(lr=0.5, olr=0.01)
        )
        assert "mushin_t_settings2:SETTINGS" in _changed(before, mod.task)
        mod = _module(
            root, "mushin_t_settings2", textwrap.dedent(src).format(lr=0.1, olr=0.02)
        )
        assert "mushin_t_settings2:SETTINGS" in _changed(before, mod.task)
    finally:
        sys.modules.pop("mushin_t_settings2", None)


def test_a_set_of_functions_is_keyed_by_name_not_by_address(project):
    """A set iterates in hash order, which for functions follows memory
    addresses: its items are keyed by name, the same in every process."""
    root, _ = project
    src = "def h1(x):\n    return x\n\n\ndef h2(x):\n    return -x\n\n\nHANDLERS = {h1, h2}\nBY_FN = {h1: 1, h2: 2}\n\n\ndef task(a):\n    return dict(v=sum(h(a) for h in HANDLERS) + sum(BY_FN.values()))\n"
    mod = _module(root, "mushin_t_sets", src)
    try:
        code = project_fingerprints(mod.task)["code"]
        assert "mushin_t_sets:HANDLERS[mushin_t_sets.h1]" in code
        assert not any("0x" in k for k in code)
    finally:
        sys.modules.pop("mushin_t_sets", None)


def test_secret_shaped_values_inside_partials_and_tuples_are_never_hashed(project):
    from mushin._resume import _constant_repr

    root, _ = project
    src = "import functools\n\n\ndef fetch(x, api_key=None):\n    return x\n\n\ncall = functools.partial(fetch, api_key='{key}')\nCREDS = ('user', 'sk-abcdefghijklmnop')\n\n\ndef task(a):\n    return dict(v=call(a) + len(CREDS))\n"
    mod = _module(root, "mushin_t_psecret", src.format(key="sk-aaaaaaaaaaaaaaaa"))
    try:
        before = project_fingerprints(mod.task)
        mod = _module(root, "mushin_t_psecret", src.format(key="sk-bbbbbbbbbbbbbbbb"))
        # The key's value is not part of any hash: changing it changes nothing.
        assert _changed(before, mod.task) == []
        assert "sk-" not in _constant_repr(("user", "sk-abcdefghijklmnop"))
    finally:
        sys.modules.pop("mushin_t_psecret", None)


def test_a_partial_task_is_covered_by_its_function_and_bound_arguments(project):
    """code_fingerprint of a partial is None (getsource rejects it): the run
    hashes the function it wraps, and the record holds the bound arguments."""
    from mushin._resume import _unwrap_task

    root, _ = project
    src = "import functools\n\n\ndef train(a, epochs=1):\n    return dict(v=float(a * epochs))\n\n\ntask = functools.partial(train, epochs={e})\n"
    mod = _module(root, "mushin_t_ptask", src.format(e=10))
    try:
        assert code_fingerprint(_unwrap_task(mod.task)) is not None
        before = project_fingerprints(mod.task)
        mod = _module(root, "mushin_t_ptask", src.format(e=20))
        assert "mushin_t_ptask:train.<partial>" in _changed(before, mod.task)
    finally:
        sys.modules.pop("mushin_t_ptask", None)


def test_state_rebound_inside_a_decorated_function_is_state(project):
    root, _ = project
    src = """
        import functools

        _MODEL = None


        def log_calls(fn):
            @functools.wraps(fn)
            def wrapper(*a):
                return fn(*a)

            return wrapper


        @log_calls
        def init():
            global _MODEL
            _MODEL = object()


        def task(a):
            init()
            return dict(v=float(a) + (_MODEL is not None))
        """
    mod = _module(root, "mushin_t_dstate", src)
    try:
        before = project_fingerprints(mod.task)
        mod.task(1)
        assert _changed(before, mod.task) == []
    finally:
        sys.modules.pop("mushin_t_dstate", None)


def test_a_docstring_is_not_an_edit(project):
    root, _ = project
    src = 'def helper(x):\n    """{doc}"""\n    return x * 2\n\n\ndef task(a):\n    return dict(v=helper(a))\n'
    mod = _module(root, "mushin_t_doc", src.format(doc="Scale x."))
    try:
        before = project_fingerprints(mod.task)
        mod = _module(root, "mushin_t_doc", src.format(doc="Scale x by two."))
        assert _changed(before, mod.task) == []
    finally:
        sys.modules.pop("mushin_t_doc", None)


def test_main_is_never_foreign_and_labels_follow_the_functions_globals():
    """In a parallel worker sys.modules['__main__'] is the launcher's module:
    a function from the user's script carries its script's globals."""
    from mushin._resume import _is_foreign_module, _module_label

    assert not _is_foreign_module("__main__")
    g = {"__name__": "__main__", "__file__": "/proj/exp_script.py", "__spec__": None}
    assert _module_label("__main__", g) == "exp_script"
    g_notebook = {"__name__": "__main__"}
    assert _module_label("__main__", g_notebook) == "__main__"


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


JOBLIB_SCRIPT = """
import sys

from mushin import multirun
from mushin.workflows import MultiRunMetricsWorkflow

OFFSET = {offset}


def helper(a):
    return a + OFFSET


class W(MultiRunMetricsWorkflow):
    @staticmethod
    def task(a):
        with open({calls!r}, "a") as f:
            f.write("x")
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


@pytest.mark.parametrize("second", ["joblib", "basic"])
def test_a_script_task_under_joblib_records_the_code_its_workers_run(tmp_path, second):
    """In a loky worker, sys.modules['__main__'] is loky's own module, while the
    task's functions carry their script's globals: the script's helpers and
    constants are project code there too, labelled as in the driver, so a
    joblib resume (or a sequential one after a joblib run) sees an edit."""
    pytest.importorskip("hydra_plugins.hydra_joblib_launcher")
    root = tmp_path / "jl"
    root.mkdir()
    calls = root / "calls.txt"
    script = root / "jl_script.py"

    def write(offset):
        script.write_text(
            JOBLIB_SCRIPT.format(offset=offset, calls=str(calls), wd=str(root / "wd"))
        )

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

    write(0)
    run("joblib")
    assert calls.read_text() == "xx"
    err = run(second, "resume")
    assert "fingerprint mismatch" not in err, err
    assert calls.read_text() == "xx"  # unchanged: reused
    write(100)
    err = run(second, "resume")
    assert "code the task reaches (jl_script:OFFSET)" in err, err
    assert calls.read_text() == "xxxx"


# --- round 7: the walk is bounded and prefers missing an edit to a re-run ------


def _bounded(fn, seconds=20):
    """project_fingerprints(fn) in a thread, failing if it does not finish."""
    import threading

    out = {}
    t = threading.Thread(
        target=lambda: out.setdefault("r", project_fingerprints(fn)), daemon=True
    )
    t.start()
    t.join(seconds)
    assert not t.is_alive(), "the walk did not finish"
    return out["r"]


def test_a_closure_that_holds_itself_does_not_hang_the_walk(project):
    """A call-counter decorator and a recursive inner function: the closure
    holds the function itself (Codex-style hang, adversarial round 6)."""
    root, _ = project
    src = """
        import functools


        def counted(fn):
            @functools.wraps(fn)
            def wrapper(*a, **k):
                wrapper.calls += 1
                return fn(*a, **k)

            wrapper.calls = 0
            return wrapper


        def make():
            def depth(n):
                return 0 if not n else 1 + depth(n[1:])

            return depth


        DEPTH = make()


        @counted
        def scale(x):
            return x * 2


        def task(a):
            return dict(v=scale(a) + DEPTH("ab"))
        """
    mod = _module(root, "mushin_t_selfref", textwrap.dedent(src))
    try:
        assert "mushin_t_selfref:DEPTH" in _bounded(mod.task)["code"]
    finally:
        sys.modules.pop("mushin_t_selfref", None)


def test_an_object_that_invents_attributes_does_not_hang_the_walk(project):
    """addict.Dict and its kind return a new object for any missing attribute,
    __wrapped__ included."""
    root, _ = project
    src = """
        class AttrDict(dict):
            def __getattr__(self, name):
                return self[name] if name in self else type(self)()


        CFG = AttrDict(lr=0.1)


        def task(a):
            return dict(v=a * CFG.lr)
        """
    mod = _module(root, "mushin_t_addict", textwrap.dedent(src))
    try:
        _bounded(mod.task)
    finally:
        sys.modules.pop("mushin_t_addict", None)


def test_a_walk_past_its_bound_stops_at_the_same_place_in_every_process(
    project, monkeypatch
):
    import mushin._resume as r

    root, _ = project
    n = 300
    lines = [f"def f{i}(x):\n    return f{i + 1}(x)\n" for i in range(n)]
    lines.append(f"def f{n}(x):\n    return x\n")
    lines.append("def task(a):\n    return dict(v=f0(a))\n")
    mod = _module(root, "mushin_t_bound", "\n\n".join(lines))
    monkeypatch.setattr(r, "_WALK_MAX_OPS", 100)
    try:
        first = project_fingerprints(mod.task)
        assert 0 < len(first["code"]) < n
        assert project_fingerprints(mod.task) == first
    finally:
        sys.modules.pop("mushin_t_bound", None)


def test_secret_patterns_do_not_hide_ordinary_code(project):
    """Names and strings that only look like provider tokens are settings and
    are compared; real tokens are not."""
    from mushin._resume import _SECRET

    root, _ = project
    src = 'MODEL = "mask-rcnn_r50_fpn_1x_coco"\nCACHE_DIR = "/data/hf_cache/models"\nAUTHOR = "me"\nHF_TOKEN = "hf_abcdefghijklmnopqrst"\nVALUEISH = "sk-abcdefghijklmnopqrst"\n\n\ndef task(a):\n    return dict(v=a * len(MODEL + CACHE_DIR + AUTHOR + HF_TOKEN + VALUEISH))\n'
    mod = _module(root, "mushin_t_secret2", src)
    try:
        code = project_fingerprints(mod.task)["code"]
        for name in ("MODEL", "CACHE_DIR", "AUTHOR"):
            assert code[f"mushin_t_secret2:{name}"] != _SECRET, name
        assert code["mushin_t_secret2:HF_TOKEN"] == _SECRET  # by name
        assert code["mushin_t_secret2:VALUEISH"] == _SECRET  # by value alone
    finally:
        sys.modules.pop("mushin_t_secret2", None)


def test_a_registry_holds_code_not_entries_for_its_other_contents(project):
    """A registry filled as the code runs must not move the record: only the
    project code it holds is walked, and the container itself is no entry."""
    root, _ = project
    src = 'def cnn(x):\n    return x\n\n\nMODELS = {"cnn": cnn}\nHISTORY = [(1, 2)]\n\n\ndef task(a):\n    HISTORY.append((a, a))\n    MODELS["x" + str(a)] = 3\n    return dict(v=MODELS["cnn"](a))\n'
    mod = _module(root, "mushin_t_reg2", src)
    try:
        before = project_fingerprints(mod.task)
        assert "mushin_t_reg2:MODELS['cnn']" in before["code"]
        mod.task(1)
        assert project_fingerprints(mod.task) == before
    finally:
        sys.modules.pop("mushin_t_reg2", None)


# --- final round -----------------------------------------------------------


def test_a_computed_constant_reads_the_same_on_every_host(project, monkeypatch):
    """`N = os.cpu_count()` or `DEVICE = "cuda" if ... else "cpu"` differs from
    node to node; it is recorded by its source, so a resume on another node
    does not re-run unchanged cells, and editing the expression still shows."""
    root, _ = project
    src = 'import os\n\nN = os.cpu_count() {op} 1\nDEVICE = "cuda" if os.environ.get("MUSHIN_T_GPU") else "cpu"\n\n\ndef task(a):\n    return dict(v=float(a * N) + len(DEVICE))\n'
    mod = _module(root, "mushin_t_host", src.format(op="+"))
    try:
        before = project_fingerprints(mod.task)
        monkeypatch.setattr(os, "cpu_count", lambda: 999)
        monkeypatch.setenv("MUSHIN_T_GPU", "1")
        mod = _module(root, "mushin_t_host", src.format(op="+"))  # another node
        assert _changed(before, mod.task) == []
        mod = _module(root, "mushin_t_host", src.format(op="-"))
        assert _changed(before, mod.task) == ["mushin_t_host:N"]
    finally:
        sys.modules.pop("mushin_t_host", None)


def test_a_class_value_filled_at_runtime_is_not_an_edit(project):
    """A class-level cache a classmethod fills on first use: the class body's
    source is what is compared, so an in-process resume reuses the cells; an
    edit to a class constant still shows."""
    root, _ = project
    src = "class Scaler:\n    FACTOR = {f}\n    _table = None\n\n    @classmethod\n    def scale(cls, x):\n        if cls._table is None:\n            cls._table = {{1: 2}}\n        return x * cls.FACTOR\n\n\ndef task(a):\n    return dict(v=Scaler.scale(a))\n"
    mod = _module(root, "mushin_t_clscache", src.format(f=2))
    try:
        before = project_fingerprints(mod.task)
        mod.task(1)
        assert _changed(before, mod.task) == []
        mod = _module(root, "mushin_t_clscache", src.format(f=3))
        assert "mushin_t_clscache:Scaler" in _changed(before, mod.task)
    finally:
        sys.modules.pop("mushin_t_clscache", None)


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


def test_a_driver_script_target_does_not_hash_the_script():
    """builds(Net) in a script names __main__.Net; recording the whole script
    would make an edit to its run block an edit."""
    from mushin._resume import _CodeWalk

    walk = _CodeWalk()
    walk.walk_target("__main__.Net")
    assert walk.hashes == {}


def test_a_docstring_added_to_an_undocumented_helper_is_not_an_edit(project):
    root, _ = project
    src = "def helper(x):\n{doc}    return x * 2\n\n\ndef task(a):\n    return dict(v=helper(a))\n"
    mod = _module(root, "mushin_t_doc2", src.format(doc=""))
    try:
        before = project_fingerprints(mod.task)
        mod = _module(
            root, "mushin_t_doc2", src.format(doc='    """Now documented."""\n')
        )
        assert _changed(before, mod.task) == []
    finally:
        sys.modules.pop("mushin_t_doc2", None)


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


def test_a_walk_past_its_bound_warns_once(project, monkeypatch):
    import mushin._resume as r

    root, _ = project
    lines = [f"def f{i}(x):\n    return f{i + 1}(x)\n" for i in range(50)]
    lines.append("def f50(x):\n    return x\n")
    lines.append("def task(a):\n    return dict(v=f0(a))\n")
    mod = _module(root, "mushin_t_bound2", "\n\n".join(lines))
    monkeypatch.setattr(r, "_WALK_MAX_OPS", 10)
    monkeypatch.setattr(r, "_TRUNCATION_WARNED", [])
    try:
        with pytest.warns(UserWarning, match="larger than the fingerprint walks"):
            project_fingerprints(mod.task)
    finally:
        sys.modules.pop("mushin_t_bound2", None)
