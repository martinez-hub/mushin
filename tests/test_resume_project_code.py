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
    assert f"{EXP}:helpers" not in code  # a module is not an entry, its names are
    assert f"{HELPERS}:SCALE" in code  # read by scale, reached through it
    assert not [k for k in code if k.startswith(("mushin.", "numpy", "json"))]
    # The root's own body is code_fingerprint's: not an entry here.
    assert f"{EXP}:W.task" not in code
    assert _code(functools.partial(exp.W.task, a=1)) == code


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
    """A loaded module imported in the task's body is walked like a global; one
    this process has not imported yet is hashed by its file, without importing
    it. A relative import resolves against the task's package."""
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
        unloaded = _code(mod.task)
        assert "mushin_t_pkg.lazy (file)" in unloaded
        assert "mushin_t_pkg.lazy" not in sys.modules  # found without importing
        importlib.import_module("mushin_t_pkg.lazy")
        loaded = _code(mod.task)
        assert "mushin_t_pkg.lazy:K" in loaded
        # A process that had imported it and one that had not share no key for
        # it, so neither reads as a change to the other.
        assert (
            changed_project_code(
                {"python": PY, "code": unloaded}, {"python": PY, "code": loaded}
            )
            == []
        )
    finally:
        for name in ("mushin_t_pkg", "mushin_t_pkg.lazy", "mushin_t_pkg.exp"):
            sys.modules.pop(name, None)


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

    monkeypatch.setattr(r, "_installed_top_levels", lambda: frozenset({"clusterlib"}))
    r._is_foreign_module.cache_clear()
    try:
        assert r._is_foreign_module("clusterlib.ops")
        assert not r._is_foreign_module("mylab_project.models")
    finally:
        r._is_foreign_module.cache_clear()


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
    """A task that reaches no project code keeps main's key, so its entries from
    before this change are still found; one that does gets another; one whose
    project code could not be fingerprinted does not use the cache."""
    from mushin.workflows import _TaskRunner

    def runner(deps):
        return _TaskRunner(
            task=None,
            pre_task=None,
            swept_names=(),
            base_provenance=None,
            on_error="raise",
            inject_resume=False,
            prior_manifest=None,
            code_hash="c0de",
            cache_dir="cache",
            code_deps=deps,
        )

    two_part = hashlib.sha256(b"h:c0de").hexdigest()[:32]
    assert runner({"python": PY, "code": {}})._cache_key("h") == two_part
    assert runner({"python": PY, "code": {"m:f": "1"}})._cache_key("h") != two_part
    assert runner(None)._cache_key("h") is None


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
