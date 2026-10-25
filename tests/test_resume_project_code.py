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

    monkeypatch.setattr(
        r, "_installed_roots", lambda: {"clusterlib": (pathlib.Path("/x"),)}
    )
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


def test_a_name_that_stops_being_hashable_is_still_an_edit(project):
    """`LAYERS = 3` becoming a list, or a helper becoming a foreign function,
    keeps its key with a stand-in hash, so the edit shows."""
    root, _ = project
    src = """
        import math

        LAYERS = {layers}
        {act}


        def task(a):
            return dict(v=act(a) * len(str(LAYERS)))
        """
    mod = _module(
        root,
        "mushin_t_kinds",
        textwrap.dedent(src).format(layers=3, act="def act(x):\n    return abs(x)"),
    )
    try:
        before = project_fingerprints(mod.task)
        mod = _module(
            root,
            "mushin_t_kinds",
            textwrap.dedent(src).format(layers="[64, 64]", act="act = math.fabs"),
        )
        assert set(_changed(before, mod.task)) == {
            "mushin_t_kinds:LAYERS",
            "mushin_t_kinds:act",
        }
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
    r._is_foreign_module.cache_clear()
    try:
        assert not r._is_foreign_module(EXP)  # its file is in the project
        monkeypatch.setattr(r, "_installed_roots", lambda: {EXP: (root.resolve(),)})
        r._is_foreign_module.cache_clear()
        assert r._is_foreign_module(EXP)  # its file is inside the install dir
    finally:
        r._is_foreign_module.cache_clear()


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
        assert {"mushin_t_roots:pre_task", "target:mushin_t_roots.Model"} <= set(
            before["code"]
        )
        mod = _module(root, "mushin_t_roots", src.format(off=1, k=2))
        after = cell_project_record("tok-b", (mod.task,), (mod.pre_task,), cfg)
        assert changed_project_code(before, after) == ["mushin_t_roots:SEED_OFFSET"]
        mod = _module(root, "mushin_t_roots", src.format(off=0, k=3))
        after = cell_project_record("tok-c", (mod.task,), (mod.pre_task,), cfg)
        assert "target:mushin_t_roots.Model" in changed_project_code(before, after)
    finally:
        sys.modules.pop("mushin_t_roots", None)


def test_a_reused_cell_keeps_the_keys_only_its_prior_record_had():
    """A key only one kind of process reaches (a lazily imported module hashed
    by file in a fresh process) stays, so a later resume can compare it."""
    from mushin._resume import merge_project_records

    prior = {"python": PY, "code": {"pkg.lazy (file)": "1", "m:f": "2"}}
    current = {"python": PY, "code": {"pkg.lazy:K": "3", "m:f": "2"}}
    merged = merge_project_records(prior, current)
    assert merged["code"] == {"pkg.lazy (file)": "1", "pkg.lazy:K": "3", "m:f": "2"}
    assert merge_project_records(prior, None) == prior
    assert merge_project_records(None, current) == current
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
