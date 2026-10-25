# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
"""resume=True and cache_dir= notice edits to the project code a task reaches.

The task's own body was fingerprinted before; a helper it calls, or a constant
in its module, was not, so a resume after editing one returned the stale
result. The fingerprint now covers the module that defines the task and the
project modules it imports, read from disk when ``run()`` starts.
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
from mushin._resume import STATUS_FILE, project_fingerprints

HELPERS = "mushin_t_helpers"
EXP = "mushin_t_exp"
HELPERS_KEY = f"{HELPERS}.py"  # keys are paths relative to the task's module
EXP_KEY = f"{EXP}.py"


def _write(root, scale: int = 10, offset: int = 0):
    (root / f"{HELPERS}.py").write_text(
        textwrap.dedent(f"""
            SCALE = {scale}


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
    Yields the project root and the imported task module; ``reload()`` picks up
    an edit the way a fresh process would."""
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


def _resume(exp, wd, **kwargs):
    wf = exp.W()
    wf.run(a=multirun([1, 2]), working_dir=wd, resume=True, **kwargs)
    return wf.to_xarray()


def test_project_fingerprints_cover_the_task_module_and_its_project_imports(project):
    _, exp = project
    deps = project_fingerprints(exp.W.task)
    # The task's module, the helper it imports; not mushin, numpy or the stdlib.
    assert set(deps) == {EXP_KEY, HELPERS_KEY}
    assert all(len(h) == 16 for h in deps.values())
    assert project_fingerprints(functools.partial(exp.W.task, a=1)) == deps


def test_project_fingerprints_change_with_a_helper_edit_on_disk(project):
    root, exp = project
    before = project_fingerprints(exp.W.task)
    _write(root, scale=100)  # the file changes; the loaded module does not
    after = project_fingerprints(exp.W.task)
    assert after[EXP_KEY] == before[EXP_KEY]
    assert after[HELPERS_KEY] != before[HELPERS_KEY]


def test_project_fingerprints_is_none_for_a_function_without_a_module():
    """A function built at runtime (exec into a module that is not in
    sys.modules) cannot be located: nothing to check, like a missing hash."""
    mod = types.ModuleType("mushin_t_nowhere")
    exec("def task(a):\n    return dict(v=a)", mod.__dict__)
    assert project_fingerprints(mod.task) is None


def test_project_fingerprints_is_empty_for_a_task_inside_mushin():
    from mushin import _show

    assert project_fingerprints(_show.best) == {}


def test_resume_reruns_when_a_helper_module_changes(project):
    root, exp = project
    wd = str(root / "sweep")
    exp.W().run(a=multirun([1, 2]), working_dir=wd)
    _write(root, scale=100)
    exp = _reload(exp)
    with pytest.warns(
        UserWarning, match=rf"code the task reaches \({HELPERS_KEY}\)"
    ) as rec:
        ds = _resume(exp, wd)
    assert float(ds["v"].sel(a=2)) == 200.0  # the edited helper ran
    msgs = [str(w.message) for w in rec if "fingerprint mismatch" in str(w.message)]
    assert msgs and all("task code" not in m for m in msgs)  # only the helper moved


def test_resume_reruns_when_a_constant_in_the_task_module_changes(project):
    root, exp = project
    wd = str(root / "sweep")
    exp.W().run(a=multirun([1, 2]), working_dir=wd)
    _write(root, offset=1)  # the task body is unchanged, the module is not
    exp = _reload(exp)
    with pytest.warns(UserWarning, match=rf"code the task reaches \({EXP_KEY}\)"):
        ds = _resume(exp, wd)
    assert float(ds["v"].sel(a=2)) == 21.0


def test_resume_reuses_when_project_code_is_unchanged(project):
    root, exp = project
    wd = str(root / "sweep")
    exp.W().run(a=multirun([1, 2]), working_dir=wd)
    _write(root, scale=100)  # an edit...
    _write(root)  # ...undone before the resume: same bytes, same fingerprint
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any resume warning fails
        ds = _resume(exp, wd)
    assert float(ds["v"].sel(a=2)) == 20.0


def test_sidecar_records_the_project_fingerprints(project):
    root, exp = project
    wd = root / "sweep"
    exp.W().run(a=multirun([1]), working_dir=str(wd))
    (status,) = wd.rglob(STATUS_FILE)
    recorded = json.loads(status.read_text())["code_deps"]
    assert recorded == project_fingerprints(exp.W.task)


def test_resume_reuses_legacy_sidecars_without_project_fingerprints(project):
    """A sweep recorded before the field existed cannot be checked, which is
    not a reason to re-run: it resumes like one whose code did not change."""
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


def test_cache_dir_misses_after_a_helper_edit_and_hits_when_it_is_undone(project):
    root, exp = project
    cache = str(root / "cache")
    exp.W().run(a=multirun([1, 2]), working_dir=str(root / "s1"), cache_dir=cache)
    _write(root, scale=100)
    exp = _reload(exp)
    wf = exp.W()
    wf.run(a=multirun([1, 2]), working_dir=str(root / "s2"), cache_dir=cache)
    assert float(wf.to_xarray()["v"].sel(a=2)) == 200.0  # a miss: recomputed
    _write(root)
    exp = _reload(exp)
    # In memory only, so a recompute is observable: it would give 2000.0.
    sys.modules[HELPERS].SCALE = 1000
    wf = exp.W()
    wf.run(a=multirun([1, 2]), working_dir=str(root / "s3"), cache_dir=cache)
    assert float(wf.to_xarray()["v"].sel(a=2)) == 20.0  # the first run's entry


def test_project_fingerprints_do_not_depend_on_the_import_name(project):
    """A script run as __main__ and the same file imported by name are one
    project: keys are paths relative to the task's module, not module names."""
    import importlib.util

    root, exp = project
    spec = importlib.util.spec_from_file_location(
        "mushin_t_other_name", root / f"{EXP}.py"
    )
    other = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = other
    try:
        spec.loader.exec_module(other)
        assert project_fingerprints(other.W.task) == project_fingerprints(exp.W.task)
    finally:
        sys.modules.pop(spec.name, None)


def test_project_file_classifies_the_interpreter_and_mushin_as_foreign(project):
    import json as json_mod

    import numpy

    from mushin import _show
    from mushin._resume import _foreign_roots, _project_file

    _, exp = project
    assert _project_file(json_mod) is None  # stdlib
    assert _project_file(numpy) is None  # site-packages
    assert _project_file(_show) is None  # mushin itself
    assert _project_file(exp) is not None
    # The bare prefix is not a root: a project under it is still the project's.
    assert pathlib.Path(sys.prefix).resolve() not in _foreign_roots()


def test_a_file_edited_after_import_is_not_fingerprinted(tmp_path, monkeypatch):
    """With bytecode written, the .pyc header records the source the loaded
    code came from; a later edit on disk means the code that would run is not
    the code on disk, so nothing is recorded under the new hash."""
    from mushin._resume import _loaded_from_older_source

    root = tmp_path / "stale"
    root.mkdir()
    name = "mushin_t_stale"
    path = root / f"{name}.py"
    path.write_text("VALUE = 1\n\n\ndef task(a):\n    return dict(v=a * VALUE)\n")
    monkeypatch.setattr(sys, "dont_write_bytecode", False)
    monkeypatch.syspath_prepend(str(root))
    sys.modules.pop(name, None)
    mod = importlib.import_module(name)
    try:
        assert set(project_fingerprints(mod.task)) == {f"{name}.py"}
        path.write_text(
            "VALUE = 2000\n\n\ndef task(a):\n    return dict(v=a * VALUE)\n"
        )
        assert _loaded_from_older_source(mod, path)
        with pytest.warns(UserWarning, match="edited after this process imported it"):
            assert project_fingerprints(mod.task) is None
    finally:
        sys.modules.pop(name, None)


def test_study_reruns_when_a_method_helper_changes(project):
    """Study's task lives inside mushin; the project code it must notice is the
    user's training functions' modules and their helpers."""
    from mushin.study._sweep import run_training_sweep

    root, exp = project
    (root / "mushin_t_train.py").write_text(
        textwrap.dedent(f"""
            from pathlib import Path

            import {HELPERS} as helpers


            def train(seed):
                p = Path(f"ckpt_{{seed}}.bin")
                p.write_text(str(helpers.scale(seed + 1)))
                return str(p.resolve())
            """)
    )
    sys.modules.pop("mushin_t_train", None)
    train_mod = importlib.import_module("mushin_t_train")
    cwd = root / "cwd"
    cwd.mkdir()
    old = os.getcwd()
    os.chdir(cwd)
    try:
        wd = str(root / "study")
        run_training_sweep(
            {"m": train_mod.train}, seeds=[0], ckpt_dir=root / "ck", working_dir=wd
        )
        _write(root, scale=100)
        importlib.reload(sys.modules[HELPERS])
        train_mod = importlib.reload(train_mod)
        with pytest.warns(UserWarning, match="fingerprint mismatch"):
            ckpts = run_training_sweep(
                {"m": train_mod.train},
                seeds=[0],
                ckpt_dir=root / "ck",
                working_dir=wd,
                resume=True,
            )
        assert pathlib.Path(ckpts["m"][0]).read_text() == "100.0"
    finally:
        os.chdir(old)
        sys.modules.pop("mushin_t_train", None)


def test_project_fingerprints_see_code_not_text(project):
    """Comments, formatting, line endings and a script's run block are not
    edits; a constant is."""
    root, exp = project
    helpers = root / f"{HELPERS}.py"
    before = project_fingerprints(exp.W.task)
    helpers.write_text(helpers.read_text() + "\n# a comment\n")
    assert project_fingerprints(exp.W.task) == before
    helpers.write_bytes(helpers.read_bytes().replace(b"\n", b"\r\n"))
    assert project_fingerprints(exp.W.task) == before
    helpers.write_text(
        helpers.read_text() + 'if __name__ == "__main__":\n    print(scale(1))\n'
    )
    assert project_fingerprints(exp.W.task) == before
    _write(root, scale=100)
    assert project_fingerprints(exp.W.task) != before


def test_resume_judges_only_the_files_both_runs_reached(project):
    """A module reached by one run only (a notebook imported more since) is
    not an edit; a removed one neither (the importing file would have
    changed). The cell is reused without a warning."""
    root, exp = project
    wd = root / "sweep"
    exp.W().run(a=multirun([1, 2]), working_dir=str(wd))
    for status in wd.rglob(STATUS_FILE):
        d = json.loads(status.read_text())
        d["code_deps"]["mushin_t_extra.py"] = "deadbeefdeadbeef"
        d["code_deps"].pop(HELPERS_KEY)
        status.write_text(json.dumps(d))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ds = _resume(exp, str(wd))
    assert float(ds["v"].sel(a=2)) == 20.0


def test_cache_key_keeps_the_two_part_form_without_project_code():
    """Entries written before the project fingerprint stay reachable for a task
    that reaches no project file; one that does gets a different key."""
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
    assert runner(None)._cache_key("h") == two_part
    assert runner({})._cache_key("h") == two_part
    assert runner({"x.py": "1"})._cache_key("h") != two_part


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
        return dict(v=helpers.scale(a) + OFFSET)


if __name__ == "__main__":
    W().run(a=multirun([1, 2]), working_dir={wd!r}, resume="resume" in sys.argv)
    {tail}
"""


def test_a_task_in_the_driver_script_survives_edits_to_the_script(tmp_path):
    """The resilience loop edits the script that defines the task (a flag, a
    print, a comment) and resumes: that must reuse the cells. A constant the
    task reads is an edit."""
    root = tmp_path / "script"
    root.mkdir()
    _write(root)
    calls = root / "calls.txt"
    script = root / "exp_script.py"

    def write_script(offset=0, tail="pass"):
        script.write_text(
            SCRIPT.format(
                offset=offset, calls=str(calls), wd=str(root / "wd"), tail=tail
            )
        )

    def run(*args):
        return subprocess.run(
            [sys.executable, str(script), *args],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
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
    assert "code the task reaches (exp_script.py)" in err
    assert calls.read_text() == "xxxx"
