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
import importlib
import json
import sys
import textwrap
import types
import warnings

import pytest

from mushin import multirun
from mushin._resume import STATUS_FILE, project_fingerprints

HELPERS = "mushin_t_helpers"
EXP = "mushin_t_exp"


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
    assert set(deps) == {EXP, HELPERS}
    assert all(len(h) == 16 for h in deps.values())
    assert project_fingerprints(functools.partial(exp.W.task, a=1)) == deps


def test_project_fingerprints_change_with_a_helper_edit_on_disk(project):
    root, exp = project
    before = project_fingerprints(exp.W.task)
    _write(root, scale=100)  # the file changes; the loaded module does not
    after = project_fingerprints(exp.W.task)
    assert after[EXP] == before[EXP]
    assert after[HELPERS] != before[HELPERS]


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
        UserWarning, match=rf"code the task reaches \({HELPERS}\)"
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
    with pytest.warns(UserWarning, match=rf"code the task reaches \({EXP}\)"):
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
    wf = exp.W()
    wf.run(a=multirun([1, 2]), working_dir=str(root / "s3"), cache_dir=cache)
    assert float(wf.to_xarray()["v"].sel(a=2)) == 20.0  # the first run's entry
