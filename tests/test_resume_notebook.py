# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
"""resume=True and cache_dir= notice edits to code defined in a notebook.

A notebook has no file to compare. Its code is the cells the kernel ran, in
the order it ran them, and IPython keeps them (`In`). A task defined in a cell
is compared by what it can reach by name through the latest run of each cell,
the way a driver script is compared through its file: re-running a cell
unchanged, or the cell that extends the grid, is not an edit.
"""

from __future__ import annotations

import importlib
import pathlib
import sys
import textwrap
import warnings

import pytest

pytest.importorskip("IPython")

from mushin import multirun  # noqa: E402
from mushin._resume import (  # noqa: E402
    STATUS_FILE,
    cell_project_record,
    driver_script_entries,
    project_fingerprints,
)

KEY = "__main__ (notebook)"


@pytest.fixture
def shell(tmp_path, monkeypatch):
    """A fresh IPython shell whose history is kept in memory, never in the
    user's profile, with `__main__` restored afterwards."""
    from IPython.core.interactiveshell import InteractiveShell
    from traitlets.config import Config

    monkeypatch.setenv("IPYTHONDIR", str(tmp_path / "ipython"))
    main = sys.modules["__main__"]
    InteractiveShell.clear_instance()
    config = Config()
    config.HistoryManager.hist_file = ":memory:"
    ip = InteractiveShell.instance(config=config)
    try:
        yield ip
    finally:
        ip.history_manager.end_session()
        InteractiveShell.clear_instance()
        sys.modules["__main__"] = main


def run(ip, src):
    result = ip.run_cell(textwrap.dedent(src), store_history=True)
    result.raise_error()
    return result


def code(ip, *names):
    record = project_fingerprints(*(ip.user_ns[n] for n in names))
    assert record is not None
    return record["code"]


def test_a_helper_cell_rerun_unchanged_is_not_an_edit_and_edited_is(shell):
    run(shell, "SCALE = 2\n\ndef helper(x):\n    return x * SCALE\n")
    run(shell, "def task(a):\n    return helper(a)\n")
    before = code(shell, "task")[KEY]
    run(shell, "def helper(x):\n    return x * SCALE\n")
    assert code(shell, "task")[KEY] == before
    run(shell, "def helper(x):\n    return x * SCALE * 10\n")
    assert code(shell, "task")[KEY] != before


def test_a_constant_cell_edited_and_rerun_is_an_edit(shell):
    run(shell, "SCALE = 2")
    run(shell, "def task(a):\n    return a * SCALE\n")
    before = code(shell, "task")[KEY]
    run(shell, "SCALE = 2")
    assert code(shell, "task")[KEY] == before
    run(shell, "SCALE = 3")
    assert code(shell, "task")[KEY] != before


def test_extending_the_grid_and_rerunning_the_run_cell_is_not_an_edit(shell):
    run(shell, "SCALE = 2\n\ndef task(a):\n    return a * SCALE\n")
    run(shell, "GRID = [1, 2]\nRESULT = [task(a) for a in GRID]")
    before = code(shell, "task")[KEY]
    run(shell, "GRID = [1, 2, 3]\nRESULT = [task(a) for a in GRID]")
    run(shell, "print(RESULT)")
    assert code(shell, "task")[KEY] == before


def test_a_value_is_compared_as_the_cells_that_made_it(shell):
    """`Y = X * 2` keeps the X it was computed from until its cell runs again."""
    run(shell, "X = 1")
    run(shell, "Y = X * 2")
    run(shell, "def task(a):\n    return a * Y\n")
    before = code(shell, "task")[KEY]
    run(shell, "X = 5")  # Y is still 2
    assert code(shell, "task")[KEY] == before
    run(shell, "Y = X * 2")  # now 10
    assert code(shell, "task")[KEY] != before


def test_a_task_kept_from_before_its_cell_changed_records_its_own_cell(shell):
    """A sweep built from the old task runs the old task: its record is the
    old cell's, though the name now holds the new one."""
    run(shell, "def task(a):\n    return a * 1\n")
    run(shell, "KEPT = task")
    before = code(shell, "KEPT")[KEY]
    run(shell, "def task(a):\n    return a * 2\n")
    assert code(shell, "KEPT")[KEY] == before
    assert code(shell, "task")[KEY] != before


def test_a_notebook_record_names_the_project_modules_the_task_reaches(
    shell, tmp_path, monkeypatch
):
    """The modules come from the cells the task reaches, not from everything
    the notebook holds."""
    monkeypatch.syspath_prepend(str(tmp_path))
    (tmp_path / "mushin_t_nbuse.py").write_text("def f(x):\n    return x\n")
    (tmp_path / "mushin_t_nbother.py").write_text("def g(x):\n    return x\n")
    importlib.invalidate_caches()
    try:
        run(shell, "import mushin_t_nbuse as use\nimport mushin_t_nbother as other")
        run(shell, "def task(a):\n    return use.f(a)\n")
        record = code(shell, "task")
        assert "mushin_t_nbuse (file)" in record
        assert "mushin_t_nbother (file)" not in record
    finally:
        sys.modules.pop("mushin_t_nbuse", None)
        sys.modules.pop("mushin_t_nbother", None)


def test_a_value_made_outside_the_cells_brings_its_module(
    shell, tmp_path, monkeypatch
):
    """A name no cell binds (`%run`, an extension) is followed by the module
    of its value, inspected by type only."""
    monkeypatch.syspath_prepend(str(tmp_path))
    (tmp_path / "mushin_t_nbmade.py").write_text(
        "class Model:\n    def f(self, x):\n        return x\n"
    )
    importlib.invalidate_caches()
    try:
        made = importlib.import_module("mushin_t_nbmade")
        shell.user_ns["MODEL"] = made.Model()
        run(shell, "def task(a):\n    return MODEL.f(a)\n")
        assert "mushin_t_nbmade (file)" in code(shell, "task")
    finally:
        sys.modules.pop("mushin_t_nbmade", None)


def test_a_cell_that_does_not_parse_is_skipped(shell):
    shell.run_cell("def broken(:\n    pass\n", store_history=True)
    run(shell, "def task(a):\n    return a\n")
    assert KEY in code(shell, "task")


def test_a_cell_with_magics_is_read(shell):
    run(shell, "%time SCALE = 2\ndef task(a):\n    return a * SCALE\n")
    before = code(shell, "task")[KEY]
    run(shell, "%time SCALE = 3")
    assert code(shell, "task")[KEY] != before


def test_a_cell_run_under_a_timing_magic_is_read(shell):
    """`%%time` runs the cell's code in the notebook's namespace."""
    run(shell, "%%time\nSCALE = 2\ndef task(a):\n    return a * SCALE\n")
    before = code(shell, "task")[KEY]
    run(shell, "%%time\nSCALE = 3\n")
    assert code(shell, "task")[KEY] != before


def test_a_notebook_record_never_warns_about_staleness(shell):
    """The cells are what runs, so a changed cell is new code, never stale."""
    run(shell, "SCALE = 2\n\ndef task(a):\n    return a * SCALE\n")
    code(shell, "task")
    run(shell, "SCALE = 3")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        code(shell, "task")


def test_a_parallel_worker_uses_the_drivers_notebook_entry(shell):
    """A worker gets the task by value (cloudpickle), with globals that are
    neither the notebook's nor a file's: it records what the driver took."""
    # joblib's loky ships the cloudpickle its workers use.
    cloudpickle = pytest.importorskip("joblib.externals.cloudpickle")
    run(shell, "SCALE = 2\n\ndef task(a):\n    return a * SCALE\n")
    task = shell.user_ns["task"]
    given = driver_script_entries([task])
    assert "<notebook>" in given
    shipped = cloudpickle.loads(cloudpickle.dumps(task))
    assert shipped.__globals__ is not shell.user_ns
    record = cell_project_record("tok-nb-worker", (shipped,), (), None, scripts=given)
    assert record["code"][KEY] == given["<notebook>"][0]


def test_a_notebook_target_class_is_covered(shell):
    """`builds(Net)` on a class defined in a cell names `__main__.Net`."""
    run(shell, "class Net:\n    def f(self, x):\n        return x * 2\n")
    run(shell, "def task(net, a):\n    return net.f(a)\n")
    task = shell.user_ns["task"]
    cfg = {"net": {"_target_": "__main__.Net"}}
    target_key = "__main__:Net (notebook)"
    before = cell_project_record("tok-nbt-1", (task,), (), cfg)["code"]
    assert target_key in before
    run(shell, "class Net:\n    def f(self, x):\n        return x * 3\n")
    after = cell_project_record("tok-nbt-2", (task,), (), cfg)["code"]
    assert after[target_key] != before[target_key]


SWEEP_CELL = """
from mushin.workflows import MultiRunMetricsWorkflow

class W(MultiRunMetricsWorkflow):
    @staticmethod
    def task(a):
        with open(CALLS, "a") as f:
            f.write("x")
        return dict(v=helper(a))
"""


def test_a_notebook_resume_reruns_after_a_helper_cell_edit(shell, tmp_path):
    """The exploration loop in a notebook: resume reuses the cells after an
    unchanged re-run of the helper's cell and re-runs them after an edit."""
    calls = tmp_path / "calls.txt"
    wd = tmp_path / "wd"
    run(shell, f"CALLS = {str(calls)!r}\nWD = {str(wd)!r}")
    run(shell, "SCALE = 2\n\ndef helper(x):\n    return x * SCALE\n")
    run(shell, SWEEP_CELL)
    shell.user_ns["multirun"] = multirun
    sweep = "W().run(a=multirun([1, 2]), working_dir=WD, resume=True)"
    run(shell, sweep)
    assert calls.read_text() == "xx"
    run(shell, "def helper(x):\n    return x * SCALE\n")
    run(shell, sweep)
    assert calls.read_text() == "xx"  # unchanged: reused
    run(shell, "def helper(x):\n    return x * SCALE * 10\n")
    with pytest.warns(UserWarning, match=r"code the task reaches \(__main__ \(notebook\)\)"):
        run(shell, sweep)
    assert calls.read_text() == "xxxx"
    statuses = list(pathlib.Path(wd).rglob(STATUS_FILE))
    assert len(statuses) == 2
