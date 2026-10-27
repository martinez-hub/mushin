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


def test_a_value_made_outside_the_cells_brings_its_module(shell, tmp_path, monkeypatch):
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
    with pytest.warns(
        UserWarning, match=r"code the task reaches \(__main__ \(notebook\)\)"
    ):
        run(shell, sweep)
    assert calls.read_text() == "xxxx"
    statuses = list(pathlib.Path(wd).rglob(STATUS_FILE))
    assert len(statuses) == 2


# --- the cells as they ran (#277's review) -------------------------------------


def test_each_run_of_an_augmented_assignment_cell_counts(shell):
    run(shell, "SCALE = 1")
    run(shell, "def task(a):\n    return a * SCALE\n")
    seen = []
    for _ in range(3):
        run(shell, "SCALE += 1")
        seen.append(code(shell, "task")[KEY])
    assert len(set(seen)) == 3


def test_a_cell_that_raised_counts_only_up_to_the_error(shell):
    """IPython stops a cell at its first failing statement: what follows it
    did not run, so it is not the code the task reads."""
    run(shell, "SCALE = 2")
    run(shell, "def task(a):\n    return a * SCALE\n")
    before = code(shell, "task")[KEY]
    result = shell.run_cell("undefined_name\nSCALE = 3", store_history=True)
    assert not result.success and shell.user_ns["SCALE"] == 2
    assert code(shell, "task")[KEY] == before
    run(shell, "SCALE = 3")
    assert code(shell, "task")[KEY] != before


def test_a_cell_with_top_level_await_is_read(shell):
    run(shell, "import asyncio\nSCALE = 2")
    run(shell, "def task(a):\n    return a * SCALE\n")
    before = code(shell, "task")[KEY]
    run(shell, "SCALE = 3\nawait asyncio.sleep(0)")
    assert code(shell, "task")[KEY] != before


def test_a_value_a_cell_made_through_foreign_code_brings_its_module(
    shell, tmp_path, monkeypatch
):
    """`model = loader(...)` makes a project object no cell imports: its
    class's module is followed, by type."""
    monkeypatch.syspath_prepend(str(tmp_path))
    (tmp_path / "mushin_t_nbpk.py").write_text(
        "class Model:\n    def __call__(self, x):\n        return x\n"
    )
    importlib.invalidate_caches()
    try:
        run(
            shell,
            "import importlib\nmodel = importlib.import_module('mushin_t_nbpk').Model()",
        )
        run(shell, "def task(a):\n    return model(a)\n")
        assert "mushin_t_nbpk (file)" in code(shell, "task")
    finally:
        sys.modules.pop("mushin_t_nbpk", None)


def test_a_target_class_with_a_wrapped_method_is_placed_by_its_own_cell(shell):
    run(
        shell,
        "import functools\n\ndef timed(f):\n    @functools.wraps(f)\n"
        "    def w(*a):\n        return f(*a)\n    return w\n",
    )
    run(shell, "class Net:\n    @timed\n    def f(self, x):\n        return x * 2\n")
    run(shell, "def task(net, a):\n    return net.f(a)\n")
    task = shell.user_ns["task"]
    cfg = {"net": {"_target_": "__main__.Net"}}
    key = "__main__:Net (notebook)"
    before = cell_project_record("tok-nbw-1", (task,), (), cfg)["code"][key]
    run(shell, "class Net:\n    @timed\n    def f(self, x):\n        return x * 3\n")
    after = cell_project_record("tok-nbw-2", (task,), (), cfg)["code"][key]
    assert after != before


def test_a_kept_task_is_placed_at_the_run_of_its_cell_that_made_it(shell):
    """IPython names each run of a cell apart (`<ipython-input-N-...>`): a
    task kept from an earlier run of an identical cell is that run's, whose
    default read the SCALE of its time."""
    run(shell, "SCALE = 2")
    run(shell, "def task(a, s=SCALE):\n    return a * s\n")
    run(shell, "KEPT = task")
    before = code(shell, "KEPT")[KEY]
    run(shell, "SCALE = 3")
    run(shell, "def task(a, s=SCALE):\n    return a * s\n")
    assert code(shell, "KEPT")[KEY] == before


def test_a_cell_magic_body_is_read_as_ipython_runs_it(shell):
    """A `%%time` body is itself IPython input: a shell line in it does not
    hide the cell's bindings."""
    run(shell, "SCALE = 2")
    run(shell, "def task(a):\n    return a * SCALE\n")
    before = code(shell, "task")[KEY]
    run(shell, "%%time\n!true\nSCALE = 3\n")
    assert code(shell, "task")[KEY] != before


def test_a_notebook_sweep_under_joblib_resumes_from_the_drivers_entry(shell, tmp_path):
    """A real parallel run from the shell: the workers record the driver's
    entry, so an unchanged re-run of the helper's cell resumes every cell."""
    pytest.importorskip("hydra_plugins.hydra_joblib_launcher")
    calls = tmp_path / "calls"
    wd = tmp_path / "wd"
    run(shell, f"CALLS = {str(calls)!r}\nWD = {str(wd)!r}")
    run(shell, "SCALE = 2\n\ndef helper(x):\n    return x * SCALE\n")
    run(
        shell,
        SWEEP_CELL.replace(
            'with open(CALLS, "a") as f:\n            f.write("x")',
            "import os, uuid\n        os.makedirs(CALLS, exist_ok=True)\n"
            '        open(os.path.join(CALLS, uuid.uuid4().hex), "w").close()',
        ),
    )
    shell.user_ns["multirun"] = multirun
    sweep = (
        "W().run(a=multirun([1, 2]), working_dir=WD, resume=True,"
        " launcher='joblib', launcher_config={'n_jobs': 2})"
    )
    run(shell, sweep)
    assert len(list(calls.iterdir())) == 2
    entry = driver_script_entries([shell.user_ns["W"].task])["<notebook>"][0]
    for status in pathlib.Path(wd).rglob(STATUS_FILE):
        import json

        assert json.loads(status.read_text())["code_deps"]["code"][KEY] == entry
    run(shell, "def helper(x):\n    return x * SCALE\n")
    run(shell, sweep)
    assert len(list(calls.iterdir())) == 2  # unchanged: reused


def test_a_capture_cell_is_read(shell):
    run(shell, "SCALE = 2")
    run(shell, "def task(a):\n    return a * SCALE\n")
    before = code(shell, "task")[KEY]
    run(shell, "%%capture out\nSCALE = 3\nprint(SCALE)\n")
    assert code(shell, "task")[KEY] != before


def test_a_decorated_task_is_placed_by_its_decorators_line(shell):
    """A decorated function's code starts at its first decorator, after other
    statements in its cell: a task kept from before its cell changed is still
    placed at its own definition, not found by name."""
    run(shell, "import functools")
    run(shell, "SCALE = 2\n\n@functools.lru_cache\ndef task(a):\n    return a * 1\n")
    run(shell, "KEPT = task")
    before = code(shell, "KEPT")[KEY]
    run(shell, "SCALE = 2\n\n@functools.lru_cache\ndef task(a):\n    return a * 2\n")
    assert code(shell, "KEPT")[KEY] == before
    assert code(shell, "task")[KEY] != before


def test_a_cell_named_by_its_text_is_placed_at_its_latest_run(shell):
    """A Jupyter kernel names a cell's code by its text alone, so identical
    runs share it: the root is placed at the latest run of that text."""
    import linecache

    from mushin._resume import _cell_position, _cell_statements

    text = "def task(a):\n    return a\n"
    for _ in range(2):
        run(shell, text)
    run(shell, "pass")
    name = "/tmp/ipykernel_0/123456.py"
    linecache.cache[name] = (len(text), None, text.splitlines(True), name)
    try:
        ns = {}
        exec(compile(text, name, "exec"), ns)
        cells = list(shell.history_manager.input_hist_parsed)
        starts, total = [], 0
        for cell in cells:
            part = _cell_statements(cell) or []
            starts.append((total, part))
            total += len(part)
        assert _cell_position(ns["task"], cells, starts) == starts[2][0]
    finally:
        linecache.cache.pop(name, None)


def test_a_file_less_main_without_ipython_records_its_namespaces_modules(
    tmp_path, monkeypatch
):
    """A REPL's `__main__` (no file, no IPython shell for it): its code cannot
    be read, so the project modules of the objects it holds are recorded, as
    before."""
    monkeypatch.syspath_prepend(str(tmp_path))
    (tmp_path / "mushin_t_repl.py").write_text("def f(x):\n    return x\n")
    importlib.invalidate_caches()
    try:
        ns = {"__name__": "__main__"}
        exec("import mushin_t_repl as lib\n\ndef task(a):\n    return lib.f(a)\n", ns)
        record = project_fingerprints(ns["task"])["code"]
        assert "mushin_t_repl (file)" in record
        assert KEY not in record
    finally:
        sys.modules.pop("mushin_t_repl", None)
