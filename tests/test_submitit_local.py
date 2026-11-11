# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
"""The SLURM launcher's path, without a cluster (#344).

`launcher="submitit_local"` runs hydra-submitit-launcher's own `launch()`, the
one `launcher="submitit_slurm"` uses: it pickles each cell, starts it as a job in
a fresh process, and waits on every job's result. It does not exercise SLURM
arrays, signals, preemption or requeue; those stay in the cluster runbook
(docs/superpowers/cluster-validation-runbook.md).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("hydra_plugins.hydra_submitit_launcher")
submitit = pytest.importorskip("submitit")

pytestmark = pytest.mark.skipif(
    sys.platform == "win32",
    reason="submitit's local executor sends POSIX signals (SIGKILL) Windows lacks",
)

import _submitit_tasks as tasks  # noqa: E402
import mushin  # noqa: E402
from mushin import multirun  # noqa: E402

TESTS = Path(__file__).resolve().parent
# A short time limit, so a job orphaned by an aborted test ends soon.
LOCAL = dict(launcher="submitit_local", launcher_config={"timeout_min": 1})


@pytest.fixture(autouse=True)
def _jobs_can_import_the_tasks(monkeypatch):
    """A job imports `_submitit_tasks` by name. It inherits os.environ but not
    pytest's `pythonpath`, so put tests/ on its PYTHONPATH."""
    path = [str(TESTS), os.environ.get("PYTHONPATH", "")]
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(p for p in path if p))


def _job_dirs(working_dir: Path) -> list[Path]:
    """The folders submitit made under the sweep, one per submitted job. A local
    job's id is a number; the default `submitit_folder` (`.submitit/%j`) also
    leaves a literal `%j` folder behind, which is not a job."""
    root = working_dir / ".submitit"
    return [p for p in root.iterdir() if p.is_dir() and p.name.isdigit()]


def test_each_cell_runs_as_its_own_job(tmp_path):
    wd = tmp_path / "s"
    ds = tasks.record_pid.run(seed=multirun([0, 1, 2]), working_dir=str(wd), **LOCAL)
    assert ds["v"].values.tolist() == [0.0, 1.0, 2.0]
    pids = ds["pid"].values.tolist()
    assert os.getpid() not in pids  # out of process
    assert len(set(pids)) == 3  # a process per cell
    assert len(_job_dirs(wd)) == 3  # a job per cell, under the sweep
    assert (wd / "mushin_sweep_manifest.json").is_file()


def test_a_cell_that_raises_raises_its_own_error_by_default(tmp_path):
    # SLURM users see this by default: the task's own exception, not
    # submitit's wrapper around it.
    with pytest.raises(RuntimeError, match="boom"):
        tasks.fail_on_one.run(
            seed=multirun([0, 1, 2]), working_dir=str(tmp_path / "s"), **LOCAL
        )


def test_a_cell_that_raises_is_nan_under_on_error_nan(tmp_path):
    with pytest.warns(UserWarning, match="failed"):
        ds = tasks.fail_on_one.run(
            seed=multirun([0, 1, 2]),
            working_dir=str(tmp_path / "s"),
            on_error="nan",
            **LOCAL,
        )
    v = ds["v"].values
    assert v[0] == 0.0 and np.isnan(v[1]) and v[2] == 2.0


def test_resume_reuses_the_completed_cells(tmp_path):
    wd = str(tmp_path / "s")
    first = tasks.record_pid.run(seed=multirun([0, 1, 2]), working_dir=wd, **LOCAL)
    again = tasks.record_pid.run(
        seed=multirun([0, 1, 2]), working_dir=wd, resume=True, **LOCAL
    )
    # The pids are the first run's: no cell was computed again.
    assert again["pid"].values.tolist() == first["pid"].values.tolist()


def test_sample_computes_only_the_sampled_cells(tmp_path):
    with pytest.warns(UserWarning, match="skip"):
        ds = tasks.record_pid.run(
            seed=multirun([0, 1, 2, 3]),
            working_dir=str(tmp_path / "s"),
            sample=1,
            **LOCAL,
        )
    assert int((~np.isnan(ds["v"].values)).sum()) == 1


# A job that ends without a result (#345): killed at its time limit, out of
# memory, on a failed node, or exited outside Python. `os._exit` stands in.


def _status(cell: Path) -> str:
    return json.loads((cell / "mushin_cell_status.json").read_text())["status"]


def test_a_job_that_dies_outside_python_is_nan_under_on_error_nan(tmp_path):
    wd = tmp_path / "s"
    with pytest.warns(UserWarning, match=r"failed: \['seed=2'\]"):
        ds = tasks.die_on_two.run(
            seed=multirun([0, 1, 2]), working_dir=str(wd), on_error="nan", **LOCAL
        )
    v = ds["v"].values
    assert v[0] == 0.0 and v[1] == 1.0 and np.isnan(v[2])
    # The cell says so on disk, with the job's log, not `running` forever.
    assert _status(wd / "2") == "failed"
    error = (wd / "2" / "mushin_error.txt").read_text()
    assert "UncompletedJobError" in error and ".submitit" in error
    manifest = json.loads((wd / "mushin_sweep_manifest.json").read_text())
    (failed,) = [c for c in manifest["cells"].values() if c["status"] == "failed"]
    assert "ended without a result" in failed["error"]
    assert "\n" not in failed["error"]  # the log tail stays in the cell
    assert {r["status"] for r in mushin.show(wd).rows} == {"completed", "failed"}


def test_every_job_is_waited_on_when_the_first_dies(tmp_path):
    """The plugin waited in submission order and stopped at the first dead
    job; the cell between the two dead ones still finishes."""
    wd = tmp_path / "s"
    with pytest.warns(UserWarning, match="2 run"):
        ds = tasks.die_on_zero_and_two.run(
            seed=multirun([0, 1, 2]), working_dir=str(wd), on_error="nan", **LOCAL
        )
    v = ds["v"].values
    assert np.isnan(v[0]) and v[1] == 1.0 and np.isnan(v[2])
    assert [_status(wd / n) for n in "012"] == ["failed", "completed", "failed"]


def test_a_dead_job_raises_after_the_others_end_by_default(tmp_path):
    wd = tmp_path / "s"
    with pytest.raises(submitit.core.utils.UncompletedJobError):
        tasks.die_on_zero_and_two.run(
            seed=multirun([0, 1, 2]), working_dir=str(wd), **LOCAL
        )
    # Each dead cell is recorded before the error propagates.
    assert [_status(wd / n) for n in "012"] == ["failed", "completed", "failed"]


def test_a_job_that_fails_before_its_cell_starts_is_a_failed_cell(
    tmp_path, monkeypatch
):
    """No job can import the task's module, so none reaches mushin's runner
    and no cell dir exists: each cell is still recorded as failed."""
    monkeypatch.setenv("PYTHONPATH", "")
    monkeypatch.chdir(tmp_path)  # `python -m` puts the cwd on a job's path
    wd = tmp_path / "s"
    with pytest.warns(UserWarning, match="2 run"):
        tasks.record_pid.run(
            seed=multirun([0, 1]), working_dir=str(wd), on_error="nan", **LOCAL
        )
    for cell in ("0", "1"):
        assert _status(wd / cell) == "failed"
        error = (wd / cell / "mushin_error.txt").read_text()
        assert "FailedJobError" in error and "_submitit_tasks" in error
    assert {r["status"] for r in mushin.show(wd).rows} == {"failed"}


def test_a_plain_hydra_task_on_mushins_launcher_sees_the_plugins_error(tmp_path):
    """Only mushin's runner takes a dead job as a failed cell."""
    from hydra_zen import launch, make_config, zen

    with pytest.raises(submitit.core.utils.UncompletedJobError):
        launch(
            make_config(seed=0),
            zen(tasks.die_plainly),
            overrides=[
                "seed=0,1",
                "hydra/launcher=mushin_submitit_local",
                "hydra.launcher.timeout_min=1",
                f"hydra.sweep.dir={tmp_path / 's'}",
            ],
            multirun=True,
            version_base="1.3",
        )


@pytest.mark.parametrize("raw", [False, True], ids=["launcher=", "override"])
def test_both_spellings_run_on_mushins_launcher(tmp_path, raw):
    wd = tmp_path / "s"
    how = (
        dict(overrides=["hydra/launcher=submitit_local"])
        if raw
        else dict(launcher="submitit_local")
    )
    tasks.record_pid.run(seed=multirun([0]), working_dir=str(wd), **how)
    hydra_yaml = (wd / "0" / ".hydra" / "hydra.yaml").read_text()
    assert "hydra_plugins.mushin_submitit.launcher.MushinLocalLauncher" in hydra_yaml


def test_a_resume_reuses_a_sweep_the_plugins_launcher_ran(tmp_path, monkeypatch):
    """A 0.19 sweep ran on the plugin's launcher; resuming it here reuses it."""
    import mushin.workflows as workflows

    wd = str(tmp_path / "s")
    with monkeypatch.context() as m:
        m.setattr(workflows, "_route_to_mushin_launchers", lambda o: o)
        first = tasks.record_pid.run(seed=multirun([0, 1]), working_dir=wd, **LOCAL)
    assert "hydra_submitit_launcher" in (tmp_path / "s/0/.hydra/hydra.yaml").read_text()
    again = tasks.record_pid.run(
        seed=multirun([0, 1]), working_dir=wd, resume=True, **LOCAL
    )
    assert again["pid"].values.tolist() == first["pid"].values.tolist()


# Known gap (#345), pinned so a fix turns it into a failure to update.


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="#345: a resume submits every cell; each job decides reuse itself",
)
def test_resume_submits_only_the_cells_it_must_compute(tmp_path):
    wd = tmp_path / "s"
    tasks.record_pid.run(seed=multirun([0, 1, 2]), working_dir=str(wd), **LOCAL)
    before = len(_job_dirs(wd))
    tasks.record_pid.run(
        seed=multirun([0, 1, 2]), working_dir=str(wd), resume=True, **LOCAL
    )
    assert len(_job_dirs(wd)) == before
