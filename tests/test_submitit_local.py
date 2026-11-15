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

import functools
import json
import os
import re
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
from mushin._resume import snapshot_run_stamp  # noqa: E402
from mushin.workflows import MultiRunMetricsWorkflow  # noqa: E402

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
    if not root.exists():  # nothing was submitted
        return []
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
    # One line, shaped like a task's repr; the log tail stays in the cell.
    assert failed["error"].startswith("UncompletedJobError('submitit job ")
    assert "gave no result" in failed["error"] and "\n" not in failed["error"]
    assert {r["status"] for r in mushin.show(wd).rows} == {"completed", "failed"}


def test_a_dead_jobs_long_error_stream_keeps_its_start_and_end(tmp_path):
    """submitit puts a dead job's whole error stream in its error; the cell
    keeps its start and its end, where the cause is."""
    wd = tmp_path / "s"
    with pytest.warns(UserWarning, match=r"failed: \['seed=1'\]"):
        tasks.die_loudly.run(
            seed=multirun([0, 1]), working_dir=str(wd), on_error="nan", **LOCAL
        )
    error = (wd / "1" / "mushin_error.txt").read_text()
    assert len(error) < 25_000 and "characters cut by mushin" in error
    assert error.startswith("UncompletedJobError(")
    assert error.rstrip().endswith("the cause, last")


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
    # Each dead cell is recorded before the error propagates; as for a task
    # that raised, mushin_error.txt is left to on_error="nan".
    assert [_status(wd / n) for n in "012"] == ["failed", "completed", "failed"]
    assert not list(wd.glob("*/mushin_error.txt"))


@pytest.mark.parametrize("on_error", ["raise", "nan"])
def test_a_dead_job_stops_a_batched_sweep_as_a_raising_task_does(tmp_path, on_error):
    """With one job per batch, "raise" stops after the dead job's batch, as
    it does after a task that raises; "nan" goes on to every batch."""
    wd = tmp_path / "s"
    run = functools.partial(
        tasks.die_on_zero_and_two.run,
        seed=multirun([0, 1, 2]),
        working_dir=str(wd),
        on_error=on_error,
        overrides=["hydra.sweeper.max_batch_size=1"],
        **LOCAL,
    )
    if on_error == "raise":
        with pytest.raises(submitit.core.utils.UncompletedJobError):
            run()
        assert len(_job_dirs(wd)) == 1 and not (wd / "1").exists()
    else:
        with pytest.warns(UserWarning, match="2 run"):
            ds = run()
        assert len(_job_dirs(wd)) == 3 and ds["v"].values[1] == 1.0


def test_a_job_that_fails_before_its_cell_starts_is_a_failed_cell(
    tmp_path, monkeypatch
):
    """No job can import the task's module, so none reaches mushin's runner
    and no cell dir exists: each cell is still recorded as failed, and the
    sweep reloads with both cells failed."""
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
        assert "ModuleNotFoundError" in error  # the job's own error, kept
    # The one line says the job raised, not that it gave no result.
    manifest = json.loads((wd / "mushin_sweep_manifest.json").read_text())
    for c in manifest["cells"].values():
        assert c["error"].startswith("FailedJobError('submitit job ")
        assert "raised an error instead of returning" in c["error"]
        assert "\n" not in c["error"]
    assert {r["status"] for r in mushin.show(wd).rows} == {"failed"}
    reloaded = MultiRunMetricsWorkflow().load_from_dir(wd, "mushin_metrics.json")
    assert sorted(f["combo"] for f in reloaded.failures) == ["seed=0", "seed=1"]


def test_one_cell_dead_before_it_starts_reloads_failed_and_resumes(
    tmp_path, monkeypatch
):
    """Job 1 dies while Hydra computes its dir, in a sweep that otherwise
    runs. The driver writes the dir Hydra never made, so a reload sees a
    failed cell (not one never run), and a resume computes only that cell."""
    wd = str(tmp_path / "s")
    sweep = dict(seed=multirun([0, 1, 2]), working_dir=wd, **LOCAL)
    monkeypatch.setenv("MUSHIN_TEST_DIE_BEFORE_START", "1")
    with pytest.warns(UserWarning, match=r"failed: \['seed=1'\]"):
        first = tasks.die_when_told.run(
            on_error="nan", overrides=[tasks.DIE_BEFORE_START], **sweep
        )
    assert (tmp_path / "s/1/.hydra/hydra.yaml").is_file()
    dead = json.loads((tmp_path / "s/1/mushin_cell_status.json").read_text())
    assert dead["status"] == "failed" and dead["combo"] == {"seed": 1}
    reloaded = MultiRunMetricsWorkflow().load_from_dir(wd, "mushin_metrics.json")
    assert [f["combo"] for f in reloaded.failures] == ["seed=1"]
    assert reloaded.skipped == []

    monkeypatch.delenv("MUSHIN_TEST_DIE_BEFORE_START")
    again = tasks.die_when_told.run(
        resume=True, overrides=[tasks.DIE_BEFORE_START], **sweep
    )
    assert again["v"].values.tolist() == [0.0, 1.0, 2.0]
    before, after = first["pid"].values, again["pid"].values
    assert after[0] == before[0] and after[2] == before[2]  # reused


def test_a_dead_job_for_a_cell_sample_left_out_is_skipped(tmp_path, monkeypatch):
    """`sample=` still submits every cell; one it left out is skipped, as the
    runner skips it, even when its job dies before the runner starts."""
    wd = tmp_path / "s"
    monkeypatch.setenv("MUSHIN_TEST_DIE_BEFORE_START", "0,1,2")
    with pytest.warns(UserWarning, match="2 run"):
        tasks.die_when_told.run(
            seed=multirun([0, 1, 2]),
            working_dir=str(wd),
            sample=2,
            on_error="nan",
            overrides=[tasks.DIE_BEFORE_START],
            **LOCAL,
        )
    statuses = sorted(_status(wd / n) for n in "012")
    assert statuses == ["failed", "failed", "skipped"]
    assert len(list(wd.glob("*/mushin_error.txt"))) == 2


def test_a_cell_dead_while_running_resumes(tmp_path, monkeypatch):
    wd = str(tmp_path / "s")
    sweep = dict(seed=multirun([0, 1, 2]), working_dir=wd, **LOCAL)
    monkeypatch.setenv("MUSHIN_TEST_DIE", "1")
    with pytest.warns(UserWarning, match=r"failed: \['seed=1'\]"):
        first = tasks.die_when_told.run(on_error="nan", **sweep)
    dead = json.loads((tmp_path / "s/1/mushin_cell_status.json").read_text())
    assert dead["status"] == "failed" and dead["combo"] == {"seed": 1}
    monkeypatch.delenv("MUSHIN_TEST_DIE")
    again = tasks.die_when_told.run(resume=True, **sweep)
    assert again["v"].values.tolist() == [0.0, 1.0, 2.0]
    before, after = first["pid"].values, again["pid"].values
    assert after[0] == before[0] and after[2] == before[2]  # reused


def test_a_plain_hydra_task_on_mushins_launcher_sees_the_plugins_error(
    tmp_path, monkeypatch, mushin_local
):
    """Only mushin's runner takes a dead job as a failed cell."""
    from hydra_zen import launch, make_config, zen

    warned = []
    monkeypatch.setattr(mushin_local.log, "warning", warned.append)
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
    # Only mushin's runner names its jobs for `show` (#345), and another
    # task is not even tried.
    assert not (tmp_path / "s" / "mushin_jobs").exists() and warned == []


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


def test_resume_submits_only_the_cells_it_must_compute(tmp_path):
    wd = tmp_path / "s"
    tasks.record_pid.run(seed=multirun([0, 1, 2]), working_dir=str(wd), **LOCAL)
    before = len(_job_dirs(wd))
    tasks.record_pid.run(
        seed=multirun([0, 1, 2]), working_dir=str(wd), resume=True, **LOCAL
    )
    assert len(_job_dirs(wd)) == before


def _result_too_late_for(monkeypatch, seed):
    """submitit's `results()` raises for `seed`'s job as it does when a job's
    result appears only after submitit stopped waiting for it (a slow shared
    filesystem): the job finished, and its cell recorded the result. Patched
    in this process, the driver; returns the ids it raised for."""
    from submitit.core import core
    from submitit.core.utils import UncompletedJobError

    real = core.Job.results
    raised = []

    def results(self):
        out = real(self)
        if out[0].cfg.seed == seed:
            raised.append(self.job_id)
            raise UncompletedJobError(
                f"Job {self.job_id} (task: 0) has not produced any output "
                "(state: FINISHED)"
            )
        return out

    monkeypatch.setattr(core.Job, "results", results)
    return raised


@pytest.mark.parametrize("on_error", ["raise", "nan"])
def test_a_job_whose_cell_completed_keeps_its_result(tmp_path, monkeypatch, on_error):
    wd = tmp_path / "s"
    raised = _result_too_late_for(monkeypatch, seed=1)
    ds = tasks.record_pid.run(
        seed=multirun([0, 1, 2]), working_dir=str(wd), on_error=on_error, **LOCAL
    )
    assert len(raised) == 1  # the dead-job path ran
    assert ds["v"].values.tolist() == [0.0, 1.0, 2.0]
    manifest = json.loads((wd / "mushin_sweep_manifest.json").read_text())
    assert {c["status"] for c in manifest["cells"].values()} == {"completed"}
    assert not (wd / "1" / "mushin_error.txt").exists()
    record = json.loads((wd / "1" / "mushin_cell_status.json").read_text())
    assert record["status"] == "completed" and isinstance(record["metrics"], str)
    assert record["job_id"] == raised[0]  # the job's own id, as submitit names it


@pytest.mark.parametrize("on_error", ["raise", "nan"])
def test_a_job_whose_cell_returned_none_keeps_that_result(
    tmp_path, monkeypatch, on_error
):
    """A cell whose task returned None records that it wrote no metrics
    (#364): when its job's result is lost, the record stands for it, and the
    run and a reload agree that it completed."""
    wd = tmp_path / "s"
    raised = _result_too_late_for(monkeypatch, seed=1)
    ds = tasks.none_on_one.run(
        seed=multirun([0, 1, 2]), working_dir=str(wd), on_error=on_error, **LOCAL
    )
    assert len(raised) == 1  # the dead-job path ran
    v = ds["v"].values
    assert v[0] == 0.0 and np.isnan(v[1]) and v[2] == 2.0
    manifest = json.loads((wd / "mushin_sweep_manifest.json").read_text())
    assert {c["status"] for c in manifest["cells"].values()} == {"completed"}
    reloaded = MultiRunMetricsWorkflow().load_from_dir(wd, "mushin_metrics.json")
    assert reloaded.failures == [] and reloaded.is_complete


def _results_that_raised(monkeypatch):
    """The ids of the jobs whose `results()` raised in this process, the
    driver: a job submitit reports as failed. A job of several tasks reads
    each task's `results()`, under the job's id."""
    from submitit.core import core

    real = core.Job.results
    raised = []

    def results(self):
        try:
            return real(self)
        except Exception:
            raised.append(self.job_id)
            raise

    monkeypatch.setattr(core.Job, "results", results)
    return raised


def test_a_job_of_several_tasks_fails_its_cell_when_one_dies(tmp_path, monkeypatch):
    """Every task of the job runs mushin's runner in the cell's dir, so the
    cell's completed record is not known to be the first task's (the job's
    result): the cell fails, as when the record could not be read, the
    record says so too, and a resume runs the cell again (#406)."""
    raised = _results_that_raised(monkeypatch)
    wd = tmp_path / "s"
    with pytest.warns(UserWarning, match=r"failed: \['seed=1'\]"):
        ds = tasks.second_task_dies.run(
            seed=multirun([0, 1]),
            working_dir=str(wd),
            on_error="nan",
            launcher="submitit_local",
            launcher_config={"timeout_min": 1, "tasks_per_node": 2},
        )
    # Seed 1's job (its tasks' `results()` count too): the dead-job path ran.
    assert len(set(raised)) == 1
    v = ds["v"].values
    assert v[0] == 0.0 and np.isnan(v[1])
    # Failed, as the run reports it (#406): before, the first task's completed
    # record and its metrics file stayed, and a reload took them.
    record = json.loads((wd / "1" / "mushin_cell_status.json").read_text())
    assert record["status"] == "failed"
    assert not (wd / "1" / "mushin_metrics.json").exists()
    assert "UncompletedJobError" in (wd / "1" / "mushin_error.txt").read_text()


@pytest.mark.parametrize("on_error", ["raise", "nan"])
def test_a_result_submitit_could_not_save_fails_its_cell(tmp_path, on_error):
    """The job's runner recorded the cell completed, then submitit could not
    pickle the result (nor the error, so it left no result file). The
    metrics file holds that value only as its text, so the record does not
    say it holds the result, and the cell fails rather than take it; its
    record and metrics file say so too (#406)."""
    from submitit.core.utils import UncompletedJobError

    wd = tmp_path / "s"
    sweep = dict(seed=multirun([0, 1]), working_dir=str(wd), on_error=on_error)
    if on_error == "raise":
        # No result file: submitit cannot tell this from a job that died.
        with pytest.raises(UncompletedJobError):
            tasks.unpicklable_result.run(**sweep, **LOCAL)
        assert not (wd / "0" / "mushin_error.txt").exists()
    else:
        with pytest.warns(UserWarning, match=r"failed: \['seed=0'\]"):
            ds = tasks.unpicklable_result.run(**sweep, **LOCAL)
        v = ds["v"].values
        assert np.isnan(v[0]) and v[1] == 1.0
        error = (wd / "0" / "mushin_error.txt").read_text()
        assert error.startswith("UncompletedJobError('submitit job ")
        assert "gave no result" in error
    # Failed, as the run reports it (#406): before, the job's completed record
    # and the metrics file holding the value as text stayed, and a reload
    # took them.
    record = json.loads((wd / "0" / "mushin_cell_status.json").read_text())
    assert record["status"] == "failed" and "metrics" not in record
    assert not (wd / "0" / "mushin_metrics.json").exists()


def test_a_dead_job_leaves_another_cells_record_in_a_reused_dir(tmp_path, monkeypatch):
    """A second sweep over other values reuses the first one's numbered dirs.
    Its job 1 dies before its cell starts. The dir's record, metrics and any
    checkpoint are the first sweep's cell: recording this cell over them would
    let a resume take that checkpoint for this cell's, so they stay, and the
    cell fails for this run. A resume then starts it afresh."""
    wd = tmp_path / "s"
    tasks.die_when_told.run(seed=multirun([0, 1, 2]), working_dir=str(wd), **LOCAL)
    monkeypatch.setenv("MUSHIN_TEST_DIE_BEFORE_START", "1")
    second = dict(seed=multirun([5, 6, 7]), working_dir=str(wd), **LOCAL)
    with pytest.warns(UserWarning, match=r"failed: \['seed=6'\]"):
        tasks.die_when_told.run(
            on_error="nan", overrides=[tasks.DIE_BEFORE_START], **second
        )
    record = json.loads((wd / "1" / "mushin_cell_status.json").read_text())
    assert record["status"] == "completed" and record["combo"] == {"seed": 1}
    assert (wd / "1" / "mushin_metrics.json").exists()
    # Beside them, this run's snapshot of seed 6's job, so a reload sets the
    # first sweep's record aside and takes the cell's end from this run (#406).
    this_run = json.loads((wd / "0" / "mushin_cell_status.json").read_text())
    assert (
        snapshot_run_stamp(wd / "1" / ".hydra" / "hydra.yaml") == (this_run["run_id"])
    )
    assert "seed: 6" in (wd / "1" / ".hydra" / "config.yaml").read_text()
    monkeypatch.delenv("MUSHIN_TEST_DIE_BEFORE_START")
    again = tasks.die_when_told.run(resume=True, **second)
    assert again["v"].values.tolist() == [5.0, 6.0, 7.0]
    record = json.loads((wd / "1" / "mushin_cell_status.json").read_text())
    assert record["combo"] == {"seed": 6} and record["attempt"] == 1


def test_a_dead_job_leaves_this_cells_earlier_result_for_a_resume(
    tmp_path, monkeypatch
):
    """An earlier run's completed record of the same cell, under the same
    config and code, is what a resume reuses: a job that dies before its cell
    starts fails the cell for its own run and leaves that result alone,
    beside its own run's snapshot, so a reload counts the cell failed."""
    wd = tmp_path / "s"
    sweep = dict(seed=multirun([0, 1, 2]), working_dir=str(wd), **LOCAL)
    first = tasks.die_when_told.run(**sweep)
    first_run = json.loads((wd / "1" / "mushin_cell_status.json").read_text())
    monkeypatch.setenv("MUSHIN_TEST_DIE_BEFORE_START", "1")
    with pytest.warns(UserWarning, match=r"failed: \['seed=1'\]"):
        tasks.die_when_told.run(
            on_error="nan", overrides=[tasks.DIE_BEFORE_START], **sweep
        )
    record = json.loads((wd / "1" / "mushin_cell_status.json").read_text())
    assert record["status"] == "completed"
    # The first run's record, beside the second run's snapshot (#406).
    second_run = json.loads((wd / "0" / "mushin_cell_status.json").read_text())
    assert record["run_id"] == first_run["run_id"] != second_run["run_id"]
    assert (
        snapshot_run_stamp(wd / "1" / ".hydra" / "hydra.yaml") == (second_run["run_id"])
    )
    monkeypatch.delenv("MUSHIN_TEST_DIE_BEFORE_START")
    again = tasks.die_when_told.run(resume=True, **sweep)
    # The first run's result, reused; the second run recomputed the others.
    assert again["pid"].values[1] == first["pid"].values[1]


def _combos(cells: list[dict]) -> list[str]:
    return sorted(cell["combo"] for cell in cells)


def _dead_job_case(case: str, wd: Path, monkeypatch):
    """Run ``case``'s sweep into ``wd`` under ``on_error="nan"``: the last
    run's dataset and workflow, and the cells it must report failed."""
    if case == "several-tasks":
        # A job of two tasks, one of which dies after the other completed.
        ds = tasks.second_task_dies.run(
            seed=multirun([0, 1]),
            working_dir=str(wd),
            on_error="nan",
            launcher="submitit_local",
            launcher_config={"timeout_min": 1, "tasks_per_node": 2},
        )
        return ds, tasks.second_task_dies.workflow, ["seed=1"]
    if case == "unsaved-result":
        ds = tasks.unpicklable_result.run(
            seed=multirun([0, 1]), working_dir=str(wd), on_error="nan", **LOCAL
        )
        return ds, tasks.unpicklable_result.workflow, ["seed=0"]
    if case == "guard-kept-result":
        raised = _result_too_late_for(monkeypatch, seed=1)
        ds = tasks.record_pid.run(
            seed=multirun([0, 1, 2]), working_dir=str(wd), on_error="nan", **LOCAL
        )
        assert len(raised) == 1  # the dead-job path ran
        return ds, tasks.record_pid.workflow, []
    if case == "earlier-interpolated":
        # `x` resolves in each job (#386). The first run's seed-1 job dies
        # while running, so the dir holds that run's failed record of the
        # cell when the second run's job 1 dies before its cell starts.
        sweep = dict(
            x=multirun(["${hydra:job.num}"]),
            seed=multirun([0, 1]),
            working_dir=str(wd),
            on_error="nan",
            overrides=[tasks.DIE_BEFORE_START],
            **LOCAL,
        )
        with monkeypatch.context() as m:
            m.setenv("MUSHIN_TEST_DIE", "1")
            tasks.die_when_told_over_x.run(**sweep)
        assert _status(wd / "1") == "failed"
        monkeypatch.setenv("MUSHIN_TEST_DIE_BEFORE_START", "1")
        ds = tasks.die_when_told_over_x.run(**sweep)
        return ds, tasks.die_when_told_over_x.workflow, ["seed=1,x=0"]
    # A fresh run (not a resume) into a dir an earlier run left, whose job 1
    # dies before its cell starts, so the dir keeps the earlier run's end.
    first, second, failed = {
        "earlier-run": ([0, 1, 2], [0, 1, 2], ["seed=1"]),
        "earlier-sweep": ([0, 1, 2], [5, 6, 7], ["seed=6"]),
        "earlier-skipped": ([0, 1, 2], [0, 1, 2], ["seed=1"]),
    }[case]
    sweep = dict(working_dir=str(wd), overrides=[tasks.DIE_BEFORE_START], **LOCAL)
    if case == "earlier-skipped":
        # Leaves job 1 out: Random("3:2:1").sample(range(3), 2) == [0, 2].
        tasks.die_when_told.run(seed=multirun(first), sample=2, sample_seed=1, **sweep)
        assert _status(wd / "1") == "skipped"
    else:
        tasks.die_when_told.run(seed=multirun(first), **sweep)
        assert _status(wd / "1") == "completed"
    monkeypatch.setenv("MUSHIN_TEST_DIE_BEFORE_START", "1")
    ds = tasks.die_when_told.run(seed=multirun(second), on_error="nan", **sweep)
    return ds, tasks.die_when_told.workflow, failed


@pytest.mark.parametrize(
    "case",
    [
        "several-tasks",
        "unsaved-result",
        "earlier-run",
        "earlier-sweep",
        "earlier-skipped",
        "earlier-interpolated",
        # A guard (it passes before #406 too): a result the record holds whole
        # is kept, by the run and the reload alike.
        "guard-kept-result",
    ],
)
def test_a_reload_says_what_the_run_said_of_a_dead_jobs_cell(
    tmp_path, monkeypatch, case
):
    """A cell whose job ended without a result reloads (`load_from_dir`) as
    the run that wrote it reported it (#406): failed when the run failed it,
    whether its dir holds the job's own completed record (a job of several
    tasks, a result submitit could not save) or an earlier run's end."""
    wd = tmp_path / "s"
    ds, run, failed = _dead_job_case(case, wd, monkeypatch)
    reloaded = MultiRunMetricsWorkflow().load_from_dir(wd, "mushin_metrics.json")
    again = reloaded.to_xarray()
    # The run's verdict first, so a case whose job did not die where it
    # means to cannot pass.
    assert _combos(run.failures) == failed
    # Failed and skipped cells, not only values: an earlier sweep's record
    # of another cell reloads with the same values, as a cell not run.
    assert _combos(reloaded.failures) == failed
    assert _combos(reloaded.skipped) == _combos(run.skipped)
    assert set(again.coords) == set(ds.coords)
    assert set(again.data_vars) == set(ds.data_vars)
    for name in [*ds.coords, *ds.data_vars]:
        np.testing.assert_array_equal(again[name].values, ds[name].values)
    assert again.attrs.get("mushin_failures") == ds.attrs.get("mushin_failures")


def test_a_resume_runs_again_a_cell_whose_dead_job_the_run_could_not_keep(tmp_path):
    """The run failed the cell of a job of several tasks one of which died,
    so neither a resume nor a later run with the same `cache_dir` takes the
    value its first task recorded: a resume submits the cell again (#406)."""
    wd, cache = tmp_path / "s", tmp_path / "cache"
    sweep = dict(
        seed=multirun([0, 1]),
        working_dir=str(wd),
        on_error="nan",
        cache_dir=str(cache),
        launcher="submitit_local",
        launcher_config={"timeout_min": 1, "tasks_per_node": 2},
    )
    with pytest.warns(UserWarning, match=r"failed: \['seed=1'\]"):
        tasks.second_task_dies.run(**sweep)
    # Seed 0's entry alone: seed 1's went with its completed record.
    assert len(list(cache.glob("*/mushin_metrics.json"))) == 1
    submitted = len(_job_dirs(wd))
    # Rank 0 waits for this file, which rank 1 makes before it dies. Left from
    # the first run, rank 0 would not wait: it could complete the cell before
    # rank 1 starts, and rank 1 would then keep that record, not die.
    (wd / "1" / "rank1_started_1").unlink()
    # Seed 1's rank 1 dies again, which shows the cell ran again.
    with pytest.warns(UserWarning, match=r"failed: \['seed=1'\]"):
        tasks.second_task_dies.run(resume=True, **sweep)
    assert len(_job_dirs(wd)) == submitted + 1


def test_each_jobs_snapshot_names_the_run_its_record_does(tmp_path):
    """#355: a reload tells a record an earlier run left from this run's by
    the run its job's hydra.yaml names."""
    from mushin._hydra_yaml import hydra_run_stamp, load_hydra_record

    wd = tmp_path / "s"
    tasks.record_pid.run(seed=multirun([0, 1]), working_dir=str(wd), **LOCAL)
    for cell in ("0", "1"):
        record = json.loads((wd / cell / "mushin_cell_status.json").read_text())
        snapshot = load_hydra_record(wd / cell / ".hydra" / "hydra.yaml")
        assert hydra_run_stamp(snapshot) == record["run_id"]


def test_a_task_that_returns_none_keeps_the_first_tasks_metrics(tmp_path):
    """A job of several tasks runs the cell in each; one that returns None
    after the first task wrote the metrics leaves them (#355)."""
    wd = tmp_path / "s"
    ds = tasks.first_task_reports.run(
        seed=multirun([0, 1]),
        working_dir=str(wd),
        launcher="submitit_local",
        launcher_config={"timeout_min": 2, "tasks_per_node": 2},
    )
    assert ds["v"].values.tolist() == [0.0, 1.0]
    for cell in ("0", "1"):
        assert (wd / cell / "mushin_metrics.json").is_file()


def test_a_job_of_several_tasks_runs_its_cell_without_the_budget(tmp_path):
    """submitit's local tasks set only their own variables; each still finds
    it is one of several and disables a budget, as a SLURM task does, so a
    late task cannot record the cell skipped (#365)."""
    wd = tmp_path / "s"
    ds = tasks.first_task_reports.run(
        seed=multirun([0, 1]),
        working_dir=str(wd),
        launcher="submitit_local",
        launcher_config={"timeout_min": 2, "tasks_per_node": 2},
        max_total_seconds=600,
    )
    assert ds["v"].values.tolist() == [0.0, 1.0]
    logs = sorted((wd / ".submitit").glob("*/*_log.err"))
    assert len(logs) == 4  # two tasks for each cell
    for log in logs:
        assert "`max_total_seconds` is disabled" in log.read_text()


def test_a_resume_submits_no_job_for_a_cell_that_returned_none(tmp_path):
    """The driver settles a completed cell whose task returned None as it
    settles one with metrics, so a resume submits nothing (#373)."""
    wd = tmp_path / "s"
    sweep = dict(seed=multirun([0, 1, 2]), working_dir=str(wd), **LOCAL)
    tasks.none_on_one.run(**sweep)
    submitted = len(_job_dirs(wd))
    again = tasks.none_on_one.run(resume=True, **sweep)
    assert len(_job_dirs(wd)) == submitted
    v = again["v"].values
    assert v[0] == 0.0 and np.isnan(v[1]) and v[2] == 2.0


@pytest.mark.parametrize("on_error", ["raise", "nan"])
@pytest.mark.parametrize(
    "task",
    ["second_task_raises_late", "first_task_raises_early", "second_task_raises_early"],
)
def test_a_job_whose_task_raises_fails_its_cell_and_its_record(
    tmp_path, on_error, task
):
    """A job of several tasks fails its cell when any task raises, whichever
    task records last, and the cell's record says so, so a reload agrees
    with the run (#365). On main the run took the first task's return."""
    wd = tmp_path / "s"
    sweep = dict(
        seed=multirun([0]),
        working_dir=str(wd),
        on_error=on_error,
        launcher="submitit_local",
        launcher_config={"timeout_min": 2, "tasks_per_node": 2},
    )
    if on_error == "raise":
        with pytest.raises(RuntimeError, match="task . of seed 0"):
            getattr(tasks, task).run(**sweep)
    else:
        with pytest.warns(UserWarning, match=r"failed: \['seed=0'\]"):
            ds = getattr(tasks, task).run(**sweep)
        assert "v" not in ds or np.isnan(ds["v"].values).all()
        # The task's own error, not a wait that timed out: the order ran.
        error = (wd / "0" / "mushin_error.txt").read_text()
        assert re.search(r"RuntimeError: task . of seed 0", error)
    record = json.loads((wd / "0" / "mushin_cell_status.json").read_text())
    assert record["status"] == "failed"
    # The failed record still names the job (#345).
    assert record["submitit"]["job_id"] == record["job_id"]
    assert not (wd / "0" / "mushin_metrics.json").exists()
    loaded = MultiRunMetricsWorkflow().load_from_dir(wd, "mushin_metrics.json")
    assert [f["combo"] for f in loaded.failures] == ["seed=0"]


# Cells that need no job are settled in the driver (#345): only the rest are
# submitted, under their own job numbers.


def _records(wd: Path) -> dict[str, dict]:
    """Each cell's record, without what names the run or the job; a metrics
    digest only as present (each run's metrics hold its own pids)."""
    out = {}
    for path in sorted(wd.glob("[0-9]*/mushin_cell_status.json")):
        record = json.loads(path.read_text())
        for key in ("run_id", "job_id", "submitit"):
            record.pop(key, None)
        if "metrics" in record:
            record["metrics"] = "a digest"
        out[path.parent.name] = record
    return out


def _cells(wd: Path) -> dict[str, str]:
    manifest = json.loads((wd / "mushin_sweep_manifest.json").read_text())
    return {key: cell["status"] for key, cell in manifest["cells"].items()}


def _without_the_driver_pass(monkeypatch):
    from mushin.workflows import _TaskRunner

    monkeypatch.setattr(_TaskRunner, "settles_in_driver", lambda self: False)


@pytest.mark.parametrize("driver", [True, False], ids=["driver", "jobs"])
def test_sample_submits_only_the_sampled_cells(tmp_path, monkeypatch, driver):
    if not driver:
        _without_the_driver_pass(monkeypatch)
    wd = tmp_path / "s"
    with pytest.warns(UserWarning, match="skip"):
        ds = tasks.record_pid.run(
            seed=multirun([0, 1, 2, 3]), working_dir=str(wd), sample=1, **LOCAL
        )
    assert int((~np.isnan(ds["v"].values)).sum()) == 1
    assert len(_job_dirs(wd)) == (1 if driver else 4)
    records = _records(wd)
    assert sorted(r["status"] for r in records.values()) == ["completed"] + 3 * [
        "skipped"
    ]
    for cell in records:
        assert (wd / cell / ".hydra" / "hydra.yaml").is_file()
    if driver:  # a settled cell's snapshot names no job, but no missing value
        snapshots = [(wd / c / ".hydra" / "hydra.yaml").read_text() for c in records]
        assert sum("id: driver_" in text for text in snapshots) == 3
    reloaded = MultiRunMetricsWorkflow().load_from_dir(wd, "mushin_metrics.json")
    assert len(reloaded.skipped) == 3


def _sweep_then_resume(tmp_path, monkeypatch, driver):
    """A sweep whose cell 1 dies, then a resume; returns the runs' datasets,
    the dir, and the jobs each run submitted."""
    if not driver:
        _without_the_driver_pass(monkeypatch)
    wd = tmp_path / "s"
    sweep = dict(seed=multirun([0, 1, 2]), working_dir=str(wd), **LOCAL)
    monkeypatch.setenv("MUSHIN_TEST_DIE", "1")
    with pytest.warns(UserWarning, match="failed"):
        first = tasks.die_when_told.run(on_error="nan", **sweep)
    submitted = len(_job_dirs(wd))
    monkeypatch.delenv("MUSHIN_TEST_DIE")
    again = tasks.die_when_told.run(resume=True, **sweep)
    return first, again, wd, (submitted, len(_job_dirs(wd)) - submitted)


def test_a_resume_submits_only_the_cell_it_must_compute(tmp_path, monkeypatch):
    first, again, wd, (_, resubmitted) = _sweep_then_resume(
        tmp_path, monkeypatch, driver=True
    )
    assert resubmitted == 1
    assert again["v"].values.tolist() == [0.0, 1.0, 2.0]
    before, after = first["pid"].values, again["pid"].values
    assert after[0] == before[0] and after[2] == before[2]  # reused


def test_settling_in_the_driver_leaves_what_the_jobs_would(tmp_path, monkeypatch):
    """The same sweep and resume with the driver's pass and without it: the
    same dataset, manifest and cell records (but for the ids of the run and
    the job); only the jobs submitted differ."""
    outcomes = []
    for driver in (True, False):
        with monkeypatch.context() as m:
            _, again, wd, counts = _sweep_then_resume(tmp_path / str(driver), m, driver)
        outcomes.append((again["v"].values.tolist(), _cells(wd), _records(wd), counts))
    (v1, cells1, records1, counts1), (v2, cells2, records2, counts2) = outcomes
    assert (v1, cells1, records1) == (v2, cells2, records2)
    assert counts1 == (3, 1) and counts2 == (3, 3)


def test_a_cache_hit_submits_no_job(tmp_path):
    cache = tmp_path / "cache"
    sweep = dict(seed=multirun([0, 1]), cache_dir=str(cache), **LOCAL)
    first = tasks.record_pid.run(working_dir=str(tmp_path / "a"), **sweep)
    again = tasks.record_pid.run(working_dir=str(tmp_path / "b"), **sweep)
    assert again["pid"].values.tolist() == first["pid"].values.tolist()
    # Nothing left to submit: no executor, so not even its folder.
    assert not (tmp_path / "b" / ".submitit").exists()


def test_a_changed_config_on_resume_submits_every_cell(tmp_path):
    wd = tmp_path / "s"
    tasks.scaled.run(seed=multirun([0, 1]), scale=1.0, working_dir=str(wd), **LOCAL)
    # The driver leaves a changed cell to its job, which says so in its log.
    again = tasks.scaled.run(
        seed=multirun([0, 1]), scale=2.0, working_dir=str(wd), resume=True, **LOCAL
    )
    assert again["v"].values.tolist() == [0.0, 2.0]
    assert len(_job_dirs(wd)) == 4


def test_a_cell_the_driver_cannot_settle_is_left_to_its_job(tmp_path, monkeypatch):
    from mushin.workflows import _TaskRunner

    wd = tmp_path / "s"
    sweep = dict(seed=multirun([0, 1]), working_dir=str(wd), **LOCAL)
    first = tasks.record_pid.run(**sweep)

    def broken(self, ret):
        raise RuntimeError("cannot settle here")

    monkeypatch.setattr(_TaskRunner, "settle_in_driver", broken)
    with pytest.warns(UserWarning, match=r"2 of 2 cells could not be settled.*here"):
        again = tasks.record_pid.run(resume=True, **sweep)
    # Submitted, and reused by the jobs themselves, as before.
    assert len(_job_dirs(wd)) == 4
    assert again["pid"].values.tolist() == first["pid"].values.tolist()


def _cache_twice(tmp_path, monkeypatch, driver):
    """A sweep into a cache_dir, then the same sweep into a new working_dir;
    returns the second run's dataset and dir, and the jobs it submitted."""
    if not driver:
        _without_the_driver_pass(monkeypatch)
    sweep = dict(seed=multirun([0, 1]), cache_dir=str(tmp_path / "cache"), **LOCAL)
    tasks.record_pid.run(working_dir=str(tmp_path / "a"), **sweep)
    again = tasks.record_pid.run(working_dir=str(tmp_path / "b"), **sweep)
    wd = tmp_path / "b"
    return again, wd, len(_job_dirs(wd))


def test_a_cache_hit_settled_in_the_driver_leaves_what_its_job_would(
    tmp_path, monkeypatch
):
    outcomes = []
    for driver in (True, False):
        with monkeypatch.context() as m:
            again, wd, submitted = _cache_twice(tmp_path / str(driver), m, driver)
        outcomes.append(
            (again["v"].values.tolist(), _cells(wd), _records(wd), submitted)
        )
    (v1, cells1, records1, n1), (v2, cells2, records2, n2) = outcomes
    assert (v1, cells1, records1) == (v2, cells2, records2)
    assert (n1, n2) == (0, 2)


def test_a_resume_over_a_renumbered_grid_reuses_every_earlier_cell(tmp_path):
    """A value put first on the axis shifts every job number by one, so each
    cell's earlier result is in the dir the next cell now takes: deciding
    every cell before writing any keeps them all."""
    wd = tmp_path / "s"
    first = tasks.record_pid.run(seed=multirun([1, 2, 3]), working_dir=str(wd), **LOCAL)
    submitted = len(_job_dirs(wd))
    again = tasks.record_pid.run(
        seed=multirun([0, 1, 2, 3]), working_dir=str(wd), resume=True, **LOCAL
    )
    assert len(_job_dirs(wd)) - submitted == 1  # only seed 0
    assert again["v"].values.tolist() == [0.0, 1.0, 2.0, 3.0]
    assert again["pid"].values.tolist()[1:] == first["pid"].values.tolist()


def test_a_config_that_reads_the_environment_is_left_to_its_jobs(tmp_path, monkeypatch):
    """`hydra.job.env_set` sets a variable in each job but not in the driver,
    where `${oc.env:...}` gives its default on both runs: the first run's
    value, so the driver would reuse its cells where the jobs see a new one."""
    monkeypatch.delenv("MUSHIN_TEST_N", raising=False)
    wd = tmp_path / "s"
    sweep = dict(
        seed=multirun([0, 1]), n="${oc.env:MUSHIN_TEST_N,1}", working_dir=str(wd)
    )

    def env(value):
        return [f"++hydra.job.env_set.MUSHIN_TEST_N={value}"]

    first = tasks.env_scaled.run(**sweep, overrides=env(1), **LOCAL)
    assert first["v"].values.tolist() == [1.0, 1.0]
    again = tasks.env_scaled.run(**sweep, overrides=env(8), resume=True, **LOCAL)
    assert again["v"].values.tolist() == [8.0, 8.0]


def test_a_sweep_with_hydra_callbacks_settles_nothing_in_the_driver(tmp_path):
    # A settled cell would skip the callbacks Hydra's run_job calls per job.
    wd = tmp_path / "s"
    callback = (
        "+hydra.callbacks.log_return._target_="
        "hydra.experimental.callbacks.LogJobReturnCallback"
    )
    sweep = dict(seed=multirun([0, 1]), working_dir=str(wd), overrides=[callback])
    tasks.record_pid.run(**sweep, **LOCAL)
    tasks.record_pid.run(**sweep, resume=True, **LOCAL)
    assert len(_job_dirs(wd)) == 4


def test_a_cell_whose_prior_dir_a_job_reads_is_left_to_its_job(tmp_path, monkeypatch):
    """A resume over a renumbered grid, where one cell (seed 1) is left to
    its job, which reads seed 1's prior result from dir 0. Seed 3 would take
    dir 0, so it is submitted, and its job reads seed 3's prior result from
    dir 2; seed 2 would take dir 2, so it is submitted too."""
    from mushin.workflows import _TaskRunner

    wd = tmp_path / "s"
    tasks.record_pid.run(seed=multirun([1, 2, 3]), working_dir=str(wd), **LOCAL)
    submitted = len(_job_dirs(wd))
    original = _TaskRunner.settle_in_driver

    def picky(self, ret):
        return None if ret.cfg.seed == 1 else original(self, ret)

    monkeypatch.setattr(_TaskRunner, "settle_in_driver", picky)
    again = tasks.record_pid.run(
        seed=multirun([3, 1, 2]), working_dir=str(wd), resume=True, **LOCAL
    )
    assert again["v"].values.tolist() == [3.0, 1.0, 2.0]
    assert len(_job_dirs(wd)) - submitted == 3


def test_setting_variables_alone_still_settles_in_the_driver(tmp_path):
    wd = tmp_path / "s"
    sweep = dict(
        seed=multirun([0, 1]),
        working_dir=str(wd),
        overrides=["++hydra.job.env_set.OMP_NUM_THREADS=1"],
    )
    tasks.record_pid.run(**sweep, **LOCAL)
    submitted = len(_job_dirs(wd))
    tasks.record_pid.run(**sweep, resume=True, **LOCAL)
    assert len(_job_dirs(wd)) == submitted


def test_settling_is_best_effort_under_warnings_as_errors(tmp_path, monkeypatch):
    import warnings

    from mushin.workflows import _TaskRunner

    wd = tmp_path / "s"
    sweep = dict(seed=multirun([0, 1]), working_dir=str(wd), **LOCAL)
    first = tasks.record_pid.run(**sweep)

    def broken(self, ret):
        raise RuntimeError("cannot settle here")

    monkeypatch.setattr(_TaskRunner, "settle_in_driver", broken)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        again = tasks.record_pid.run(resume=True, **sweep)
    assert again["pid"].values.tolist() == first["pid"].values.tolist()


def test_the_drivers_judging_shows_no_warning(tmp_path, monkeypatch):
    """What the driver sees while judging a cell (a module it finds edited,
    say) is the job's to warn about, in the job's process."""
    import warnings

    from mushin.workflows import _TaskRunner

    wd = tmp_path / "s"
    sweep = dict(seed=multirun([0, 1]), working_dir=str(wd), **LOCAL)
    tasks.record_pid.run(**sweep)
    original = _TaskRunner.settle_in_driver

    def noisy(self, ret):
        warnings.warn("seen while judging", UserWarning)
        return original(self, ret)

    monkeypatch.setattr(_TaskRunner, "settle_in_driver", noisy)
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        tasks.record_pid.run(resume=True, **sweep)
    assert not [w for w in seen if "while judging" in str(w.message)]
    assert len(_job_dirs(wd)) == 2  # both settled: nothing submitted


def test_a_sweep_with_nothing_to_submit_keeps_the_sweep_dirs_mode(tmp_path):
    sweep = dict(
        seed=multirun([0, 1]),
        cache_dir=str(tmp_path / "cache"),
        overrides=["+hydra.sweep.mode=700"],
        **LOCAL,
    )
    tasks.record_pid.run(working_dir=str(tmp_path / "a"), **sweep)
    tasks.record_pid.run(working_dir=str(tmp_path / "b"), **sweep)
    assert not (tmp_path / "b" / ".submitit").exists()
    assert ((tmp_path / "b").stat().st_mode & 0o777) == 0o700


def test_the_driver_judges_a_cell_under_the_jobs_env_set(tmp_path, monkeypatch):
    """A custom resolver reads what `hydra.job.env_set` sets in each job; the
    driver judges under the same variables, so a resume that changes one
    recomputes, as the jobs would."""
    monkeypatch.delenv("MUSHIN_TEST_N", raising=False)
    wd = tmp_path / "s"
    sweep = dict(seed=multirun([0, 1]), n="${mushin_test_n:}", working_dir=str(wd))

    def env(value):
        return [f"++hydra.job.env_set.MUSHIN_TEST_N={value}"]

    first = tasks.env_scaled.run(**sweep, overrides=env(1), **LOCAL)
    assert first["v"].values.tolist() == [1.0, 1.0]
    again = tasks.env_scaled.run(**sweep, overrides=env(8), resume=True, **LOCAL)
    assert again["v"].values.tolist() == [8.0, 8.0]


def test_a_variable_only_the_jobs_have_still_lets_sample_settle(tmp_path, monkeypatch):
    """`hydra.job.env_set` may read what only a job's environment holds; a
    cell `sample=` leaves out resolves none of it in the driver."""
    import warnings

    monkeypatch.delenv("SUBMITIT_LOCAL_JOB_ID", raising=False)
    wd = tmp_path / "s"
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        ds = tasks.record_pid.run(
            seed=multirun([0, 1, 2, 3]),
            sample=1,
            working_dir=str(wd),
            overrides=[
                "++hydra.job.env_set.MUSHIN_JOB=${oc.env:SUBMITIT_LOCAL_JOB_ID}"
            ],
            **LOCAL,
        )
    assert int((~np.isnan(ds["v"].values)).sum()) == 1
    assert len(_job_dirs(wd)) == 1
    assert not [w for w in seen if "could not be settled" in str(w.message)]


def test_variables_the_driver_cannot_resolve_leave_the_cell_to_its_job(
    tmp_path, monkeypatch
):
    """A resume whose `hydra.job.env_set` holds a value only the jobs can
    resolve (a resolver reading their environment) leaves each cell to its
    job, quietly, and the jobs reuse the cells."""
    import warnings

    monkeypatch.delenv("SUBMITIT_LOCAL_JOB_ID", raising=False)
    wd = tmp_path / "s"
    sweep = dict(
        seed=multirun([0, 1, 2]),
        working_dir=str(wd),
        overrides=["++hydra.job.env_set.MUSHIN_JOB=${mushin_test_job_only:}"],
    )
    first = tasks.record_pid.run(**sweep, **LOCAL)
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        again = tasks.record_pid.run(**sweep, resume=True, **LOCAL)
    assert not [w for w in seen if "could not be settled" in str(w.message)]
    assert again["v"].values.tolist() == [0.0, 1.0, 2.0]
    assert again["pid"].values.tolist() == first["pid"].values.tolist()


# Each record names the submitit job that wrote it: its id, log folder and
# host, so a cell can be traced to its job and its logs (#345).


def _entry(wd: Path, cell: str) -> tuple[dict, dict]:
    record = json.loads((wd / cell / "mushin_cell_status.json").read_text())
    return record, record.get("submitit")


def _logs_of(entry: dict) -> Path:
    return Path(entry["logs"]) / f"{entry['job_id']}_0_log.out"


def test_each_record_names_its_jobs_host_and_logs(tmp_path):
    import socket

    wd = tmp_path / "s"
    tasks.record_pid.run(seed=multirun([0, 1]), working_dir=str(wd), **LOCAL)
    for cell in ("0", "1"):
        record, entry = _entry(wd, cell)
        assert entry["job_id"] == record["job_id"]
        assert entry["host"] == socket.gethostname()
        assert Path(entry["logs"]).resolve().parent == (wd / ".submitit").resolve()
        assert _logs_of(entry).is_file()
        assert "partition" not in entry  # not on SLURM


def test_a_cell_dead_while_running_keeps_its_host_and_names_its_logs(
    tmp_path, monkeypatch
):
    """The driver records the dead cell from the job it waited on: the job's
    log folder, and the host its own 'running' record named."""
    import socket

    wd = tmp_path / "s"
    monkeypatch.setenv("MUSHIN_TEST_DIE", "1")
    with pytest.warns(UserWarning, match="failed"):
        tasks.die_when_told.run(
            seed=multirun([0, 1, 2]), working_dir=str(wd), on_error="nan", **LOCAL
        )
    record, entry = _entry(wd, "1")
    assert record["status"] == "failed"
    assert entry["job_id"] == record["job_id"]
    assert entry["host"] == socket.gethostname()
    assert _logs_of(entry).is_file()


def test_a_cell_dead_before_its_runner_names_its_logs_not_a_host(tmp_path, monkeypatch):
    wd = tmp_path / "s"
    monkeypatch.setenv("MUSHIN_TEST_DIE_BEFORE_START", "1")
    with pytest.warns(UserWarning, match="failed"):
        tasks.die_when_told.run(
            seed=multirun([0, 1, 2]),
            working_dir=str(wd),
            on_error="nan",
            overrides=[tasks.DIE_BEFORE_START],
            **LOCAL,
        )
    record, entry = _entry(wd, "1")
    assert record["status"] == "failed"
    assert entry["job_id"] == record["job_id"] and "host" not in entry
    assert _logs_of(entry).is_file()


@pytest.mark.parametrize("driver", [True, False], ids=["driver", "jobs"])
def test_a_resume_keeps_the_entry_of_the_job_that_computed_the_cell(
    tmp_path, monkeypatch, driver
):
    """A reused cell's record names the job that computed its result, whose
    log holds that computation; the recomputed cell names its new job."""
    if not driver:
        _without_the_driver_pass(monkeypatch)
    wd = tmp_path / "s"
    sweep = dict(seed=multirun([0, 1, 2]), working_dir=str(wd), **LOCAL)
    monkeypatch.setenv("MUSHIN_TEST_DIE", "1")
    with pytest.warns(UserWarning, match="failed"):
        tasks.die_when_told.run(on_error="nan", **sweep)
    before = {cell: _entry(wd, cell)[1] for cell in ("0", "1", "2")}
    monkeypatch.delenv("MUSHIN_TEST_DIE")
    tasks.die_when_told.run(resume=True, **sweep)
    after = {cell: _entry(wd, cell)[1] for cell in ("0", "1", "2")}
    assert after["0"] == before["0"] and after["2"] == before["2"]
    assert after["1"]["job_id"] != before["1"]["job_id"]
    assert _logs_of(after["1"]).is_file()


# A launch names its jobs in `mushin_jobs/` until it has waited on them, so
# `mushin.show` lists the cells whose jobs have not started as queued (#345).


@pytest.fixture
def mushin_local():
    """The launcher module, once Hydra's plugin scan has run: the scan
    executes the module again, rebinding its classes, so a patch made
    before the first Hydra app of the process would be lost."""
    from hydra.core.plugins import Plugins

    Plugins.instance()
    return sys.modules["hydra_plugins.mushin_submitit.launcher"]


def _gate(monkeypatch, tmp_path, nums) -> Path:
    """Hold the jobs of cells ``nums`` before their runner starts (with
    `tasks.GATE`) until the returned file exists."""
    gate = tmp_path / "gate"
    monkeypatch.setenv("MUSHIN_TEST_GATE", str(gate))
    monkeypatch.setenv("MUSHIN_TEST_GATE_NUMS", ",".join(map(str, nums)))
    return gate


def _launches(wd: Path) -> list[str]:
    return sorted(p.name for p in (wd / "mushin_jobs").glob("*.json"))


def _completed(cell: Path) -> bool:
    try:
        return _status(cell) == "completed"
    except (OSError, ValueError):  # not written yet, or mid-replace
        return False


def _until(condition, seconds=30) -> bool:
    import time

    deadline = time.monotonic() + seconds
    while not condition():
        if time.monotonic() > deadline:
            return False
        time.sleep(0.05)
    return True


def _show(wd: Path) -> dict:
    """`show`'s status per seed, or the error it raised."""
    try:
        return {r["seed"]: r["status"] for r in mushin.show(wd).rows}
    except Exception as exc:  # noqa: BLE001 - the test asserts on it
        return {"error": exc}


def _watch(monkeypatch, module, wd, gate=None, ungated=()):
    """Wrap the launcher's `_wait`: once the ``ungated`` cells completed, note
    the launch files, `show`'s rows (with the jobs' columns) and the jobs'
    ids, open the gate, wait as the launcher does, then note `show`'s
    statuses again. It asserts nothing: an AssertionError raised there would
    take the launcher's error path."""
    seen = {}
    original = module.MushinLocalLauncher._wait

    def wait(self, jobs, job_params):
        try:
            _until(lambda: all(_completed(wd / str(n)) for n in ungated))
            seen["ids"] = [str(job.job_id) for job in jobs]
            seen["launches"] = _launches(wd)
            seen["before"] = _show(wd)
            try:
                seen["rows"] = mushin.show(wd, jobs=True).rows
            except Exception as exc:  # noqa: BLE001 - the test asserts on it
                seen["rows"] = exc
        finally:
            if gate is not None:
                gate.touch()
        waited = original(self, jobs, job_params)
        seen["after"] = _show(wd)
        seen["launches after"] = _launches(wd)
        return waited

    monkeypatch.setattr(module.MushinLocalLauncher, "_wait", wait)
    return seen


def test_show_lists_queued_cells_of_a_live_sweep(tmp_path, monkeypatch, mushin_local):
    wd = tmp_path / "s"
    gate = _gate(monkeypatch, tmp_path, [1, 2])
    seen = _watch(monkeypatch, mushin_local, wd, gate, ungated=[0])
    tasks.record_pid.run(
        seed=multirun([0, 1, 2]), working_dir=str(wd), overrides=[tasks.GATE], **LOCAL
    )
    run_id = json.loads((wd / "0" / "mushin_cell_status.json").read_text())["run_id"]
    assert seen["launches"] == [f"{run_id}.0.json"]
    ids = seen["ids"]
    rows = [(r["seed"], r["status"], r["job"]) for r in seen["rows"]]
    assert rows == [
        (0, "completed", ids[0]),
        (1, "queued", ids[1]),
        (2, "queued", ids[2]),
    ]
    assert _show(wd) == {0: "completed", 1: "completed", 2: "completed"}
    assert (wd / "mushin_jobs").is_dir() and _launches(wd) == []


def test_resume_shows_a_resubmitted_cell_queued(tmp_path, monkeypatch, mushin_local):
    wd = tmp_path / "s"
    sweep = dict(
        seed=multirun([0, 1, 2]), working_dir=str(wd), overrides=[tasks.GATE], **LOCAL
    )
    monkeypatch.setenv("MUSHIN_TEST_DIE", "1")
    with pytest.warns(UserWarning, match="failed"):
        tasks.die_when_told.run(on_error="nan", **sweep)
    monkeypatch.delenv("MUSHIN_TEST_DIE")
    gate = _gate(monkeypatch, tmp_path, [1])
    seen = _watch(monkeypatch, mushin_local, wd, gate)
    tasks.die_when_told.run(resume=True, **sweep)
    # Cells 0 and 2 were reused in the driver; cell 1's earlier record is
    # the first run's failure, which its queued job replaces.
    assert seen["before"] == {0: "completed", 1: "queued", 2: "completed"}
    assert _show(wd) == {0: "completed", 1: "completed", 2: "completed"}


def test_dead_before_start_over_an_earlier_terminal_record(
    tmp_path, monkeypatch, mushin_local
):
    """A resubmitted job dies before its runner, in a dir holding the first
    run's failure, which the driver does not overwrite. The cell is queued
    until the job has ended; then, its launch no longer read (a local job's
    process has exited), the dir's earlier record shows."""
    wd = tmp_path / "s"
    sweep = dict(seed=multirun([0, 1, 2]), working_dir=str(wd), **LOCAL)
    monkeypatch.setenv("MUSHIN_TEST_DIE", "1")
    with pytest.warns(UserWarning, match="failed"):
        tasks.die_when_told.run(on_error="nan", **sweep)
    first = (wd / "1" / "mushin_cell_status.json").read_text()
    monkeypatch.delenv("MUSHIN_TEST_DIE")
    monkeypatch.setenv("MUSHIN_TEST_DIE_BEFORE_START", "1")
    seen = _watch(monkeypatch, mushin_local, wd)
    with pytest.warns(UserWarning, match="failed"):
        tasks.die_when_told.run(
            on_error="nan", resume=True, overrides=[tasks.DIE_BEFORE_START], **sweep
        )
    assert seen["before"][1] == "queued"
    assert len(seen["launches after"]) == 1  # still there after `_wait`
    assert seen["after"][1] == "failed"
    assert (wd / "1" / "mushin_cell_status.json").read_text() == first
    assert _show(wd)[1] == "failed" and _launches(wd) == []


def test_dead_before_start_in_a_fresh_dir_is_queued_then_failed(
    tmp_path, monkeypatch, mushin_local
):
    wd = tmp_path / "s"
    monkeypatch.setenv("MUSHIN_TEST_DIE_BEFORE_START", "1")
    seen = _watch(monkeypatch, mushin_local, wd)
    with pytest.warns(UserWarning, match="failed"):
        tasks.die_when_told.run(
            seed=multirun([0, 1, 2]),
            working_dir=str(wd),
            on_error="nan",
            overrides=[tasks.DIE_BEFORE_START],
            **LOCAL,
        )
    assert seen["before"].get(1) == "queued"
    assert seen["after"][1] == "failed"  # this run's record of the dead job


def test_wait_raising_after_waiting_removes_the_launch(
    tmp_path, monkeypatch, mushin_local
):
    wd = tmp_path / "s"
    seen = {}
    original = mushin_local.MushinLocalLauncher._wait

    def wait(self, jobs, job_params):
        seen["launches"] = _launches(wd)
        original(self, jobs, job_params)
        raise RuntimeError("after waiting")

    monkeypatch.setattr(mushin_local.MushinLocalLauncher, "_wait", wait)
    with pytest.raises(RuntimeError, match="after waiting"):
        tasks.record_pid.run(seed=multirun([0, 1]), working_dir=str(wd), **LOCAL)
    assert len(seen["launches"]) == 1
    assert _launches(wd) == []


def test_launch_kept_when_a_jobs_end_is_unknown(tmp_path, monkeypatch, mushin_local):
    """`_wait` raised, and submitit cannot tell that a job ended: it may
    still be queued, so its launch stays."""
    wd = tmp_path / "s"
    original = mushin_local.MushinLocalLauncher._wait

    def wait(self, jobs, job_params):
        original(self, jobs, job_params)
        raise RuntimeError("after waiting")

    monkeypatch.setattr(mushin_local.MushinLocalLauncher, "_wait", wait)
    monkeypatch.setattr(mushin_local, "_ended", lambda job: False)
    with pytest.raises(RuntimeError, match="after waiting"):
        tasks.record_pid.run(seed=multirun([0, 1]), working_dir=str(wd), **LOCAL)
    assert len(_launches(wd)) == 1


def test_ctrl_c_leaves_the_launch(tmp_path, monkeypatch, mushin_local):
    wd = tmp_path / "s"
    gate = _gate(monkeypatch, tmp_path, [0, 1])
    held = []  # the jobs, for the teardown to cancel
    asked = []  # the jobs the launcher asked the scheduler about

    def wait(self, jobs, job_params):
        held.extend(jobs)
        raise KeyboardInterrupt

    def ended(job):
        asked.append(job)
        return True

    monkeypatch.setattr(mushin_local.MushinLocalLauncher, "_wait", wait)
    monkeypatch.setattr(mushin_local, "_ended", ended)
    try:
        with pytest.raises(KeyboardInterrupt):
            tasks.record_pid.run(
                seed=multirun([0, 1]),
                working_dir=str(wd),
                overrides=[tasks.GATE],
                **LOCAL,
            )
        # Interrupted, the driver asks the scheduler nothing more.
        assert len(_launches(wd)) == 1 and asked == []
        assert _show(wd) == {0: "queued", 1: "queued"}
        gate.touch()
        assert _until(lambda: _completed(wd / "0") and _completed(wd / "1"))
        # Each job's record is its launch's: the launch no longer counts.
        assert _show(wd) == {0: "completed", 1: "completed"}
    finally:
        gate.touch()
        for job in held:
            job.cancel(check=False)


def test_cancel_leaves_a_local_launch_to_its_host(tmp_path, monkeypatch, mushin_local):
    """`mushin.cancel` on the launch a driver killed mid-sweep leaves behind:
    a local job's id is a process id on the driver's host, never SLURM's, so
    it warns and asks no scheduler while the jobs live, and drops the launch
    once their processes have all exited."""
    import warnings

    from mushin import _jobs

    wd = tmp_path / "s"
    gate = _gate(monkeypatch, tmp_path, [0, 1])
    held = []  # the jobs, for the teardown to cancel

    def wait(self, jobs, job_params):
        held.extend(jobs)
        raise KeyboardInterrupt

    def scheduler(*args, **kwargs):
        raise AssertionError(f"the scheduler was asked: {args!r}")

    monkeypatch.setattr(mushin_local.MushinLocalLauncher, "_wait", wait)
    monkeypatch.setattr(_jobs, "_run", scheduler)
    monkeypatch.setattr(_jobs, "_which", scheduler)
    try:
        with pytest.raises(KeyboardInterrupt):
            tasks.record_pid.run(
                seed=multirun([0, 1]),
                working_dir=str(wd),
                overrides=[tasks.GATE],
                **LOCAL,
            )
        (launch,) = _launches(wd)
        with pytest.warns(UserWarning, match="process ids"):
            assert mushin.cancel(wd) == []
        assert _launches(wd) == [launch]
        assert _show(wd) == {0: "queued", 1: "queued"}
        gate.touch()
        # `done()` polls each job's process, which reaps it: an exited but
        # unreaped child still holds its pid.
        assert _until(lambda: all(job.done() for job in held))
        (read,), _ = _jobs.read_launches(wd)
        assert _jobs.ended_locally(read)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert mushin.cancel(wd) == []
        assert _launches(wd) == []
    finally:
        gate.touch()
        for job in held:
            job.cancel(check=False)


@pytest.mark.config_group("m345")
def test_launch_combos_match_records(
    tmp_path, monkeypatch, mushin_local, restore_config_group
):
    """A launch's combos are its cells' overrides as parsed, which match the
    records' but where a typed field converts the value."""
    from hydra.core.config_store import ConfigStore

    from mushin._sweep_io import combo_key

    ConfigStore.instance().store(group="m345", name="small", node={"width": 4})
    wd = tmp_path / "s"
    seen = []
    original = mushin_local.MushinLocalLauncher._wait

    def wait(self, jobs, job_params):
        seen.extend(
            json.loads(p.read_text()) for p in (wd / "mushin_jobs").glob("*.json")
        )
        return original(self, jobs, job_params)

    monkeypatch.setattr(mushin_local.MushinLocalLauncher, "_wait", wait)
    tasks.Combos(tasks.CombosConfig).run(
        seed=multirun([0, 1]),
        lr=multirun([1]),  # a float field given an int
        n=multirun([3]),
        x=multirun([0.5]),
        name=multirun(["a b"]),
        flag=multirun([True]),
        xs=multirun([[1, 2]]),
        **{"opt.beta": multirun([0.9])},
        m345=multirun(["small"]),
        working_dir=str(wd),
        **LOCAL,
    )
    (launch,) = seen
    assert [cell["num"] for cell in launch["cells"]] == [0, 1]
    for cell in launch["cells"]:
        record = json.loads(
            (wd / str(cell["num"]) / "mushin_cell_status.json").read_text()
        )
        submitted, recorded = cell["combo"], record["combo"]
        assert submitted.pop("lr") == 1 and recorded.pop("lr") == 1.0
        assert submitted == recorded
        assert recorded["xs"] == "[1, 2]" and recorded["m345"] == "small"
        # The typed float keys apart: `show` matches such a cell by its job.
        assert combo_key({**submitted, "lr": 1}) != combo_key({**recorded, "lr": 1.0})


# Several cells per job (`cells_per_job`, #348): each job runs its cells one
# after another, each through the plugin's own per-cell `__call__`. A job of
# k cells must report, cell by cell, what k one-cell jobs would have, except
# where a job that dies takes its later cells with it. These tests cannot
# fail before #348 on their assertions (the key is unknown there): each names
# the mutant of the launcher it catches.

GROUPS = dict(
    launcher="submitit_local",
    launcher_config={"timeout_min": 1, "cells_per_job": 3},
)


def _grouped(k: int) -> dict:
    """The launcher arguments of a sweep of ``k`` cells per job."""
    config = {"timeout_min": 1}
    if k > 1:
        config["cells_per_job"] = k
    return dict(launcher="submitit_local", launcher_config=config)


def _record(cell: Path) -> dict:
    return json.loads((cell / "mushin_cell_status.json").read_text())


def _reloaded(wd: Path):
    return MultiRunMetricsWorkflow().load_from_dir(wd, "mushin_metrics.json")


def test_a_jobs_cells_keep_their_own_dirs_and_records(tmp_path):
    """Mutant caught: groups of one cell (`_submit_groups` with size 1)."""
    wd = tmp_path / "s"
    ds = tasks.record_pid.run(
        seed=multirun(list(range(5))), working_dir=str(wd), **_grouped(2)
    )
    assert ds["v"].values.tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]
    pids = ds["pid"].values.tolist()
    assert len(_job_dirs(wd)) == 3  # jobs of cells 0-1, 2-3 and 4
    assert pids[0] == pids[1] and pids[2] == pids[3] and len(set(pids)) == 3
    records = [_record(wd / str(n)) for n in range(5)]
    for n, record in enumerate(records):
        assert record["status"] == "completed" and record["job_num"] == n
        assert (wd / str(n) / ".hydra" / "hydra.yaml").is_file()
    ids = [record["job_id"] for record in records]
    assert ids[0] == ids[1] and ids[2] == ids[3] and len(set(ids)) == 3
    again = _reloaded(wd).to_xarray()
    for name in ("v", "pid"):
        np.testing.assert_array_equal(again[name].values, ds[name].values)


# What a sweep leaves, compared between one cell per job and three. A dead
# job's one-line summary names its id and log, and its kind can differ: a
# cell whose return would not load is, in a job of several, one that "raised
# an error instead of returning" (`_CellGroup`'s check), where its own job
# reports what submitit managed ("gave no result", under the error the
# driver's read raised). The verdict is failed either way.
_SUMMARY = re.compile(
    r"\w+\('submitit job \S+ (gave no result|raised an error instead of "
    r"returning) \(state: \w+\); its log: [^']*'\)"
)
# The last line of a traceback's error, which the error file of either
# shape holds (a job's error file ends with its log paths, a cell's in a job
# of several with the traceback).
_CAUSE = re.compile(r"^[\w.]+(Error|Exception)\b.*$", re.MULTILINE)


def _summary_free(text):
    return text if text is None else _SUMMARY.sub("<a dead job's summary>", text)


def _left(wd: Path, run, ds) -> dict:
    """What a sweep into ``wd`` left, without ids, paths or times: each
    record's fields, each error file's first line and cause, the manifest's
    cells, the run's failed and skipped cells, and the dataset."""
    records, errors = {}, {}
    for path in sorted(wd.glob("*/mushin_cell_status.json")):
        record = json.loads(path.read_text())
        records[path.parent.name] = (
            record["status"],
            record["combo"],
            record["attempt"],
            record.get("job_num"),
            "metrics" in record,
            record.get("no_metrics", False),
        )
    for path in sorted(wd.glob("*/mushin_error.txt")):
        text = path.read_text()
        causes = [m.group(0) for m in _CAUSE.finditer(text)]
        errors[path.parent.name] = (
            _summary_free(text.splitlines()[0]),
            causes[-1] if causes else None,
        )
    manifest = wd / "mushin_sweep_manifest.json"
    cells = None
    if manifest.exists():
        cells = {
            key: (cell["status"], cell.get("dir"), _summary_free(cell.get("error")))
            for key, cell in json.loads(manifest.read_text())["cells"].items()
        }
    left = dict(records=records, errors=errors, manifest=cells)
    if run is not None:
        left["failures"] = sorted(
            (f["combo"], _summary_free(f["exception"])) for f in run.failures
        )
        left["skipped"] = sorted(s["combo"] for s in run.skipped)
    if ds is not None:
        # NaN as None, which compares equal to itself.
        left["values"] = {
            n: [None if x != x else x for x in ds[n].values.tolist()]
            for n in [*ds.coords, *ds.data_vars]
        }
    return left


def _scenario(name, wd, launch, monkeypatch, cache):
    """Run scenario ``name`` into ``wd``: its dataset and workflow (None
    when it raised), the error it raised, and the cells it fails."""
    import warnings

    nan = dict(on_error="nan", working_dir=str(wd), **launch)
    seeds = multirun([0, 1, 2])
    with monkeypatch.context() as m, warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if name == "raised-under-raise":
            with pytest.raises(RuntimeError, match="boom") as raised:
                tasks.fail_on_one.run(seed=seeds, working_dir=str(wd), **launch)
            return None, None, raised.value, ["seed=1"]
        if name == "completed":
            task, sweep, failed = tasks.fail_on_one, dict(seed=multirun([0, 2, 3])), []
        elif name == "raised":
            task, sweep, failed = tasks.fail_on_one, dict(seed=seeds), ["seed=1"]
        elif name == "none":
            task, sweep, failed = tasks.none_on_one, dict(seed=seeds), []
        elif name == "unpicklable":
            task, sweep, failed = tasks.unpicklable_result, dict(seed=seeds), ["seed=0"]
        elif name == "unloadable":
            task, sweep, failed = (
                tasks.two_arg_error_on_one,
                dict(seed=seeds),
                ["seed=1"],
            )
        elif name == "outside-the-task":
            m.setenv("MUSHIN_TEST_RAISE_BEFORE_START", "1")
            task, failed = tasks.fail_on_one, ["seed=2"]
            sweep = dict(seed=multirun([0, 2, 3]), overrides=[tasks.RAISE_BEFORE_START])
        elif name == "sample":
            task, failed = tasks.fail_on_one, []
            sweep = dict(seed=multirun([0, 2, 3, 4, 5]), sample=3, sample_seed=1)
        elif name == "resume":
            m.setenv("MUSHIN_TEST_RAISE", "1")
            tasks.raise_when_told.run(seed=multirun([0, 1, 2, 3]), **nan)
            m.delenv("MUSHIN_TEST_RAISE")
            task, failed = tasks.raise_when_told, []
            sweep = dict(seed=multirun([0, 1, 2, 3]), resume=True)
        elif name == "cache":
            # Two of four cells cached by a run into another dir.
            first = {**nan, "working_dir": f"{wd}-first"}
            tasks.fail_on_one.run(seed=multirun([0, 2]), cache_dir=str(cache), **first)
            task, failed = tasks.fail_on_one, []
            sweep = dict(seed=multirun([0, 2, 3, 4]), cache_dir=str(cache))
        else:
            raise AssertionError(f"no scenario {name!r}")
        ds = task.run(**sweep, **nan)
        return ds, task.workflow, None, failed


@pytest.mark.parametrize(
    "name",
    [
        "completed",
        "raised",
        "raised-under-raise",
        "none",
        "unpicklable",
        "unloadable",
        "outside-the-task",
        "sample",
        "resume",
        "cache",
    ],
)
def test_a_job_of_several_cells_records_what_one_cell_jobs_would(
    tmp_path, monkeypatch, name
):
    """Each scenario runs with one cell per job and with three, and leaves
    the same records, error files, manifest, failed and skipped cells and
    dataset (the budget, which a job's late cells meet sooner, is
    `test_the_budget_skips_the_cells_a_job_reaches_late`).

    Mutants caught: the job stops after a failed cell (raised, unloadable,
    outside-the-task); no per-cell `try` (outside-the-task); `_wait` takes
    the group's result as one cell's return (every scenario); the cell's
    return only dumped, not loaded back (unloadable)."""
    left, raised, jobs = {}, {}, {}
    for k in (1, 3):
        wd = tmp_path / f"k{k}"
        ds, run, error, failed = _scenario(
            name, wd, _grouped(k), monkeypatch, tmp_path / f"cache{k}"
        )
        left[k] = _left(wd, run, ds)
        raised[k] = None if error is None else (type(error), str(error))
        jobs[k] = len(_job_dirs(wd))
        # The run's verdict first, so a scenario that did not go where it
        # means to cannot pass on two runs that went wrong alike.
        if run is not None:
            assert [f[0] for f in left[k]["failures"]] == failed, k
    assert jobs[3] < jobs[1]  # cells were grouped
    assert raised[3] == raised[1]
    assert left[3].keys() == left[1].keys()
    for part in left[1]:
        assert left[3][part] == left[1][part], part


@pytest.mark.parametrize("on_error", ["nan", "raise"])
@pytest.mark.parametrize("where", ["while-running", "before-its-dir"])
def test_a_job_that_dies_part_way_keeps_its_finished_cells(
    tmp_path, monkeypatch, on_error, where
):
    """A job of cells 0 to 2 dies at cell 1: cell 0 keeps its result (its
    record holds it), cells 1 and 2 fail, as the reload says too (#406). A
    resume reuses cell 0 and runs 1 and 2, in one job. With one cell per job
    cell 2 would have run: the difference a job of several cells gives up.

    Mutant caught: a dead job's first cell's return used for its other
    cells."""
    wd = tmp_path / "s"
    sweep = dict(seed=multirun([0, 1, 2]), working_dir=str(wd), **GROUPS)
    if where == "while-running":
        monkeypatch.setenv("MUSHIN_TEST_DIE", "1")
    else:
        monkeypatch.setenv("MUSHIN_TEST_DIE_BEFORE_START", "1")
        sweep["overrides"] = [tasks.DIE_BEFORE_START]
    if on_error == "raise":
        with pytest.raises(submitit.core.utils.UncompletedJobError):
            tasks.die_when_told.run(**sweep)
    else:
        with pytest.warns(UserWarning, match="2 run"):
            ds = tasks.die_when_told.run(on_error="nan", **sweep)
        assert ds["v"].values[0] == 0.0 and np.isnan(ds["v"].values[1:]).all()
    assert [_status(wd / n) for n in "012"] == ["completed", "failed", "failed"]
    errors = sorted(p.parent.name for p in wd.glob("*/mushin_error.txt"))
    assert errors == (["1", "2"] if on_error == "nan" else [])
    reloaded = _reloaded(wd)
    again = reloaded.to_xarray()["v"].values
    assert again[0] == 0.0 and np.isnan(again[1:]).all()
    if on_error == "nan":  # under "raise" no manifest says failed
        assert _combos(reloaded.failures) == ["seed=1", "seed=2"]
    before = _record(wd / "0")

    monkeypatch.delenv("MUSHIN_TEST_DIE", raising=False)
    monkeypatch.delenv("MUSHIN_TEST_DIE_BEFORE_START", raising=False)
    ds = tasks.die_when_told.run(resume=True, on_error=on_error, **sweep)
    assert ds["v"].values.tolist() == [0.0, 1.0, 2.0]
    # Reused: its record names the job that computed it.
    after = _record(wd / "0")
    assert after["attempt"] == before["attempt"]
    assert after["submitit"]["job_id"] == before["submitit"]["job_id"]
    pids = ds["pid"].values
    assert pids[1] == pids[2] != pids[0]  # one job, for the two cells it ran


def test_a_dead_jobs_unreached_cell_leaves_an_earlier_runs_record(
    tmp_path, monkeypatch
):
    """A fresh run (not a resume) into a dir an earlier run filled, whose job
    dies at cell 1: cell 2, never reached, keeps the earlier run's record,
    with this run's snapshot beside it, so the reload reports it failed as
    the run did (#406, through a job of several cells). A guard of the dead
    path per cell; the mutant of the test above fails it too."""
    wd = tmp_path / "s"
    sweep = dict(seed=multirun([0, 1, 2]), working_dir=str(wd), **GROUPS)
    tasks.die_when_told.run(**sweep)
    earlier = (wd / "2" / "mushin_cell_status.json").read_text()
    monkeypatch.setenv("MUSHIN_TEST_DIE", "1")
    with pytest.warns(UserWarning, match="2 run"):
        tasks.die_when_told.run(on_error="nan", **sweep)
    this_run = _record(wd / "0")["run_id"]
    assert this_run != json.loads(earlier)["run_id"]
    assert (wd / "2" / "mushin_cell_status.json").read_text() == earlier
    assert snapshot_run_stamp(wd / "2" / ".hydra" / "hydra.yaml") == this_run
    reloaded = _reloaded(wd)
    assert _combos(reloaded.failures) == ["seed=1", "seed=2"]
    assert reloaded.to_xarray()["v"].values[0] == 0.0


def test_cells_sharing_a_dir_in_one_job_never_take_each_others_result(
    tmp_path, monkeypatch
):
    """Two cells of one job in one dir (a constant `hydra.sweep.subdir`); the
    second dies before its runner starts. Its job id is the first's, so only
    the job number tells the first's completed record from its own.

    Mutant caught: `record_dead_cell` matching by job id alone (the commit
    before this one), which reports 0.1 for both."""
    monkeypatch.setenv("MUSHIN_TEST_SHARED_DIE", "1")
    with pytest.warns(UserWarning, match="1 run"):
        ds = tasks.lr_value.run(
            lr=multirun([0.1, 0.2]),
            seed=0,
            working_dir=str(tmp_path / "s"),
            on_error="nan",
            overrides=[tasks.SHARED_DIE],
            **GROUPS,
        )
    v = ds["v"].values
    assert v[0] == 0.1 and np.isnan(v[1])


def _runs(path: Path) -> list[int]:
    """The seeds `signal_self_once` logged, a line per run of its cell."""
    return [int(line.split()[0]) for line in path.read_text().splitlines()]


def test_a_requeued_job_skips_the_cells_it_finished(tmp_path, monkeypatch):
    """Cell 1 sends its job the signal of a preemption: submitit checkpoints
    the job and requeues it. The requeued job skips cell 0, which it
    finished, and runs cell 1 again (its attempt 2), then cell 2. A large
    time limit, so submitit takes the signal for a preemption (with one
    minute, any signal is a time limit's).

    Mutant caught: `checkpoint()` not carrying the finished cells, which
    runs cell 0 again."""
    runs = tmp_path / "runs.txt"
    monkeypatch.setenv("MUSHIN_TEST_RUNS", str(runs))
    wd = tmp_path / "s"
    ds = tasks.signal_self_once.run(
        seed=multirun([0, 1, 2]),
        working_dir=str(wd),
        launcher="submitit_local",
        launcher_config={"timeout_min": 20, "cells_per_job": 3},
    )
    assert ds["v"].values.tolist() == [0.0, 1.0, 2.0]
    assert _runs(runs) == [0, 1, 1, 2]
    records = [_record(wd / n) for n in "012"]
    assert [r["attempt"] for r in records] == [1, 2, 1]
    assert {r["status"] for r in records} == {"completed"}


def test_out_of_time_fails_the_running_cell_and_the_job_runs_on(tmp_path, monkeypatch):
    """With no requeue left at the time limit, submitit raises "Job not
    requeued" in the running cell, which fails with that error; the job then
    runs its next cell, as one-cell jobs would have. A guard of the design's
    one rule (every cell of a job runs), which the job-stops-after-a-failed-
    cell mutant fails."""
    runs = tmp_path / "runs.txt"
    monkeypatch.setenv("MUSHIN_TEST_RUNS", str(runs))
    wd = tmp_path / "s"
    with pytest.warns(UserWarning, match=r"failed: \['seed=1'\]"):
        ds = tasks.signal_self_once.run(
            seed=multirun([0, 1, 2]),
            working_dir=str(wd),
            on_error="nan",
            overrides=["++hydra.launcher.max_num_timeout=0"],
            **GROUPS,
        )
    v = ds["v"].values
    assert v[0] == 0.0 and np.isnan(v[1]) and v[2] == 2.0
    assert "not requeued" in (wd / "1" / "mushin_error.txt").read_text()
    assert _status(wd / "2") == "completed"
    assert _runs(runs) == [0, 1, 2]


def test_show_lists_the_cells_a_job_has_not_reached_as_queued(
    tmp_path, monkeypatch, mushin_local
):
    """Mid-run, the job has finished cell 0 and waits before cell 1: cells 1
    and 2 are queued, with the job's id. The launch file names 3 cells and
    one job id, and is gone once the job has been waited on.

    Mutant caught: `show`'s rule 1 matching by a job id its launch shares
    (the commit before this one), which lists no row for cells 1 and 2."""
    wd = tmp_path / "s"
    gate = _gate(monkeypatch, tmp_path, [1, 2])
    seen = {}
    watched = _watch(monkeypatch, mushin_local, wd, gate, ungated=[0])
    wait = mushin_local.MushinLocalLauncher._wait

    def wait_noting_the_launch(self, jobs, job_params):
        seen["launch"] = [
            json.loads(p.read_text()) for p in (wd / "mushin_jobs").glob("*.json")
        ]
        return wait(self, jobs, job_params)

    monkeypatch.setattr(
        mushin_local.MushinLocalLauncher, "_wait", wait_noting_the_launch
    )
    tasks.record_pid.run(
        seed=multirun([0, 1, 2]), working_dir=str(wd), overrides=[tasks.GATE], **GROUPS
    )
    (launch,) = seen["launch"]
    assert [c["num"] for c in launch["cells"]] == [0, 1, 2]
    (job,) = {c["job_id"] for c in launch["cells"]}
    assert watched["ids"] == [job] * 3
    rows = [(r["seed"], r["status"], r["job"]) for r in watched["rows"]]
    assert rows == [(0, "completed", job), (1, "queued", job), (2, "queued", job)]
    assert _show(wd) == {0: "completed", 1: "completed", 2: "completed"}
    assert _launches(wd) == []


@pytest.mark.parametrize(
    "config",
    [
        {"timeout_min": 1, "cells_per_job": 2, "tasks_per_node": 2},
        {"timeout_min": 1, "cells_per_job": 0},
    ],
    ids=["several-tasks", "zero"],
)
def test_cells_per_job_is_refused_before_anything_is_written(tmp_path, config):
    """Refused in the launcher's `setup`, which Hydra runs before it writes
    the sweep's `multirun.yaml`. (OmegaConf takes `"2"` as 2 and refuses
    `True` and 1.5 itself, before `setup`.) Mutant caught: no check."""
    wd = tmp_path / "s"
    with pytest.raises(ValueError, match="cells_per_job"):
        tasks.record_pid.run(
            seed=multirun([0, 1]),
            working_dir=str(wd),
            launcher="submitit_local",
            launcher_config=config,
        )
    assert not (wd / "multirun.yaml").exists()
    assert _job_dirs(wd) == []


def test_the_budget_skips_the_cells_a_job_reaches_late(tmp_path, monkeypatch):
    """Every cell checks the budget when it starts, so a job's cells after
    the deadline are skipped, and a resume with time to spare runs them. A
    guard that a job of several cells keeps the runner's check per cell;
    mutant caught: `_TaskRunner._budget_expired` always False."""
    wd = tmp_path / "s"
    sweep = dict(seed=multirun([0, 1, 2]), working_dir=str(wd), **GROUPS)
    with pytest.warns(UserWarning, match="skip"):
        ds = tasks.slow.run(max_total_seconds=1, **sweep)
    v = ds["v"].values
    assert v[0] == 0.0 and np.isnan(v[1:]).all()
    assert [_status(wd / n) for n in "012"] == ["completed", "skipped", "skipped"]
    assert _combos(tasks.slow.workflow.skipped) == ["seed=1", "seed=2"]
    monkeypatch.setenv("MUSHIN_TEST_SLEEP", "0")
    again = tasks.slow.run(resume=True, **sweep)
    assert again["v"].values.tolist() == [0.0, 1.0, 2.0]


def test_a_cell_keeps_its_own_return_when_the_next_reuses_the_object(tmp_path):
    """A task that fills and returns one dict for every cell: each cell of a
    job keeps the return it gave, not what a later cell of the job left in
    the dict (#412)."""
    ds = tasks.shared_metrics.run(
        seed=multirun([0, 1, 2]), working_dir=str(tmp_path / "s"), **GROUPS
    )
    assert ds["v"].values.tolist() == [0.0, 1.0, 2.0]
