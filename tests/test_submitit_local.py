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
    result): the cell fails, as when the record could not be read, and the
    record is left for a resume."""
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
    record = json.loads((wd / "1" / "mushin_cell_status.json").read_text())
    assert record["status"] == "completed"


def test_a_result_submitit_could_not_save_fails_its_cell(tmp_path):
    """The job's runner recorded the cell completed, then submitit could not
    pickle the result (nor the error, so it left no result file). The
    metrics file holds that value only as its text, so the record does not
    say it holds the result, and the cell fails rather than take it."""
    wd = tmp_path / "s"
    with pytest.warns(UserWarning, match=r"failed: \['seed=0'\]"):
        ds = tasks.unpicklable_result.run(
            seed=multirun([0, 1]), working_dir=str(wd), on_error="nan", **LOCAL
        )
    v = ds["v"].values
    assert np.isnan(v[0]) and v[1] == 1.0
    record = json.loads((wd / "0" / "mushin_cell_status.json").read_text())
    assert record["status"] == "completed" and "metrics" not in record


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
    starts fails the cell for its own run and leaves that result alone."""
    wd = tmp_path / "s"
    sweep = dict(seed=multirun([0, 1, 2]), working_dir=str(wd), **LOCAL)
    first = tasks.die_when_told.run(**sweep)
    monkeypatch.setenv("MUSHIN_TEST_DIE_BEFORE_START", "1")
    with pytest.warns(UserWarning, match=r"failed: \['seed=1'\]"):
        tasks.die_when_told.run(
            on_error="nan", overrides=[tasks.DIE_BEFORE_START], **sweep
        )
    record = json.loads((wd / "1" / "mushin_cell_status.json").read_text())
    assert record["status"] == "completed"
    monkeypatch.delenv("MUSHIN_TEST_DIE_BEFORE_START")
    again = tasks.die_when_told.run(resume=True, **sweep)
    # The first run's result, reused; the second run recomputed the others.
    assert again["pid"].values[1] == first["pid"].values[1]


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
    record = json.loads((wd / "0" / "mushin_cell_status.json").read_text())
    assert record["status"] == "failed"
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
        for key in ("run_id", "job_id"):
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
