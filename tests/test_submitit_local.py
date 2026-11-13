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
    assert record["status"] == "completed"
    assert record["job_id"] == raised[0]  # the job's own id, as submitit names it


def test_a_job_whose_second_task_died_keeps_the_first_tasks_result(tmp_path):
    wd = tmp_path / "s"
    ds = tasks.second_task_dies.run(
        seed=multirun([0, 1]),
        working_dir=str(wd),
        on_error="nan",
        launcher="submitit_local",
        launcher_config={"timeout_min": 1, "tasks_per_node": 2},
    )
    assert ds["v"].values.tolist() == [0.0, 1.0]


def test_a_dead_job_replaces_another_cells_record_in_a_reused_dir(
    tmp_path, monkeypatch
):
    """A second sweep over other values reuses the first one's numbered dirs.
    Its job 1 dies before its cell starts: the dir's completed record, its
    metrics and its snapshot are the first sweep's cell, and must not stand
    for this one's."""
    wd = tmp_path / "s"
    tasks.die_when_told.run(seed=multirun([0, 1, 2]), working_dir=str(wd), **LOCAL)
    monkeypatch.setenv("MUSHIN_TEST_DIE_BEFORE_START", "1")
    with pytest.warns(UserWarning, match=r"failed: \['seed=6'\]"):
        tasks.die_when_told.run(
            seed=multirun([5, 6, 7]),
            working_dir=str(wd),
            on_error="nan",
            overrides=[tasks.DIE_BEFORE_START],
            **LOCAL,
        )
    record = json.loads((wd / "1" / "mushin_cell_status.json").read_text())
    assert record["status"] == "failed" and record["combo"] == {"seed": 6}
    assert record["attempt"] == 1
    assert not (wd / "1" / "mushin_metrics.json").exists()
    assert "seed: 6" in (wd / "1" / ".hydra" / "config.yaml").read_text()
    reloaded = MultiRunMetricsWorkflow().load_from_dir(wd, "mushin_metrics.json")
    assert reloaded.to_xarray()["seed"].values.tolist() == [5, 6, 7]
    assert [f["combo"] for f in reloaded.failures] == ["seed=6"]


def test_a_resumed_failed_cell_whose_job_dies_gets_this_jobs_error(
    tmp_path, monkeypatch
):
    wd = tmp_path / "s"
    sweep = dict(seed=multirun([0, 1]), working_dir=str(wd), on_error="nan", **LOCAL)
    monkeypatch.setenv("MUSHIN_TEST_RAISE", "1")
    with pytest.warns(UserWarning, match="failed"):
        tasks.raise_when_told.run(**sweep)
    assert "boom" in (wd / "1" / "mushin_error.txt").read_text()
    monkeypatch.delenv("MUSHIN_TEST_RAISE")
    monkeypatch.setenv("MUSHIN_TEST_DIE_BEFORE_START", "1")
    with pytest.warns(UserWarning, match="failed"):
        tasks.raise_when_told.run(
            resume=True, overrides=[tasks.DIE_BEFORE_START], **sweep
        )
    error = (wd / "1" / "mushin_error.txt").read_text()
    assert "boom" not in error and error.startswith("UncompletedJobError(")
    record = json.loads((wd / "1" / "mushin_cell_status.json").read_text())
    assert record["status"] == "failed" and record["attempt"] == 2


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
