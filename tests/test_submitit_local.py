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


# Known gaps (#345), pinned so a fix turns them into failures to update.


@pytest.mark.xfail(
    strict=True,
    raises=submitit.core.utils.UncompletedJobError,
    reason="#345: a job that dies outside Python aborts the driver",
)
def test_a_job_that_dies_outside_python_is_nan_under_on_error_nan(tmp_path):
    ds = tasks.die_on_one.run(
        seed=multirun([0, 1, 2]),
        working_dir=str(tmp_path / "s"),
        on_error="nan",
        **LOCAL,
    )
    v = ds["v"].values
    assert v[0] == 0.0 and np.isnan(v[1]) and v[2] == 2.0


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
