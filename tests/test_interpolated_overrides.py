# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
"""A sweep's override holding an interpolation that cannot be resolved once
its jobs have run is read as written, instead of stopping `run()` and a
reload (#351). One that can be resolved reads as before."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from hydra_zen import make_config

from mushin import MultiRunMetricsWorkflow, multirun
from mushin._sweep_io import MANIFEST_FILE

HYDRA_DIR = "${hydra:runtime.output_dir}"


class OwnDir(MultiRunMetricsWorkflow):
    @staticmethod
    def task(x, out):
        # 1.0 when the task got its own cell directory, as Hydra resolves it
        return dict(score=float(x), own_dir=float(Path(out) == Path.cwd()))


class Plain(MultiRunMetricsWorkflow):
    @staticmethod
    def task(x):
        return dict(score=1.0)


def _scores(wf):
    return wf.to_xarray().score.values.tolist()


def test_a_fixed_hydra_interpolation_does_not_stop_the_sweep(tmp_path):
    wf = OwnDir()
    wf.run(x=multirun([1, 2]), out=HYDRA_DIR, working_dir=tmp_path)

    assert (tmp_path / MANIFEST_FILE).exists()
    ds = wf.to_xarray()
    assert ds.score.values.tolist() == [1.0, 2.0]
    assert ds.own_dir.values.tolist() == [1.0, 1.0]
    assert wf.multirun_task_overrides["out"] == HYDRA_DIR
    # The path that turns a fixed value into a coordinate.
    singleton = wf.to_xarray(non_multirun_params_as_singleton_dims=True)
    assert singleton.coords["out"].values.tolist() == [HYDRA_DIR]

    # Reloaded in this process, where HydraConfig is no longer set.
    loaded = MultiRunMetricsWorkflow()
    loaded.load_from_dir(tmp_path, "mushin_metrics.json")
    assert _scores(loaded) == [1.0, 2.0]
    assert loaded.multirun_task_overrides["out"] == HYDRA_DIR


def test_a_fixed_hydra_interpolation_reloads_in_a_new_process(tmp_path):
    # In a new process the `hydra` resolver is not registered at all.
    OwnDir().run(x=multirun([1, 2]), out=HYDRA_DIR, working_dir=tmp_path)
    code = (
        "import json, sys\n"
        "from mushin import MultiRunMetricsWorkflow\n"
        "wf = MultiRunMetricsWorkflow()\n"
        "wf.load_from_dir(sys.argv[1], 'mushin_metrics.json')\n"
        "print(json.dumps([wf.to_xarray().score.values.tolist(),"
        " wf.multirun_task_overrides['out']]))\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert json.loads(out.splitlines()[-1]) == [[1.0, 2.0], HYDRA_DIR]


def test_target_dir_overrides_read_a_fixed_hydra_interpolation(tmp_path):
    first = OwnDir()
    first.run(x=multirun([1, 2]), out=HYDRA_DIR, working_dir=tmp_path / "a")

    class Second(MultiRunMetricsWorkflow):
        @staticmethod
        def task(out):
            return dict(score=1.0)

    # The second sweep's own overrides hold one too.
    second = Second()
    second.run(
        target_job_dirs=first.multirun_working_dirs,
        out=HYDRA_DIR,
        working_dir=tmp_path / "b",
    )
    assert second.target_dir_multirun_overrides == {
        "x": [1, 2],
        "out": [HYDRA_DIR, HYDRA_DIR],
    }


class Rate(MultiRunMetricsWorkflow):
    @staticmethod
    def task(lr, base_lr):
        return dict(score=float(lr))


@pytest.mark.parametrize("swept", [False, True], ids=["fixed", "swept"])
def test_an_interpolation_that_resolves_reads_its_value(tmp_path, swept):
    # A value of the config, unlike `hydra:`, resolves after the launch as it
    # did in the jobs: it reads as that value, as it always has.
    lr = multirun(["${base_lr}", 0.1]) if swept else "${base_lr}"
    expected = ["0.01", 0.1] if swept else "0.01"
    wf = Rate(eval_task_cfg=make_config(base_lr=0.01))
    wf.run(lr=lr, working_dir=tmp_path)
    assert wf.multirun_task_overrides["lr"] == expected
    loaded = Rate()
    loaded.load_from_dir(tmp_path, "mushin_metrics.json")
    assert loaded.multirun_task_overrides["lr"] == expected
    scores = [0.01, 0.1] if swept else [0.01]
    assert np.ravel(loaded.to_xarray().score.values).tolist() == scores


def test_a_swept_hydra_interpolation_warns_instead_of_stopping_the_sweep(tmp_path):
    # Each job resolves the value to its own directory, so the grid's text
    # cannot key that cell; the run returns and says so.
    wf = Plain()
    with pytest.warns(UserWarning, match="cannot resolve the sweep"):
        wf.run(x=multirun([f"{HYDRA_DIR}/a", "b"]), working_dir=tmp_path)
    ds = wf.to_xarray()
    assert ds.coords["x"].values.tolist() == [f"{HYDRA_DIR}/a", "b"]
    score = ds.score.values
    assert np.isnan(score[0]) and score[1] == 1.0
    assert (tmp_path / MANIFEST_FILE).exists()
