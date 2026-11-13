# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT
"""A value holding an interpolation, fixed or swept, is read back unresolved
(#351): mushin never resolves it after the jobs ran."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf

from mushin import MultiRunMetricsWorkflow, hydra_list, multirun
from mushin._sweep_io import MANIFEST_FILE
from mushin.workflows import _read_swept_value, _swept_combo

HYDRA_DIR = "${hydra:runtime.output_dir}"
TICK = "${mushin_test_tick:}"


@pytest.fixture
def tick():
    """A resolver that gives a new value on every call, as ``now`` does over
    time: a read that resolves it again cannot get what the job had."""
    count = iter(range(1, 10**6))
    OmegaConf.register_new_resolver(
        "mushin_test_tick", lambda: f"tick-{next(count)}", replace=True
    )
    yield
    OmegaConf.clear_resolver("mushin_test_tick")


class OwnDir(MultiRunMetricsWorkflow):
    @staticmethod
    def task(x, out):
        # 1.0 when the task got its own cell directory, as Hydra resolves it
        return dict(score=float(x), own_dir=float(Path(out) == Path.cwd()))


class Swept(MultiRunMetricsWorkflow):
    @staticmethod
    def task(x):
        return dict(score=1.0)


def _scores(wf):
    return wf.to_xarray().score.values.tolist()


def test_fixed_hydra_interpolation_does_not_crash_the_sweep(tmp_path):
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

    # Reloaded in this process: HydraConfig is no longer set.
    loaded = MultiRunMetricsWorkflow()
    loaded.load_from_dir(tmp_path, "mushin_metrics.json")
    assert _scores(loaded) == [1.0, 2.0]
    assert loaded.multirun_task_overrides["out"] == HYDRA_DIR


def test_fixed_hydra_interpolation_reloads_in_a_new_process(tmp_path):
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


@pytest.mark.usefixtures("tick")
def test_fixed_interpolation_reads_back_unresolved(tmp_path):
    wf = OwnDir()
    wf.run(x=multirun([1, 2]), out=HYDRA_DIR, stamp=TICK, working_dir=tmp_path)
    assert wf.multirun_task_overrides["stamp"] == TICK
    for _ in range(2):  # the same on every read
        loaded = MultiRunMetricsWorkflow()
        loaded.load_from_dir(tmp_path, "mushin_metrics.json")
        assert loaded.multirun_task_overrides["stamp"] == TICK


@pytest.mark.usefixtures("tick")
def test_swept_interpolation_keeps_every_cell(tmp_path):
    # Resolved when read, the grid's coordinate was a new tick that no cell
    # had, so a completed cell came back NaN.
    wf = Swept()
    wf.run(x=multirun([TICK, "b"]), working_dir=tmp_path)

    ds = wf.to_xarray(include_working_subdirs_as_data_var=True)
    assert ds.coords["x"].values.tolist() == [TICK, "b"]
    assert ds.score.values.tolist() == [1.0, 1.0]
    assert all(ds.working_subdir.values.tolist())
    assert wf.failures == [] and wf.skipped == []
    manifest = json.loads((tmp_path / MANIFEST_FILE).read_text())
    assert {c["status"] for c in manifest["cells"].values()} == {"completed"}
    for status in tmp_path.glob("*/mushin_cell_status.json"):
        assert json.loads(status.read_text())["combo"]["x"] in (TICK, "b")

    loaded = Swept()
    loaded.load_from_dir(tmp_path, "mushin_metrics.json")
    assert _scores(loaded) == [1.0, 1.0]
    assert loaded.failures == [] and loaded.skipped == []


@pytest.mark.usefixtures("tick")
def test_swept_list_with_an_interpolation_keeps_every_cell(tmp_path):
    wf = Swept()
    wf.run(
        x=multirun([hydra_list([TICK, "z"]), hydra_list(["b"])]), working_dir=tmp_path
    )
    assert _scores(wf) == [1.0, 1.0]
    loaded = Swept()
    loaded.load_from_dir(tmp_path, "mushin_metrics.json")
    assert _scores(loaded) == [1.0, 1.0]


def test_target_dir_overrides_read_unresolved(tmp_path):
    first = OwnDir()
    first.run(x=multirun([1, 2]), out=HYDRA_DIR, working_dir=tmp_path / "a")

    class Second(MultiRunMetricsWorkflow):
        @staticmethod
        def task():
            return dict(score=1.0)

    second = Second()
    second.run(target_job_dirs=first.multirun_working_dirs, working_dir=tmp_path / "b")
    assert second.target_dir_multirun_overrides == {
        "x": [1, 2],
        "out": [HYDRA_DIR, HYDRA_DIR],
    }


@pytest.mark.parametrize(
    ("cfg", "name", "expected"),
    [
        ({"x": "${now:%Y}"}, "x", "${now:%Y}"),  # a plain dict is read as is
        (OmegaConf.create({"x": HYDRA_DIR}), "x", HYDRA_DIR),
        (OmegaConf.create({"m": {"w": HYDRA_DIR}}), "m.w", HYDRA_DIR),
        (OmegaConf.create({"m": {"w": 3}}), "m.w", 3),
        # a literal key wins over the nested path, as before
        (OmegaConf.create({"a.b": HYDRA_DIR, "a": {"b": 5}}), "a.b", HYDRA_DIR),
        # the parent is itself an interpolation
        (
            OmegaConf.create({"m": "${base}", "base": {"w": HYDRA_DIR}}),
            "m.w",
            HYDRA_DIR,
        ),
    ],
)
def test_read_swept_value_leaves_an_interpolation_unresolved(cfg, name, expected):
    assert _read_swept_value(cfg, name) == expected


def test_swept_combo_leaves_a_list_element_unresolved():
    cfg = OmegaConf.create({"x": [HYDRA_DIR, "z"]})
    combo = _swept_combo(cfg, ["x"], lambda name: None, lambda v: v)
    assert combo == {"x": [HYDRA_DIR, "z"]}


def test_a_value_without_interpolation_reads_as_before(tmp_path):
    wf = Swept()
    wf.run(x=multirun([0.5, 1.5]), working_dir=tmp_path)
    assert wf.multirun_task_overrides == {"x": [0.5, 1.5]}
    assert np.array_equal(wf.to_xarray().coords["x"].values, [0.5, 1.5])
