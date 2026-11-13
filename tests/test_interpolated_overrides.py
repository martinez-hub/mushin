# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
"""A fixed value holding an interpolation that cannot be resolved once the
sweep's jobs have run is read as written, instead of stopping `run()` and a
reload (#351). One that can be resolved reads as before, and a swept one
that cannot still raises (#353)."""

import itertools
import json
import math
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest
from hydra_zen import make_config
from omegaconf import OmegaConf

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
    child = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path)], capture_output=True, text=True
    )
    assert child.returncode == 0, child.stderr
    assert json.loads(child.stdout.splitlines()[-1]) == [[1.0, 2.0], HYDRA_DIR]


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


def test_a_swept_interpolation_that_cannot_be_resolved_reads_its_cells(tmp_path):
    # Each job resolves it for itself, to its own dir: main raised reading
    # the sweep back after both cells completed (#353). Its one cell's value
    # is the coordinate, in the run and in a reload.
    wf = Plain()
    wf.run(x=multirun([f"{HYDRA_DIR}/a", "b"]), working_dir=tmp_path)
    own = str(tmp_path / "0" / "a")
    assert wf.multirun_task_overrides["x"] == [own, "b"]
    assert _scores(wf) == [1.0, 1.0] and wf.is_complete
    loaded = Plain()
    loaded.load_from_dir(tmp_path, "mushin_metrics.json")
    assert loaded.multirun_task_overrides["x"] == [own, "b"]
    assert _scores(loaded) == [1.0, 1.0] and loaded.is_complete
    # With two cells per element, which resolve it differently, its text.
    two = Plain()
    two.run(
        x=multirun([f"{HYDRA_DIR}/a", "b"]),
        overrides=["+seed=0,1"],
        working_dir=tmp_path / "two",
    )
    assert two.multirun_task_overrides["x"] == [f"{HYDRA_DIR}/a", "b"]
    assert two.is_complete


def test_target_dir_overrides_cache_nothing_from_a_read_that_raised(tmp_path):
    first = Plain()
    first.run(x=multirun(["a", "b"]), working_dir=tmp_path / "a")
    second = Plain()
    second.run(
        target_job_dirs=first.multirun_working_dirs,
        x="c",
        working_dir=tmp_path / "b",
    )
    (first.multirun_working_dirs[1] / ".hydra" / "overrides.yaml").unlink()
    for _ in range(2):  # not half the dirs on the second read
        with pytest.raises(FileNotFoundError):
            second.target_dir_multirun_overrides  # noqa: B018


# A swept value written as an interpolation is keyed by what its cells
# resolved: the value they agree on, else the element's text (#353).


@pytest.fixture
def counter():
    """A resolver that gives a new value on each call."""
    count = itertools.count()
    OmegaConf.register_new_resolver("mushin_test_count", lambda: f"c{next(count)}")
    yield "${mushin_test_count:}"
    OmegaConf.clear_resolver("mushin_test_count")


def _values(wf, name="v"):
    return np.ravel(wf.to_xarray()[name].values).tolist()


class Seeded(MultiRunMetricsWorkflow):
    @staticmethod
    def task(x, seed):
        return dict(v=float(seed))


def test_a_value_its_cells_resolve_differently_keys_by_its_text(tmp_path, counter):
    """The issue's first case: each cell of the element resolved its own
    value, so the coordinate is the element's text, and every completed cell
    holds its result (main left them NaN)."""
    wf = Seeded()
    wf.run(x=multirun([counter, "b"]), seed=multirun([0, 1]), working_dir=tmp_path)
    assert [str(x) for x in wf.to_xarray()["x"].values] == [counter, "b"]
    assert _values(wf) == [0.0, 1.0, 0.0, 1.0] and wf.is_complete
    back = Seeded().load_from_dir(tmp_path, "mushin_metrics.json")
    assert [str(x) for x in back.to_xarray()["x"].values] == [counter, "b"]
    assert _values(back) == [0.0, 1.0, 0.0, 1.0] and back.is_complete


class Named(MultiRunMetricsWorkflow):
    @staticmethod
    def task(model_name, name):
        return dict(v=float(len(name)))


def test_an_interpolation_of_another_swept_value(tmp_path):
    """`${model_name}_x` cannot be resolved in multirun.yaml (main raised);
    its cells resolve it to m1_x and m2_x, so it keys by its text."""
    wf = Named()
    wf.run(
        model_name=multirun(["m1", "m2"]),
        name=multirun(["${model_name}_x", "b"]),
        working_dir=tmp_path,
    )
    assert [str(x) for x in wf.to_xarray()["name"].values] == ["${model_name}_x", "b"]
    assert _values(wf) == [4.0, 1.0, 4.0, 1.0]
    back = Named().load_from_dir(tmp_path, "mushin_metrics.json")
    assert _values(back) == [4.0, 1.0, 4.0, 1.0]


class SampledRate(MultiRunMetricsWorkflow):
    @staticmethod
    def task(lr, seed, base_lr=None):
        return dict(v=float(lr) + seed)


def test_an_element_no_cell_ran_keys_as_the_driver_reads_it(tmp_path):
    """With `sample=` leaving out every cell of `${base_lr}`, nothing votes:
    the coordinate is the driver's reading, as on main, not the text."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for sample_seed in range(30):
            wf = SampledRate()
            wf.run(
                base_lr=0.01,
                lr=multirun(["${base_lr}", 0.1]),
                seed=multirun([0, 1]),
                working_dir=tmp_path / str(sample_seed),
                sample=2,
                sample_seed=sample_seed,
            )
            left_out = wf.to_xarray()["v"].sel(lr="0.01").values
            if all(math.isnan(x) for x in left_out):
                break
        else:
            pytest.skip("no sample left out every ${base_lr} cell")
    assert wf.multirun_task_overrides["lr"] == ["0.01", 0.1]


class Listed(MultiRunMetricsWorkflow):
    @staticmethod
    def task(xs, base_lr=None):
        return dict(v=float(len(xs)))


def test_an_interpolation_inside_a_swept_list(tmp_path):
    """main keyed the element as the driver read it (['0.01', 1]) and the
    cell as it resolved it ([0.01, 1]): a NaN cell."""
    wf = Listed()
    wf.run(base_lr=0.01, xs=multirun([["${base_lr}", 1], [3]]), working_dir=tmp_path)
    assert _values(wf) == [2.0, 1.0]
    back = Listed().load_from_dir(tmp_path, "mushin_metrics.json")
    assert _values(back) == [2.0, 1.0]


class FailsOnB(MultiRunMetricsWorkflow):
    @staticmethod
    def task(x, seed):
        if x == "b":
            raise RuntimeError("b fails")
        return dict(v=float(seed))


def test_failures_key_the_same_in_the_run_and_a_reload(tmp_path, counter):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        wf = FailsOnB()
        wf.run(
            x=multirun([counter, "b"]),
            seed=multirun([0, 1]),
            working_dir=tmp_path,
            on_error="nan",
        )
    back = FailsOnB().load_from_dir(tmp_path, "mushin_metrics.json")
    assert sorted(f["combo"] for f in wf.failures) == ["seed=0,x=b", "seed=1,x=b"]
    assert sorted(f["combo"] for f in back.failures) == sorted(
        f["combo"] for f in wf.failures
    )
    assert _values(wf)[:2] == [0.0, 1.0] == _values(back)[:2]
