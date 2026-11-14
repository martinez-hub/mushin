# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
"""A fixed value holding an interpolation that cannot be resolved once the
sweep's jobs have run is read as written, instead of stopping `run()` and a
reload (#351). A swept one is keyed by what the sweep's cells resolved it to,
as the run decided once and saved in its manifest (#353)."""

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

from mushin import MultiRunMetricsWorkflow, hydra_list, multirun
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
    own = f"{tmp_path / '0'}/a"  # the job's dir, then "/a" as written
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
    """With `sample=` leaving out every cell of `${base_lr}` (sample_seed 9
    does, checked below), the coordinate is the driver's reading, as on
    main, not the text: a left-out cell is recorded with its own value."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        wf = SampledRate()
        wf.run(
            base_lr=0.01,
            lr=multirun(["${base_lr}", 0.1]),
            seed=multirun([0, 1]),
            working_dir=tmp_path,
            sample=2,
            sample_seed=9,
        )
    left_out = wf.to_xarray()["v"].sel(lr="0.01").values
    assert all(math.isnan(x) for x in left_out)
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


def test_a_nan_inside_a_swept_list_with_an_interpolation(tmp_path):
    """A job's own list element holding a NaN maps to its element (a NaN
    equals a NaN there too), so its cell keeps its value in the run and in a
    reload (#386)."""
    wf = Listed()
    wf.run(
        base_lr=0.01,
        xs=multirun([["${base_lr}", float("nan")], [3]]),
        working_dir=tmp_path,
    )
    assert _values(wf) == [2.0, 1.0]
    back = Listed().load_from_dir(tmp_path, "mushin_metrics.json")
    assert _values(back) == [2.0, 1.0]
    assert [str(x) for x in back.multirun_task_overrides["xs"]] == [
        str(x) for x in wf.multirun_task_overrides["xs"]
    ]


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


class Paced(MultiRunMetricsWorkflow):
    @staticmethod
    def task(lr, seed, base_lr=None):
        if float(lr) == 0.01:
            raise RuntimeError("0.01 fails")
        return dict(v=float(lr) + seed)


def test_a_reload_ignores_an_earlier_sweeps_dirs(tmp_path):
    """An earlier, larger sweep in the same working_dir left dirs whose
    records the reload must not take for this run's (#353 review)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Paced().run(
            base_lr=0.02,
            lr=multirun([0.1, "${base_lr}"]),
            seed=multirun([0, 1, 2]),
            working_dir=tmp_path,
        )
        wf = Paced()
        wf.run(
            base_lr=0.01,
            lr=multirun([0.1, "${base_lr}"]),
            seed=multirun([0, 1]),
            working_dir=tmp_path,
            on_error="nan",
        )
    back = Paced().load_from_dir(tmp_path, "mushin_metrics.json")
    run, reload_ = wf.to_xarray(), back.to_xarray()
    assert [str(x) for x in reload_["lr"].values] == [str(x) for x in run["lr"].values]
    assert np.array_equal(run["v"].values, reload_["v"].values, equal_nan=True)
    assert sorted(f["combo"] for f in back.failures) == sorted(
        f["combo"] for f in wf.failures
    )


def test_a_failed_elements_coordinate_is_the_same_on_every_read(tmp_path, counter):
    """Its cells failed, and each resolved the counter for itself: the
    coordinate is the run's decision, read back, not resolved again."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        wf = FailsOnB()
        wf.run(
            x=multirun(["b", counter]),
            seed=multirun([0, 1]),
            working_dir=tmp_path,
            on_error="nan",
        )
    keys = sorted(f["combo"] for f in wf.failures)
    for _ in range(2):
        back = FailsOnB().load_from_dir(tmp_path, "mushin_metrics.json")
        assert sorted(f["combo"] for f in back.failures) == keys


def test_a_base_workflow_reads_an_interpolated_override(tmp_path):
    """Only MultiRunMetricsWorkflow keeps cells; a BaseWorkflow reads the
    overrides as before (#353 review: it raised AttributeError)."""
    from mushin.workflows import BaseWorkflow

    seen = {}

    class Base(BaseWorkflow):
        @staticmethod
        def task(lr, base_lr, x):
            return None

        def jobs_post_process(self):
            self.working_dir = tmp_path
            seen.update(self.multirun_task_overrides)

    Base().run(
        base_lr=0.01,
        lr=multirun(["${base_lr}", 0.1]),
        x="${base_lr}",
        working_dir=tmp_path,
    )
    assert seen["lr"] == ["0.01", 0.1] and seen["x"] == "0.01"


def test_a_workflow_object_reused_for_a_load_reads_that_sweep(tmp_path):
    """A load replaces the jobs of a run the same object made."""
    a, b = tmp_path / "a", tmp_path / "b"
    first = SampledRate()
    first.run(
        base_lr=0.01,
        lr=multirun(["${base_lr}", 0.1]),
        seed=multirun([0]),
        working_dir=a,
    )
    SampledRate().run(
        base_lr=0.01,
        lr=multirun([0.1, "${base_lr}"]),
        seed=multirun([0]),
        working_dir=b,
    )
    first.load_from_dir(b, "mushin_metrics.json")
    fresh = SampledRate().load_from_dir(b, "mushin_metrics.json")
    assert first.multirun_task_overrides["lr"] == fresh.multirun_task_overrides["lr"]


def test_an_element_that_cannot_be_read_leaves_its_siblings_as_read(tmp_path):
    """`${hydra:job.num}` cannot be resolved after the run; `${base_lr}`
    beside it still reads as the driver reads it (a string, as main has it
    beside 0.1), so `sel(lr="0.01")` and `diff` pairing hold."""
    wf = SampledRate()
    wf.run(
        base_lr=0.01,
        lr=multirun(["${base_lr}", "${hydra:job.num}"]),
        seed=multirun([0]),
        working_dir=tmp_path,
    )
    assert wf.multirun_task_overrides["lr"][0] == "0.01"


class Inf(MultiRunMetricsWorkflow):
    @staticmethod
    def task(x, base_lr=None):
        return dict(v=1.0)


def test_a_non_finite_element_beside_an_interpolation(tmp_path):
    wf = Inf()
    wf.run(base_lr=0.01, x=multirun(["${base_lr}", float("inf")]), working_dir=tmp_path)
    assert wf.multirun_task_overrides["x"][1] == float("inf")
    back = Inf().load_from_dir(tmp_path, "mushin_metrics.json")
    assert back.multirun_task_overrides["x"][1] == float("inf")


class Rated(MultiRunMetricsWorkflow):
    @staticmethod
    def task(lr, base_lr=None):
        return dict(v=1.0)


def test_a_nan_element_beside_an_interpolation(tmp_path):
    wf = Rated()
    wf.run(
        base_lr=0.01,
        lr=multirun([float("nan"), "${base_lr}", 0.5]),
        working_dir=tmp_path,
    )
    assert _values(wf) == [1.0, 1.0, 1.0]
    back = Rated().load_from_dir(tmp_path, "mushin_metrics.json")
    assert math.isnan(back.multirun_task_overrides["lr"][0])
    assert _values(back) == [1.0, 1.0, 1.0]


class Based(MultiRunMetricsWorkflow):
    @staticmethod
    def task(x, base=None):
        return dict(v=1.0 if x == x else 0.0)


def test_an_interpolation_that_resolves_to_nan_keys_as_nan(tmp_path):
    """A record stores a combo's NaN tagged (#382), so the cell whose element
    resolves to NaN keys as ``nan`` (not as a swept None) in the run and in a
    reload (#386). The coordinate is the driver's reading, as on main."""
    wf = Based()
    wf.run(base=float("nan"), x=multirun(["${base}", 0.5]), working_dir=tmp_path)
    back = Based().load_from_dir(tmp_path, "mushin_metrics.json")
    for read in (wf, back):
        assert [str(x) for x in read.multirun_task_overrides["x"]] == ["nan", "0.5"]
        assert _values(read) == [0.0, 1.0]


def test_a_sweep_from_before_run_stamps_reads_its_records(tmp_path):
    """A sweep written before runs were named has no run stamp: its records
    count as its own, as a reload takes them elsewhere."""
    import re

    Named().run(
        model_name=multirun(["m1", "m2"]),
        name=multirun(["${model_name}_x", "b"]),
        working_dir=tmp_path,
    )
    stamp = re.compile(r"^.*mushin_run_id.*\n", re.M)
    for path in [tmp_path / "multirun.yaml", *tmp_path.glob("*/.hydra/*.yaml")]:
        path.write_text(stamp.sub("", path.read_text()))
    for path in tmp_path.glob("*/mushin_cell_status.json"):
        record = json.loads(path.read_text())
        record.pop("run_id", None)
        path.write_text(json.dumps(record))
    (tmp_path / MANIFEST_FILE).unlink()
    back = Named().load_from_dir(tmp_path, "mushin_metrics.json")
    assert _values(back) == [4.0, 1.0, 4.0, 1.0]


class AnyRate(MultiRunMetricsWorkflow):
    @staticmethod
    def task(lr, seed, base_lr=None):
        return dict(v=1.0)


def test_a_repeated_interpolation_that_cannot_be_read(tmp_path):
    wf = AnyRate()
    wf.run(
        base_lr=0.01,
        lr=multirun(["${base_lr}", "${hydra:job.num}", "${hydra:job.num}_b"]),
        seed=multirun([0]),
        working_dir=tmp_path,
    )
    assert wf.multirun_task_overrides["lr"][0] == "0.01"


class Tagged(MultiRunMetricsWorkflow):
    @staticmethod
    def task(name, tag=None):
        return dict(v=float(len(name)))


def test_an_interpolation_that_reads_as_several_elements(tmp_path):
    """An unquoted `${tag}` that resolves to "foo,bar" reads, in the driver,
    as three elements where the axis has two: that reading is not the
    axis's. With no record or saved decision for the element (its job died,
    and so did the driver), it keys by its text, not by the reading's first
    part."""
    wf = Tagged()
    wf.run(tag="foo,bar", overrides=["+name=${tag},b"], working_dir=tmp_path)
    assert _values(wf) == [7.0, 1.0]
    (tmp_path / MANIFEST_FILE).unlink()
    (tmp_path / "0" / "mushin_cell_status.json").unlink()
    back = Tagged().load_from_dir(tmp_path, "mushin_metrics.json")
    assert back.multirun_task_overrides["name"] == ["${tag}", "b"]


def test_a_load_replaces_a_runs_manifest(tmp_path):
    first = Plain()
    first.run(x=multirun(["a"]), working_dir=tmp_path / "a", notes="A")
    Plain().run(x=multirun(["b"]), working_dir=tmp_path / "b")
    first.load_from_dir(tmp_path / "b", "mushin_metrics.json")
    fresh = Plain().load_from_dir(tmp_path / "b", "mushin_metrics.json")
    assert first.notes == fresh.notes


def _seen(value):
    """Keep the value a task received in its cell's dir (the cwd): what its
    job resolved, whatever provenance records."""
    Path("seen.json").write_text(json.dumps(value))


def _seen_in(cell: Path):
    return json.loads((cell / "seen.json").read_text())


class Stamped(MultiRunMetricsWorkflow):
    @staticmethod
    def task(x, stamp):
        _seen(stamp)
        return dict(v=float(x))


@pytest.mark.parametrize("spec", ["%Y", "%H-%M-%S-%f"])
def test_a_fixed_interpolation_is_the_runs_value_on_every_read(tmp_path, spec):
    """The issue's second case: a fixed value is the one its jobs resolved
    when they agree (the run's year), else its text, not the reader's
    reading at reload time."""
    stamp = "${now:" + spec + "}"
    wf = Stamped()
    wf.run(x=multirun([1, 2]), stamp=stamp, working_dir=tmp_path)
    seen = [_seen_in(tmp_path / d) for d in ("0", "1")]
    expected = seen[0] if seen[0] == seen[1] else stamp
    assert wf.multirun_task_overrides["stamp"] == expected
    for _ in range(2):
        back = Stamped().load_from_dir(tmp_path, "mushin_metrics.json")
        assert back.multirun_task_overrides["stamp"] == expected


class Listed2(MultiRunMetricsWorkflow):
    @staticmethod
    def task(x, stamps):
        _seen(list(stamps))
        return dict(v=float(x))


def test_a_fixed_container_is_read_only_from_jobs_that_resolved_it(tmp_path):
    """A cell `sample=` left out never resolved its config; a list holding
    an interpolation there is not read (it would resolve now, in the
    driver), so the run's value is the computed cell's (#392)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        wf = Listed2()
        wf.run(
            x=multirun([1, 2]),
            stamps=hydra_list(["${now:%H-%M-%S-%f}"]),
            working_dir=tmp_path,
            sample=1,
            sample_seed=0,
        )
    ran = [d for d in ("0", "1") if (tmp_path / d / "mushin_metrics.json").exists()]
    assert len(ran) == 1
    assert wf.multirun_task_overrides["stamps"] == _seen_in(tmp_path / ran[0])


@pytest.mark.config_group("model")
def test_a_fixed_interpolated_group_choice_reads_its_option_name(
    tmp_path, restore_config_group, monkeypatch
):
    """A config group's option chosen through an interpolation reads as the
    option's name, as on main, not as the composed node, in the run and in a
    reload (#392). Read from the environment, it is not saved: a reload reads
    it again, as main does."""
    from hydra.core.config_store import ConfigStore

    cs = ConfigStore.instance()
    cs.store(group="model", name="small", node={"width": 4})
    cs.store(group="model", name="big", node={"width": 8})

    class Grouped(MultiRunMetricsWorkflow):
        @staticmethod
        def task(seed, model=None):
            return dict(w=float(model["width"]))

    monkeypatch.setenv("MUSHIN_TEST_MODEL", "small")
    wf = Grouped(make_config(model=None))
    wf.run(
        model="${oc.env:MUSHIN_TEST_MODEL}",
        seed=multirun([0, 1]),
        working_dir=tmp_path,
    )
    assert wf.multirun_task_overrides["model"] == "small"
    back = Grouped(make_config(model=None)).load_from_dir(
        tmp_path, "mushin_metrics.json"
    )
    assert back.multirun_task_overrides["model"] == "small"


class Weighted(MultiRunMetricsWorkflow):
    @staticmethod
    def task(x, ws, base_lr=None):
        return dict(v=float(x))


def test_a_fixed_container_keeps_its_nan_on_a_reload(tmp_path):
    """The run saves a fixed list with its NaN tagged, so a reload reads NaN
    there rather than None (#392)."""
    wf = Weighted()
    wf.run(
        base_lr=0.01,
        x=multirun([1, 2]),
        ws=hydra_list([float("nan"), "${base_lr}"]),
        working_dir=tmp_path,
    )
    ran = wf.multirun_task_overrides["ws"]
    back = Weighted().load_from_dir(tmp_path, "mushin_metrics.json")
    read = back.multirun_task_overrides["ws"]
    assert str(read) == str(ran)
    assert read[0] is not None


class Mutates(MultiRunMetricsWorkflow):
    @staticmethod
    def task(x):  # raw: called with the whole config
        x.lr = 99
        return dict(v=1.0)


def test_a_raw_task_that_changes_a_fixed_value_does_not_report_its_change(tmp_path):
    """A task run without a wrapper gets the config itself and may change a
    fixed value; the value reported is the one the sweep was launched with,
    not the task's change (#392)."""
    wf = Mutates()
    wf.run(
        base_lr=0.01,
        lr="${base_lr}",
        x=multirun([1, 2]),
        working_dir=tmp_path,
        task_fn_wrapper=None,
    )
    assert wf.multirun_task_overrides["lr"] != 99
    back = Mutates().load_from_dir(tmp_path, "mushin_metrics.json")
    assert back.multirun_task_overrides["lr"] != 99
    assert str(back.multirun_task_overrides["lr"]) == str(
        wf.multirun_task_overrides["lr"]
    )


class Mapped(MultiRunMetricsWorkflow):
    @staticmethod
    def task(d, base=None):
        return dict(v=float(len(str(d))))


def test_a_nan_inside_a_dict_element_reloads_as_nan(tmp_path):
    """A dict element written as itself keeps its NaN in the saved
    coordinates, so a reload does not read it as None and key it with a
    swept None (#386)."""
    wf = Mapped()
    wf.run(
        base=1,
        d=multirun([{"k": float("nan")}, {"k": "${base}"}, {"k": None}]),
        working_dir=tmp_path,
    )
    back = Mapped().load_from_dir(tmp_path, "mushin_metrics.json")
    for read in (wf, back):
        first = read.multirun_task_overrides["d"][0]
        assert isinstance(first["k"], float) and math.isnan(first["k"])
        assert not any(math.isnan(v) for v in _values(read))


def test_a_user_dict_shaped_like_a_tag_is_kept(tmp_path):
    """A swept dict whose only key is the NaN tag's, holding something that
    is not a tag, is a value of the user's: it is read as stored, and the
    run does not raise (#386)."""
    wf = Mapped()
    wf.run(
        base=1,
        d=multirun([{"__mushin_nonfinite__": [1]}, {"k": "${base}"}]),
        working_dir=tmp_path,
    )
    assert wf.multirun_task_overrides["d"][0] == {"__mushin_nonfinite__": [1]}
    assert not any(math.isnan(v) for v in _values(wf))


def test_dict_elements_whose_keys_differ_only_in_type_are_two_elements():
    """A key ``True`` and a key ``1`` are equal to Python's dict; as parsed
    override values they are two elements (#386)."""
    from mushin.workflows import _same_element

    assert not _same_element({True: "a"}, {1: "a"})
    assert _same_element({1: "a"}, {1: "a"})


def test_a_coordinate_json_cannot_keep_reloads_as_it_ran(tmp_path):
    """JSON keeps a dict's keys only as text: a coordinate holding ``{1: "a"}``
    is not saved in that lossy form, so a reload reads it as the run did
    (#386)."""
    wf = Listed()
    wf.run(base_lr=1, xs=multirun([[{1: "a"}], ["${base_lr}"]]), working_dir=tmp_path)
    back = Listed().load_from_dir(tmp_path, "mushin_metrics.json")
    assert [str(x) for x in back.multirun_task_overrides["xs"]] == [
        str(x) for x in wf.multirun_task_overrides["xs"]
    ]
    assert _values(back) == _values(wf)


def test_a_dict_element_with_a_nan_key_is_its_own_element():
    """A NaN key equals a NaN key, as a NaN value does (#386)."""
    from mushin.workflows import _same_element

    assert _same_element({float("nan"): "a"}, {float("nan"): "a"})
    assert not _same_element({float("nan"): "a"}, {float("nan"): "b"})
    assert not _same_element({float("nan"): "a"}, {1.0: "a"})


class Tokened(MultiRunMetricsWorkflow):
    @staticmethod
    def task(x, token):
        return dict(v=float(x))


def test_a_fixed_value_read_from_the_environment_is_not_saved(tmp_path, monkeypatch):
    """A credential a fixed override reads from the environment never
    reaches the manifest; it is read again, as before (#392)."""
    monkeypatch.setenv("MUSHIN_TEST_TOKEN", "tok-should-not-be-saved")
    wf = Tokened()
    wf.run(
        x=multirun([1, 2]), token="${oc.env:MUSHIN_TEST_TOKEN}", working_dir=tmp_path
    )
    assert "tok-should-not-be-saved" not in (tmp_path / MANIFEST_FILE).read_text()
    back = Tokened().load_from_dir(tmp_path, "mushin_metrics.json")
    assert back.multirun_task_overrides["token"] == wf.multirun_task_overrides["token"]


class Keyed(MultiRunMetricsWorkflow):
    @staticmethod
    def task(x, d, base_lr=None):
        return dict(v=float(x))


def test_a_fixed_dict_with_a_number_key_reloads_as_it_ran(tmp_path):
    """JSON keeps a dict's keys only as text: a fixed value holding ``{1: ...}``
    is not saved in that lossy form, and reloads as the run read it (#392)."""
    wf = Keyed()
    wf.run(
        base_lr=0.01,
        x=multirun([1, 2]),
        overrides=["+d={1:'${base_lr}'}"],
        working_dir=tmp_path,
    )
    back = Keyed().load_from_dir(tmp_path, "mushin_metrics.json")
    ran, read = wf.multirun_task_overrides["d"], back.multirun_task_overrides["d"]
    assert list(read) == list(ran) and type(next(iter(read))) is int


class Dirred(MultiRunMetricsWorkflow):
    @staticmethod
    def task(x):
        return dict(v=float(Path.cwd().name))


def test_an_element_that_resolves_to_anothers_value_keeps_its_cell(tmp_path):
    """Job 0's ``${hydra:job.num}`` resolves to 0, the other element's
    value: keyed by it, the two cells would share a key and one result would
    overwrite the other. The element keeps its text instead (#386)."""
    wf = Dirred()
    wf.run(x=multirun(["${hydra:job.num}", 0]), working_dir=tmp_path)
    assert sorted(_values(wf)) == [0.0, 1.0]
    back = Dirred().load_from_dir(tmp_path, "mushin_metrics.json")
    assert sorted(_values(back)) == [0.0, 1.0]
    assert [str(x) for x in back.multirun_task_overrides["x"]] == [
        str(x) for x in wf.multirun_task_overrides["x"]
    ]


@pytest.mark.parametrize("where", ["override", "config"])
def test_a_fixed_value_resolved_through_an_environment_read_is_not_saved(
    tmp_path, monkeypatch, where
):
    """``token="${api_key}"`` reads no environment itself, but ``api_key``
    does (an override, or the base config): no fixed value of a run whose
    config reads the environment is saved, so the credential never reaches
    the manifest (#392)."""
    monkeypatch.setenv("MUSHIN_TEST_TOKEN", "tok-indirect-secret")

    class Creds(MultiRunMetricsWorkflow):
        @staticmethod
        def task(x, token, api_key=None):
            return dict(v=float(x))

    env = "${oc.env:MUSHIN_TEST_TOKEN}"
    if where == "override":
        wf = Creds()
        wf.run(
            api_key=env, token="${api_key}", x=multirun([1, 2]), working_dir=tmp_path
        )
    else:
        wf = Creds(make_config(api_key=env))
        wf.run(token="${api_key}", x=multirun([1, 2]), working_dir=tmp_path)
    assert wf.multirun_task_overrides["token"] == "tok-indirect-secret"
    assert "tok-indirect-secret" not in (tmp_path / MANIFEST_FILE).read_text()
