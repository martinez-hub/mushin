# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT
import inspect
import string
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import hypothesis.strategies as st
import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch as tr
import xarray as xr
from hydra.core.config_store import ConfigStore
from hydra.plugins.sweeper import Sweeper
from hydra_zen import builds, load_from_yaml, make_config
from hydra_zen.errors import HydraZenValidationError
from hypothesis import given, settings
from hypothesis.extra.numpy import array_shapes, arrays
from numpy.testing import assert_allclose
from xarray.testing import assert_duckarray_equal, assert_identical

from mushin import multirun
from mushin.workflows import (
    BaseWorkflow,
    MultiRunMetricsWorkflow,
    RobustnessCurve,
    hydra_list,
)

common_shape = array_shapes(min_dims=2, max_dims=2)

epsilons = arrays(
    shape=st.integers(1, 5),
    dtype="float",
    elements=st.floats(-1000, 1000),
)


class MyWorkflow(BaseWorkflow):
    @staticmethod
    def task():
        return dict(result=1)

    def jobs_post_process(self):
        return


def create_workflow(as_array=False):
    class LocalRobustness(RobustnessCurve):
        @staticmethod
        def task(epsilon):
            val = 100 - epsilon**2
            if as_array:
                val = [val]

            result = dict(result=val)

            tr.save(result, "test_metrics.pt")
            return result

    return LocalRobustness


def test_robustnesscurve_raises_notimplemented():
    task = MyWorkflow()

    with pytest.raises(NotImplementedError):
        task.plot()

    with pytest.raises(NotImplementedError):
        task.to_xarray()


@pytest.mark.usefixtures("cleandir")
def test_robustnesscurve_validate():
    LocalRobustness = create_workflow()

    task = LocalRobustness(make_config(epsilon=0))
    task.validate()

    task = LocalRobustness()
    with pytest.raises(HydraZenValidationError):
        task.validate()


@pytest.mark.usefixtures("cleandir")
@pytest.mark.parametrize(
    "config", [None, make_config(epsilon=0), make_config(epsilon=None)]
)
@pytest.mark.parametrize("as_array", [True, False])
def test_robustnesscurve_run(config, as_array):
    epsilon = [0, 1, 2, 3.0]
    LocalRobustness = create_workflow(as_array)
    task = LocalRobustness(config)
    task.run(epsilon=epsilon)

    assert "result" in task.metrics
    assert len(task.metrics["result"]) == len(epsilon)

    multirun_task_overrides = task.multirun_task_overrides
    assert "epsilon" in multirun_task_overrides

    extracted_epsilon = multirun_task_overrides["epsilon"]
    assert isinstance(extracted_epsilon, Sequence)
    assert len(extracted_epsilon) == len(epsilon)

    # will raise if not set correctly
    task.plot("result")


@pytest.mark.usefixtures("cleandir")
def test_robustnesscurve_working_dir():
    LocalRobustness = create_workflow()
    task = LocalRobustness(make_config(epsilon=0))
    task.run(epsilon="0,1,2,3", working_dir="test_dir")

    assert str(task.working_dir).endswith("test_dir")
    assert Path("test_dir").exists()


@pytest.mark.usefixtures("cleandir")
@pytest.mark.parametrize(
    "sweeper,launcher", [(None, None), (None, "basic"), ("basic", None)]
)
def test_robustnesscurve_hydra(sweeper, launcher):
    LocalRobustness = create_workflow()
    task = LocalRobustness(make_config(epsilon=0))
    task.run(epsilon=[0, 1, 2, 3], sweeper=sweeper, launcher=launcher)


@pytest.mark.usefixtures("cleandir")
def test_robustnesscurve_override():
    LocalRobustness = create_workflow()
    task = LocalRobustness(make_config(epsilon=0))

    overrides = ["hydra.sweep.dir=test_sweep_dir"]
    task.run(epsilon=[0, 1, 2, 3], overrides=overrides)
    assert Path("test_sweep_dir").exists()

    # make sure overrides is not modified
    assert len(overrides) == 1


@pytest.mark.usefixtures("cleandir")
@settings(deadline=None, max_examples=10)
@given(epsilon=epsilons)
def test_robustnesscurve_to_data(epsilon):
    LocalRobustness = create_workflow()
    task = LocalRobustness(make_config(epsilon=0))
    task.run(epsilon=epsilon)

    xd = task.to_xarray(non_multirun_params_as_singleton_dims=True)

    assert isinstance(xd, xr.Dataset)
    assert len(xd["epsilon"]) == len(epsilon)


@pytest.mark.usefixtures("cleandir")
def test_robustnesscurve_load_from_dir():
    LocalRobustness = create_workflow()
    task = LocalRobustness()
    task.run(epsilon=[0, 1, 2, 3], working_dir="test_dir")

    working_dir = "test_dir"
    LocalRobustness = create_workflow()
    load_task = LocalRobustness()
    load_task.load_from_dir(working_dir, "test_metrics.pt")

    for k, v in task.multirun_task_overrides.items():
        assert k in load_task.multirun_task_overrides
        assert v == load_task.multirun_task_overrides[k]

    for k, v in task.metrics.items():
        assert k in load_task.metrics
        assert v == load_task.metrics[k]

    for cfg in task.cfgs:
        assert "epsilon" in cfg


@pytest.mark.usefixtures("cleandir")
@pytest.mark.parametrize("fake_param_string", [True, False])
def test_robustnesscurve_extra_param(fake_param_string: bool):
    class BadVal:
        pass

    LocalRobustness = create_workflow()
    task = LocalRobustness(make_config(epsilon=0))

    if fake_param_string:
        task.run(epsilon=[0, 1, 2, 3], fake_param="some_value")
    else:
        with pytest.raises(TypeError):
            task.run(epsilon=[0, 1, 2, 3], fake_param=BadVal)  # type: ignore
        return

    assert "fake_param" in task.multirun_task_overrides
    assert task.multirun_task_overrides["fake_param"] == "some_value"
    task.plot("result", group="fake_param", non_multirun_params_as_singleton_dims=True)


@pytest.mark.usefixtures("cleandir")
@pytest.mark.parametrize("fake_param_string", [True, False])
def test_robustnesscurve_extra_param_multirun(fake_param_string: bool):
    LocalRobustness = create_workflow()
    task = LocalRobustness(make_config(epsilon=0))

    if fake_param_string:
        task.run(epsilon=[0, 1, 2, 3], fake_param="1,2")
        task.plot("result")
    else:
        with pytest.raises(TypeError):
            task.run(epsilon=[0, 1, 2, 3], fake_param=[1, 2])  # type: ignore
        return

    multirun_task_overrides = task.multirun_task_overrides
    assert "fake_param" in multirun_task_overrides

    extracted_fake_param = multirun_task_overrides["fake_param"]
    assert isinstance(extracted_fake_param, Sequence)
    assert len(extracted_fake_param) == 2


@pytest.mark.usefixtures("cleandir")
@pytest.mark.parametrize("ax", [None, plt.subplots()[1]])
def test_robustnesscurve_plot_save(ax):
    LocalRobustness = create_workflow()
    task = LocalRobustness()
    task.run(epsilon=[0, 1, 2, 3])
    task.plot("result", save_filename="test_save.png", ax=ax)
    assert Path("test_save.png").exists()


class MultiDimMetrics(RobustnessCurve):
    # returns     "images" -> shape-(4, 1)
    #         "accuracies" -> scalar
    @staticmethod
    def task(epsilon):
        val = 100 - epsilon**2
        result = dict(images=[[val] * 1] * 4, accuracies=val + 2)
        tr.save(result, "test_metrics.pt")
        return result


@pytest.mark.usefixtures("cleandir")
@pytest.mark.parametrize(
    "foo, foo_expected", [("val", "val"), (hydra_list(["val"]), "['val']")]
)
@pytest.mark.parametrize("bar", [multirun(["a", "b"]), multirun(["[a,b]", "[c,d]"])])
def test_robustness_with_multidim_metrics(foo, foo_expected, bar):
    wf = MultiDimMetrics()
    wf.run(epsilon=[1.0, 3.0, 2.0], foo=foo, bar=bar)
    xarray = wf.to_xarray()
    assert list(xarray.data_vars.keys()) == ["images", "accuracies"]
    assert list(xarray.coords.keys()) == [
        "bar",
        "epsilon",
        "images_dim0",
        "images_dim1",
    ]
    assert xarray.accuracies.shape == (2, 3)
    assert xarray.images.shape == (2, 3, 4, 1)
    assert {k: v for k, v in xarray.attrs.items() if k != "provenance"} == {
        "foo": foo_expected
    }

    for eps, expected in zip([1.0, 2.0, 3.0], [99.0, 96.0, 91.0]):
        # test that results were organized as-expected
        sub_xray = xarray.sel(epsilon=eps)
        assert np.all(sub_xray.accuracies == expected + 2).item()
        assert np.all(sub_xray.images == expected).item()


class MultiDimIterationMetrics(MultiRunMetricsWorkflow):
    # returns "images" -> shape-(N, 4, 4)
    #         "accuracies" -> N
    #         "epochs" -> (N, 10)

    @staticmethod
    def task(epsilon, as_tensor: bool):
        assert isinstance(as_tensor, bool)
        backend = np if as_tensor else tr
        val = 100 * backend.ones(10) - epsilon**2
        epochs = backend.arange(10)
        images = 100 * backend.ones((10, 4, 4)) - epsilon**2
        result = dict(images=images, accuracies=val + 2, epochs=epochs)
        tr.save(result, "test_metrics.pt")
        return result


@pytest.mark.usefixtures("cleandir")
@pytest.mark.parametrize("as_tensor", [False, True])
def test_robustness_with_multidim_metrics_with_iteration(as_tensor: bool):
    wf = MultiDimIterationMetrics()
    wf.run(
        epsilon=multirun([1.0, 3.0, 2.0]),
        foo="val",
        bar=multirun(["a", "b"]),
        as_tensor=as_tensor,
    )
    xarray = wf.to_xarray(coord_from_metrics="epochs")
    assert list(xarray.data_vars.keys()) == ["images", "accuracies"]
    assert list(xarray.coords.keys()) == [
        "epsilon",
        "bar",
        "epochs",
        "images_dim1",
        "images_dim2",
    ]
    assert xarray.accuracies.shape == (3, 2, 10)
    assert xarray.images.shape == (3, 2, 10, 4, 4)
    assert {k: v for k, v in xarray.attrs.items() if k != "provenance"} == {
        "foo": "val",
        "as_tensor": as_tensor,
    }

    for eps, expected in zip([1.0, 2.0, 3.0], [99.0, 96.0, 91.0]):
        # test that results were organized as-expected
        sub_xray = xarray.sel(epsilon=eps)
        assert np.all(sub_xray.accuracies == expected + 2).item()
        assert np.all(sub_xray.images == expected).item()

    with pytest.raises(ValueError):
        wf.to_xarray(coord_from_metrics="key_not_in_metrics")


@pytest.mark.usefixtures("cleandir")
def test_xarray_from_loaded_workflow():
    wf = MultiDimMetrics()
    wf.run(epsilon=[1.0, 3.0, 2.0], foo="val", bar=multirun(["a", "b"]))
    xarray1 = wf.to_xarray()

    wf2 = MultiDimMetrics().load_from_dir(wf.working_dir, "test_metrics.pt")
    xarray2 = wf2.to_xarray()
    assert_identical(xarray1, xarray2)

    wf3 = MultiDimMetrics(working_dir=wf.working_dir)
    xarray3 = wf3.to_xarray(metrics_filename="test_metrics.pt")
    assert_identical(xarray1, xarray3)

    wf4 = MultiDimMetrics().load_from_dir(wf.working_dir, metrics_filename=None)
    xarray4 = wf4.to_xarray(metrics_filename="test_metrics.pt")
    assert_identical(xarray1, xarray4)


class LocalBasicSweeper(Sweeper):
    def setup(self, *, hydra_context, task_function, config):
        pass

    def sweep(self, arguments):
        return dict(hi=1)


@pytest.mark.usefixtures("cleandir")
def test_return_not_list_jobreturn():
    cs = ConfigStore.instance()
    cs.store(group="hydra/sweeper", name="local_test", node=builds(LocalBasicSweeper))

    wf = MyWorkflow()
    wf.run(epsilon=multirun([1.0, 3.0, 2.0]), overrides=["hydra/sweeper=local_test"])
    assert wf.jobs == dict(hi=1)


class FirstMultiRun(RobustnessCurve):
    # returns     "images" -> shape-(4, 1)
    #         "accuracies" -> scalar
    @staticmethod
    def task(epsilon, acc):
        val = 100 - epsilon**2
        result = dict(images=np.array([[val] * 1] * 4), accuracies=acc)
        tr.save(result, "test_metrics.pt")
        return result


class ScndMultiRun(MultiRunMetricsWorkflow):
    # loads test metrics, multiplies each by `val` and saves
    @staticmethod
    def task(job_dir, val):
        result = tr.load(f"{job_dir}/test_metrics.pt", weights_only=False)

        # val multiplies each metric
        result = {k: v * val for k, v in result.items()}
        tr.save(result, "test_metrics.pt")
        return result


@pytest.mark.parametrize("load_from_working_dir", [False, True])
@pytest.mark.usefixtures("cleandir")
@pytest.mark.filterwarnings("ignore:invalid value encountered in cast")
def test_multirun_over_jobdir(load_from_working_dir):
    # Runs a standard multirun workflow and then runs
    # a multirun over the resulting folders, loading in
    # their metrics and re-returning them
    wf = FirstMultiRun()
    wf.run(
        epsilon=multirun([1.0, 2.0, 3.0]),
        acc=multirun([1, 2]),
        list_vals=multirun([[0, 1]]),  # ensure that multiruns over lists work
        working_dir="first",
    )

    snd_wf = ScndMultiRun()
    # runs over a total of epsilon-3 x acc-2 -> 6 job-dirs and 2 val
    snd_wf.run(
        target_job_dirs=wf.multirun_working_dirs,
        val=multirun([1, 2]),
        working_dir="second",
    )

    if load_from_working_dir:
        snd_wf = ScndMultiRun().load_from_dir(snd_wf.working_dir, "test_metrics.pt")

    assert wf.target_dir_multirun_overrides == {}
    assert snd_wf.target_dir_multirun_overrides == {
        "acc": [1, 1, 1, 2, 2, 2],
        "epsilon": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
        "list_vals": [[0, 1]] * 6,
    }
    xr1 = wf.to_xarray()
    xr2 = snd_wf.to_xarray()

    # `list_vals` is a length-1 multirun, so it is a swept (length-1) dimension
    assert xr1.sizes == {
        "acc": 2,
        "list_vals": 1,
        "epsilon": 3,
        "images_dim0": 4,
        "images_dim1": 1,
    }
    assert xr2.sizes == {"val": 2, "job_dir": 6, "images_dim0": 4, "images_dim1": 1}

    # drop the length-1 list_vals dim from xr1 for comparison with xr2 selections
    xr1 = xr1.sel(list_vals="[0, 1]").drop_vars("list_vals")

    xr2 = xr2.set_index(job_dir=["epsilon", "acc", "list_vals"]).unstack("job_dir")
    xr2 = xr2.transpose(
        "list_vals", "val", "acc", "epsilon", "images_dim0", "images_dim1"
    )

    assert_identical(xr1.epsilon, xr2.epsilon)
    assert_identical(xr1.acc, xr2.acc)

    assert_duckarray_equal(xr1.images, xr2.images.sel(val=1, list_vals="[0, 1]"))
    assert_duckarray_equal(2 * xr1.images, xr2.images.sel(val=2, list_vals="[0, 1]"))
    assert_duckarray_equal(
        xr1.accuracies, xr2.accuracies.sel(val=1, list_vals="[0, 1]")
    )
    assert_duckarray_equal(
        2 * xr1.accuracies, xr2.accuracies.sel(val=2, list_vals="[0, 1]")
    )


class NoMetrics(MultiRunMetricsWorkflow):
    @staticmethod
    def task(x: int, y: int):
        pass


@pytest.mark.parametrize("load_from_working_dir", [False, True])
@pytest.mark.usefixtures("cleandir")
def test_multirun_metrics_workflow_no_metrics(load_from_working_dir):
    wf = NoMetrics()
    wf.run(x=multirun([-1, 0, 1]), y=multirun([-10, 10]))
    assert wf.multirun_task_overrides == {"x": [-1, 0, 1], "y": [-10, 10]}

    if load_from_working_dir:
        wf = NoMetrics().load_from_dir(wf.working_dir, metrics_filename=None)

    xdata = wf.to_xarray()
    assert xdata.sizes == {"x": 3, "y": 2}
    assert_allclose(xdata.coords["x"].data, [-1, 0, 1])
    assert_allclose(xdata.coords["y"].data, [-10, 10])
    assert len(xdata.data_vars) == 0


class GridMetrics(MultiRunMetricsWorkflow):
    @staticmethod
    def task(x: int, y: int):
        results = dict(xx=x, yy=y)
        tr.save(results, "test_metrics.pt")
        return results


@pytest.mark.parametrize("hydra_sweep_dir", [None, "cross_validation/", "."])
@pytest.mark.parametrize("hydra_sweep_subdir", [None, "x_${x}_y_${y}"])
@pytest.mark.parametrize("load_from_working_dir", [False, True])
@pytest.mark.usefixtures("cleandir")
def test_working_subdirs(
    hydra_sweep_dir: str | None,
    hydra_sweep_subdir: str | None,
    load_from_working_dir: bool,
):
    overrides = [
        f"{k.replace('_', '.')}={v}"
        for k, v in locals().items()
        if k.startswith("hydra") and v is not None
    ]

    # just to keep these from looking like they are unused
    del hydra_sweep_dir, hydra_sweep_subdir
    wf = GridMetrics()
    wf.run(x=multirun([-1, 0, 1]), y=multirun([-10, 10]), overrides=overrides)

    if load_from_working_dir:
        wf = GridMetrics().load_from_dir(
            wf.working_dir, metrics_filename="test_metrics.pt"
        )

    xdata = wf.to_xarray(include_working_subdirs_as_data_var=True)

    # ensure data variables are set appropriately
    yy, xx = np.meshgrid([-10, 10], [-1, 0, 1])
    assert_allclose(actual=xdata.xx.data, desired=xx)
    assert_allclose(actual=xdata.yy.data, desired=yy)

    # ensure working_subdir points to correct dir
    for x_coord in xdata.x:
        for y_coord in xdata.y:
            dd = Path(xdata.working_subdir.sel(x=x_coord, y=y_coord).item())
            cfg = load_from_yaml(dd / ".hydra" / "config.yaml")
            assert cfg == dict(x=x_coord.item(), y=y_coord.item())

    # ensure working_subdir is serializable
    xdata.to_netcdf("tmp.nc")


@pytest.mark.usefixtures("cleandir")
@pytest.mark.parametrize(
    "file_pattern", ["*.pt", ("*.pt",), ("images.pt", "acc.pt"), ("images.*", "acc.*")]
)
def test_globbed_xarray(file_pattern):
    class MultiSaveFile(MultiRunMetricsWorkflow):
        # returns     "images" -> shape-(4, 1)
        #         "accuracies" -> scalar
        @staticmethod
        def task(epsilon, acc):
            val = 100 - epsilon**2
            result = dict(images=np.array([[val] * 1] * 4), accuracies=acc)
            tr.save(dict(images=result["images"]), "images.pt")
            tr.save(dict(accuracies=result["accuracies"]), "acc.pt")

            return result

    # saves multiple metrics files that we load/merge via glob pattern
    wf = MultiSaveFile()
    wf.run(epsilon=multirun([1, 2, 3]), acc=multirun([0.9, 0.95, 0.99]))
    xdata1 = wf.to_xarray()

    wf2 = MultiSaveFile(working_dir=wf.working_dir)
    xdata2 = wf2.to_xarray(metrics_filename=file_pattern)

    assert_identical(xdata1, xdata2)


@pytest.mark.parametrize("seed", [0, 123])
@pytest.mark.usefixtures("cleandir")
def test_pre_task_seeding(seed: int):
    class HasPreTask(MultiRunMetricsWorkflow):
        @staticmethod
        def pre_task(seed: int):
            np.random.seed(seed)

        @staticmethod
        def task(rand_val: int):
            return {"rand_val": rand_val}

    wf = HasPreTask(make_config(rand_val=builds(np.random.rand)))
    wf.run(seed=seed)
    actual = wf.jobs[0].return_value["rand_val"]

    np.random.seed(seed)
    expected = np.random.rand()
    assert expected == actual


def test_raises_on_non_static_method():
    class NonStaticEvalTask(MultiRunMetricsWorkflow):
        def task(self):
            pass

    class NonStaticPreTask(MultiRunMetricsWorkflow):
        def pre_task(self):
            pass

    with pytest.raises(TypeError, match="task must be a static method"):
        NonStaticEvalTask().run()

    with pytest.raises(TypeError, match="pre_task must be a static method"):
        NonStaticPreTask().run()


@pytest.mark.usefixtures("cleandir")
@settings(max_examples=10, deadline=None)
@given(
    int_=st.integers(),
    bool_=st.booleans(),
    float_=st.floats(-10, 10),
    list_=st.lists(st.integers()),
    # Exclude tokens that Hydra's override parser reinterprets as non-strings
    # (e.g. "null" -> None, "inf"/"nan" -> float), which can't round-trip as str.
    str_=st.text(alphabet=string.ascii_lowercase).filter(
        lambda x: x not in {"true", "false", "null", "none", "nan", "inf"}
    ),
    mrun=st.lists(
        st.booleans() | st.lists(st.integers()),
        min_size=2,
        max_size=5,
    ).map(multirun),
)
def test_overrides_roundtrip(
    int_,
    bool_,
    float_,
    str_,
    list_,
    mrun,
):
    class WorkFlow(MultiDimIterationMetrics):
        @staticmethod
        def task():
            pass

    wf = WorkFlow()
    overrides: dict[str, Any] = dict(
        int_=int_,
        float_=float_,
        str_=str_,
        bool_=bool_,
        list_=hydra_list(list_),
        mrun=mrun,
    )
    wf.run(**overrides)

    assert wf.multirun_task_overrides == overrides
    xdata = wf.to_xarray(non_multirun_params_as_singleton_dims=True)
    assert xdata.int_.item() == int_
    assert xdata.bool_.item() == bool_
    assert xdata.str_.item() == str_
    if not isinstance(mrun[0], list) and all(
        isinstance(i, type(mrun[0])) for i in mrun
    ):
        assert xdata.mrun.data.tolist() == mrun
    else:
        assert xdata.mrun.data.tolist() == [str(i) for i in mrun]


@pytest.mark.usefixtures("cleandir")
def test_custom_metric_load_fn():
    import pickle

    class PickleWorkFlow(MultiRunMetricsWorkflow):
        def metric_load_fn(self, file_path: Path):
            with file_path.open("rb") as f:
                return pickle.load(f)

        @staticmethod
        def task(a, b):
            with open("./metrics.pkl", "wb") as f:
                pickle.dump(dict(a=[[a] * 2], b=b), f)

    wf = PickleWorkFlow()
    wf.run(a=multirun([1, 2, 3]), b=False)
    wf.load_metrics("metrics.pkl")
    assert wf.metrics == dict(a=[[1] * 2, [2] * 2, [3] * 2], b=[False] * 3)


@pytest.mark.usefixtures("cleandir")
def test_g1_multirun_string_value_with_comma_not_split():
    # G1: a multirun whose string elements contain commas must launch exactly
    # one job per element (not silently re-split into more jobs), and the
    # labels must round-trip exactly.
    class Blank(MultiRunMetricsWorkflow):
        @staticmethod
        def task(label):
            return None

    wf = Blank()
    wf.run(label=multirun(["hello world", "foo,bar"]))

    assert len(wf.jobs) == 2
    assert wf.multirun_task_overrides["label"] == multirun(["hello world", "foo,bar"])

    xdata = wf.to_xarray()
    assert xdata.sizes == {"label": 2}
    assert xdata.coords["label"].data.tolist() == ["hello world", "foo,bar"]


@pytest.mark.usefixtures("cleandir")
def test_g2_length_one_multirun_is_a_sweep():
    # G2: a length-1 multirun must remain a swept dimension (coord), not be
    # collapsed into a scalar attribute.
    class Blank(MultiRunMetricsWorkflow):
        @staticmethod
        def task(a):
            return None

    wf = Blank()
    wf.run(a=multirun([5]))

    assert len(wf.jobs) == 1
    # recorded as a sweep (list), not a scalar
    assert wf.multirun_task_overrides["a"] == multirun([5])

    xdata = wf.to_xarray()
    assert "a" in xdata.coords
    assert "a" not in xdata.attrs
    assert xdata.sizes == {"a": 1}
    assert xdata.coords["a"].data.tolist() == [5]


@pytest.mark.usefixtures("cleandir")
def test_g4_load_from_dir_resets_cached_overrides():
    # G4: a reused workflow object must not return stale multirun overrides
    # after load_from_dir points it at a different directory.
    class Blank(MultiRunMetricsWorkflow):
        @staticmethod
        def task(a):
            return None

    wf_a = Blank()
    wf_a.run(a=multirun([1, 2, 3]), working_dir="dir_a")

    wf_b = Blank()
    wf_b.run(a=multirun([7, 8]), working_dir="dir_b")

    loader = Blank()
    loader.load_from_dir(wf_a.working_dir, metrics_filename=None)
    # populate the cache for dir_a
    assert loader.multirun_task_overrides["a"] == multirun([1, 2, 3])
    assert loader.to_xarray().sizes == {"a": 3}

    # now point it at dir_b -- must reflect the NEW dir, not stale dir_a
    loader.load_from_dir(wf_b.working_dir, metrics_filename=None)
    assert loader.multirun_task_overrides["a"] == multirun([7, 8])
    assert loader.to_xarray().sizes == {"a": 2}


@pytest.mark.usefixtures("cleandir")
def test_g5_dict_override_rejected():
    # G5: dict workflow_overrides were advertised but emitted invalid Hydra
    # syntax; dict support has been dropped, so it must be rejected.
    class Blank(MultiRunMetricsWorkflow):
        @staticmethod
        def task(x):
            return None

    wf = Blank()
    with pytest.raises(TypeError):
        wf.run(x={"a": 1})  # type: ignore


@pytest.mark.usefixtures("cleandir")
def test_g3_target_dir_overrides_with_equals_in_value():
    # G3: target_dir_multirun_overrides used o.split("=") (no maxsplit), which
    # crashed when an override value contained '='. The previous run's
    # overrides include a quoted value with '='; flattening must not crash.
    class First(MultiRunMetricsWorkflow):
        @staticmethod
        def task(tag, eps):
            tr.save(dict(eps=eps), "test_metrics.pt")
            return dict(eps=eps)

    class Second(MultiRunMetricsWorkflow):
        @staticmethod
        def task(job_dir):
            return None

    first = First()
    first.run(
        tag=multirun(["a=b", "c=d"]),
        eps=multirun([1.0, 2.0]),
        working_dir="first",
    )

    second = Second()
    second.run(target_job_dirs=first.multirun_working_dirs, working_dir="second")

    flattened = second.target_dir_multirun_overrides
    assert flattened["tag"] == ["a=b", "a=b", "c=d", "c=d"]
    assert flattened["eps"] == [1.0, 2.0, 1.0, 2.0]


@pytest.mark.usefixtures("cleandir")
def test_regression_68():
    # https://github.com/mit-ll-responsible-ai/responsible-ai-toolbox/pull/68
    class Blank(MultiRunMetricsWorkflow):
        @staticmethod
        def task():
            pass

    wf1 = Blank()
    wf1.run(
        list_vals=multirun([[0, 1], [2, 3]]),  # note: multi-run over list-values
        working_dir="first",
    )

    wf2 = Blank()
    wf2.run(
        target_job_dirs=wf1.multirun_working_dirs,
        val=multirun([1, 2]),
        working_dir="second",
    )

    xr1_coords = wf1.to_xarray().list_vals.data
    xr2_coords = wf2.to_xarray().list_vals.data
    assert np.all(xr1_coords == xr2_coords)


def test_to_override_element_handles_numpy_scalars():
    """NumPy/torch 0-d scalars (e.g. from np.arange or RobustnessCurve's epsilon
    array) are emitted unquoted so Hydra sweeps over numbers, not strings."""
    import numpy as np

    from mushin.workflows import _to_override_element

    assert _to_override_element(np.int64(3)) == "3"
    assert _to_override_element(np.float32(1.5)) == "1.5"
    assert _to_override_element(np.bool_(True)) == "true"
    # strings still quoted (commas/spaces preserved)
    assert _to_override_element("a,b") == "'a,b'"


@pytest.mark.parametrize(
    "cls", [BaseWorkflow, MultiRunMetricsWorkflow, RobustnessCurve]
)
@pytest.mark.parametrize("param", ["config_name", "job_name"])
def test_run_defaults_are_not_rai_branded(cls, param):
    default = inspect.signature(cls.run).parameters[param].default
    assert default == "mushin_workflow"


@pytest.mark.parametrize("param", ["resume", "capture_env"])
def test_robustnesscurve_run_forwards_resilience_params(param):
    # RobustnessCurve.run must expose (and forward) resume/capture_env, mirroring
    # on_error, so these resilience knobs are reachable on the public subclass.
    params = inspect.signature(RobustnessCurve.run).parameters
    assert param in params
    assert params[param].default is False


@pytest.mark.usefixtures("cleandir")
@pytest.mark.filterwarnings("ignore:invalid value encountered in cast")
def test_original_cwd_inside_real_hydra_run_returns_launch_dir(cleandir):
    """Integration: exercise the REAL (non-monkeypatched) Hydra branch of
    mushin.original_cwd(). Inside a Hydra job the process cwd is the per-job
    output dir, but original_cwd() must resolve back to the launch dir."""
    import os

    import mushin

    launch_dir = Path(cleandir).resolve()

    class CwdCapture(MultiRunMetricsWorkflow):
        @staticmethod
        def task(epsilon):
            captured = dict(
                per_job_cwd=os.getcwd(),
                original_cwd=str(mushin.original_cwd()),
            )
            tr.save(captured, "cwd_capture.pt")
            tr.save(dict(result=float(epsilon)), "test_metrics.pt")
            return dict(result=float(epsilon))

    wf = CwdCapture()
    wf.run(epsilon=multirun([0.0]))

    job_dir = Path(wf.multirun_working_dirs[0])
    captured = tr.load(job_dir / "cwd_capture.pt", weights_only=False)

    # Inside the Hydra job the process cwd is the per-job output dir...
    assert Path(captured["per_job_cwd"]).resolve() != launch_dir
    # ...but original_cwd() resolves back to where .run() was invoked.
    assert Path(captured["original_cwd"]).resolve() == launch_dir


@pytest.mark.usefixtures("cleandir")
def test_to_xarray_nan_fills_missing_combo(tmp_path):
    import numpy as np

    from mushin import multirun
    from mushin.workflows import MultiRunMetricsWorkflow

    class Holey(MultiRunMetricsWorkflow):
        @staticmethod
        def task(a, b):
            if a == 2 and b == 1:  # emulate a missing result for this cell
                return None
            return dict(val=float(a * 10 + b))

    wf = Holey()
    wf.run(a=multirun([1, 2]), b=multirun([0, 1]), working_dir=str(tmp_path / "s"))
    ds = wf.to_xarray()
    assert ds.sizes == {"a": 2, "b": 2}
    assert np.isnan(float(ds["val"].sel(a=2, b=1)))  # hole -> NaN
    assert float(ds["val"].sel(a=1, b=1)) == 11.0  # others intact


@pytest.mark.usefixtures("cleandir")
def test_singleton_dim_with_real_metrics_not_nan(tmp_path):
    # non_multirun_params_as_singleton_dims=True with a non-multirun scalar
    # param must NOT nan-fill the (present) metric values.
    class WF(MultiRunMetricsWorkflow):
        @staticmethod
        def task(a, b):
            return dict(val=float(a * 10 + b))

    wf = WF()
    wf.run(a=multirun([1, 2]), b=5, working_dir=str(tmp_path / "s"))
    ds = wf.to_xarray(non_multirun_params_as_singleton_dims=True)
    assert not np.isnan(ds["val"].data).any()
    assert_allclose(ds["val"].data, [[15.0], [25.0]])


@pytest.mark.usefixtures("cleandir")
def test_non_square_grid_no_transpose(tmp_path):
    # A non-square multi-param grid must not transpose cells.
    class WF(MultiRunMetricsWorkflow):
        @staticmethod
        def task(a, b):
            return dict(val=float(a * 100 + b))

    wf = WF()
    wf.run(
        a=multirun([1, 2]), b=multirun([10, 20, 30]), working_dir=str(tmp_path / "s")
    )
    ds = wf.to_xarray()
    assert ds.sizes == {"a": 2, "b": 3}
    assert float(ds["val"].sel(a=1, b=30)) == 130.0
    assert float(ds["val"].sel(a=2, b=10)) == 210.0
    assert float(ds["val"].sel(a=2, b=30)) == 230.0


@pytest.mark.usefixtures("cleandir")
def test_integer_metric_dtype_preserved(tmp_path):
    # A clean sweep of an integer-valued metric must keep an integer dtype.
    class WF(MultiRunMetricsWorkflow):
        @staticmethod
        def task(a, b):
            return dict(val=int(a * 10 + b))

    wf = WF()
    wf.run(a=multirun([1, 2]), b=multirun([0, 1]), working_dir=str(tmp_path / "s"))
    ds = wf.to_xarray()
    assert np.issubdtype(ds["val"].dtype, np.integer)


@pytest.mark.usefixtures("cleandir")
def test_run_writes_metrics_sidecar_per_job(tmp_path):
    from mushin import multirun
    from mushin._sweep_io import read_metrics_sidecar
    from mushin.workflows import MultiRunMetricsWorkflow

    class W(MultiRunMetricsWorkflow):
        @staticmethod
        def task(x):
            return dict(y=float(x) * 2)

    wf = W()
    wf.run(x=multirun([1, 2, 3]), working_dir=str(tmp_path / "s"))
    for d in wf.multirun_working_dirs:
        assert read_metrics_sidecar(d) is not None
        assert read_metrics_sidecar(d)["y"] is not None


def _grid_with_one_failure():
    from mushin.workflows import MultiRunMetricsWorkflow

    class W(MultiRunMetricsWorkflow):
        @staticmethod
        def task(a, b):
            if a == 2 and b == 1:
                raise RuntimeError("boom")
            return dict(val=float(a * 10 + b))

    return W


def test_on_error_raise_is_default(tmp_path):
    from mushin import multirun

    with pytest.raises(Exception):
        _grid_with_one_failure()().run(
            a=multirun([1, 2]), b=multirun([0, 1]), working_dir=str(tmp_path / "s")
        )


def test_on_error_nan_records_and_continues(tmp_path):
    import numpy as np

    from mushin import multirun

    wf = _grid_with_one_failure()()
    with pytest.warns(UserWarning, match="fail"):
        wf.run(
            a=multirun([1, 2]),
            b=multirun([0, 1]),
            working_dir=str(tmp_path / "s"),
            on_error="nan",
        )
    assert wf.is_complete is False
    assert any("a=2" in f["combo"] for f in wf.failures)
    ds = wf.to_xarray()
    assert np.isnan(float(ds["val"].sel(a=2, b=1)))
    assert float(ds["val"].sel(a=1, b=0)) == 10.0
    assert ds.attrs["mushin_failures"]  # non-empty list


def test_resume_reruns_only_failed_cell(tmp_path):
    from mushin import multirun
    from mushin.workflows import MultiRunMetricsWorkflow

    CALLS = {"n": 0}

    class W(MultiRunMetricsWorkflow):
        FAIL = True

        @staticmethod
        def task(a, b):
            CALLS["n"] += 1
            if a == 2 and b == 1 and W.FAIL:
                raise RuntimeError("boom")
            return dict(val=float(a * 10 + b))

    wd = str(tmp_path / "s")
    W.FAIL = True
    wf = W()
    wf.run(a=multirun([1, 2]), b=multirun([0, 1]), working_dir=wd, on_error="nan")
    W.FAIL = False
    CALLS["n"] = 0
    wf2 = W()
    wf2.run(a=multirun([1, 2]), b=multirun([0, 1]), working_dir=wd, resume=True)
    assert CALLS["n"] == 1  # only the previously-failed cell actually ran
    assert wf2.is_complete
    ds = wf2.to_xarray()
    assert float(ds["val"].sel(a=2, b=1)) == 21.0  # cell filled in place
    assert ds.sizes == {"a": 2, "b": 2}  # same shape, no growth


def test_resume_with_relative_working_dir_reruns_only_failed_cell(
    tmp_path, monkeypatch
):
    # Regression: a RELATIVE working_dir must still short-circuit completed cells
    # on resume. The short-circuit's sidecar read runs INSIDE the chdir'd Hydra
    # job, so an unresolved (relative) manifest root resolves against the job cwd
    # and finds no sidecar -> every completed cell silently re-executes. The main
    # process must resolve working_dir to absolute before wrapping.
    from mushin import multirun
    from mushin.workflows import MultiRunMetricsWorkflow

    monkeypatch.chdir(tmp_path)
    CALLS = {"n": 0}

    class W(MultiRunMetricsWorkflow):
        FAIL = True

        @staticmethod
        def task(a, b):
            CALLS["n"] += 1
            if a == 2 and b == 1 and W.FAIL:
                raise RuntimeError("boom")
            return dict(val=float(a * 10 + b))

    wd = "rel_sweep"  # RELATIVE to the (monkeypatched) cwd
    W.FAIL = True
    wf = W()
    with pytest.warns(UserWarning, match="fail"):
        wf.run(a=multirun([1, 2]), b=multirun([0, 1]), working_dir=wd, on_error="nan")

    W.FAIL = False
    CALLS["n"] = 0
    wf2 = W()
    wf2.run(a=multirun([1, 2]), b=multirun([0, 1]), working_dir=wd, resume=True)
    # Only the previously-failed cell re-ran; the 3 completed cells short-circuited.
    assert CALLS["n"] == 1
    assert wf2.is_complete
    ds = wf2.to_xarray()
    assert float(ds["val"].sel(a=2, b=1)) == 21.0


def test_narrowed_grid_reusing_dir_is_complete(tmp_path):
    # Regression: a fresh sweep that reuses a working_dir with a NARROWED grid
    # must not inherit stale failed cells from the prior sweep's manifest. Cells
    # absent from the current grid must be pruned so `is_complete` reflects only
    # the current grid.
    from mushin import multirun
    from mushin.workflows import MultiRunMetricsWorkflow

    class W(MultiRunMetricsWorkflow):
        FAIL_SEED = 1

        @staticmethod
        def task(seed):
            if seed == W.FAIL_SEED:
                raise RuntimeError("boom")
            return dict(val=float(seed))

    wd = str(tmp_path / "D")

    # pass 1: seed in {0,1,2}, seed=1 fails -> incomplete
    wf = W()
    with pytest.warns(UserWarning, match="fail"):
        wf.run(seed=multirun([0, 1, 2]), working_dir=wd, on_error="nan")
    assert wf.is_complete is False

    # pass 2: NEW sweep, SAME dir, narrowed grid {0,2}; all succeed -> complete
    wf2 = W()
    wf2.run(seed=multirun([0, 2]), working_dir=wd)
    assert wf2.is_complete is True


def test_run_sets_hydra_chdir_explicitly_and_silences_deprecation(tmp_path):
    # Regression: under version_base "1.1" Hydra emits a UserWarning that future
    # versions will no longer change the job working directory by default. The
    # workflow depends on the chdir behavior (each job reads/writes its metrics
    # sidecar in its own dir), so `run` sets `hydra.job.chdir=True` explicitly —
    # preserving behavior and silencing the warning (which fires only while the
    # setting is left implicit).
    import warnings

    class W(MultiRunMetricsWorkflow):
        @staticmethod
        def task(seed):
            return dict(val=float(seed))

    wf = W()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        wf.run(seed=multirun([0, 1]), working_dir=str(tmp_path / "d"))

    chdir_warnings = [
        w
        for w in caught
        if "no longer change working directory" in str(w.message)
    ]
    assert not chdir_warnings, [str(w.message) for w in chdir_warnings]
    # behavior preserved: the sweep still assembles a labeled dataset
    assert wf.to_xarray().sizes == {"seed": 2}


def test_run_does_not_duplicate_a_caller_hydra_chdir_override(tmp_path):
    # A caller who passes their own `hydra.job.chdir` override wins — `run` must
    # not append a second, conflicting one. (Use `=True`, which matches the
    # workflow's required behavior, so the sweep still completes normally.)
    class W(MultiRunMetricsWorkflow):
        @staticmethod
        def task(seed):
            return dict(val=float(seed))

    captured = {}

    import mushin.workflows as _wf_mod

    orig_launch = _wf_mod.launch

    def _spy_launch(*args, **kwargs):
        captured["overrides"] = list(kwargs.get("overrides", []))
        return orig_launch(*args, **kwargs)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(_wf_mod, "launch", _spy_launch)
        W().run(
            seed=multirun([0, 1]),
            working_dir=str(tmp_path / "d"),
            overrides=["hydra.job.chdir=True"],
        )

    chdir_overrides = [
        o for o in captured["overrides"] if o.split("=", 1)[0] == "hydra.job.chdir"
    ]
    assert chdir_overrides == ["hydra.job.chdir=True"]  # exactly one, not duplicated
