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
    # A sweep axis cannot repeat a value (duplicates now raise); keep epsilon
    # values distinct so the generated curve has one cell per epsilon.
    unique=True,
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
        # An extra sweep axis is declared explicitly with multirun (a bare
        # string like "1,2" is now a fixed literal value, not an auto-sweep).
        task.run(epsilon=[0, 1, 2, 3], fake_param=multirun([1, 2]))
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
        # Sweep axes reject duplicate values; keep the multirun elements distinct.
        unique_by=repr,
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

    # The failing cell raises RuntimeError("boom"); the default on_error="raise"
    # must propagate exactly that (not merely "some exception", which would pass
    # even on an unrelated API/signature regression).
    with pytest.raises(RuntimeError, match="boom"):
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


@pytest.mark.parametrize("relative", [False, True])
def test_resume_reruns_only_failed_cell(tmp_path, monkeypatch, relative):
    # Resume must re-run ONLY the previously-failed cell and short-circuit the
    # completed ones. `relative=True` is a regression guard: a RELATIVE working_dir
    # must still short-circuit — the short-circuit's sidecar read runs INSIDE the
    # chdir'd Hydra job, so the main process must resolve working_dir to absolute
    # before wrapping, else every completed cell silently re-executes.
    from mushin import multirun
    from mushin.workflows import MultiRunMetricsWorkflow

    if relative:
        monkeypatch.chdir(tmp_path)
        wd = "rel_sweep"  # RELATIVE to the (monkeypatched) cwd
    else:
        wd = str(tmp_path / "s")

    CALLS = {"n": 0}

    class W(MultiRunMetricsWorkflow):
        FAIL = True

        @staticmethod
        def task(a, b):
            CALLS["n"] += 1
            if a == 2 and b == 1 and W.FAIL:
                raise RuntimeError("boom")
            return dict(val=float(a * 10 + b))

    W.FAIL = True
    wf = W()
    # fail-soft emits a UserWarning naming the failed cell — assert it (this also
    # keeps it out of the pytest warnings summary).
    with pytest.warns(UserWarning, match="fail"):
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
        w for w in caught if "no longer change working directory" in str(w.message)
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


def test_cell_status_sidecar_written_completed(tmp_path):
    from mushin._resume import read_cell_status

    class W(MultiRunMetricsWorkflow):
        @staticmethod
        def task(seed):
            return dict(val=float(seed))

    wd = str(tmp_path / "s")
    W().run(seed=multirun([0, 1]), working_dir=wd)
    statuses = [
        read_cell_status(d)
        for d in Path(wd).iterdir()
        if d.is_dir() and (d / "mushin_cell_status.json").exists()
    ]
    assert statuses and all(s["status"] == "completed" for s in statuses)


def test_cell_status_sidecar_written_failed_under_fail_soft(tmp_path):
    from mushin._resume import read_cell_status

    class W(MultiRunMetricsWorkflow):
        @staticmethod
        def task(seed):
            if seed == 1:
                raise RuntimeError("boom")
            return dict(val=float(seed))

    wd = str(tmp_path / "s")
    with pytest.warns(UserWarning, match="fail"):
        W().run(seed=multirun([0, 1]), working_dir=wd, on_error="nan")
    got = {
        read_cell_status(d)["combo"]["seed"]: read_cell_status(d)["status"]
        for d in Path(wd).iterdir()
        if d.is_dir() and (d / "mushin_cell_status.json").exists()
    }
    assert got == {0: "completed", 1: "failed"}


def test_cell_status_combo_for_override_string_multirun(tmp_path):
    # A multirun supplied via `overrides=[...]` (not the multirun() kwarg) must
    # still record distinct, correctly-projected combos in the status sidecars —
    # no crash, no collapse to an empty combo.
    from mushin._resume import read_cell_status

    class W(MultiRunMetricsWorkflow):
        @staticmethod
        def task(seed):
            return dict(val=float(seed))

    wd = str(tmp_path / "s")
    W().run(working_dir=wd, overrides=["+seed=0,1,2"])
    combos = [
        read_cell_status(d)["combo"]
        for d in Path(wd).iterdir()
        if d.is_dir() and (d / "mushin_cell_status.json").exists()
    ]
    assert sorted(c["seed"] for c in combos) == [0, 1, 2]


def test_task_receives_resume_context_on_reexecution(tmp_path):
    seen = {}

    class W(MultiRunMetricsWorkflow):
        FAIL = True

        @staticmethod
        def task(seed, mushin_resume=None):
            seen[seed] = mushin_resume
            if mushin_resume is not None and mushin_resume.dir is not None:
                (mushin_resume.dir / "last.ckpt").write_text("state")
            if seed == 0 and W.FAIL:
                raise RuntimeError("boom")
            return dict(val=float(seed))

    wd = str(tmp_path / "s")
    W.FAIL = True
    with pytest.warns(UserWarning, match="fail"):
        W().run(seed=multirun([0, 1]), working_dir=wd, on_error="nan")
    assert seen[0] is not None and seen[0].is_resume is False and seen[0].attempt == 1

    W.FAIL = False
    seen.clear()
    wf = W()
    wf.run(seed=multirun([0, 1]), working_dir=wd, resume=True)
    assert 1 not in seen  # seed 1 completed -> short-circuited (task not called)
    rc = seen[0]
    assert rc.is_resume is True
    assert rc.attempt == 2
    assert rc.last_ckpt is not None and rc.last_ckpt.name == "last.ckpt"
    assert wf.is_complete


def test_task_without_mushin_resume_param_is_unaffected(tmp_path):
    # Introspection gate: a task NOT declaring mushin_resume is called as today,
    # with no Hydra/zen config error.
    class W(MultiRunMetricsWorkflow):
        @staticmethod
        def task(seed):
            return dict(val=float(seed))

    wf = W()
    wf.run(seed=multirun([0, 1]), working_dir=str(tmp_path / "s"))
    assert wf.to_xarray().sizes == {"seed": 2}


def _task_with_resume(seed, mushin_resume=None):
    return {"v": float(seed), "got": mushin_resume}


def _plain_task(seed):
    return {"v": float(seed)}


def test_resume_injector_is_picklable_and_hides_param():
    import inspect
    import pickle

    from hydra_zen import zen

    from mushin.workflows import _prepare_task, _ResumeInjector

    prepared, wants = _prepare_task(_task_with_resume)
    assert wants is True
    assert isinstance(prepared, _ResumeInjector)
    # hidden from the signature zen inspects:
    assert list(inspect.signature(prepared).parameters) == ["seed"]
    # picklable, and zen(prepared) picklable:
    pickle.loads(pickle.dumps(prepared))
    pickle.loads(pickle.dumps(zen(prepared)))

    # a task WITHOUT mushin_resume is returned unchanged:
    prepared2, wants2 = _prepare_task(_plain_task)
    assert wants2 is False and prepared2 is _plain_task


class _PicklableRunnerWF(MultiRunMetricsWorkflow):
    @staticmethod
    def task(seed):
        return dict(v=float(seed))


def test_task_runner_is_picklable(tmp_path):
    import pickle

    import mushin.workflows as wf_mod

    captured = {}
    orig = wf_mod.launch

    def capture(cfg, task_call, **k):  # launch(cfg, task_call, **kwargs)
        captured["task_call"] = task_call
        return orig(cfg, task_call, **k)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(wf_mod, "launch", capture)
        _PicklableRunnerWF().run(
            seed=multirun([0, 1]), working_dir=str(tmp_path / "s"), on_error="nan"
        )

    tc = captured["task_call"]
    assert isinstance(tc, wf_mod._TaskRunner)
    pickle.loads(pickle.dumps(tc))  # the whole dispatch object is picklable


def test_bare_list_sweep_arg_gives_actionable_multirun_error(tmp_path):
    # A new user's most likely slip: forgetting multirun() and passing a bare list.
    # The error must point at the fix (multirun) and the escape hatch (hydra_list),
    # not just list the internal accepted types.
    class W(MultiRunMetricsWorkflow):
        @staticmethod
        def task(lr):
            return dict(v=float(lr))

    with pytest.raises(TypeError, match=r"mushin\.multirun\("):
        W().run(lr=[0.01, 0.1, 1.0])
    with pytest.raises(TypeError, match=r"mushin\.hydra_list\("):
        W().run(lr=[0.01, 0.1, 1.0])
    # a legitimate multirun still works (must NOT be caught by the bare-list
    # guard); use tmp_path so the sweep doesn't scatter multirun/ into the cwd.
    W().run(lr=multirun([0.01, 0.1]), working_dir=str(tmp_path / "s"))


def test_failures_attr_survives_netcdf_roundtrip(tmp_path):
    # attrs["mushin_failures"] must be netCDF-portable: a raw list of strings
    # collapses to a scalar str for 1-element lists (netCDF4) or fails to write
    # (scipy engine), so it is stored as a JSON string like `provenance`.
    import json

    from mushin import multirun

    wf = _grid_with_one_failure()()
    with pytest.warns(UserWarning, match="fail"):
        wf.run(
            a=multirun([1, 2]),
            b=multirun([0, 1]),
            working_dir=str(tmp_path / "s"),
            on_error="nan",
        )
    ds = wf.to_xarray()
    combos = json.loads(ds.attrs["mushin_failures"])
    assert any("a=2" in c for c in combos)

    p = tmp_path / "roundtrip.nc"
    ds.to_netcdf(p)
    back = xr.load_dataset(p)
    assert json.loads(back.attrs["mushin_failures"]) == combos


def test_on_error_nan_preserves_traceback_on_disk(tmp_path):
    """on_error='nan' must not discard the failure's stack trace: the repr in
    wf.failures is one line; the full traceback is written into the cell dir."""
    from mushin import multirun

    wf = _grid_with_one_failure()()
    with pytest.warns(UserWarning, match="fail"):
        wf.run(
            a=multirun([1, 2]),
            b=multirun([0, 1]),
            working_dir=str(tmp_path / "s"),
            on_error="nan",
        )
    (failure,) = wf.failures
    err_file = Path(failure["working_dir"]) / "mushin_error.txt"
    assert err_file.exists()
    text = err_file.read_text()
    assert "RuntimeError" in text and "boom" in text and "Traceback" in text


def test_dotted_override_sweep_axis(tmp_path):
    """A nested (dotted) override like model.width=4,8 must be usable as a
    sweep axis: combo extraction reads it as a config *path*, not a literal
    key, and it becomes a normal xarray dimension."""
    from mushin import multirun
    from mushin.workflows import MultiRunMetricsWorkflow

    class W(MultiRunMetricsWorkflow):
        @staticmethod
        def task(model={"width": 4}):  # noqa: B006 - config template, not mutated
            return dict(w=float(model["width"]))

    wf = W()
    wf.run(
        working_dir=str(tmp_path / "s"),
        **{"model.width": multirun([4, 8])},
    )
    ds = wf.to_xarray()
    assert sorted(ds["model.width"].values.tolist()) == [4, 8]
    assert float(ds["w"].sel({"model.width": 8})) == 8.0


@pytest.mark.config_group("model")
def test_config_group_sweep_coords_are_choice_names(tmp_path, restore_config_group):
    """Sweeping a Hydra config GROUP (model=small,big) must key the xarray
    dimension by the chosen option name, not a stringified sub-config."""
    from hydra.core.config_store import ConfigStore
    from hydra_zen import make_config

    from mushin import multirun
    from mushin.workflows import MultiRunMetricsWorkflow

    cs = ConfigStore.instance()
    cs.store(group="model", name="small", node={"width": 4})
    cs.store(group="model", name="big", node={"width": 8})

    class W(MultiRunMetricsWorkflow):
        @staticmethod
        def task(model=None):
            return dict(w=float(model["width"]))

    # The group's target field must exist on the config for Hydra to compose
    # into it; declaring it None is the supported pattern.
    wf = W(make_config(model=None))
    wf.run(working_dir=str(tmp_path / "s"), model=multirun(["small", "big"]))
    ds = wf.to_xarray()
    assert sorted(str(v) for v in ds["model"].values) == ["big", "small"]
    assert float(ds["w"].sel(model="big")) == 8.0


def test_parse_overrides_expands_range_and_rejects_interval():
    """Hydra range(...) sweeps form a grid and are expanded; continuous
    interval(...) (Bayesian-sweeper syntax) cannot and must raise a clear
    error instead of AttributeError."""
    from mushin import multirun
    from mushin.workflows import MultiRunMetricsWorkflow

    out = MultiRunMetricsWorkflow._parse_overrides(["x=range(1,5)"])
    assert isinstance(out["x"], multirun) and list(out["x"]) == [1, 2, 3, 4]

    with pytest.raises(ValueError, match="interval"):
        MultiRunMetricsWorkflow._parse_overrides(["x=interval(0,1)"])


def test_resume_reruns_completed_cells_when_config_changed(tmp_path):
    """A cached cell is only reused if the resolved config that produced it
    matches the current one -- changing a NON-swept value and resuming must
    re-run every cell, not silently mix results from two configurations."""
    from mushin import multirun
    from mushin.workflows import MultiRunMetricsWorkflow

    CALLS = {"n": 0}

    class W(MultiRunMetricsWorkflow):
        @staticmethod
        def task(a, scale=1.0):
            CALLS["n"] += 1
            return dict(val=float(a) * scale)

    wd = str(tmp_path / "s")
    W().run(a=multirun([1, 2]), scale=1.0, working_dir=wd)
    assert CALLS["n"] == 2

    CALLS["n"] = 0
    wf2 = W()
    with pytest.warns(UserWarning, match="config"):
        wf2.run(a=multirun([1, 2]), scale=2.0, working_dir=wd, resume=True)
    assert CALLS["n"] == 2  # both cells re-ran under the new config
    ds = wf2.to_xarray()
    assert float(ds["val"].sel(a=2)) == 4.0  # new config's result, not stale


def test_resume_reuses_legacy_sidecars_without_config_hash(tmp_path):
    """Sweeps recorded before the config-hash existed must still resume
    (reuse) their completed cells."""
    import json as _json

    from mushin import multirun
    from mushin._resume import STATUS_FILE
    from mushin.workflows import MultiRunMetricsWorkflow

    CALLS = {"n": 0}

    class W(MultiRunMetricsWorkflow):
        @staticmethod
        def task(a):
            CALLS["n"] += 1
            return dict(val=float(a))

    wd = tmp_path / "s"
    W().run(a=multirun([1, 2]), working_dir=str(wd))
    # Simulate a pre-hash sweep: strip config_hash from every status sidecar.
    for status_file in wd.glob(f"*/{STATUS_FILE}"):
        d = _json.loads(status_file.read_text())
        d.pop("config_hash", None)
        status_file.write_text(_json.dumps(d))

    CALLS["n"] = 0
    W().run(a=multirun([1, 2]), working_dir=str(wd), resume=True)
    assert CALLS["n"] == 0  # legacy cells reused, nothing re-ran


def test_task_runner_ships_only_completed_cells(tmp_path):
    """Out-of-process launchers pickle the runner to every worker; it must
    carry only the completed {combo_key: dir} map, not the full manifest
    (O(N^2) serialized bytes across N cells)."""
    from mushin import multirun
    from mushin._sweep_io import Manifest
    from mushin.workflows import _PriorCells

    class W(_grid_with_one_failure()):
        pass

    wd = str(tmp_path / "s")
    with pytest.warns(UserWarning, match="fail"):
        W().run(a=multirun([1, 2]), b=multirun([0, 1]), working_dir=wd, on_error="nan")

    m = Manifest.from_cell_status(Path(wd), ["a", "b"])
    prior = _PriorCells.from_manifest(m)
    assert len(prior.completed) == 3  # the failed cell is not shipped
    assert all("a=" in k for k in prior.completed)


def test_dataset_netcdf_roundtrip_with_string_coords(tmp_path):
    """The headline share-your-results flow: a sweep dataset with a string
    sweep axis must survive to_netcdf -> load_dataset identically (values,
    coords, and attrs)."""
    from mushin import multirun
    from mushin.workflows import MultiRunMetricsWorkflow

    class W(MultiRunMetricsWorkflow):
        @staticmethod
        def task(opt: str, lr: float):
            return dict(score=float(len(opt)) * lr)

    wf = W()
    wf.run(
        opt=multirun(["adam", "sgd"]),
        lr=multirun([0.1, 0.2]),
        working_dir=str(tmp_path / "s"),
    )
    ds = wf.to_xarray()
    p = tmp_path / "r.nc"
    ds.to_netcdf(p)
    back = xr.load_dataset(p)
    xr.testing.assert_identical(ds, back)
    assert float(back["score"].sel(opt="adam", lr=0.2)) == pytest.approx(0.8)


def test_to_dataframe_tidy_view(tmp_path):
    """The pandas exit ramp: one call, one row per sweep cell, sweep params
    and metrics as plain columns -- no xarray knowledge required."""
    from mushin import multirun
    from mushin.workflows import MultiRunMetricsWorkflow

    class W(MultiRunMetricsWorkflow):
        @staticmethod
        def task(a, b):
            return dict(val=float(a * 10 + b))

    wf = W()
    wf.run(a=multirun([1, 2]), b=multirun([0, 1]), working_dir=str(tmp_path / "s"))
    df = wf.to_dataframe()
    assert {"a", "b", "val"} <= set(df.columns)
    assert len(df) == 4
    assert float(df[(df.a == 2) & (df.b == 1)]["val"].iloc[0]) == 21.0
    # kwargs forward to to_xarray
    df2 = wf.to_dataframe(include_working_subdirs_as_data_var=True)
    assert "working_subdir" in df2.columns


def test_resume_reruns_when_task_code_changes(tmp_path):
    # The stale-code-reuse bug: resume=True with an edited task body but an
    # UNCHANGED config must re-run the cell (new code, new result), not return
    # the previous run's cached metrics.
    from mushin import multirun
    from mushin.workflows import MultiRunMetricsWorkflow

    class V1(MultiRunMetricsWorkflow):
        @staticmethod
        def task(a):
            return dict(v=float(a) * 10)

    class V2(MultiRunMetricsWorkflow):
        @staticmethod
        def task(a):
            return dict(v=float(a) * 100)  # edited body, same signature/config

    wd = str(tmp_path / "s")
    V1().run(a=multirun([1, 2]), working_dir=wd)
    wf2 = V2()
    with pytest.warns(UserWarning, match="task code"):
        wf2.run(a=multirun([1, 2]), working_dir=wd, resume=True)
    ds = wf2.to_xarray()
    assert float(ds["v"].sel(a=2)) == 200.0  # V2 ran; not the stale 20.0


def test_resume_reuses_when_task_code_unchanged(tmp_path):
    # Same code + same config -> reuse (no re-run, no warning).
    import warnings

    from mushin import multirun
    from mushin.workflows import MultiRunMetricsWorkflow

    CALLS = {"n": 0}

    class W(MultiRunMetricsWorkflow):
        @staticmethod
        def task(a):
            CALLS["n"] += 1
            return dict(v=float(a))

    wd = str(tmp_path / "s")
    W().run(a=multirun([1, 2]), working_dir=wd)
    CALLS["n"] = 0
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any resume warning would fail here
        W().run(a=multirun([1, 2]), working_dir=wd, resume=True)
    assert CALLS["n"] == 0  # both cells reused


def test_resume_reuses_legacy_sidecars_without_code_hash(tmp_path):
    # Sweeps recorded before the code-hash existed must still resume (reuse)
    # their completed cells rather than spuriously re-running.
    import json

    from mushin import multirun
    from mushin._resume import STATUS_FILE
    from mushin.workflows import MultiRunMetricsWorkflow

    CALLS = {"n": 0}

    class W(MultiRunMetricsWorkflow):
        @staticmethod
        def task(a):
            CALLS["n"] += 1
            return dict(v=float(a))

    wd = tmp_path / "s"
    W().run(a=multirun([1, 2]), working_dir=str(wd))
    for sf in wd.glob(f"*/{STATUS_FILE}"):
        d = json.loads(sf.read_text())
        d.pop("code_hash", None)  # simulate a pre-code-hash sweep
        sf.write_text(json.dumps(d))

    CALLS["n"] = 0
    W().run(a=multirun([1, 2]), working_dir=str(wd), resume=True)
    assert CALLS["n"] == 0  # legacy cells reused


def test_resume_reruns_full_grid_when_axis_added(tmp_path):
    # Adding a sweep axis changes the grid shape. A resume must NOT project the
    # new cells onto the old params and reuse the wrong cell (which silently
    # dropped the new axis and marked the sweep complete); it re-runs the full
    # new grid and warns.
    import json

    from mushin import multirun
    from mushin._resume import STATUS_FILE
    from mushin.workflows import MultiRunMetricsWorkflow

    CALLS = {"n": 0}

    class W(MultiRunMetricsWorkflow):
        @staticmethod
        def task(a, b=0):
            CALLS["n"] += 1
            return dict(v=float(a) * 100 + float(b))

    wd = tmp_path / "s"
    W().run(a=multirun([1, 2]), working_dir=str(wd))
    # simulate a legacy (pre-fingerprint) sweep: strip the hashes so the config
    # guard can't catch the shape change -- the grid-shape guard must.
    for sf in wd.glob(f"*/{STATUS_FILE}"):
        d = json.loads(sf.read_text())
        d.pop("config_hash", None)
        d.pop("code_hash", None)
        sf.write_text(json.dumps(d))

    CALLS["n"] = 0
    wf2 = W()
    with pytest.warns(UserWarning, match="grid changed"):
        wf2.run(
            a=multirun([1, 2]), b=multirun([10, 20]), working_dir=str(wd), resume=True
        )
    assert CALLS["n"] == 4  # the full 2x2 grid re-ran (nothing wrongly reused)
    ds = wf2.to_xarray()
    assert set(ds.dims) == {"a", "b"}
    assert float(ds["v"].sel(a=1, b=20)) == 120.0
    assert float(ds["v"].sel(a=2, b=10)) == 210.0
    assert wf2.is_complete


def test_resume_dir_reuse_keeps_metrics_consistent_with_config(tmp_path):
    # gridevo case4: a resume that reuses a numeric job dir a prior sweep filled
    # with a DIFFERENT cell must refresh that dir's metrics so config and metrics
    # stay in sync -- otherwise the manifest and an offline load_from_dir mis-key
    # the reused cell (e.g. a=3 read as the stale a=1 value).
    import json

    from mushin import multirun
    from mushin.workflows import MultiRunMetricsWorkflow

    class W(MultiRunMetricsWorkflow):
        @staticmethod
        def metric_load_fn(p):  # the sidecar is JSON, not a torch pickle
            with open(p) as f:
                return json.load(f)

        @staticmethod
        def task(a):
            return dict(v=float(a))

    wd = tmp_path / "s"
    W().run(a=multirun([1, 2, 3]), working_dir=str(wd))
    # resume with a shifted grid: a=3 is reused, a=5/a=6 run into reused dirs
    W().run(a=multirun([3, 5, 6]), working_dir=str(wd), resume=True)

    # every job dir's metrics must match the config Hydra wrote into it
    for d in sorted(p for p in wd.glob("[0-9]*") if p.is_dir()):
        overrides = (d / ".hydra" / "overrides.yaml").read_text()
        a_val = float(overrides.split("a=")[1].split()[0])
        v = json.loads((d / "mushin_metrics.json").read_text())["v"]
        assert v == a_val, f"{d}: config a={a_val} but metrics v={v}"

    # offline reload reads the correct per-cell value (the reused a=3 -> 3.0)
    wf = W()
    wf.load_from_dir(str(wd), "mushin_metrics.json")
    ds = wf.to_xarray()
    assert float(ds["v"].sel(a=3)) == 3.0
    assert float(ds["v"].sel(a=5)) == 5.0
    assert float(ds["v"].sel(a=6)) == 6.0


def test_default_metric_load_fn_reads_json_sidecar(tmp_path):
    # A task returning a dict writes a JSON sidecar (mushin_metrics.json). The
    # DEFAULT metric_load_fn must read it -- previously it was torch.load, which
    # crashed on JSON, so offline reload of a @mushin.sweep/decorator sweep was
    # broken unless the user overrode metric_load_fn.
    from mushin import multirun
    from mushin.workflows import MultiRunMetricsWorkflow

    class W(MultiRunMetricsWorkflow):
        @staticmethod
        def task(a):
            return dict(v=float(a))

    wd = tmp_path / "s"
    W().run(a=multirun([1, 2, 3]), working_dir=str(wd))

    wf = MultiRunMetricsWorkflow()  # fresh instance, DEFAULT loader
    wf.load_from_dir(str(wd), "mushin_metrics.json")
    ds = wf.to_xarray()
    assert float(ds["v"].sel(a=2)) == 2.0
    assert set(ds["v"].sizes) == {"a"}


def test_default_metric_load_fn_still_reads_torch_files(tmp_path):
    # The torch.save path (MetricsCallback .pt files) must still load.
    import torch

    from mushin.workflows import MultiRunMetricsWorkflow

    p = tmp_path / "fit_metrics.pt"
    torch.save({"loss": [0.5, 0.4]}, p)
    assert MultiRunMetricsWorkflow.metric_load_fn(p) == {"loss": [0.5, 0.4]}
