# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT
import inspect
import string
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import hypothesis.strategies as st
import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch as tr
import xarray as xr
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


@pytest.mark.usefixtures("cleandir")
def test_return_not_list_jobreturn():
    # `hydra/sweeper=local_test` is the plugin in tests/plugins/hydra_plugins/
    # mushin_test_sweeper: Hydra only admits sweepers defined under
    # `hydra_plugins`, and hydra-zen's launch is moving onto Hydra's registry.
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
def test_rerun_same_instance_resets_cached_overrides():
    # A second run() on the SAME workflow instance must not reuse the first
    # run's cached overrides: the grid (coords, cell keys) must come from the
    # new sweep, not silently NaN-fill against the old one.
    class WF(MultiRunMetricsWorkflow):
        @staticmethod
        def task(a):
            return dict(m=float(a) * 10)

    wf = WF()
    wf.run(a=multirun([1.0, 2.0]), working_dir="dir_a")
    ds1 = wf.to_xarray()  # populates the override cache for dir_a
    assert ds1["m"].values.tolist() == [10.0, 20.0]

    wf.run(a=multirun([3.0, 4.0]), working_dir="dir_b")
    ds2 = wf.to_xarray()
    assert ds2.coords["a"].values.tolist() == [3.0, 4.0]
    assert ds2["m"].values.tolist() == [30.0, 40.0]


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


def test_run_defaults_pin_chdir_and_version_base(tmp_path):
    # The workflow depends on Hydra's per-job chdir (each job reads/writes its
    # metrics sidecar in its own dir), which is OFF by default under
    # version_base >= 1.2 — so `run` must pin `hydra.job.chdir=True` explicitly
    # and default version_base to "1.3" (the 1.1 compat mode is deprecated and
    # removed in Hydra 1.4). Assert the mechanism at the launch boundary:
    # reverting either default must fail here.
    class W(MultiRunMetricsWorkflow):
        @staticmethod
        def task(seed):
            return dict(val=float(seed))

    captured = {}

    import mushin.workflows as _wf_mod

    orig_launch = _wf_mod.launch

    def _spy_launch(*args, **kwargs):
        captured["overrides"] = list(kwargs.get("overrides", []))
        captured["version_base"] = kwargs.get("version_base")
        return orig_launch(*args, **kwargs)

    wf = W()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(_wf_mod, "launch", _spy_launch)
        wf.run(seed=multirun([0, 1]), working_dir=str(tmp_path / "d"))

    assert captured["version_base"] == "1.3"
    assert "hydra.job.chdir=True" in captured["overrides"]
    # behavior preserved: the sweep still assembles a labeled dataset
    assert wf.to_xarray().sizes == {"seed": 2}


def test_run_forwards_an_explicit_legacy_version_base(tmp_path):
    # Until Hydra 1.4 removes the 1.1 compat mode, a caller may still pass
    # version_base="1.1" explicitly — it must be forwarded verbatim and the
    # sweep must complete. (hydra-core >= 1.3.5 emits a Hydra14MigrationWarning
    # for it; that is expected and the caller's concern, not an error.)
    import warnings

    class W(MultiRunMetricsWorkflow):
        @staticmethod
        def task(seed):
            return dict(val=float(seed))

    captured = {}

    import mushin.workflows as _wf_mod

    orig_launch = _wf_mod.launch

    def _spy_launch(*args, **kwargs):
        captured["version_base"] = kwargs.get("version_base")
        return orig_launch(*args, **kwargs)

    wf = W()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(_wf_mod, "launch", _spy_launch)
        with warnings.catch_warnings():
            # hydra-core >= 1.3.5 warns on 1.1 compat; expected, not under test
            warnings.simplefilter("ignore")
            wf.run(
                seed=multirun([0, 1]),
                working_dir=str(tmp_path / "d"),
                version_base="1.1",
            )

    assert captured["version_base"] == "1.1"
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


@pytest.mark.config_group("model")
def test_config_group_sweep_survives_offline_reload(tmp_path, restore_config_group):
    """A finished config-GROUP sweep must reload from disk with real values:
    load_from_dir must recover each cell's runtime choice from its hydra.yaml
    so combo keys match the choice-name coordinates (not NaN-fill the grid)."""
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

    wf = W(make_config(model=None))
    wf.run(working_dir=str(tmp_path / "s"), model=multirun(["small", "big"]))

    loader = W(make_config(model=None)).load_from_dir(
        tmp_path / "s", "mushin_metrics.json"
    )
    ds = loader.to_xarray()
    assert sorted(str(v) for v in ds["model"].values) == ["big", "small"]
    assert float(ds["w"].sel(model="big")) == 8.0
    assert float(ds["w"].sel(model="small")) == 4.0


@pytest.mark.config_group("model")
def test_config_group_cell_records_the_choice_name(tmp_path, restore_config_group):
    """Each cell records its config-GROUP coordinate as the chosen option
    name, as Hydra chose it in that job, so the offline views (built from the
    cell records, not from hydra.yaml) show `small`/`big`, not a sub-config."""
    from hydra.core.config_store import ConfigStore
    from hydra_zen import make_config

    import mushin
    from mushin import multirun
    from mushin.workflows import MultiRunMetricsWorkflow

    cs = ConfigStore.instance()
    cs.store(group="model", name="small", node={"width": 4})
    cs.store(group="model", name="big", node={"width": 8})

    class W(MultiRunMetricsWorkflow):
        @staticmethod
        def task(model=None):
            return dict(w=float(model["width"]))

    wd = tmp_path / "s"
    W(make_config(model=None)).run(
        working_dir=str(wd), model=multirun(["small", "big"])
    )
    rows = {r["model"]: r["w"] for r in mushin.show(wd).rows}
    assert rows == {"small": 4.0, "big": 8.0}


def test_a_subclass_sanitizer_keys_cells_and_grid_alike(tmp_path):
    """A workflow that overrides `_sanitize_coordinate_for_xarray` keys each
    cell's result with it, as it keys the grid, in-process and reloaded."""
    from mushin import multirun
    from mushin.workflows import MultiRunMetricsWorkflow

    class W(MultiRunMetricsWorkflow):
        @staticmethod
        def task(a):
            return dict(v=float(a))

        @staticmethod
        def _sanitize_coordinate_for_xarray(value):
            if isinstance(value, multirun):
                return [f"x{v}" for v in value]
            return f"x{value}"

    wd = tmp_path / "s"
    wf = W()
    wf.run(a=multirun([1, 2]), working_dir=str(wd))
    loaded = W().load_from_dir(wd, "mushin_metrics.json")
    for w in (wf, loaded):
        ds = w.to_xarray()
        assert list(ds["a"].values) == ["x1", "x2"]
        assert ds["v"].values.tolist() == [1.0, 2.0]
    assert loaded.failures == [] and loaded.skipped == []


def test_load_from_dir_without_job_dirs_says_so(tmp_path):
    """A sweep killed before its first job has a multirun.yaml and no cells;
    the reload names that instead of failing to unpack an empty sort."""
    from omegaconf import OmegaConf

    from mushin.workflows import MultiRunMetricsWorkflow

    OmegaConf.save({"hydra": {"output_subdir": ".hydra"}}, tmp_path / "multirun.yaml")
    with pytest.raises(FileNotFoundError, match="no job directories"):
        MultiRunMetricsWorkflow().load_from_dir(tmp_path, "mushin_metrics.json")


def test_load_from_dir_names_a_null_output_subdir(tmp_path):
    """With hydra.output_subdir=null, no job keeps a record to reload from;
    the error says so instead of blaming a killed sweep."""
    from omegaconf import OmegaConf

    from mushin.workflows import MultiRunMetricsWorkflow

    OmegaConf.save({"hydra": {"output_subdir": None}}, tmp_path / "multirun.yaml")
    with pytest.raises(FileNotFoundError, match="output_subdir=null"):
        MultiRunMetricsWorkflow().load_from_dir(tmp_path, "mushin_metrics.json")


def test_load_from_dir_that_fails_partway_reloads_on_retry(tmp_path):
    """A cell missing its config.yaml fails the load before any of the job
    directories or configs are kept. On a workflow that held another sweep,
    a later to_xarray then reloads (and fails again, until the file is back)
    instead of reading the earlier sweep's cells under the new directory."""
    from mushin import multirun
    from mushin.workflows import MultiRunMetricsWorkflow

    class W(MultiRunMetricsWorkflow):
        @staticmethod
        def task(a):
            return dict(m=float(a))

    W().run(a=multirun([1, 2, 3]), working_dir=str(tmp_path / "s"))
    cfg = tmp_path / "s" / "2" / ".hydra" / "config.yaml"
    kept = cfg.read_text()
    cfg.unlink()
    # Prior state from a load, not a run: an in-process run's jobs outlive a
    # reload (as on main), which is a separate issue.
    W().run(a=multirun([7, 8]), working_dir=str(tmp_path / "other"))
    wf = W().load_from_dir(tmp_path / "other", "mushin_metrics.json")
    with pytest.raises(FileNotFoundError, match="config.yaml not found"):
        wf.load_from_dir(tmp_path / "s", "mushin_metrics.json")
    assert wf.multirun_working_dirs is None
    with pytest.raises(FileNotFoundError, match="config.yaml not found"):
        wf.to_xarray(metrics_filename="mushin_metrics.json")

    cfg.write_text(kept)
    ds = wf.to_xarray(metrics_filename="mushin_metrics.json")
    assert sorted(float(v) for v in ds["m"].values) == [1.0, 2.0, 3.0]


def test_load_from_dir_reports_a_broken_file_before_a_bad_field(tmp_path):
    """Every job's hydra.yaml is loaded before any field is read, as with the
    full OmegaConf load, so a sweep holding both a file that does not parse
    and a `???` job number reports the parse error, in either order."""
    from omegaconf import OmegaConf

    from mushin import multirun
    from mushin.workflows import MultiRunMetricsWorkflow

    class W(MultiRunMetricsWorkflow):
        @staticmethod
        def task(a):
            return dict(m=float(a))

    W().run(a=multirun([1, 2]), working_dir=str(tmp_path / "s"))
    files = sorted((tmp_path / "s").glob("*/.hydra/hydra.yaml"))
    unset = OmegaConf.to_yaml(OmegaConf.create({"hydra": {"job": {"num": "???"}}}))
    for bad, missing in [(files[0], files[1]), (files[1], files[0])]:
        missing.write_text(unset)
        bad.write_text("hydra: [\n")
        with pytest.raises(Exception) as err:
            MultiRunMetricsWorkflow().load_from_dir(
                tmp_path / "s", "mushin_metrics.json"
            )
        assert type(err.value).__module__.startswith("yaml"), repr(err.value)


def test_load_from_dir_with_a_mistyped_path_changes_nothing(tmp_path):
    """The path is checked before any state is reset, so a typo leaves a
    workflow that ran as it was."""
    from mushin import multirun
    from mushin.workflows import MultiRunMetricsWorkflow

    class W(MultiRunMetricsWorkflow):
        @staticmethod
        def task(a):
            return dict(m=float(a))

    wf = W()
    wf.run(a=multirun([1, 2]), working_dir=str(tmp_path / "s"))
    dirs, jobs = list(wf.multirun_working_dirs), list(wf.jobs)
    with pytest.raises(FileNotFoundError):
        wf.load_from_dir(tmp_path / "typo", "mushin_metrics.json")
    assert wf.multirun_working_dirs == dirs and wf.jobs == jobs
    assert sorted(float(v) for v in wf.to_xarray()["m"].values) == [1.0, 2.0]


def test_load_from_dir_nan_fills_failed_cells(tmp_path):
    """A fail-soft sweep must be reloadable offline: a failed cell has no
    metrics sidecar, so load_metrics must NaN-fill it (consulting the cell's
    recorded status) instead of raising FileNotFoundError."""
    from mushin import multirun
    from mushin.workflows import MultiRunMetricsWorkflow

    class W(MultiRunMetricsWorkflow):
        @staticmethod
        def task(a):
            if a == 2:
                raise RuntimeError("boom")
            return dict(m=float(a) * 10)

    wf = W()
    with pytest.warns(UserWarning, match="failed"):
        wf.run(a=multirun([1, 2, 3]), working_dir=str(tmp_path / "s"), on_error="nan")

    loader = W().load_from_dir(tmp_path / "s", "mushin_metrics.json")
    ds = loader.to_xarray()
    vals = ds["m"].values.tolist()
    assert vals[0] == 10.0 and vals[2] == 30.0
    assert vals[1] != vals[1]  # NaN


def test_load_from_dir_nan_fills_sampled_out_cells(tmp_path):
    """A sampled sweep records un-sampled cells as 'skipped' with no metrics;
    reloading it offline must NaN-fill those cells, not raise."""
    from mushin import multirun
    from mushin.workflows import MultiRunMetricsWorkflow

    class W(MultiRunMetricsWorkflow):
        @staticmethod
        def task(a):
            return dict(m=float(a) * 10)

    wf = W()
    with pytest.warns(UserWarning, match="skipped"):
        wf.run(a=multirun([1, 2, 3, 4]), working_dir=str(tmp_path / "s"), sample=2)

    loader = W().load_from_dir(tmp_path / "s", "mushin_metrics.json")
    ds = loader.to_xarray()
    vals = ds["m"].values.tolist()
    present = [v for v in vals if v == v]
    assert len(present) == 2 and len(vals) == 4


def test_load_from_dir_restores_the_failure_record(tmp_path):
    """A reloaded fail-soft sweep must carry the `mushin_failures` attr the
    dataset from `.run()` had. Without it `compare_methods` cannot tell that the
    NaN cells are missing results, and compares the sweep as if it were complete."""
    from mushin import multirun
    from mushin.workflows import MultiRunMetricsWorkflow

    class W(MultiRunMetricsWorkflow):
        @staticmethod
        def task(a, b):
            if a == 2 and b == 1:
                raise RuntimeError("boom")
            return dict(m=float(a) * 10 + b)

    wf = W()
    with pytest.warns(UserWarning, match="failed"):
        wf.run(
            a=multirun([1, 2, 3]),
            b=multirun([0, 1]),
            working_dir=str(tmp_path / "s"),
            on_error="nan",
        )
    ran = wf.to_xarray()

    loader = W().load_from_dir(tmp_path / "s", "mushin_metrics.json")
    assert loader.failures == wf.failures
    assert loader.skipped == []
    assert loader.to_xarray().attrs["mushin_failures"] == ran.attrs["mushin_failures"]


def test_load_from_dir_restores_the_skip_record(tmp_path):
    """Likewise for `mushin_skipped`: a sampled-out cell is a missing result."""
    from mushin import multirun
    from mushin.workflows import MultiRunMetricsWorkflow

    class W(MultiRunMetricsWorkflow):
        @staticmethod
        def task(a):
            return dict(m=float(a) * 10)

    wf = W()
    with pytest.warns(UserWarning, match="skipped"):
        wf.run(a=multirun([1, 2, 3, 4]), working_dir=str(tmp_path / "s"), sample=2)
    ran = wf.to_xarray()

    loader = W().load_from_dir(tmp_path / "s", "mushin_metrics.json")
    assert loader.skipped == wf.skipped
    assert loader.failures == []
    attrs = loader.to_xarray().attrs
    assert attrs["mushin_skipped"] == ran.attrs["mushin_skipped"]
    assert "mushin_failures" not in attrs


def test_load_from_dir_of_a_complete_sweep_adds_no_record(tmp_path):
    from mushin import multirun
    from mushin.workflows import MultiRunMetricsWorkflow

    class W(MultiRunMetricsWorkflow):
        @staticmethod
        def task(a):
            return dict(m=float(a))

    W().run(a=multirun([1, 2]), working_dir=str(tmp_path / "s"))
    loader = W().load_from_dir(tmp_path / "s", "mushin_metrics.json")
    assert loader.failures == [] and loader.skipped == []
    assert loader.is_complete is True
    attrs = loader.to_xarray().attrs
    assert "mushin_failures" not in attrs and "mushin_skipped" not in attrs


def test_a_sweep_loaded_without_metrics_knows_it_is_incomplete(tmp_path):
    """`MultiRunMetricsWorkflow(working_dir=...)` and `load_from_dir(d, None)`
    load no metrics; the cells' statuses alone still give the record."""
    from mushin import multirun
    from mushin.workflows import MultiRunMetricsWorkflow

    class W(MultiRunMetricsWorkflow):
        @staticmethod
        def task(a):
            if a == 2:
                raise RuntimeError("boom")
            return dict(m=float(a))

    wd = tmp_path / "s"
    with pytest.warns(UserWarning, match="failed"):
        W().run(a=multirun([1, 2, 3]), working_dir=str(wd), on_error="nan")

    for loader in (W(working_dir=wd), W().load_from_dir(wd, None)):
        assert [f["combo"] for f in loader.failures] == ["a=2"]
        assert loader.skipped == [] and loader.is_complete is False


def test_load_from_dir_record_of_a_sweep_from_before_status_sidecars(tmp_path):
    """Such a sweep has only the end-of-run manifest: its failed cell must stay
    a failure, and a completed cell whose metrics file holds `{}` a result."""
    import json

    from mushin import multirun
    from mushin.workflows import MultiRunMetricsWorkflow

    class W(MultiRunMetricsWorkflow):
        @staticmethod
        def task(a):
            if a == 2:
                raise RuntimeError("boom")
            return {} if a == 3 else dict(m=float(a))

    wd = tmp_path / "s"
    with pytest.warns(UserWarning, match="failed"):
        W().run(a=multirun([1, 2, 3]), working_dir=str(wd), on_error="nan")
    for status_file in wd.glob("*/mushin_cell_status.json"):
        status_file.unlink()
    (wd / "2" / "mushin_metrics.json").write_text(json.dumps({}))

    loader = W().load_from_dir(wd, None)
    assert [f["combo"] for f in loader.failures] == ["a=2"]
    assert loader.skipped == [] and loader.is_complete is False

    # Loaded with metrics: the `{}` file is a result too.
    wd2 = tmp_path / "s2"
    W().run(a=multirun([1, 3]), working_dir=str(wd2))
    for status_file in wd2.glob("*/mushin_cell_status.json"):
        status_file.unlink()
    loader = W().load_from_dir(wd2, "mushin_metrics.json")
    assert loader.failures == [] and loader.skipped == []
    assert loader.is_complete is True


def test_load_from_dir_record_ignores_an_earlier_sweep_of_the_dir(tmp_path):
    """Job dirs an earlier, wider sweep left in a reused working_dir are not
    cells of the latest sweep: their 'failed' status must not mark the latest
    (complete) sweep incomplete. Nor may a dir over other axes crash the load."""
    from mushin import multirun
    from mushin.workflows import MultiRunMetricsWorkflow

    class W(MultiRunMetricsWorkflow):
        @staticmethod
        def task(method, seed=0):
            if method == "c":
                raise RuntimeError("boom")
            return dict(m=float(seed))

    wd = str(tmp_path / "s")
    with pytest.warns(UserWarning, match="failed"):
        W().run(method=multirun(["a", "b", "c"]), working_dir=wd, on_error="nan")
    # The latest sweep: fewer cells, and a new axis, in the same dir. Dir 2
    # (method=c, failed, no seed in its config) is left from the first sweep.
    W().run(method=multirun(["a"]), seed=multirun([0, 1]), working_dir=wd)

    loader = W().load_from_dir(wd, "mushin_metrics.json")
    assert loader.failures == [] and loader.skipped == []
    assert loader.is_complete is True
    ds = loader.to_xarray()
    assert "mushin_failures" not in ds.attrs
    assert ds["m"].values.tolist() == [[0.0, 1.0]]


def test_load_from_dir_records_cells_that_never_finished(tmp_path):
    """A sweep killed between cells leaves no job dir (and no manifest) for the
    cells it never started; one killed mid-cell leaves a 'running' status. Both
    are cells without a result, recorded as not run, not silently NaN-filled."""
    import json
    import shutil

    from mushin import multirun
    from mushin._sweep_io import MANIFEST_FILE, METRICS_FILE
    from mushin.workflows import MultiRunMetricsWorkflow

    class W(MultiRunMetricsWorkflow):
        @staticmethod
        def task(a):
            return dict(m=float(a))

    wd = tmp_path / "s"
    W().run(a=multirun([1, 2, 3, 4]), working_dir=str(wd))
    # Simulate the kill: no end-of-run manifest, cell 3 never started, and
    # cell 2 was interrupted after it marked itself running.
    (wd / MANIFEST_FILE).unlink()
    shutil.rmtree(wd / "3")
    status_file = wd / "2" / "mushin_cell_status.json"
    status = json.loads(status_file.read_text())
    status_file.write_text(json.dumps({**status, "status": "running"}))
    (wd / "2" / METRICS_FILE).unlink()

    loader = W().load_from_dir(wd, "mushin_metrics.json")
    assert loader.failures == []
    assert [s["combo"] for s in loader.skipped] == ["a=3", "a=4"]
    assert loader.skipped[1]["working_dir"] is None
    assert loader.is_complete is False
    ds = loader.to_xarray()
    assert json.loads(ds.attrs["mushin_skipped"]) == ["a=3", "a=4"]
    vals = ds["m"].values.tolist()
    assert vals[:2] == [1.0, 2.0] and vals[2] != vals[2] and vals[3] != vals[3]


def test_load_from_dir_record_of_a_complete_sweep_over_inf(tmp_path):
    """Cells keyed by a non-finite swept value must match the grid's keys, or a
    complete sweep would be reported incomplete."""
    from mushin import multirun
    from mushin.workflows import MultiRunMetricsWorkflow

    class W(MultiRunMetricsWorkflow):
        @staticmethod
        def task(clip, seed):
            return dict(m=float(seed))

    W().run(
        clip=multirun([1.0, float("inf")]),
        seed=multirun([0, 1]),
        working_dir=str(tmp_path / "s"),
    )
    loader = W().load_from_dir(tmp_path / "s", "mushin_metrics.json")
    assert loader.failures == [] and loader.skipped == []
    assert loader.is_complete is True


def test_load_from_dir_record_and_data_agree_after_a_second_load(tmp_path):
    """Reloading metrics must refresh the data along with the record: a cell
    recorded as not run must not keep the value an earlier load cached."""
    import json

    from mushin import multirun
    from mushin.workflows import MultiRunMetricsWorkflow

    class W(MultiRunMetricsWorkflow):
        @staticmethod
        def task(a):
            return dict(m=float(a))

    wd = tmp_path / "s"
    W().run(a=multirun([1, 2, 3]), working_dir=str(wd))
    loader = W().load_from_dir(wd, "mushin_metrics.json")
    assert loader.to_xarray()["m"].values.tolist() == [1.0, 2.0, 3.0]

    status_file = wd / "1" / "mushin_cell_status.json"  # a=2 is re-running
    status_file.write_text(
        json.dumps({**json.loads(status_file.read_text()), "status": "running"})
    )
    vals = loader.to_xarray(metrics_filename="mushin_metrics.json")["m"].values
    assert [s["combo"] for s in loader.skipped] == ["a=2"]
    assert vals[0] == 1.0 and vals[1] != vals[1] and vals[2] == 3.0


def test_load_from_dir_counts_a_completed_cell_with_no_metrics_as_done(tmp_path):
    """As `.run()` does: a task that returned no metrics still completed."""
    from mushin import multirun
    from mushin.workflows import MultiRunMetricsWorkflow

    class W(MultiRunMetricsWorkflow):
        @staticmethod
        def task(a):
            return {} if a == 2 else dict(m=float(a))

    wf = W()
    wf.run(a=multirun([1, 2]), working_dir=str(tmp_path / "s"))
    assert wf.is_complete is True
    loader = W().load_from_dir(tmp_path / "s", "mushin_metrics.json")
    assert loader.skipped == [] and loader.is_complete is True


def test_load_from_dir_failure_text_is_not_from_an_older_attempt(tmp_path):
    """The manifest is written at the end of a run, so a resume killed before
    it leaves the previous attempt's error text in place: use it only when the
    manifest is newer than the cell's status."""
    import os
    import time

    from mushin import multirun
    from mushin.workflows import MultiRunMetricsWorkflow

    class W(MultiRunMetricsWorkflow):
        @staticmethod
        def task(a):
            if a == 2:
                raise RuntimeError("first attempt")
            return dict(m=float(a))

    wd = tmp_path / "s"
    with pytest.warns(UserWarning, match="failed"):
        W().run(a=multirun([1, 2]), working_dir=str(wd), on_error="nan")
    loader = W().load_from_dir(wd, "mushin_metrics.json")
    assert "first attempt" in loader.failures[0]["exception"]

    later = time.time() + 60  # the cell failed again after the manifest was saved
    os.utime(wd / "1" / "mushin_cell_status.json", (later, later))
    loader = W().load_from_dir(wd, "mushin_metrics.json")
    assert [f["combo"] for f in loader.failures] == ["a=2"]
    assert loader.failures[0]["exception"] is None


def test_load_from_dir_record_paths_are_absolute(tmp_path, monkeypatch):
    from mushin import multirun
    from mushin.workflows import MultiRunMetricsWorkflow

    class W(MultiRunMetricsWorkflow):
        @staticmethod
        def task(a):
            if a == 2:
                raise RuntimeError("boom")
            return dict(m=float(a))

    monkeypatch.chdir(tmp_path)
    wf = W()
    with pytest.warns(UserWarning, match="failed"):
        wf.run(a=multirun([1, 2]), working_dir="s", on_error="nan")
    loader = W().load_from_dir("s", "mushin_metrics.json")
    assert loader.failures == wf.failures
    assert Path(loader.failures[0]["working_dir"]).is_absolute()


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


@pytest.mark.parametrize("legacy_key", ["config_hash", "code_hash"])
def test_resume_reuses_legacy_sidecars(tmp_path, legacy_key):
    """Sweeps recorded before the config hash (or the code hash) existed must
    still resume (reuse) their completed cells rather than re-run them."""
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
    # Simulate an older sweep: strip the key from every status sidecar.
    status_files = list(wd.glob(f"*/{STATUS_FILE}"))
    assert len(status_files) == 2
    for status_file in status_files:
        d = _json.loads(status_file.read_text())
        assert d.pop(legacy_key) is not None
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


def test_failed_rerun_removes_stale_metrics_sidecar(tmp_path):
    """When a cell fails, a metrics sidecar left by a prior successful attempt
    in the same dir must be removed — otherwise offline tools present the old
    metrics as if the failed attempt produced them."""
    from mushin import multirun
    from mushin.workflows import MultiRunMetricsWorkflow

    class Good(MultiRunMetricsWorkflow):
        @staticmethod
        def task(a):
            return dict(m=float(a))

    class Bad(MultiRunMetricsWorkflow):
        @staticmethod
        def task(a):
            raise RuntimeError("boom")

    wd = tmp_path / "s"
    Good().run(a=multirun([1.0]), working_dir=str(wd))
    assert (wd / "0" / "mushin_metrics.json").exists()

    with pytest.warns(UserWarning, match="failed"):
        Bad().run(a=multirun([1.0]), working_dir=str(wd), on_error="nan")
    assert not (wd / "0" / "mushin_metrics.json").exists()


def test_dry_run_and_failed_validation_preserve_loaded_state(tmp_path):
    """dry_run previews (and run() calls that fail argument validation) must
    not wipe a loaded workflow's reload state — otherwise a later to_xarray()
    silently NaN-fills, the exact failure mode the offline fixes address."""
    from mushin import multirun
    from mushin.workflows import MultiRunMetricsWorkflow

    class W(MultiRunMetricsWorkflow):
        @staticmethod
        def task(a):
            if a == 2:
                raise RuntimeError("boom")
            return dict(m=float(a) * 10)

    with pytest.warns(UserWarning, match="failed"):
        W().run(a=multirun([1, 2, 3]), working_dir=str(tmp_path / "s"), on_error="nan")

    loader = W().load_from_dir(tmp_path / "s", "mushin_metrics.json")

    loader.run(a=multirun([1, 2, 3]), dry_run=True)  # documented no-op preview
    vals = loader.to_xarray()["m"].values.tolist()
    assert vals[0] == 10.0 and vals[2] == 30.0 and vals[1] != vals[1]

    with pytest.raises(ValueError, match="resume"):
        loader.run(a=multirun([1]), resume=True)  # fails validation, no launch
    vals = loader.to_xarray()["m"].values.tolist()
    assert vals[0] == 10.0 and vals[2] == 30.0 and vals[1] != vals[1]


def test_pre_task_failure_marks_cell_failed_and_removes_stale_metrics(tmp_path):
    """A setup-phase (pre_task) failure under fail-soft must leave the same
    durable 'failed' marker — and remove a stale metrics sidecar — as a
    task-body failure, or offline tools present the prior sweep's metrics as
    this sweep's results."""
    import json

    from mushin import multirun
    from mushin.workflows import MultiRunMetricsWorkflow

    class Good(MultiRunMetricsWorkflow):
        @staticmethod
        def task(a):
            return dict(m=float(a) * 10)

    class Bad(Good):
        @staticmethod
        def pre_task(a):
            raise RuntimeError("setup boom")

    wd = tmp_path / "s"
    Good().run(a=multirun([1.0]), working_dir=str(wd))
    assert (wd / "0" / "mushin_metrics.json").exists()

    with pytest.warns(UserWarning, match="failed"):
        Bad().run(a=multirun([1.0]), working_dir=str(wd), on_error="nan")

    status = json.loads((wd / "0" / "mushin_cell_status.json").read_text())
    assert status["status"] == "failed"
    assert not (wd / "0" / "mushin_metrics.json").exists()


def test_skipped_cell_removes_stale_metrics_sidecar(tmp_path):
    """A cell skipped by sample= (or an exhausted budget) in a reused dir must
    not retain the previous sweep's metrics sidecar — show/export would list
    old metrics on a 'skipped' row while to_xarray NaN-fills it."""
    import json

    from mushin import multirun
    from mushin.workflows import MultiRunMetricsWorkflow

    class W(MultiRunMetricsWorkflow):
        @staticmethod
        def task(a):
            return dict(m=float(a) * 10)

    wd = tmp_path / "s"
    W().run(a=multirun([1, 2, 3]), working_dir=str(wd))  # all completed

    with pytest.warns(UserWarning, match="skipped"):
        W().run(a=multirun([1, 2, 3]), working_dir=str(wd), sample=1)

    for d in wd.iterdir():
        if not d.is_dir():
            continue
        status = json.loads((d / "mushin_cell_status.json").read_text())["status"]
        if status == "skipped":
            assert not (d / "mushin_metrics.json").exists()


class _ReusedDir(MultiRunMetricsWorkflow):
    @staticmethod
    def task(x, lr):
        if x == 13:
            raise RuntimeError("boom")
        return dict(v=float(x) * lr)


def _a_job_died_before_its_first_record(
    tmp_path, first, second, *, manifest=False, run_id=True, stamp=True, dead="0"
):
    """Sweep ``first`` into a dir, then leave that dir as a later sweep
    ``second`` leaves it when its job 0 dies after Hydra wrote the cell's
    ``.hydra/`` and before mushin's runner wrote anything (#355): the later
    sweep's multirun.yaml, the dead cell's (``dead``, cell 0 by default)
    snapshot over the earlier cell's record and metrics, and the later
    sweep's other cells, complete. ``manifest`` brings the
    later sweep's manifest (it finished the other cells); ``run_id=False``
    drops the earlier record's run id, as a released version wrote none;
    ``stamp=False`` drops the run from the later snapshot, as a version
    before it wrote."""
    import json
    import os
    import shutil

    wd, later = tmp_path / "s", tmp_path / "later"
    _ReusedDir().run(working_dir=str(wd), **first)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # a failed cell, on purpose
        _ReusedDir().run(working_dir=str(later), on_error="nan", **second)
    shutil.copy(later / "multirun.yaml", wd / "multirun.yaml")
    for cell in later.glob("[0-9]*"):
        if cell.name == dead:
            shutil.rmtree(wd / dead / ".hydra")
            shutil.copytree(cell / ".hydra", wd / dead / ".hydra")
        else:
            shutil.rmtree(wd / cell.name, ignore_errors=True)
            shutil.copytree(cell, wd / cell.name)
    path = wd / dead / "mushin_cell_status.json"
    if not run_id:
        record = json.loads(path.read_text())
        del record["run_id"]
        path.write_text(json.dumps(record))
    if not stamp:
        hydra = wd / dead / ".hydra" / "hydra.yaml"
        text = hydra.read_text()
        assert "mushin_run_id" in text
        hydra.write_text(
            "\n".join(line for line in text.splitlines() if "mushin_run_id" not in line)
        )
    # Hydra writes the snapshot after the earlier run ended (its record, then
    # its manifest), whatever the copy did, and the later sweep's manifest after
    # that. Dated from the earlier manifest too: a slow runner can take more
    # than the second below between the earlier record and that manifest.
    earlier = max(
        path.stat().st_mtime_ns, (wd / "mushin_sweep_manifest.json").stat().st_mtime_ns
    )
    for name in ("config.yaml", "hydra.yaml", "overrides.yaml"):
        os.utime(wd / dead / ".hydra" / name, ns=(earlier + 10**9, earlier + 10**9))
    if manifest:
        # Written after the snapshot, or in the same tick of a coarse clock.
        shutil.copy(later / "mushin_sweep_manifest.json", wd)
        tick = 10**9 if manifest == "same tick" else 2 * 10**9
        os.utime(wd / "mushin_sweep_manifest.json", ns=(earlier + tick,) * 2)
    return wd


@pytest.mark.parametrize(
    ("first", "second", "run_id"),
    [
        # Another cell of another grid, with and without a run id in its record.
        (dict(x=multirun([1, 2]), lr=1.0), dict(x=multirun([3, 4]), lr=1.0), True),
        (dict(x=multirun([1, 2]), lr=1.0), dict(x=multirun([3, 4]), lr=1.0), False),
        # The same combo under another fixed value: the earlier run's result.
        (dict(x=multirun([1, 2]), lr=1.0), dict(x=multirun([1, 2]), lr=10.0), True),
    ],
    ids=["another-cell", "another-cell-no-run-id", "another-config"],
)
def test_a_reload_does_not_take_an_earlier_runs_record_for_a_job_that_died(
    tmp_path, first, second, run_id
):
    import mushin

    wd = _a_job_died_before_its_first_record(tmp_path, first, second, run_id=run_id)
    loaded = _ReusedDir().load_from_dir(wd, "mushin_metrics.json")
    values = loaded.to_xarray()["v"].values.tolist()
    later = float(list(second["x"])[1]) * second["lr"]
    assert values[0] != values[0]  # NaN: the job left no result
    assert values[1] == later
    assert not loaded.is_complete and not loaded.failures
    # show, best and export read the same: the earlier result is not this run's.
    from mushin._show import _read_cells

    assert mushin.best(wd, "v", mode="min").value == later
    assert [cell["dir"].name for cell in _read_cells(wd)] == ["1"]


@pytest.mark.parametrize("manifest", [True, "same tick"])
def test_a_cell_whose_job_died_is_failed_as_its_runs_manifest_says(tmp_path, manifest):
    first = dict(x=multirun([1, 2]), lr=1.0)
    second = dict(x=multirun([13, 4]), lr=1.0)  # 13 fails in the later sweep
    wd = _a_job_died_before_its_first_record(tmp_path, first, second, manifest=manifest)
    loaded = _ReusedDir().load_from_dir(wd, "mushin_metrics.json")
    assert [f["combo"] for f in loaded.failures] == ["x=13"]
    # An earlier run's manifest says nothing about this run's cell, even
    # where the earlier run failed the same combo.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        wd = _a_job_died_before_its_first_record(
            tmp_path / "b", dict(second, on_error="nan"), dict(second, lr=10.0)
        )
    loaded = _ReusedDir().load_from_dir(wd, "mushin_metrics.json")
    assert not loaded.failures


def test_a_snapshot_from_before_the_run_stamp_says_nothing(tmp_path):
    """A snapshot that names no run (written before it was stamped) leaves
    the record standing, as before. The earlier sweep's value then shows
    for this cell: a known limit for sweeps from before the stamp, pinned so
    a change to it is deliberate."""
    first = dict(x=multirun([1, 2]), lr=1.0)
    second = dict(x=multirun([3, 4]), lr=1.0)
    wd = _a_job_died_before_its_first_record(tmp_path, first, second, stamp=False)
    loaded = _ReusedDir().load_from_dir(wd, "mushin_metrics.json")
    assert loaded.to_xarray()["v"].values.tolist() == [1.0, 4.0]


def test_another_runs_record_is_found_whatever_the_dirs_order(tmp_path):
    """Dirs sort by job number, not by name ("10" after "9"): the cell
    judged is the dead job's, not the one at its place in name order."""
    first = dict(x=multirun(list(range(12))), lr=1.0)
    second = dict(x=multirun(list(range(12))), lr=10.0)
    wd = _a_job_died_before_its_first_record(tmp_path, first, second, dead="10")
    values = _ReusedDir().load_from_dir(wd, "mushin_metrics.json").to_xarray()
    v = values["v"].values.tolist()
    assert v[10] != v[10]
    assert [x for i, x in enumerate(v) if i != 10] == [
        10.0 * i for i in range(12) if i != 10
    ]


def test_a_run_after_a_load_judges_no_record_by_that_load(tmp_path):
    first = dict(x=multirun([1, 2]), lr=1.0)
    second = dict(x=multirun([3, 4]), lr=1.0)
    wd = _a_job_died_before_its_first_record(tmp_path, first, second)
    wf = _ReusedDir()
    wf.load_from_dir(wd, "mushin_metrics.json")
    wf.run(x=multirun([5, 6]), lr=1.0, working_dir=str(tmp_path / "fresh"))
    assert wf.load_metrics("mushin_metrics.json") == {"v": [5.0, 6.0]}


def test_a_cell_that_completes_after_a_load_counts_on_the_next_read(tmp_path):
    """A reload mid-sweep finds a job between its snapshot and its first
    record; once that job records its result, the same workflow reads it."""
    import json

    from mushin._hydra_yaml import hydra_run_stamp, load_hydra_record
    from mushin._sweep_io import write_metrics_sidecar

    first = dict(x=multirun([1, 2]), lr=1.0)
    second = dict(x=multirun([3, 4]), lr=1.0)
    wd = _a_job_died_before_its_first_record(tmp_path, first, second)
    wf = _ReusedDir()
    wf.load_from_dir(wd, "mushin_metrics.json")
    assert wf._job_metrics_by_index[0] is None
    # The job finishes: its run's record and metrics.
    run = hydra_run_stamp(load_hydra_record(wd / "0" / ".hydra" / "hydra.yaml"))
    path = wd / "0" / "mushin_cell_status.json"
    record = json.loads(path.read_text())
    record.update(run_id=run, combo={"x": 3})
    path.write_text(json.dumps(record))
    write_metrics_sidecar(wd / "0", {"v": 3.0})
    assert wf.load_metrics("mushin_metrics.json") == {"v": [3.0, 4.0]}


@pytest.mark.parametrize(
    "line",
    [
        None,
        "    mushin_run_id: abc123",
        "    mushin_run_id: '1e500000000000000000000000000000'",
        '    mushin_run_id: "0123"',
        "    mushin_run_id: '???'",
        "    mushin_run_id: ???",
        "    mushin_run_id: ${hydra.job.name}",
        "    mushin_run_id: ''",
        "    mushin_run_id: null",
        "    mushin_run_id: 12345",
        "    mushin_run_id: 1e5",
    ],
)
def test_the_stamp_line_reads_as_the_parsed_file(tmp_path, line):
    """show reads a snapshot's run from its line, without a YAML parser
    (`snapshot_run_stamp`); it must give what the parse gives."""
    from mushin._hydra_yaml import hydra_run_stamp, load_hydra_record
    from mushin._resume import snapshot_run_stamp

    lines = ["hydra:", "  job:", "    name: abc"] + ([line] if line else [])
    path = tmp_path / "hydra.yaml"
    path.write_text("\n".join(lines) + "\n")
    assert snapshot_run_stamp(path) == hydra_run_stamp(load_hydra_record(path))


def test_the_stamp_line_is_read_from_the_job_section_only(tmp_path):
    """The same key under another node of ``hydra:`` (a user override such
    as ``++hydra.run.mushin_run_id=x``) is not the job's stamp."""
    from mushin._hydra_yaml import hydra_run_stamp, load_hydra_record
    from mushin._resume import snapshot_run_stamp

    path = tmp_path / "hydra.yaml"
    path.write_text(
        "hydra:\n  run:\n    dir: out\n    mushin_run_id: decoy\n"
        "  job:\n    name: abc\n    mushin_run_id: abc123\n"
        "  runtime:\n    mushin_run_id: later\n"
    )
    assert snapshot_run_stamp(path) == "abc123"
    assert hydra_run_stamp(load_hydra_record(path)) == "abc123"


def test_the_stamp_line_reads_real_snapshots(tmp_path, monkeypatch):
    from mushin._hydra_yaml import hydra_run_stamp, load_hydra_record
    from mushin._resume import snapshot_run_stamp
    from mushin.workflows import _TaskRunner

    init = _TaskRunner.__init__
    for run_id in ("1e5" + "0" * 29, "1" * 32, "abc" + "0" * 29):

        def given(self, *args, run_id=run_id, **kwargs):
            init(self, *args, **kwargs)
            self.run_id = run_id

        monkeypatch.setattr(_TaskRunner, "__init__", given)
        wd = tmp_path / run_id[:4]
        _ReusedDir().run(x=multirun([1]), lr=1.0, working_dir=str(wd))
        path = wd / "0" / ".hydra" / "hydra.yaml"
        assert snapshot_run_stamp(path) == run_id
        assert hydra_run_stamp(load_hydra_record(path)) == run_id


@pytest.mark.parametrize("case", ["copied", "before-the-stamp"])
def test_show_keeps_an_older_record_of_the_same_run(tmp_path, case):
    """show, best and export keep a record older than its snapshot when both
    name one run (a copy without its files' times), or when the snapshot
    names none (written before the stamp)."""
    import os

    import mushin
    from mushin._show import _read_cells

    wd = tmp_path / "s"
    _ReusedDir().run(x=multirun([1, 2]), lr=2.0, working_dir=str(wd))
    for cell in ("0", "1"):
        hydra = wd / cell / ".hydra" / "hydra.yaml"
        if case == "before-the-stamp":
            text = hydra.read_text()
            hydra.write_text(
                "\n".join(x for x in text.splitlines() if "mushin_run_id" not in x)
            )
        record = (wd / cell / "mushin_cell_status.json").stat().st_mtime_ns
        os.utime(hydra, ns=(record + 10**9, record + 10**9))
    assert [c["dir"].name for c in _read_cells(wd)] == ["0", "1"]
    assert mushin.best(wd, "v", mode="min").value == 2.0


def test_a_run_that_raises_after_a_load_keeps_the_loads_judgment(tmp_path):
    """A run that raises before its jobs are collected leaves the loaded
    dirs in place; their record of another run must stay judged."""
    first = dict(x=multirun([1, 2]), lr=1.0)
    second = dict(x=multirun([3, 4]), lr=1.0)
    wd = _a_job_died_before_its_first_record(tmp_path, first, second)
    wf = _ReusedDir()
    wf.load_from_dir(wd, "mushin_metrics.json")
    with pytest.raises(Exception):
        wf.run(x=multirun([13]), lr=1.0, working_dir=str(tmp_path / "fresh"))
    v = wf.to_xarray(metrics_filename="mushin_metrics.json")["v"].values.tolist()
    assert v[0] != v[0] and v[1] == 4.0


def test_a_cell_a_later_run_fills_counts_on_the_next_read(tmp_path):
    """After a load, a resume from elsewhere computes the dead cell: its new
    record is no longer older than its snapshot, and the same workflow reads
    it, as a fresh load would."""
    first = dict(x=multirun([1, 2]), lr=1.0)
    second = dict(x=multirun([3, 4]), lr=1.0)
    wd = _a_job_died_before_its_first_record(tmp_path, first, second)
    wf = _ReusedDir()
    wf.load_from_dir(wd, "mushin_metrics.json")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _ReusedDir().run(working_dir=str(wd), resume=True, **second)
    assert wf.load_metrics("mushin_metrics.json") == {"v": [3.0, 4.0]}


def test_each_jobs_snapshot_names_the_run_its_record_does(tmp_path):
    import json

    from mushin._hydra_yaml import hydra_run_stamp, load_hydra_record

    _ReusedDir().run(x=multirun([1, 2]), lr=1.0, working_dir=str(tmp_path))
    stamps = set()
    for cell in ("0", "1"):
        record = json.loads((tmp_path / cell / "mushin_cell_status.json").read_text())
        snapshot = load_hydra_record(tmp_path / cell / ".hydra" / "hydra.yaml")
        assert hydra_run_stamp(snapshot) == record["run_id"]
        stamps.add(record["run_id"])
    assert len(stamps) == 1


class _Copied(MultiRunMetricsWorkflow):
    @staticmethod
    def task(x, lr, tag):
        return dict(v=float(x) * lr)


@pytest.mark.parametrize(
    "sweep",
    [
        dict(x=multirun([1, 2]), lr=2.0, tag="a"),
        dict(x=multirun([1, 2]), lr=multirun([1.0, float("inf")]), tag="a"),
        dict(x=multirun([1, 2]), lr=2.0, tag="${now:%H%M%S%f}"),
    ],
    ids=["plain", "non-finite", "time"],
)
def test_a_reload_keeps_a_cell_whose_snapshot_is_newer_than_its_record(tmp_path, sweep):
    """A sweep copied without its files' times (`scp -r`, `rsync` without
    `-t`) can land a cell's hydra.yaml after its record: both name the same
    run, so its result stays, whatever its config resolves to now."""
    import os

    wd = tmp_path / "s"
    wf = _Copied()
    wf.run(working_dir=str(wd), **sweep)
    ran = wf.to_xarray()
    for cell in wd.glob("[0-9]*"):
        record = (cell / "mushin_cell_status.json").stat().st_mtime_ns
        for name in ("config.yaml", "hydra.yaml", "overrides.yaml"):
            later = record + 10**9
            os.utime(cell / ".hydra" / name, ns=(later, later))
    loaded = _Copied().load_from_dir(wd, "mushin_metrics.json")
    assert loaded.to_xarray()["v"].values.tolist() == ran["v"].values.tolist()
    assert loaded.is_complete


def test_a_numeric_looking_run_id_stays_the_runs_stamp(tmp_path, monkeypatch):
    """The stamp is quoted: an id such as 1e5... or all digits must read
    back as itself, not as a number."""
    import json

    from mushin._hydra_yaml import hydra_run_stamp, load_hydra_record
    from mushin.workflows import _TaskRunner

    init = _TaskRunner.__init__
    for run_id in ("1e5" + "0" * 29, "1" * 32):

        def numeric(self, *args, run_id=run_id, **kwargs):
            init(self, *args, **kwargs)
            self.run_id = run_id

        monkeypatch.setattr(_TaskRunner, "__init__", numeric)
        wd = tmp_path / run_id[:4]
        _ReusedDir().run(x=multirun([1]), lr=1.0, working_dir=str(wd))
        record = json.loads((wd / "0" / "mushin_cell_status.json").read_text())
        snapshot = load_hydra_record(wd / "0" / ".hydra" / "hydra.yaml")
        assert record["run_id"] == run_id
        assert hydra_run_stamp(snapshot) == run_id


@pytest.mark.parametrize(
    ("stamp", "read"),
    [
        ("abc123", "abc123"),
        (None, None),
        ("", None),
        ("???", None),
        ("${hydra.job.name}", None),
        (5, None),
        ("interpolated job", None),  # `hydra.job` itself an interpolation
    ],
)
def test_a_run_stamp_reads_alike_on_both_read_paths(stamp, read):
    from omegaconf import OmegaConf

    from mushin._hydra_yaml import hydra_run_stamp

    job = {"name": "abc"} if stamp is None else {"name": "abc", "mushin_run_id": stamp}
    tree = {"hydra": {"job": job}}
    if stamp == "interpolated job":
        tree = {"hydra": {"job": "${other}"}, "other": {"mushin_run_id": "abc123"}}
    assert hydra_run_stamp(tree) == read
    assert hydra_run_stamp(OmegaConf.create(tree)) == read


class _SometimesNone(MultiRunMetricsWorkflow):
    @staticmethod
    def task(x):
        return None if x == 2 else dict(v=float(x))


def test_a_reload_reads_a_cell_that_returned_no_metrics_as_the_run_does(tmp_path):
    """#364: a completed cell whose task returned None has no metrics file,
    and its record says so: the reload gives it NaN, as the run does,
    instead of raising."""
    wf = _SometimesNone()
    wf.run(x=multirun([1, 2]), working_dir=str(tmp_path))
    ran = wf.to_xarray()["v"].values.tolist()
    loaded = _SometimesNone().load_from_dir(tmp_path, "mushin_metrics.json")
    values = loaded.to_xarray()["v"].values.tolist()
    assert values[0] == ran[0] == 1.0
    assert values[1] != values[1] and ran[1] != ran[1]
    assert loaded.is_complete == wf.is_complete
    assert [f["combo"] for f in loaded.failures] == [f["combo"] for f in wf.failures]
    assert [s["combo"] for s in loaded.skipped] == [s["combo"] for s in wf.skipped]


def test_a_metrics_file_a_cell_wrote_and_lost_still_raises(tmp_path):
    """Only a task that returned None is recorded as writing no metrics: a
    missing file of a cell whose task returned metrics was lost."""
    _SometimesNone().run(x=multirun([1, 2]), working_dir=str(tmp_path))
    (tmp_path / "0" / "mushin_metrics.json").unlink()
    with pytest.raises(FileNotFoundError, match="No files") as info:
        _SometimesNone().load_from_dir(tmp_path, "mushin_metrics.json")
    assert "earlier mushin version" not in str(info.value)


def test_a_record_from_before_the_flag_still_raises_with_a_hint(tmp_path):
    """A released version's record says nothing of a task that returned
    None, so it cannot tell that from a lost file: a reload raises, and says
    why."""
    import json

    _SometimesNone().run(x=multirun([1, 2]), working_dir=str(tmp_path))
    path = tmp_path / "1" / "mushin_cell_status.json"
    record = json.loads(path.read_text())
    for key in ("no_metrics", "run_id"):
        del record[key]
    path.write_text(json.dumps(record))
    with pytest.raises(FileNotFoundError, match="No files.*earlier mushin version"):
        _SometimesNone().load_from_dir(tmp_path, "mushin_metrics.json")


def test_the_earlier_version_hint_is_only_for_the_sidecar(tmp_path):
    import json

    _SometimesNone().run(x=multirun([1, 2]), working_dir=str(tmp_path))
    path = tmp_path / "0" / "mushin_cell_status.json"
    record = json.loads(path.read_text())
    del record["run_id"]
    path.write_text(json.dumps(record))
    with pytest.raises(FileNotFoundError, match="fit_metrics.pt") as info:
        _SometimesNone().load_from_dir(
            tmp_path, ["mushin_metrics.json", "fit_metrics.pt"]
        )
    assert "earlier mushin version" not in str(info.value)


@pytest.mark.parametrize("name", ["fit_metrics.pt", "nothing_*.json"])
def test_a_missing_file_of_the_users_own_still_raises(tmp_path, name):
    """A cell that returned None: only the sidecar is not looked for. Its
    job is the first, so no other cell's missing file raises first."""
    _SometimesNone().run(x=multirun([2, 1]), working_dir=str(tmp_path))
    with pytest.raises(FileNotFoundError, match="No files") as info:
        _SometimesNone().load_from_dir(tmp_path, ["mushin_metrics.json", name])
    assert str(tmp_path / "0") in str(info.value)


def test_the_sidecar_may_be_named_as_a_path(tmp_path):
    _SometimesNone().run(x=multirun([1, 2]), working_dir=str(tmp_path))
    loaded = _SometimesNone().load_from_dir(tmp_path, "./mushin_metrics.json")
    values = loaded.to_xarray()["v"].values.tolist()
    assert values[0] == 1.0 and values[1] != values[1]


def test_a_none_cell_whose_metrics_file_exists_loads_it(tmp_path):
    """In a job of several tasks, one returning None can write its record
    over that of the task that wrote the metrics: a file there still loads."""
    from mushin._sweep_io import write_metrics_sidecar

    _SometimesNone().run(x=multirun([1, 2]), working_dir=str(tmp_path))
    write_metrics_sidecar(tmp_path / "1", {"v": 2.0})
    loaded = _SometimesNone().load_from_dir(tmp_path, "mushin_metrics.json")
    assert loaded.to_xarray()["v"].values.tolist() == [1.0, 2.0]


def test_a_cell_that_returns_none_keeps_no_metrics_an_earlier_sweep_left(tmp_path):
    """#355: a task that returns None has no metrics, so a metrics file an
    earlier sweep left in its reused dir is not this cell's."""
    from mushin import multirun
    from mushin.workflows import MultiRunMetricsWorkflow

    class First(MultiRunMetricsWorkflow):
        @staticmethod
        def task(x):
            return dict(v=float(x))

    class Second(MultiRunMetricsWorkflow):
        @staticmethod
        def task(x):
            return None

    First().run(x=multirun([1, 2]), working_dir=tmp_path)
    assert len(list(tmp_path.glob("*/mushin_metrics.json"))) == 2
    Second().run(x=multirun([1, 2]), working_dir=tmp_path)
    assert list(tmp_path.glob("*/mushin_metrics.json")) == []


@pytest.mark.parametrize("same_run", [False, True], ids=["another-run", "this-run"])
def test_a_cell_removes_only_another_runs_metrics_file(tmp_path, monkeypatch, same_run):
    """At its start a cell removes a metrics file only when the record it
    replaces is another run's: in a job of several tasks each runs the cell,
    and one that starts after another finished must keep what it wrote. Two
    runs given one run id stand in for two tasks of one job."""
    from mushin import multirun
    from mushin.workflows import MultiRunMetricsWorkflow, _TaskRunner

    class First(MultiRunMetricsWorkflow):
        @staticmethod
        def task(x):
            return dict(v=float(x))

    class Second(MultiRunMetricsWorkflow):
        @staticmethod
        def task(x):
            return None

    if same_run:
        init = _TaskRunner.__init__

        def one_run(self, *args, **kwargs):
            init(self, *args, **kwargs)
            self.run_id = "one run"

        monkeypatch.setattr(_TaskRunner, "__init__", one_run)
    First().run(x=multirun([1]), working_dir=tmp_path)
    Second().run(x=multirun([1]), working_dir=tmp_path)
    assert (tmp_path / "0" / "mushin_metrics.json").exists() is same_run


def test_offline_reload_restores_inf_metrics(tmp_path):
    """The tagged ±Inf sidecar marker must be decoded by the default
    metric_load_fn too — an offline reload must yield real Inf values, not
    leak {'__mushin_nonfinite__': ...} dicts into the dataset."""
    import math

    from mushin import multirun
    from mushin.workflows import MultiRunMetricsWorkflow

    class W(MultiRunMetricsWorkflow):
        @staticmethod
        def task(a):
            return dict(m=float("inf") if a == 1 else float(a))

    wd = tmp_path / "s"
    W().run(a=multirun([1, 2]), working_dir=str(wd))

    loader = W().load_from_dir(wd, "mushin_metrics.json")
    ds = loader.to_xarray()
    assert float(ds["m"].sel(a=1)) == math.inf
    assert float(ds["m"].sel(a=2)) == 2.0


def test_working_dir_with_equals_or_comma_in_name(tmp_path):
    """Experiment dirs are commonly named after params (`lr=0.1`); the
    working_dir override must be quoted so such paths don't break Hydra's
    override parser (or get split as an accidental sweep)."""
    from mushin import multirun
    from mushin.workflows import MultiRunMetricsWorkflow

    class W(MultiRunMetricsWorkflow):
        @staticmethod
        def task(a):
            return dict(m=float(a))

    for name in ("lr=0.1", "run,2"):
        wd = tmp_path / name
        wf = W()
        wf.run(a=multirun([1.0]), working_dir=str(wd))
        assert wf.is_complete
        assert (wd / "0" / "mushin_metrics.json").exists()


def test_target_job_dirs_tells_a_file_from_a_missing_dir(tmp_path):
    """A target that exists but is a file is reported as not a directory; one
    that is missing, as not existing. Both stay FileNotFoundError, as before."""
    from mushin.workflows import MultiRunMetricsWorkflow

    class W(MultiRunMetricsWorkflow):
        @staticmethod
        def task(job_dir):
            return dict(m=1.0)

    a_file = tmp_path / "metrics.json"
    a_file.write_text("{}", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="is not a directory"):
        W().run(target_job_dirs=[a_file], working_dir=str(tmp_path / "s"))
    with pytest.raises(FileNotFoundError, match="does not exist"):
        W().run(target_job_dirs=[tmp_path / "gone"], working_dir=str(tmp_path / "s"))


def test_a_target_job_dir_hydra_cannot_write_is_refused_before_launch(tmp_path):
    """The job directories are swept as strings, so one Hydra cannot write
    unquoted (a `#`, a non-ASCII name) failed inside the launch, after the
    sweep directory was made; it is refused before (#319)."""
    from mushin.workflows import MultiRunMetricsWorkflow

    class W(MultiRunMetricsWorkflow):
        @staticmethod
        def task(job_dir):
            return dict(m=1.0)

    plain, odd = tmp_path / "plain", tmp_path / "a#b"
    plain.mkdir()
    odd.mkdir()
    with pytest.raises(
        ValueError, match=r"target job directory '.*a#b' would not parse"
    ):
        W().run(target_job_dirs=[plain, odd], working_dir=str(tmp_path / "s"))
    assert not (tmp_path / "s").exists()


def test_the_target_job_dir_refusal_names_the_other_axes_too(tmp_path):
    """A target job directory and a swept string are refused in one message,
    so fixing one does not uncover the other on the next run (#319)."""
    from mushin.workflows import MultiRunMetricsWorkflow

    class W(MultiRunMetricsWorkflow):
        @staticmethod
        def task(job_dir, a):
            return dict(m=1.0)

    odd = tmp_path / "a#b"
    odd.mkdir()
    with pytest.raises(ValueError) as err:
        W().run(
            target_job_dirs=[odd],
            a=multirun(["true", "x"]),
            working_dir=str(tmp_path / "s"),
        )
    message = str(err.value)
    assert "`a`: 'true' would reach the task as True." in message
    assert "a#b' would not parse in its job" in message


def test_a_job_dir_parameter_swept_by_hand_gets_the_sweep_advice(tmp_path):
    """Only the axis `target_job_dirs=` builds is worded as a directory: a
    task's own `job_dir` swept with multirun gets the advice for a sweep
    (#319)."""
    from mushin.workflows import MultiRunMetricsWorkflow

    class W(MultiRunMetricsWorkflow):
        @staticmethod
        def task(job_dir):
            return dict(m=1.0)

    with pytest.raises(ValueError) as err:
        W().run(job_dir=multirun(["true", "x"]), dry_run=True)
    message = str(err.value)
    assert "`job_dir`: 'true' would reach the task as True." in message
    assert "target job directory" not in message


def test_a_completed_record_says_whether_its_metrics_file_holds_the_result(tmp_path):
    """A launcher that loses a job's result can take the metrics file for it
    only when the record names that file, written whole (#345)."""
    import json

    class Some(MultiRunMetricsWorkflow):
        @staticmethod
        def task(x):
            return dict(v=1.0) if x == 1 else None

    Some().run(x=multirun([1, 2]), working_dir=tmp_path)
    records = {
        json.loads(p.read_text())["combo"]["x"]: json.loads(p.read_text())
        for p in tmp_path.glob("*/mushin_cell_status.json")
    }
    from mushin._sweep_io import METRICS_FILE, _digest

    written = (tmp_path / "0" / METRICS_FILE, tmp_path / "1" / METRICS_FILE)
    text = next(p.read_text() for p in written if p.exists())
    assert records[1]["metrics"] == _digest(text)  # the file it wrote
    assert "metrics" not in records[2]


def test_a_job_resolves_its_config_once(tmp_path):
    """#353: a resolver that gives a new value each call (a counter) is
    resolved once for the cell's record and its task, so the record holds
    the value the task ran with."""
    import itertools
    import json

    from omegaconf import OmegaConf

    counter = itertools.count()
    OmegaConf.register_new_resolver("mushin_test_count", lambda: f"c{next(counter)}")
    seen = []

    class W(MultiRunMetricsWorkflow):
        @staticmethod
        def task(x, seed):
            seen.append(x)
            return dict(v=float(seed))

    try:
        W().run(
            x=multirun(["${mushin_test_count:}"]),
            seed=multirun([0, 1]),
            working_dir=str(tmp_path),
        )
    finally:
        OmegaConf.clear_resolver("mushin_test_count")
    for cell, value in zip(sorted(tmp_path.glob("[0-9]*")), seen, strict=True):
        record = json.loads((cell / "mushin_cell_status.json").read_text())
        assert record["combo"]["x"] == value


def test_a_jobs_fingerprint_is_its_config_as_composed(tmp_path):
    """The fingerprint is taken before the config is resolved in place, as
    the driver takes it from the composed config: an escaped interpolation
    (resolved in place, it becomes a live one) keeps its fingerprint."""
    import json

    from omegaconf import OmegaConf

    from mushin._resume import resolved_config, resolved_config_hash

    class W(MultiRunMetricsWorkflow):
        @staticmethod
        def task(x, tmpl):
            return dict(v=float(x))

    from hydra_zen import make_config

    W(make_config(tmpl="\\${x}")).run(x=multirun([1]), working_dir=str(tmp_path))
    cell = tmp_path / "0"
    composed = OmegaConf.load(cell / ".hydra" / "config.yaml")
    record = json.loads((cell / "mushin_cell_status.json").read_text())
    assert record["config_hash"] == resolved_config_hash(resolved_config(composed))


def test_a_wrapper_that_does_not_resolve_reads_after_pre_task(tmp_path, monkeypatch):
    """Only a wrapper that resolves the config before it calls (zen's
    default) has it resolved in place; with one that opted out, the task
    resolves its interpolations when it reads them, after `pre_task`."""
    import os

    from hydra_zen import zen

    monkeypatch.setenv("MUSHIN_TEST_PRE", "unset")
    monkeypatch.delenv("MUSHIN_TEST_PRE")
    seen = []

    class W(MultiRunMetricsWorkflow):
        @staticmethod
        def pre_task(x):
            os.environ["MUSHIN_TEST_PRE"] = "from-pre-task"

        @staticmethod
        def task(x, d):
            seen.append(d)
            return dict(v=1.0)

    def lazy(fn):
        return zen(fn, resolve_pre_call=False)

    W().run(
        x=multirun([1]),
        d="${oc.env:MUSHIN_TEST_PRE,default}",
        working_dir=str(tmp_path),
        task_fn_wrapper=lazy,
        pre_task_fn_wrapper=lazy,
    )
    assert seen == ["from-pre-task"]


def test_a_non_resolving_pre_task_runs_before_the_tasks_resolution(
    tmp_path, monkeypatch
):
    """When only the task's wrapper resolves, the config is resolved just
    before the task, after `pre_task`, as that wrapper would (Codex, #385):
    an environment variable `pre_task` sets reaches the task, and the cell's
    record holds what the task got."""
    import json
    import os

    from hydra_zen import zen

    monkeypatch.setenv("MUSHIN_TEST_PRE", "unset")
    monkeypatch.delenv("MUSHIN_TEST_PRE")
    seen = []

    class W(MultiRunMetricsWorkflow):
        @staticmethod
        def pre_task(x):
            os.environ["MUSHIN_TEST_PRE"] = "from-pre-task"

        @staticmethod
        def task(x, d):
            seen.append(d)
            return dict(v=1.0)

    W().run(
        x=multirun([1]),
        d=multirun(["${oc.env:MUSHIN_TEST_PRE,default}"]),
        working_dir=str(tmp_path),
        pre_task_fn_wrapper=lambda fn: zen(fn, resolve_pre_call=False),
    )
    assert seen == ["from-pre-task"]
    record = json.loads((tmp_path / "0" / "mushin_cell_status.json").read_text())
    assert record["combo"]["d"] == "from-pre-task"
