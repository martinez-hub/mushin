import numpy as np
import pytest
import xarray as xr

from mushin.benchmark import compare_methods
from mushin.benchmark._stats import IncompleteSweepError


def _ds():
    return xr.Dataset(
        {"acc": (("method", "seed"), np.random.RandomState(0).rand(2, 3))},
        coords={"method": ["a", "b"], "seed": [0, 1, 2]},
    )


def test_compare_methods_refuses_incomplete_sweep():
    ds = _ds()
    ds.attrs["mushin_failures"] = ["method=a,seed=1"]
    with pytest.raises(IncompleteSweepError, match="fail"):
        compare_methods(ds)


def test_compare_methods_runs_on_complete_sweep():
    compare_methods(
        _ds()
    )  # no mushin_failures attr -> runs fine (returns/does not raise)


def test_plain_user_dataset_unaffected():
    ds = _ds()  # no mushin attrs at all
    compare_methods(ds)  # must not raise
