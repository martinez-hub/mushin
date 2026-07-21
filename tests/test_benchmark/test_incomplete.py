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


def test_json_string_failures_attr_counts_combos_not_characters():
    # to_xarray stores the attr as a JSON string (netCDF attrs cannot hold
    # lists of strings portably); the count must be of combos, not characters.
    import json

    ds = _ds()
    ds.attrs["mushin_failures"] = json.dumps(["a=1,b=2"])
    with pytest.raises(IncompleteSweepError, match=r"1 run\(s\) failed \(a=1,b=2\)"):
        compare_methods(ds)


def test_legacy_scalar_failures_attr_counts_one():
    # A legacy 1-element list attr round-trips netCDF4 as a bare str; it must
    # count as ONE failed combo, not len(str) failures.
    ds = _ds()
    ds.attrs["mushin_failures"] = "a=1,b=2"
    with pytest.raises(IncompleteSweepError, match=r"1 run\(s\) failed"):
        compare_methods(ds)


def test_compare_methods_runs_on_complete_sweep():
    compare_methods(
        _ds()
    )  # no mushin_failures attr -> runs fine (returns/does not raise)


def test_plain_user_dataset_unaffected():
    ds = _ds()  # no mushin attrs at all
    compare_methods(ds)  # must not raise
