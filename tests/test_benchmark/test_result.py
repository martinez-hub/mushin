from mushin.benchmark._aggregate import to_dataset
from mushin.benchmark._result import BenchmarkResult
from mushin.benchmark._stats import compare_methods


def _result():
    results = {
        "ours": [{"accuracy": 0.90}, {"accuracy": 0.91}, {"accuracy": 0.92}],
        "base": [{"accuracy": 0.70}, {"accuracy": 0.71}, {"accuracy": 0.72}],
    }
    ds = to_dataset(results)
    comparisons = compare_methods(ds, test="welch", alpha=0.05)
    return BenchmarkResult(data=ds, comparisons=comparisons, alpha=0.05)


def test_summary_has_row_per_method_metric():
    summary = _result().summary()
    assert len(summary) == 2
    assert set(summary.columns) >= {
        "method",
        "metric",
        "mean",
        "ci_low",
        "ci_high",
        "significant_vs_ref",
    }


def test_summary_marks_significant_against_reference():
    summary = _result().summary()  # reference defaults to first method ("ours")
    base_row = summary[summary["method"] == "base"].iloc[0]
    assert base_row["significant_vs_ref"] == "*"


def test_summary_rejects_unknown_reference():
    """A typo'd reference must raise, not silently blank every significance
    marker in a publication-ready table."""
    import pandas as pd
    import pytest
    import xarray as xr

    from mushin.benchmark._result import BenchmarkResult

    ds = xr.Dataset(
        {"acc": (("method", "seed"), [[0.9, 0.8], [0.7, 0.6]])},
        coords={"method": ["a", "b"], "seed": [0, 1]},
    )
    comps = pd.DataFrame(
        [
            {
                "metric": "acc",
                "method_a": "a",
                "method_b": "b",
                "significant": True,
            }
        ]
    )
    res = BenchmarkResult(data=ds, comparisons=comps, alpha=0.05)
    with pytest.raises(ValueError, match="TYPO"):
        res.summary(reference="TYPO")
