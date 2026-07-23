"""Regression tests for metric-value coercion into the JSON sidecar.

Pins fixes for the value-coercion divergence: the sidecar normalizer must
recurse into nested dicts (a nested numpy value used to crash the write *after*
the task had already succeeded), non-finite floats must serialize as valid JSON
and round-trip back to NaN, and a non-JSON-native metric value must be
stringified rather than abort the sweep.
"""

import json
import math

import numpy as np
import pytest

from mushin import multirun
from mushin._sweep_io import read_metrics_sidecar, write_metrics_sidecar
from mushin.workflows import MultiRunMetricsWorkflow


def test_nested_numpy_metric_serializes_and_roundtrips(tmp_path):
    metrics = {
        "acc": np.float64(0.9),
        "counts": {"tp": np.int64(5), "row": np.array([1, 2, 3])},
        "seq": [np.float32(1.5), np.int32(2)],
    }
    write_metrics_sidecar(tmp_path, metrics)  # must not raise
    back = read_metrics_sidecar(tmp_path)
    assert back["acc"] == 0.9
    assert back["counts"] == {"tp": 5, "row": [1, 2, 3]}
    assert back["seq"] == [1.5, 2]
    # on-disk file is strict, valid JSON
    json.loads((tmp_path / "mushin_metrics.json").read_text())


def test_nonfinite_metric_is_valid_json_and_restores_nonfinite(tmp_path):
    write_metrics_sidecar(tmp_path, {"loss": float("nan"), "grad": float("inf")})
    raw = (tmp_path / "mushin_metrics.json").read_text()
    assert "NaN" not in raw and "Infinity" not in raw  # not python-only literals
    json.loads(raw)  # strict parse succeeds
    back = read_metrics_sidecar(tmp_path)
    assert math.isnan(back["loss"])
    assert back["grad"] == math.inf  # ±Inf keeps its sign (no NaN collapse)


def test_nonnative_metric_value_is_stringified_not_crashing(tmp_path):
    import datetime

    write_metrics_sidecar(tmp_path, {"when": datetime.datetime(2026, 1, 1)})  # no crash
    back = read_metrics_sidecar(tmp_path)
    assert isinstance(back["when"], str) and "2026" in back["when"]


def test_task_with_nested_numpy_metric_completes_the_sweep(tmp_path):
    # The end-to-end failure: a SUCCESSFUL task returning a nested-numpy metric
    # used to crash the sidecar write, leaving the cell 'running' and aborting
    # (or, under on_error='nan', silently failing) an otherwise-fine sweep.
    class W(MultiRunMetricsWorkflow):
        @staticmethod
        def task(a):
            return dict(v=float(a), detail={"tp": np.int64(a)})

    wf = W()
    wf.run(a=multirun([1, 2]), working_dir=str(tmp_path / "s"))
    assert wf.is_complete
    ds = wf.to_xarray()
    assert float(ds["v"].sel(a=2)) == 2.0


def test_as_float_rejects_nonscalar_numpy_with_clear_message():
    from mushin.benchmark._inference import _as_float

    with pytest.raises(TypeError, match="battery metrics must return scalar"):
        _as_float(np.array([1.0, 2.0]))
