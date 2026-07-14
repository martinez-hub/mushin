# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
"""End-to-end integration test for the sweep-resilience feature.

Exercises a *real* Hydra multirun (``MultiRunMetricsWorkflow``, real launch, a
tmp working dir) through the whole fail-soft -> refuse-stats -> fix -> resume ->
compare loop, rather than unit-testing the pieces in isolation.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from mushin import multirun
from mushin._sweep_io import MANIFEST_FILE, METRICS_FILE, combo_key
from mushin.benchmark import IncompleteSweepError, compare_methods
from mushin.workflows import MultiRunMetricsWorkflow

PROVENANCE_FILE = "mushin_provenance.json"

# The single grid cell that fails on the first pass.
_FAILED_METHOD, _FAILED_SEED = "beta", 1
_BASE = {"alpha": 0.0, "beta": 1.0}


def test_fail_soft_resume_compare_end_to_end(tmp_path):
    calls: dict[str, int] = {"n": 0}

    class Sweep(MultiRunMetricsWorkflow):
        FAIL = True

        @staticmethod
        def task(method, seed):
            calls["n"] += 1
            if Sweep.FAIL and method == _FAILED_METHOD and seed == _FAILED_SEED:
                raise RuntimeError("boom: transient training failure")
            # a per-seed-varying score so the downstream stats test has a real
            # sampling distribution (not a constant that gets masked out).
            return {"score": _BASE[method] + 0.1 * float(seed)}

    working_dir = str(tmp_path / "sweep")
    grid = dict(method=multirun(["alpha", "beta"]), seed=multirun([0, 1, 2]))

    # --- pass 1: fail-soft -------------------------------------------------
    Sweep.FAIL = True
    wf = Sweep()
    with pytest.warns(UserWarning, match="fail"):
        wf.run(working_dir=working_dir, on_error="nan", **grid)

    failed_key = combo_key({"method": _FAILED_METHOD, "seed": _FAILED_SEED})

    # the failed cell is NaN; a completed cell carries its real value
    ds = wf.to_xarray()
    failed_cell = {"method": _FAILED_METHOD, "seed": _FAILED_SEED}
    assert bool(np.isnan(float(ds["score"].sel(failed_cell))))
    assert float(ds["score"].sel({"method": "alpha", "seed": 2})) == pytest.approx(0.2)

    # failures recorded, sweep flagged incomplete, dataset carries the signal
    assert wf.failures, "expected a recorded failure"
    assert any(f["combo"] == failed_key for f in wf.failures)
    assert wf.is_complete is False
    assert ds.attrs["mushin_failures"], "dataset must carry the failure signal"

    # on-disk manifest exists and marks that cell failed
    manifest = json.loads((tmp_path / "sweep" / MANIFEST_FILE).read_text())
    assert manifest["cells"][failed_key]["status"] == "failed"

    # every *completed* job dir has both sidecars; the failed dir has provenance
    # but no metrics (the task raised before writing metrics).
    failed_dirs = {f["working_dir"] for f in wf.failures}
    for d in wf.multirun_working_dirs:
        assert (d / PROVENANCE_FILE).exists(), f"missing provenance in {d}"
        if str(d) in failed_dirs:
            assert not (d / METRICS_FILE).exists()
        else:
            assert (d / METRICS_FILE).exists(), f"missing metrics in {d}"

    # --- stats refuse an incomplete sweep ----------------------------------
    with pytest.raises(IncompleteSweepError):
        compare_methods(ds)

    # --- pass 2: fix the cause and resume ----------------------------------
    Sweep.FAIL = False
    calls["n"] = 0
    wf2 = Sweep()
    wf2.run(working_dir=working_dir, resume=True, **grid)

    # only the previously-failed cell actually re-executed
    assert calls["n"] == 1
    assert wf2.is_complete is True

    ds2 = wf2.to_xarray()
    # cell filled in place, same shape as before, no lingering failure signal
    assert float(ds2["score"].sel(failed_cell)) == pytest.approx(1.1)
    assert ds2.sizes == ds.sizes
    assert "mushin_failures" not in ds2.attrs

    # stats now succeed on the completed dataset
    result = compare_methods(ds2, test="welch")
    assert result is not None
