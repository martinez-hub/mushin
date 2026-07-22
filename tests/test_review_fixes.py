# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
"""Regression tests for defects found in the adversarial review of the
exploration features (cache_dir path handling, sample+resume, overrides= axis
counting, allow_incomplete NaN handling, show() sort ordering)."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
import xarray as xr

import mushin
from mushin import multirun
from mushin.benchmark import compare_methods
from mushin.workflows import MultiRunMetricsWorkflow


class _W(MultiRunMetricsWorkflow):
    @staticmethod
    def task(x):
        return dict(v=float(x))


# --- 1. cache_dir with a relative / ~ path -----------------------------------


def test_cache_dir_relative_path_reuses_across_working_dirs(tmp_path, monkeypatch):
    # A relative cache_dir must resolve to a single shared location, not be
    # dereferenced inside each per-cell job dir (hydra.job.chdir=True).
    monkeypatch.chdir(tmp_path)
    calls = {"n": 0}

    class W(MultiRunMetricsWorkflow):
        @staticmethod
        def task(x):
            calls["n"] += 1
            return dict(v=float(x))

    W().run(x=multirun([1, 2]), cache_dir="relcache", working_dir="a")
    assert calls["n"] == 2
    calls["n"] = 0
    W().run(x=multirun([1, 2]), cache_dir="relcache", working_dir="b")
    assert calls["n"] == 0  # reused from the relative cache


# --- 2. resume + sample must not silently destroy completed cells ------------


def test_resume_with_sample_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="sample"):
        _W().run(
            x=multirun([1, 2, 3]),
            resume=True,
            sample=2,
            working_dir=str(tmp_path / "s"),
        )


# --- 3. num_cells must count axes supplied via overrides=[...] ----------------


def test_cell_count_gate_counts_overrides_list_axis(tmp_path):
    with pytest.raises(ValueError, match="cell"):
        _W().run(
            overrides=["+x=0,1,2,3"],  # a real 4-way sweep axis
            confirm_above=2,
            working_dir=str(tmp_path / "s"),
        )


def test_dry_run_counts_overrides_list_axis(tmp_path):
    summary = _W().run(
        overrides=["+x=0,1,2,3"], dry_run=True, working_dir=str(tmp_path / "s")
    )
    assert summary["num_cells"] == 4


# --- 4. allow_incomplete must compute over the completed cells ---------------


def test_allow_incomplete_computes_over_completed_cells():
    data = np.array([[1.0, 2.0, 3.0, 4.0, 5.0], [1.4, 2.6, 3.3, 4.5, np.nan]])
    ds = xr.Dataset(
        {"acc": (("method", "seed"), data)},
        coords={"method": ["a", "b"], "seed": [0, 1, 2, 3, 4]},
    )
    ds.attrs["mushin_skipped"] = json.dumps(["method=b,seed=4"])
    with pytest.warns(UserWarning, match="incomplete"):
        out = compare_methods(ds, test="welch", allow_incomplete=True)
    row = out[(out.method_a == "a") & (out.method_b == "b")].iloc[0]
    assert not np.isnan(row["p_value"])  # over the 4 completed seed pairs, not NaN


# --- 5. show() sort ordering -------------------------------------------------


def test_sortable_puts_nan_last():
    from mushin._show import _sortable

    out = sorted([float("nan"), 1.0, 2.0, 3.0], key=_sortable)
    assert out[:3] == [1.0, 2.0, 3.0]
    assert math.isnan(out[3])


def test_show_sort_by_unknown_column_raises(tmp_path):
    wd = tmp_path / "s"
    _W().run(x=multirun([1, 2, 3]), working_dir=str(wd))
    with pytest.raises(ValueError, match="sort"):
        mushin.show(wd, sort="does_not_exist")
