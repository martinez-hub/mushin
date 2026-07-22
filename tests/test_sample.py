# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
"""`run(..., sample=K)` runs a deterministic random K-cell subset of the grid
(the rest skipped → NaN) for fast exploration. The selection is reproducible and
resume-safe: resuming without `sample` fills in the remaining cells."""

from __future__ import annotations

import numpy as np
import pytest

from mushin import multirun
from mushin.workflows import MultiRunMetricsWorkflow


def test_sample_runs_only_k_cells(tmp_path):
    calls = []

    class W(MultiRunMetricsWorkflow):
        @staticmethod
        def task(a, b):
            calls.append((a, b))
            return dict(v=float(a) + float(b))

    wf = W()
    with pytest.warns(UserWarning, match="skip"):
        wf.run(
            a=multirun([0, 1, 2, 3]),
            b=multirun([0, 1, 2, 3]),  # 16 cells
            sample=5,
            working_dir=str(tmp_path / "s"),
        )
    assert len(calls) == 5  # only 5 of 16 ran
    assert wf.is_complete is False
    assert len(wf.skipped) == 11
    ds = wf.to_xarray()
    assert int(np.isnan(ds["v"].values).sum()) == 11


def test_sample_selection_is_deterministic(tmp_path):
    def ran_cells(wd):
        calls = []

        class W(MultiRunMetricsWorkflow):
            @staticmethod
            def task(a, b):
                calls.append((a, b))
                return dict(v=float(a))

        with pytest.warns(UserWarning, match="skip"):
            W().run(
                a=multirun([0, 1, 2, 3]),
                b=multirun([0, 1, 2, 3]),
                sample=5,
                working_dir=str(wd),
            )
        return set(calls)

    assert ran_cells(tmp_path / "s1") == ran_cells(tmp_path / "s2")


def test_sample_seed_varies_the_subset(tmp_path):
    def ran_cells(wd, seed):
        calls = []

        class W(MultiRunMetricsWorkflow):
            @staticmethod
            def task(a, b):
                calls.append((a, b))
                return dict(v=float(a))

        with pytest.warns(UserWarning, match="skip"):
            W().run(
                a=multirun([0, 1, 2, 3]),
                b=multirun([0, 1, 2, 3]),
                sample=5,
                sample_seed=seed,
                working_dir=str(wd),
            )
        return set(calls)

    assert ran_cells(tmp_path / "s1", 0) != ran_cells(tmp_path / "s2", 1)


def test_sample_at_least_grid_runs_all(tmp_path):
    calls = []

    class W(MultiRunMetricsWorkflow):
        @staticmethod
        def task(a):
            calls.append(a)
            return dict(v=float(a))

    wf = W()
    wf.run(a=multirun([0, 1, 2, 3]), sample=100, working_dir=str(tmp_path / "s"))
    assert len(calls) == 4  # sample >= grid -> run everything
    assert wf.is_complete


def test_sample_nonpositive_rejected(tmp_path):
    class W(MultiRunMetricsWorkflow):
        @staticmethod
        def task(a):
            return dict(v=float(a))

    with pytest.raises(ValueError, match="sample"):
        W().run(a=multirun([0, 1]), sample=0, working_dir=str(tmp_path / "s"))


def test_sample_then_resume_fills_the_rest(tmp_path):
    class W(MultiRunMetricsWorkflow):
        ran: list = []

        @staticmethod
        def task(a, b):
            W.ran.append((a, b))
            return dict(v=float(a) + float(b))

    wd = str(tmp_path / "s")
    with pytest.warns(UserWarning, match="skip"):
        W().run(
            a=multirun([0, 1, 2, 3]),
            b=multirun([0, 1]),  # 8 cells
            sample=3,
            working_dir=wd,
        )
    assert len(W.ran) == 3

    W.ran.clear()
    wf2 = W()
    wf2.run(a=multirun([0, 1, 2, 3]), b=multirun([0, 1]), resume=True, working_dir=wd)
    assert len(W.ran) == 5  # the 5 previously-skipped cells ran; the 3 done were reused
    assert wf2.is_complete
