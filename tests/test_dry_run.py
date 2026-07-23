# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
"""`dry_run=True` previews a sweep — cell count and per-axis values — and
returns without launching a single job, so a range typo (`lr=multirun(range(
100))`) is caught before burning compute."""

from __future__ import annotations

import mushin
from mushin import multirun
from mushin.workflows import MultiRunMetricsWorkflow, _format_dry_run


def test_dry_run_returns_summary_without_launching(tmp_path):
    calls = {"n": 0}

    class W(MultiRunMetricsWorkflow):
        @staticmethod
        def task(lr=0.1, seed=0, epochs=1):
            calls["n"] += 1
            return dict(v=float(lr))

    wf = W()
    wd = tmp_path / "s"
    summary = wf.run(
        lr=multirun([0.1, 0.2, 0.3]),
        seed=multirun([0, 1]),
        epochs=10,
        working_dir=str(wd),
        dry_run=True,
    )
    assert calls["n"] == 0  # nothing ran
    assert summary["num_cells"] == 6  # 3 x 2
    assert summary["axes"] == {"lr": [0.1, 0.2, 0.3], "seed": [0, 1]}
    assert summary["fixed"]["epochs"] == 10
    assert not wd.exists()  # no output directory created


def test_dry_run_single_cell_when_no_axes(tmp_path):
    class W(MultiRunMetricsWorkflow):
        @staticmethod
        def task(epochs=1):
            return dict(v=1.0)

    summary = W().run(epochs=5, working_dir=str(tmp_path / "s"), dry_run=True)
    assert summary["num_cells"] == 1
    assert summary["axes"] == {}
    assert summary["fixed"]["epochs"] == 5


def test_dry_run_decorator_returns_summary(tmp_path):
    calls = {"n": 0}

    @mushin.sweep
    def exp(lr=0.1, seed=0):
        calls["n"] += 1
        return dict(v=float(lr))

    summary = exp.run(
        lr=multirun([0.1, 0.2]),
        seed=multirun([0, 1, 2]),
        working_dir=str(tmp_path / "s"),
        dry_run=True,
    )
    assert calls["n"] == 0
    assert summary["num_cells"] == 6


def test_format_dry_run_lists_cells_axes_and_fixed():
    text = _format_dry_run(
        {
            "num_cells": 6,
            "axes": {"lr": [0.1, 0.2, 0.3], "seed": [0, 1]},
            "fixed": {"epochs": 10},
            "working_dir": "/tmp/x",
        }
    )
    assert "6" in text  # cell count
    assert "lr" in text and "0.1" in text  # an axis and its values
    assert "seed" in text
    assert "epochs" in text  # fixed value


def test_user_param_starting_with_hydra_counts_in_grid(tmp_path):
    """A user axis whose name merely STARTS with 'hydra' (e.g. `hydraulic`)
    must count toward the grid — only real `hydra.*`/`hydra/*` plumbing is
    excluded from the preview/gate/sample cell count."""
    from mushin import multirun
    from mushin.workflows import MultiRunMetricsWorkflow

    class W(MultiRunMetricsWorkflow):
        @staticmethod
        def task(hydraulic, a):
            return dict(m=1.0)

    s = W().run(dry_run=True, hydraulic=multirun([1, 2, 3]), a=multirun([1, 2]))
    assert s["num_cells"] == 6
    assert set(s["axes"]) == {"hydraulic", "a"}


def test_dry_run_reports_sample(tmp_path):
    """With `sample=` set, the preview must say how many cells will actually
    compute, not just the full grid size."""
    from mushin import multirun
    from mushin.workflows import MultiRunMetricsWorkflow

    class W(MultiRunMetricsWorkflow):
        @staticmethod
        def task(a):
            return dict(m=1.0)

    s = W().run(dry_run=True, sample=2, a=multirun([1, 2, 3]))
    assert s["num_cells"] == 3
    assert s["sample"] == 2
