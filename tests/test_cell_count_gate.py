# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
"""A pre-launch cell-count gate: `confirm_above=` (per call) and the
`MUSHIN_MAX_CELLS` environment default refuse an over-large sweep before it
launches, so an accidentally huge grid fails fast instead of burning compute."""

from __future__ import annotations

import pytest

import mushin
from mushin import multirun
from mushin.workflows import MultiRunMetricsWorkflow


def _counting_workflow(counter):
    class W(MultiRunMetricsWorkflow):
        @staticmethod
        def task(lr=0.1, seed=0):
            counter["n"] += 1
            return dict(v=float(lr))

    return W


def test_confirm_above_blocks_oversized_sweep(tmp_path):
    counter = {"n": 0}
    wf = _counting_workflow(counter)()
    with pytest.raises(ValueError, match="cell") as exc:
        wf.run(
            lr=multirun([0.1, 0.2, 0.3]),
            seed=multirun([0, 1]),  # 6 cells
            confirm_above=4,
            working_dir=str(tmp_path / "s"),
        )
    assert counter["n"] == 0  # nothing launched
    assert "6" in str(exc.value) and "4" in str(exc.value)


def test_confirm_above_allows_sweep_at_or_below_limit(tmp_path):
    counter = {"n": 0}
    wf = _counting_workflow(counter)()
    wf.run(
        lr=multirun([0.1, 0.2, 0.3]),
        seed=multirun([0, 1]),  # 6 cells
        confirm_above=6,
        working_dir=str(tmp_path / "s"),
    )
    assert counter["n"] == 6


def test_max_cells_env_blocks_oversized_sweep(tmp_path, monkeypatch):
    monkeypatch.setenv("MUSHIN_MAX_CELLS", "4")
    counter = {"n": 0}
    wf = _counting_workflow(counter)()
    with pytest.raises(ValueError, match="MUSHIN_MAX_CELLS"):
        wf.run(
            lr=multirun([0.1, 0.2, 0.3]),
            seed=multirun([0, 1]),  # 6 cells
            working_dir=str(tmp_path / "s"),
        )
    assert counter["n"] == 0


def test_explicit_confirm_above_overrides_env(tmp_path, monkeypatch):
    # An explicit confirm_above wins over the environment default (both ways).
    monkeypatch.setenv("MUSHIN_MAX_CELLS", "2")
    counter = {"n": 0}
    wf = _counting_workflow(counter)()
    wf.run(
        lr=multirun([0.1, 0.2, 0.3]),
        seed=multirun([0, 1]),  # 6 cells
        confirm_above=100,  # explicit, roomy -> runs despite env=2
        working_dir=str(tmp_path / "s"),
    )
    assert counter["n"] == 6


def test_malformed_env_is_ignored(tmp_path, monkeypatch):
    monkeypatch.setenv("MUSHIN_MAX_CELLS", "not-a-number")
    counter = {"n": 0}
    wf = _counting_workflow(counter)()
    wf.run(
        lr=multirun([0.1, 0.2]),
        working_dir=str(tmp_path / "s"),
    )
    assert counter["n"] == 2  # malformed env -> no gate


def test_dry_run_bypasses_the_gate(tmp_path):
    # dry_run is exactly how you'd preview an over-limit sweep, so the gate must
    # not block it.
    counter = {"n": 0}
    wf = _counting_workflow(counter)()
    summary = wf.run(
        lr=multirun([0.1, 0.2, 0.3]),
        seed=multirun([0, 1]),  # 6 cells
        confirm_above=1,
        dry_run=True,
        working_dir=str(tmp_path / "s"),
    )
    assert counter["n"] == 0
    assert summary["num_cells"] == 6


def test_gate_applies_to_decorator_sweep(tmp_path):
    counter = {"n": 0}

    @mushin.sweep
    def exp(lr=0.1, seed=0):
        counter["n"] += 1
        return dict(v=float(lr))

    with pytest.raises(ValueError, match="cell"):
        exp.run(
            lr=multirun([0.1, 0.2, 0.3]),
            seed=multirun([0, 1]),  # 6 cells
            confirm_above=4,
            working_dir=str(tmp_path / "s"),
        )
    assert counter["n"] == 0


def test_gate_uses_sampled_cell_count(tmp_path):
    """`sample=` bounds what actually computes, so the cell-count gate must
    compare the SAMPLED count against the limit, not the full grid."""
    from mushin import multirun
    from mushin.workflows import MultiRunMetricsWorkflow

    class W(MultiRunMetricsWorkflow):
        @staticmethod
        def task(a):
            return dict(m=float(a))

    wf = W()
    with pytest.warns(UserWarning, match="skipped"):
        wf.run(
            a=multirun([1, 2, 3, 4, 5, 6]),
            sample=2,
            confirm_above=3,
            working_dir=str(tmp_path / "s"),
        )
    assert len(wf.skipped) == 4
