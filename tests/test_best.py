# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
"""`mushin.best(root, metric)` returns the winning cell of a sweep — offline,
reading the same per-cell sidecars as `mushin.show`."""

from __future__ import annotations

from pathlib import Path

import pytest

import mushin
from mushin import multirun
from mushin.workflows import MultiRunMetricsWorkflow


def _run(wd, *, on_error="raise"):
    class W(MultiRunMetricsWorkflow):
        @staticmethod
        def task(x):
            if on_error == "nan" and x == 5.0:
                raise RuntimeError("boom")
            return dict(score=float(x), other=float(-x))

    W().run(x=multirun([1.0, 5.0, 3.0]), working_dir=str(wd), on_error=on_error)


def test_best_returns_max_cell_by_default(tmp_path):
    wd = tmp_path / "s"
    _run(wd)
    r = mushin.best(wd, "score")
    assert r.combo == {"x": 5.0}
    assert r.value == pytest.approx(5.0)
    assert r.status == "completed"
    assert r.metrics["score"] == pytest.approx(5.0)
    assert Path(r.dir).is_dir()


def test_best_mode_min(tmp_path):
    wd = tmp_path / "s"
    _run(wd)
    r = mushin.best(wd, "score", mode="min")
    assert r.combo == {"x": 1.0}
    assert r.value == pytest.approx(1.0)


def test_best_optimizes_the_named_metric(tmp_path):
    wd = tmp_path / "s"
    _run(wd)
    # `other` == -x, so its max is at x == 1.0
    r = mushin.best(wd, "other")
    assert r.combo == {"x": 1.0}
    assert r.value == pytest.approx(-1.0)


def test_best_ignores_failed_cells(tmp_path):
    # With x==5 failing (NaN), the best completed score is at x==3.
    wd = tmp_path / "s"
    with pytest.warns(UserWarning, match="fail"):
        _run(wd, on_error="nan")
    r = mushin.best(wd, "score")
    assert r.combo == {"x": 3.0}
    assert r.value == pytest.approx(3.0)


def test_best_unknown_metric_raises_with_available(tmp_path):
    wd = tmp_path / "s"
    _run(wd)
    with pytest.raises(ValueError, match="score") as exc:
        mushin.best(wd, "nope")
    assert "nope" in str(exc.value)


def test_best_invalid_mode_raises(tmp_path):
    wd = tmp_path / "s"
    _run(wd)
    with pytest.raises(ValueError, match="mode"):
        mushin.best(wd, "score", mode="highest")


def test_best_missing_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        mushin.best(tmp_path / "nope", "score")
