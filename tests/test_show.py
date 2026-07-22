# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
"""`mushin.show(root)` reads a sweep directory's per-cell sidecars and returns a
status/metrics table — offline, dependency-free, and readable mid-sweep."""

from __future__ import annotations

import pytest

import mushin
from mushin import multirun
from mushin.workflows import MultiRunMetricsWorkflow


def _run_sweep(wd):
    class W(MultiRunMetricsWorkflow):
        @staticmethod
        def task(lr, seed):
            return dict(acc=float(lr) + float(seed), loss=float(seed))

    W().run(lr=multirun([0.1, 0.2]), seed=multirun([0, 1]), working_dir=str(wd))


def test_show_returns_one_row_per_cell(tmp_path):
    wd = tmp_path / "s"
    _run_sweep(wd)
    res = mushin.show(wd)
    assert len(res.rows) == 4  # 2 x 2 grid
    # every row carries its swept params, a status, and its metrics
    r = next(row for row in res.rows if row["lr"] == 0.2 and row["seed"] == 1)
    assert r["status"] == "completed"
    assert r["acc"] == pytest.approx(1.2)
    assert r["loss"] == pytest.approx(1.0)


def test_show_table_lists_columns_and_values(tmp_path):
    wd = tmp_path / "s"
    _run_sweep(wd)
    table = str(mushin.show(wd))
    for col in ("lr", "seed", "status", "acc", "loss"):
        assert col in table
    assert "completed" in table


def test_show_reports_incomplete_cells_mid_sweep(tmp_path):
    # A cell whose status sidecar says "running" (no metrics yet) must appear
    # with that status and blank metrics — the point of watching a live sweep.
    from mushin._resume import write_cell_status

    wd = tmp_path / "s"
    _run_sweep(wd)
    # simulate a still-running cell by overwriting one cell's status
    running_dir = next(
        d
        for d in wd.iterdir()
        if d.is_dir() and (d / "mushin_cell_status.json").exists()
    )
    import json

    combo = json.loads((running_dir / "mushin_cell_status.json").read_text())["combo"]
    write_cell_status(running_dir, status="running", combo=combo, attempt=1)

    res = mushin.show(wd)
    running = [r for r in res.rows if r["status"] == "running"]
    assert len(running) == 1


def test_show_metrics_filter_limits_columns(tmp_path):
    wd = tmp_path / "s"
    _run_sweep(wd)
    res = mushin.show(wd, metrics=["acc"])
    table = str(res)
    assert "acc" in table
    assert "loss" not in table
    # rows still carry the swept params and status
    assert all("lr" in r and "status" in r for r in res.rows)


def test_show_missing_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        mushin.show(tmp_path / "nope")


def test_show_empty_dir_returns_no_rows(tmp_path):
    (tmp_path / "empty").mkdir()
    res = mushin.show(tmp_path / "empty")
    assert res.rows == []


def test_param_and_metric_sharing_a_name_do_not_collide(tmp_path):
    """A task reporting a metric named like a swept param (e.g. the effective
    `lr`) must not overwrite the param column — both values must survive."""
    from mushin import export

    class W(MultiRunMetricsWorkflow):
        @staticmethod
        def task(lr):
            return dict(lr=float(lr) * 2, loss=1.0)  # effective lr, a common pattern

    wd = tmp_path / "s"
    W().run(lr=multirun([0.001, 0.01]), working_dir=str(wd))

    rows = mushin.show(wd).rows
    r = next(row for row in rows if row["lr"] == 0.001)  # swept value intact
    assert r["lr (metric)"] == pytest.approx(0.002)

    csv_lines = export.table(wd).strip().splitlines()
    header = csv_lines[0].split(",")
    assert header.count("lr") == 1
    assert "lr (metric)" in header


def test_unknown_metric_filter_raises(tmp_path):
    """A typo'd metrics= entry must raise (like sort= does), not silently drop
    the column; and the requested order must be honored."""
    from mushin import export

    wd = tmp_path / "s"
    _run_sweep(wd)

    with pytest.raises(ValueError, match="wrong_metric"):
        mushin.show(wd, metrics=["wrong_metric"])
    with pytest.raises(ValueError, match="wrong_metric"):
        export.table(wd, metrics=["wrong_metric"])

    res = mushin.show(wd, metrics=["loss", "acc"])
    cols = res.table.splitlines()[0].split()
    assert cols.index("loss") < cols.index("acc")  # requested order wins


def test_metrics_filter_deduplicates_repeated_names(tmp_path):
    """A repeated metrics= entry must not emit a duplicated table/CSV column
    (a duplicate CSV header makes pandas mangle it to acc/acc.1)."""
    from mushin import export

    wd = tmp_path / "s"
    _run_sweep(wd)

    cols = mushin.show(wd, metrics=["acc", "acc"]).table.splitlines()[0].split()
    assert cols.count("acc") == 1

    header = export.table(wd, metrics=["acc", "acc"]).strip().splitlines()[0]
    assert header.split(",").count("acc") == 1
