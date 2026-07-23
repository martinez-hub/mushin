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


def test_offline_tools_scope_to_latest_sweep_grid(tmp_path):
    """Re-running a narrowed grid in the same working_dir leaves stale cell
    dirs behind; show/best/export must scope to the LATEST sweep's manifest,
    not resurrect cells that are no longer part of the grid."""

    from mushin import export

    class W(MultiRunMetricsWorkflow):
        @staticmethod
        def task(lr):
            return dict(acc=float(lr))

    wd = tmp_path / "s"
    W().run(lr=multirun([0.001, 0.01, 0.1]), working_dir=str(wd))
    W().run(lr=multirun([0.001, 0.01]), working_dir=str(wd))  # narrowed re-run

    res = mushin.show(wd)
    assert len(res.rows) == 2
    assert all(r["lr"] != 0.1 for r in res.rows)

    b = mushin.best(wd, "acc")
    assert b.combo == {"lr": 0.01}  # NOT the stale lr=0.1 cell

    csv_text = export.table(wd)
    assert len(csv_text.strip().splitlines()) == 1 + 2  # header + 2 cells


def test_offline_tools_ignore_duplicate_dirs_not_in_manifest(tmp_path):
    """When two dirs claim the same combo, the manifest's assignment wins —
    the winner must not depend on filesystem iteration order."""
    import shutil

    class W(MultiRunMetricsWorkflow):
        @staticmethod
        def task(lr):
            return dict(acc=float(lr))

    wd = tmp_path / "s"
    W().run(lr=multirun([0.001, 0.01]), working_dir=str(wd))
    shutil.copytree(wd / "0", wd / "9")  # duplicate claimant for dir 0's combo
    # give the impostor DIFFERENT metrics, so the assertion below proves the
    # manifest's dir won (not merely that a duplicate was dropped)
    (wd / "9" / "mushin_metrics.json").write_text('{"acc": 999.0}')

    res = mushin.show(wd)
    assert len(res.rows) == 2  # the copy is not double-counted
    assert all(r["acc"] != 999.0 for r in res.rows)  # manifest's dir wins


def test_show_keeps_cells_of_a_newer_inflight_sweep(tmp_path):
    """Mid-sweep view of a NEW grid in a reused dir: a cell whose sidecar is
    newer than the (previous sweep's) manifest belongs to the in-flight sweep
    and must be shown even though the stale manifest doesn't list it."""
    from mushin._resume import write_cell_status

    class W(MultiRunMetricsWorkflow):
        @staticmethod
        def task(lr):
            return dict(acc=float(lr))

    wd = tmp_path / "s"
    W().run(lr=multirun([0.001, 0.01]), working_dir=str(wd))

    d = wd / "7"
    d.mkdir()
    write_cell_status(d, status="running", combo={"lr": 0.5}, attempt=1)
    # Force the sidecar mtime past the manifest's, so the test doesn't depend
    # on sub-second filesystem mtime resolution.
    import os

    manifest_mtime = (wd / "mushin_sweep_manifest.json").stat().st_mtime
    os.utime(d / "mushin_cell_status.json", (manifest_mtime + 5, manifest_mtime + 5))

    rows = mushin.show(wd).rows
    assert any(r["lr"] == 0.5 and r["status"] == "running" for r in rows)


def test_offline_tools_survive_wrong_shape_manifest(tmp_path):
    """A valid-JSON manifest of the wrong shape (list payload, or non-dict
    cells) must degrade to the unscoped scan, not crash show/best/export."""
    wd = tmp_path / "s"
    _run_sweep(wd)

    for payload in ("[]", '{"cells": []}', '{"cells": {"k": 5}}'):
        (wd / "mushin_sweep_manifest.json").write_text(payload)
        assert len(mushin.show(wd).rows) == 4
        assert mushin.best(wd, "acc").combo == {"lr": 0.2, "seed": 1}
