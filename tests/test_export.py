# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
"""`mushin.export.table(root)` writes a dependency-free CSV of a sweep — one row
per cell (swept params, status, metrics) — for pandas/spreadsheet users."""

from __future__ import annotations

import csv
import io
from pathlib import Path

import pytest

import mushin
from mushin import multirun
from mushin.workflows import MultiRunMetricsWorkflow


def _run(wd):
    class W(MultiRunMetricsWorkflow):
        @staticmethod
        def task(lr, seed):
            return dict(acc=float(lr) + float(seed), loss=float(seed))

    W().run(lr=multirun([0.1, 0.2]), seed=multirun([0, 1]), working_dir=str(wd))


def _parse(text):
    return list(csv.DictReader(io.StringIO(text)))


def test_export_table_returns_csv_string(tmp_path):
    wd = tmp_path / "s"
    _run(wd)
    text = mushin.export.table(wd)
    rows = _parse(text)
    assert len(rows) == 4
    header = set(rows[0].keys())
    assert {"lr", "seed", "status", "acc", "loss"} <= header
    r = next(row for row in rows if row["lr"] == "0.2" and row["seed"] == "1")
    assert r["status"] == "completed"
    assert float(r["acc"]) == pytest.approx(1.2)


def test_export_table_writes_file_and_returns_path(tmp_path):
    wd = tmp_path / "s"
    _run(wd)
    out = tmp_path / "table.csv"
    ret = mushin.export.table(wd, path=out)
    assert Path(ret) == out
    assert out.exists()
    rows = _parse(out.read_text())
    assert len(rows) == 4
    assert {"lr", "seed", "status", "acc", "loss"} <= set(rows[0].keys())


def test_export_table_metrics_filter(tmp_path):
    wd = tmp_path / "s"
    _run(wd)
    rows = _parse(mushin.export.table(wd, metrics=["acc"]))
    header = set(rows[0].keys())
    assert "acc" in header
    assert "loss" not in header


def test_export_table_missing_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        mushin.export.table(tmp_path / "nope")
