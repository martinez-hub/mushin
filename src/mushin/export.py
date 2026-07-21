# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
"""Dependency-free exports of a sweep directory. ``mushin.export.table`` renders
the per-cell sidecars (swept params, status, metrics) as CSV — a durable,
pandas-/spreadsheet-friendly substrate that needs neither Hydra nor xarray."""

from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Any

from ._show import _fmt, _ordered_union, _read_cells, _sortable


def _csv_cell(v: Any):
    """A CSV-friendly value: scalars pass through (so pandas parses numbers
    numerically), None becomes empty, a length-1 list is unwrapped, and any other
    non-scalar is rendered compactly."""
    if v is None:
        return ""
    if isinstance(v, bool | int | float | str):
        return v
    if isinstance(v, list | tuple) and len(v) == 1:
        return _csv_cell(v[0])
    return _fmt(v)


def table(root, *, path=None, metrics: list[str] | None = None):
    """Export a sweep's cells as CSV: one row per cell with its swept params,
    ``status``, and metrics.

    Parameters
    ----------
    root :
        A sweep ``working_dir``.
    path :
        If given, write the CSV there and return the ``Path``. Otherwise return
        the CSV as a string.
    metrics : list[str] | None
        Restrict the metric columns to these names (default: every metric found).

    Raises
    ------
    FileNotFoundError
        If ``root`` is not a directory.
    """
    cells = _read_cells(root)

    param_cols = _ordered_union([c["combo"] for c in cells])
    metric_cols = _ordered_union([c["metrics"] for c in cells])
    if metrics is not None:
        metric_cols = [m for m in metric_cols if m in set(metrics)]
    columns = [*param_cols, "status", *metric_cols]

    rows = []
    for c in cells:
        row: dict[str, Any] = {p: c["combo"].get(p) for p in param_cols}
        row["status"] = c["status"]
        for m in metric_cols:
            row[m] = c["metrics"].get(m)
        rows.append(row)
    if param_cols:
        rows.sort(key=lambda r: tuple(_sortable(r.get(p)) for p in param_cols))

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(columns)
    for row in rows:
        writer.writerow([_csv_cell(row.get(col)) for col in columns])
    text = buf.getvalue()

    if path is not None:
        path = Path(path)
        path.write_text(text, newline="")
        return path
    return text
