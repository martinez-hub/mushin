# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
"""`mushin.show`: a dependency-free, offline view of a sweep directory.

Reads each cell's status sidecar (its swept-param ``combo`` and ``status``) and
metrics sidecar directly — pure JSON, so it works mid-sweep and needs neither
Hydra nor xarray. Handy for watching a live sweep or eyeballing a finished one
before committing to the full ``to_xarray`` load."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _ordered_union(dicts: list[dict]) -> list[str]:
    """Keys across ``dicts`` in first-seen order (preserves sweep-axis and
    metric ordering rather than sorting alphabetically)."""
    seen: list[str] = []
    for d in dicts:
        for k in d:
            if k not in seen:
                seen.append(k)
    return seen


def _fmt(v: Any) -> str:
    """Render one table cell compactly."""
    if v is None:
        return ""
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, float):
        return "nan" if v != v else f"{v:.6g}"
    if isinstance(v, int | str):
        return str(v)
    if isinstance(v, list | tuple):
        return _fmt(v[0]) if len(v) == 1 else f"[{len(v)} values]"
    return str(v)


def _numeric_dir_key(name: str):
    """Sort job dirs by their numeric name when possible, else lexicographically."""
    try:
        return (0, int(name))
    except ValueError:
        return (1, name)


def _read_cells(root) -> list[dict]:
    """Scan a sweep ``root`` and return one dict per cell:
    ``{"combo", "status", "metrics", "dir"}`` (``dir`` is the job dir Path).
    Reads only the per-cell JSON sidecars — no Hydra/xarray. Raises
    ``FileNotFoundError`` if ``root`` is not a directory."""
    from ._resume import read_cell_status
    from ._sweep_io import read_metrics_sidecar

    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"sweep directory not found: {root}")

    cells: list[dict] = []
    for d in sorted(
        (p for p in root.iterdir() if p.is_dir()),
        key=lambda p: _numeric_dir_key(p.name),
    ):
        s = read_cell_status(d)
        if s is None or not isinstance(s.get("combo"), dict):
            continue
        m = read_metrics_sidecar(d) or {}
        if not isinstance(m, dict):
            m = {}
        cells.append(
            {
                "combo": s["combo"],
                "status": s.get("status", "pending"),
                "metrics": m,
                "dir": d,
            }
        )
    return cells


class ShowResult:
    """The result of :func:`show`: ``rows`` (one dict per cell, with raw swept
    params, ``status``, and metric values) plus a rendered ``table``. ``str()``
    returns the table."""

    def __init__(self, rows: list[dict], table: str):
        self.rows = rows
        self.table = table

    def __str__(self) -> str:
        return self.table

    def __repr__(self) -> str:
        return self.table

    def __len__(self) -> int:
        return len(self.rows)

    def __iter__(self):
        return iter(self.rows)


def show(
    root, *, metrics: list[str] | None = None, sort: str | None = None
) -> ShowResult:
    """Read the sweep under ``root`` and return a status/metrics table.

    Parameters
    ----------
    root :
        A sweep ``working_dir`` (the parent of the numeric per-cell job dirs).
    metrics : list[str] | None
        Restrict the metric columns to these names (default: every metric found).
    sort : str | None
        Column to sort rows by (a swept param, ``"status"``, or a metric).
        Defaults to the swept params in grid order.

    Returns
    -------
    ShowResult
        ``.rows`` (one dict per cell) and ``.table`` (the rendered string); the
        table is also printed.
    """
    cells = _read_cells(root)

    param_cols = _ordered_union([c["combo"] for c in cells])
    metric_cols = _ordered_union([c["metrics"] for c in cells])
    if metrics is not None:
        metric_cols = [m for m in metric_cols if m in set(metrics)]

    rows: list[dict] = []
    for c in cells:
        row: dict[str, Any] = {p: c["combo"].get(p) for p in param_cols}
        row["status"] = c["status"]
        for m in metric_cols:
            row[m] = c["metrics"].get(m)
        rows.append(row)

    if sort is not None:
        rows.sort(key=lambda r: (r.get(sort) is None, _sortable(r.get(sort))))
    elif param_cols:
        rows.sort(key=lambda r: tuple(_sortable(r.get(p)) for p in param_cols))

    columns = [*param_cols, "status", *metric_cols]
    table = _render(columns, rows)
    print(table)
    return ShowResult(rows, table)


def _sortable(v: Any):
    """A total-order key: numbers sort together, everything else by string."""
    if isinstance(v, bool):
        return (1, str(v))
    if isinstance(v, int | float):
        return (0, float(v))
    return (1, str(v))


class BestResult:
    """The winning cell of a sweep (see :func:`best`): its swept-param ``combo``,
    the optimized metric ``value``, the full ``metrics`` dict, its ``status``,
    and its job ``dir`` (for locating checkpoints/artifacts)."""

    def __init__(self, combo: dict, status: str, metrics: dict, value: float, dir):
        self.combo = combo
        self.status = status
        self.metrics = metrics
        self.value = value
        self.dir = dir

    def __repr__(self) -> str:
        return (
            f"BestResult(combo={self.combo!r}, value={self.value!r}, dir={self.dir!r})"
        )


def _finite_number(v: Any) -> bool:
    return isinstance(v, int | float) and not isinstance(v, bool) and v == v


def best(root, metric: str, *, mode: str = "max") -> BestResult:
    """Return the completed cell that optimizes ``metric`` across the sweep.

    Parameters
    ----------
    root :
        A sweep ``working_dir``.
    metric :
        The metric to optimize; it must be a finite scalar in each cell.
    mode : str
        ``"max"`` (default) or ``"min"``.

    Raises
    ------
    ValueError
        If ``mode`` is invalid, or no completed cell has ``metric`` as a finite
        scalar (the error lists the scalar metrics that are available).
    FileNotFoundError
        If ``root`` is not a directory.
    """
    if mode not in ("max", "min"):
        raise ValueError(f"`mode` must be 'max' or 'min', got {mode!r}")

    cells = _read_cells(root)
    candidates = [
        c
        for c in cells
        if c["status"] == "completed" and _finite_number(c["metrics"].get(metric))
    ]
    if not candidates:
        available = sorted(
            {
                k
                for c in cells
                if c["status"] == "completed"
                for k, v in c["metrics"].items()
                if _finite_number(v)
            }
        )
        raise ValueError(
            f"no completed cell has a finite scalar metric {metric!r}. "
            f"Available scalar metrics: {', '.join(available) or '(none)'}."
        )

    pick = (max if mode == "max" else min)(
        candidates, key=lambda c: c["metrics"][metric]
    )
    return BestResult(
        combo=dict(pick["combo"]),
        status=pick["status"],
        metrics=dict(pick["metrics"]),
        value=pick["metrics"][metric],
        dir=pick["dir"],
    )


def _render(columns: list[str], rows: list[dict]) -> str:
    if not rows:
        return "(no cells found)"
    cells = [[_fmt(r.get(c)) for c in columns] for r in rows]
    widths = [
        max(len(columns[i]), *(len(row[i]) for row in cells))
        for i in range(len(columns))
    ]
    header = "  ".join(columns[i].ljust(widths[i]) for i in range(len(columns)))
    sep = "  ".join("-" * widths[i] for i in range(len(columns)))
    body = "\n".join(
        "  ".join(row[i].ljust(widths[i]) for i in range(len(columns))) for row in cells
    )
    return f"{header}\n{sep}\n{body}"
