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


def _read_provenance(root) -> dict | None:
    """A representative provenance record for a sweep (the first cell's
    ``mushin_provenance.json``), or None if none is readable."""
    import json

    for d in sorted(
        (p for p in Path(root).iterdir() if p.is_dir()),
        key=lambda p: _numeric_dir_key(p.name),
    ):
        p = d / "mushin_provenance.json"
        if p.exists():
            try:
                data = json.loads(p.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(data, dict):
                return data
    return None


# Fields that vary per cell / per run rather than describing the environment;
# excluded from the provenance delta so it reports only real env changes.
_PROV_VOLATILE = frozenset({"timestamp", "config"})


def _flatten(d: dict, prefix: str = "") -> dict:
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key + "."))
        else:
            out[key] = v
    return out


def _diff_provenance(a: dict | None, b: dict | None) -> dict:
    """Flattened field-by-field diff of two provenance records, excluding
    volatile fields. Returns ``{dotted_key: (a_value, b_value)}`` for every field
    whose value differs (a field present on only one side reads as ``None`` on
    the other)."""
    fa = _flatten({k: v for k, v in (a or {}).items() if k not in _PROV_VOLATILE})
    fb = _flatten({k: v for k, v in (b or {}).items() if k not in _PROV_VOLATILE})
    changed = {}
    for k in sorted(set(fa) | set(fb)):
        va, vb = fa.get(k), fb.get(k)
        if va != vb:
            changed[k] = (va, vb)
    return changed


class DiffResult:
    """The result of :func:`diff`: ``rows`` (one per shared cell, each with a
    ``deltas`` map of ``metric -> (a, b, b - a)``), ``only_in_a`` / ``only_in_b``
    (combos unique to each sweep), the ``provenance`` delta, and the rendered
    ``table``. ``str()`` returns the table."""

    def __init__(self, rows, only_in_a, only_in_b, provenance, table):
        self.rows = rows
        self.only_in_a = only_in_a
        self.only_in_b = only_in_b
        self.provenance = provenance
        self.table = table

    def __str__(self) -> str:
        return self.table

    def __repr__(self) -> str:
        return self.table


def diff(a, b, *, metrics: list[str] | None = None) -> DiffResult:
    """Compare two sweep directories ``a`` and ``b``.

    Cells are aligned by their swept-param combination. For each shared cell the
    delta ``b - a`` is computed for every metric that is a finite scalar in both.
    Cells present in only one sweep are reported separately, along with a diff of
    the two runs' environment provenance (git/packages/python/accelerator).

    Returns
    -------
    DiffResult
        ``.rows`` (shared-cell deltas), ``.only_in_a`` / ``.only_in_b`` (combos),
        ``.provenance`` (``{field: (a, b)}``), and ``.table``; the table prints.
    """
    from ._sweep_io import combo_key

    cells_a = {combo_key(c["combo"]): c for c in _read_cells(a)}
    cells_b = {combo_key(c["combo"]): c for c in _read_cells(b)}

    keys_a, keys_b = set(cells_a), set(cells_b)
    only_in_a = [dict(cells_a[k]["combo"]) for k in sorted(keys_a - keys_b)]
    only_in_b = [dict(cells_b[k]["combo"]) for k in sorted(keys_b - keys_a)]

    want = set(metrics) if metrics is not None else None
    rows = []
    for k in sorted(keys_a & keys_b):
        ca, cb = cells_a[k], cells_b[k]
        deltas: dict[str, tuple] = {}
        for name, va in ca["metrics"].items():
            if want is not None and name not in want:
                continue
            vb = cb["metrics"].get(name)
            if _finite_number(va) and _finite_number(vb):
                deltas[name] = (va, vb, vb - va)
        row: dict[str, Any] = dict(ca["combo"])
        row["deltas"] = deltas
        rows.append(row)

    provenance = _diff_provenance(_read_provenance(a), _read_provenance(b))
    table = _render_diff(rows, only_in_a, only_in_b, provenance)
    print(table)
    return DiffResult(rows, only_in_a, only_in_b, provenance, table)


def _render_diff(rows, only_in_a, only_in_b, provenance) -> str:
    param_cols = _ordered_union(
        [{k: v for k, v in r.items() if k != "deltas"} for r in rows]
    )
    metric_cols = _ordered_union([r["deltas"] for r in rows])
    columns = [*param_cols, *(f"Δ{m}" for m in metric_cols)]
    flat_rows = []
    for r in rows:
        fr = {p: r.get(p) for p in param_cols}
        for m in metric_cols:
            fr[f"Δ{m}"] = r["deltas"][m][2] if m in r["deltas"] else None
        flat_rows.append(fr)
    parts = [_render(columns, flat_rows) if rows else "(no shared cells)"]
    if only_in_a:
        parts.append(f"only in a ({len(only_in_a)}): {only_in_a}")
    if only_in_b:
        parts.append(f"only in b ({len(only_in_b)}): {only_in_b}")
    if provenance:
        parts.append("provenance changes:")
        parts.extend(f"  {k}: {va!r} -> {vb!r}" for k, (va, vb) in provenance.items())
    return "\n".join(parts)
