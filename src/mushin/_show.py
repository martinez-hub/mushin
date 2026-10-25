# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
"""`mushin.show`: a dependency-free, offline view of a sweep directory.

Reads each cell's status sidecar (its swept-param ``combo`` and ``status``) and
metrics sidecar directly: pure JSON, so it works mid-sweep and needs neither
Hydra nor xarray. Handy for watching a live sweep or eyeballing a finished one
before committing to the full ``to_xarray`` load."""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence


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


def _manifest_cells(root: Path) -> tuple[dict, float]:
    """The latest sweep's manifest cells and the manifest's mtime, or
    ``({}, 0.0)`` when none is usable (missing, corrupt, wrong shape): the scan
    is then unscoped, and ``best(over=)`` cannot take the sweep as finished."""
    from ._sweep_io import MANIFEST_FILE, Manifest

    try:
        cells = Manifest.load_or_new(root, []).cells
        if not isinstance(cells, dict) or not cells:
            return {}, 0.0
        return cells, (root / MANIFEST_FILE).stat().st_mtime
    except Exception:  # noqa: BLE001 - unreadable manifest -> unscoped scan
        return {}, 0.0


def _read_cells(root) -> list[dict]:
    """Scan a sweep ``root`` and return one dict per cell:
    ``{"combo", "status", "metrics", "dir"}`` (``dir`` is the job dir Path).
    Reads only the per-cell JSON sidecars: no Hydra/xarray. Raises
    ``FileNotFoundError`` if ``root`` is not a directory.

    When a sweep manifest exists it scopes the result to the LATEST sweep's
    grid: a dir whose combo the manifest does not list (or lists under another
    dir) is stale residue of an earlier/wider sweep in a reused ``working_dir``
    and is excluded, unless its status sidecar is newer than the manifest,
    which marks a cell of a newer sweep currently in flight (the mid-sweep
    view). Without a manifest (fresh dir mid-sweep, pre-manifest sweeps) every
    cell dir is kept."""
    return _scan(root)[0]


def _scan(root) -> tuple[list[dict], float]:
    """``_read_cells`` plus the mtime of the manifest that scoped the scan
    (0.0 when none did), read once so the two never disagree."""
    from ._resume import STATUS_FILE, read_cell_status
    from ._sweep_io import combo_key, read_metrics_sidecar

    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"sweep directory not found: {root}")

    manifest_cells, manifest_mtime = _manifest_cells(root)

    cells: list[dict] = []
    for d in sorted(
        (p for p in root.iterdir() if p.is_dir()),
        key=lambda p: _numeric_dir_key(p.name),
    ):
        s = read_cell_status(d)
        if s is None or not isinstance(s.get("combo"), dict):
            continue
        if manifest_cells:
            entry = manifest_cells.get(combo_key(s["combo"]))
            in_grid = isinstance(entry, dict) and entry.get("dir") == d.name
            if not in_grid:
                try:
                    newer = (d / STATUS_FILE).stat().st_mtime > manifest_mtime
                except OSError:
                    newer = False
                if not newer:
                    continue  # stale cell from a previous sweep of this dir
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
    return cells, manifest_mtime


def _metric_column_names(param_cols: list[str], metric_cols: list[str]) -> dict:
    """Column name per metric; a metric colliding with a swept-param column (or
    ``status``) is disambiguated as ``"<name> (metric)"`` so both values
    survive; e.g. a task reporting the effective ``lr`` alongside a swept
    ``lr``."""
    taken = set(param_cols) | {"status"}
    return {m: (f"{m} (metric)" if m in taken else m) for m in metric_cols}


def _apply_metric_filter(
    metrics: list[str] | None, metric_cols: list[str]
) -> list[str]:
    """Restrict/reorder metric columns to ``metrics``. Unknown names raise
    (like ``sort=`` does) (a typo silently dropping a column is worse) and
    the caller's requested order is honored, with repeats deduplicated (a
    duplicate CSV header would be mangled by downstream consumers)."""
    if metrics is None:
        return metric_cols
    unknown = [m for m in metrics if m not in set(metric_cols)]
    if unknown:
        raise ValueError(
            f"metric column(s) {', '.join(map(repr, unknown))} not found; "
            f"available metrics: {', '.join(metric_cols) or '(none)'}"
        )
    return list(dict.fromkeys(metrics))


class ShowResult:
    """The result of ``show``: ``rows`` (one dict per cell, with raw swept
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
        Restrict the metric columns to these names, in this order (default:
        every metric found, discovery order). An unknown name raises
        ``ValueError`` (like ``sort=``); repeats are deduplicated.
    sort : str | None
        Column to sort rows by (a swept param, ``"status"``, or a metric).
        Defaults to the swept params in grid order. A metric that shares a
        swept param's name is shown (and sorted) as ``"<name> (metric)"``.

    Returns
    -------
    ShowResult
        ``.rows`` (one dict per cell) and ``.table`` (the rendered string); the
        table is also printed.
    """
    cells = _read_cells(root)

    param_cols = _ordered_union([c["combo"] for c in cells])
    metric_cols = _ordered_union([c["metrics"] for c in cells])
    metric_cols = _apply_metric_filter(metrics, metric_cols)
    colname = _metric_column_names(param_cols, metric_cols)

    rows: list[dict] = []
    for c in cells:
        row: dict[str, Any] = {p: c["combo"].get(p) for p in param_cols}
        row["status"] = c["status"]
        for m in metric_cols:
            row[colname[m]] = c["metrics"].get(m)
        rows.append(row)

    columns = [*param_cols, "status", *(colname[m] for m in metric_cols)]
    if sort is not None:
        if sort not in columns:
            raise ValueError(
                f"sort column {sort!r} not found; available columns: "
                f"{', '.join(columns)}"
            )
        rows.sort(key=lambda r: (r.get(sort) is None, _sortable(r.get(sort))))
    elif param_cols:
        rows.sort(key=lambda r: tuple(_sortable(r.get(p)) for p in param_cols))
    table = _render(columns, rows)
    print(table)
    return ShowResult(rows, table)


def _sortable(v: Any):
    """A total-order key: finite numbers first (numeric order), then everything
    else by string, then NaN last. NaN gets its own bucket so it cannot break the
    sort's total order (``nan < x`` and ``x < nan`` are both False)."""
    if isinstance(v, bool):
        return (1, str(v))
    if isinstance(v, int | float):
        f = float(v)
        if f != f:  # NaN
            return (2, 0.0)
        return (0, f)
    return (1, str(v))


class BestResult:
    """The winner of a sweep (see ``best``).

    Attributes
    ----------
    combo : dict
        The winning cell's swept parameters. With ``best(..., over=...)`` the
        averaged parameters are omitted, so it names a configuration.
    status : str
        The cell's status, always ``"completed"``.
    metrics : dict
        Every metric of the cell. With ``over=`` the mean of every metric that
        is a finite scalar in each cell of the group; the others are omitted.
    value : float
        The optimized metric (its mean over the group with ``over=``).
    dir : Path | None
        The cell's job dir, for locating checkpoints and artifacts. ``None``
        with ``over=``, since the winner is then a group of cells.
    dirs : list[Path]
        The job dirs of every cell behind the result (``[dir]`` for a single
        cell), in job order.
    n : int
        How many cells the result averages; 1 without ``over=``.
    """

    def __init__(
        self,
        combo: dict,
        status: str,
        metrics: dict,
        value: float,
        dir: Path | None,
        *,
        dirs: list[Path] | None = None,
    ):
        self.combo = combo
        self.status = status
        self.metrics = metrics
        self.value = value
        self.dir = dir
        # ``dir is None`` means "a group", which must say which cells it averages.
        if dirs is None:
            if dir is None:
                raise TypeError("a BestResult without a dir needs dirs")
            dirs = [dir]
        self.dirs = list(dirs)
        self.n = len(self.dirs)

    def __repr__(self) -> str:
        if self.dir is None:
            return (
                f"BestResult(combo={self.combo!r}, value={self.value!r}, "
                f"n={self.n}, dirs={self.dirs!r})"
            )
        return (
            f"BestResult(combo={self.combo!r}, value={self.value!r}, dir={self.dir!r})"
        )


def _finite_number(v: Any) -> bool:
    # isfinite, not just `v == v`: an inf metric must not be "best", and a group
    # holding +inf and -inf would make fmean raise instead of being incomplete.
    if isinstance(v, bool) or not isinstance(v, int | float):
        return False
    try:
        return math.isfinite(v)
    except OverflowError:  # an int beyond float range (JSON allows it): no float
        return False


def _is_candidate(cell: dict, metric: str) -> bool:
    return cell["status"] == "completed" and _finite_number(cell["metrics"].get(metric))


def _over_names(over) -> list[str]:
    """The ``over`` argument of ``best`` as a list of distinct names, in the
    caller's order. Checked before the sweep is read, like ``mode``, so a wrong
    argument is not masked by a sweep that has no finite metric yet."""
    if isinstance(over, str):
        names = [over]
    else:
        try:
            names = list(over)
        except TypeError:
            names = []
    if not names or not all(isinstance(n, str) for n in names):
        raise ValueError(
            f"`over` must be a swept-param name or a sequence of them, got {over!r}"
        )
    return list(dict.fromkeys(names))


def _check_swept(over: list[str], cells: list[dict]) -> None:
    """A name no cell sweeps raises: a typo would otherwise silently average
    over nothing and return the best single cell."""
    swept = _ordered_union([c["combo"] for c in cells])
    unknown = [n for n in over if n not in swept]
    if unknown:
        raise ValueError(
            f"`over` names {', '.join(map(repr, unknown))}, which the sweep does "
            f"not vary; swept params: {', '.join(swept) or '(none)'}"
        )


def _mean(values: list[float]) -> float | None:
    """``fmean``, or None when it has no finite answer: fsum raises on
    intermediate overflow (two values near 1e308), and a mean of ``inf`` is no
    mean to compare."""
    from statistics import fmean

    try:
        m = fmean(values)
    except OverflowError:
        return None
    return m if math.isfinite(m) else None


def _best_group(
    root,
    cells: list[dict],
    manifest_mtime: float,
    metric: str,
    mode: str,
    over: list[str],
) -> BestResult:
    """``best(over=...)``: group cells by their combo minus ``over``, average
    ``metric`` within each group and pick the best mean.

    Means are comparable only over the same seeds, so a group counts only when
    it is complete: it has a result (a completed cell with a finite ``metric``)
    at every ``over`` value where any group has one, and one result per value.
    That bar, rather than "every cell present completed", is what makes the
    rule hold: a cell that has not started has no job dir at all, so a
    configuration still missing seeds would otherwise pass mid-sweep, or after
    a kill, and could win on one lucky seed. And rather than "every value any
    cell has", so that a seed that failed for every configuration drops out
    instead of making the sweep unanswerable. While the sweep is unfinished
    (a cell running, a sidecar newer than the manifest that ``run()`` writes
    last, or no usable manifest) the bar is provisional and the result says
    so; once finished, a single value of ``over`` means there is nothing to
    average, and raises. Skipped groups, and cells whose combo lacks an
    ``over`` param (another grid in flight in a reused ``working_dir``), are
    reported with a warning; no complete group raises."""
    import warnings

    from ._resume import STATUS_FILE
    from ._sweep_io import combo_key

    def over_key(c: dict) -> str:
        return combo_key({k: c["combo"][k] for k in over})

    lacking = 0
    groups: dict[str, list[dict]] = {}
    for c in cells:
        if any(k not in c["combo"] for k in over):
            lacking += 1
            continue
        key = combo_key({k: v for k, v in c["combo"].items() if k not in over})
        groups.setdefault(key, []).append(c)
    grouped = [c for g in groups.values() for c in g]

    results = {
        key: [c for c in g if _is_candidate(c, metric)] for key, g in groups.items()
    }
    bar = {over_key(c) for rs in results.values() for c in rs}
    # No bar (no grouped cell has a result) means no complete group, not every
    # group complete over nothing.
    complete = [
        rs
        for rs in results.values()
        if bar and len(rs) == len(bar) and {over_key(c) for c in rs} == bar
    ]
    # A mean that overflows is no mean to compare: the group is skipped too.
    scored = [
        (m, rs)
        for rs in complete
        if (m := _mean([c["metrics"][metric] for c in rs])) is not None
    ]

    names = ", ".join(map(repr, over))
    notes = []
    # Unfinished: a grouped cell still running (a resume re-runs cells under the
    # prior run's manifest), a status sidecar newer than the manifest (a re-run
    # of the same grid overwrites cells in place, and run() writes the manifest
    # last), or no usable manifest (never written, corrupt, or a sweep that
    # predates it).
    in_flight = sum(c["status"] in ("running", "pending") for c in grouped)

    def newer(c: dict) -> bool:
        try:
            return (c["dir"] / STATUS_FILE).stat().st_mtime > manifest_mtime
        except OSError:
            return False

    if in_flight:
        why = f"{in_flight} cell(s) still running"
    elif not manifest_mtime:
        why = "no usable manifest: it is written when run() ends, or predates it"
    elif any(newer(c) for c in grouped):
        why = "cells newer than the manifest, so a re-run is in flight"
    else:
        why = None
    if why:
        notes.append(
            f"the sweep has not finished ({why}), so the bar is the values of "
            f"{names} reached so far, and a configuration whose other cells have "
            "not started looks complete"
        )
    elif len(bar) == 1:  # an empty bar is "no complete group", below
        raise ValueError(
            f"only one value of {names} has a result anywhere (the sweep has one, "
            f"or the others failed everywhere), so there is nothing to average; "
            f"`best()` without `over=` returns the best single cell."
        )
    skipped = len(groups) - len(scored)
    if skipped:
        notes.append(
            f"skipped {skipped} of {len(groups)} groups that lack a result (a "
            f"completed cell with a finite {metric!r}) at a value of {names} where "
            f"another group has one, hold two cells at one value, or whose mean "
            f"overflows"
        )
    if lacking:
        notes.append(
            f"{lacking} cell(s) whose params do not include {names} were left out "
            f"(if they are the sweep you meant, it does not vary {names})"
        )
    if not scored:
        raise ValueError(
            f"no group of cells that share their params other than {names} is "
            f"complete, so no mean is comparable: {'; '.join(notes)}. `best()` "
            f"without `over=` still returns the best single cell."
        )
    if notes:
        warnings.warn(
            f"best(over={over!r}): {'; '.join(notes)}; the result compares the "
            f"{len(scored)} complete group(s).",
            UserWarning,
            stacklevel=3,
        )

    value, pick = (max if mode == "max" else min)(scored, key=lambda t: t[0])
    means = {}
    for m in _ordered_union([c["metrics"] for c in pick]):
        if all(_finite_number(c["metrics"].get(m)) for c in pick):
            mean = _mean([c["metrics"][m] for c in pick])
            if mean is not None:
                means[m] = mean
    return BestResult(
        combo={k: v for k, v in pick[0]["combo"].items() if k not in over},
        status="completed",
        metrics=means,
        value=value,
        dir=None,
        dirs=[c["dir"] for c in pick],
    )


def best(
    root,
    metric: str,
    *,
    mode: str = "max",
    over: str | Sequence[str] | None = None,
) -> BestResult:
    """Return the completed cell that optimizes ``metric`` across the sweep, or,
    with ``over=``, the configuration whose mean ``metric`` over the named
    swept params is best.

    Parameters
    ----------
    root :
        A sweep ``working_dir``.
    metric :
        The metric to optimize; it must be a finite scalar in each cell.
    mode : str
        ``"max"`` (default) or ``"min"``.
    over : str | Sequence[str] | None
        Swept params to average over, typically ``"seed"``. The default picks
        the single best cell, which is one seed; ``over="seed"`` groups the
        cells that share every other param, averages ``metric`` within each
        group and returns the best group, so a lucky seed cannot win. Means
        are comparable only over the same seeds, so only a complete group
        counts: one with a result (a completed cell with a finite ``metric``)
        at every seed where any group has one. The others (a failed cell, a
        ``sample=`` sweep's holes, a configuration whose seeds have not all
        run) are skipped with a warning; a seed that failed for every
        configuration drops out instead. On a sweep still running the seeds
        reached so far set the bar, and the warning says so; the definitive answer comes once the sweep has finished. A finished sweep with one value of ``over`` has nothing to average, and raises.

    Returns
    -------
    BestResult
        With ``over=``, its ``combo`` omits the averaged params, ``value`` and
        ``metrics`` are means, ``n`` counts the cells, ``dirs`` lists their job
        dirs and ``dir`` is ``None``.

    Raises
    ------
    ValueError
        If ``mode`` is invalid, no completed cell has ``metric`` as a finite
        scalar (the error lists the scalar metrics that are available), ``over``
        is not a swept-param name or a sequence of them, names a param the
        sweep does not vary, or no group is complete.
    FileNotFoundError
        If ``root`` is not a directory.
    """
    if mode not in ("max", "min"):
        raise ValueError(f"`mode` must be 'max' or 'min', got {mode!r}")
    names = None if over is None else _over_names(over)

    cells, manifest_mtime = _scan(root)
    if names is not None:
        _check_swept(names, cells)
    candidates = [c for c in cells if _is_candidate(c, metric)]
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

    if names is not None:
        return _best_group(root, cells, manifest_mtime, metric, mode, names)

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
    ``mushin_provenance.json`` that has an accelerator, else the first), or
    None if none is readable."""
    from ._provenance import representative_record

    dirs = sorted(
        (p for p in Path(root).iterdir() if p.is_dir()),
        key=lambda p: _numeric_dir_key(p.name),
    )
    return representative_record(d / "mushin_provenance.json" for d in dirs)


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


def _accelerator_unrecorded(record: dict | None) -> bool:
    # A record that is missing altogether reports every field as (None, value),
    # the accelerator among them; the rule is for a record that has none.
    return record is not None and not isinstance(record.get("accelerator"), dict)


def _diff_provenance(a: dict | None, b: dict | None) -> dict:
    """Flattened field-by-field diff of two provenance records, excluding
    volatile fields and an accelerator that either side did not record.
    Returns ``{dotted_key: (a_value, b_value)}`` for every field whose value
    differs (a field present on only one side reads as ``None`` on the
    other)."""
    skip = set(_PROV_VOLATILE)
    # A null accelerator means "not recorded": the cell's process never
    # imported torch, so a torch sweep against a scikit-learn sweep on the same
    # box is no hardware change. A dict is recorded whenever torch was loaded
    # (its cuda and cudnn are the wheel's build, its device None until the
    # task initialized CUDA) and still compares, so a CPU-only torch build
    # against a GPU box is reported, as it always was.
    if _accelerator_unrecorded(a) or _accelerator_unrecorded(b):
        skip.add("accelerator")
    fa = _flatten({k: v for k, v in (a or {}).items() if k not in skip})
    fb = _flatten({k: v for k, v in (b or {}).items() if k not in skip})
    changed = {}
    for k in sorted(set(fa) | set(fb)):
        va, vb = fa.get(k), fb.get(k)
        if va != vb:
            changed[k] = (va, vb)
    return changed


class DiffResult:
    """The result of ``diff``: ``rows`` (one per shared cell, each with a
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
    delta ``b - a`` is computed for every metric that is a finite scalar in both,
    and only when the cell is ``completed`` on BOTH sides (a failed/skipped
    cell may carry a stale sidecar from a prior attempt; its row appears with
    empty ``deltas``). Cells present in only one sweep are reported separately,
    along with a diff of the two runs' environment provenance
    (git/packages/python, and the accelerator when both runs recorded one).

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
        # Deltas only between COMPLETED cells: a failed/skipped cell may still
        # carry a stale metrics sidecar from a prior attempt, and a Δ computed
        # from it would present the failure as an unchanged result.
        both_completed = ca["status"] == "completed" and cb["status"] == "completed"
        for name, va in ca["metrics"].items() if both_completed else ():
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
