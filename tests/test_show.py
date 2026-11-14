# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
"""`mushin.show(root)` reads a sweep directory's per-cell sidecars and returns a
status/metrics table — offline, dependency-free, and readable mid-sweep."""

from __future__ import annotations

import os
import sys

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
    # the printed table lists every column and the values
    table = str(res)
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
    """`metrics=` limits the columns. A typo'd entry must raise (like sort=
    does), not silently drop the column; the requested order must be honored;
    and a repeated entry must not emit a duplicated table/CSV column (a
    duplicate CSV header makes pandas mangle it to acc/acc.1)."""
    from mushin import export

    wd = tmp_path / "s"
    _run_sweep(wd)
    res = mushin.show(wd, metrics=["acc"])
    table = str(res)
    assert "acc" in table
    assert "loss" not in table
    # rows still carry the swept params and status
    assert all("lr" in r and "status" in r for r in res.rows)

    with pytest.raises(ValueError, match="wrong_metric"):
        mushin.show(wd, metrics=["wrong_metric"])
    with pytest.raises(ValueError, match="wrong_metric"):
        export.table(wd, metrics=["wrong_metric"])

    res = mushin.show(wd, metrics=["loss", "acc"])
    cols = res.table.splitlines()[0].split()
    assert cols.index("loss") < cols.index("acc")  # requested order wins

    cols = mushin.show(wd, metrics=["acc", "acc"]).table.splitlines()[0].split()
    assert cols.count("acc") == 1

    header = export.table(wd, metrics=["acc", "acc"]).strip().splitlines()[0]
    assert header.split(",").count("acc") == 1


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


def test_a_param_named_status_keeps_its_values(tmp_path):
    """A swept param named ``status`` must not be overwritten by the cell
    status (#306). It is the column ``status (param)``, a metric named
    ``status`` is ``status (metric)``, and ``status`` stays the cell status."""
    import json

    from mushin import export
    from mushin._resume import write_cell_status

    class W(MultiRunMetricsWorkflow):
        @staticmethod
        def task(status):
            return dict(status=-float(len(status)))

    wd = tmp_path / "s"
    W().run(status=multirun(["bb", "a"]), working_dir=str(wd))

    # The default sort is by the swept value, so "a" comes before "bb".
    rows = mushin.show(wd).rows
    assert [r["status (param)"] for r in rows] == ["a", "bb"]
    assert [r["status"] for r in rows] == ["completed", "completed"]
    assert [r["status (metric)"] for r in rows] == [-1.0, -2.0]
    by_metric = mushin.show(wd, sort="status (metric)").rows
    assert [r["status (param)"] for r in by_metric] == ["bb", "a"]

    lines = export.table(wd).strip().splitlines()
    assert lines[0].split(",") == ["status (param)", "status", "status (metric)"]
    assert [line.split(",") for line in lines[1:]] == [
        ["a", "completed", "-1.0"],
        ["bb", "completed", "-2.0"],
    ]

    # sort="status" sorts by the cell status, not by the param.
    for d in wd.iterdir():
        sidecar = d / "mushin_cell_status.json"
        if sidecar.exists():
            combo = json.loads(sidecar.read_text())["combo"]
            if combo["status"] == "a":
                write_cell_status(d, status="running", combo=combo, attempt=1)
    by_status = mushin.show(wd, sort="status").rows
    assert [(r["status"], r["status (param)"]) for r in by_status] == [
        ("completed", "bb"),
        ("running", "a"),
    ]


def test_cell_rows_never_repeat_a_column():
    """No two columns share a name, whatever the metrics are called: a renamed
    metric is kept clear of the param columns as shown and of every other
    metric, even one found after it (``lr`` comes before ``lr (metric)``)."""
    from mushin._show import _cell_rows

    cell = {
        "combo": {"status": "a", "lr": 0.1},
        "status": "completed",
        "metrics": {
            "status": 1.0,
            "status (param)": 2.0,
            "lr": 4.0,
            "lr (metric)": 3.0,
        },
    }
    _, columns, rows = _cell_rows([cell], None)
    assert rows == [
        {
            "status (param)": "a",
            "lr": 0.1,
            "status": "completed",
            "status (metric)": 1.0,
            "status (param) (metric)": 2.0,
            "lr (metric) (metric)": 4.0,
            "lr (metric)": 3.0,
        }
    ]
    assert columns == list(rows[0])


def test_show_sorts_an_int_metric_beyond_float_range(tmp_path):
    """JSON carries ints of any size; float() of one raises OverflowError, so
    sorting by that metric must place it at the far end instead of crashing."""
    import json

    wd = tmp_path / "s"
    _run_sweep(wd)
    d = next(p for p in wd.iterdir() if (p / "mushin_metrics.json").exists())
    payload = json.loads((d / "mushin_metrics.json").read_text())
    payload["acc"] = 10**400
    (d / "mushin_metrics.json").write_text(json.dumps(payload))
    other = next(
        p for p in wd.iterdir() if p != d and (p / "mushin_metrics.json").exists()
    )
    payload = json.loads((other / "mushin_metrics.json").read_text())
    payload["acc"] = -(10**400)
    (other / "mushin_metrics.json").write_text(json.dumps(payload))
    rows = mushin.show(wd, sort="acc").rows
    assert rows[0]["acc"] == -(10**400)
    assert rows[-1]["acc"] == 10**400


@pytest.mark.parametrize("x", [float("inf"), float("nan")], ids=["inf", "nan"])
def test_show_lists_a_swept_non_finite_cell(tmp_path, x):
    """A swept NaN or inf is stored tagged in the cell's record; show keys it
    as the manifest does, so the finished cell is listed (#355)."""
    import mushin
    from mushin import multirun
    from mushin._show import _read_cells

    @mushin.sweep
    def t(x):
        return dict(v=1.0)

    t.run(x=multirun([1.0, x]), working_dir=str(tmp_path))
    cells = [c["combo"]["x"] for c in _read_cells(tmp_path)]
    assert len(cells) == 2 and cells[0] == 1.0
    assert repr(cells[1]) == repr(x)


# Queued cells (#345). mushin's submitit launchers write a launch file per
# submission under `mushin_jobs/`, which `show` reads to list the cells whose
# jobs have not started. These sweeps are built by hand: records and launch
# files with set times, so each rule is tested on its own.

R0, R1, R2 = "a" * 32, "b" * 32, "c" * 32  # run ids
S = 10**9  # a second, in ns: whole seconds, which every filesystem keeps
T = 1_700_000_000 * S  # a launch file's time, in ns


def _put_record(
    wd,
    name,
    *,
    run,
    combo,
    at,
    status="completed",
    num=None,
    job_id=None,
    entry=None,
    metrics=None,
):
    """A cell's record in ``wd/name``, written at ``at`` (ns)."""
    import os

    from mushin._resume import STATUS_FILE, write_cell_status
    from mushin._sweep_io import write_metrics_sidecar

    d = wd / name
    d.mkdir(parents=True, exist_ok=True)
    write_cell_status(
        d,
        status=status,
        combo=combo,
        attempt=1,
        job_num=num,
        choices=None if num is None else {},
        run_id=run,
        job_id=job_id,
        submitit=entry,
    )
    if metrics is not None:
        write_metrics_sidecar(d, metrics)
    os.utime(d / STATUS_FILE, ns=(at, at))


def _put_launch(wd, *, run, cells, at=T, name=None, **fields):
    """A launch file as the launcher writes it, naming ``cells`` (``(num,
    job_id, combo)``), written at ``at`` (ns). A SLURM launch by default:
    a local one is read only while its jobs' processes live."""
    import json
    import os

    payload = {
        "schema": 1,
        "run_id": run,
        "executor": "slurm",
        "folder": str(wd / ".submitit"),
        "clusters": None,
        "host": "login1",
        "depth": 1,
        "cells": [{"num": n, "job_id": j, "combo": c} for n, j, c in cells],
    }
    payload.update(fields)
    path = wd / "mushin_jobs" / (name or f"{run}.{cells[0][0]}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
    os.utime(path, ns=(at, at))
    return path


def _statuses(wd, **kwargs) -> list[tuple]:
    return [(r["lr"], r["status"]) for r in mushin.show(wd, **kwargs).rows]


def test_show_lists_submitted_cells_as_queued(tmp_path):
    _put_record(tmp_path, "0", run=R1, combo={"lr": 0.1}, num=0, at=T + 5 * S)
    _put_launch(
        tmp_path, run=R1, cells=[(1, "100_0", {"lr": 0.2}), (2, "100_1", {"lr": 0.3})]
    )
    assert _statuses(tmp_path) == [(0.1, "completed"), (0.2, "queued"), (0.3, "queued")]


def test_queued_cell_replaces_an_earlier_runs_record_in_its_dir(tmp_path):
    # The launch holds `lr=1` as typed; the record, a float field's 1.0.
    _put_record(
        tmp_path, "1", run=R0, combo={"lr": 1.0}, num=1, status="failed", at=T - 9 * S
    )
    _put_launch(tmp_path, run=R1, cells=[(1, "100", {"lr": 1})])
    rows = mushin.show(tmp_path).rows
    assert [(r["lr"], r["status"]) for r in rows] == [(1, "queued")]


def test_queued_cell_found_by_combo_under_a_custom_subdir(tmp_path):
    _put_record(
        tmp_path,
        "lr_1",
        run=R0,
        combo={"lr": 1.0},
        num=7,
        status="failed",
        at=T - 9 * S,
    )
    _put_launch(tmp_path, run=R1, cells=[(1, "100", {"lr": 1})])
    assert _statuses(tmp_path) == [(1, "queued")]


def test_a_cell_of_the_launchs_own_run_that_keys_alike_is_not_its_record(tmp_path):
    """`1` and `'1'` key alike, so the combo match of a custom subdir would
    take cell 2's record, of the launch's own run, for cell 1's."""
    _put_launch(
        tmp_path, run=R1, cells=[(1, "100_0", {"lr": 1}), (2, "100_1", {"lr": "1"})]
    )
    _put_record(
        tmp_path,
        "2",
        run=R1,
        combo={"lr": "1"},
        num=2,
        job_id="100_1",
        status="running",
        at=T + 5 * S,
    )
    assert _statuses(tmp_path) == [(1, "queued"), ("1", "running")]


def test_started_cell_shows_its_record(tmp_path):
    _put_record(
        tmp_path,
        "1",
        run=R1,
        combo={"lr": 0.2},
        num=1,
        job_id="100_1",
        status="running",
        at=T + 5 * S,
    )
    _put_launch(tmp_path, run=R1, cells=[(1, "100_1", {"lr": 0.2})])
    assert _statuses(tmp_path) == [(0.2, "running")]


def test_newer_record_of_another_run_retires_a_launch_per_cell(tmp_path):
    """A killed driver's launch stays; a later run's newer record retires
    it for that cell alone."""
    _put_launch(
        tmp_path, run=R1, cells=[(1, "100_0", {"lr": 0.2}), (2, "100_1", {"lr": 0.3})]
    )
    _put_record(tmp_path, "1", run=R2, combo={"lr": 0.2}, num=1, at=T + 10 * S)
    _put_record(
        tmp_path, "2", run=R0, combo={"lr": 0.3}, num=2, status="failed", at=T - 10 * S
    )
    assert _statuses(tmp_path) == [(0.2, "completed"), (0.3, "queued")]


def test_leftover_launch_after_a_later_run_lists_nothing_queued(tmp_path):
    lrs = [0.1, 0.2, 0.3]
    _put_launch(
        tmp_path,
        run=R1,
        cells=[(n, f"100_{n}", {"lr": lr}) for n, lr in enumerate(lrs)],
    )
    # Computed, reused (naming the job that computed it) and cached.
    entry = {"job_id": "100_1", "logs": "/s/.submitit/100_1"}
    _put_record(
        tmp_path, "0", run=R2, combo={"lr": 0.1}, num=0, job_id="200_0", at=T + 10 * S
    )
    _put_record(
        tmp_path, "1", run=R2, combo={"lr": 0.2}, num=1, entry=entry, at=T + 10 * S
    )
    _put_record(tmp_path, "2", run=R2, combo={"lr": 0.3}, num=2, at=T + 10 * S)
    assert _statuses(tmp_path) == [(lr, "completed") for lr in lrs]


def test_record_and_launch_with_one_mtime_show_queued(tmp_path):
    """A resume within a coarse mtime tick of the earlier run: the launch,
    which came after, is what stands."""
    _put_record(tmp_path, "1", run=R0, combo={"lr": 0.2}, num=1, at=T)
    _put_launch(tmp_path, run=R1, cells=[(1, "100", {"lr": 0.2})], at=T)
    assert _statuses(tmp_path) == [(0.2, "queued")]


def test_two_launches_name_one_job_number(tmp_path):
    _put_launch(tmp_path, run=R1, cells=[(1, "100_1", {"lr": 0.2})], at=T)
    _put_launch(
        tmp_path,
        run=R2,
        cells=[(1, "200_0", {"lr": 0.2}), (2, "200_1", {"lr": 0.3})],
        at=T + 10 * S,
    )
    rows = mushin.show(tmp_path, jobs=True).rows
    assert [(r["lr"], r["status"], r["job"]) for r in rows] == [
        (0.2, "queued", "200_0"),
        (0.3, "queued", "200_1"),
    ]


def test_two_launches_of_one_time_are_ordered_by_name(tmp_path):
    _put_launch(tmp_path, run=R2, cells=[(1, "200_0", {"lr": 0.2})], at=T)
    _put_launch(tmp_path, run=R1, cells=[(1, "100_1", {"lr": 0.2})], at=T)
    (row,) = mushin.show(tmp_path, jobs=True).rows
    assert row["job"] == "200_0"  # R2's name sorts after R1's


def test_unreadable_launch_files_are_ignored(tmp_path):
    _put_launch(tmp_path, run=R1, cells=[(0, "100_0", {"lr": 0.1})])
    bad = tmp_path / "mushin_jobs" / f"{R0}.1.json"
    bad.write_text("{not json")
    _put_launch(tmp_path, run=R0, cells=[(2, "100", {"lr": 0.3})], schema=2)
    _put_launch(tmp_path, run=R0, cells=[(3, "1; x", {"lr": 0.4})])
    _put_launch(tmp_path, run=R2, cells=[(4, "100", {"lr": 0.5})], name=f"{R0}.4.json")
    for kind, fields in {
        5: {"executor": "pbs"},
        6: {"folder": "relative/.submitit"},
        7: {"clusters": ""},
        8: {"depth": 0},
        9: {"host": ""},
    }.items():
        _put_launch(
            tmp_path, run=R0, cells=[(kind, "100", {"lr": float(kind)})], **fields
        )
    _put_launch(
        tmp_path, run=R2, cells=[(10, "100", {"lr": 1.0}), (10, "101", {"lr": 1.1})]
    )
    _put_launch(
        tmp_path, run=R2, cells=[(11, "100", {"lr": 1.2}), (True, "101", {"lr": 1.3})]
    )
    # A writer's temporary file, and a name off the pattern, are not launches.
    _put_launch(
        tmp_path,
        run=R2,
        cells=[(12, "100", {"lr": 1.4})],
        name=f"{R2}.12.json.9.ab12.tmp",
    )
    _put_launch(tmp_path, run=R2, cells=[(13, "100", {"lr": 1.5})], name="notes.json")
    assert _statuses(tmp_path) == [(0.1, "queued")]


def test_show_jobs_columns(tmp_path):
    entry = {"job_id": "100_0", "logs": "/s/.submitit/100_0", "host": "node7"}
    _put_record(
        tmp_path,
        "0",
        run=R1,
        combo={"lr": 0.1},
        num=0,
        job_id="100_0",
        entry=entry,
        at=T + 5 * S,
    )
    _put_record(tmp_path, "1", run=R0, combo={"lr": 0.2}, num=1, at=T - 5 * S)
    _put_launch(
        tmp_path, run=R1, cells=[(0, "100_0", {"lr": 0.1}), (2, "100_1", {"lr": 0.3})]
    )
    res = mushin.show(tmp_path, jobs=True)
    assert [
        {k: r[k] for k in ("lr", "status", "job", "host", "logs")} for r in res.rows
    ] == [
        {
            "lr": 0.1,
            "status": "completed",
            "job": "100_0",
            "host": "node7",
            "logs": "/s/.submitit/100_0",
        },
        {"lr": 0.2, "status": "completed", "job": None, "host": None, "logs": None},
        {"lr": 0.3, "status": "queued", "job": "100_1", "host": None, "logs": None},
    ]
    assert res.table.splitlines()[0].split() == ["lr", "status", "job", "host", "logs"]
    plain = mushin.show(tmp_path)
    assert plain.table.splitlines()[0].split() == ["lr", "status"]
    assert all(set(r) == {"lr", "status"} for r in plain.rows)


def test_jobs_column_names_reserved_only_with_jobs(tmp_path):
    for n, host in enumerate(["b", "a"]):
        _put_record(
            tmp_path,
            str(n),
            run=R1,
            combo={"host": host},
            num=n,
            entry={"job_id": f"10{n}", "logs": "/s", "host": f"node{n}"},
            metrics={"job": float(n), "logs": 1.0},
            at=T,
        )
    rows = mushin.show(tmp_path, jobs=True, sort="host").rows
    assert [
        (r["host (param)"], r["host"], r["job (metric)"], r["job"]) for r in rows
    ] == [
        ("b", "node0", 0.0, "100"),
        ("a", "node1", 1.0, "101"),
    ]
    assert [r["logs (metric)"] for r in rows] == [1.0, 1.0]
    # Without `jobs`, the names are the sweep's own.
    plain = mushin.show(tmp_path, sort="host").rows
    assert [(r["host"], r["job"], r["logs"]) for r in plain] == [
        ("a", 1.0, 1.0),
        ("b", 0.0, 1.0),
    ]


def test_best_over_seed_counts_queued_as_in_flight(tmp_path):
    for n, (lr, seed) in enumerate([(0.1, 0), (0.1, 1), (0.2, 0)]):
        combo = {"lr": lr, "seed": seed}
        _put_record(
            tmp_path,
            str(n),
            run=R1,
            combo=combo,
            num=n,
            metrics={"acc": lr},
            at=T + 5 * S,
        )
    _put_launch(tmp_path, run=R1, cells=[(3, "100_3", {"lr": 0.2, "seed": 1})])
    with pytest.warns(UserWarning, match=r"1 cell\(s\) queued") as rec:
        r = mushin.best(tmp_path, "acc", over="seed")
    assert r.combo == {"lr": 0.1}
    assert "has not finished" in str(rec[0].message)


def test_export_and_diff_ignore_launches(tmp_path):
    from mushin import export

    a, b = tmp_path / "a", tmp_path / "b"
    for wd in (a, b):
        _put_record(
            wd,
            "0",
            run=R1,
            combo={"lr": 0.1},
            num=0,
            metrics={"acc": 1.0},
            at=T - 5 * S,
        )
        _put_record(
            wd,
            "1",
            run=R1,
            combo={"lr": 0.2},
            num=1,
            metrics={"acc": 2.0},
            at=T - 5 * S,
        )
    before = (export.table(a), mushin.diff(a, b).table)
    _put_launch(a, run=R2, cells=[(1, "200", {"lr": 0.2}), (2, "201", {"lr": 0.3})])
    assert (export.table(a), mushin.diff(a, b).table) == before
    assert _statuses(a) == [(0.1, "completed"), (0.2, "queued"), (0.3, "queued")]


def test_started_matched_by_job_id(tmp_path):
    """A record of the launch's run that names its job, but no job number
    (no Hydra job fields), is that job's."""
    _put_record(
        tmp_path,
        "1",
        run=R1,
        combo={"lr": 0.2},
        job_id="100_1",
        status="running",
        at=T + 5 * S,
    )
    _put_launch(tmp_path, run=R1, cells=[(1, "100_1", {"lr": 0.2})])
    assert _statuses(tmp_path) == [(0.2, "running")]


def test_records_scoped_out_still_count(tmp_path):
    """A later run of a narrower grid makes `show` drop the leftover
    launch's cell 2 as stale; its record still says the job ran."""
    import os

    from mushin._sweep_io import MANIFEST_FILE, Manifest

    lrs = [0.1, 0.2, 0.3]
    _put_launch(
        tmp_path,
        run=R1,
        cells=[(n, f"100_{n}", {"lr": lr}) for n, lr in enumerate(lrs)],
    )
    _put_record(
        tmp_path, "2", run=R1, combo={"lr": 0.3}, num=2, job_id="100_2", at=T + 5 * S
    )
    manifest = Manifest(tmp_path, ["lr"])
    for n, lr in enumerate(lrs[:2]):
        _put_record(tmp_path, str(n), run=R2, combo={"lr": lr}, num=n, at=T + 10 * S)
        manifest.mark({"lr": lr}, dir=str(n), status="completed")
    manifest.save()
    os.utime(tmp_path / MANIFEST_FILE, ns=(T + 20 * S, T + 20 * S))
    assert _statuses(tmp_path) == [(0.1, "completed"), (0.2, "completed")]


def test_a_launch_of_nested_cell_dirs_lists_nothing_queued(tmp_path):
    """`show` reads only the sweep root's direct children, so a launch
    whose `hydra.sweep.subdir` nests deeper cannot tell its cells' records."""
    _put_launch(tmp_path, run=R1, cells=[(1, "100", {"lr": 0.2})], depth=2)
    assert mushin.show(tmp_path).rows == []


@pytest.mark.skipif(sys.platform == "win32", reason="a pid is probed on POSIX only")
def test_a_local_launch_is_read_while_its_jobs_live(tmp_path):
    """A local launch's job ids are process ids on its host. Once none of
    them lives there, the launch is not read (a driver killed before
    removing it), elsewhere it is."""
    import socket
    import subprocess

    done = subprocess.Popen([sys.executable, "-c", "pass"])
    done.wait()  # reaped: the pid names no process
    here = socket.gethostname()

    def queued(pid, host):
        for path in (tmp_path / "mushin_jobs").glob("*.json"):
            path.unlink()
        _put_launch(
            tmp_path,
            run=R1,
            cells=[(1, str(pid), {"lr": 0.2})],
            executor="local",
            host=host,
        )
        return _statuses(tmp_path) == [(0.2, "queued")]

    assert not queued(done.pid, here)
    assert queued(os.getpid(), here)
    assert queued(done.pid, here + "-elsewhere")


@pytest.mark.parametrize("launcher", ["basic", "joblib"])
def test_other_launchers_record_no_launch(tmp_path, launcher):
    @mushin.sweep
    def t(lr):
        return dict(v=float(lr))

    how = {} if launcher == "basic" else {"launcher": launcher}
    t.run(lr=multirun([0.1, 0.2]), working_dir=str(tmp_path / "s"), **how)
    assert not (tmp_path / "s" / "mushin_jobs").exists()


def test_a_launch_file_reads_back_as_written(tmp_path):
    """What the launcher writes, `show` reads: a swept NaN tagged as in a
    record, and a folder of either path flavor (a cluster's sweep read from a
    desktop over a share)."""
    import math

    from mushin._jobs import LaunchCell, read_launches, record_launch

    for n, folder in enumerate(["/scratch/s/.submitit", r"C:\s\.submitit"]):
        path = record_launch(
            tmp_path / str(n),
            run_id=R1,
            executor="slurm",
            folder=folder,
            clusters=None,
            host="login1",
            depth=1,
            cells=[LaunchCell(3, "7_0", {"lr": float("nan"), "m": None})],
        )
        assert path.name == f"{R1}.3.json"
        (launch,), bad = read_launches(tmp_path / str(n))
        assert bad == [] and launch.folder == folder and launch.run_id == R1
        ((num, job_id, combo),) = [(c.num, c.job_id, c.combo) for c in launch.cells]
        assert (num, job_id, combo["m"]) == (3, "7_0", None)
        assert math.isnan(combo["lr"])
