# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
"""`mushin.diff(a, b)` compares two sweep directories — per-cell metric deltas,
which cells are unique to each, and what changed in the environment."""

from __future__ import annotations

import pytest

import mushin
from mushin import multirun
from mushin.workflows import MultiRunMetricsWorkflow


def _run(wd, *, bump=0.0, seeds=(0, 1)):
    class W(MultiRunMetricsWorkflow):
        @staticmethod
        def task(seed):
            return dict(acc=float(seed) + bump)

    W().run(seed=multirun(list(seeds)), working_dir=str(wd))


def test_diff_reports_metric_deltas(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    _run(a, bump=0.0)
    _run(b, bump=10.0)
    d = mushin.diff(a, b)
    assert len(d.rows) == 2  # both grids share seed 0 and 1
    row = next(r for r in d.rows if r["seed"] == 1)
    va, vb, delta = row["deltas"]["acc"]
    assert va == pytest.approx(1.0)
    assert vb == pytest.approx(11.0)
    assert delta == pytest.approx(10.0)
    assert "acc" in str(d)  # the rendered table names the metric


def test_diff_reports_cells_unique_to_each(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    _run(a, seeds=(0, 1, 2))
    _run(b, seeds=(0, 1))
    d = mushin.diff(a, b)
    assert {c["seed"] for c in d.only_in_a} == {2}
    assert d.only_in_b == []
    assert len(d.rows) == 2  # shared cells only


def test_diff_identical_env_has_no_provenance_change(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    _run(a)
    _run(b)
    d = mushin.diff(a, b)
    assert d.provenance == {}  # same process/env; volatile timestamp excluded


def test_diff_provenance_helper_flattens_and_excludes_volatile():
    from mushin._show import _diff_provenance

    a = {
        "timestamp": "t1",
        "git": {"sha": "aaa", "dirty": False},
        "packages": {"torch": "2.2"},
    }
    b = {
        "timestamp": "t2",
        "git": {"sha": "bbb", "dirty": False},
        "packages": {"torch": "2.3"},
    }
    ch = _diff_provenance(a, b)
    assert ch["git.sha"] == ("aaa", "bbb")
    assert ch["packages.torch"] == ("2.2", "2.3")
    assert "timestamp" not in ch  # volatile field excluded
    assert "git.dirty" not in ch  # unchanged field omitted


def test_diff_missing_dir_raises(tmp_path):
    _run(tmp_path / "a")
    with pytest.raises(FileNotFoundError):
        mushin.diff(tmp_path / "a", tmp_path / "nope")


def test_diff_excludes_non_completed_cells(tmp_path):
    """A failed cell that still has a stale metrics sidecar on disk must not
    produce metric deltas as if it were an unchanged completed result."""
    import json

    a, b = tmp_path / "a", tmp_path / "b"
    _run(a, bump=0.0)
    _run(b, bump=0.0)

    # mark seed=1 in b as failed, leaving its (now stale) metrics sidecar
    status_files = sorted(b.glob("*/mushin_cell_status.json"))
    target = next(
        p for p in status_files if json.loads(p.read_text())["combo"] == {"seed": 1}
    )
    payload = json.loads(target.read_text())
    payload["status"] = "failed"
    target.write_text(json.dumps(payload))

    d = mushin.diff(a, b)
    failed_row = next(r for r in d.rows if r["seed"] == 1)
    assert failed_row["deltas"] == {}  # no Δ from a failed cell's stale metrics
    ok_row = next(r for r in d.rows if r["seed"] == 0)
    assert "acc" in ok_row["deltas"]
