import pytest

from mushin._sweep_io import (
    Manifest,
    combo_key,
    read_metrics_sidecar,
    write_metrics_sidecar,
)


def test_combo_key_is_canonical_and_order_stable():
    assert combo_key({"lr": 0.1, "seed": 2}) == combo_key({"seed": 2, "lr": 0.1})
    assert combo_key({"lr": 0.1, "seed": 2}) == "lr=0.1,seed=2"


def test_metrics_sidecar_roundtrip(tmp_path):
    write_metrics_sidecar(tmp_path, {"accuracy": 0.9, "loss": 0.1})
    assert read_metrics_sidecar(tmp_path) == {"accuracy": 0.9, "loss": 0.1}
    assert (tmp_path / "mushin_metrics.json").exists()
    assert read_metrics_sidecar(tmp_path / "nope") is None  # absent -> None


def test_read_metrics_sidecar_corrupt_returns_none(tmp_path):
    # A corrupt (non-JSON) sidecar must be treated like a missing one (None),
    # not raise, so a resume can proceed and simply re-run that cell.
    (tmp_path / "mushin_metrics.json").write_text("{not: valid json,,,")
    assert read_metrics_sidecar(tmp_path) is None


def test_manifest_tracks_and_replaces_cells(tmp_path):
    m = Manifest.load_or_new(tmp_path, params=["lr", "seed"])
    m.mark({"lr": 0.1, "seed": 0}, dir="0", status="completed")
    m.mark({"lr": 0.1, "seed": 2}, dir="5", status="failed", error="OOM")
    m.save()

    m2 = Manifest.load_or_new(tmp_path, params=["lr", "seed"])
    assert m2.cells == {
        "lr=0.1,seed=0": {"dir": "0", "status": "completed"},
        "lr=0.1,seed=2": {"dir": "5", "status": "failed", "error": "OOM"},
    }
    # a re-run REPLACES in place (no duplicate entry)
    m2.mark({"lr": 0.1, "seed": 2}, dir="9", status="completed")
    assert m2.cells["lr=0.1,seed=2"] == {"dir": "9", "status": "completed"}
    assert len(m2.cells) == 2
    assert m2.is_complete()
    m2.mark({"lr": 1.0, "seed": 1}, dir="3", status="failed", error="boom")
    assert not m2.is_complete()


def test_manifest_load_or_new_corrupt_returns_fresh(tmp_path):
    """A manifest truncated by a mid-write kill must not make the sweep
    un-resumable: treat it like a missing manifest (the per-cell status
    sidecars are the durable source of truth)."""
    from mushin._sweep_io import MANIFEST_FILE, Manifest

    (tmp_path / MANIFEST_FILE).write_text('{"schema": 1, "cells": {tru')
    m = Manifest.load_or_new(tmp_path, ["a", "b"])
    assert m.cells == {}
    assert m.params == ["a", "b"]


@pytest.mark.parametrize(
    "content",
    [b"\xff\xfe{}", b"[" * 100_000],
    # Short ids: pytest would otherwise name a case after its 100,000 bytes.
    ids=["not-utf8", "deep-nesting"],
)
def test_manifest_load_or_new_unreadable_returns_fresh(tmp_path, content):
    """Bytes that are not UTF-8, or JSON nested past the recursion limit, read
    as a corrupt manifest, as they do for the status and metrics sidecars."""
    from mushin._sweep_io import MANIFEST_FILE, Manifest

    (tmp_path / MANIFEST_FILE).write_bytes(content)
    m = Manifest.load_or_new(tmp_path, ["a", "b"])
    assert m.cells == {}


def test_from_cell_status_survives_corrupt_manifest(tmp_path):
    from mushin._resume import write_cell_status
    from mushin._sweep_io import MANIFEST_FILE, Manifest, combo_key

    cell = tmp_path / "0"
    cell.mkdir()
    write_cell_status(cell, status="completed", combo={"a": 1}, attempt=1)
    (tmp_path / MANIFEST_FILE).write_text("not json at all")
    m = Manifest.from_cell_status(tmp_path, ["a"])
    assert m.cells[combo_key({"a": 1})]["status"] == "completed"


def test_atomic_write_uses_unique_temp_names(tmp_path, monkeypatch):
    """Two writers of the same file must not share a temp name -- a fixed
    <path>.tmp lets concurrent processes truncate each other mid-write and
    rename half-written content over the target."""
    from pathlib import Path

    from mushin._sweep_io import _atomic_write_json

    seen = []
    orig = Path.replace

    def spy(self, target):
        seen.append(self.name)
        return orig(self, target)

    monkeypatch.setattr(Path, "replace", spy)
    _atomic_write_json(tmp_path / "m.json", {"a": 1})
    _atomic_write_json(tmp_path / "m.json", {"a": 2})
    assert len(seen) == 2 and seen[0] != seen[1]
    assert not list(tmp_path.glob("*.tmp"))  # no residue


def test_metrics_sidecar_roundtrips_infinities(tmp_path):
    """±Inf metric values must survive the sidecar round-trip with their sign
    (not collapse to NaN), including inside nested containers, while the file
    stays strict JSON (no Infinity/NaN literals)."""
    import json
    import math

    from mushin._sweep_io import (
        METRICS_FILE,
        read_metrics_sidecar,
        write_metrics_sidecar,
    )

    write_metrics_sidecar(
        tmp_path,
        {
            "pos": float("inf"),
            "neg": float("-inf"),
            "nan": float("nan"),
            "nested": {"v": [1.0, float("inf")]},
        },
    )
    back = read_metrics_sidecar(tmp_path)
    assert back is not None
    assert back["pos"] == math.inf
    assert back["neg"] == -math.inf
    assert back["nan"] != back["nan"]  # NaN still round-trips to NaN
    assert back["nested"]["v"] == [1.0, math.inf]

    def _reject_constant(name):  # Infinity/NaN literals are not strict JSON
        raise AssertionError(f"non-strict JSON literal in sidecar: {name}")

    json.loads((tmp_path / METRICS_FILE).read_text(), parse_constant=_reject_constant)


def test_user_dict_resembling_nonfinite_tag_is_not_corrupted(tmp_path):
    """Only the exact internal marker values decode to ±Inf — a user metric
    dict that merely shares the tag key must round-trip unchanged."""
    from mushin._sweep_io import read_metrics_sidecar, write_metrics_sidecar

    write_metrics_sidecar(tmp_path, {"weird": {"__mushin_nonfinite__": "hello"}})
    back = read_metrics_sidecar(tmp_path)
    assert back == {"weird": {"__mushin_nonfinite__": "hello"}}


def test_a_status_write_waits_for_a_reader_to_let_go(tmp_path):
    """On Windows a replace fails while another process has the target open
    (#357): the write waits for the reader rather than fail the cell. On other
    systems the replace never waited."""
    import threading

    from mushin._resume import STATUS_FILE, read_cell_status, write_cell_status

    write_cell_status(tmp_path, status="running", combo={"x": 1}, attempt=1)
    reader = open(tmp_path / STATUS_FILE, encoding="utf-8")  # noqa: SIM115
    timer = threading.Timer(0.3, reader.close)
    timer.start()
    try:
        write_cell_status(tmp_path, status="completed", combo={"x": 1}, attempt=1)
    finally:
        timer.join()
        reader.close()
    assert read_cell_status(tmp_path)["status"] == "completed"


def test_a_replace_is_retried_only_on_windows(tmp_path, monkeypatch):
    from mushin import _sweep_io

    calls = []
    real = type(tmp_path).replace

    def held_twice(self, target):
        calls.append(target)
        if len(calls) <= 2:
            raise PermissionError(13, "Access is denied")
        return real(self, target)

    monkeypatch.setattr(type(tmp_path), "replace", held_twice)
    monkeypatch.setattr(_sweep_io.time, "sleep", lambda s: None)
    monkeypatch.setattr(_sweep_io.sys, "platform", "win32")
    _sweep_io._atomic_write_text(tmp_path / "f.json", "{}")
    assert (tmp_path / "f.json").read_text() == "{}" and len(calls) == 3

    calls.clear()
    monkeypatch.setattr(_sweep_io.sys, "platform", "linux")
    with pytest.raises(PermissionError):
        _sweep_io._atomic_write_text(tmp_path / "g.json", "{}")
    assert len(calls) == 1
