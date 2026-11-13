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


@pytest.mark.skipif(
    __import__("sys").platform != "win32", reason="a reader blocks a replace only there"
)
def test_a_status_write_waits_for_a_reader_to_let_go(tmp_path):
    """On Windows a replace fails while another process has the target open
    (#357): the write waits for the reader rather than fail the cell."""
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


def _held(monkeypatch, times, ops=("replace", "unlink")):
    """`Path.replace` and `Path.unlink` (those in ``ops``) raise as Windows
    does while another process holds the file, ``times`` times; then they
    act."""
    from pathlib import Path

    from mushin import _sweep_io

    calls = []
    real = {"replace": Path.replace, "unlink": Path.unlink}

    def stand_in(name):
        def held(self, *args, **kwargs):
            calls.append(name)
            if len(calls) <= times:
                raise PermissionError(13, "Access is denied")
            return real[name](self, *args, **kwargs)

        return held

    for name in ops:
        monkeypatch.setattr(Path, name, stand_in(name))
    monkeypatch.setattr(_sweep_io, "_HELD_WAITS", (0,) * len(_sweep_io._HELD_WAITS))
    return calls


@pytest.mark.parametrize("windows", [True, False], ids=["windows", "elsewhere"])
def test_a_held_sidecar_is_waited_for_only_on_windows(tmp_path, monkeypatch, windows):
    from mushin import _sweep_io

    monkeypatch.setattr(_sweep_io, "_WINDOWS", windows)
    target = tmp_path / "f.json"
    target.write_text("old")
    calls = _held(monkeypatch, times=2)
    if windows:
        _sweep_io._atomic_write_text(target, "{}")
        assert target.read_text() == "{}" and calls.count("replace") == 3
        calls.clear()
        _sweep_io.remove_sidecar(target)  # held twice more, then deleted
        assert not target.exists() and calls == ["unlink"] * 3
    else:
        with pytest.raises(PermissionError):
            _sweep_io._atomic_write_text(target, "{}")
        assert calls.count("replace") == 1 and target.read_text() == "old"


def test_a_sidecar_held_past_the_waits_fails_and_leaves_no_temp_file(
    tmp_path, monkeypatch
):
    from mushin import _sweep_io

    monkeypatch.setattr(_sweep_io, "_WINDOWS", True)
    # Only the target is held, by its reader: the temp file is not.
    calls = _held(monkeypatch, times=10**6, ops=("replace",))
    with pytest.raises(PermissionError):
        _sweep_io._atomic_write_text(tmp_path / "f.json", "{}")
    assert calls.count("replace") == len(_sweep_io._HELD_WAITS) + 1
    monkeypatch.undo()
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    ("metrics", "whole"),
    [
        (dict(acc=0.9, loss=[1.0, float("inf")], n=3, name="a", ok=True), True),
        (dict(nested=dict(acc=0.5)), True),
        (dict(acc=0.9, lock=__import__("threading").Lock()), False),
        (dict(when=__import__("datetime").date(2026, 1, 1)), False),
        ({1: 0.5}, False),  # a key that is not text reads back as text
    ],
)
def test_write_metrics_sidecar_says_whether_the_file_holds_the_result(
    tmp_path, metrics, whole
):
    from mushin._sweep_io import read_metrics_sidecar_written

    written = write_metrics_sidecar(tmp_path, metrics)
    assert (written is not None) is whole
    if whole:
        assert read_metrics_sidecar_written(tmp_path, written) is not None
        assert read_metrics_sidecar_written(tmp_path, "0" * 16) is None
