# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
from mushin._resume import (
    STATUS_FILE,
    ResumeContext,
    discover_last_ckpt,
    read_cell_status,
    write_cell_status,
)


def test_write_then_read_cell_status(tmp_path):
    write_cell_status(tmp_path, status="running", combo={"seed": 0}, attempt=1)
    got = read_cell_status(tmp_path)
    assert got["status"] == "running"
    assert got["attempt"] == 1
    assert got["combo"] == {"seed": 0}
    assert (tmp_path / STATUS_FILE).exists()


def test_read_cell_status_missing_or_corrupt_returns_none(tmp_path):
    assert read_cell_status(tmp_path) is None
    (tmp_path / STATUS_FILE).write_text("{not json")
    assert read_cell_status(tmp_path) is None


def test_discover_last_ckpt_prefers_last_then_newest(tmp_path):
    assert discover_last_ckpt(tmp_path) is None
    (tmp_path / "epoch=0.ckpt").write_text("a")
    (tmp_path / "epoch=1.ckpt").write_text("b")
    newest = discover_last_ckpt(tmp_path)
    assert newest is not None and newest.suffix == ".ckpt"
    (tmp_path / "last.ckpt").write_text("c")
    assert discover_last_ckpt(tmp_path).name == "last.ckpt"


def test_resume_context_is_frozen():
    rc = ResumeContext(dir=None, is_resume=False, last_ckpt=None, attempt=1)
    try:
        rc.is_resume = True  # type: ignore[misc]
    except Exception as e:
        assert "cannot assign" in str(e).lower() or "frozen" in str(e).lower()
    else:  # pragma: no cover
        raise AssertionError("ResumeContext must be frozen")


def test_build_resume_context_combo_match_guard(tmp_path):
    from mushin._resume import build_resume_context, write_cell_status

    # first attempt of combo {"seed": 0}: fresh
    rc = build_resume_context(tmp_path, {"seed": 0})
    assert rc.is_resume is False and rc.attempt == 1 and rc.last_ckpt is None

    # a prior attempt of the SAME combo left a checkpoint -> resume it
    write_cell_status(tmp_path, status="failed", combo={"seed": 0}, attempt=1)
    (tmp_path / "last.ckpt").write_text("state")
    rc = build_resume_context(tmp_path, {"seed": 0})
    assert rc.is_resume is True and rc.attempt == 2
    assert rc.last_ckpt is not None and rc.last_ckpt.name == "last.ckpt"

    # SAME dir now queried for a DIFFERENT combo (numeric dir reused after a grid
    # change) -> must NOT resume or surface the other cell's checkpoint
    rc = build_resume_context(tmp_path, {"seed": 9})
    assert rc.is_resume is False and rc.attempt == 1 and rc.last_ckpt is None
