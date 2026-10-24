# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
import json
import os

import pytest

from mushin._resume import (
    STATUS_FILE,
    ResumeContext,
    discover_last_ckpt,
    read_cell_status,
    sidecar_job_fields,
    write_cell_status,
)


def test_write_then_read_cell_status(tmp_path):
    write_cell_status(tmp_path, status="running", combo={"seed": 0}, attempt=1)
    got = read_cell_status(tmp_path)
    assert got["status"] == "running"
    assert got["attempt"] == 1
    assert got["combo"] == {"seed": 0}
    assert (tmp_path / STATUS_FILE).exists()


def test_write_cell_status_records_job_fields_only_when_known(tmp_path):
    """Outside a Hydra job there is no job number, and the sidecar says
    nothing rather than something made up."""
    write_cell_status(tmp_path, status="running", combo={}, attempt=1)
    got = read_cell_status(tmp_path)
    assert "job_num" not in got and "choices" not in got
    assert sidecar_job_fields(tmp_path) is None

    choices = {"model": "small", "hydra/env": None}
    write_cell_status(
        tmp_path, status="running", combo={}, attempt=1, job_num=2, choices=choices
    )
    got = read_cell_status(tmp_path)
    assert got["job_num"] == 2 and got["choices"] == choices
    assert sidecar_job_fields(tmp_path) == (2, choices)


@pytest.mark.parametrize(
    "job_num, choices",
    [
        ("3", None),  # a string job number
        (True, None),  # JSON true reads as a Python bool, an int subclass
        (3.0, None),
        (None, None),
        (3, ["adam"]),  # choices must map group names to option names
        (3, {"optimizer": 1}),
        (3, {"optimizer": "adam", "model": 2.5}),
    ],
)
def test_sidecar_job_fields_rejects_what_hydra_does_not_write(
    tmp_path, job_num, choices
):
    payload = {"status": "completed", "combo": {}, "attempt": 1}
    payload["job_num"], payload["choices"] = job_num, choices
    (tmp_path / STATUS_FILE).write_text(json.dumps(payload))
    assert sidecar_job_fields(tmp_path) is None


@pytest.mark.parametrize(
    "job_num, choices",
    [(3, None), (0, {}), (3, {"optimizer": None, "model": "small"})],
)
def test_sidecar_job_fields_accepts_what_hydra_writes(tmp_path, job_num, choices):
    payload = {"status": "completed", "combo": {}, "attempt": 1}
    payload["job_num"], payload["choices"] = job_num, choices
    (tmp_path / STATUS_FILE).write_text(json.dumps(payload))
    assert sidecar_job_fields(tmp_path) == (job_num, choices)


def test_sidecar_job_fields_without_a_sidecar(tmp_path):
    assert sidecar_job_fields(tmp_path) is None
    (tmp_path / STATUS_FILE).write_text("{not json")
    assert sidecar_job_fields(tmp_path) is None


def test_sidecar_job_fields_defers_to_a_newer_hydra_yaml(tmp_path):
    """Hydra writes hydra.yaml before the task writes the sidecar, so a
    sidecar older than hydra.yaml is a previous sweep's, and is not used. A
    missing hydra.yaml, or one of the same age, leaves the sidecar in use."""
    write_cell_status(
        tmp_path, status="completed", combo={}, attempt=1, job_num=5, choices=None
    )
    hydra_yaml = tmp_path / "hydra.yaml"
    assert sidecar_job_fields(tmp_path, hydra_yaml) == (5, None)  # no such file

    hydra_yaml.write_text("hydra:\n  job:\n    num: 0\n")
    t = (tmp_path / STATUS_FILE).stat().st_mtime_ns
    os.utime(hydra_yaml, ns=(t + 10**9, t + 10**9))
    assert sidecar_job_fields(tmp_path, hydra_yaml) is None
    os.utime(hydra_yaml, ns=(t, t))
    assert sidecar_job_fields(tmp_path, hydra_yaml) == (5, None)
    os.utime(hydra_yaml, ns=(t - 10**9, t - 10**9))
    assert sidecar_job_fields(tmp_path, hydra_yaml) == (5, None)


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


def test_resume_context_is_public():
    import mushin

    assert mushin.ResumeContext is not None


def test_build_resume_context_tolerates_sidecar_missing_attempt(tmp_path):
    # A malformed / older-schema sidecar (has combo, lacks attempt) must not crash
    # the task wrapper — attempt degrades to 1.
    import json

    from mushin._resume import STATUS_FILE, build_resume_context

    (tmp_path / STATUS_FILE).write_text(
        json.dumps({"status": "failed", "combo": {"seed": 0}})
    )
    rc = build_resume_context(tmp_path, {"seed": 0})
    assert rc.is_resume is True
    assert rc.attempt == 1  # missing attempt -> 0 + 1
