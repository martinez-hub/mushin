# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
import json
import os
from pathlib import Path

import pytest

from mushin._resume import (
    STATUS_FILE,
    ResumeContext,
    build_resume_context,
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
    # Never one field without the other: a reload needs both to skip hydra.yaml.
    write_cell_status(tmp_path, status="running", combo={}, attempt=1, job_num=2)
    got = read_cell_status(tmp_path)
    assert "job_num" not in got and "choices" not in got

    choices = {"model": "small", "hydra/env": None}
    write_cell_status(
        tmp_path, status="running", combo={}, attempt=1, job_num=2, choices=choices
    )
    got = read_cell_status(tmp_path)
    assert got["job_num"] == 2 and got["choices"] == choices
    assert sidecar_job_fields(tmp_path) == (2, choices)

    # The writer applies the reader's rule, and never coerces: a bool is not
    # job number 1, and a value that is not an option name is not recorded.
    for bad in [dict(job_num=True, choices={}), dict(job_num=2, choices={"m": 3})]:
        write_cell_status(tmp_path, status="running", combo={}, attempt=1, **bad)
        got = read_cell_status(tmp_path)
        assert "job_num" not in got and "choices" not in got, bad


def test_sidecar_job_fields_defers_to_a_hydra_yaml_it_cannot_examine(
    tmp_path, monkeypatch
):
    """Only a missing hydra.yaml means there is nothing to defer to. One that
    is there but cannot be examined sends the reload to the hydra.yaml path,
    which reports the problem as it always did.

    The failure is injected. An on-disk fixture (a regular file where the
    `.hydra` directory should be) is NotADirectoryError on POSIX but
    FileNotFoundError on Windows, where it reads as a missing hydra.yaml."""
    write_cell_status(
        tmp_path, status="completed", combo={}, attempt=1, job_num=5, choices={}
    )
    hydra_yaml = tmp_path / "hydra.yaml"
    hydra_yaml.write_text("hydra:\n  job:\n    num: 0\n")
    real_stat = Path.stat

    def stat(self, *args, **kwargs):
        if self.name == hydra_yaml.name:
            raise PermissionError(13, "Permission denied", str(self))
        return real_stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", stat)
    assert sidecar_job_fields(tmp_path, hydra_yaml) is None


def test_sidecar_job_fields_without_the_file_it_read(tmp_path, monkeypatch):
    """A sidecar removed between its read and the age check (a concurrent
    resume or gc) has no claim over the hydra.yaml that is there."""
    from mushin import _resume

    monkeypatch.setattr(
        _resume, "read_cell_status", lambda d: {"job_num": 1, "choices": {}}
    )
    hydra_yaml = tmp_path / "hydra.yaml"
    hydra_yaml.write_text("hydra:\n  job:\n    num: 0\n")
    assert sidecar_job_fields(tmp_path, hydra_yaml) is None
    assert sidecar_job_fields(tmp_path) == (1, {})  # no hydra.yaml to defer to


@pytest.mark.parametrize(
    "job_num, choices",
    [
        ("3", {}),  # a string job number
        (True, {}),  # JSON true reads as a Python bool, an int subclass
        (True, {"model": "small"}),
        (3.0, {}),
        (None, {}),
        (3, ["adam"]),  # choices must map group names to option names
        (3, {"optimizer": 1}),
        (3, {"optimizer": "adam", "model": 2.5}),
        (3, None),  # mushin never writes a number without the choices
        (3, {"model": "???"}),  # not option names, as the hydra.yaml read has it
        (3, {"model": "${oc.env:MODEL}"}),
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
    [(0, {}), (3, {"optimizer": None, "model": "small"})],
)
def test_sidecar_job_fields_accepts_what_hydra_writes(tmp_path, job_num, choices):
    payload = {"status": "completed", "combo": {}, "attempt": 1}
    payload["job_num"], payload["choices"] = job_num, choices
    (tmp_path / STATUS_FILE).write_text(json.dumps(payload))
    assert sidecar_job_fields(tmp_path) == (job_num, choices)


def test_no_sidecar_reads_as_none(tmp_path):
    assert read_cell_status(tmp_path) is None
    assert sidecar_job_fields(tmp_path) is None


def test_sidecar_job_fields_defers_to_a_newer_hydra_yaml(tmp_path):
    """Hydra writes hydra.yaml before the task writes the sidecar, so a
    sidecar older than hydra.yaml is a previous sweep's, and is not used. A
    missing hydra.yaml, or one of the same age, leaves the sidecar in use."""
    write_cell_status(
        tmp_path, status="completed", combo={}, attempt=1, job_num=5, choices={}
    )
    hydra_yaml = tmp_path / "hydra.yaml"
    assert sidecar_job_fields(tmp_path, hydra_yaml) == (5, {})  # no such file

    hydra_yaml.write_text("hydra:\n  job:\n    num: 0\n")
    t = (tmp_path / STATUS_FILE).stat().st_mtime_ns
    os.utime(hydra_yaml, ns=(t + 10**9, t + 10**9))
    assert sidecar_job_fields(tmp_path, hydra_yaml) is None
    os.utime(hydra_yaml, ns=(t, t))
    assert sidecar_job_fields(tmp_path, hydra_yaml) == (5, {})
    os.utime(hydra_yaml, ns=(t - 10**9, t - 10**9))
    assert sidecar_job_fields(tmp_path, hydra_yaml) == (5, {})


@pytest.mark.parametrize(
    "content",
    [
        b"{not json",
        b"\xff\xfe{}",  # not UTF-8
        b'{"job_num": ' + b"9" * 5000 + b"}",  # past Python's int digit limit
        b"[" * 100_000,  # nested past the recursion limit
    ],
    # pytest would otherwise name the cases after the bytes themselves (5,000
    # and 100,000 characters), which the Windows CI jobs could not handle.
    ids=["not-json", "not-utf8", "huge-int", "deep-nesting"],
)
def test_read_cell_status_treats_any_unreadable_file_as_corrupt(tmp_path, content):
    """The MCP server lists directories it did not write; one planted or
    damaged sidecar must not make the whole directory unqueryable."""
    (tmp_path / STATUS_FILE).write_bytes(content)
    assert read_cell_status(tmp_path) is None
    assert sidecar_job_fields(tmp_path) is None


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


# A swept NaN or inf is stored tagged in a cell's record; every reader reads
# it as the value it was swept (#355).


def test_a_combo_reads_back_as_swept(tmp_path):
    import math

    combo = {
        "a": float("inf"),
        "b": -float("inf"),
        "c": float("nan"),
        "d": None,
        "e": [1, float("nan")],
    }
    write_cell_status(tmp_path, status="completed", combo=combo, attempt=1)
    back = read_cell_status(tmp_path)["combo"]
    assert back["a"] == float("inf") and back["b"] == -float("inf")
    assert math.isnan(back["c"]) and back["d"] is None
    assert back["e"][0] == 1 and math.isnan(back["e"][1])
    assert math.isnan(combo["c"])  # the caller's combo is not changed
    # A swept param named like the tag stays a param.
    write_cell_status(
        tmp_path, status="completed", combo={"__mushin_nonfinite__": "inf"}, attempt=1
    )
    assert read_cell_status(tmp_path)["combo"] == {"__mushin_nonfinite__": "inf"}


def test_a_record_too_deep_to_restore_keeps_its_stored_combo(tmp_path):
    value: object = 1
    for _ in range(500):
        value = [value]
    (tmp_path / STATUS_FILE).write_text(
        json.dumps({"status": "completed", "combo": {"x": value}})
    )
    record = read_cell_status(tmp_path)
    assert record is not None and "x" in record["combo"]


class _NoText:
    def __str__(self):
        raise RuntimeError("no text")


@pytest.mark.parametrize(
    "stored, live, same",
    [
        ({"x": float("nan")}, {"x": float("nan")}, True),
        ({"x": None}, {"x": float("nan")}, False),
        ({"x": float("nan")}, {"x": None}, False),
        ({"x": float("inf")}, {"x": float("inf")}, True),
        ({"x": 1}, {"x": True}, False),
        ({"x": 1}, {"x": 1.0}, False),
        ({"x": [1, 2]}, {"x": (1, 2)}, True),
        ({"x": 1}, {"x": _NoText()}, False),
        (None, {}, False),
    ],
)
def test_same_cell_compares_combos_as_stored(stored, live, same):
    from mushin._resume import same_cell

    assert same_cell(stored, live) is same


def test_a_nan_cell_does_not_resume_a_none_cells_checkpoint(tmp_path):
    """A record from a swept None in a reused dir is another cell's: both
    were stored null before NaN was tagged."""
    write_cell_status(tmp_path, status="failed", combo={"x": None}, attempt=1)
    (tmp_path / "last.ckpt").write_text("weights of x=None")
    rc = build_resume_context(tmp_path, {"x": float("nan")})
    assert rc.is_resume is False and rc.last_ckpt is None


@pytest.mark.parametrize("x", [float("inf"), float("nan")], ids=["inf", "nan"])
def test_a_swept_non_finite_cell_resumes_its_own_checkpoint(tmp_path, x):
    import warnings

    from mushin import multirun
    from mushin.workflows import MultiRunMetricsWorkflow

    seen = {}

    class W(MultiRunMetricsWorkflow):
        @staticmethod
        def task(x, mushin_resume=None):
            last = mushin_resume.last_ckpt
            seen[mushin_resume.attempt] = None if last is None else last.read_text()
            if mushin_resume.attempt == 1:
                (Path.cwd() / "last.ckpt").write_text(f"x={x}")
                raise RuntimeError("first attempt fails")
            return dict(v=1.0)

    sweep = dict(x=multirun([x]), working_dir=str(tmp_path))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        W().run(on_error="nan", **sweep)
    W().run(resume=True, **sweep)
    assert seen == {1: None, 2: f"x={x}"}


@pytest.mark.parametrize("x", [float("inf"), float("nan")], ids=["inf", "nan"])
def test_a_resume_without_its_manifest_reuses_a_non_finite_cell(tmp_path, x):
    """The manifest is rebuilt from the cells' records when it is gone (a
    driver killed before its last save): a NaN cell keys as the live one."""
    from mushin import multirun
    from mushin.workflows import MultiRunMetricsWorkflow

    calls = []

    class W(MultiRunMetricsWorkflow):
        @staticmethod
        def task(x):
            calls.append(x)
            return dict(v=1.0)

    W().run(x=multirun([1.0, x]), working_dir=str(tmp_path))
    (tmp_path / "mushin_sweep_manifest.json").unlink()
    calls.clear()
    W().run(x=multirun([1.0, x]), working_dir=str(tmp_path), resume=True)
    assert calls == []
