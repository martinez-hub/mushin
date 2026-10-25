# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
import pytest


def test_provenance_written_per_job(tmp_path):
    import json

    from mushin import multirun
    from mushin.workflows import MultiRunMetricsWorkflow

    class W(MultiRunMetricsWorkflow):
        @staticmethod
        def task(x):
            return dict(y=float(x))

    wf = W()
    wf.run(x=multirun([1, 2]), working_dir=str(tmp_path / "s"))
    d = wf.multirun_working_dirs[0]
    prov = json.loads((d / "mushin_provenance.json").read_text())
    assert "python" in prov and "packages" in prov and "git" in prov
    assert prov["packages"]["mushin-py"]
    assert "sha" in prov["git"]


def test_capture_provenance_no_git(tmp_path):
    from mushin._provenance import capture

    p = capture()
    assert "sha" in p["git"]  # must not raise even if git absent


def test_git_captured_once_per_sweep_not_per_cell(monkeypatch, tmp_path):
    # Regression / perf: the sweep-constant provenance (git state) is captured
    # ONCE per run(), not per cell — an N-cell sweep must not spawn 3N git
    # subprocesses. Every cell's record still carries the full git data.
    import json

    from mushin import _provenance, multirun
    from mushin.workflows import MultiRunMetricsWorkflow

    calls = {"n": 0}
    real_git = _provenance._git

    def counting_git():
        calls["n"] += 1
        return real_git()

    monkeypatch.setattr(_provenance, "_git", counting_git)

    class W(MultiRunMetricsWorkflow):
        @staticmethod
        def task(x):
            return dict(y=float(x))

    wf = W()
    wf.run(x=multirun([0, 1, 2, 3, 4]), working_dir=str(tmp_path / "s"))

    assert calls["n"] == 1  # captured once for the whole 5-cell sweep
    # ...and every cell's provenance still has the git sha
    for d in wf.multirun_working_dirs:
        prov = json.loads((d / "mushin_provenance.json").read_text())
        assert "sha" in prov["git"]
        assert prov["packages"]["mushin-py"]


def test_capture_records_this_process_accelerator():
    """GPU numerics depend on the CUDA/cuDNN build and the physical device,
    not just the torch wheel version -- the provenance record must carry them
    (None values on CPU-only builds, but the keys must exist) once torch is
    loaded in the process. The record is this process's view, whatever the
    sweep-constant base from another process carried."""
    import torch  # noqa: F401 - the accelerator is recorded only with torch loaded

    from mushin._provenance import capture, capture_base

    assert "accelerator" not in capture_base()
    acc = capture()["accelerator"]
    assert set(acc) == {"cuda", "cudnn", "device"}
    login_node = {"accelerator": {"cuda": "11.8", "cudnn": 8, "device": "login"}}
    assert capture(base=login_node)["accelerator"] == acc


def test_accelerator_does_not_import_torch_for_a_sweep_without_it(
    tmp_path, monkeypatch
):
    """A process that has not imported torch has no accelerator numerics to
    record, and importing torch to learn that is the dominant cost of a
    provenance capture; the record says "not recorded" (null) and torch stays
    unimported."""
    import json
    import sys

    from mushin import _provenance

    # An import attempt is observed at the finders, not through sys.modules:
    # re-importing torch over its loaded extension fails, and a failed import
    # leaves sys.modules as it was, which would look like no attempt.
    attempted = []

    class Recorder:
        @staticmethod
        def find_spec(name, path=None, target=None):
            if name == "torch" or name.startswith("torch."):
                attempted.append(name)
            return None

    monkeypatch.setattr(sys, "meta_path", [Recorder, *sys.meta_path])
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    assert _provenance._accelerator() is None
    # The parent's placeholder (a login node with torch, say) is not kept.
    assert (
        _provenance.capture(base={"accelerator": {"device": "x"}})["accelerator"]
        is None
    )
    # record_accelerator leaves the cell's record as written, and does not
    # import torch either.
    record = json.dumps({"git": {"sha": "x"}, "accelerator": None})
    (tmp_path / "mushin_provenance.json").write_text(record)
    monkeypatch.setattr(_provenance, "_ACCELERATOR_SEEN", {})
    _provenance.record_accelerator(tmp_path, None)
    assert (tmp_path / "mushin_provenance.json").read_text() == record
    assert attempted == []
    # Nothing is cached either: a later cell whose task imports torch must
    # still get its accelerator recorded.
    assert _provenance._ACCELERATOR_SEEN == {}


def test_run_does_not_import_torch_for_a_sweep_without_it(tmp_path):
    """The whole run() path, in a fresh interpreter: a sweep whose task never
    imports torch finishes without mushin importing it, and records a null
    accelerator. Guards the run path, not just the provenance helpers."""
    import subprocess
    import sys

    code = f"""
import json, sys
from mushin import multirun
from mushin.workflows import MultiRunMetricsWorkflow

class W(MultiRunMetricsWorkflow):
    @staticmethod
    def task(x):
        return dict(y=float(x))

wf = W()
wf.run(x=multirun([0, 1]), working_dir={str(tmp_path / "s")!r})
prov = json.loads((wf.multirun_working_dirs[1] / "mushin_provenance.json").read_text())
print("torch" in sys.modules, json.dumps(prov["accelerator"]))
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr[-2000:]
    # Hydra logs its launches to stdout; the probe's line comes last.
    assert result.stdout.strip().splitlines()[-1] == "False null"


def test_record_accelerator_rewrites_only_that_field(tmp_path, monkeypatch):
    """The cell's process rewrites the record's accelerator, keeping every
    other field, atomically, and does nothing when the record is missing (the
    best-effort write before the task failed). It never initializes CUDA: the
    device is named only once the task has, and each CUDA state's answer is
    queried once per process."""
    import json

    import torch

    from mushin import _provenance

    monkeypatch.setattr(_provenance, "_ACCELERATOR_SEEN", {})
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    # An NVIDIA box, wherever the test runs.
    monkeypatch.setattr(
        torch.backends.mps, "is_available", lambda: False, raising=False
    )
    # The query is best-effort (its exceptions are swallowed), so the calls
    # are recorded rather than made to raise.
    named, probed = [], []

    def get_device_name(i):
        named.append(i)
        return f"NVIDIA H100 #{i}"

    def is_available():
        probed.append(True)
        return True

    monkeypatch.setattr(torch.cuda, "get_device_name", get_device_name)
    monkeypatch.setattr(torch.cuda, "is_available", is_available)
    # A plain write leaves no residue either; the atomic one is observed by
    # the writer it goes through.
    atomic = []
    real_atomic_write_text = _provenance._atomic_write_text

    def atomic_write_text(path, text, **kwargs):
        atomic.append(path)
        real_atomic_write_text(path, text, **kwargs)

    monkeypatch.setattr(_provenance, "_atomic_write_text", atomic_write_text)
    before = {"git": {"sha": "x"}, "accelerator": None}
    (tmp_path / "mushin_provenance.json").write_text(json.dumps(before))
    _provenance.record_accelerator(tmp_path, None)
    after = json.loads((tmp_path / "mushin_provenance.json").read_text())
    assert after["git"] == {"sha": "x"}
    assert set(after["accelerator"]) == {"cuda", "cudnn", "device"}
    assert after["accelerator"]["device"] is None  # CUDA not initialized: no name
    assert named == []  # get_device_name would have initialized it
    assert probed == []  # is_available alone fixes the visible devices
    # Cached for the uninitialized state only: a later cell may initialize it.
    assert set(_provenance._ACCELERATOR_SEEN) == {False}

    # The task initialized CUDA: the device is named (device 0, the pinned GPU,
    # whatever current_device says), and the answer cached for that state.
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 3)
    _provenance.record_accelerator(tmp_path, None)
    after = json.loads((tmp_path / "mushin_provenance.json").read_text())
    assert after["accelerator"]["device"] == "NVIDIA H100 #0"
    assert named == [0]
    assert _provenance._ACCELERATOR_SEEN[True] == after["accelerator"]
    assert atomic == [tmp_path / "mushin_provenance.json"] * 2  # both rewrites
    assert not list(tmp_path.glob("*.tmp"))  # and no residue

    # Cached: a later cell reuses the answer without querying torch again...
    (tmp_path / "mushin_provenance.json").write_text(json.dumps(before))
    _provenance.record_accelerator(tmp_path, None)
    after = json.loads((tmp_path / "mushin_provenance.json").read_text())
    assert after["accelerator"]["device"] == "NVIDIA H100 #0"
    assert named == [0]
    # ...gets it at its first write, so there is nothing to rewrite...
    record = _provenance.capture(base=before)
    assert record["accelerator"] == after["accelerator"]
    # The record is the caller's to change; the cache is not.
    record["accelerator"]["device"] = "changed by the caller"
    assert _provenance._ACCELERATOR_SEEN[True]["device"] == "NVIDIA H100 #0"
    # ...and told so, does not even read the record back.
    stale = json.dumps({**before, "accelerator": {"device": "stale"}})
    (tmp_path / "mushin_provenance.json").write_text(stale)
    _provenance.record_accelerator(tmp_path, after["accelerator"])
    assert (tmp_path / "mushin_provenance.json").read_text() == stale
    # A first write that failed leaves whatever is there (a prior sweep's
    # record in a reused directory) alone, rather than stamping it.
    _provenance.record_accelerator(tmp_path, _provenance.NOT_WRITTEN)
    assert (tmp_path / "mushin_provenance.json").read_text() == stale

    _provenance.record_accelerator(tmp_path / "no-such-cell", None)
    assert not (tmp_path / "no-such-cell").exists()


def test_sweep_records_the_accelerator_the_cell_saw_after_its_task(
    tmp_path, monkeypatch
):
    """The accelerator in each cell's record is the one the cell's process
    reports after the task ran, for a completed and for a failed cell: a task
    that imports torch inside its body, or that runs on another node than the
    one that called run(), records what it used."""
    import json

    import torch  # noqa: F401 - record_accelerator acts only with torch loaded

    from mushin import _provenance, multirun
    from mushin.workflows import MultiRunMetricsWorkflow

    def seen_by(x):
        return {"cuda": "12.8", "cudnn": 91000, "device": f"NVIDIA H100 for x={x}"}

    class W(MultiRunMetricsWorkflow):
        @staticmethod
        def task(x):
            # What a task body's `import torch` changes: the process's view
            # of the accelerator, after capture_base ran in the parent. Each
            # cell sees its own value, so a record that merely carries the
            # previous cell's cached answer from its first write is caught.
            monkeypatch.setattr(_provenance, "_ACCELERATOR_SEEN", {})
            monkeypatch.setattr(_provenance, "_accelerator", lambda: seen_by(x))
            if x == 1:
                raise RuntimeError("boom")
            return dict(y=float(x))

    wf = W()
    wf.run(x=multirun([0, 1]), working_dir=str(tmp_path / "s"), on_error="nan")
    for x, d in zip([0, 1], wf.multirun_working_dirs, strict=True):
        prov = json.loads((d / "mushin_provenance.json").read_text())
        assert prov["accelerator"] == seen_by(x), d
        assert "sha" in prov["git"]


def test_provenance_keeps_non_finite_config_values(tmp_path, monkeypatch):
    """The record is written atomically but as plain JSON, not through the
    metrics normalizer: a config value of inf stays `Infinity` and reads back
    as inf, as it always did (the normalizer would store a marker dict)."""
    import json

    import torch  # noqa: F401 - so record_accelerator has something to record
    from omegaconf import OmegaConf

    from mushin import _provenance
    from mushin._provenance import record_accelerator, write_provenance

    atomic = []
    real_atomic_write_text = _provenance._atomic_write_text

    def atomic_write_text(path, text, **kwargs):
        atomic.append(path)
        real_atomic_write_text(path, text, **kwargs)

    monkeypatch.setattr(_provenance, "_atomic_write_text", atomic_write_text)

    path = tmp_path / "mushin_provenance.json"
    cfg = OmegaConf.create({"clip": float("inf"), "lr": 0.1})
    written = write_provenance(tmp_path, config=cfg)
    prov = json.loads(path.read_text())
    assert prov["config"] == {"clip": float("inf"), "lr": 0.1}
    # The return value is what the runner hands record_accelerator to skip the
    # read-back: the accelerator the record now holds.
    assert isinstance(written, dict) and written == prov["accelerator"]
    # A rewrite (here forced by blanking the accelerator) keeps it too.
    path.write_text(json.dumps({**prov, "accelerator": None}))
    record_accelerator(tmp_path, None)
    prov = json.loads(path.read_text())
    assert prov["accelerator"] is not None
    assert prov["config"]["clip"] == float("inf")
    assert atomic == [path, path]  # the write and the rewrite, both atomic
    assert not list(tmp_path.glob("*.tmp"))
    # A cell directory that is not there is not invented: the write raises
    # into the runner's best-effort guard, as it always did.
    with pytest.raises(FileNotFoundError):
        write_provenance(tmp_path / "gone", config=cfg)
    assert not (tmp_path / "gone").exists()


def test_cell_in_a_worker_without_torch_records_null_even_if_the_driver_has_it(
    tmp_path,
):
    """Out of process, the record is the worker's view, never the driver's: a
    driver that imported torch (this test process has) running a task that
    does not, under joblib, writes `null` in every cell. Otherwise a login
    node's torch view would stand in for a cell that never used torch, and
    diff would compare it against real GPU records."""
    import json

    import torch  # noqa: F401 - the driver has torch loaded

    from mushin import multirun
    from mushin.workflows import MultiRunMetricsWorkflow

    class W(MultiRunMetricsWorkflow):
        @staticmethod
        def task(x):
            import sys

            return dict(torch_loaded=float("torch" in sys.modules))

    # loky keeps its worker processes across Parallel calls in this process,
    # so an earlier joblib test with a torch-importing task would hand over
    # workers that already loaded torch. Start from fresh ones.
    from joblib.externals.loky import get_reusable_executor

    get_reusable_executor().shutdown(wait=True, kill_workers=True)
    wf = W()
    wf.run(
        x=multirun([0, 1]),
        launcher="joblib",
        launcher_config={"n_jobs": 2},
        working_dir=str(tmp_path / "s"),
    )
    for d in wf.multirun_working_dirs:
        metrics = json.loads((d / "mushin_metrics.json").read_text())
        assert metrics["torch_loaded"] == 0.0, "the worker loaded torch; test moot"
        prov = json.loads((d / "mushin_provenance.json").read_text())
        assert prov["accelerator"] is None, d


def test_failed_device_query_is_not_cached(tmp_path, monkeypatch):
    """With CUDA initialized the device is always named, so a None device
    there is a query that failed; the answer is recorded for this cell but
    not kept for every later cell of the worker."""
    import json

    import torch

    from mushin import _provenance

    monkeypatch.setattr(_provenance, "_ACCELERATOR_SEEN", {})
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    monkeypatch.setattr(
        torch.backends.mps, "is_available", lambda: False, raising=False
    )

    def broken(i):
        raise RuntimeError("CUDA error: device-side assert triggered")

    monkeypatch.setattr(torch.cuda, "get_device_name", broken)
    path = tmp_path / "mushin_provenance.json"
    path.write_text(json.dumps({"accelerator": None}))
    _provenance.record_accelerator(tmp_path, None)
    assert json.loads(path.read_text())["accelerator"]["device"] is None
    assert _provenance._ACCELERATOR_SEEN == {}

    monkeypatch.setattr(torch.cuda, "get_device_name", lambda i: "NVIDIA H100")
    _provenance.record_accelerator(tmp_path, None)
    assert json.loads(path.read_text())["accelerator"]["device"] == "NVIDIA H100"
    assert _provenance._ACCELERATOR_SEEN[True]["device"] == "NVIDIA H100"


def test_representative_record_prefers_a_cell_with_an_accelerator(tmp_path):
    """wf.provenance, show and diff stand one cell's record for the sweep. The
    accelerator varies per cell now (a first cell that failed before its task
    imported torch has none), so the first record with one is preferred, within
    the first few cells; a sweep without any keeps its first record."""
    import json

    from mushin._provenance import representative_record
    from mushin._show import _read_provenance

    def cell(i, record):
        d = tmp_path / str(i)
        d.mkdir()
        (d / "mushin_provenance.json").write_text(json.dumps(record))
        return d / "mushin_provenance.json"

    gpu = {"git": {"sha": "b"}, "accelerator": {"cuda": "12.8", "device": "H100"}}
    paths = [
        cell(0, {"git": {"sha": "a"}, "accelerator": None}),
        cell(1, ["not a record"]),
        cell(2, gpu),
    ]
    (tmp_path / "3").mkdir()  # a cell without a record
    assert representative_record(paths) == gpu
    assert _read_provenance(tmp_path) == gpu
    # Only the first few cells are looked at: a sweep without torch is not
    # read whole for a record it does not have.
    far = [
        cell(i, {"git": {"sha": str(i)}, "accelerator": None}) for i in range(10, 30)
    ]
    assert representative_record(far + [paths[2]]) == {
        "git": {"sha": "10"},
        "accelerator": None,
    }
    # Cells without a record count toward that bound too.
    missing = [tmp_path / "3" / "mushin_provenance.json"] * 10
    assert representative_record(missing + [paths[2]]) is None
    assert representative_record([]) is None
    assert representative_record(missing[:1]) is None


def test_env_snapshot_never_overwrites_prior_run(tmp_path):
    """A resume runs in a possibly-different environment; it must not
    overwrite the snapshot recorded for the cells that ran earlier."""
    from mushin.workflows import _write_env_snapshot

    _write_env_snapshot(tmp_path)
    first = tmp_path / "mushin_env.txt"
    assert first.exists()
    first.write_text("SENTINEL-ORIGINAL-ENV")
    _write_env_snapshot(tmp_path)
    assert first.read_text() == "SENTINEL-ORIGINAL-ENV"
    siblings = sorted(p.name for p in tmp_path.glob("mushin_env*.txt"))
    assert len(siblings) == 2  # the new snapshot landed beside, not over, it


def test_env_interpolated_secret_not_resolved_into_provenance(monkeypatch):
    # A `${oc.env:...}` interpolation must NOT be resolved into the record —
    # otherwise a secret env var is baked into mushin_provenance.json.
    import json

    from omegaconf import OmegaConf

    from mushin._provenance import capture

    monkeypatch.setenv("MUSHIN_TEST_SECRET", "sk-live-should-not-appear")
    cfg = OmegaConf.create({"token": "${oc.env:MUSHIN_TEST_SECRET}", "lr": 0.1})
    rec = capture(config=cfg)
    blob = json.dumps(rec)
    assert "sk-live-should-not-appear" not in blob
    assert rec["config"]["lr"] == 0.1  # non-secret values preserved


def test_literal_secret_valued_keys_are_redacted():
    from omegaconf import OmegaConf

    from mushin._provenance import capture

    cfg = OmegaConf.create(
        {
            "api_key": "sk-literal-secret",
            "nested": {"auth_token": "hf_abcdef", "width": 8},
            "lr": 0.1,
        }
    )
    rec = capture(config=cfg)
    assert rec["config"]["api_key"] == "***REDACTED***"
    assert rec["config"]["nested"]["auth_token"] == "***REDACTED***"
    assert rec["config"]["nested"]["width"] == 8  # non-secret preserved
    assert rec["config"]["lr"] == 0.1


def test_secret_shaped_value_under_innocent_key_is_redacted():
    from omegaconf import OmegaConf

    from mushin._provenance import capture

    cfg = OmegaConf.create({"note": "sk-ABCDEFGHIJKLMNOPQRSTUV", "ok": "hello"})
    rec = capture(config=cfg)
    assert rec["config"]["note"] == "***REDACTED***"
    assert rec["config"]["ok"] == "hello"


def test_mcp_get_provenance_serves_redacted_config(tmp_path):
    import json

    from mushin.mcp.server import _get_provenance

    d = tmp_path / "exp" / "0"
    d.mkdir(parents=True)
    (d / ".hydra").mkdir()
    (d / "mushin_provenance.json").write_text(
        json.dumps({"config": {"api_key": "***REDACTED***", "lr": 0.1}})
    )
    out = _get_provenance(tmp_path / "exp", include_config=True)
    served = json.dumps(out)
    assert "sk-" not in served
    assert "***REDACTED***" in served


def test_git_dirty_is_unknown_when_status_fails(monkeypatch):
    """A failed/timed-out `git status` must record dirty=None (unknown), never
    a confident dirty=False the tool could not verify."""
    import subprocess

    from mushin import _provenance

    real_run = subprocess.run

    def fake_run(args, **kwargs):
        if "status" in args:
            raise subprocess.TimeoutExpired(args, 5)
        return real_run(args, **kwargs)

    monkeypatch.setattr(subprocess, "run", fake_run)
    out = _provenance._git()
    assert out["sha"] is not None  # we ARE in a git repo
    assert out["dirty"] is None  # ...but cleanliness could not be verified


def test_accelerator_records_mps_when_available(monkeypatch):
    """On Apple Silicon (no CUDA), the accelerator record must not be all-None:
    an MPS-trained run's provenance should say what it ran on."""
    import torch

    from mushin import _provenance

    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True, raising=False)
    out = _provenance._accelerator()
    assert out["device"] is not None and out["device"].startswith("mps")


def test_apple_chip_parses_only_apple_brands(monkeypatch):
    """_apple_chip must return the chip only for a successful sysctl reporting
    an Apple chip — an Intel CPU brand (Intel Macs can have MPS via a Metal
    GPU) or a failed command must yield None, not a wrong hardware label."""
    import subprocess as _subprocess
    from types import SimpleNamespace

    from mushin import _provenance

    def fake(args, **kwargs):
        return SimpleNamespace(returncode=0, stdout="Apple M3\n")

    monkeypatch.setattr(_subprocess, "run", fake)
    assert _provenance._apple_chip() == "Apple M3"

    monkeypatch.setattr(
        _subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(
            returncode=0, stdout="Intel(R) Core(TM) i9-9880H\n"
        ),
    )
    assert _provenance._apple_chip() is None

    monkeypatch.setattr(
        _subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(returncode=1, stdout="Apple M3\n"),
    )
    assert _provenance._apple_chip() is None


def test_accelerator_mps_labels(monkeypatch):
    """The MPS device label must be the real Apple chip when known, the CPU
    architecture otherwise — never a hardcoded 'Apple Silicon' claim that
    could mislabel an Intel Mac with a Metal GPU."""
    import platform

    import torch

    from mushin import _provenance

    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True, raising=False)

    monkeypatch.setattr(_provenance, "_apple_chip", lambda: "Apple M3")
    assert _provenance._accelerator()["device"] == "mps (Apple M3)"

    monkeypatch.setattr(_provenance, "_apple_chip", lambda: None)
    assert _provenance._accelerator()["device"] == f"mps ({platform.machine()})"


def test_accelerator_device_none_without_cuda_or_mps(monkeypatch):
    import torch

    from mushin import _provenance

    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    monkeypatch.setattr(
        torch.backends.mps, "is_available", lambda: False, raising=False
    )
    assert _provenance._accelerator()["device"] is None
