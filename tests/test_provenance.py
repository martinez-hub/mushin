# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT


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


def test_capture_base_records_accelerator():
    """GPU numerics depend on the CUDA/cuDNN build and the physical device,
    not just the torch wheel version -- the provenance record must carry them
    (None values on CPU-only builds, but the keys must exist)."""
    from mushin._provenance import capture_base

    base = capture_base()
    acc = base["accelerator"]
    assert set(acc) == {"cuda", "cudnn", "device"}


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
