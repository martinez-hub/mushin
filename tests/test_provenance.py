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
