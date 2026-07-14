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
