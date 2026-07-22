# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
"""`run(..., cache_dir=...)` is a content-addressed store of completed cells:
a cell whose resolved config AND task source match a previously-computed cell —
in ANY working_dir — reuses that result instead of recomputing."""

from __future__ import annotations

from hydra_zen import make_config

from mushin import multirun
from mushin.workflows import MultiRunMetricsWorkflow


def test_cache_dir_reuses_cells_across_working_dirs(tmp_path):
    cache = tmp_path / "cache"
    calls = {"n": 0}

    class W(MultiRunMetricsWorkflow):
        @staticmethod
        def task(x):
            calls["n"] += 1
            return dict(v=float(x) * 2)

    W().run(
        x=multirun([1, 2, 3]), cache_dir=str(cache), working_dir=str(tmp_path / "a")
    )
    assert calls["n"] == 3

    calls["n"] = 0
    wf2 = W()  # fresh working_dir, same cache
    wf2.run(
        x=multirun([1, 2, 3]), cache_dir=str(cache), working_dir=str(tmp_path / "b")
    )
    assert calls["n"] == 0  # every cell served from the cache
    assert wf2.is_complete
    ds = wf2.to_xarray()
    assert float(ds["v"].sel(x=2)) == 4.0


def test_cache_dir_partial_overlap_runs_only_new_cells(tmp_path):
    cache = tmp_path / "cache"
    calls = {"n": 0}

    class W(MultiRunMetricsWorkflow):
        @staticmethod
        def task(x):
            calls["n"] += 1
            return dict(v=float(x))

    W().run(x=multirun([1, 2]), cache_dir=str(cache), working_dir=str(tmp_path / "a"))
    assert calls["n"] == 2

    calls["n"] = 0
    W().run(
        x=multirun([1, 2, 3, 4]), cache_dir=str(cache), working_dir=str(tmp_path / "b")
    )
    assert calls["n"] == 2  # only x=3 and x=4 were new


def test_cache_dir_misses_on_changed_task_body(tmp_path):
    cache = tmp_path / "cache"

    class W1(MultiRunMetricsWorkflow):
        @staticmethod
        def task(x):
            return dict(v=float(x))

    W1().run(x=multirun([1, 2]), cache_dir=str(cache), working_dir=str(tmp_path / "a"))

    calls = {"n": 0}

    class W2(MultiRunMetricsWorkflow):
        @staticmethod
        def task(x):
            calls["n"] += 1
            return dict(v=float(x) + 100.0)  # different source -> different code hash

    wf2 = W2()
    wf2.run(x=multirun([1, 2]), cache_dir=str(cache), working_dir=str(tmp_path / "b"))
    assert calls["n"] == 2  # changed code -> cache miss -> recomputed
    assert float(wf2.to_xarray()["v"].sel(x=2)) == 102.0


def test_cache_dir_misses_on_changed_config(tmp_path):
    cache = tmp_path / "cache"

    class W(MultiRunMetricsWorkflow):
        @staticmethod
        def task(x, scale):
            return dict(v=float(x) * float(scale))

    W(make_config(scale=1.0)).run(
        x=multirun([1, 2]), cache_dir=str(cache), working_dir=str(tmp_path / "a")
    )

    wf2 = W(make_config(scale=2.0))  # different non-swept config value
    wf2.run(x=multirun([1, 2]), cache_dir=str(cache), working_dir=str(tmp_path / "b"))
    # x=2 recomputed at scale=2 -> 4.0, not the cached scale=1 result of 2.0
    assert float(wf2.to_xarray()["v"].sel(x=2)) == 4.0


def test_cache_dir_is_populated_after_a_run(tmp_path):
    cache = tmp_path / "cache"

    class W(MultiRunMetricsWorkflow):
        @staticmethod
        def task(x):
            return dict(v=float(x))

    W().run(
        x=multirun([1, 2, 3]), cache_dir=str(cache), working_dir=str(tmp_path / "a")
    )
    entries = [d for d in cache.iterdir() if d.is_dir()]
    assert len(entries) == 3
