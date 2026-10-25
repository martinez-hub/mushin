# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
"""`mushin.best(root, metric)` returns the winning cell of a sweep — offline,
reading the same per-cell sidecars as `mushin.show`."""

from __future__ import annotations

from pathlib import Path

import pytest

import mushin
from mushin import multirun
from mushin.workflows import MultiRunMetricsWorkflow


def _run(wd, *, on_error="raise"):
    class W(MultiRunMetricsWorkflow):
        @staticmethod
        def task(x):
            if on_error == "nan" and x == 5.0:
                raise RuntimeError("boom")
            return dict(score=float(x), other=float(-x))

    W().run(x=multirun([1.0, 5.0, 3.0]), working_dir=str(wd), on_error=on_error)


def test_best_returns_max_cell_by_default(tmp_path):
    wd = tmp_path / "s"
    _run(wd)
    r = mushin.best(wd, "score")
    assert r.combo == {"x": 5.0}
    assert r.value == pytest.approx(5.0)
    assert r.status == "completed"
    assert r.metrics["score"] == pytest.approx(5.0)
    assert Path(r.dir).is_dir()


def test_best_mode_min(tmp_path):
    wd = tmp_path / "s"
    _run(wd)
    r = mushin.best(wd, "score", mode="min")
    assert r.combo == {"x": 1.0}
    assert r.value == pytest.approx(1.0)


def test_best_optimizes_the_named_metric(tmp_path):
    wd = tmp_path / "s"
    _run(wd)
    # `other` == -x, so its max is at x == 1.0
    r = mushin.best(wd, "other")
    assert r.combo == {"x": 1.0}
    assert r.value == pytest.approx(-1.0)


def test_best_ignores_failed_cells(tmp_path):
    # With x==5 failing (NaN), the best completed score is at x==3.
    wd = tmp_path / "s"
    with pytest.warns(UserWarning, match="fail"):
        _run(wd, on_error="nan")
    r = mushin.best(wd, "score")
    assert r.combo == {"x": 3.0}
    assert r.value == pytest.approx(3.0)


def test_best_unknown_metric_raises_with_available(tmp_path):
    wd = tmp_path / "s"
    _run(wd)
    with pytest.raises(ValueError, match="score") as exc:
        mushin.best(wd, "nope")
    assert "nope" in str(exc.value)


def test_best_invalid_mode_raises(tmp_path):
    wd = tmp_path / "s"
    _run(wd)
    with pytest.raises(ValueError, match="mode"):
        mushin.best(wd, "score", mode="highest")


def test_best_missing_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        mushin.best(tmp_path / "nope", "score")


def _run_seeded(wd, *, fail=()):
    """A 2 x 3 (lr x seed) grid where the best SINGLE cell belongs to the
    configuration with the worse MEAN: lr=0.1 scores -5, -5, 10 (mean 0.0) and
    lr=0.2 scores 1, 1, 1 (mean 1.0). ``only0`` is finite at seed 0 only and
    ``signed_inf`` is infinite at seeds 0 and 1. ``fail`` lists (lr, seed)
    cells that raise, which ``on_error="nan"`` records as failed."""
    scores = {0.1: [-5.0, -5.0, 10.0], 0.2: [1.0, 1.0, 1.0]}

    class W(MultiRunMetricsWorkflow):
        @staticmethod
        def task(lr, seed):
            if (lr, seed) in fail:
                raise RuntimeError("boom")
            score = scores[lr][seed]
            return dict(
                score=score,
                loss=-score,
                only0=1.0 if seed == 0 else float("nan"),
                # +inf, -inf, 1.0: a mean over them is undefined (fsum raises)
                signed_inf=[float("inf"), float("-inf"), 1.0][seed],
            )

    W().run(
        lr=multirun([0.1, 0.2]),
        seed=multirun([0, 1, 2]),
        working_dir=str(wd),
        on_error="nan" if fail else "raise",
    )


def _combos(dirs):
    from mushin._resume import read_cell_status

    return [read_cell_status(d)["combo"] for d in dirs]


def _cell_dir(wd, **combo):
    for c in mushin._show._read_cells(wd):
        if all(c["combo"].get(k) == v for k, v in combo.items()):
            return c["dir"]
    raise AssertionError(f"no cell dir for {combo}")


def _touch_newer_than_manifest(wd, path):
    """_read_cells keeps a dir the manifest does not list only when its status
    sidecar is NEWER than the manifest, and a coarse-mtime filesystem can give
    both the same second, so bump it by one."""
    import os

    from mushin._resume import STATUS_FILE
    from mushin._sweep_io import MANIFEST_FILE

    newer = (wd / MANIFEST_FILE).stat().st_mtime + 1
    os.utime(path / STATUS_FILE, (newer, newer))


def test_best_over_seed_picks_the_best_mean_not_the_lucky_cell(tmp_path):
    wd = tmp_path / "s"
    _run_seeded(wd)
    single = mushin.best(wd, "score")
    assert single.combo == {"lr": 0.1, "seed": 2}
    assert single.n == 1
    assert single.dirs == [single.dir]

    r = mushin.best(wd, "score", over="seed")
    assert r.combo == {"lr": 0.2}
    assert r.value == pytest.approx(1.0)
    assert r.status == "completed"
    assert r.n == 3
    assert r.dir is None
    # the winning group's own cells, in job order
    assert _combos(r.dirs) == [{"lr": 0.2, "seed": s} for s in (0, 1, 2)]
    # means of the metrics finite in every cell; `only0` is NaN at seeds 1, 2
    # and `signed_inf` is infinite at seeds 0, 1
    assert r.metrics == {"score": pytest.approx(1.0), "loss": pytest.approx(-1.0)}
    assert "n=3" in repr(r)


def test_best_over_mode_min(tmp_path):
    wd = tmp_path / "s"
    _run_seeded(wd)
    r = mushin.best(wd, "score", mode="min", over="seed")
    assert r.combo == {"lr": 0.1}
    assert r.value == pytest.approx(0.0)


def test_best_over_several_params_averages_everything(tmp_path):
    wd = tmp_path / "s"
    _run_seeded(wd)
    r = mushin.best(wd, "score", over=["seed", "lr", "seed"])
    assert r.combo == {}
    assert r.n == 6
    assert r.value == pytest.approx((-5 - 5 + 10 + 1 + 1 + 1) / 6)


def test_best_over_skips_incomplete_groups_with_a_warning(tmp_path):
    # lr=0.2 has a failed seed where lr=0.1 has a result, so its mean is over
    # fewer cells and must not compete; the complete lr=0.1 group wins even
    # though its mean is lower.
    wd = tmp_path / "s"
    with pytest.warns(UserWarning, match="fail"):
        _run_seeded(wd, fail={(0.2, 1)})
    with pytest.warns(UserWarning, match=r"skipped 1 of 2 groups") as rec:
        r = mushin.best(wd, "score", over="seed")
    assert r.combo == {"lr": 0.1}
    assert r.value == pytest.approx(0.0)
    assert r.n == 3
    ours = [w for w in rec if "skipped" in str(w.message)]
    assert all(w.filename == __file__ for w in ours)
    assert "no manifest" not in str(ours[0].message)  # the sweep finished


def test_best_over_drops_a_seed_that_failed_everywhere(tmp_path):
    """Means are comparable over the SAME seeds, not necessarily all of them:
    with seed 0 failed for both configurations, both are complete over seeds
    1 and 2 (lr=0.1: (-5 + 10) / 2 = 2.5 beats lr=0.2's 1.0)."""
    wd = tmp_path / "s"
    with pytest.warns(UserWarning, match="fail"):
        _run_seeded(wd, fail={(0.1, 0), (0.2, 0)})
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # nothing to warn about
        r = mushin.best(wd, "score", over="seed")
    assert r.combo == {"lr": 0.1}
    assert r.value == pytest.approx(2.5)
    assert r.n == 2
    assert _combos(r.dirs) == [{"lr": 0.1, "seed": 1}, {"lr": 0.1, "seed": 2}]


def test_best_over_with_no_complete_group_raises(tmp_path):
    # each configuration lacks a result at a seed where the other has one
    wd = tmp_path / "s"
    with pytest.warns(UserWarning, match="fail"):
        _run_seeded(wd, fail={(0.1, 0), (0.2, 1)})
    assert mushin.best(wd, "score").combo == {"lr": 0.1, "seed": 2}  # still works
    with pytest.raises(ValueError, match="skipped 2 of 2 groups.*without `over=`"):
        mushin.best(wd, "score", over="seed")


def test_best_over_treats_missing_seeds_as_incomplete(tmp_path):
    """A cell that has not started has no job dir at all, so a group with every
    PRESENT cell completed can still be short of seeds. Removing two of lr=0.2's
    seeds stands in for a sweep still running or killed: lr=0.2 holds one seed
    (score 1.0), and must not beat lr=0.1's complete three (mean 0.0)."""
    import shutil

    wd = tmp_path / "s"
    _run_seeded(wd)
    for seed in (1, 2):
        shutil.rmtree(_cell_dir(wd, lr=0.2, seed=seed))
    with pytest.warns(UserWarning, match=r"skipped 1 of 2 groups"):
        r = mushin.best(wd, "score", over="seed")
    assert r.combo == {"lr": 0.1}
    assert r.n == 3


def test_best_over_warns_that_a_running_sweep_sets_a_provisional_bar(tmp_path):
    """Mid-sweep at a seed boundary (seed the outer loop, or a batch launcher)
    every configuration has the same single seed, so every group IS complete
    at n=1 and only the missing manifest tells that the bar is provisional."""
    import shutil

    from mushin._sweep_io import MANIFEST_FILE

    wd = tmp_path / "s"
    _run_seeded(wd)
    (wd / MANIFEST_FILE).unlink()
    for lr in (0.1, 0.2):
        for seed in (1, 2):
            shutil.rmtree(_cell_dir(wd, lr=lr, seed=seed))
    with pytest.warns(UserWarning, match="no manifest yet") as rec:
        r = mushin.best(wd, "score", over="seed")
    assert r.combo == {"lr": 0.2} and r.n == 1
    assert "skipped" not in str(rec[0].message)  # only what happened


def test_best_over_rejects_a_group_with_two_cells_at_one_seed(tmp_path):
    """A reused working_dir can hold a stale cell and a newer one at the same
    combo; averaging both would double-weight that seed."""
    import shutil

    wd = tmp_path / "s"
    _run_seeded(wd)
    extra = wd / "99"
    shutil.copytree(_cell_dir(wd, lr=0.2, seed=0), extra)
    _touch_newer_than_manifest(wd, extra)
    with pytest.warns(UserWarning, match="two cells at one value"):
        r = mushin.best(wd, "score", over="seed")
    assert r.combo == {"lr": 0.1}


def test_best_over_leaves_out_cells_without_the_param(tmp_path):
    """A newer sweep in flight in a reused working_dir can have another grid;
    its cells, lacking `seed`, are reported and left out rather than competing
    as one-cell groups."""
    import shutil

    from mushin._resume import write_cell_status

    wd = tmp_path / "s"
    _run_seeded(wd)
    extra = wd / "99"
    shutil.copytree(_cell_dir(wd, lr=0.1, seed=2), extra)  # score 10.0, no seed
    write_cell_status(extra, status="completed", combo={"lr": 0.3}, attempt=1)
    _touch_newer_than_manifest(wd, extra)
    assert any(c["combo"] == {"lr": 0.3} for c in mushin._show._read_cells(wd))

    with pytest.warns(UserWarning, match=r"1 cell\(s\) whose params") as rec:
        r = mushin.best(wd, "score", over="seed")
    assert r.combo == {"lr": 0.2}
    assert r.value == pytest.approx(1.0)
    assert "skipped" not in str(rec[0].message)  # no group was skipped


def test_best_over_unknown_param_raises_with_swept_params(tmp_path):
    wd = tmp_path / "s"
    _run_seeded(wd)
    with pytest.raises(ValueError, match="'sed'.*lr, seed"):
        mushin.best(wd, "score", over="sed")
    with pytest.raises(ValueError, match="`over`"):
        mushin.best(wd, "score", over=5)
    with pytest.raises(ValueError, match="`over`"):
        mushin.best(wd, "score", over=[])
    # a wrong `over` is reported even when no cell has the metric yet
    with pytest.raises(ValueError, match="`over`"):
        mushin.best(wd, "nope", over=5)
    with pytest.raises(ValueError, match="'sed'"):
        mushin.best(wd, "nope", over="sed")


def test_best_over_unknown_metric_raises_the_usual_error(tmp_path):
    wd = tmp_path / "s"
    _run_seeded(wd)
    with pytest.raises(ValueError, match="no completed cell has a finite scalar"):
        mushin.best(wd, "nope", over="seed")


def test_best_over_non_finite_values_are_not_results(tmp_path):
    """A completed cell whose metric is NaN or infinite is no result: `only0`
    has one at seed 0 only and `signed_inf` at seed 2 only, in every group, so
    the bar is that one seed and +inf/-inf never reach fmean."""
    wd = tmp_path / "s"
    _run_seeded(wd)
    r = mushin.best(wd, "only0", over="seed")
    assert r.n == 1 and r.value == pytest.approx(1.0)
    r = mushin.best(wd, "signed_inf", over="seed")
    assert r.n == 1 and r.value == pytest.approx(1.0)
    assert _combos(r.dirs)[0]["seed"] == 2


def test_best_never_picks_an_infinite_cell(tmp_path):
    wd = tmp_path / "s"
    _run_seeded(wd)
    r = mushin.best(wd, "signed_inf")  # +inf at seed 0 is not finite
    assert r.combo["seed"] == 2 and r.value == pytest.approx(1.0)
    r = mushin.best(wd, "signed_inf", mode="min")  # nor is -inf at seed 1
    assert r.combo["seed"] == 2


def test_best_result_without_a_dir_needs_dirs():
    from mushin._show import BestResult

    with pytest.raises(TypeError, match="dirs"):
        BestResult({}, "completed", {}, 0.0, None)
    r = BestResult({}, "completed", {}, 0.0, None, dirs=["a", "b"])
    assert r.n == 2 and "n=2" in repr(r)


def test_best_skips_an_int_metric_beyond_float_range(tmp_path):
    """JSON carries ints of any size and math.isfinite raises OverflowError on
    one beyond float range; such a value is no result, not a crash."""
    import json

    wd = tmp_path / "s"
    _run_seeded(wd)
    lucky = _cell_dir(wd, lr=0.1, seed=2)  # score 10.0, the single best
    (lucky / "mushin_metrics.json").write_text(json.dumps({"score": 10**400}))
    assert mushin.best(wd, "score").combo["lr"] == 0.2
    with pytest.warns(UserWarning, match="skipped 1 of 2"):
        assert mushin.best(wd, "score", over="seed").combo == {"lr": 0.2}


def test_best_over_with_results_only_in_foreign_cells_raises(tmp_path):
    """The metric exists only in a cell that lacks `seed`: no grouped cell has
    a result, so the bar is empty and that is "no complete group", not every
    group complete over nothing (fmean of nothing)."""
    import json
    import shutil

    from mushin._resume import write_cell_status

    wd = tmp_path / "s"
    _run_seeded(wd)
    extra = wd / "99"
    shutil.copytree(_cell_dir(wd, lr=0.1, seed=2), extra)
    (extra / "mushin_metrics.json").write_text(json.dumps({"fresh": 1.0}))
    write_cell_status(extra, status="completed", combo={"lr": 0.3}, attempt=1)
    _touch_newer_than_manifest(wd, extra)
    with pytest.raises(ValueError, match="no group of cells.*left out"):
        mushin.best(wd, "fresh", over="seed")


def test_best_over_warns_while_a_resumed_cell_is_running(tmp_path):
    """A resume re-runs cells under the prior run's manifest, so the manifest
    alone does not say the sweep finished; a running cell does."""
    from mushin._resume import write_cell_status

    wd = tmp_path / "s"
    _run_seeded(wd)
    write_cell_status(
        _cell_dir(wd, lr=0.2, seed=2),
        status="running",
        combo={"lr": 0.2, "seed": 2},
        attempt=2,
    )
    with pytest.warns(UserWarning, match=r"1 cell\(s\) still running") as rec:
        r = mushin.best(wd, "score", over="seed")
    assert r.combo == {"lr": 0.1}
    assert "skipped 1 of 2" in str(rec[0].message)


def test_best_over_treats_a_corrupt_manifest_as_unfinished(tmp_path):
    """_read_cells scans unscoped when the manifest is unusable; the result
    must say so rather than take the file's presence as "finished"."""
    from mushin._sweep_io import MANIFEST_FILE

    wd = tmp_path / "s"
    _run_seeded(wd)
    (wd / MANIFEST_FILE).write_text("{not json")
    with pytest.warns(UserWarning, match="no manifest yet"):
        r = mushin.best(wd, "score", over="seed")
    assert r.combo == {"lr": 0.2} and r.n == 3
