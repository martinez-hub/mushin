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
    lr=0.2 scores 1, 1, 1 (mean 1.0). ``only0`` is finite at seed 0 only, so a
    mean of it is undefined. ``fail`` lists (lr, seed) cells that raise, which
    ``on_error="nan"`` records as failed."""
    scores = {0.1: [-5.0, -5.0, 10.0], 0.2: [1.0, 1.0, 1.0]}

    class W(MultiRunMetricsWorkflow):
        @staticmethod
        def task(lr, seed):
            if (lr, seed) in fail:
                raise RuntimeError("boom")
            score = scores[lr][seed]
            return dict(
                score=score, loss=-score, only0=1.0 if seed == 0 else float("nan")
            )

    W().run(
        lr=multirun([0.1, 0.2]),
        seed=multirun([0, 1, 2]),
        working_dir=str(wd),
        on_error="nan" if fail else "raise",
    )


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
    assert len(r.dirs) == 3 and all(Path(d).is_dir() for d in r.dirs)
    # means of the metrics finite in every cell; `only0` is NaN at seeds 1, 2
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
    # lr=0.2 has a failed seed, so its mean is over fewer cells and must not
    # compete; the complete lr=0.1 group wins even though its mean is lower.
    wd = tmp_path / "s"
    with pytest.warns(UserWarning, match="fail"):
        _run_seeded(wd, fail={(0.2, 1)})
    with pytest.warns(UserWarning, match=r"skipped 1 of 2 groups") as rec:
        r = mushin.best(wd, "score", over="seed")
    assert r.combo == {"lr": 0.1}
    assert r.value == pytest.approx(0.0)
    assert r.n == 3
    assert all(w.filename == __file__ for w in rec if "skipped" in str(w.message))


def test_best_over_with_no_complete_group_raises(tmp_path):
    wd = tmp_path / "s"
    with pytest.warns(UserWarning, match="fail"):
        _run_seeded(wd, fail={(0.1, 0), (0.2, 0)})
    assert mushin.best(wd, "score").combo == {"lr": 0.1, "seed": 2}  # still works
    with pytest.raises(ValueError, match="without `over=`"):
        mushin.best(wd, "score", over="seed")


def test_best_over_unknown_param_raises_with_swept_params(tmp_path):
    wd = tmp_path / "s"
    _run_seeded(wd)
    with pytest.raises(ValueError, match="'sed'.*lr, seed"):
        mushin.best(wd, "score", over="sed")
    with pytest.raises(ValueError, match="`over`"):
        mushin.best(wd, "score", over=5)
    with pytest.raises(ValueError, match="`over`"):
        mushin.best(wd, "score", over=[])


def test_best_over_unknown_metric_raises_the_usual_error(tmp_path):
    wd = tmp_path / "s"
    _run_seeded(wd)
    with pytest.raises(ValueError, match="no completed cell has a finite scalar"):
        mushin.best(wd, "nope", over="seed")


def _cell_dir(wd, **combo):
    import json

    for d in wd.iterdir():
        if d.is_dir() and (d / "mushin_cell_status.json").exists():
            c = json.loads((d / "mushin_cell_status.json").read_text())["combo"]
            if all(c.get(k) == v for k, v in combo.items()):
                return d
    raise AssertionError(f"no cell dir for {combo}")


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
    assert any(c["combo"] == {"lr": 0.3} for c in mushin._show._read_cells(wd))

    with pytest.warns(UserWarning, match=r"1 cell\(s\) whose params do not include"):
        r = mushin.best(wd, "score", over="seed")
    assert r.combo == {"lr": 0.2}
    assert r.value == pytest.approx(1.0)


def test_best_result_without_dir_has_no_dirs():
    from mushin._show import BestResult

    r = BestResult({}, "completed", {}, 0.0, None)
    assert r.dirs == [] and r.n == 0
    assert "n=0" in repr(r)
