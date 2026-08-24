# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
"""Tests for the Inspect AI adapter in ``examples/inspect_ai_compare.py``.

Inspect AI is not a mushin dependency, so the log objects are duck-typed here.
That is enough: the risky part is the ALIGNMENT (samples paired by id across
models, epochs mapped to runs), and getting it wrong silently compares unrelated
questions rather than failing.
"""

from __future__ import annotations

import importlib.util
import re
import sys
import types
import warnings
from pathlib import Path

import numpy as np
import pytest

_SPEC = importlib.util.spec_from_file_location(
    "inspect_ai_compare",
    Path(__file__).resolve().parent.parent / "examples" / "inspect_ai_compare.py",
)
adapter = importlib.util.module_from_spec(_SPEC)
sys.modules["inspect_ai_compare"] = adapter
_SPEC.loader.exec_module(adapter)


def _log(model, rows, scorer="accuracy"):
    """rows: list of (sample_id, epoch, value) -> a duck-typed EvalLog."""
    samples = [
        types.SimpleNamespace(
            id=sid, epoch=ep, scores={scorer: types.SimpleNamespace(value=val)}
        )
        for sid, ep, val in rows
    ]
    return types.SimpleNamespace(
        eval=types.SimpleNamespace(model=model), samples=samples
    )


def test_samples_are_aligned_by_id_not_position():
    """The whole point: two logs may list their samples in different orders."""
    a = _log("m/a", [("q1", 1, 1.0), ("q2", 1, 0.0), ("q3", 1, 1.0)])
    b = _log("m/b", [("q3", 1, 0.0), ("q1", 1, 1.0), ("q2", 1, 1.0)])  # shuffled
    scores, ids = adapter.scores_from_logs([a, b])
    assert ids == ["q1", "q2", "q3"]
    np.testing.assert_array_equal(scores["m/a"], [[1.0, 0.0, 1.0]])
    # aligned by id: q1=1, q2=1, q3=0 — NOT the file order [0, 1, 1]
    np.testing.assert_array_equal(scores["m/b"], [[1.0, 1.0, 0.0]])


def test_epochs_become_runs():
    rows = [(sid, ep, float(ep)) for ep in (1, 2, 3) for sid in ("q1", "q2")]
    scores, ids = adapter.scores_from_logs([_log("m/a", rows), _log("m/b", rows)])
    assert scores["m/a"].shape == (3, 2)  # (n_runs, n_items)
    np.testing.assert_array_equal(scores["m/a"], [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])


def test_correct_incorrect_verdicts_are_converted():
    a = _log("m/a", [("q1", 1, "C"), ("q2", 1, "I"), ("q3", 1, "P")])
    b = _log("m/b", [("q1", 1, True), ("q2", 1, False), ("q3", 1, 0.25)])
    scores, _ = adapter.scores_from_logs([a, b])
    np.testing.assert_array_equal(scores["m/a"], [[1.0, 0.0, 0.5]])
    np.testing.assert_array_equal(scores["m/b"], [[1.0, 0.0, 0.25]])


def test_mismatched_sample_sets_are_rejected():
    """Different datasets cannot be paired — mushin pairs item i to item i."""
    a = _log("m/a", [("q1", 1, 1.0), ("q2", 1, 1.0)])
    b = _log("m/b", [("q1", 1, 1.0), ("q9", 1, 1.0)])
    with pytest.raises(ValueError, match="different questions"):
        adapter.scores_from_logs([a, b])


def test_missing_epoch_cell_is_rejected():
    a = _log(
        "m/a", [("q1", 1, 1.0), ("q2", 1, 1.0), ("q1", 2, 1.0)]
    )  # q2 epoch 2 absent
    b = _log("m/b", [("q1", 1, 1.0), ("q2", 1, 1.0)])
    with pytest.raises(ValueError, match="missing a sample/epoch cell"):
        adapter.scores_from_logs([a, b])


def test_ambiguous_and_missing_scorer_are_rejected():
    two = types.SimpleNamespace(
        eval=types.SimpleNamespace(model="m/a"),
        samples=[
            types.SimpleNamespace(
                id="q1",
                epoch=1,
                scores={
                    "acc": types.SimpleNamespace(value=1.0),
                    "f1": types.SimpleNamespace(value=0.5),
                },
            )
        ],
    )
    with pytest.raises(ValueError, match="pass scorer="):
        adapter.scores_from_logs([two])
    with pytest.raises(ValueError, match="missing from sample"):
        adapter.scores_from_logs([two], scorer="nope")
    scores, _ = adapter.scores_from_logs([two], scorer="f1")
    np.testing.assert_array_equal(scores["m/a"], [[0.5]])


def test_unconvertible_score_value_raises_rather_than_guessing():
    a = _log("m/a", [("q1", 1, {"nested": 1})])
    with pytest.raises((TypeError, ValueError), match="unsupported|cannot score"):
        adapter.scores_from_logs([a])


def test_header_only_log_gives_an_actionable_error():
    empty = types.SimpleNamespace(eval=types.SimpleNamespace(model="m/a"), samples=None)
    with pytest.raises(ValueError, match="header_only"):
        adapter.scores_from_logs([empty])


def test_end_to_end_into_compare_scores():
    """The adapter's output must be exactly what compare_scores accepts."""
    pytest.importorskip("scipy")
    from mushin.llm import compare_scores

    rng = np.random.default_rng(0)
    ids = [f"q{i}" for i in range(40)]
    rows_a = [(sid, ep, float(rng.random() < 0.8)) for ep in (1, 2, 3) for sid in ids]
    rows_b = [(sid, ep, float(rng.random() < 0.5)) for ep in (1, 2, 3) for sid in ids]
    scores, sample_ids = adapter.scores_from_logs(
        [_log("m/a", rows_a), _log("m/b", rows_b)]
    )
    assert sample_ids == sorted(ids, key=str)
    result = compare_scores(scores)
    row = result.comparisons.iloc[0]
    assert not np.isnan(row["item_diff"])
    assert not np.isnan(row["p_value"])  # 3 epochs -> seed dimension exists


def test_demo_runs_and_reaches_the_right_verdicts(capsys):
    """`--demo` must work with no Inspect AI and no logs, and be *correct*.

    It is the example's own regression test: two identical models must not be
    called different, and a genuinely better model must be.
    """
    pytest.importorskip("scipy")
    assert adapter.main(["--demo"]) == 0
    out = capsys.readouterr().out
    scenarios = out.split("SCENARIO")
    assert len(scenarios) == 3  # preamble + two scenarios
    identical, better = scenarios[1], scenarios[2]
    assert "not a difference you can defend" in identical
    assert "a difference worth acting on" in better


def test_report_handles_a_single_run(capsys):
    """One epoch cannot answer the re-run question; say so instead of guessing."""
    pytest.importorskip("scipy")
    rng = np.random.default_rng(0)
    scores = {
        "a": (rng.random((1, 40)) < 0.8).astype(float),
        "b": (rng.random((1, 40)) < 0.5).astype(float),
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        adapter.report(scores)
    out = capsys.readouterr().out
    assert "only one epoch" in out  # re-run question unanswerable
    assert "would OTHER QUESTIONS agree?" in out  # item question still answered


def _rows(model, n_items=30, n_epochs=3, bonus=0.0, seed=0):
    rng = np.random.default_rng(seed)
    return _log(
        model,
        [
            (f"q{i}", ep, 1.0 if rng.random() < 0.5 + bonus else 0.0)
            for ep in range(1, n_epochs + 1)
            for i in range(n_items)
        ],
    )


def test_ci_is_oriented_to_the_announced_lead(capsys):
    """When the SECOND model wins, the interval must not print negative.

    The bounds are signed method_a - method_b, and method_a is just the first key
    of the dict — i.e. the order the log files were passed. Printing them raw
    means about half of real invocations show a positive lead above a wholly
    negative interval.
    """
    pytest.importorskip("scipy")
    scores, _ = adapter.scores_from_logs(
        [_rows("weak", bonus=-0.25, seed=1), _rows("strong", bonus=0.25, seed=2)]
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        adapter.report(scores)
    out = capsys.readouterr().out
    assert "strong leads weak" in out
    ci = out.split("95% CI [")[1].split("]")[0]
    assert not ci.strip().startswith("-"), f"interval contradicts the lead: {ci}"


def _printed_pvalues(out):
    """Every ``p=`` number in report()'s output, in printed order."""
    return [float(m) for m in re.findall(r"p=([0-9.]+)", out)]


def test_three_models_print_the_corrected_p_values_on_BOTH_axes(capsys):
    """With 3+ models there are 3 pairs; the raw p-values over-state significance.

    Asserting the printed NUMBERS, not just the word "Holm-corrected": a test that
    only checks the banner passes with either p-value column wired up, which is
    how a raw item p sat under a "corrected" banner in the first place.
    """
    pytest.importorskip("scipy")
    from mushin.llm import compare_scores

    scores, _ = adapter.scores_from_logs(
        [_rows("m1", seed=1), _rows("m2", seed=2), _rows("m3", seed=3)]
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        expected = compare_scores(scores)
        adapter.report(scores)
    out = capsys.readouterr().out
    assert out.count("Is that real?") == 3

    corrected = [
        p
        for _, row in expected.comparisons.iterrows()
        for p in (row["p_corrected"], row["item_p_corrected"])
        if not np.isnan(p)
    ]
    raw = [
        p
        for _, row in expected.comparisons.iterrows()
        for p in (row["p_value"], row["item_p"])
        if not np.isnan(p)
    ]
    # The correction has to actually bite, or this test proves nothing.
    assert any(c > r + 1e-9 for c, r in zip(corrected, raw, strict=True))
    assert _printed_pvalues(out) == pytest.approx(
        [round(p, 4) for p in corrected], abs=5e-5
    )


def test_item_p_is_corrected_over_the_same_family_as_the_seed_p():
    """The library, not the example, owns the correction — check it directly."""
    pytest.importorskip("scipy")
    from mushin.benchmark._stats import holm_correction
    from mushin.llm import compare_scores

    scores, _ = adapter.scores_from_logs(
        [_rows("m1", seed=1), _rows("m2", seed=2), _rows("m3", seed=3)]
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = compare_scores(scores)
    np.testing.assert_allclose(
        result.comparisons["item_p_corrected"].to_numpy(),
        holm_correction(result.comparisons["item_p"].to_numpy()),
    )


def test_two_models_leave_the_p_values_uncorrected_and_say_nothing(capsys):
    """One comparison is not a family: no banner, and no inflated p-value."""
    pytest.importorskip("scipy")
    from mushin.llm import compare_scores

    scores, _ = adapter.scores_from_logs([_rows("m1", seed=1), _rows("m2", seed=2)])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        expected = compare_scores(scores)
        adapter.report(scores)
    out = capsys.readouterr().out
    assert "Holm-corrected" not in out
    row = expected.comparisons.iloc[0]
    assert row["item_p_corrected"] == pytest.approx(row["item_p"])


def test_logs_from_different_tasks_are_refused():
    """Inspect numbers samples 1..N per task, so ids collide across tasks."""
    a = _log("m/a", [(i, 1, 1.0) for i in range(1, 6)])
    b = _log("m/b", [(i, 1, 0.0) for i in range(1, 6)])
    a.eval.task = "theory_of_mind"
    b.eval.task = "gsm8k"
    with pytest.raises(ValueError, match="different Inspect tasks"):
        adapter.scores_from_logs([a, b])
    # ...and the same task on both logs is fine.
    b.eval.task = "theory_of_mind"
    scores, _ = adapter.scores_from_logs([a, b])
    assert set(scores) == {"m/a", "m/b"}


def test_non_fractional_scores_are_not_printed_as_percentages(capsys):
    """A 1-10 rubric scorer must not render 7.4 as 744.2%."""
    pytest.importorskip("scipy")
    rng = np.random.default_rng(0)
    scores = {
        "a": 7.0 + rng.normal(scale=0.4, size=(4, 30)),
        "b": 6.0 + rng.normal(scale=0.4, size=(4, 30)),
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        adapter.report(scores)
    out = capsys.readouterr().out
    assert "score units" in out
    assert "points" not in out
    # "95% CI" is the only legitimate percent sign; no SCORE may be rendered as one.
    assert re.sub(r"95% CI", "", out).count("%") == 0
    assert "7.0" in out or "6.9" in out or "7.1" in out  # the mean, in its own units


def test_an_unmeasurable_check_is_reported_as_unproven_not_as_a_refutation(capsys):
    """A temperature-0 model has no re-run distribution. That is not a failure."""
    pytest.importorskip("scipy")
    rng = np.random.default_rng(1)
    items = rng.random(40)
    scores = {
        # identical every epoch: the seed axis cannot be tested...
        "temp0": np.tile(items + 0.25, (4, 1)),
        # ...but the item axis is decisive.
        "other": items + rng.normal(scale=0.01, size=(4, 40)),
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        adapter.report(scores)
    out = capsys.readouterr().out
    assert "no re-run distribution" in out
    assert "holds up" in out
    assert "UNPROVEN" in out
    assert "not a difference you can defend" not in out


def test_a_degenerate_item_bootstrap_is_not_blamed_on_missing_scores(capsys):
    """Per-question scores are always present on this path; say the real cause."""
    pytest.importorskip("scipy")
    rng = np.random.default_rng(2)
    base = rng.random((4, 30))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        adapter.report({"a": base + 0.1, "b": base})  # identical gap on every item
    out = capsys.readouterr().out
    assert "every question shows the same gap" in out
    assert "per-question scores unavailable" not in out


def test_masked_seed_test_is_not_blamed_on_a_single_epoch(capsys):
    """A deterministic model masks the seed test even with many epochs."""
    pytest.importorskip("scipy")
    const = _log("const", [(f"q{i}", ep, 1.0) for ep in (1, 2, 3) for i in range(30)])
    scores, _ = adapter.scores_from_logs([const, _rows("varies", seed=4)])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        adapter.report(scores)
    out = capsys.readouterr().out
    assert "no re-run distribution" in out
    assert "only one epoch" not in out


def test_single_log_says_so_instead_of_printing_nothing(capsys):
    scores, _ = adapter.scores_from_logs([_rows("solo")])
    adapter.report(scores)
    assert "at least two logs" in capsys.readouterr().out


def test_unscored_sample_gets_its_own_error():
    """An errored Inspect sample has scores=None; that is not scorer ambiguity."""
    bad = types.SimpleNamespace(
        eval=types.SimpleNamespace(model="m"),
        samples=[types.SimpleNamespace(id="q1", epoch=1, scores=None)],
    )
    with pytest.raises(ValueError, match="no scores"):
        adapter.scores_from_logs([bad])


def _real_eval_log(model, task, rows):
    """Build a genuine Inspect `EvalLog`, not a duck-type."""
    from inspect_ai.log import (
        EvalConfig,
        EvalDataset,
        EvalLog,
        EvalSample,
        EvalSpec,
        EvalStats,
    )
    from inspect_ai.scorer import Score

    return EvalLog(
        version=2,
        status="success",
        eval=EvalSpec(
            task=task,
            task_id="t1",
            task_version=0,
            model=model,
            dataset=EvalDataset(),
            config=EvalConfig(),
            created="2026-01-01",
        ),
        stats=EvalStats(started_at="2026-01-01", completed_at="2026-01-01"),
        samples=[
            EvalSample(
                id=sid,
                epoch=ep,
                input="q",
                target="t",
                scores={"accuracy": Score(value=val)},
            )
            for sid, ep, val in rows
        ],
    )


def test_adapter_works_against_real_inspect_ai_objects():
    """The other tests duck-type `EvalLog`; this one uses the real class.

    Duck-types encode our *belief* about Inspect's shape. If `sample.scores`,
    `sample.epoch`, `sample.id` or `eval.model` were renamed upstream, every
    other test here would still pass while real logs broke on first contact.
    Skipped when inspect-ai is absent — it pulls ~110 packages, so it is not a
    dev dependency.
    """
    pytest.importorskip("inspect_ai")
    pytest.importorskip("scipy")
    rng = np.random.default_rng(0)
    rows_a = [
        (i, ep, "C" if rng.random() < 0.75 else "I")
        for ep in (1, 2, 3)
        for i in range(1, 21)
    ]
    rows_b = [
        (i, ep, "C" if rng.random() < 0.50 else "I")
        for ep in (1, 2, 3)
        for i in range(1, 21)
    ]
    scores, ids = adapter.scores_from_logs(
        [
            _real_eval_log("openai/gpt-4", "theory_of_mind", rows_a),
            _real_eval_log("anthropic/claude-3-5-sonnet", "theory_of_mind", rows_b),
        ]
    )
    assert set(scores) == {"openai/gpt-4", "anthropic/claude-3-5-sonnet"}
    assert all(arr.shape == (3, 20) for arr in scores.values())
    # Inspect numbers samples 1..N; they must come back in NUMERIC order, since
    # this list is what a caller lines `clusters=` up against.
    assert ids == list(range(1, 21))
    # "C"/"I" resolve through the real Score object.
    assert set(np.unique(scores["openai/gpt-4"])) <= {0.0, 1.0}


def test_real_inspect_logs_from_different_tasks_are_refused():
    """Inspect ids are 1..N per task, so cross-task pairing is meaningless."""
    pytest.importorskip("inspect_ai")
    rows = [(i, 1, "C") for i in range(1, 6)]
    with pytest.raises(ValueError, match="different Inspect tasks"):
        adapter.scores_from_logs(
            [
                _real_eval_log("m/a", "theory_of_mind", rows),
                _real_eval_log("m/b", "gsm8k", rows),
            ]
        )


def test_integer_sample_ids_sort_numerically_not_lexicographically():
    """Inspect's default ids are ints; str-sorting gives 1, 10, 11, ... 2, 20.

    The pairing stays correct either way (every model gets the same order), but
    `question_ids` is what a caller maps `clusters=` onto, so a surprising order
    silently misaligns the groups.
    """
    ids = list(range(1, 13))
    logs = [
        _log("m/a", [(i, 1, 1.0) for i in ids]),
        _log("m/b", [(i, 1, 0.0) for i in ids]),
    ]
    assert adapter.scores_from_logs(logs)[1] == ids
    # Mixed id types have no total order — must still be deterministic.
    mixed = [1, "b", 2]
    both = [
        _log("m/a", [(i, 1, 1.0) for i in mixed]),
        _log("m/b", [(i, 1, 0.0) for i in mixed]),
    ]
    assert adapter.scores_from_logs(both)[1] == [1, 2, "b"]
