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
import sys
import types
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
    with pytest.raises(ValueError, match="different samples"):
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
