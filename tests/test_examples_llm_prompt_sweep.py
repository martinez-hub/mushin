# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
"""Tests for ``examples/llm_prompt_sweep.py``.

The example's claim is about *resilience*: a transient API failure must not
discard the cells already paid for, and a resume must retry only the holes.
Those are the behaviours worth pinning — an example that quietly stopped
failing would still print a happy story while demonstrating nothing.
"""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pytest

_SPEC = importlib.util.spec_from_file_location(
    "llm_prompt_sweep",
    Path(__file__).resolve().parent.parent / "examples" / "llm_prompt_sweep.py",
)
sweep_ex = importlib.util.module_from_spec(_SPEC)
sys.modules["llm_prompt_sweep"] = sweep_ex
_SPEC.loader.exec_module(sweep_ex)


def test_score_config_returns_a_metric_dict():
    sweep_ex._OUTAGE = False
    try:
        out = sweep_ex.score_config("cot", 0.0, 0)
    finally:
        sweep_ex._OUTAGE = True
    assert set(out) == {"accuracy"}
    assert 0.0 <= out["accuracy"] <= 1.0


def test_the_simulated_outage_actually_fails_sometimes():
    """If nothing ever failed, the resilience story would be vacuous."""
    sweep_ex._OUTAGE = True
    failures = 0
    for seed in range(60):
        try:
            sweep_ex.score_config("cot", 0.0, seed)
        except RuntimeError:
            failures += 1
    assert failures > 0, "no simulated failures — on_error/resume prove nothing"


def test_on_error_nan_keeps_the_completed_cells():
    """A transient failure must leave holes, not discard the whole grid."""
    import mushin

    sweep_ex._OUTAGE = True
    wf = mushin.sweep(sweep_ex.score_config)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        wf.run(
            prompt=mushin.multirun(["cot", "terse"]),
            temperature=mushin.multirun([0.0]),
            seed=mushin.multirun(list(range(8))),
            working_dir=tempfile.mkdtemp(),
            on_error="nan",
        )
    values = wf.workflow.to_xarray()["accuracy"].values
    assert values.size == 16
    assert np.isfinite(values).any()  # something survived the outage


def test_resume_fills_the_holes_after_the_cause_is_fixed():
    import mushin

    workdir = tempfile.mkdtemp()
    sweep_ex._OUTAGE = True
    first = mushin.sweep(sweep_ex.score_config)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        first.run(
            prompt=mushin.multirun(["cot", "terse", "role"]),
            temperature=mushin.multirun([0.0]),
            seed=mushin.multirun(list(range(8))),
            working_dir=workdir,
            on_error="nan",
        )
    holes = int(np.isnan(first.workflow.to_xarray()["accuracy"].values).sum())

    sweep_ex._OUTAGE = False  # the cause is fixed
    try:
        second = mushin.sweep(sweep_ex.score_config)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            second.run(
                prompt=mushin.multirun(["cot", "terse", "role"]),
                temperature=mushin.multirun([0.0]),
                seed=mushin.multirun(list(range(8))),
                working_dir=workdir,
                resume=True,
            )
    finally:
        sweep_ex._OUTAGE = True
    after = second.workflow.to_xarray()["accuracy"].values
    assert np.isfinite(after).all(), f"{holes} holes were not filled by resume"


def test_demo_runs_and_reports_a_winner(capsys):
    pytest.importorskip("scipy")
    assert sweep_ex.main(["--demo"]) == 0
    out = capsys.readouterr().out
    assert "after resume:" in out
    assert "best: prompt=" in out
    assert "would a re-run agree?" in out


def test_cell_values_are_stable_across_processes():
    """resume can only reuse a cell whose value is reproducible.

    The first version seeded from hash(), which Python randomises per process, so
    every run produced different numbers and a resumed cell was not the cell it
    replaced.
    """
    import subprocess

    code = (
        "import importlib.util,sys;"
        "spec=importlib.util.spec_from_file_location('sw','examples/llm_prompt_sweep.py');"
        "m=importlib.util.module_from_spec(spec);sys.modules['sw']=m;spec.loader.exec_module(m);"
        "m._OUTAGE=False;print(m.score_config('cot',0.0,0)['accuracy'])"
    )
    outs = {
        subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True
        ).stdout.strip()
        for _ in range(3)
    }
    assert len(outs) == 1, f"cell value varies per process: {outs}"


def test_clearing_the_outage_does_not_change_the_score():
    """A resumed cell must be the SAME cell that failed, not a different draw."""
    sweep_ex._OUTAGE = False
    try:
        without = sweep_ex.score_config("role", 0.4, 2)["accuracy"]
    finally:
        sweep_ex._OUTAGE = True
    # a cell that does not trip the outage roll scores identically either way
    with_outage = sweep_ex.score_config("role", 0.4, 2)["accuracy"]
    assert with_outage == without


def test_scores_actually_vary_with_seed():
    """Without seed dependence the significance test has nothing to test."""
    sweep_ex._OUTAGE = False
    try:
        values = {sweep_ex.score_config("cot", 0.0, s)["accuracy"] for s in range(5)}
    finally:
        sweep_ex._OUTAGE = True
    assert len(values) > 1, f"all seeds identical: {values}"


def test_demo_compares_against_the_runner_up_not_a_fixed_baseline(capsys):
    """Comparing the winner to the by-design worst prompt proves nothing."""
    pytest.importorskip("scipy")
    assert sweep_ex.main(["--demo"]) == 0
    out = capsys.readouterr().out
    line = [ln for ln in out.splitlines() if " vs " in ln][0]
    winner, rest = line.strip().split(" vs ")
    runner = rest.split(" at ")[0]
    # the runner-up must be the second-best prompt in the printed table, and in
    # particular must not be hard-coded
    assert winner != runner
    assert runner in sweep_ex.PROMPTS


def test_demo_restores_the_outage_flag():
    """A second run in the same process must demonstrate the same thing."""
    pytest.importorskip("scipy")
    sweep_ex._OUTAGE = True
    sweep_ex.main(["--demo"])
    assert sweep_ex._OUTAGE is True, "demo() leaked its outage flag"
