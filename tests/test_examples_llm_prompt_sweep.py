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

    Each subprocess gets a different PYTHONHASHSEED so a hash()-based
    implementation must disagree, runs outside the repo root so nothing depends
    on the caller's cwd, and is checked for success — comparing raw stdout meant
    three identical *failures* passed as three identical values.
    """
    import os
    import subprocess

    path = Path(__file__).resolve().parent.parent / "examples" / "llm_prompt_sweep.py"
    code = (
        "import importlib.util,sys;"
        f"spec=importlib.util.spec_from_file_location('sw',{str(path)!r});"
        "m=importlib.util.module_from_spec(spec);sys.modules['sw']=m;spec.loader.exec_module(m);"
        "m._OUTAGE=False;print(m.score_config('cot',0.0,0)['accuracy'])"
    )
    values = set()
    for hashseed in ("0", "1", "12345"):
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONHASHSEED": hashseed},
            cwd=tempfile.gettempdir(),
        )
        assert proc.returncode == 0, f"subprocess failed: {proc.stderr}"
        values.add(float(proc.stdout.strip()))  # raises if it printed nothing
    assert len(values) == 1, f"cell value varies per process: {values}"
    assert 0.0 < next(iter(values)) < 1.0  # a constant would also be "stable"


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
    """Comparing the winner to the by-design worst prompt proves nothing.

    Asserting only "runner != winner and runner in PROMPTS" passes with the
    hard-coded "terse" baseline restored, because terse is neither the winner nor
    absent from PROMPTS. So this reads the printed table and checks the runner-up
    really is the second-best prompt at the winning temperature — and that it is
    NOT 'terse', the by-design worst.
    """
    pytest.importorskip("scipy")
    assert sweep_ex.main(["--demo"]) == 0
    out = capsys.readouterr().out
    line = [ln for ln in out.splitlines() if " vs " in ln][0]
    winner, rest = line.strip().split(" vs ")
    runner = rest.split(" at ")[0]
    temperature = float(rest.split("temperature=")[1])
    assert winner != runner
    assert runner in sweep_ex.PROMPTS

    # Recompute the ranking at that temperature from the example's own scorer.
    sweep_ex._OUTAGE = False
    try:
        ranked = sorted(
            sweep_ex.PROMPTS,
            key=lambda p: (
                -np.mean(
                    [
                        sweep_ex.score_config(p, temperature, s)["accuracy"]
                        for s in range(5)
                    ]
                )
            ),
        )
    finally:
        sweep_ex._OUTAGE = True
    assert winner == ranked[0], f"winner {winner!r} is not the best: {ranked}"
    assert runner == ranked[1], f"runner-up {runner!r} is not second: {ranked}"
    assert runner != "terse"  # the fixed baseline the first version used


def test_runner_up_selection_survives_a_tie_at_the_top(monkeypatch, capsys):
    """Two prompts tied at the winning temperature must not select the winner.

    A descending `sortby` REVERSES tied entries, so `values[1]` was exactly the
    winner whenever one other prompt tied it — and comparing a system against
    itself collapses compare_scores to a single system and raises.
    """
    pytest.importorskip("scipy")
    # Exactly two prompts tied at the top, which is the case that triggers it.
    monkeypatch.setattr(
        sweep_ex,
        "_simulated_call",
        lambda prompt, temperature, seed, question: float(prompt in ("cot", "role")),
    )
    assert sweep_ex.main(["--demo"]) == 0
    out = capsys.readouterr().out
    line = [ln for ln in out.splitlines() if " vs " in ln][0]
    winner, rest = line.strip().split(" vs ")
    assert winner != rest.split(" at ")[0]


def test_demo_reports_the_failed_cells_instead_of_hiding_them(capsys):
    """mushin says "N run(s) failed"; a blanket filter used to swallow it."""
    pytest.importorskip("scipy")
    assert sweep_ex.main(["--demo"]) == 0
    out = capsys.readouterr().out
    assert "run(s) failed" in out
    assert "grid cells set to NaN" in out


def test_demo_restores_the_outage_flag():
    """A second run in the same process must demonstrate the same thing."""
    pytest.importorskip("scipy")
    sweep_ex._OUTAGE = True
    sweep_ex.main(["--demo"])
    assert sweep_ex._OUTAGE is True, "demo() leaked its outage flag"
