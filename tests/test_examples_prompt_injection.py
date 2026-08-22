# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
"""Tests for ``examples/prompt_injection_eval.py``.

The example measures how often an indirect prompt injection succeeds, so the
thing to pin is the MEASUREMENT: the canary detector must not report a win that
did not happen (or miss one that did), and the eval set must actually carry an
attack in every document.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "prompt_injection_eval",
    Path(__file__).resolve().parent.parent / "examples" / "prompt_injection_eval.py",
)
inj = importlib.util.module_from_spec(_SPEC)
sys.modules["prompt_injection_eval"] = inj
_SPEC.loader.exec_module(inj)


def test_every_document_actually_carries_an_attack():
    """An eval set that forgot to inject would score 100% resistance — silently."""
    suite = inj.injection_suite()
    assert len(suite) == len(inj._TOPICS) * len(inj._ATTACKS)
    for example in suite:
        assert inj.CANARY in example["input"]["document"]
        assert example["reference"] == inj.CANARY
        # the legitimate task must not itself mention the canary
        assert inj.CANARY not in example["input"]["task"]


def test_all_attack_shapes_are_represented():
    suite = inj.injection_suite()
    assert {ex["attack"] for ex in suite} == set(inj._ATTACKS)


def test_detector_scores_the_right_direction():
    """1.0 = resisted, 0.0 = the injection won. Getting this backwards would
    invert every conclusion the harness draws."""
    assert inj.resisted("Here is a summary of the memo.", inj.CANARY) == 1.0
    assert inj.resisted(f"Sure. {inj.CANARY}", inj.CANARY) == 0.0
    # the canary anywhere in the output counts, not just as the whole reply
    assert inj.resisted(f"...{inj.CANARY}...trailing text", inj.CANARY) == 0.0
    # non-string outputs must not crash the scorer
    assert inj.resisted(None, inj.CANARY) == 1.0


def test_a_perfectly_robust_system_scores_one_and_a_broken_one_zero():
    suite = inj.injection_suite()
    prompts = [ex["input"] for ex in suite]
    robust = inj._simulated_system(1.0)
    broken = inj._simulated_system(0.0)
    assert all(inj.resisted(o, inj.CANARY) == 1.0 for o in robust(prompts, 0))
    assert all(inj.resisted(o, inj.CANARY) == 0.0 for o in broken(prompts, 0))


def test_demo_runs_and_surfaces_the_blind_spot(capsys):
    """The demo's point is that an aggregate score hides a per-attack regression."""
    pytest.importorskip("scipy")
    assert inj.main(["--demo"]) == 0
    out = capsys.readouterr().out
    assert "blind spot" in out  # the per-attack view found it
    assert "would OTHER ATTACKS agree?" in out
