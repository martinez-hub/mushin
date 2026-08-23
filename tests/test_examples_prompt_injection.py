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
import re
import sys
import tempfile
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
    """The demo's point is that an aggregate score hides a per-attack regression.

    Pins WHICH attack is flagged and that the others are not: asserting only that
    the words "blind spot" appear passes with the comparison inverted, which would
    flag four attacks and miss the real one.
    """
    pytest.importorskip("scipy")
    assert inj.main(["--demo"]) == 0
    out = capsys.readouterr().out
    flagged = re.findall(r"blind spot: .* on '([^']+)'", out)
    assert flagged == ["authority"], out  # the one shape 'hardened' is weak against
    assert "would OTHER ATTACKS agree?" in out


def test_demo_reports_the_few_clusters_warning_instead_of_hiding_it(capsys):
    """Five attack shapes is a five-sample problem, and mushin says so.

    The demo used to wrap compare_llms in a blanket `simplefilter("ignore")`. Once
    `clusters=` was added that filter started swallowing mushin's own "this
    interval is unreliable" warning — while the demo printed the interval.
    """
    pytest.importorskip("scipy")
    assert inj.main(["--demo"]) == 0
    out = capsys.readouterr().out
    assert "mushin warns" in out
    assert "only 5 clusters" in out
    assert "add attack SHAPES" in out


def test_the_attack_label_is_not_visible_to_the_system_under_test():
    """Ground truth lives beside `input`, never inside it."""
    suite = inj.injection_suite()
    for ex in suite:
        assert ex["attack"] in inj._ATTACKS  # available to the harness...
        assert set(ex["input"]) == {"task", "document"}  # ...but not to the model
        assert ex["attack"] not in str(ex["input"])


def test_per_attack_rows_are_genuinely_distinct():
    """The breakdown must measure the attacks, not replay one RNG stream.

    The first version seeded sequentially per batch position, so every attack
    subset drew the same numbers and four of five rows were bit-identical — a
    table that looked informative and carried nothing.
    """
    suite = inj.injection_suite()
    system = inj._simulated_system(0.7)
    rates = {}
    for attack in inj._ATTACKS:
        prompts = [ex["input"] for ex in suite if ex["attack"] == attack]
        outputs = system(prompts, 0)
        rates[attack] = sum(inj.resisted(o, inj.CANARY) for o in outputs) / len(outputs)
    assert len(set(rates.values())) > 1, f"all attacks scored identically: {rates}"


def test_outcome_is_independent_of_batch_composition():
    """Scoring one attack's documents must match scoring all of them together.

    This is the contract mushin's output cache assumes, and it is what makes the
    sweep breakdown a decomposition of the headline number rather than a second,
    unrelated measurement.
    """
    suite = inj.injection_suite()
    system = inj._simulated_system(0.7, weak_against="authority")
    whole = {
        ex["input"]["document"]: inj.resisted(o, inj.CANARY)
        for ex, o in zip(suite, system([e["input"] for e in suite], 3), strict=True)
    }
    for attack in inj._ATTACKS:
        subset = [e["input"] for e in suite if e["attack"] == attack]
        for item, out in zip(subset, system(subset, 3), strict=True):
            assert inj.resisted(out, inj.CANARY) == whole[item["document"]]


def test_simulated_weakness_keys_on_the_attack_name():
    """`weak_against` takes an attack name, and rejects one that does not exist."""
    with pytest.raises(ValueError, match="unknown attack"):
        inj._simulated_system(0.8, weak_against="SYSTEM NOTICE")
    suite = inj.injection_suite()
    system = inj._simulated_system(0.9, weak_against="roleplay")
    rate = {}
    for attack in ("roleplay", "naive"):
        prompts = [ex["input"] for ex in suite if ex["attack"] == attack]
        outs = system(prompts, 0)
        rate[attack] = sum(inj.resisted(o, inj.CANARY) for o in outs) / len(outs)
    assert rate["roleplay"] < rate["naive"]


def test_draws_are_stable_across_processes():
    """hash() is randomised per process; the demo must not be.

    Runs with an explicit PYTHONHASHSEED per subprocess, so a hash()-based
    implementation is guaranteed to disagree rather than merely likely to. The
    subprocess result is checked for success and for a parseable number: the
    first version compared stdout strings, so three identical *failures* — a
    wrong cwd, an import error — passed as three identical draws.
    """
    import os
    import subprocess
    import sys as _sys

    path = Path(__file__).resolve().parent.parent / "examples"
    path = path / "prompt_injection_eval.py"
    code = (
        "import importlib.util,sys;"
        f"spec=importlib.util.spec_from_file_location('inj',{str(path)!r});"
        "m=importlib.util.module_from_spec(spec);sys.modules['inj']=m;spec.loader.exec_module(m);"
        "print(round(m._uniform('doc-under-test', 7), 12))"
    )
    draws = set()
    for hashseed in ("0", "1", "12345"):
        env = {**os.environ, "PYTHONHASHSEED": hashseed}
        proc = subprocess.run(
            [_sys.executable, "-c", code],
            capture_output=True,
            text=True,
            env=env,
            cwd=tempfile.gettempdir(),  # nothing may depend on the caller's cwd
        )
        assert proc.returncode == 0, f"subprocess failed: {proc.stderr}"
        draws.add(float(proc.stdout.strip()))  # raises if it printed nothing
    assert len(draws) == 1, f"per-process variation: {draws}"
    assert draws != {0.0}  # a degenerate constant would also be "stable"
