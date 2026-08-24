# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
"""Finding the best prompt, without paying for the same call twice.

Tuning an LLM system means sweeping the things you control — prompt template,
temperature, retrieval depth, model — across several seeds, and reading off what
actually helped. That sweep has properties a training sweep does not:

* **every cell costs money**, so re-running from scratch after a crash is a real
  loss, not just lost time;
* **cells fail transiently** — rate limits, timeouts, a 503 — and one failure
  must not discard the hours already spent;
* **the grid is easy to under-estimate**: 4 prompts x 3 temperatures x 5 seeds is
  60 API calls per eval item.

This example shows the sweep half of mushin handling exactly that:

``on_error="nan"``     a rate-limited cell becomes NaN; the rest of the grid finishes
``resume=True``        re-running reuses completed cells and only retries the holes
``to_xarray()``        results labelled by prompt/temperature/seed — no bookkeeping
``max_total_seconds``  a wall-clock budget, so a runaway sweep stops itself
``sample=``            try a fraction of the grid before committing to all of it

The demo exercises the first three, and then hands the winner to the statistics,
so "prompt B is better" is a claim about a difference that survived re-running
rather than a single lucky run. The budget and sampling knobs are listed because
they belong to the same problem and take one argument each — the demo does not
exercise them; see the resilient-sweeps guide.

**The mushin part, in full.** The rest of this file simulates an LLM and prints
the result::

    import mushin

    sweep = mushin.sweep(score_config)       # score_config(prompt, temperature, seed)
    sweep.run(
        prompt=mushin.multirun(list(PROMPTS)),
        temperature=mushin.multirun([0.0, 0.4, 0.8]),
        seed=mushin.multirun(list(range(5))),
        working_dir=workdir,
        on_error="nan",                      # a 429 must not discard the grid
    )
    ds = sweep.workflow.to_xarray()          # labelled by prompt/temperature/seed

    sweep.run(..., working_dir=workdir, resume=True)   # retries only the holes

Run the demo (no keys, no network — failures are simulated)::

    pip install "mushin-py[eval]"     # the significance step needs the extra
    python examples/llm_prompt_sweep.py --demo
"""

from __future__ import annotations

import hashlib
import sys
import tempfile
from collections.abc import Sequence

import numpy as np

import mushin

#: Prompt templates to compare. In a real sweep these are your candidates.
PROMPTS = {
    "terse": "Answer in one sentence: {q}",
    "cot": "Think step by step, then answer: {q}",
    "role": "You are a careful domain expert. Answer precisely: {q}",
    "fewshot": "Q: 2+2? A: 4\nQ: capital of France? A: Paris\nQ: {q} A:",
}

#: A stand-in for "the eval set".
QUESTIONS = [f"question-{i}" for i in range(40)]

#: How often a simulated CELL fails transiently, to exercise on_error/resume.
#: Rolled once per configuration, not per call — a rate limit takes out the
#: request you are making, not each of the 40 questions independently.
_FAILURE_RATE = 0.15

#: Stands in for "we fixed the cause" between the two passes: with the outage
#: over, the resume pass retries the holes and they succeed. Mirrors the
#: sentinel in the resilient-sweeps notebook.
_OUTAGE = True


def _draw(*key) -> float:
    """A stable [0, 1) draw for a key, independent of process and call order.

    blake2b rather than hash(), which numpy would happily seed from but which
    Python randomises per process: the same cell would then score differently on
    every run, and `resume` would be reusing values it could not reproduce.
    """
    digest = hashlib.blake2b("|".join(map(str, key)).encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") / 2**64


def _simulated_call(
    prompt_name: str, temperature: float, seed: int, question: str
) -> float:
    """Score one (prompt, temperature, seed, question).

    Depends on the seed, because that is exactly what the seed dimension
    measures: re-running one configuration must be able to give a different
    answer, or the significance test has nothing to test.
    """
    # cot helps; high temperature hurts; role helps a little
    base = {"terse": 0.55, "cot": 0.72, "role": 0.63, "fewshot": 0.66}[prompt_name]
    draw = _draw("score", prompt_name, temperature, seed, question)
    return float(draw < base - 0.25 * temperature)


def score_config(prompt: str, temperature: float, seed: int) -> dict:
    """One grid cell: run the eval set under this configuration.

    Whatever dict this returns becomes data variables in the labelled dataset,
    keyed by the swept parameters — that is the whole contract.
    """
    # The outage roll is drawn from its own stream, so clearing _OUTAGE does not
    # shift the scores: a cell filled in by `resume` is the SAME cell that failed,
    # which is what makes reuse meaningful rather than merely convenient.
    if _OUTAGE and _draw("outage", prompt, temperature, seed) < _FAILURE_RATE:
        raise RuntimeError("429 rate limited (simulated)")
    scores = [_simulated_call(prompt, temperature, seed, q) for q in QUESTIONS]
    return {"accuracy": float(np.mean(scores))}


def demo(argv: Sequence[str]) -> int:
    import warnings

    workdir = tempfile.mkdtemp()

    print("PASS 1 — the sweep, with transient API failures\n")
    sweep = mushin.sweep(score_config)
    # Captured, not silenced: mushin reports "N run(s) failed" and that is the
    # headline of this pass, not noise to be filtered out.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sweep.run(
            prompt=mushin.multirun(list(PROMPTS)),
            temperature=mushin.multirun([0.0, 0.4, 0.8]),
            seed=mushin.multirun(list(range(5))),
            working_dir=workdir,
            on_error="nan",  # a 429 must not discard the whole grid
        )
    wf = sweep.workflow
    ds = wf.to_xarray()
    total = int(ds["accuracy"].size)
    failed = int(np.isnan(ds["accuracy"].values).sum())
    print(f"   grid: {dict(ds.sizes)}  = {total} cells")
    print(f"   completed: {total - failed},  failed transiently: {failed}")
    print(f"   is_complete: {wf.is_complete}")
    for w in caught:
        # mushin says it out loud; a harness that hides this is how a half-empty
        # sweep gets written up as a finished one.
        print(f"   mushin: {w.message}")
    if failed:
        done = total - failed
        print(
            f"   -> a strict harness would have discarded those {done} completed "
            f"cells too — {done * len(QUESTIONS)} paid calls"
        )

    print("\nPASS 2 — the outage is over; resume retries only the holes\n")
    global _OUTAGE
    _OUTAGE = False  # "we fixed the cause", as in the resilient-sweeps notebook
    try:
        _resume_and_report(workdir, sweep)
    finally:
        # Restore it, or a second demo() in the same process would start with the
        # outage already over and silently demonstrate nothing.
        _OUTAGE = True
    return 0


def _resume_and_report(workdir: str, sweep) -> None:
    import warnings

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sweep2 = mushin.sweep(score_config)
        sweep2.run(
            prompt=mushin.multirun(list(PROMPTS)),
            temperature=mushin.multirun([0.0, 0.4, 0.8]),
            seed=mushin.multirun(list(range(5))),
            working_dir=workdir,
            resume=True,
            on_error="nan",  # a cell still failing must not abort the retry
        )
    ds2 = sweep2.workflow.to_xarray()
    still_missing = int(np.isnan(ds2["accuracy"].values).sum())
    print(
        f"   after resume: {ds2['accuracy'].size - still_missing} completed, "
        f"{still_missing} still missing"
    )
    for w in caught:
        print(f"   mushin: {w.message}")

    print("\nWHICH CONFIGURATION WON — one reduction over named dimensions\n")
    mean_acc = ds2["accuracy"].mean("seed")
    table = mean_acc.to_dataframe().unstack("temperature")
    print(table.to_string(float_format=lambda v: f"{v:.1%}"))
    # argmax over the flattened grid: `where(== max, drop=True)` keeps a whole
    # sub-block when cells tie, and taking [0] of each coordinate can then name a
    # cell that is not the maximum at all (or a NaN one).
    flat = mean_acc.stack(cell=("prompt", "temperature"))
    best_cell = flat.isel(cell=int(flat.argmax("cell")))
    bp = str(best_cell["prompt"].values)
    bt = float(best_cell["temperature"].values)
    print(f"\n   best: prompt={bp!r} temperature={bt}  ({float(best_cell):.1%})")

    print("\nIS THE WINNER ACTUALLY BETTER? — the per-seed scores, tested\n")
    from mushin.llm import compare_scores

    # The RUNNER-UP at the winning temperature, not a fixed baseline: comparing
    # the winner against the by-design worst prompt makes the test trivially easy
    # and says nothing about the choice you actually face.
    at_best = mean_acc.sel(temperature=bt)
    # Drop the winner FIRST. Sorting and taking [1] returns the winner itself
    # whenever another prompt ties it — the sort is not guaranteed to put the
    # winning cell first among equals — and comparing a system against itself
    # collapses compare_scores to one system and raises.
    others = at_best.drop_sel(prompt=bp)
    if others.sizes.get("prompt", 0) == 0:
        print("   only one prompt swept — nothing to compare the winner against.")
        return
    runner_up = str(others["prompt"].values[int(others.argmax("prompt"))])
    at = lambda p: ds2["accuracy"].sel(prompt=p, temperature=bt).values[:, None]  # noqa: E731
    with warnings.catch_warnings(record=True) as stat_warnings:
        warnings.simplefilter("always")
        row = compare_scores({bp: at(bp), runner_up: at(runner_up)}).comparisons.iloc[0]
    print(f"   {bp} vs {runner_up} at temperature={bt}")
    print(f"      mean difference {row['mean_diff']:+.1%}")
    print(
        f"      would a re-run agree?  "
        f"{'yes' if row['p_value'] < 0.05 else 'not established'} "
        f"(p={row['p_value']:.4f})"
    )
    # One expected warning is filtered by MESSAGE, not by a blanket ignore: the
    # (n_seeds, 1) shape is deliberate here — each cell is already a mean over
    # the 40 questions, so there is one "item" per run and no item-level
    # statistics are possible. That is the trade-off the closing note describes.
    # Anything mushin says beyond it is printed.
    for w in stat_warnings:
        if "no item-level statistics are possible" not in str(w.message):
            print(f"      ! mushin warns: {w.message}")
    print(
        "\n(The seeds here are whole re-runs of the eval set, so this asks whether\n"
        " the prompt difference survives sampling noise — not whether it would\n"
        " survive different questions; pass per-question scores for that.)"
    )


def main(argv: Sequence[str]) -> int:
    if "--demo" in argv:
        return demo(argv)
    print(__doc__)
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
