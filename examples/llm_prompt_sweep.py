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
``max_total_seconds``  a wall-clock budget, so a runaway sweep stops itself
``sample=``            try a fraction of the grid before committing to all of it
``to_xarray()``        results labelled by prompt/temperature/seed — no bookkeeping

and then hands the winner to the statistics, so "prompt B is better" is a claim
about a difference that survived re-running rather than a single lucky run.

Run the demo (no keys, no network — failures are simulated)::

    python examples/llm_prompt_sweep.py --demo
"""

from __future__ import annotations

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


def _simulated_call(prompt_name: str, temperature: float, question: str, rng) -> float:
    """Score one (prompt, temperature, question)."""
    # cot helps; high temperature hurts; role helps a little
    base = {"terse": 0.55, "cot": 0.72, "role": 0.63, "fewshot": 0.66}[prompt_name]
    return float(rng.random() < base - 0.25 * temperature)


def score_config(prompt: str, temperature: float, seed: int) -> dict:
    """One grid cell: run the eval set under this configuration.

    Whatever dict this returns becomes data variables in the labelled dataset,
    keyed by the swept parameters — that is the whole contract.
    """
    rng = np.random.default_rng(abs(hash((prompt, temperature, seed))) % 2**32)
    if _OUTAGE and rng.random() < _FAILURE_RATE:
        raise RuntimeError("429 rate limited (simulated)")
    scores = [_simulated_call(prompt, temperature, q, rng) for q in QUESTIONS]
    return {"accuracy": float(np.mean(scores))}


def demo(argv: Sequence[str]) -> int:
    import warnings

    workdir = tempfile.mkdtemp()

    print("PASS 1 — the sweep, with transient API failures\n")
    sweep = mushin.sweep(score_config)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
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
    if failed:
        print(
            "   -> a strict harness would have lost the other "
            f"{total - failed} paid calls too"
        )

    print("\nPASS 2 — the outage is over; resume retries only the holes\n")
    global _OUTAGE
    _OUTAGE = False  # "we fixed the cause", as in the resilient-sweeps notebook
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sweep2 = mushin.sweep(score_config)
        sweep2.run(
            prompt=mushin.multirun(list(PROMPTS)),
            temperature=mushin.multirun([0.0, 0.4, 0.8]),
            seed=mushin.multirun(list(range(5))),
            working_dir=workdir,
            resume=True,
        )
    ds2 = sweep2.workflow.to_xarray()
    still_missing = int(np.isnan(ds2["accuracy"].values).sum())
    print(
        f"   after resume: {ds2['accuracy'].size - still_missing} completed, "
        f"{still_missing} still missing"
    )

    print("\nWHICH CONFIGURATION WON — one reduction over named dimensions\n")
    mean_acc = ds2["accuracy"].mean("seed")
    table = mean_acc.to_dataframe().unstack("temperature")
    print(table.to_string(float_format=lambda v: f"{v:.1%}"))
    best = mean_acc.where(mean_acc == mean_acc.max(), drop=True)
    bp = str(best["prompt"].values[0])
    bt = float(best["temperature"].values[0])
    print(
        f"\n   best: prompt={bp!r} temperature={bt}  ({float(best.values.ravel()[0]):.1%})"
    )

    print("\nIS THE WINNER ACTUALLY BETTER? — the per-seed scores, tested\n")
    from mushin.llm import compare_scores

    runner_up = "terse" if bp != "terse" else "role"
    at = lambda p: ds2["accuracy"].sel(prompt=p, temperature=bt).values[:, None]  # noqa: E731
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        row = compare_scores({bp: at(bp), runner_up: at(runner_up)}).comparisons.iloc[0]
    print(f"   {bp} vs {runner_up} at temperature={bt}")
    print(f"      mean difference {row['mean_diff']:+.1%}")
    print(
        f"      would a re-run agree?  "
        f"{'yes' if row['p_value'] < 0.05 else 'not established'} "
        f"(p={row['p_value']:.4f})"
    )
    print(
        "\n(The seeds here are whole re-runs of the eval set, so this asks whether\n"
        " the prompt difference survives sampling noise — not whether it would\n"
        " survive different questions; pass per-question scores for that.)"
    )
    return 0


def main(argv: Sequence[str]) -> int:
    if "--demo" in argv:
        return demo(argv)
    print(__doc__)
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
