"""llm_judge: turn a user judge-call into a pointwise metric."""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any


def parse_score(reply: str) -> float:
    """Extract a [0,1] score from a judge reply: a leading 0/1 float, yes/no, or
    `score: X`. Raise ValueError if none is found."""
    text = reply.strip().lower()
    yn = re.match(r"(yes|no)\b", text)  # whole word, so "nope"/"yesterday" don't match
    if yn:
        return 1.0 if yn.group(1) == "yes" else 0.0
    m = re.search(r"score\s*[:=]\s*([0-9]*\.?[0-9]+)", text) or re.match(
        r"([0-9]*\.?[0-9]+)", text
    )
    if m:
        score = float(m.group(1))
        if not 0.0 <= score <= 1.0:
            raise ValueError(
                f"judge returned a score of {score}, outside [0, 1] "
                f"(reply: {reply!r}). Use a 0-1 rubric, or pass a custom `parse` "
                "that rescales (e.g. a 1-10 scale)."
            )
        return score
    raise ValueError(f"could not parse a score from judge reply: {reply!r}")


def default_template(rubric: str, output: Any, reference: Any) -> str:
    ref = "" if reference is None else f"\nReference answer:\n{reference}\n"
    return (
        f"{rubric}\n\nCandidate answer:\n{output}\n{ref}\n"
        "Reply with `yes`/`no` or `score: <0-1>`."
    )


def llm_judge(
    judge: Callable[[str, int], str],
    rubric: str,
    *,
    parse: Callable[[str], float] = parse_score,
    template: Callable[[str, Any, Any], str] = default_template,
    seed: int = 0,
) -> Callable[[Any, Any], float]:
    """Return a pointwise metric `(output, reference) -> float` that asks `judge`
    (a user-supplied, provider-agnostic `judge(prompt, seed) -> reply`) to score
    each output against `rubric`, parsing the reply to a float."""

    def metric(output: Any, reference: Any, seed: int = seed) -> float:
        # `seed` defaults to the llm_judge seed; compare_llms overrides it with the
        # per-trial seed so a stochastic judge is tied to (and reproducible per) run.
        prompt = template(rubric, output, reference)
        return parse(judge(prompt, seed))

    return metric
