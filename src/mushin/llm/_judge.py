"""llm_judge: turn a user judge-call into a pointwise metric."""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

# A number, optionally written as a fraction `X/N` (e.g. an "8/10" scale).
_NUM = r"([0-9]*\.?[0-9]+)\s*(?:/\s*([0-9]*\.?[0-9]+))?"


def _to_float(num: str, denom: str | None) -> float:
    """Resolve a captured number / fraction to a float (`1/10` -> 0.1)."""
    value = float(num)
    if denom is not None:
        d = float(denom)
        if d == 0:
            raise ValueError(f"judge returned a zero denominator: {num}/{denom}")
        value /= d
    return value


def parse_score(reply: str) -> float:
    """Extract a [0,1] score from a judge reply: a `score: X` (or `X/N` fraction),
    a yes/no verdict, or a leading number. Raise ValueError if none is found.

    An explicit ``score:`` wins over a leading yes/no, so a hedged reply such as
    ``"no, but score: 0.9"`` uses the stated number rather than the verdict."""
    text = reply.strip().lower()
    m = re.search(rf"score\s*[:=]\s*{_NUM}", text)
    if m:
        score = _to_float(m.group(1), m.group(2))
    else:
        yn = re.match(r"(yes|no)\b", text)  # whole word: "nope"/"yesterday" don't match
        if yn:
            return 1.0 if yn.group(1) == "yes" else 0.0
        m = re.match(_NUM, text)
        if not m:
            raise ValueError(f"could not parse a score from judge reply: {reply!r}")
        score = _to_float(m.group(1), m.group(2))
    if not 0.0 <= score <= 1.0:
        raise ValueError(
            f"judge returned a score of {score}, outside [0, 1] "
            f"(reply: {reply!r}). Use a 0-1 rubric, or pass a custom `parse` "
            "that rescales (e.g. a 1-10 scale)."
        )
    return score


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
