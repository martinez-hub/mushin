"""llm_judge: turn a user judge-call into a pointwise metric."""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any


def parse_score(reply: str) -> float:
    """Extract a [0,1] score from a judge reply: a leading 0/1 float, yes/no, or
    `score: X`. Raise ValueError if none is found."""
    text = reply.strip().lower()
    if text.startswith("yes"):
        return 1.0
    if text.startswith("no"):
        return 0.0
    m = re.search(r"score\s*[:=]\s*([0-9]*\.?[0-9]+)", text) or re.match(
        r"([0-9]*\.?[0-9]+)", text
    )
    if m:
        return float(m.group(1))
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

    def metric(output: Any, reference: Any) -> float:
        prompt = template(rubric, output, reference)
        return parse(judge(prompt, seed))

    return metric
