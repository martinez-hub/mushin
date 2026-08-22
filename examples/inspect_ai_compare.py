# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
"""Compare Inspect AI eval logs with mushin's statistics.

`Inspect AI <https://inspect.aisi.org.uk>`_ runs the evaluation — solvers, tool
use, scoring, sandboxing — and writes a log per model. It reports per-sample
scores and standard errors, but not model-vs-model inference. mushin takes it
from there: paired item-level bootstrap, seed/epoch variance, and
multiple-comparison correction across models.

The two dimensions line up naturally:

===================  ==========================================
Inspect AI           mushin
===================  ==========================================
one sample           one **item**   (eval-set uncertainty)
one epoch            one **run**    (decoding/sampling noise)
===================  ==========================================

So an Inspect eval run with ``--epochs 5`` gives exactly the
``(n_runs, n_items)`` array :func:`mushin.llm.compare_scores` wants.

Run Inspect first, once per model::

    inspect eval theory_of_mind.py --model openai/gpt-4 --epochs 5
    inspect eval theory_of_mind.py --model anthropic/claude-3-5-sonnet --epochs 5

then::

    python examples/inspect_ai_compare.py logs/*.eval

Requires ``pip install inspect-ai "mushin-py[eval]"``.
"""

from __future__ import annotations

import sys
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np

# Inspect's own CORRECT/INCORRECT sentinels, as they appear in a log.
_VERDICTS = {"C": 1.0, "I": 0.0, "P": 0.5, "N": 0.0}


def _to_float(value: Any) -> float:
    """One Inspect ``Score.value`` -> float.

    Score values are not always numeric: the built-in scorers record
    ``"C"``/``"I"`` for correct/incorrect, and a grouped scorer can return a
    mapping. Anything unrecognised raises rather than being silently coerced —
    a wrong number here is invisible downstream.
    """
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        key = value.strip().upper()
        if key in _VERDICTS:
            return _VERDICTS[key]
        try:
            return float(key)
        except ValueError as exc:
            raise ValueError(
                f"cannot score Inspect value {value!r}; pass a `value_fn` that "
                "maps your scorer's values to floats"
            ) from exc
    raise TypeError(
        f"unsupported Inspect score value type {type(value).__name__} ({value!r}); "
        "pass a `value_fn` to convert it"
    )


def scores_from_logs(
    logs: Iterable[Any],
    *,
    scorer: str | None = None,
    value_fn=_to_float,
) -> tuple[dict[str, np.ndarray], list[Any]]:
    """Inspect ``EvalLog`` objects -> ``({model: (n_runs, n_items)}, sample_ids)``.

    Samples are aligned **by ``sample.id``**, not by position. Two logs can list
    their samples in different orders (retries, parallelism, a shuffled dataset),
    and ``compare_scores`` pairs item *i* of one model with item *i* of another —
    so aligning positionally would silently compare unrelated questions. Every
    model must cover the same sample ids, or this raises.

    ``scorer`` picks one entry from ``sample.scores`` when the task has several;
    with a single scorer it can be omitted.
    """
    per_model: dict[str, dict[tuple[Any, int], float]] = {}
    ids_seen: dict[str, set[Any]] = {}

    for log in logs:
        model = log.eval.model
        if model in per_model:
            raise ValueError(
                f"two logs for model {model!r}; pass one log per model (or rename "
                "them) so each column of the comparison is one system"
            )
        cells: dict[tuple[Any, int], float] = {}
        samples = log.samples or []
        if not samples:
            raise ValueError(
                f"log for {model!r} has no samples; it was probably read with "
                "header_only=True — use read_eval_log(...) or "
                "read_eval_log_sample_summaries(...)"
            )
        for sample in samples:
            scores = sample.scores or {}
            if scorer is None:
                if len(scores) != 1:
                    raise ValueError(
                        f"sample {sample.id!r} has scorers {sorted(scores)}; pass "
                        "scorer=<name> to choose one"
                    )
                (score,) = scores.values()
            else:
                if scorer not in scores:
                    raise ValueError(
                        f"scorer {scorer!r} missing from sample {sample.id!r} "
                        f"(has {sorted(scores)})"
                    )
                score = scores[scorer]
            epoch = getattr(sample, "epoch", 1)
            key = (sample.id, epoch)
            if key in cells:
                raise ValueError(
                    f"duplicate (id={sample.id!r}, epoch={epoch}) in the log for "
                    f"{model!r}"
                )
            cells[key] = float(value_fn(score.value))
        per_model[model] = cells
        ids_seen[model] = {sid for sid, _ in cells}

    models = list(per_model)
    shared = ids_seen[models[0]]
    for model in models[1:]:
        if ids_seen[model] != shared:
            missing = sorted(map(str, shared - ids_seen[model]))[:5]
            extra = sorted(map(str, ids_seen[model] - shared))[:5]
            raise ValueError(
                f"models evaluate different samples, so they cannot be paired: "
                f"{model!r} is missing {missing} and adds {extra}. Run every model "
                "on the same dataset."
            )

    sample_ids = sorted(shared, key=str)  # deterministic, order-independent
    arrays: dict[str, np.ndarray] = {}
    for model, cells in per_model.items():
        epochs = sorted({ep for _, ep in cells})
        try:
            arrays[model] = np.array(
                [[cells[(sid, ep)] for sid in sample_ids] for ep in epochs],
                dtype=float,
            )
        except KeyError as exc:
            raise ValueError(
                f"log for {model!r} is missing a sample/epoch cell {exc.args[0]!r}; "
                "every sample must be scored in every epoch"
            ) from exc
    return arrays, sample_ids


def main(paths: Sequence[str]) -> int:
    try:
        from inspect_ai.log import read_eval_log
    except ImportError:
        print("this example needs Inspect AI:  pip install inspect-ai")
        return 1
    from mushin.llm import compare_scores

    if not paths:
        print(__doc__)
        return 1

    logs = [read_eval_log(p) for p in paths]
    scores, sample_ids = scores_from_logs(logs)
    for model, arr in scores.items():
        print(f"  {model}: {arr.shape[0]} epoch(s) x {arr.shape[1]} samples")

    result = compare_scores(scores)
    print()
    print(result.summary().to_string(index=False))
    print()
    cols = [
        "method_a",
        "method_b",
        "p_value",
        "item_diff",
        "item_ci_low",
        "item_ci_high",
        "item_p",
    ]
    print(result.comparisons[cols].to_string(index=False))
    print(
        "\n`p_value` is epoch-to-epoch (sampling) noise; `item_*` is whether the "
        "difference survives a different sample of eval items — usually the "
        "larger uncertainty. Read both."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
