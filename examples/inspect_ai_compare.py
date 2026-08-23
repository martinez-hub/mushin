# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
"""Is that eval gap real? Statistics for `Inspect AI <https://inspect.aisi.org.uk>`_ logs.

You ran the same eval on two models. One scored 60%, the other 56.7%. **Should
you switch?**

Inspect AI gives you those headline numbers — it runs the eval, the solvers, the
tool use, the scoring — but it does not tell you whether a gap is real. A gap can
be luck in two different ways, and this script measures both:

1. **Sampling luck.** Models generate different answers run to run. Re-run the
   same eval and the score wiggles. Inspect's ``--epochs`` gives you the repeats;
   mushin turns them into a p-value.
2. **Question luck.** You picked 50 questions. A *different* 50 might rank the
   models the other way round. This is usually the larger risk and the one
   nobody measures — a bootstrap over the eval items answers it.

A difference worth acting on has to survive both.

Try it without installing anything or running a real eval::

    python examples/inspect_ai_compare.py --demo

That runs two scenarios with **known ground truth** — two identical models, and
one genuinely better model — so you can see the checks correctly refuse the
first and accept the second.

On your own logs::

    inspect eval theory_of_mind.py --model openai/gpt-4 --epochs 5
    inspect eval theory_of_mind.py --model anthropic/claude-3-5-sonnet --epochs 5
    python examples/inspect_ai_compare.py logs/*.eval

Requires ``pip install "mushin-py[eval]"``; the log-reading path additionally
needs ``pip install inspect-ai``.

How the two tools line up:

===================  ===============================================
Inspect AI           mushin
===================  ===============================================
one sample           one **item**  — "would other questions agree?"
one epoch            one **run**   — "would a re-run agree?"
===================  ===============================================
"""

from __future__ import annotations

import sys
import types
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np

# Inspect's CORRECT/INCORRECT sentinels, as they appear in a log.
_VERDICTS = {"C": 1.0, "I": 0.0, "P": 0.5, "N": 0.0}


def _to_float(value: Any) -> float:
    """One Inspect ``Score.value`` -> float.

    Score values are not always numeric: the built-in scorers record ``"C"``/
    ``"I"`` for correct/incorrect. Anything unrecognised raises rather than being
    coerced — a silently wrong number here is invisible downstream.
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
    """Inspect ``EvalLog`` objects -> ``({model: (n_runs, n_items)}, question_ids)``.

    Questions are matched across models **by ``sample.id``, never by position**.
    Two Inspect logs can list their samples in different orders — retries,
    parallelism, a shuffled dataset — and the comparison pairs question *i* of one
    model with question *i* of the other. Matching positionally would compare
    "capital of France" against "solve this integral" and report a confident,
    meaningless answer. Every model must cover the same question ids, or this
    raises.

    ``scorer`` picks one entry from ``sample.scores`` when the task has several.
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
                "header_only=True — use read_eval_log(...) instead"
            )
        for sample in samples:
            if sample.scores is None:
                raise ValueError(
                    f"sample {sample.id!r} in the log for {model!r} has no scores "
                    "(an errored or unscored Inspect sample). Re-run or filter "
                    "those samples out before comparing."
                )
            scores = sample.scores
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
                f"models answered different questions, so they cannot be paired: "
                f"{model!r} is missing {missing} and adds {extra}. Run every model "
                "on the same dataset."
            )

    question_ids = sorted(shared, key=str)  # deterministic, order-independent
    arrays: dict[str, np.ndarray] = {}
    for model, cells in per_model.items():
        epochs = sorted({ep for _, ep in cells})
        try:
            arrays[model] = np.array(
                [[cells[(sid, ep)] for sid in question_ids] for ep in epochs],
                dtype=float,
            )
        except KeyError as exc:
            raise ValueError(
                f"log for {model!r} is missing a sample/epoch cell {exc.args[0]!r}; "
                "every sample must be scored in every epoch"
            ) from exc
    return arrays, question_ids


def report(scores: dict[str, np.ndarray], *, clusters=None) -> None:
    """Print the headline numbers, then whether each gap survives both checks.

    ``clusters`` is positional against the item axis, which is the sorted
    question-id order :func:`scores_from_logs` returns — pass its second return
    value through your grouping lookup, not the raw log order.
    """
    from mushin.llm import compare_scores

    names = list(scores)
    if len(names) < 2:
        print(
            f"only one model ({names[0]}) — a comparison needs at least two logs."
            if names
            else "no models to compare."
        )
        return

    n_runs = {name: arr.shape[0] for name, arr in scores.items()}
    # Recomputed from the per-sample scores, so it is the mean over the epochs
    # and samples present in the log — not necessarily the figure Inspect's own
    # aggregate metric printed (a different reducer would give a different one).
    print("Mean score per model (recomputed from the per-sample scores):")
    for name in names:
        print(f"   {name:<32} {scores[name].mean():6.1%}  ({n_runs[name]} epoch(s))")

    result = compare_scores(scores, clusters=clusters)
    n_pairs = len(result.comparisons)
    if n_pairs > 1:
        print(
            f"\n{n_pairs} pairwise comparisons — p-values below are Holm-corrected "
            "for multiplicity."
        )

    for _, row in result.comparisons.iterrows():
        a, b = row["method_a"], row["method_b"]
        gap = row["mean_diff"]
        leader, trailer = (a, b) if gap >= 0 else (b, a)
        print(
            f"\n{leader} leads {trailer} by {abs(gap) * 100:.1f} points. Is that real?"
        )

        # 1. sampling noise across Inspect epochs. Use the CORRECTED p-value:
        # with three or more logs the raw one over-states significance, and the
        # library already did the correction for us.
        p_seed = row["p_corrected"]
        if np.isnan(p_seed):
            if min(n_runs[a], n_runs[b]) < 2:
                why = "only one epoch, so re-run variation is unmeasured"
            else:
                why = (
                    "one model scored identically in every epoch, so it has no "
                    "re-run distribution to test"
                )
            verdict = f"no answer — {why}"
        elif p_seed < 0.05:
            verdict = f"survives re-running (p={p_seed:.4f})"
        else:
            verdict = f"could be re-run noise (p={p_seed:.4f})"
        print(f"   would a RE-RUN agree?          {verdict}")

        # 2. eval-set uncertainty across questions. The interval is signed
        # method_a - method_b; orient it to the lead just announced, or it reads
        # as contradicting the sentence above.
        lo, hi = row["item_ci_low"], row["item_ci_high"]
        if gap < 0:
            lo, hi = -hi, -lo
        if np.isnan(row["item_p"]):
            items = "no answer — per-question scores unavailable"
        elif row["item_p"] < 0.05:
            items = f"holds up (p={row['item_p']:.4f}, 95% CI [{lo * 100:+.1f}, {hi * 100:+.1f}] points)"
        else:
            items = (
                f"NOT established (p={row['item_p']:.4f}, 95% CI "
                f"[{lo * 100:+.1f}, {hi * 100:+.1f}] points includes 0 — another question set "
                "could flip it)"
            )
        print(f"   would OTHER QUESTIONS agree?   {items}")

        real = bool(row["significant"]) and row["item_p"] < 0.05
        print(
            f"   -> {'a difference worth acting on' if real else 'not a difference you can defend'}"
        )


def _fake_log(model: str, question_skill, bonus: float, rng, epochs: int = 5):
    """A stand-in for one `inspect eval --epochs N` run, log-shaped."""
    rows = []
    for epoch in range(1, epochs + 1):
        for qid, skill in question_skill.items():
            p = 1.0 / (1.0 + np.exp(-(skill + bonus + rng.normal(scale=0.4))))
            rows.append((qid, epoch, "C" if rng.random() < p else "I"))
    rng.shuffle(rows)  # real logs are not ordered; alignment is by id
    samples = [
        types.SimpleNamespace(
            id=qid, epoch=ep, scores={"accuracy": types.SimpleNamespace(value=v)}
        )
        for qid, ep, v in rows
    ]
    return types.SimpleNamespace(
        eval=types.SimpleNamespace(model=model), samples=samples
    )


def demo() -> None:
    """Two scenarios with known ground truth, so the verdicts can be checked."""
    for title, bonus, truth in [
        ("SCENARIO 1 — the two models are IDENTICAL", 0.0, "any gap here is luck"),
        ("SCENARIO 2 — one model is GENUINELY BETTER", 1.6, "the gap is real"),
    ]:
        rng = np.random.default_rng(2)
        skill = {f"q{i}": rng.normal() for i in range(50)}
        logs = [
            _fake_log("gpt-4", skill, bonus, rng),
            _fake_log("claude-3-5", skill, 0.0, rng),
        ]
        scores, _ = scores_from_logs(logs)
        print("=" * 72)
        print(f"{title}  ({truth})")
        print("=" * 72)
        report(scores)
        print()


def main(argv: Sequence[str]) -> int:
    if "--demo" in argv:
        demo()
        return 0
    if not argv:
        print(__doc__)
        return 1
    try:
        from inspect_ai.log import read_eval_log
    except ImportError:
        print("reading real logs needs Inspect AI:  pip install inspect-ai")
        print("or try:  python examples/inspect_ai_compare.py --demo")
        return 1
    scores, question_ids = scores_from_logs([read_eval_log(p) for p in argv])
    print(f"{len(question_ids)} questions, aligned by sample id\n")
    report(scores)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
